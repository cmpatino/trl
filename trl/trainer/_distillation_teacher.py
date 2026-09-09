# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teacher registry, scoring executor and window store for managed multi-teacher distillation.

Private helpers of [`DistillationTrainer`]: the registry resolves routing IDs to immutable teacher sources, the
executor keeps one reloadable CPU body slot and scores the exact generated tokens through
`torch.func.functional_call`, and the window store owns the CPU hidden-target blocks the managed loss consumes.
See `/data/workspaces/mopd/implementation/interfaces.md` for the seam contract.
"""

import contextlib
import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate.utils import is_peft_model
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel

from ._distillation_heads import HeadIdentity, HeadSource, TargetGroup
from ._distillation_identity import tokenizer_fingerprint
from .utils import create_model_from_path


logger = logging.getLogger(__name__)

# Loading options that change placement or numerics in ways the managed scoring path does not support: the executor
# needs a plain, complete, dense CPU source it can substitute into `functional_call`.
_UNSUPPORTED_INIT_KWARGS = ("device_map", "quantization_config", "load_in_8bit", "load_in_4bit")
# Loading metadata copied into the manifest. Everything else (credentials in particular) is dropped.
_MANIFEST_LOADING_KEYS = ("dtype", "revision", "attn_implementation", "low_cpu_mem_usage", "trust_remote_code")
# Version of the scoring adapter (backbone `functional_call` + causal shift). Bump when its output changes.
_ADAPTER_VERSION = 1
# Version of the head transformations (linear projection, optional bias, logit scale, softcap).
_TRANSFORM_VERSION = 1
# Per-block bookkeeping charged on top of the raw hidden bytes when planning a window.
_TARGET_BLOCK_OVERHEAD_BYTES = 4096
# Rows per staging tile fill. One tile is reused for every block; copies block until complete (`non_blocking=False`).
_STAGING_ROWS = 256
# Block size for the streaming content digests of checkpoint files and preloaded tensors.
_HASH_BLOCK_BYTES = 8 << 20


def _canonical_json(payload) -> str:
    """Serialize `payload` with sorted keys and no whitespace so digests ignore cosmetic JSON formatting."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(payload) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safetensors_header(path: Path) -> dict:
    """Read one safetensors file's JSON header (8-byte little-endian length, then the header) without any weights."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def _checkpoint_files(path: str) -> list[Path]:
    """
    The immutable files that define a local checkpoint's content.

    `config.json`, the shard index when the checkpoint is sharded (it decides which shard every tensor loads from),
    then the safetensors shards in name order.
    """
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Teacher checkpoint '{path}' has no config.json.")
    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(
            f"Teacher checkpoint '{path}' contains no safetensors weight file. Managed multi-teacher distillation "
            "reloads teacher bodies from immutable safetensors snapshots; re-save the checkpoint with "
            "`save_pretrained`."
        )
    index_path = root / "model.safetensors.index.json"
    index_files = [index_path] if index_path.is_file() else []
    return [config_path, *index_files, *weight_files]


def _content_digest(path: str) -> tuple[str, int]:
    """
    Stream a local checkpoint's bytes through sha256 in bounded blocks.

    Tensor *values* are part of the digest: headers, shapes, dtypes, sizes and timestamps do not change when weights
    are overwritten in place, and a local path is mutable. The executor recomputes this before every body reload, so
    the cost (reported as `ExecutorStats.verify_bytes`) is one extra read of the checkpoint per reload.

    Args:
        path (`str`):
            Local checkpoint directory.

    Returns:
        `tuple[str, int]`: hex sha256 digest and the number of bytes hashed.
    """
    digest = hashlib.sha256()
    hashed = 0
    for file_path in _checkpoint_files(path):
        digest.update(file_path.name.encode("utf-8"))
        with file_path.open("rb") as handle:
            while True:
                block = handle.read(_HASH_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
                hashed += len(block)
    return digest.hexdigest(), hashed


def _hash_tensor_blocks(digest, tensor: torch.Tensor) -> int:
    """
    Feed one tensor's values through `digest` in bounded blocks, without ever materializing a full copy.

    Slices along the first dimension are made contiguous one block at a time, so a non-contiguous source (a
    transposed parameter, say) costs one block rather than a copy of the whole tensor. A single row is the smallest
    block, so the bound is `max(_HASH_BLOCK_BYTES, one row)`. Logical element order is what gets hashed, so a
    contiguous tensor and a non-contiguous view of the same values hash identically.

    Args:
        digest (`hashlib._Hash`):
            Digest to update in place.
        tensor (`torch.Tensor`):
            Tensor to hash; moved to CPU without a copy when it already lives there.

    Returns:
        `int`: number of bytes hashed.
    """
    tensor = tensor.detach().to("cpu")
    if tensor.numel() == 0:
        return 0
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    rows = tensor.shape[0]
    row_bytes = max(1, tensor.numel() // rows * tensor.element_size())
    block_rows = max(1, _HASH_BLOCK_BYTES // row_bytes)
    hashed = 0
    for start in range(0, rows, block_rows):
        block = tensor[start : start + block_rows].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(block)
        hashed += len(block)
    return hashed


def _tensor_content_digest(model: PreTrainedModel) -> tuple[str, int]:
    """
    Stream a preloaded model's parameter and buffer values through sha256 in bounded blocks.

    A caller-provided model has no file identity, so its values are its identity: without them two models sharing a
    config and parameter shapes would resolve to the same source, and manifest validation could not reject a changed
    teacher on resume. Non-persistent buffers are included — `state_dict()` omits them, but they take part in the
    forward (Qwen3's rotary `inv_freq` is one), so a change to them changes the teacher's distribution.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            CPU model to fingerprint. Parameters and buffers are hashed in sorted-name order.

    Returns:
        `tuple[str, int]`: hex sha256 digest and the number of tensor bytes hashed.
    """
    digest = hashlib.sha256()
    hashed = 0
    tensors = sorted([*model.named_parameters(), *model.named_buffers()], key=lambda item: item[0])
    for name, tensor in tensors:
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        hashed += _hash_tensor_blocks(digest, tensor)
    return digest.hexdigest(), hashed


def _checkpoint_inventory(path: str) -> dict:
    """
    Inventory a local checkpoint directory from its safetensors headers, without materializing any weights.

    The inventory is the basis for a local source's storage and loading-transient accounting; its content identity is
    the streaming `content_digest`, which covers tensor values, because a local path is mutable and headers, sizes and
    timestamps do not change when weights are overwritten in place.

    Args:
        path (`str`):
            Local checkpoint directory.

    Returns:
        `dict` with keys:
            - `tensors` (`dict[str, list]`):
                Tensor name to `[dtype, shape]` as recorded in the file header.
            - `files` (`list[list]`):
                Per weight file `[name, size_bytes, header_digest]`, sorted by name.
            - `config_digest` (`str`):
                sha256 of `config.json`.
            - `numel` (`int`):
                Total number of checkpoint elements.
            - `largest_file_bytes` (`int`):
                Size of the largest weight file, used as the loading transient estimate.
            - `content_digest` (`str`):
                Streaming sha256 over `config.json` and every safetensors file, tensor values included.
            - `hashed_bytes` (`int`):
                Bytes hashed to produce `content_digest`.
    """
    content_files = _checkpoint_files(path)
    config_path = content_files[0]
    # The shard index is part of the content digest but has no tensor header to read.
    weight_files = [file_path for file_path in content_files if file_path.suffix == ".safetensors"]
    tensors = {}
    files = []
    numel = 0
    for weight_file in weight_files:
        header = _safetensors_header(weight_file)
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = [spec["dtype"], list(spec["shape"])]
            count = 1
            for dim in spec["shape"]:
                count *= dim
            numel += count
        files.append([weight_file.name, weight_file.stat().st_size, _sha256_json(header)])
    content_digest, hashed_bytes = _content_digest(path)
    return {
        "tensors": tensors,
        "files": files,
        "config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "numel": numel,
        "largest_file_bytes": max(entry[1] for entry in files),
        "content_digest": content_digest,
        "hashed_bytes": hashed_bytes,
    }


def _resolve_hub_snapshot(repo_id: str, revision: str | None, token: str | bool | None = None) -> tuple[str, str]:
    """
    Resolve a Hub branch/tag/commit to an immutable commit hash and download that commit's files locally.

    Pinning the snapshot up front is what makes CPU body reload independent of the network.

    Args:
        repo_id (`str`):
            Hub model repository.
        revision (`str`, *optional*):
            Branch, tag or commit. `None` selects the repository's default branch.
        token (`str` or `bool`, *optional*):
            Token forwarded to `huggingface_hub`.

    Returns:
        `tuple[str, str]`: the local snapshot directory and the resolved commit hash.
    """
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi(token=token).model_info(repo_id, revision=revision).sha
    local_dir = snapshot_download(repo_id, revision=commit, token=token)
    return local_dir, commit


def _resolve_dtype(loading_kwargs: dict, config) -> torch.dtype:
    """Resolve the dtype the CPU source materializes in, mirroring `create_model_from_path`'s dtype handling."""
    dtype = loading_kwargs.get("dtype", "float32")
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in ("auto", None):
        return config.dtype if isinstance(config.dtype, torch.dtype) else torch.float32
    if dtype in ("bfloat16", "float16", "float32"):
        return getattr(torch, dtype)
    raise ValueError(
        "Invalid `dtype` passed for a teacher. Expected either 'auto' or a string representing a valid `torch.dtype` "
        f"(e.g., 'float32'), but got {dtype}."
    )


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _logit_transforms(text_config) -> tuple[float, float | None]:
    """Read the teacher's pre-softmax logit transformations, matching `DistillationTrainer._compute_loss`."""
    # `logit_scale` is None on models that don't scale (e.g. MPT); read that as unscaled (1.0). A real 0.0 is kept
    # as-is. Muse Glimmer applies the same pre-softcap multiplier under the name `output_multiplier`.
    logit_scale = getattr(text_config, "logit_scale", None)
    if logit_scale is None:
        logit_scale = getattr(text_config, "output_multiplier", None)
    logit_scale = 1.0 if logit_scale is None else logit_scale
    return float(logit_scale), getattr(text_config, "final_logit_softcapping", None)


def _teacher_backbone(model: PreTrainedModel):
    """
    Return the backbone producing hidden states, the same object `DistillationTrainer._get_last_hidden_state` runs.

    `base_model` skips `lm_head`. Managed teachers are plain text decoders: PEFT wrappers and pre-5.0 VLM shells
    (where `base_model is model`, so scoring would re-run `lm_head`) need their own adapter and are rejected here.
    """
    if is_peft_model(model):
        raise ValueError(
            "Managed multi-teacher distillation cannot score a PEFT-wrapped teacher: merge the adapter into the base "
            "weights and register the merged checkpoint instead."
        )
    backbone = model.base_model
    if backbone is model:
        raise ValueError(
            f"Teacher architecture {type(model).__name__} exposes no separate backbone (`base_model` is the model "
            "itself), so scoring would re-run its output head. This architecture needs a dedicated teacher adapter."
        )
    return backbone


@contextlib.contextmanager
def _plain_loading_env():
    """
    Scope Accelerate's student-only RAM-efficient loading policy off while a teacher body loads.

    `transformers.distributed.fsdp.is_fsdp_enabled()` is true when `ACCELERATE_USE_FSDP` and
    `FSDP_CPU_RAM_EFFICIENT_LOADING` are both set under an initialized process group; `from_pretrained` then fills
    every process that is not local rank zero with zeros and waits for the student's state synchronization. Each rank
    needs its own complete teacher, so both variables are forced off for the duration of the load and restored on
    exit. DeepSpeed ZeRO-3's `zero.Init` is not env-scoped and stays unsupported (rejected by the trainer).
    """
    overrides = {"ACCELERATE_USE_FSDP": "false", "FSDP_CPU_RAM_EFFICIENT_LOADING": "false"}
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class TeacherEntry:
    """
    Everything the trainer knows about one registered teacher, independent of any loaded model object.

    Three dtypes are kept apart on purpose. `source_dtype` is what the CPU source materializes in. `target_dtype` is
    what hidden targets are stored in (the executor's autocast dtype when set, otherwise `source_dtype`), and window
    planning sizes blocks with it. `hidden_dtype` is the dtype the backbone actually returned: it stays `None` until a
    forward has been observed (by scoring or by [`~TeacherExecutor.probe_hidden_dtype`]) and is never inferred from
    configuration. `projection_dtype` is the matmul execution dtype the managed loss reproduces on replay.

    `content_digest` is the source's value-level identity: a streaming digest of the checkpoint files for path/Hub
    sources, or of the parameter and buffer values for a preloaded model. `identity_source` is the part of the source
    that belongs to that identity — a Hub repository id, or `None` for a local checkpoint or a preloaded model, whose
    identity is entirely their content. `source` and `load_path` stay as display and reload metadata: the same
    checkpoint staged under different per-rank paths must produce one identity.
    """

    teacher_id: str
    index: int
    source: str | None
    identity_source: str | None
    resolved_revision: str | None
    source_key: str
    config_class: str
    hidden_size: int
    vocab_size: int
    source_dtype: torch.dtype
    target_dtype: torch.dtype
    hidden_dtype: torch.dtype | None
    projection_dtype: torch.dtype
    tokenizer_fingerprint: str
    head_identity: HeadIdentity
    storage_bytes: int
    evictable: bool
    adapter_version: int
    load_path: str | None
    loading_kwargs: dict
    loading_transient_bytes: int
    content_digest: str
    content_hashed_bytes: int

    @property
    def head_bytes(self) -> int:
        """Bytes of the retained CPU head source (weight plus optional bias)."""
        rows, columns = self.head_identity.weight_shape
        item = _dtype_bytes(self.head_identity.source_dtype)
        return rows * columns * item + (rows * item if self.head_identity.has_bias else 0)


class TeacherRegistry:
    """
    Immutable identity and routing table for the registered teachers.

    Routing IDs are user-facing names: several IDs may point at the same repository, and each ID's `revision` is
    resolved once to a commit hash whose files are snapshotted locally, so two revisions of one repository are two
    entries that never share a source, head or cache. Shared tokenization is a v1 requirement, so every teacher's
    tokenizer fingerprint must equal the student's and every teacher's vocabulary size must equal the student's.

    Args:
        teacher_models (`dict[str, str` or [`~transformers.PreTrainedModel`]`]`):
            Mapping from routing ID to a checkpoint path/Hub ID, or to a frozen CPU model owned by the caller.
        student_tokenizer ([`~transformers.PreTrainedTokenizerFast`]):
            The student's processing class; prompts are rendered once with it and scored as token IDs.
        student_vocab_size (`int`):
            The student's `config.get_text_config().vocab_size`.
        common_init_kwargs (`dict`, *optional*):
            Loading kwargs applied to every path/Hub source, before per-ID overrides.
        per_teacher_init_kwargs (`dict[str, dict]`, *optional*):
            Per-ID loading overrides, including `revision`. IDs must exist in `teacher_models`; preloaded models
            accept none.
        teacher_tokenizers (`dict[str, ~transformers.PreTrainedTokenizerFast]`, *optional*):
            Tokenizers for sources whose tokenizer cannot be resolved from the checkpoint. Required for preloaded
            models.
        student_model ([`~transformers.PreTrainedModel`], *optional*):
            Unwrapped student, used only to reject preloaded teachers that alias the student's parameter storage.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Forwarded to config/tokenizer/model loading.
    """

    def __init__(
        self,
        teacher_models: dict[str, str | PreTrainedModel],
        *,
        student_tokenizer,
        student_vocab_size: int,
        common_init_kwargs: dict | None = None,
        per_teacher_init_kwargs: dict[str, dict] | None = None,
        teacher_tokenizers: dict | None = None,
        student_model: PreTrainedModel | None = None,
        trust_remote_code: bool = False,
    ):
        if not teacher_models:
            raise ValueError(
                "`teacher_models` is empty. Register at least one teacher, or use the singular `teacher_model` "
                "argument for single-teacher distillation."
            )
        common_init_kwargs = dict(common_init_kwargs or {})
        per_teacher_init_kwargs = {key: dict(value) for key, value in (per_teacher_init_kwargs or {}).items()}
        teacher_tokenizers = dict(teacher_tokenizers or {})
        unknown = sorted(set(per_teacher_init_kwargs) - set(teacher_models))
        if unknown:
            raise ValueError(
                f"`teacher_model_init_kwargs_by_teacher` has entries for unknown teacher IDs {unknown}. Registered "
                f"IDs are {sorted(teacher_models)}."
            )
        unknown = sorted(set(teacher_tokenizers) - set(teacher_models))
        if unknown:
            raise ValueError(f"`teacher_tokenizers` has entries for unknown teacher IDs {unknown}.")

        self.student_tokenizer_fingerprint = tokenizer_fingerprint(student_tokenizer)
        self.student_vocab_size = student_vocab_size
        self.trust_remote_code = trust_remote_code
        self.entries: list[TeacherEntry] = []
        self._by_id: dict[str, TeacherEntry] = {}
        self._preloaded: dict[int, PreTrainedModel] = {}
        student_storages = _parameter_storages(student_model) if student_model is not None else set()

        for index, (teacher_id, source) in enumerate(teacher_models.items()):
            overrides = per_teacher_init_kwargs.get(teacher_id, {})
            if isinstance(source, PreTrainedModel):
                if overrides:
                    raise ValueError(
                        f"Teacher '{teacher_id}' is a preloaded model, so it cannot take per-teacher loading "
                        f"overrides {sorted(overrides)}. Register a checkpoint path to apply loading options."
                    )
                entry = self._register_preloaded(
                    teacher_id, index, source, teacher_tokenizers.get(teacher_id), student_storages
                )
                self._preloaded[index] = source
            else:
                loading_kwargs = {**common_init_kwargs, **overrides}
                entry = self._register_path(teacher_id, index, source, loading_kwargs, teacher_tokenizers)
            self.entries.append(entry)
            self._by_id[teacher_id] = entry

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, teacher_id: str) -> TeacherEntry:
        if teacher_id not in self._by_id:
            raise ValueError(f"Unknown teacher ID {teacher_id!r}. Registered IDs are {sorted(self._by_id)}.")
        return self._by_id[teacher_id]

    @property
    def teacher_ids(self) -> list[str]:
        return [entry.teacher_id for entry in self.entries]

    def preloaded_model(self, index: int) -> PreTrainedModel:
        """Return the caller-owned model registered at `index`, or `None` for reloadable path/Hub sources."""
        return self._preloaded.get(index)

    def resolve_ids(self, teacher_ids: list[str | None]) -> list[int]:
        """
        Map per-row routing IDs to registry indices.

        Args:
            teacher_ids (`list[str` or `None]`):
                One ID per row. `None` is allowed only when exactly one teacher is registered.

        Returns:
            `list[int]`: the matching registry indices.
        """
        indices = []
        for teacher_id in teacher_ids:
            if teacher_id is None:
                if len(self.entries) != 1:
                    raise ValueError(
                        f"A row has no `teacher_id`, but {len(self.entries)} teachers are registered "
                        f"({sorted(self._by_id)}). Every row needs a known `teacher_id` unless exactly one teacher "
                        "is registered."
                    )
                indices.append(0)
            else:
                indices.append(self[teacher_id].index)
        return indices

    def identity_digest(self) -> str:
        """
        Digest over every entry's identity fields, compared once across ranks by the trainer.

        Covers content digests, Hub repository/revision, precision policy and head identity, but no local path and no
        observed dtype: ranks that stage the same checkpoint under different directories, or that have not scored yet,
        must still agree.
        """
        payload = [
            {
                "teacher_id": entry.teacher_id,
                "index": entry.index,
                "identity_source": entry.identity_source,
                "resolved_revision": entry.resolved_revision,
                "source_key": entry.source_key,
                "config_class": entry.config_class,
                "hidden_size": entry.hidden_size,
                "vocab_size": entry.vocab_size,
                "source_dtype": str(entry.source_dtype),
                "target_dtype": str(entry.target_dtype),
                "projection_dtype": str(entry.projection_dtype),
                "content_digest": entry.content_digest,
                "tokenizer_fingerprint": entry.tokenizer_fingerprint,
                "weight_shape": list(entry.head_identity.weight_shape),
                "has_bias": entry.head_identity.has_bias,
                "logit_scale": entry.head_identity.logit_scale,
                "final_logit_softcapping": entry.head_identity.final_logit_softcapping,
                "transform_version": entry.head_identity.transform_version,
                "adapter_version": entry.adapter_version,
            }
            for entry in self.entries
        ]
        return _sha256_json({"student_tokenizer": self.student_tokenizer_fingerprint, "teachers": payload})

    def manifest(self) -> dict:
        """
        Allowlisted registry description consumed by [`TeacherManifest.from_registry`]; never contains credentials.

        Dtypes stay `torch.dtype` objects and `head["weight_shape"]` a tuple: `from_registry` serializes them. The
        allowlisted `hidden_dtype` is the observed backbone output dtype once a forward has been seen and the planned
        `target_dtype` before that; `hidden_dtype_observed` says which, and `target_dtype` is reported separately so
        the source -> hidden -> target chain stays visible.

        Returns:
            `dict` with keys:
                - `version` (`int`):
                    Manifest schema version.
                - `identity_digest` (`str`):
                    Value of [`~TeacherRegistry.identity_digest`].
                - `student_tokenizer_fingerprint` (`str`):
                    Fingerprint every teacher tokenizer had to match.
                - `teachers` (`list[dict]`):
                    One entry per teacher in registry order, with keys `id`, `index`, `source`,
                    `resolved_revision`, `source_key`, `tokenizer_fingerprint`, `source_dtype`, `hidden_dtype`,
                    `projection_dtype`, `adapter_version`, `head` (`weight_shape`, `has_bias`, `source_dtype`,
                    `logit_scale`, `final_logit_softcapping`, `transform_version`), plus the registry's own
                    `config_class`, `hidden_size`, `vocab_size`, `target_dtype`, `hidden_dtype_observed`,
                    `content_digest`, `content_hashed_bytes`, `evictable`, `storage_bytes` and allowlisted
                    `loading` kwargs.
        """
        teachers = []
        for entry in self.entries:
            loading = {
                key: str(entry.loading_kwargs[key]) for key in _MANIFEST_LOADING_KEYS if key in entry.loading_kwargs
            }
            teachers.append(
                {
                    "id": entry.teacher_id,
                    "index": entry.index,
                    "source": entry.source,
                    "resolved_revision": entry.resolved_revision,
                    "source_key": entry.source_key,
                    "tokenizer_fingerprint": entry.tokenizer_fingerprint,
                    "config_class": entry.config_class,
                    "hidden_size": entry.hidden_size,
                    "vocab_size": entry.vocab_size,
                    "source_dtype": entry.source_dtype,
                    "hidden_dtype": entry.hidden_dtype or entry.target_dtype,
                    "hidden_dtype_observed": entry.hidden_dtype is not None,
                    "target_dtype": entry.target_dtype,
                    "projection_dtype": entry.projection_dtype,
                    "adapter_version": entry.adapter_version,
                    "content_digest": entry.content_digest,
                    "content_hashed_bytes": entry.content_hashed_bytes,
                    "head": {
                        "weight_shape": entry.head_identity.weight_shape,
                        "has_bias": entry.head_identity.has_bias,
                        "source_dtype": entry.head_identity.source_dtype,
                        "logit_scale": entry.head_identity.logit_scale,
                        "final_logit_softcapping": entry.head_identity.final_logit_softcapping,
                        "transform_version": entry.head_identity.transform_version,
                    },
                    "evictable": entry.evictable,
                    "storage_bytes": entry.storage_bytes,
                    "loading": loading,
                }
            )
        return {
            "version": 1,
            "identity_digest": self.identity_digest(),
            "student_tokenizer_fingerprint": self.student_tokenizer_fingerprint,
            "teachers": teachers,
        }

    def _register_path(
        self, teacher_id: str, index: int, source: str, loading_kwargs: dict, teacher_tokenizers: dict
    ) -> TeacherEntry:
        unsupported = sorted(key for key in _UNSUPPORTED_INIT_KWARGS if key in loading_kwargs)
        if unsupported:
            raise ValueError(
                f"Teacher '{teacher_id}' passes unsupported loading options {unsupported}. Managed teachers are "
                "loaded as plain dense CPU models: placement and quantization options are not supported."
            )
        revision = loading_kwargs.get("revision")
        if Path(source).is_dir():
            if revision is not None:
                raise ValueError(
                    f"Teacher '{teacher_id}' points at the local checkpoint '{source}' and also passes "
                    f"revision={revision!r}. Local paths carry no revision; register the two checkpoints under "
                    "separate teacher IDs instead."
                )
            load_path, resolved_revision = source, None
            # A local checkpoint is identified by its content alone: the path is where it happens to be staged, and
            # ranks may stage the same files under different directories.
            identity_source = None
        else:
            load_path, resolved_revision = _resolve_hub_snapshot(
                source, revision, loading_kwargs.get("token", loading_kwargs.get("use_auth_token"))
            )
            identity_source = source
        inventory = _checkpoint_inventory(load_path)
        config = AutoConfig.from_pretrained(load_path, trust_remote_code=self.trust_remote_code)
        text_config = config.get_text_config()
        source_dtype = _resolve_dtype(loading_kwargs, text_config)
        head_shape = inventory["tensors"].get("lm_head.weight")
        weight_shape = (
            (head_shape[1][0], head_shape[1][1])
            if head_shape is not None
            else (text_config.vocab_size, text_config.hidden_size)
        )
        has_bias = "lm_head.bias" in inventory["tensors"]
        source_key = _sha256_json(
            {
                "kind": "path" if resolved_revision is None else "hub",
                "source": identity_source,
                "revision": resolved_revision,
                "content": {
                    "digest": inventory["content_digest"],
                    "config": inventory["config_digest"],
                    "files": inventory["files"],
                    "tensors": inventory["tensors"],
                },
                "loading": _loading_identity(loading_kwargs, source_dtype),
                "adapter_version": _ADAPTER_VERSION,
                "transform_version": _TRANSFORM_VERSION,
            }
        )
        tokenizer = teacher_tokenizers.get(teacher_id)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=self.trust_remote_code)
        return self._build_entry(
            teacher_id=teacher_id,
            index=index,
            source=source,
            identity_source=identity_source,
            resolved_revision=resolved_revision,
            source_key=source_key,
            text_config=text_config,
            config_class=type(config).__name__,
            source_dtype=source_dtype,
            weight_shape=weight_shape,
            has_bias=has_bias,
            tokenizer=tokenizer,
            storage_bytes=inventory["numel"] * _dtype_bytes(source_dtype),
            evictable=True,
            load_path=load_path,
            loading_kwargs=loading_kwargs,
            loading_transient_bytes=inventory["largest_file_bytes"],
            content_digest=inventory["content_digest"],
            content_hashed_bytes=inventory["hashed_bytes"],
        )

    def _register_preloaded(
        self, teacher_id: str, index: int, model: PreTrainedModel, tokenizer, student_storages: set[int]
    ) -> TeacherEntry:
        _teacher_backbone(model)
        devices = {tensor.device.type for tensor in model.parameters()}
        if devices != {"cpu"}:
            raise ValueError(
                f"Preloaded teacher '{teacher_id}' has parameters on {sorted(devices)}. Preloaded teachers must be "
                "frozen CPU sources; the executor uploads disposable copies for scoring."
            )
        if tokenizer is None:
            raise ValueError(
                f"Preloaded teacher '{teacher_id}' needs its tokenizer in `teacher_tokenizers`: shared tokenization "
                "is validated by fingerprint and cannot be resolved from a model object."
            )
        aliased = sorted(
            name
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
            if tensor.untyped_storage().data_ptr() in student_storages
        )
        if aliased:
            raise ValueError(
                f"Preloaded teacher '{teacher_id}' shares storage with the student for {aliased}. Teachers are frozen "
                "targets and must be independent of the student's parameters; pass a separate copy."
            )
        head = model.get_output_embeddings()
        config = model.config
        text_config = config.get_text_config()
        content_digest, hashed_bytes = _tensor_content_digest(model)
        source_key = _sha256_json(
            {
                "kind": "preloaded",
                # A caller model has no file identity, so the routing ID pins the slot and the streamed parameter and
                # buffer values pin the content: re-supplying an equal model on resume matches, a changed one does not.
                "teacher_id": teacher_id,
                "content": content_digest,
                "config": json.loads(config.to_json_string(use_diff=False)),
                "parameters": sorted(
                    [name, str(tensor.dtype), list(tensor.shape)] for name, tensor in model.named_parameters()
                ),
                "adapter_version": _ADAPTER_VERSION,
                "transform_version": _TRANSFORM_VERSION,
            }
        )
        return self._build_entry(
            teacher_id=teacher_id,
            index=index,
            source=None,
            identity_source=None,
            resolved_revision=None,
            source_key=source_key,
            text_config=text_config,
            config_class=type(config).__name__,
            source_dtype=head.weight.dtype,
            weight_shape=(head.weight.shape[0], head.weight.shape[1]),
            has_bias=head.bias is not None,
            tokenizer=tokenizer,
            storage_bytes=sum(
                storage.nbytes()
                for storage in {
                    tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
                    for tensor in list(model.parameters()) + list(model.buffers())
                }.values()
            ),
            evictable=False,
            load_path=None,
            loading_kwargs={},
            loading_transient_bytes=0,
            content_digest=content_digest,
            content_hashed_bytes=hashed_bytes,
        )

    def _build_entry(
        self,
        *,
        teacher_id,
        index,
        source,
        identity_source,
        resolved_revision,
        source_key,
        text_config,
        config_class,
        source_dtype,
        weight_shape,
        has_bias,
        tokenizer,
        storage_bytes,
        evictable,
        load_path,
        loading_kwargs,
        loading_transient_bytes,
        content_digest,
        content_hashed_bytes,
    ) -> TeacherEntry:
        if text_config.vocab_size != self.student_vocab_size:
            raise ValueError(
                f"Teacher '{teacher_id}' has vocab_size {text_config.vocab_size} but the student has vocab_size "
                f"{self.student_vocab_size}. Distillation compares full next-token distributions, which requires a "
                "shared vocabulary."
            )
        if weight_shape[0] != text_config.vocab_size:
            raise ValueError(
                f"Teacher '{teacher_id}' has an output head with {weight_shape[0]} rows but vocab_size "
                f"{text_config.vocab_size}. Managed distillation projects targets through the teacher's own head and "
                "cannot reconcile a padded head."
            )
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint != self.student_tokenizer_fingerprint:
            raise ValueError(
                f"Teacher '{teacher_id}' has tokenizer fingerprint {fingerprint[:12]}... but the student's is "
                f"{self.student_tokenizer_fingerprint[:12]}.... Managed multi-teacher distillation renders prompts "
                "once with the student's processing class and scores those exact token IDs, so the tokenizer "
                "serialization and special-token roles/IDs must be identical."
            )
        logit_scale, softcapping = _logit_transforms(text_config)
        head_identity = HeadIdentity(
            teacher_id=teacher_id,
            source_key=source_key,
            weight_shape=weight_shape,
            has_bias=has_bias,
            source_dtype=source_dtype,
            logit_scale=logit_scale,
            final_logit_softcapping=softcapping,
            transform_version=_TRANSFORM_VERSION,
        )
        return TeacherEntry(
            teacher_id=teacher_id,
            index=index,
            source=source,
            identity_source=identity_source,
            resolved_revision=resolved_revision,
            source_key=source_key,
            config_class=config_class,
            hidden_size=weight_shape[1],
            vocab_size=text_config.vocab_size,
            source_dtype=source_dtype,
            target_dtype=source_dtype,
            hidden_dtype=None,
            projection_dtype=source_dtype,
            tokenizer_fingerprint=fingerprint,
            head_identity=head_identity,
            storage_bytes=storage_bytes,
            evictable=evictable,
            adapter_version=_ADAPTER_VERSION,
            load_path=load_path,
            loading_kwargs=loading_kwargs,
            loading_transient_bytes=loading_transient_bytes,
            content_digest=content_digest,
            content_hashed_bytes=content_hashed_bytes,
        )


def _parameter_storages(model: PreTrainedModel) -> set[int]:
    """Data pointers of every storage `model`'s parameters and buffers own, used for alias rejection."""
    return {tensor.untyped_storage().data_ptr() for tensor in list(model.parameters()) + list(model.buffers())}


def _loading_identity(loading_kwargs: dict, source_dtype: torch.dtype) -> dict:
    """Effective loading/precision settings that belong to the source identity, with credentials removed."""
    secrets = ("token", "use_auth_token", "hf_token")
    settings = {key: str(value) for key, value in loading_kwargs.items() if key not in secrets}
    settings["dtype"] = str(source_dtype)
    settings.pop("revision", None)
    return settings


@dataclass
class HiddenTargetBlock:
    """
    One contiguous CPU allocation of hidden targets, shared by every group of the same width and dtype in a window.

    Groups take zero-copy row slices of `hidden`; `row_samples` records the `(sample_id, completion_position)` each row
    was scored for, so consumption never relies on row order alone.
    """

    block_id: int
    hidden: torch.Tensor
    row_samples: list[tuple]

    @property
    def nbytes(self) -> int:
        return self.hidden.numel() * self.hidden.element_size()


class TargetWriter:
    """
    Bounded writer copying scored device rows into reserved rows of a CPU [`HiddenTargetBlock`].

    Copies go through the executor's staging tile in `_STAGING_ROWS`-row pieces and block until complete
    (`non_blocking=False`), so a tile is never rewritten before its copy finishes and the block never needs a second
    full host copy. The tile carries the block's dtype, so it also performs the hidden-dtype cast.

    Args:
        block ([`HiddenTargetBlock`]):
            Destination block.
        rows (`torch.Tensor`):
            CPU int64 destination row indices, aligned with the request's `positions`.
        staging (`torch.Tensor`):
            Reused staging tile, `(tile_rows, H)` in the block's dtype.
    """

    def __init__(self, block: HiddenTargetBlock, rows: torch.Tensor, staging: torch.Tensor):
        self.block = block
        self.rows = rows
        self.staging = staging
        self.rows_written = 0

    def write(self, offset: int, hidden: torch.Tensor) -> None:
        """
        Copy `hidden` into the rows reserved at `offset`.

        Args:
            offset (`int`):
                Index into `rows` the first copied row belongs to.
            hidden (`torch.Tensor`):
                Scored rows `(n, H)` on the scoring device, in `rows[offset : offset + n]` order.
        """
        count = hidden.shape[0]
        if offset + count > self.rows.numel():
            raise RuntimeError(
                f"teacher scoring wrote {offset + count} rows into a block slice reserved for {self.rows.numel()}"
            )
        tile_rows = self.staging.shape[0]
        for start in range(0, count, tile_rows):
            piece = hidden[start : start + tile_rows]
            staged = self.staging[: piece.shape[0]]
            staged.copy_(piece, non_blocking=False)
            self.block.hidden.index_copy_(0, self.rows[offset + start : offset + start + piece.shape[0]], staged)
        self.rows_written += count


@dataclass(frozen=True)
class ScoreRequest:
    """
    One teacher's scoring request for one microbatch: exact tokens, alignment metadata and requested positions.

    `input_ids`/`attention_mask` are the generation payload's `prompt_ids`/`completion_ids` (left-padded prompts,
    right-padded completions) concatenated; `positions` are the flat `b * K + k` completion positions whose hidden
    states become targets, in ascending order. Masked positions stay in the input context and simply are not requested.
    """

    generation_id: int
    microbatch_index: int
    teacher_index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int
    completion_length: int
    positions: torch.Tensor
    sample_ids: tuple
    representation_version: int = 1


@dataclass(frozen=True)
class ScoreResult:
    """Completed target metadata for one [`ScoreRequest`]."""

    teacher_index: int
    rows_written: int
    hidden_dtype: torch.dtype
    forward_calls: int


@dataclass
class ExecutorStats:
    """
    Performance counters of the scoring executor; the `device` counters cover the disposable scoring copies.

    `body_loads` counts CPU body materializations and `cpu_reloads` the subset that re-materialized a previously
    evicted source.
    """

    body_loads: int = 0
    cpu_reloads: int = 0
    disk_bytes: int = 0
    verify_bytes: int = 0
    backbone_uploads: int = 0
    upload_bytes: int = 0
    upload_seconds: float = 0.0
    forward_calls: int = 0
    rows_scored: int = 0
    live_cpu_weight_bytes: int = 0
    peak_cpu_weight_bytes: int = 0
    live_device_weight_bytes: int = 0
    peak_device_weight_bytes: int = 0
    staging_bytes: int = 0


class TeacherExecutor:
    """
    Loads teacher bodies, scores exact generated tokens and retains the exact CPU head sources.

    One reloadable CPU body slot serves every path/Hub source: loading another source releases the current body's
    module and parameter dictionaries, and the replacement is reloaded from the registry's pinned local snapshot, so
    host residency depends on the largest teacher instead of the number registered. Caller-provided models are
    borrowed, never evicted, and charged against the CPU budget for the executor's lifetime. Head sources are compact
    CPU copies retained independently of the body, so a checkpointed loss replay can re-project a teacher whose body
    is long gone without any disk or network work.

    Scoring substitutes disposable device parameters/buffers into the backbone with `torch.func.functional_call`
    (`tie_weights=True`, `strict=True`), under an explicit `torch.no_grad()` and the recorded autocast context, since
    `_prepare_inputs` runs outside the student wrapper's precision context. The backbone is asked for full-length
    outputs (never `logits_to_keep`), the output length is verified, and only then is the causal shift applied and the
    requested completion positions selected — the same arithmetic as
    `DistillationTrainer._get_last_hidden_state` followed by the loss mask.

    Args:
        registry ([`TeacherRegistry`]):
            Registry whose entries this executor serves.
        head_cache (`TeacherHeadCache`):
            One-slot device head cache the retained head sources are registered with.
        device (`torch.device`):
            Device the scoring forward runs on.
        cpu_weight_budget_bytes (`int`, *optional*):
            Cap on retained head sources, non-evictable models, the body and its loading transient.
        gpu_weight_budget_bytes (`int`, *optional*):
            Cap on the disposable device weights of one scoring forward.
        scoring_batch_size (`int`, *optional*, defaults to `1`):
            Maximum rows per teacher forward.
        autocast_dtype (`torch.dtype`, *optional*):
            When set, scoring runs under `torch.autocast(device_type, dtype=autocast_dtype)` and targets are stored in
            that dtype. When unset there is no autocast and targets are stored in the source dtype.
        staging_rows (`int`, *optional*, defaults to `256`):
            Rows per staging tile fill.
    """

    def __init__(
        self,
        registry: TeacherRegistry,
        head_cache,
        device: torch.device,
        *,
        cpu_weight_budget_bytes: int | None = None,
        gpu_weight_budget_bytes: int | None = None,
        scoring_batch_size: int = 1,
        autocast_dtype: torch.dtype | None = None,
        staging_rows: int = _STAGING_ROWS,
    ):
        self.registry = registry
        self.head_cache = head_cache
        self.device = torch.device(device)
        self.cpu_weight_budget_bytes = cpu_weight_budget_bytes
        self.gpu_weight_budget_bytes = gpu_weight_budget_bytes
        self.scoring_batch_size = scoring_batch_size
        self.autocast_dtype = autocast_dtype
        self.staging_rows = staging_rows
        self.capabilities = {"hidden_targets": True, "requires_collective_schedule": False}
        self.stats = ExecutorStats()
        # Recorded precision policy: targets are stored in `target_dtype` and `projection_dtype` is the matmul
        # execution dtype the managed loss reproduces on replay. `hidden_dtype` stays unset until a real forward (or
        # `probe_hidden_dtype`) shows what the backbone returns; it is never inferred from configuration.
        for entry in registry.entries:
            entry.target_dtype = autocast_dtype or entry.source_dtype
            entry.projection_dtype = autocast_dtype or entry.source_dtype
        self._body: PreTrainedModel | None = None
        self._body_key: str | None = None
        self._loaded_keys: set[str] = set()
        self._head_sources: dict[int, HeadSource] = {}
        self._head_refcounts: dict[int, int] = {}
        self._staging: torch.Tensor | None = None
        self._closed = False
        self._non_evictable_bytes = sum(entry.storage_bytes for entry in registry.entries if not entry.evictable)
        self._account_cpu()

    def retain_head_source(self, index: int) -> HeadSource:
        """
        Retain the exact CPU head of teacher `index`, registering it with the head cache on first use.

        Args:
            index (`int`):
                Registry index.

        Returns:
            [`HeadSource`]: the retained CPU head, valid until the matching number of
            [`~TeacherExecutor.release_head_source`] calls.
        """
        self._check_open()
        if index not in self._head_sources:
            self._head_sources[index] = self._build_head_source(self.registry.entries[index])
            self.head_cache.retain_head_source(self._head_sources[index])
            self._account_cpu()
        self._head_refcounts[index] = self._head_refcounts.get(index, 0) + 1
        return self._head_sources[index]

    def release_head_source(self, index: int) -> None:
        """Drop one retention of teacher `index`'s head source, unregistering it when the last one goes away."""
        count = self._head_refcounts.get(index, 0)
        if count == 0:
            raise RuntimeError(f"head source for teacher index {index} is not retained")
        if count > 1:
            self._head_refcounts[index] = count - 1
            return
        self._head_refcounts.pop(index)
        source = self._head_sources.pop(index)
        self.head_cache.release_head_source(source.identity)
        self._account_cpu()

    def score(self, request: ScoreRequest, writer: TargetWriter) -> ScoreResult:
        """
        Score one microbatch's rows for one teacher and copy the requested hidden states into reserved CPU rows.

        Args:
            request ([`ScoreRequest`]):
                Tokens, alignment metadata and requested completion positions.
            writer ([`TargetWriter`]):
                Writer bound to the reserved block rows, in `request.positions` order.

        Returns:
            [`ScoreResult`]: the completed target metadata.
        """
        self._check_open()
        entry = self.registry.entries[request.teacher_index]
        positions = request.positions
        if positions.numel() != writer.rows.numel():
            raise RuntimeError(
                f"teacher '{entry.teacher_id}' was asked for {positions.numel()} targets but {writer.rows.numel()} "
                "block rows are reserved"
            )
        if positions.numel() > 1 and bool((positions[1:] <= positions[:-1]).any()):
            raise RuntimeError(
                f"teacher '{entry.teacher_id}' received unsorted completion positions; row order must follow the "
                "(sample, position) bookkeeping"
            )
        prompt_length, completion_length = request.prompt_length, request.completion_length
        if request.input_ids.shape[1] != prompt_length + completion_length:
            raise RuntimeError(
                f"scoring request has {request.input_ids.shape[1]} tokens but prompt_length={prompt_length} and "
                f"completion_length={completion_length}"
            )
        body = self._load_body(request.teacher_index)
        backbone = _teacher_backbone(body)
        device_state, device_bytes = self._device_state(backbone)
        forward_calls = 0
        offset = 0
        try:
            for start in range(0, request.input_ids.shape[0], self.scoring_batch_size):
                stop = min(start + self.scoring_batch_size, request.input_ids.shape[0])
                selected = positions[(positions >= start * completion_length) & (positions < stop * completion_length)]
                if selected.numel() == 0:
                    continue
                input_ids = request.input_ids[start:stop].to(self.device)
                attention_mask = request.attention_mask[start:stop].to(self.device)
                with torch.no_grad(), self._autocast():
                    output = torch.func.functional_call(
                        backbone,
                        device_state,
                        args=(),
                        kwargs={"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False},
                        tie_weights=True,
                        strict=True,
                    )
                hidden = output.last_hidden_state
                self._record_hidden_dtype(entry, hidden.dtype)
                if hidden.shape[1] != input_ids.shape[1]:
                    raise RuntimeError(
                        f"teacher '{entry.teacher_id}' returned {hidden.shape[1]} hidden states for "
                        f"{input_ids.shape[1]} input tokens; the managed adapter requires full-length outputs before "
                        "the causal shift"
                    )
                # Same alignment as `_get_last_hidden_state`: drop the next-token prediction, then keep the completion
                # window, so row `k` is the state that predicts completion token `k`.
                completion_hidden = hidden[:, :-1][:, prompt_length - 1 : prompt_length - 1 + completion_length]
                rows = completion_hidden.reshape(-1, completion_hidden.shape[-1])
                local = (selected - start * completion_length).to(self.device)
                writer.write(offset, rows.index_select(0, local))
                offset += selected.numel()
                forward_calls += 1
        finally:
            device_state.clear()
            self.stats.live_device_weight_bytes -= device_bytes
        self.stats.forward_calls += forward_calls
        self.stats.rows_scored += offset
        if offset != positions.numel():
            raise RuntimeError(
                f"teacher '{entry.teacher_id}' scored {offset} of {positions.numel()} requested targets"
            )
        return ScoreResult(request.teacher_index, offset, entry.hidden_dtype, forward_calls)

    def align_target_dtypes(self) -> None:
        """
        Store targets in the dtype the backbones really return, probing every teacher that has not been observed.

        Target blocks are planned and written in `target_dtype`, which defaults to the autocast dtype. When a backbone
        returns something wider than that — CPU autocast keeps the final RMS norm in float32, for instance — storing
        targets in the autocast dtype rounds the teacher's own output and changes the objective relative to the
        single-teacher path, which keeps the teacher hidden states exactly as its forward produced them. Call this
        once before the first [`~WindowStore.plan_window`] so storage is lossless; it costs one two-token forward per
        teacher (and one body load, through the same single slot as scoring).
        """
        for entry in self.registry.entries:
            if entry.hidden_dtype is None:
                self.probe_hidden_dtype(entry.index)
            entry.target_dtype = entry.hidden_dtype

    def probe_hidden_dtype(self, index: int) -> torch.dtype:
        """
        Observe teacher `index`'s backbone output dtype with a two-token forward under the recorded autocast policy.

        The trainer can call this before planning a window or saving a manifest so the entry carries a measured
        `hidden_dtype` instead of a configuration guess. Scoring records the same value on its first forward, so this
        is optional.

        Args:
            index (`int`):
                Registry index.

        Returns:
            `torch.dtype`: the dtype the backbone returned.
        """
        self._check_open()
        entry = self.registry.entries[index]
        backbone = _teacher_backbone(self._load_body(index))
        device_state, device_bytes = self._device_state(backbone)
        try:
            input_ids = torch.zeros((1, 2), dtype=torch.long, device=self.device)
            with torch.no_grad(), self._autocast():
                output = torch.func.functional_call(
                    backbone,
                    device_state,
                    args=(),
                    kwargs={
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                        "use_cache": False,
                    },
                    tie_weights=True,
                    strict=True,
                )
            self._record_hidden_dtype(entry, output.last_hidden_state.dtype)
        finally:
            device_state.clear()
            self.stats.live_device_weight_bytes -= device_bytes
        return entry.hidden_dtype

    def _record_hidden_dtype(self, entry: TeacherEntry, dtype: torch.dtype) -> None:
        """Record the dtype the backbone actually returned, before any conversion into the target block dtype."""
        if entry.hidden_dtype is None:
            entry.hidden_dtype = dtype
            logger.debug(
                "Teacher '%s' backbone returns %s; targets are stored as %s",
                entry.teacher_id,
                dtype,
                entry.target_dtype,
            )
        elif entry.hidden_dtype != dtype:
            raise RuntimeError(
                f"teacher '{entry.teacher_id}' returned {dtype} hidden states but {entry.hidden_dtype} was recorded "
                "earlier; the scoring precision policy must be identical for every forward"
            )

    def target_writer(self, block: HiddenTargetBlock, rows: torch.Tensor) -> TargetWriter:
        """Bind a writer for `rows` of `block` to the staging tile, reallocating the tile when its shape changes."""
        width = block.hidden.shape[1]
        if (
            self._staging is None
            or self._staging.shape[1] != width
            or self._staging.dtype != block.hidden.dtype
            or self._staging.shape[0] != self.staging_rows
        ):
            self._staging = torch.empty(
                (self.staging_rows, width), dtype=block.hidden.dtype, pin_memory=self.device.type == "cuda"
            )
            self.stats.staging_bytes = self._staging.numel() * self._staging.element_size()
        return TargetWriter(block, rows, self._staging)

    def evict_idle_gpu(self) -> None:
        """Release idle device allocations: the head cache's slot. Scoring weights are released by [`score`] itself."""
        self.head_cache.evict_idle_gpu()

    def close(self) -> None:
        """Release the body, the staging tile and every head source; rejects live target consumers. Idempotent."""
        if self._head_refcounts:
            raise RuntimeError(
                f"cannot close the teacher executor while head sources are retained for teacher indices "
                f"{sorted(self._head_refcounts)}; release the live targets first"
            )
        for index in list(self._head_sources):
            source = self._head_sources.pop(index)
            self.head_cache.release_head_source(source.identity)
        self._release_body()
        self._staging = None
        self.stats.staging_bytes = 0
        self._closed = True
        self._account_cpu()

    def reopen(self) -> None:
        """Allow loading again after [`close`]; bodies are reloaded from the registry's pinned snapshots on demand."""
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the teacher executor is closed; call `reopen()` before scoring again")

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _load_body(self, index: int) -> PreTrainedModel:
        """Return the CPU body of teacher `index`, evicting the current occupant of the single slot when needed."""
        self._check_open()
        entry = self.registry.entries[index]
        if not entry.evictable:
            model = self.registry.preloaded_model(index)
            model.eval()
            return model
        if self._body_key == entry.source_key:
            return self._body
        self._release_body()
        budget = self.cpu_weight_budget_bytes
        required = self._live_cpu_bytes() + entry.storage_bytes + entry.loading_transient_bytes
        if budget is not None and required > budget:
            raise ValueError(
                f"Loading teacher '{entry.teacher_id}' needs {required} bytes of host weights (retained heads, "
                f"non-evictable models, its {entry.storage_bytes}-byte body and a "
                f"{entry.loading_transient_bytes}-byte loading transient) but `teacher_cpu_weight_budget_bytes` is "
                f"{budget}."
            )
        digest, hashed = _content_digest(entry.load_path)
        self.stats.verify_bytes += hashed
        if digest != entry.content_digest:
            raise ValueError(
                f"The checkpoint of teacher '{entry.teacher_id}' at '{entry.load_path}' changed since registration "
                f"(content digest {digest[:12]}... instead of {entry.content_digest[:12]}...). Managed multi-teacher "
                "distillation pins immutable sources: retained head sources, cached targets and the saved manifest "
                "all describe the registered content. Point the teacher at an immutable snapshot, or restart training "
                "so it is registered again."
            )
        loading_kwargs = {key: value for key, value in entry.loading_kwargs.items() if key != "revision"}
        loading_kwargs["dtype"] = entry.source_dtype
        loading_kwargs["device_map"] = None
        loading_kwargs.setdefault("low_cpu_mem_usage", True)
        loading_kwargs.setdefault("trust_remote_code", self.registry.trust_remote_code)
        with _plain_loading_env():
            model = create_model_from_path(entry.load_path, **loading_kwargs)
        model.eval()
        model.requires_grad_(False)
        self._body = model
        self._body_key = entry.source_key
        self.stats.body_loads += 1
        self.stats.disk_bytes += entry.storage_bytes
        if entry.source_key in self._loaded_keys:
            self.stats.cpu_reloads += 1
        self._loaded_keys.add(entry.source_key)
        self._account_cpu()
        logger.debug(
            "Loaded teacher '%s' body from %s (%d bytes)", entry.teacher_id, entry.load_path, entry.storage_bytes
        )
        return model

    def _release_body(self) -> None:
        """Drop every alias of the current body: the module, its parameter/buffer dictionaries and their storages."""
        if self._body is None:
            return
        body = self._body
        self._body = None
        self._body_key = None
        # A CPU module keeps its storages alive through its own `_parameters`/`_buffers` dicts, so dropping our
        # reference is not enough if anything else (a profiler, a traceback frame) still sees the module. Clearing the
        # dicts releases the weights either way; a retained head source keeps only its own tensor alive.
        for module in body.modules():
            module._parameters.clear()
            module._buffers.clear()
        del body
        self._account_cpu()

    def _build_head_source(self, entry: TeacherEntry) -> HeadSource:
        body = self._load_body(entry.index)
        head = body.get_output_embeddings()
        weight = head.weight.detach()
        if weight.shape != entry.head_identity.weight_shape or weight.dtype != entry.head_identity.source_dtype:
            raise RuntimeError(
                f"teacher '{entry.teacher_id}' registered head {entry.head_identity.weight_shape} "
                f"{entry.head_identity.source_dtype} but loaded {tuple(weight.shape)} {weight.dtype}"
            )
        pins_extra_storage = weight.untyped_storage().nbytes() != weight.numel() * weight.element_size()
        if body.config.get_text_config().tie_word_embeddings or pins_extra_storage or not weight.is_contiguous():
            # Tied or viewing heads would keep the input embedding (or an unrelated flattened buffer) alive for as long
            # as any target is live; a compact copy costs one head transient and frees the body completely.
            weight = weight.clone(memory_format=torch.contiguous_format)
        bias = head.bias.detach().clone() if head.bias is not None else None
        return HeadSource(identity=entry.head_identity, weight=weight, bias=bias)

    def _device_state(self, backbone) -> tuple[dict, int]:
        """
        Build the disposable device parameter/buffer dict for one scoring forward.

        `named_buffers()` yields non-persistent buffers too (persistence only affects `state_dict`), and tied
        parameters appear once, which `functional_call(tie_weights=True)` re-ties. On a CPU device `.to()` returns the
        source tensors themselves, so no copy happens there; the counters still record the dict build so eviction and
        transfer accounting stay observable without an accelerator.
        """
        started = time.perf_counter()
        state = {name: tensor.detach().to(self.device) for name, tensor in backbone.named_parameters()}
        state.update({name: tensor.detach().to(self.device) for name, tensor in backbone.named_buffers()})
        missing = sorted(set(backbone.state_dict().keys()) - set(state))
        if missing:
            raise RuntimeError(
                f"the teacher backbone exposes state entries {missing} that the scoring adapter cannot substitute"
            )
        device_bytes = sum(
            storage.nbytes()
            for storage in {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage() for tensor in state.values()
            }.values()
        )
        if self.gpu_weight_budget_bytes is not None and device_bytes > self.gpu_weight_budget_bytes:
            state.clear()
            raise ValueError(
                f"Scoring this teacher needs {device_bytes} bytes of device weights but "
                f"`teacher_gpu_weight_budget_bytes` is {self.gpu_weight_budget_bytes}."
            )
        self.stats.backbone_uploads += 1
        self.stats.upload_bytes += device_bytes
        self.stats.upload_seconds += time.perf_counter() - started
        self.stats.live_device_weight_bytes += device_bytes
        self.stats.peak_device_weight_bytes = max(
            self.stats.peak_device_weight_bytes, self.stats.live_device_weight_bytes
        )
        return state, device_bytes

    def _live_cpu_bytes(self) -> int:
        body_bytes = 0 if self._body_key is None else self.registry.entries[self._body_index()].storage_bytes
        head_bytes = sum(self.registry.entries[index].head_bytes for index in self._head_sources)
        return self._non_evictable_bytes + body_bytes + head_bytes

    def _body_index(self) -> int:
        return next(entry.index for entry in self.registry.entries if entry.source_key == self._body_key)

    def _account_cpu(self) -> None:
        self.stats.live_cpu_weight_bytes = self._live_cpu_bytes()
        self.stats.peak_cpu_weight_bytes = max(self.stats.peak_cpu_weight_bytes, self.stats.live_cpu_weight_bytes)


@dataclass(frozen=True)
class _PlannedGroup:
    """One teacher's requested target rows inside one microbatch of the planned window."""

    microbatch_index: int
    teacher_index: int
    positions: torch.Tensor
    block_key: tuple


@dataclass
class WindowPlan:
    """A consecutive window of whole microbatches that fits the CPU target and host weight budgets."""

    generation_id: int
    microbatch_indices: list[int]
    groups: list[_PlannedGroup]
    block_rows: dict[tuple, int]
    teacher_indices: list[int]
    target_bytes: int
    weight_bytes: int


class WindowStore:
    """
    Owns the CPU hidden targets of the current scoring window, keyed by `(generation_id, microbatch_index)`.

    A window is a prefix of whole microbatches of the current generation batch: planning never splits a microbatch or
    truncates the objective, it only shrinks the window. Targets of one width and dtype share one contiguous block, so
    each [`TargetGroup`] is a zero-copy row slice; a block and the head sources it needs are released after the last
    consuming microbatch finishes. Consumption is exact-once: a released key cannot be read again, and the token/mask
    fingerprint recorded at scoring time must match the microbatch presented for training.

    Args:
        registry ([`TeacherRegistry`]):
            Registry the planned teacher indices refer to. The executor must already have recorded its precision
            policy on the entries (it does so at construction), because planning sizes blocks in `hidden_dtype`.
    """

    def __init__(self, registry: TeacherRegistry):
        self.registry = registry
        self._executor: TeacherExecutor | None = None
        self._blocks: dict[int, HiddenTargetBlock] = {}
        self._block_refs: dict[int, int] = {}
        self._targets: dict[tuple[int, int], list[TargetGroup]] = {}
        self._key_blocks: dict[tuple[int, int], list[int]] = {}
        self._key_teachers: dict[tuple[int, int], list[int]] = {}
        self._consumers: dict[tuple[int, int], int] = {}
        self._fingerprints: dict[tuple[int, int], tuple] = {}
        self._released: set[tuple[int, int]] = set()
        self._next_block_id = 0

    def plan_window(
        self,
        microbatches: list[dict],
        *,
        target_cache_bytes: int,
        cpu_weight_budget_bytes: int | None = None,
        generation_id: int = 0,
        start_index: int = 0,
    ) -> WindowPlan:
        """
        Choose the largest prefix of whole microbatches whose targets and teacher weights fit the budgets.

        Args:
            microbatches (`list[dict]`):
                The split microbatch dicts, in consumption order. Each needs `completion_mask`, a per-row
                `teacher_index` (registry indices, as returned by [`~TeacherRegistry.resolve_ids`]) and optionally
                `tool_mask`; masked positions stay in the input context but get no target.
            target_cache_bytes (`int`):
                Per-rank CPU target allocation limit (`teacher_target_cache_bytes`).
            cpu_weight_budget_bytes (`int`, *optional*):
                Cap on retained head sources, non-evictable models and the largest body plus its loading transient
                (`teacher_cpu_weight_budget_bytes`).
            generation_id (`int`, *optional*, defaults to `0`):
                ID of the generation batch these microbatches come from; part of every target key.
            start_index (`int`, *optional*, defaults to `0`):
                First microbatch of this window. Pass the full generation batch and the index the previous window
                ended at to plan the next window; `microbatch_index` stays absolute in every target key.

        Returns:
            [`WindowPlan`]: the planned window.
        """
        groups: list[_PlannedGroup] = []
        per_microbatch = [
            self._microbatch_groups(microbatches[index], index) for index in range(start_index, len(microbatches))
        ]
        best = 0
        best_bytes = (0, 0)
        for count in range(1, len(per_microbatch) + 1):
            window = [group for entry in per_microbatch[:count] for group in entry]
            target_bytes = self._target_bytes(window)
            weight_bytes = self._weight_bytes(window)
            if target_bytes > target_cache_bytes or (
                cpu_weight_budget_bytes is not None and weight_bytes > cpu_weight_budget_bytes
            ):
                if count == 1:
                    control, required, budget = (
                        ("teacher_target_cache_bytes", target_bytes, target_cache_bytes)
                        if target_bytes > target_cache_bytes
                        else ("teacher_cpu_weight_budget_bytes", weight_bytes, cpu_weight_budget_bytes)
                    )
                    raise ValueError(
                        f"Microbatch {start_index} of generation batch {generation_id} needs {required} bytes but "
                        f"`{control}` is {budget}. Raise `{control}`, or lower the per-device batch size / "
                        "completion length; the objective is never truncated to make a microbatch fit."
                    )
                break
            best, best_bytes = count, (target_bytes, weight_bytes)
            groups = window
        teacher_indices = sorted({group.teacher_index for group in groups})
        block_rows: dict[tuple, int] = {}
        for group in groups:
            block_rows[group.block_key] = block_rows.get(group.block_key, 0) + group.positions.numel()
        plan = WindowPlan(
            generation_id=generation_id,
            microbatch_indices=list(range(start_index, start_index + best)),
            groups=groups,
            block_rows=block_rows,
            teacher_indices=teacher_indices,
            target_bytes=best_bytes[0],
            weight_bytes=best_bytes[1],
        )
        logger.info(
            "Planned teacher scoring window: microbatches %d-%d of %d, %d target bytes in %d block(s), %d host "
            "weight bytes, %d teacher load(s) for teachers %s",
            start_index,
            start_index + best - 1,
            len(microbatches),
            plan.target_bytes,
            len(block_rows),
            plan.weight_bytes,
            len(teacher_indices),
            [self.registry.entries[index].teacher_id for index in teacher_indices],
        )
        return plan

    def score_window(self, plan: WindowPlan, executor: TeacherExecutor, inputs: list[dict]) -> None:
        """
        Fill the window's target blocks, loading one teacher at a time and releasing its device weights after its rows.

        Args:
            plan ([`WindowPlan`]):
                Plan returned by [`~WindowStore.plan_window`].
            executor ([`TeacherExecutor`]):
                Executor scoring the rows and retaining the head sources.
            inputs (`list[dict]`):
                The same microbatch dicts that were planned, indexable by `microbatch_index`. Each needs
                `prompt_ids`, `prompt_mask`, `completion_ids`, `completion_mask`, a per-row `teacher_index` and
                optionally `tool_mask` and `sample_ids`.
        """
        self._executor = executor
        keys = [(plan.generation_id, index) for index in plan.microbatch_indices]
        live = [key for key in keys if key in self._targets]
        if live:
            raise RuntimeError(
                f"target keys {live} are still live; release the previous window (or call `reset()`) before scoring a "
                "new one"
            )
        blocks = {}
        for block_key, rows in plan.block_rows.items():
            width, dtype = block_key
            self._next_block_id += 1
            block = HiddenTargetBlock(self._next_block_id, torch.zeros((rows, width), dtype=dtype), [None] * rows)
            blocks[block_key] = block
            self._blocks[block.block_id] = block
            self._block_refs[block.block_id] = 0
        offsets = dict.fromkeys(plan.block_rows, 0)
        ranges = []
        for group in plan.groups:
            start = offsets[group.block_key]
            offsets[group.block_key] = start + group.positions.numel()
            ranges.append((start, offsets[group.block_key]))
        for key in keys:
            self._targets[key] = []
            self._key_blocks[key] = []
            self._key_teachers[key] = []
            self._consumers[key] = 1
            self._fingerprints[key] = _microbatch_fingerprint(inputs[key[1]])
            self._released.discard(key)
        for teacher_index in plan.teacher_indices:
            entry = self.registry.entries[teacher_index]
            for group, (start, end) in zip(plan.groups, ranges, strict=True):
                if group.teacher_index != teacher_index:
                    continue
                microbatch = inputs[group.microbatch_index]
                block = blocks[group.block_key]
                key = (plan.generation_id, group.microbatch_index)
                executor.retain_head_source(teacher_index)
                completion_length = microbatch["completion_ids"].shape[1]
                sample_ids = _sample_ids(microbatch, plan.generation_id, group.microbatch_index)
                request = ScoreRequest(
                    generation_id=plan.generation_id,
                    microbatch_index=group.microbatch_index,
                    teacher_index=teacher_index,
                    input_ids=torch.cat([microbatch["prompt_ids"], microbatch["completion_ids"]], dim=1),
                    attention_mask=torch.cat([microbatch["prompt_mask"], microbatch["completion_mask"]], dim=1),
                    prompt_length=microbatch["prompt_ids"].shape[1],
                    completion_length=completion_length,
                    positions=group.positions,
                    sample_ids=sample_ids,
                )
                writer = executor.target_writer(block, torch.arange(start, end))
                executor.score(request, writer)
                for row, position in enumerate(group.positions.tolist()):
                    block.row_samples[start + row] = (
                        sample_ids[position // completion_length],
                        position % completion_length,
                    )
                self._targets[key].append(
                    TargetGroup(
                        identity=entry.head_identity,
                        teacher_index=teacher_index,
                        hidden=block.hidden[start:end],
                        positions=group.positions,
                    )
                )
                self._key_blocks[key].append(block.block_id)
                self._key_teachers[key].append(teacher_index)
                self._block_refs[block.block_id] += 1

    @property
    def live_keys(self) -> list[tuple[int, int]]:
        """Keys whose targets are scored and not yet released, so the trainer can refuse to close teachers."""
        return sorted(self._targets)

    def targets_for(self, key: tuple[int, int], microbatch: dict | None = None) -> list[TargetGroup]:
        """
        Return the target groups of one microbatch as zero-copy views.

        Args:
            key (`tuple[int, int]`):
                `(generation_id, microbatch_index)`.
            microbatch (`dict`, *optional*):
                The microbatch about to be trained on. When given, its token/mask fingerprint must match the one
                recorded at scoring time.

        Returns:
            `list[TargetGroup]`: one group per teacher present in the microbatch, ordered by registry index.
        """
        if key in self._released:
            raise RuntimeError(f"targets for {key} were already released; each microbatch consumes its targets once")
        if key not in self._targets:
            raise RuntimeError(f"no scored targets for {key}; live keys are {sorted(self._targets)}")
        if microbatch is not None and _microbatch_fingerprint(microbatch) != self._fingerprints[key]:
            raise RuntimeError(
                f"the microbatch presented for {key} does not match the tokens and masks its targets were scored for"
            )
        return self._targets[key]

    def release(self, key: tuple[int, int]) -> None:
        """Drop one consumer of `key`; the last one frees its block rows and head-source retentions."""
        if key not in self._consumers:
            raise RuntimeError(f"no scored targets for {key}; live keys are {sorted(self._targets)}")
        self._consumers[key] -= 1
        if self._consumers[key] > 0:
            return
        self._free_key(key)
        self._released.add(key)

    def reset(self) -> None:
        """Drop every window: used when the generation buffer is renewed or training resumes from a checkpoint."""
        for key in list(self._consumers):
            self._free_key(key)
        self._released.clear()
        self._blocks.clear()
        self._block_refs.clear()

    def _free_key(self, key: tuple[int, int]) -> None:
        for teacher_index in self._key_teachers.pop(key):
            self._executor.release_head_source(teacher_index)
        for block_id in self._key_blocks.pop(key):
            self._block_refs[block_id] -= 1
            if self._block_refs[block_id] == 0:
                self._blocks.pop(block_id)
                self._block_refs.pop(block_id)
        self._targets.pop(key)
        self._consumers.pop(key)
        self._fingerprints.pop(key)

    def _microbatch_groups(self, microbatch: dict, index: int) -> list[_PlannedGroup]:
        loss_mask = _loss_mask(microbatch)
        if loss_mask.shape[0] == 0:
            raise ValueError(
                f"microbatch {index} has no rows; managed multi-teacher distillation requires nonempty scheduled "
                "microbatches"
            )
        completion_length = loss_mask.shape[1]
        row_teacher = microbatch["teacher_index"].repeat_interleave(completion_length)
        flat = loss_mask.reshape(-1) > 0
        groups = []
        for teacher_index in sorted(set(microbatch["teacher_index"].tolist())):
            entry = self.registry.entries[teacher_index]
            positions = (flat & (row_teacher == teacher_index)).nonzero().flatten()
            if positions.numel() > 0:
                groups.append(_PlannedGroup(index, teacher_index, positions, (entry.hidden_size, entry.target_dtype)))
        return groups

    def _target_bytes(self, groups: list[_PlannedGroup]) -> int:
        block_keys = {group.block_key for group in groups}
        rows = sum(group.positions.numel() * group.block_key[0] * _dtype_bytes(group.block_key[1]) for group in groups)
        return rows + len(block_keys) * _TARGET_BLOCK_OVERHEAD_BYTES

    def _weight_bytes(self, groups: list[_PlannedGroup]) -> int:
        """Host weight bytes the window needs: retained heads, non-evictable models and the largest body it loads."""
        present = {group.teacher_index for group in groups}
        entries = [self.registry.entries[index] for index in present]
        heads = sum(entry.head_bytes for entry in entries)
        non_evictable = sum(entry.storage_bytes for entry in self.registry.entries if not entry.evictable)
        bodies = [entry.storage_bytes + entry.loading_transient_bytes for entry in entries if entry.evictable]
        return heads + non_evictable + (max(bodies) if bodies else 0)


def _loss_mask(microbatch: dict) -> torch.Tensor:
    """Completion mask restricted to the positions the loss trains on, matching `DistillationTrainer._compute_loss`."""
    completion_mask = microbatch["completion_mask"]
    return completion_mask if "tool_mask" not in microbatch else completion_mask * microbatch["tool_mask"]


def _microbatch_fingerprint(microbatch: dict) -> tuple:
    """Token/mask shapes and content summary validated before targets are consumed."""
    loss_mask = _loss_mask(microbatch)
    return (
        tuple(microbatch["prompt_ids"].shape),
        tuple(microbatch["completion_ids"].shape),
        int(loss_mask.sum()),
        tuple(int(index) for index in microbatch["teacher_index"]),
    )


def _sample_ids(microbatch: dict, generation_id: int, microbatch_index: int) -> tuple:
    """Stable per-row IDs; synthesized from the window key when the generation payload carries none."""
    if "sample_ids" in microbatch:
        return tuple(microbatch["sample_ids"])
    return tuple(f"{generation_id}:{microbatch_index}:{row}" for row in range(microbatch["completion_ids"].shape[0]))
