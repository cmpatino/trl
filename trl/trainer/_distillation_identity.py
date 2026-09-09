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

"""Tokenizer fingerprinting, the teacher manifest, and metric-key helpers for managed multi-teacher distillation.

Shared seam between the teacher registry (`_distillation_teacher.py`, owned by another worker), the managed loss
(`_distillation_loss.py`), and the trainer. See `/data/workspaces/mopd/implementation/interfaces.md` for the contract.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from transformers import PreTrainedTokenizerFast


def canonical_tokenizer_payload(tokenizer) -> dict:
    """
    Build the canonical, JSON-serializable payload used to fingerprint a tokenizer.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to canonicalize. Must be a fast (Rust-backed) tokenizer.

    Returns:
        `dict`: The parsed complete fast-tokenizer serialization (vocabulary, merges, added tokens, normalizer,
        pre-tokenizer, post-processor, decoder) with two additional top-level keys: `special_tokens`, mapping each
        special-token role to its `(token string, token id)` pair, and `model_input_names`. `model_input_names` is
        included because it changes which tensors the tokenizer produces from the same token IDs (e.g. whether
        `token_type_ids` are rendered), which changes what the student/teacher actually consume; `padding_side` and
        `truncation_side` are excluded because they only affect batch layout around already-decided token IDs, not
        the IDs themselves.

    Raises:
        `TypeError`: If `tokenizer` is not a fast tokenizer. Slow tokenizers need a dedicated validator; none exists
            yet.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"Tokenizer fingerprinting requires a fast (Rust-backed) tokenizer, got {type(tokenizer).__name__}. "
            "Slow tokenizers need a dedicated validator, which does not exist yet."
        )

    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    # `extra_special_tokens`/`extra_special_tokens_ids` is the transformers name for what the design calls
    # "additional_special_tokens"; keep the payload key name stable across transformers versions.
    payload["special_tokens"] = {
        "bos": (tokenizer.bos_token, tokenizer.bos_token_id),
        "eos": (tokenizer.eos_token, tokenizer.eos_token_id),
        "pad": (tokenizer.pad_token, tokenizer.pad_token_id),
        "unk": (tokenizer.unk_token, tokenizer.unk_token_id),
        "sep": (tokenizer.sep_token, tokenizer.sep_token_id),
        "cls": (tokenizer.cls_token, tokenizer.cls_token_id),
        "mask": (tokenizer.mask_token, tokenizer.mask_token_id),
        "additional_special_tokens": list(
            zip(tokenizer.extra_special_tokens, tokenizer.extra_special_tokens_ids, strict=True)
        ),
    }
    payload["model_input_names"] = tokenizer.model_input_names
    return payload


def tokenizer_fingerprint(tokenizer) -> str:
    """
    Fingerprint a tokenizer's complete rendering behavior.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to fingerprint. Must be a fast (Rust-backed) tokenizer.

    Returns:
        `str`: sha256 hex digest of the canonical JSON serialization (sorted keys, no whitespace) of
        [`canonical_tokenizer_payload`]. Cosmetic JSON byte formatting (whitespace, key order) never changes the
        digest; any change to vocabulary, merges, added tokens, normalization, pre-tokenization, post-processing,
        decoding, or special-token roles/IDs does.
    """
    payload = canonical_tokenizer_payload(tokenizer)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Per-teacher keys copied verbatim from a `TeacherRegistry.manifest()` entry into the saved manifest. Listed
# explicitly (rather than persisting the whole entry) so loading kwargs, credentials, or any other extra key a
# registry entry may carry never reaches disk.
_TEACHER_ENTRY_KEYS = (
    "id",
    "index",
    "source",
    "resolved_revision",
    "source_key",
    "tokenizer_fingerprint",
    "adapter_version",
)
_TEACHER_DTYPE_KEYS = ("source_dtype", "hidden_dtype", "projection_dtype")
_HEAD_IDENTITY_KEYS = ("weight_shape", "has_bias", "logit_scale", "final_logit_softcapping", "transform_version")


@dataclass
class TeacherManifest:
    """
    Teacher identities and numerical conventions saved alongside a student checkpoint.

    Excludes transient state (targets, leases, caches, loaded model objects) and any loading credentials. Compared
    on resume via [`~TeacherManifest.check_compatible`], never diffed field-by-field by the caller.

    Args:
        version (`int`):
            Manifest schema version.
        teachers (`list[dict]`):
            One allowlisted entry per teacher, in registry order. Each entry has keys `id`, `index`, `source`,
            `resolved_revision`, `source_key`, `tokenizer_fingerprint`, `source_dtype`, `hidden_dtype`,
            `projection_dtype`, `adapter_version`, and `head` (itself a dict with `weight_shape`, `has_bias`,
            `source_dtype`, `logit_scale`, `final_logit_softcapping`, `transform_version`). Dtypes are strings (e.g.
            `"torch.bfloat16"`).
        student_tokenizer_fingerprint (`str`):
            [`tokenizer_fingerprint`] of the student's tokenizer at save time.
        beta (`float`):
            Divergence interpolation coefficient used for training.
        temperature (`float`):
            Loss temperature used for training.
        chunk_size (`int`):
            Chunk size used by the managed loss.
    """

    version: int
    teachers: list[dict[str, Any]]
    student_tokenizer_fingerprint: str
    beta: float
    temperature: float
    chunk_size: int

    @classmethod
    def from_registry(
        cls,
        registry_manifest: dict[str, Any],
        *,
        student_tokenizer_fingerprint: str,
        beta: float,
        temperature: float,
        chunk_size: int,
    ) -> "TeacherManifest":
        """
        Build a manifest from a [`TeacherRegistry.manifest`] dict.

        Args:
            registry_manifest (`dict`):
                `{"teachers": [...]}` as returned by `TeacherRegistry.manifest()`. Each entry has the keys listed on
                [`TeacherManifest`], with `source_dtype`/`hidden_dtype`/`projection_dtype`/`head.source_dtype` as
                `torch.dtype` objects and `head.weight_shape` as a `tuple[int, int]`; only these allowlisted keys are
                read, so extra keys (e.g. loading credentials) on the registry entry are never copied.
            student_tokenizer_fingerprint (`str`):
                [`tokenizer_fingerprint`] of the student's tokenizer.
            beta (`float`):
                Divergence interpolation coefficient used for training.
            temperature (`float`):
                Loss temperature used for training.
            chunk_size (`int`):
                Chunk size used by the managed loss.

        Returns:
            [`TeacherManifest`]: Manifest with `version=1` and one JSON-serializable entry per teacher.
        """
        teachers = []
        for entry in registry_manifest["teachers"]:
            teacher = {key: entry[key] for key in _TEACHER_ENTRY_KEYS}
            for key in _TEACHER_DTYPE_KEYS:
                teacher[key] = str(entry[key])
            head = entry["head"]
            teacher["head"] = {key: head[key] for key in _HEAD_IDENTITY_KEYS}
            teacher["head"]["weight_shape"] = list(head["weight_shape"])
            teacher["head"]["source_dtype"] = str(head["source_dtype"])
            teachers.append(teacher)
        return cls(
            version=1,
            teachers=teachers,
            student_tokenizer_fingerprint=student_tokenizer_fingerprint,
            beta=beta,
            temperature=temperature,
            chunk_size=chunk_size,
        )

    def save(self, output_dir: str) -> None:
        """
        Write `teacher_manifest.json` under `output_dir`.

        Args:
            output_dir (`str`):
                Directory to write into (e.g. a checkpoint directory). Must already exist.
        """
        path = os.path.join(output_dir, "teacher_manifest.json")
        payload = {
            "version": self.version,
            "teachers": self.teachers,
            "student_tokenizer_fingerprint": self.student_tokenizer_fingerprint,
            "beta": self.beta,
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, output_dir: str) -> "TeacherManifest":
        """
        Read `teacher_manifest.json` from `output_dir`.

        Args:
            output_dir (`str`):
                Directory previously written by [`~TeacherManifest.save`].

        Returns:
            [`TeacherManifest`]: The saved manifest.
        """
        path = os.path.join(output_dir, "teacher_manifest.json")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            version=payload["version"],
            teachers=payload["teachers"],
            student_tokenizer_fingerprint=payload["student_tokenizer_fingerprint"],
            beta=payload["beta"],
            temperature=payload["temperature"],
            chunk_size=payload["chunk_size"],
        )

    def check_compatible(self, other: "TeacherManifest") -> None:
        """
        Compare this manifest (freshly built from the current run) against a manifest loaded from a checkpoint.

        Args:
            other (`TeacherManifest`):
                Manifest to compare against, typically loaded via [`~TeacherManifest.load`].

        Raises:
            `ValueError`: Listing every differing field, if `self` and `other` disagree on teacher IDs/order, source
                keys, resolved revisions, tokenizer fingerprints, head identity fields, dtypes, beta, temperature, or
                chunk size.
        """
        differences = []

        self_ids = [teacher["id"] for teacher in self.teachers]
        other_ids = [teacher["id"] for teacher in other.teachers]
        if self_ids != other_ids:
            differences.append(f"teacher ids/order: {self_ids!r} != {other_ids!r}")
        else:
            per_teacher_keys = _TEACHER_ENTRY_KEYS[2:] + _TEACHER_DTYPE_KEYS  # skip id/index, already compared above
            for self_teacher, other_teacher in zip(self.teachers, other.teachers, strict=True):
                for key in per_teacher_keys:
                    if self_teacher[key] != other_teacher[key]:
                        differences.append(
                            f"teacher {self_teacher['id']!r} {key}: {self_teacher[key]!r} != {other_teacher[key]!r}"
                        )
                for key in _HEAD_IDENTITY_KEYS:
                    if self_teacher["head"][key] != other_teacher["head"][key]:
                        differences.append(
                            f"teacher {self_teacher['id']!r} head.{key}: "
                            f"{self_teacher['head'][key]!r} != {other_teacher['head'][key]!r}"
                        )

        for attribute in ("student_tokenizer_fingerprint", "beta", "temperature", "chunk_size"):
            self_value = getattr(self, attribute)
            other_value = getattr(other, attribute)
            if self_value != other_value:
                differences.append(f"{attribute}: {self_value!r} != {other_value!r}")

        if differences:
            raise ValueError("Teacher manifest incompatible with the checkpoint:\n" + "\n".join(differences))


def teacher_metric_key(prefix: str, teacher_id: str) -> str:
    """
    Build a collision-free metric key for one teacher.

    Args:
        prefix (`str`):
            Metric family, e.g. `"teacher_jsd"`.
        teacher_id (`str`):
            Routing ID of the teacher.

    Returns:
        `str`: `f"{prefix}/{encoded_teacher_id}"`, where `/` and whitespace in `teacher_id` are percent-encoded so
        two distinct IDs (e.g. `"a/b"` and `"a"` scored under `"b"`) never collide on the same key.
    """
    return f"{prefix}/{quote(teacher_id, safe='')}"


def teacher_metrics_from_stats(stats, teacher_ids: list[str]) -> dict[str, float]:
    """
    Reduce accumulated per-teacher statistics into loggable metrics.

    Args:
        stats (`torch.Tensor`):
            Float32 `[3, num_teachers]` tensor, summed over ranks/steps since the last log. Rows are
            `(divergence_sum, teacher_entropy_sum, valid_token_count)`, matching `managed_chunked_divergence_loss`'s
            `teacher_stats` output; zero columns mark teachers absent from the accumulated window.
        teacher_ids (`list[str]`):
            Routing ID for each column of `stats`, in registry order.

    Returns:
        `dict[str, float]`: Empty if every column is zero (no teacher was scored). Otherwise, for every teacher,
        `teacher_token_frac/{id}` (that teacher's valid tokens over all valid tokens, `0.0` for an absent teacher),
        plus `teacher_jsd/{id}` and `teacher_entropy/{id}` (the per-token mean divergence/entropy), omitted for
        teachers with zero valid tokens since their mean is undefined.
    """
    divergence_sum, entropy_sum, count = stats[0], stats[1], stats[2]
    total_count = count.sum().item()
    if total_count == 0:
        return {}

    metrics = {}
    for index, teacher_id in enumerate(teacher_ids):
        teacher_count = count[index].item()
        metrics[teacher_metric_key("teacher_token_frac", teacher_id)] = teacher_count / total_count
        if teacher_count > 0:
            metrics[teacher_metric_key("teacher_jsd", teacher_id)] = divergence_sum[index].item() / teacher_count
            metrics[teacher_metric_key("teacher_entropy", teacher_id)] = entropy_sum[index].item() / teacher_count
    return metrics
