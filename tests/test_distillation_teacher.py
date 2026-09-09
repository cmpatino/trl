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

"""Tests for the managed multi-teacher registry, scoring executor and window store (`_distillation_teacher.py`)."""

import hashlib
import json
import shutil
import weakref
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.testing_utils import torch_device

from trl.trainer import _distillation_teacher as teacher_module
from trl.trainer._distillation_heads import TeacherHeadCache
from trl.trainer._distillation_identity import TeacherManifest, tokenizer_fingerprint
from trl.trainer._distillation_teacher import (
    HiddenTargetBlock,
    ScoreRequest,
    TeacherExecutor,
    TeacherRegistry,
    WindowStore,
    _checkpoint_files,
    _checkpoint_inventory,
    _content_digest,
    _hash_tensor_blocks,
    _resolve_hub_snapshot,
)

from .testing_utils import require_torch_accelerator


VOCAB_SIZE = 64
HIDDEN_SIZE = 8


class FakeHeadCache:
    """Duck-typed stand-in for W1's `TeacherHeadCache`, which is not on this branch yet."""

    def __init__(self):
        self.sources = {}
        self.released = []
        self.evictions = 0

    def retain_head_source(self, source):
        self.sources.setdefault(source.identity, source)

    def release_head_source(self, identity):
        self.sources.pop(identity)
        self.released.append(identity)

    def evict_idle_gpu(self):
        self.evictions += 1


def build_tokenizer():
    """A tiny fast tokenizer: the fingerprint covers a complete serialization without parsing a 4 MB vocabulary."""
    vocab = {f"tok{index}": index for index in range(VOCAB_SIZE - 3)}
    vocab.update({"<pad>": VOCAB_SIZE - 3, "<eos>": VOCAB_SIZE - 2, "<unk>": VOCAB_SIZE - 1})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")


def build_model(hidden_size=HIDDEN_SIZE, tie_word_embeddings=False, seed=0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen3ForCausalLM(config).eval()


def save_source(directory, tokenizer, **kwargs):
    model = build_model(**kwargs)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def overwrite_weights(directory, tensor_name="model.embed_tokens.weight", value=0.5):
    """Replace one tensor's values in place, keeping its shape, dtype, the file size and `config.json` identical."""
    path = Path(directory) / "model.safetensors"
    tensors = load_file(str(path))
    tensors[tensor_name] = torch.full_like(tensors[tensor_name], value)
    save_file(tensors, str(path), metadata={"format": "pt"})


def manifest_of(registry, tokenizer):
    return TeacherManifest.from_registry(
        registry.manifest(),
        student_tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        beta=1.0,
        temperature=1.0,
        chunk_size=256,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


@pytest.fixture(scope="module")
def sources(tmp_path_factory, tokenizer):
    """Three immutable local checkpoints: two revisions of one model and one with a different hidden width."""
    root = tmp_path_factory.mktemp("teacher-sources")
    return {
        "a": save_source(root / "repoA", tokenizer, seed=0),
        "a_v2": save_source(root / "repoA-v2", tokenizer, seed=1),
        "wide": save_source(root / "repoWide", tokenizer, hidden_size=16, seed=2),
        "tied": save_source(root / "repoTied", tokenizer, tie_word_embeddings=True, seed=3),
    }


def make_registry(teacher_models, tokenizer, **kwargs):
    return TeacherRegistry(teacher_models, student_tokenizer=tokenizer, student_vocab_size=VOCAB_SIZE, **kwargs)


def make_executor(registry, **kwargs):
    return TeacherExecutor(registry, FakeHeadCache(), torch.device("cpu"), **kwargs)


def make_microbatch(teacher_indices, *, prompt_length=4, completion_length=3, tool_mask=None, seed=7):
    """A generation-payload microbatch: left-padded prompts, right-padded completions, optional tool mask."""
    generator = torch.Generator().manual_seed(seed)
    rows = len(teacher_indices)
    prompt_ids = torch.randint(0, VOCAB_SIZE - 3, (rows, prompt_length), generator=generator)
    prompt_mask = torch.ones(rows, prompt_length, dtype=torch.long)
    prompt_mask[0, 0] = 0  # left padding on the first row
    prompt_ids[0, 0] = VOCAB_SIZE - 3
    completion_ids = torch.randint(0, VOCAB_SIZE - 3, (rows, completion_length), generator=generator)
    completion_ids[0, -1] = VOCAB_SIZE - 2  # EOS
    completion_mask = torch.ones(rows, completion_length, dtype=torch.long)
    if rows > 1:
        completion_mask[-1, -1] = 0  # right padding on the last row
    microbatch = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "teacher_index": torch.tensor(teacher_indices),
    }
    if tool_mask is not None:
        microbatch["tool_mask"] = tool_mask
    return microbatch


def reference_forward(path, microbatch, dtype=torch.float32):
    """Plain-forward reference: `base_model` hidden states and full logits at the completion positions."""
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.eval()
    input_ids = torch.cat([microbatch["prompt_ids"], microbatch["completion_ids"]], dim=1)
    attention_mask = torch.cat([microbatch["prompt_mask"], microbatch["completion_mask"]], dim=1)
    completion_length = microbatch["completion_ids"].shape[1]
    with torch.no_grad():
        hidden = model.base_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    hidden = hidden[:, :-1][:, -completion_length:]
    logits = logits[:, :-1][:, -completion_length:]
    return hidden.reshape(-1, hidden.shape[-1]), logits.reshape(-1, logits.shape[-1])


def score_single(executor, registry, microbatch, teacher_index=0, hidden_size=HIDDEN_SIZE):
    """Score every valid position of one microbatch for one teacher and return the filled block and positions."""
    loss_mask = microbatch["completion_mask"]
    if "tool_mask" in microbatch:
        loss_mask = loss_mask * microbatch["tool_mask"]
    positions = (loss_mask.reshape(-1) > 0).nonzero().flatten()
    entry = registry.entries[teacher_index]
    block = HiddenTargetBlock(
        1, torch.zeros((positions.numel(), hidden_size), dtype=entry.target_dtype), [None] * positions.numel()
    )
    request = ScoreRequest(
        generation_id=0,
        microbatch_index=0,
        teacher_index=teacher_index,
        input_ids=torch.cat([microbatch["prompt_ids"], microbatch["completion_ids"]], dim=1),
        attention_mask=torch.cat([microbatch["prompt_mask"], microbatch["completion_mask"]], dim=1),
        prompt_length=microbatch["prompt_ids"].shape[1],
        completion_length=microbatch["completion_ids"].shape[1],
        positions=positions,
        sample_ids=tuple(f"s{row}" for row in range(microbatch["prompt_ids"].shape[0])),
    )
    result = executor.score(request, executor.target_writer(block, torch.arange(positions.numel())))
    return block, positions, result


class TestTeacherRegistry:
    def test_empty_mapping(self, tokenizer):
        with pytest.raises(ValueError, match="`teacher_models` is empty"):
            make_registry({}, tokenizer)

    def test_unknown_per_teacher_kwargs(self, sources, tokenizer):
        with pytest.raises(ValueError, match="unknown teacher IDs \\['late'\\]"):
            make_registry({"early": sources["a"]}, tokenizer, per_teacher_init_kwargs={"late": {"dtype": "float32"}})

    def test_preloaded_rejects_per_teacher_kwargs(self, tokenizer):
        model = build_model()
        with pytest.raises(ValueError, match="preloaded model, so it cannot take per-teacher loading overrides"):
            make_registry(
                {"live": model},
                tokenizer,
                teacher_tokenizers={"live": tokenizer},
                per_teacher_init_kwargs={"live": {"revision": "abc"}},
            )

    def test_unsupported_loading_options(self, sources, tokenizer):
        with pytest.raises(ValueError, match="unsupported loading options \\['device_map', 'load_in_4bit'\\]"):
            make_registry(
                {"early": sources["a"]},
                tokenizer,
                common_init_kwargs={"device_map": "auto", "load_in_4bit": True},
            )

    def test_local_path_rejects_revision(self, sources, tokenizer):
        with pytest.raises(ValueError, match="Local paths carry no revision"):
            make_registry({"early": sources["a"]}, tokenizer, per_teacher_init_kwargs={"early": {"revision": "main"}})

    def test_vocab_size_mismatch(self, sources, tokenizer):
        with pytest.raises(ValueError, match="but the student has vocab_size 8"):
            TeacherRegistry({"early": sources["a"]}, student_tokenizer=tokenizer, student_vocab_size=8)

    def test_tokenizer_mismatch(self, tmp_path, sources, tokenizer):
        mismatched = tmp_path / "repoMismatch"
        save_source(mismatched, tokenizer)
        saved = AutoTokenizer.from_pretrained(str(mismatched))
        saved.add_tokens(["<extra_0>"])
        saved.save_pretrained(str(mismatched))
        with pytest.raises(ValueError, match="tokenizer serialization and special-token roles/IDs must be identical"):
            make_registry({"early": str(mismatched)}, tokenizer)

    def test_preloaded_needs_tokenizer(self, tokenizer):
        with pytest.raises(ValueError, match="needs its tokenizer in `teacher_tokenizers`"):
            make_registry({"live": build_model()}, tokenizer)

    def test_preloaded_student_alias_rejected(self, tokenizer):
        student = build_model()
        teacher = build_model()
        # Share the student's embedding storage, the aliasing the registry must reject.
        teacher.model.embed_tokens.weight = student.model.embed_tokens.weight
        with pytest.raises(ValueError, match="shares storage with the student"):
            make_registry({"live": teacher}, tokenizer, teacher_tokenizers={"live": tokenizer}, student_model=student)

    def test_preloaded_independent_copy_accepted(self, tokenizer):
        student = build_model()
        registry = make_registry(
            {"live": build_model(seed=4)},
            tokenizer,
            teacher_tokenizers={"live": tokenizer},
            student_model=student,
        )
        entry = registry["live"]
        assert entry.evictable is False
        assert entry.source is None and entry.resolved_revision is None
        assert entry.storage_bytes > 0

    def test_routing_defaults_and_indices(self, sources, tokenizer):
        single = make_registry({"only": sources["a"]}, tokenizer)
        assert single.resolve_ids([None, "only"]) == [0, 0]
        multi = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        assert multi.teacher_ids == ["early", "late"]
        assert [entry.index for entry in multi.entries] == [0, 1]
        assert multi.resolve_ids(["late", "early", "late"]) == [1, 0, 1]
        with pytest.raises(ValueError, match="A row has no `teacher_id`"):
            multi.resolve_ids([None])
        with pytest.raises(ValueError, match="Unknown teacher ID 'missing'"):
            multi.resolve_ids(["missing"])

    def test_identity_digest_is_stable_and_content_sensitive(self, sources, tokenizer):
        first = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        second = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        assert first.identity_digest() == second.identity_digest()
        assert first.identity_digest() == first.identity_digest()
        swapped = make_registry({"late": sources["a_v2"], "early": sources["a"]}, tokenizer)
        assert swapped.identity_digest() != first.identity_digest()  # indices are part of the identity
        widened = make_registry({"early": sources["a"], "late": sources["wide"]}, tokenizer)
        assert widened.identity_digest() != first.identity_digest()

    def test_manifest_is_allowlisted(self, sources, tokenizer):
        registry = make_registry(
            {"early": sources["a"]}, tokenizer, common_init_kwargs={"dtype": "float32", "token": "hf_secret"}
        )
        manifest = registry.manifest()
        assert manifest["version"] == 1
        assert manifest["identity_digest"] == registry.identity_digest()
        assert manifest["student_tokenizer_fingerprint"] == tokenizer_fingerprint(tokenizer)
        teacher = manifest["teachers"][0]
        expected = {
            "id",
            "index",
            "source",
            "resolved_revision",
            "source_key",
            "tokenizer_fingerprint",
            "config_class",
            "hidden_size",
            "vocab_size",
            "source_dtype",
            "hidden_dtype",
            "hidden_dtype_observed",
            "target_dtype",
            "projection_dtype",
            "adapter_version",
            "content_digest",
            "content_hashed_bytes",
            "head",
            "evictable",
            "storage_bytes",
            "loading",
        }
        assert set(teacher) == expected
        # `TeacherManifest.from_registry` serializes these itself, so they stay native here.
        assert teacher["source_dtype"] == torch.float32
        # No forward has run, so the reported hidden dtype is still the planned target dtype, flagged as such.
        assert teacher["hidden_dtype_observed"] is False
        assert teacher["hidden_dtype"] == teacher["target_dtype"] == torch.float32
        assert teacher["content_hashed_bytes"] > 0
        assert teacher["head"] == {
            "weight_shape": (VOCAB_SIZE, HIDDEN_SIZE),
            "has_bias": False,
            "source_dtype": torch.float32,
            "logit_scale": 1.0,
            "final_logit_softcapping": None,
            "transform_version": 1,
        }
        assert teacher["config_class"] == "Qwen3Config"
        assert "hf_secret" not in str(manifest)
        assert set(teacher["loading"]) <= {
            "dtype",
            "revision",
            "attn_implementation",
            "low_cpu_mem_usage",
            "trust_remote_code",
        }

    def test_manifest_feeds_the_identity_manifest(self, tmp_path, sources, tokenizer):
        registry = make_registry(
            {"early": sources["a"], "late": sources["a_v2"]},
            tokenizer,
            common_init_kwargs={"token": "hf_secret"},
        )
        fingerprint = tokenizer_fingerprint(tokenizer)
        saved = TeacherManifest.from_registry(
            registry.manifest(),
            student_tokenizer_fingerprint=fingerprint,
            beta=1.0,
            temperature=1.0,
            chunk_size=256,
        )
        assert [entry["id"] for entry in saved.teachers] == ["early", "late"]
        assert [entry["index"] for entry in saved.teachers] == [0, 1]
        assert saved.teachers[0]["source_dtype"] == "torch.float32"
        assert saved.teachers[0]["head"]["weight_shape"] == [VOCAB_SIZE, HIDDEN_SIZE]
        assert saved.teachers[0]["source_key"] != saved.teachers[1]["source_key"]
        saved.save(str(tmp_path))
        assert "hf_secret" not in (tmp_path / "teacher_manifest.json").read_text()
        reloaded = TeacherManifest.load(str(tmp_path))
        assert reloaded == saved
        saved.check_compatible(reloaded)

    def test_head_and_precision_records(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"], "wide": sources["wide"]}, tokenizer)
        early, wide = registry.entries
        assert early.hidden_size == HIDDEN_SIZE and wide.hidden_size == 16
        assert early.head_identity.weight_shape == (VOCAB_SIZE, HIDDEN_SIZE)
        assert early.head_identity.has_bias is False
        assert early.head_identity.logit_scale == 1.0
        assert early.head_identity.final_logit_softcapping is None
        assert early.source_dtype == torch.float32  # `create_model_from_path` defaults to float32
        assert early.head_bytes == VOCAB_SIZE * HIDDEN_SIZE * 4
        assert early.loading_transient_bytes > 0


class TestTwoRevisions:
    def test_local_paths_never_share_a_source(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        early, late = registry.entries
        assert early.source_key != late.source_key
        assert early.head_identity != late.head_identity
        manifest_ids = [entry["id"] for entry in registry.manifest()["teachers"]]
        assert manifest_ids == ["early", "late"]
        executor = make_executor(registry)
        microbatch = make_microbatch([0])
        early_block, positions, _ = score_single(executor, registry, microbatch, teacher_index=0)
        late_block, _, _ = score_single(executor, registry, microbatch, teacher_index=1)
        assert not torch.allclose(early_block.hidden, late_block.hidden)
        early_reference, _ = reference_forward(sources["a"], microbatch)
        late_reference, _ = reference_forward(sources["a_v2"], microbatch)
        assert torch.equal(early_block.hidden, early_reference[positions])
        assert torch.equal(late_block.hidden, late_reference[positions])

    def test_hub_revisions_of_one_repository(self, monkeypatch, sources, tokenizer):
        snapshots = {"v1": (sources["a"], "a" * 40), "v2": (sources["a_v2"], "b" * 40)}
        calls = []

        def fake_resolver(repo_id, revision, token=None):
            calls.append((repo_id, revision))
            return snapshots[revision]

        monkeypatch.setattr(teacher_module, "_resolve_hub_snapshot", fake_resolver)
        registry = make_registry(
            {"early": "org/teacher", "late": "org/teacher"},
            tokenizer,
            per_teacher_init_kwargs={"early": {"revision": "v1"}, "late": {"revision": "v2"}},
        )
        assert calls == [("org/teacher", "v1"), ("org/teacher", "v2")]
        early, late = registry.entries
        assert early.source == late.source == "org/teacher"
        assert (early.resolved_revision, late.resolved_revision) == ("a" * 40, "b" * 40)
        assert early.source_key != late.source_key
        assert early.load_path != late.load_path
        manifest = {entry["id"]: entry for entry in registry.manifest()["teachers"]}
        assert manifest["early"]["resolved_revision"] == "a" * 40
        assert manifest["late"]["resolved_revision"] == "b" * 40
        assert manifest["early"]["source_key"] != manifest["late"]["source_key"]
        executor = make_executor(registry)
        microbatch = make_microbatch([0])
        first, positions, _ = score_single(executor, registry, microbatch, teacher_index=0)
        second, _, _ = score_single(executor, registry, microbatch, teacher_index=1)
        assert not torch.allclose(first.hidden, second.hidden)
        assert executor.stats.body_loads == 2

    def test_same_repository_and_revision_shares_the_body_slot(self, monkeypatch, sources, tokenizer):
        monkeypatch.setattr(
            teacher_module, "_resolve_hub_snapshot", lambda repo_id, revision, token=None: (sources["a"], "c" * 40)
        )
        registry = make_registry({"early": "org/teacher", "twin": "org/teacher"}, tokenizer)
        early, twin = registry.entries
        assert early.source_key == twin.source_key  # identical immutable source
        assert early.teacher_id != twin.teacher_id  # separate routing IDs and metric columns
        executor = make_executor(registry)
        microbatch = make_microbatch([0])
        score_single(executor, registry, microbatch, teacher_index=0)
        score_single(executor, registry, microbatch, teacher_index=1)
        assert executor.stats.body_loads == 1
        assert executor.stats.cpu_reloads == 0

    def test_real_hub_revision_resolution(self):
        local_dir, commit = _resolve_hub_snapshot("trl-internal-testing/tiny-Qwen3ForCausalLM", "main")
        assert len(commit) == 40 and commit.isalnum()
        assert (Path(local_dir) / "config.json").is_file()
        pinned_dir, pinned_commit = _resolve_hub_snapshot("trl-internal-testing/tiny-Qwen3ForCausalLM", commit)
        assert (pinned_dir, pinned_commit) == (local_dir, commit)


class TestTeacherExecutor:
    def test_capabilities_and_precision_records(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        entry = registry["early"]
        assert executor.capabilities == {"hidden_targets": True, "requires_collective_schedule": False}
        assert (entry.source_dtype, entry.target_dtype, entry.projection_dtype) == (
            torch.float32,
            torch.float32,
            torch.float32,
        )
        # The backbone output dtype is measured, never taken from the configuration.
        assert entry.hidden_dtype is None
        assert executor.probe_hidden_dtype(0) == torch.float32
        assert entry.hidden_dtype == torch.float32

    def test_hidden_dtype_is_observed_at_first_scoring(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        entry = registry["early"]
        assert entry.hidden_dtype is None
        _, _, result = score_single(executor, registry, make_microbatch([0]))
        assert entry.hidden_dtype == torch.float32
        assert result.hidden_dtype == torch.float32
        manifest = registry.manifest()["teachers"][0]
        assert manifest["hidden_dtype_observed"] is True
        assert manifest["hidden_dtype"] == torch.float32

    def test_evict_idle_gpu_delegates_to_the_head_cache(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        score_single(executor, registry, make_microbatch([0]))
        executor.evict_idle_gpu()
        assert executor.head_cache.evictions == 1
        assert executor.stats.live_device_weight_bytes == 0

    def test_hidden_and_logit_parity_with_a_plain_forward(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        # Score the whole microbatch in one forward so the comparison is bitwise: splitting the rows changes the
        # matmul reduction shapes and therefore the last bits.
        executor = make_executor(registry, scoring_batch_size=3)
        microbatch = make_microbatch([0, 0, 0])
        block, positions, result = score_single(executor, registry, microbatch)
        reference_hidden, reference_logits = reference_forward(sources["a"], microbatch)
        assert result.rows_written == positions.numel()
        assert result.hidden_dtype == torch.float32
        assert torch.equal(block.hidden, reference_hidden[positions])
        head = executor.retain_head_source(0)
        # The retained head reproduces the model's own logits, but not bitwise: this is `matmul` against the model's
        # `lm_head` (an `F.linear`/`addmm`), and which kernel and reduction blocking each picks depends on the host
        # BLAS and thread count. Observed on the GPU job's container, where the two paths differ in the last bits.
        torch.testing.assert_close(block.hidden @ head.weight.t(), reference_logits[positions], rtol=1e-5, atol=1e-6)

    def test_tool_masked_position_is_excluded_but_stays_in_context(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        executor.scoring_batch_size = 2
        tool_mask = torch.ones(2, 3, dtype=torch.long)
        tool_mask[0, 1] = 0
        microbatch = make_microbatch([0, 0], tool_mask=tool_mask)
        block, positions, _ = score_single(executor, registry, microbatch)
        assert 1 not in positions.tolist()  # the tool position gets no target
        assert positions.tolist() == [0, 2, 3, 4]  # row 1's last position is right padding
        reference_hidden, _ = reference_forward(sources["a"], microbatch)
        # The reference forward sees the full sequence, so the masked token is still in context.
        assert torch.equal(block.hidden, reference_hidden[positions])

    def test_scoring_batch_size_does_not_change_targets(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        microbatch = make_microbatch([0, 0, 0])
        one = make_executor(registry, scoring_batch_size=1)
        block_one, _, result_one = score_single(one, registry, microbatch)
        three = make_executor(registry, scoring_batch_size=3)
        block_three, _, result_three = score_single(three, registry, microbatch)
        assert (result_one.forward_calls, result_three.forward_calls) == (3, 1)
        # Subbatching changes the matmul shapes, so compare numerically rather than bitwise.
        torch.testing.assert_close(block_one.hidden, block_three.hidden, rtol=1e-5, atol=1e-6)

    def test_autocast_context_is_recorded_and_applied(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry, autocast_dtype=torch.bfloat16)
        entry = registry["early"]
        assert entry.target_dtype == torch.bfloat16 and entry.projection_dtype == torch.bfloat16
        assert entry.source_dtype == torch.float32  # the CPU source keeps its own dtype
        microbatch = make_microbatch([0])
        block, positions, result = score_single(executor, registry, microbatch)
        # Targets are stored in the target dtype; the observed backbone output dtype is recorded separately.
        assert block.hidden.dtype == torch.bfloat16
        assert result.hidden_dtype == entry.hidden_dtype
        assert entry.hidden_dtype is not None
        model = AutoModelForCausalLM.from_pretrained(sources["a"], dtype=torch.float32)
        model.eval()
        input_ids = torch.cat([microbatch["prompt_ids"], microbatch["completion_ids"]], dim=1)
        attention_mask = torch.cat([microbatch["prompt_mask"], microbatch["completion_mask"]], dim=1)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            hidden = model.base_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        expected = hidden[:, :-1][:, -3:].reshape(-1, HIDDEN_SIZE).to(torch.bfloat16)
        assert torch.equal(block.hidden, expected[positions])

    def test_source_is_restored_after_a_forward_exception(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        body = executor._load_body(0)
        before = {name: tensor.clone() for name, tensor in body.state_dict().items()}
        metadata = {
            name: (tensor.dtype, tensor.device, tensor.requires_grad)
            for name, tensor in list(body.named_parameters()) + list(body.named_buffers())
        }

        class Boom(Exception):
            pass

        def raise_mid_forward(module, args):
            raise Boom("forward interrupted")

        handle = body.base_model.layers[1].register_forward_pre_hook(raise_mid_forward)
        microbatch = make_microbatch([0])
        with pytest.raises(Boom):
            score_single(executor, registry, microbatch)
        handle.remove()
        after = body.state_dict()
        assert set(after) == set(before)
        for name, value in before.items():
            assert torch.equal(value, after[name]), name
        for name, tensor in list(body.named_parameters()) + list(body.named_buffers()):
            assert (tensor.dtype, tensor.device, tensor.requires_grad) == metadata[name], name
        assert executor.stats.live_device_weight_bytes == 0  # the device parameter dict was released
        block, positions, _ = score_single(executor, registry, microbatch)
        reference_hidden, _ = reference_forward(sources["a"], microbatch)
        assert torch.equal(block.hidden, reference_hidden[positions])

    def test_source_values_unchanged_by_scoring(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        body = executor._load_body(0)
        before = {name: tensor.clone() for name, tensor in body.state_dict().items()}
        devices = {name: tensor.device for name, tensor in body.named_buffers()}
        score_single(executor, registry, make_microbatch([0]))
        for name, value in before.items():
            assert torch.equal(value, body.state_dict()[name]), name
        assert {name: tensor.device for name, tensor in body.named_buffers()} == devices
        assert executor.stats.live_device_weight_bytes == 0

    def test_body_eviction_releases_aliases_and_keeps_the_head(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        executor = make_executor(registry)
        head = executor.retain_head_source(0)
        body = executor._load_body(0)
        module_ref = weakref.ref(body)
        backbone_ref = weakref.ref(body.model.layers[0].mlp.gate_proj.weight)
        storage_ref = weakref.ref(body.model.layers[0].mlp.gate_proj.weight.untyped_storage())
        embedding_ref = weakref.ref(body.model.embed_tokens.weight)
        del body
        executor._load_body(1)  # evicts the first body from the single CPU slot
        assert module_ref() is None
        assert backbone_ref() is None
        assert storage_ref() is None  # the parameter storage itself, not just the tensor wrapper
        assert embedding_ref() is None
        # The head source survives its body and still projects the same logits.
        microbatch = make_microbatch([0])
        reference_hidden, reference_logits = reference_forward(sources["a"], microbatch)
        positions = torch.arange(reference_hidden.shape[0])
        # Same reason as in `test_hidden_and_logit_parity_with_a_plain_forward`: `matmul` here against the
        # model's own `lm_head` there, so equality holds to a tight tolerance rather than bitwise.
        torch.testing.assert_close(
            reference_hidden @ head.weight.t(), reference_logits[positions], rtol=1e-5, atol=1e-6
        )
        assert executor.stats.body_loads == 2 and executor.stats.cpu_reloads == 0
        executor._load_body(0)
        assert executor.stats.cpu_reloads == 1
        assert executor.stats.disk_bytes == 3 * registry["early"].storage_bytes

    def test_tied_head_is_retained_as_a_compact_copy(self, sources, tokenizer):
        registry = make_registry({"tied": sources["tied"], "late": sources["a_v2"]}, tokenizer)
        executor = make_executor(registry)
        body = executor._load_body(0)
        embedding = body.model.embed_tokens.weight
        source = executor.retain_head_source(0)
        assert source.weight.data_ptr() != embedding.data_ptr()  # compact copy, not the tied embedding
        assert source.weight.is_contiguous()
        assert torch.equal(source.weight, embedding)
        embedding_ref = weakref.ref(embedding)
        del body, embedding
        executor._load_body(1)
        assert embedding_ref() is None  # the tied embedding storage is gone
        assert source.weight.abs().sum() > 0  # the head copy is intact

    def test_head_source_refcounting(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        identity = registry["early"].head_identity
        first = executor.retain_head_source(0)
        second = executor.retain_head_source(0)
        assert first is second
        assert identity in executor.head_cache.sources
        executor.release_head_source(0)
        assert identity in executor.head_cache.sources  # one consumer left
        executor.release_head_source(0)
        assert identity not in executor.head_cache.sources
        assert executor.head_cache.released == [identity]
        with pytest.raises(RuntimeError, match="is not retained"):
            executor.release_head_source(0)

    def test_preloaded_model_is_non_evictable_and_budget_charged(self, sources, tokenizer):
        model = build_model(seed=9)
        registry = make_registry(
            {"live": model, "disk": sources["a"]}, tokenizer, teacher_tokenizers={"live": tokenizer}
        )
        executor = make_executor(registry)
        live_entry = registry["live"]
        assert live_entry.storage_bytes > 0
        assert executor.stats.live_cpu_weight_bytes == live_entry.storage_bytes
        assert executor._load_body(0) is model
        executor._load_body(1)
        assert executor._load_body(0) is model  # borrowed, never evicted
        assert executor.stats.body_loads == 1  # only the disk source was materialized
        assert executor.stats.live_cpu_weight_bytes == live_entry.storage_bytes + registry["disk"].storage_bytes
        microbatch = make_microbatch([0])
        block, positions, _ = score_single(executor, registry, microbatch, teacher_index=0)
        input_ids = torch.cat([microbatch["prompt_ids"], microbatch["completion_ids"]], dim=1)
        attention_mask = torch.cat([microbatch["prompt_mask"], microbatch["completion_mask"]], dim=1)
        with torch.no_grad():
            hidden = model.base_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        expected = hidden[:, :-1][:, -3:].reshape(-1, HIDDEN_SIZE)
        assert torch.equal(block.hidden, expected[positions])

    def test_budget_errors_name_their_control(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        tight_cpu = make_executor(registry, cpu_weight_budget_bytes=1024)
        with pytest.raises(ValueError, match="`teacher_cpu_weight_budget_bytes` is 1024"):
            tight_cpu._load_body(0)
        tight_gpu = make_executor(registry, gpu_weight_budget_bytes=1024)
        with pytest.raises(ValueError, match="`teacher_gpu_weight_budget_bytes` is 1024"):
            score_single(tight_gpu, registry, make_microbatch([0]))
        assert tight_gpu.stats.live_device_weight_bytes == 0

    def test_close_rejects_live_consumers_and_reopen_reloads(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = make_executor(registry)
        executor.retain_head_source(0)
        with pytest.raises(RuntimeError, match="head sources are retained"):
            executor.close()
        executor.release_head_source(0)
        executor.close()
        executor.close()  # idempotent
        assert executor.stats.live_cpu_weight_bytes == 0
        with pytest.raises(RuntimeError, match="closed"):
            score_single(executor, registry, make_microbatch([0]))
        executor.reopen()
        block, positions, _ = score_single(executor, registry, make_microbatch([0]))
        reference_hidden, _ = reference_forward(sources["a"], make_microbatch([0]))
        assert torch.equal(block.hidden, reference_hidden[positions])

    def test_loading_environment_is_scoped_and_restored(self, sources, tokenizer, monkeypatch):
        monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
        monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
        seen = {}

        original = teacher_module.create_model_from_path

        def spy(path, **kwargs):
            seen["env"] = {
                "ACCELERATE_USE_FSDP": teacher_module.os.environ["ACCELERATE_USE_FSDP"],
                "FSDP_CPU_RAM_EFFICIENT_LOADING": teacher_module.os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"],
            }
            seen["kwargs"] = kwargs
            return original(path, **kwargs)

        monkeypatch.setattr(teacher_module, "create_model_from_path", spy)
        registry = make_registry({"early": sources["a"]}, tokenizer)
        make_executor(registry)._load_body(0)
        assert seen["env"] == {"ACCELERATE_USE_FSDP": "false", "FSDP_CPU_RAM_EFFICIENT_LOADING": "false"}
        assert seen["kwargs"]["device_map"] is None
        assert seen["kwargs"]["dtype"] == torch.float32
        assert "revision" not in seen["kwargs"]
        # Restored for the student's own loading path.
        assert teacher_module.os.environ["ACCELERATE_USE_FSDP"] == "true"
        assert teacher_module.os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"


def make_window(sources, tokenizer, count=3, **executor_kwargs):
    """A two-teacher registry, executor, store and `count` two-row microbatches routed one row per teacher."""
    registry = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
    executor = make_executor(registry, scoring_batch_size=1, **executor_kwargs)
    store = WindowStore(registry)
    microbatches = [make_microbatch([0, 1], seed=10 + index) for index in range(count)]
    return registry, executor, store, microbatches


# Per microbatch: row 0 contributes 3 valid positions, row 1 contributes 2 (its last position is right padding).
ROWS_PER_MICROBATCH = 5
BLOCK_BYTES = HIDDEN_SIZE * 4


class TestWindowStore:
    def test_full_window_fits(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        assert plan.microbatch_indices == [0, 1, 2]
        assert plan.teacher_indices == [0, 1]
        assert plan.target_bytes == 3 * ROWS_PER_MICROBATCH * BLOCK_BYTES + 4096
        assert plan.weight_bytes == 2 * registry["early"].head_bytes + max(
            entry.storage_bytes + entry.loading_transient_bytes for entry in registry.entries
        )
        store.score_window(plan, executor, microbatches)
        assert executor.stats.body_loads == 2  # one load per teacher present, not per microbatch
        for index, microbatch in enumerate(microbatches):
            groups = store.targets_for((0, index), microbatch)
            assert [group.teacher_index for group in groups] == [0, 1]
            assert [group.identity.teacher_id for group in groups] == ["early", "late"]
            assert [group.positions.tolist() for group in groups] == [[0, 1, 2], [3, 4]]
            early_reference, _ = reference_forward(sources["a"], microbatch)
            late_reference, _ = reference_forward(sources["a_v2"], microbatch)
            torch.testing.assert_close(groups[0].hidden, early_reference[groups[0].positions], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(groups[1].hidden, late_reference[groups[1].positions], rtol=1e-5, atol=1e-6)

    def test_groups_are_zero_copy_views_with_row_bookkeeping(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=1)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        store.score_window(plan, executor, microbatches)
        block = next(iter(store._blocks.values()))
        groups = store.targets_for((0, 0))
        for group in groups:
            assert group.hidden.untyped_storage().data_ptr() == block.hidden.untyped_storage().data_ptr()
        assert block.row_samples == [
            ("0:0:0", 0),
            ("0:0:0", 1),
            ("0:0:0", 2),
            ("0:0:1", 0),
            ("0:0:1", 1),
        ]

    def test_heterogeneous_widths_use_separate_blocks(self, sources, tokenizer):
        registry = make_registry({"narrow": sources["a"], "wide": sources["wide"]}, tokenizer)
        executor = make_executor(registry)
        store = WindowStore(registry)
        microbatches = [make_microbatch([0, 1], seed=21)]
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        assert sorted(plan.block_rows) == [(HIDDEN_SIZE, torch.float32), (16, torch.float32)]
        assert plan.target_bytes == 3 * HIDDEN_SIZE * 4 + 2 * 16 * 4 + 2 * 4096
        store.score_window(plan, executor, microbatches)
        narrow, wide = store.targets_for((0, 0))
        assert narrow.hidden.shape == (3, HIDDEN_SIZE)
        assert wide.hidden.shape == (2, 16)

    def test_small_budget_shrinks_the_window(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer)
        two_microbatches = 2 * ROWS_PER_MICROBATCH * BLOCK_BYTES + 4096
        plan = store.plan_window(microbatches, target_cache_bytes=two_microbatches)
        assert plan.microbatch_indices == [0, 1]
        assert plan.target_bytes == two_microbatches
        store.score_window(plan, executor, microbatches)
        store.targets_for((0, 1))
        with pytest.raises(RuntimeError, match="no scored targets for \\(0, 2\\)"):
            store.targets_for((0, 2))

    def test_next_window_starts_where_the_previous_one_ended(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer)
        first = store.plan_window(microbatches, target_cache_bytes=ROWS_PER_MICROBATCH * BLOCK_BYTES + 4096)
        assert first.microbatch_indices == [0]
        store.score_window(first, executor, microbatches)
        store.release((0, 0))
        second = store.plan_window(microbatches, target_cache_bytes=1 << 20, start_index=1)
        assert second.microbatch_indices == [1, 2]
        store.score_window(second, executor, microbatches)
        assert [group.teacher_index for group in store.targets_for((0, 2), microbatches[2])] == [0, 1]
        with pytest.raises(ValueError, match="Microbatch 2 of generation batch 0 needs"):
            store.plan_window(microbatches, target_cache_bytes=10, start_index=2)

    def test_over_budget_microbatch_names_the_control(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer)
        with pytest.raises(ValueError, match="needs 4256 bytes but `teacher_target_cache_bytes` is 100"):
            store.plan_window(microbatches, target_cache_bytes=100)
        with pytest.raises(ValueError, match="`teacher_cpu_weight_budget_bytes` is 1000"):
            store.plan_window(microbatches, target_cache_bytes=1 << 20, cpu_weight_budget_bytes=1000)

    def test_weight_budget_shrinks_the_window(self, sources, tokenizer):
        """A budget covering one teacher's head plus the largest body admits only single-teacher microbatches."""
        registry = make_registry({"early": sources["a"], "late": sources["a_v2"]}, tokenizer)
        store = WindowStore(registry)
        microbatches = [make_microbatch([0, 0], seed=31), make_microbatch([1, 1], seed=32)]
        budget = registry["early"].head_bytes + max(
            entry.storage_bytes + entry.loading_transient_bytes for entry in registry.entries
        )
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20, cpu_weight_budget_bytes=budget)
        assert plan.microbatch_indices == [0]
        assert plan.teacher_indices == [0]

    def test_exact_once_consumption(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=1)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        store.score_window(plan, executor, microbatches)
        assert len(executor.head_cache.sources) == 2
        store.targets_for((0, 0))
        store.release((0, 0))
        assert executor.head_cache.sources == {}  # last consumer dropped both head retentions
        assert store._blocks == {}  # and the block
        with pytest.raises(RuntimeError, match="already released"):
            store.targets_for((0, 0))
        with pytest.raises(RuntimeError, match="no scored targets"):
            store.release((0, 0))
        executor.close()

    def test_fingerprint_mismatch_is_fatal(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=1)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        store.score_window(plan, executor, microbatches)
        stale = dict(microbatches[0])
        stale["completion_mask"] = torch.zeros_like(stale["completion_mask"])
        with pytest.raises(RuntimeError, match="does not match the tokens and masks"):
            store.targets_for((0, 0), stale)
        shorter = dict(microbatches[0])
        shorter["completion_ids"] = shorter["completion_ids"][:, :-1]
        with pytest.raises(RuntimeError, match="does not match the tokens and masks"):
            store.targets_for((0, 0), shorter)

    def test_reset_drops_every_window(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=2)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        store.score_window(plan, executor, microbatches)
        store.release((0, 0))
        store.reset()
        assert executor.head_cache.sources == {}
        assert store._blocks == {}
        with pytest.raises(RuntimeError, match="no scored targets"):
            store.targets_for((0, 1))
        executor.close()  # no retention survived the reset
        # A renewed generation batch can be planned and scored again.
        executor.reopen()
        renewed = [make_microbatch([0, 1], seed=41)]
        plan = store.plan_window(renewed, target_cache_bytes=1 << 20, generation_id=1)
        store.score_window(plan, executor, renewed)
        assert [group.teacher_index for group in store.targets_for((1, 0))] == [0, 1]

    def test_overlapping_windows_are_rejected(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=1)
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        store.score_window(plan, executor, microbatches)
        with pytest.raises(RuntimeError, match="still live; release the previous window"):
            store.score_window(plan, executor, microbatches)

    def test_load_count_follows_teacher_presence(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"], "late": sources["a_v2"], "wide": sources["wide"]}, tokenizer)
        executor = make_executor(registry)
        store = WindowStore(registry)
        microbatches = [make_microbatch([0, 1], seed=51), make_microbatch([1, 0], seed=52)]
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        assert plan.teacher_indices == [0, 1]  # the third teacher is registered but absent
        store.score_window(plan, executor, microbatches)
        assert executor.stats.body_loads == 2
        assert executor.stats.cpu_reloads == 0
        assert len(executor.head_cache.sources) == 2

    def test_all_masked_microbatch_has_no_targets(self, sources, tokenizer):
        registry, executor, store, microbatches = make_window(sources, tokenizer, count=1)
        microbatches[0]["tool_mask"] = torch.zeros_like(microbatches[0]["completion_mask"])
        plan = store.plan_window(microbatches, target_cache_bytes=1 << 20)
        assert plan.groups == [] and plan.target_bytes == 0
        store.score_window(plan, executor, microbatches)
        assert store.targets_for((0, 0), microbatches[0]) == []
        store.release((0, 0))

    def test_zero_row_microbatch_is_rejected(self, sources, tokenizer):
        registry, executor, store, _ = make_window(sources, tokenizer, count=1)
        empty = {
            "prompt_ids": torch.zeros(0, 4, dtype=torch.long),
            "prompt_mask": torch.zeros(0, 4, dtype=torch.long),
            "completion_ids": torch.zeros(0, 3, dtype=torch.long),
            "completion_mask": torch.zeros(0, 3, dtype=torch.long),
            "teacher_index": torch.zeros(0, dtype=torch.long),
        }
        with pytest.raises(ValueError, match="has no rows"):
            store.plan_window([empty], target_cache_bytes=1 << 20)


class TestRealHeadCacheSeam:
    """One test wiring W1's real `TeacherHeadCache` instead of the fake, to check the retention seam end to end."""

    def test_retained_source_projects_through_a_real_lease(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        cache = TeacherHeadCache(torch.device("cpu"))
        executor = TeacherExecutor(registry, cache, torch.device("cpu"), scoring_batch_size=1)
        microbatch = make_microbatch([0])
        block, positions, _ = score_single(executor, registry, microbatch)
        identity = registry["early"].head_identity
        executor.retain_head_source(0)
        with cache.projection_lease(identity, torch.float32) as head:
            logits = block.hidden @ head.weight.t()
        _, reference_logits = reference_forward(sources["a"], microbatch)
        torch.testing.assert_close(logits, reference_logits[positions], rtol=1e-5, atol=1e-6)
        assert cache.stats.uploads == 1
        cache.evict_idle_gpu()
        executor.release_head_source(0)
        executor.close()
        cache.close()


class TestSourceContentIdentity:
    """A source's identity must follow its tensor values: local paths are mutable and model objects have no files."""

    def test_local_weight_replacement_changes_identity(self, tmp_path, tokenizer):
        mutable = str(tmp_path / "repoMutable")
        save_source(mutable, tokenizer, seed=5)
        before = _checkpoint_inventory(mutable)
        registry = make_registry({"mutable": mutable}, tokenizer)
        overwrite_weights(mutable)
        after = _checkpoint_inventory(mutable)
        # Shapes, dtypes, file sizes and the config are untouched: only the values moved.
        assert after["tensors"] == before["tensors"]
        assert [entry[:2] for entry in after["files"]] == [entry[:2] for entry in before["files"]]
        assert after["config_digest"] == before["config_digest"]
        assert after["content_digest"] != before["content_digest"]
        assert make_registry({"mutable": mutable}, tokenizer)["mutable"].source_key != registry["mutable"].source_key

    def test_mutated_checkpoint_is_rejected_before_scoring(self, tmp_path, sources, tokenizer):
        mutable = str(tmp_path / "repoMutable")
        save_source(mutable, tokenizer, seed=5)
        registry = make_registry({"mutable": mutable, "other": sources["a_v2"]}, tokenizer)
        executor = make_executor(registry)
        score_single(executor, registry, make_microbatch([0]), teacher_index=0)
        head = executor.retain_head_source(0)
        overwrite_weights(mutable)
        executor._load_body(1)  # evict the pinned body, so the next use has to reload from disk
        with pytest.raises(ValueError, match="changed since registration"):
            score_single(executor, registry, make_microbatch([0]), teacher_index=0)
        assert executor.stats.verify_bytes > 0
        # The reload never happened, so the retained head still belongs to the registered content.
        assert head.identity is registry["mutable"].head_identity
        assert executor.stats.body_loads == 2

    def test_mutated_checkpoint_fails_manifest_compatibility(self, tmp_path, sources, tokenizer):
        mutable = str(tmp_path / "repoMutable")
        save_source(mutable, tokenizer, seed=5)
        teachers = {"mutable": mutable, "other": sources["a_v2"]}
        registered = manifest_of(make_registry(teachers, tokenizer), tokenizer)
        overwrite_weights(mutable)
        mutated = manifest_of(make_registry(teachers, tokenizer), tokenizer)
        with pytest.raises(ValueError):
            registered.check_compatible(mutated)

    def test_same_content_at_two_paths_is_one_identity(self, tmp_path, tokenizer):
        first, second = str(tmp_path / "stage-a"), str(tmp_path / "stage-b")
        save_source(first, tokenizer, seed=6)
        shutil.copytree(first, second)
        staged_first = make_registry({"early": first}, tokenizer)
        staged_second = make_registry({"early": second}, tokenizer)
        # Ranks may stage the same checkpoint under different directories; the identity must not depend on where.
        assert staged_first["early"].source_key == staged_second["early"].source_key
        assert staged_first.identity_digest() == staged_second.identity_digest()
        assert staged_first["early"].identity_source is None
        # The path survives as display and reload metadata.
        assert (staged_first["early"].source, staged_first["early"].load_path) == (first, first)
        assert (staged_second["early"].source, staged_second["early"].load_path) == (second, second)
        overwrite_weights(second)
        mutated = make_registry({"early": second}, tokenizer)
        assert mutated["early"].source_key != staged_second["early"].source_key
        assert mutated.identity_digest() != staged_second.identity_digest()

    def test_hub_identity_keeps_repository_and_revision(self, monkeypatch, sources, tokenizer):
        monkeypatch.setattr(
            teacher_module, "_resolve_hub_snapshot", lambda repo_id, revision, token=None: (sources["a"], "e" * 40)
        )
        hub = make_registry({"early": "org/teacher"}, tokenizer)["early"]
        local = make_registry({"early": sources["a"]}, tokenizer)["early"]
        assert hub.identity_source == "org/teacher" and hub.resolved_revision == "e" * 40
        assert local.identity_source is None
        # Same files, but a Hub source is pinned to its repository and commit, so the identities differ.
        assert hub.content_digest == local.content_digest
        assert hub.source_key != local.source_key

    def test_shard_index_is_part_of_the_content_digest(self, tmp_path, tokenizer):
        sharded = str(tmp_path / "repoSharded")
        model = build_model(seed=7)
        model.save_pretrained(sharded, max_shard_size="4KB")
        tokenizer.save_pretrained(sharded)
        index_path = Path(sharded) / "model.safetensors.index.json"
        assert index_path.is_file()  # the checkpoint really is sharded
        assert index_path in _checkpoint_files(sharded)
        registry = make_registry({"sharded": sharded}, tokenizer)
        before, _ = _content_digest(sharded)
        # Point one tensor at another shard: the weight files are untouched, only the loading map moved.
        index = json.loads(index_path.read_text())
        names = sorted(index["weight_map"])
        shards = sorted(set(index["weight_map"].values()))
        index["weight_map"][names[0]] = shards[-1]
        index_path.write_text(json.dumps(index))
        after, _ = _content_digest(sharded)
        assert after != before
        assert make_registry({"sharded": sharded}, tokenizer)["sharded"].source_key != registry["sharded"].source_key

    def test_preloaded_identity_covers_non_persistent_buffers(self, tokenizer):
        base = build_model(seed=9)
        mutated = build_model(seed=9)
        # Qwen3's rotary `inv_freq` is a non-persistent buffer: absent from `state_dict`, but it shapes every forward.
        assert not any("inv_freq" in name for name in mutated.state_dict())
        mutated.model.rotary_emb.inv_freq.mul_(3.0)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            assert not torch.allclose(base(input_ids=input_ids).logits, mutated(input_ids=input_ids).logits)
        registries = {
            name: make_registry({"live": model}, tokenizer, teacher_tokenizers={"live": tokenizer})
            for name, model in (("base", base), ("mutated", mutated))
        }
        assert registries["base"]["live"].source_key != registries["mutated"]["live"].source_key
        base_manifest = manifest_of(registries["base"], tokenizer)
        mutated_manifest = manifest_of(registries["mutated"], tokenizer)
        with pytest.raises(ValueError):
            base_manifest.check_compatible(mutated_manifest)

    def test_non_contiguous_values_hash_identically_in_bounded_blocks(self, monkeypatch, tokenizer):
        contiguous = build_model(seed=9)
        viewed = build_model(seed=9)
        weight = viewed.model.layers[0].mlp.gate_proj.weight
        transposed = weight.data.t().contiguous().t()  # same values, non-contiguous layout
        assert not transposed.is_contiguous() and torch.equal(transposed, weight.data)
        viewed.model.layers[0].mlp.gate_proj.weight = nn.Parameter(transposed)
        registries = {
            name: make_registry({"live": model}, tokenizer, teacher_tokenizers={"live": tokenizer})
            for name, model in (("contiguous", contiguous), ("viewed", viewed))
        }
        contiguous_entry, viewed_entry = registries["contiguous"]["live"], registries["viewed"]["live"]
        assert contiguous_entry.content_digest == viewed_entry.content_digest
        assert contiguous_entry.content_hashed_bytes == viewed_entry.content_hashed_bytes
        # A tiny block size must not change the digest: slices are hashed in logical order, one block at a time.
        monkeypatch.setattr(teacher_module, "_HASH_BLOCK_BYTES", 8)
        digests = []
        for tensor in (weight.data, transposed):
            digest = hashlib.sha256()
            assert _hash_tensor_blocks(digest, tensor) == tensor.numel() * tensor.element_size()
            digests.append(digest.hexdigest())
        assert digests[0] == digests[1]

    def test_preloaded_identity_follows_parameter_values(self, tokenizer):
        registries = {
            name: make_registry({"live": build_model(seed=seed)}, tokenizer, teacher_tokenizers={"live": tokenizer})
            for name, seed in (("first", 9), ("same", 9), ("other", 10))
        }
        first, same, other = (registries[name]["live"] for name in ("first", "same", "other"))
        assert first.content_hashed_bytes > 0
        # A different object with identical values stays the same source; different values do not.
        assert first.content_digest == same.content_digest
        assert first.source_key == same.source_key
        assert first.content_digest != other.content_digest
        assert first.source_key != other.source_key
        manifest_of(registries["first"], tokenizer).check_compatible(manifest_of(registries["same"], tokenizer))
        with pytest.raises(ValueError):
            manifest_of(registries["first"], tokenizer).check_compatible(manifest_of(registries["other"], tokenizer))


@require_torch_accelerator
class TestTeacherExecutorOnAccelerator:
    """Device-only properties: pinned staging, blocking device-to-host copies and device weight release.

    The CPU suite cannot establish these; `.to()` is a no-op copy there, so the substituted parameters alias the CPU
    source and the staging tile is unpinned.
    """

    def test_scoring_on_device_keeps_the_source_on_cpu(self, sources, tokenizer):
        registry = make_registry({"early": sources["a"]}, tokenizer)
        executor = TeacherExecutor(registry, FakeHeadCache(), torch.device(torch_device), scoring_batch_size=2)
        microbatch = make_microbatch([0, 0])
        block, positions, _ = score_single(executor, registry, microbatch)
        assert block.hidden.device.type == "cpu"
        assert executor._staging.is_pinned()
        assert executor.stats.live_device_weight_bytes == 0
        assert executor.stats.peak_device_weight_bytes > 0
        body = executor._load_body(0)
        assert {tensor.device.type for tensor in body.parameters()} == {"cpu"}
        source = executor.retain_head_source(0)
        assert source.weight.device.type == "cpu"
        reference_hidden, _ = reference_forward(sources["a"], microbatch)
        torch.testing.assert_close(block.hidden, reference_hidden[positions], rtol=1e-3, atol=1e-3)
