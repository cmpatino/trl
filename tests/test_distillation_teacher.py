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

import weakref
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from trl.trainer import _distillation_teacher as teacher_module
from trl.trainer._distillation_teacher import (
    HiddenTargetBlock,
    ScoreRequest,
    TeacherExecutor,
    TeacherRegistry,
    WindowStore,
    _resolve_hub_snapshot,
    _tokenizer_fingerprint,
)


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
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


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
    return TeacherRegistry(
        teacher_models, student_tokenizer=tokenizer, student_vocab_size=VOCAB_SIZE, **kwargs
    )


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
        1, torch.zeros((positions.numel(), hidden_size), dtype=entry.hidden_dtype), [None] * positions.numel()
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
            make_registry(
                {"early": sources["a"]}, tokenizer, per_teacher_init_kwargs={"late": {"dtype": "float32"}}
            )

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
            make_registry(
                {"early": sources["a"]}, tokenizer, per_teacher_init_kwargs={"early": {"revision": "main"}}
            )

    def test_vocab_size_mismatch(self, sources, tokenizer):
        with pytest.raises(ValueError, match="but the student has vocab_size 8"):
            TeacherRegistry(
                {"early": sources["a"]}, student_tokenizer=tokenizer, student_vocab_size=8
            )

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
            make_registry(
                {"live": teacher}, tokenizer, teacher_tokenizers={"live": tokenizer}, student_model=student
            )

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
        assert manifest["student_tokenizer_fingerprint"] == _tokenizer_fingerprint(tokenizer)
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
            "projection_dtype",
            "weight_shape",
            "has_bias",
            "logit_scale",
            "final_logit_softcapping",
            "transform_version",
            "adapter_version",
            "evictable",
            "storage_bytes",
            "loading",
        }
        assert set(teacher) == expected
        assert teacher["weight_shape"] == [VOCAB_SIZE, HIDDEN_SIZE]
        assert teacher["config_class"] == "Qwen3Config"
        assert "hf_secret" not in str(manifest)
        assert set(teacher["loading"]) <= {"dtype", "revision", "attn_implementation", "low_cpu_mem_usage",
                                           "trust_remote_code"}

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
        assert executor.capabilities == {"hidden_targets": True, "requires_collective_schedule": False}
        assert registry["early"].hidden_dtype == torch.float32
        assert registry["early"].projection_dtype == torch.float32

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
        assert torch.equal(block.hidden @ head.weight.t(), reference_logits[positions])

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
        assert entry.hidden_dtype == torch.bfloat16 and entry.projection_dtype == torch.bfloat16
        assert entry.source_dtype == torch.float32  # the CPU source keeps its own dtype
        microbatch = make_microbatch([0])
        block, positions, result = score_single(executor, registry, microbatch)
        assert block.hidden.dtype == torch.bfloat16 and result.hidden_dtype == torch.bfloat16
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
        embedding_ref = weakref.ref(body.model.embed_tokens.weight)
        del body
        executor._load_body(1)  # evicts the first body from the single CPU slot
        assert module_ref() is None
        assert backbone_ref() is None
        assert embedding_ref() is None
        # The head source survives its body and still projects the same logits.
        microbatch = make_microbatch([0])
        reference_hidden, reference_logits = reference_forward(sources["a"], microbatch)
        positions = torch.arange(reference_hidden.shape[0])
        assert torch.equal(reference_hidden @ head.weight.t(), reference_logits[positions])
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
