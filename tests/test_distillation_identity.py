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

"""Tests for tokenizer fingerprinting, the teacher manifest, and the metric-key helpers."""

import json

import pytest
import torch
from tokenizers import Tokenizer as RustTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from trl.trainer._distillation_identity import (
    TeacherManifest,
    canonical_tokenizer_payload,
    teacher_metric_key,
    teacher_metrics_from_stats,
    tokenizer_fingerprint,
)

from .testing_utils import TrlTestCase


TOKENIZER_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"


class NotAFastTokenizer:
    """Stand-in for a slow tokenizer.

    transformers 5.16.1 unifies fast/slow tokenizer implementations onto a `tokenizers`-backed class for every
    checkpoint used here, so no real slow tokenizer is easily instantiable in this environment. This minimal object
    is simply not an instance of `PreTrainedTokenizerFast`, which is the only thing the guard clause checks.
    """


class TestTokenizerFingerprint(TrlTestCase):
    def test_identical_tokenizers_from_two_loads_match(self):
        tokenizer_a = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        tokenizer_b = AutoTokenizer.from_pretrained(TOKENIZER_ID)

        assert tokenizer_fingerprint(tokenizer_a) == tokenizer_fingerprint(tokenizer_b)

    def test_cosmetic_json_formatting_does_not_change_fingerprint(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

        # Reserialize the backend tokenizer with different whitespace and reordered keys, then rebuild a fast
        # tokenizer from that string. The rendering behavior is identical, so the fingerprint must be too.
        data = json.loads(tokenizer.backend_tokenizer.to_str(pretty=False))
        reformatted = json.dumps(data, indent=4, sort_keys=True)
        rust_tokenizer = RustTokenizer.from_str(reformatted)
        reformatted_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=rust_tokenizer,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
            unk_token=tokenizer.unk_token,
        )

        assert tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(reformatted_tokenizer)

    def test_added_token_changes_fingerprint(self):
        base_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        modified_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        modified_tokenizer.add_tokens(["<|my_new_token|>"])

        assert tokenizer_fingerprint(base_tokenizer) != tokenizer_fingerprint(modified_tokenizer)

    def test_changing_eos_token_changes_fingerprint(self):
        base_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        modified_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        modified_tokenizer.eos_token = modified_tokenizer.pad_token

        assert tokenizer_fingerprint(base_tokenizer) != tokenizer_fingerprint(modified_tokenizer)

    def test_slow_tokenizer_raises_type_error(self):
        with pytest.raises(TypeError, match="fast"):
            canonical_tokenizer_payload(NotAFastTokenizer())
        with pytest.raises(TypeError, match="fast"):
            tokenizer_fingerprint(NotAFastTokenizer())

    def test_payload_contains_special_tokens_and_model_input_names(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

        payload = canonical_tokenizer_payload(tokenizer)

        assert payload["special_tokens"]["eos"] == (tokenizer.eos_token, tokenizer.eos_token_id)
        assert payload["special_tokens"]["pad"] == (tokenizer.pad_token, tokenizer.pad_token_id)
        assert payload["special_tokens"]["bos"] == (None, None)
        assert payload["model_input_names"] == tokenizer.model_input_names
        assert "padding_side" not in payload
        assert "truncation_side" not in payload


def _registry_manifest() -> dict:
    return {
        "teachers": [
            {
                "id": "early",
                "index": 0,
                "source": "org/teacher",
                "resolved_revision": "commit-a",
                "source_key": "org/teacher@commit-a",
                "tokenizer_fingerprint": "fp-early",
                "source_dtype": torch.bfloat16,
                "hidden_dtype": torch.bfloat16,
                "projection_dtype": torch.float32,
                "adapter_version": 1,
                "token": "hf_should_never_be_persisted",
                "head": {
                    "weight_shape": (152000, 4096),
                    "has_bias": False,
                    "source_dtype": torch.bfloat16,
                    "logit_scale": 1.0,
                    "final_logit_softcapping": None,
                    "transform_version": 1,
                },
            },
            {
                "id": "late",
                "index": 1,
                "source": "org/teacher",
                "resolved_revision": "commit-b",
                "source_key": "org/teacher@commit-b",
                "tokenizer_fingerprint": "fp-late",
                "source_dtype": torch.float16,
                "hidden_dtype": torch.float16,
                "projection_dtype": torch.float16,
                "adapter_version": 1,
                "use_auth_token": "hf_should_also_never_be_persisted",
                "head": {
                    "weight_shape": (152000, 2048),
                    "has_bias": True,
                    "source_dtype": torch.float16,
                    "logit_scale": 1.0,
                    "final_logit_softcapping": 30.0,
                    "transform_version": 1,
                },
            },
        ]
    }


def _build_manifest() -> TeacherManifest:
    return TeacherManifest.from_registry(
        _registry_manifest(),
        student_tokenizer_fingerprint="student-fp",
        beta=1.0,
        temperature=1.0,
        chunk_size=256,
    )


class TestTeacherManifest(TrlTestCase):
    def test_from_registry_serializes_dtypes_and_shapes(self):
        manifest = _build_manifest()

        assert manifest.version == 1
        early, late = manifest.teachers
        assert early["source_dtype"] == "torch.bfloat16"
        assert early["hidden_dtype"] == "torch.bfloat16"
        assert early["projection_dtype"] == "torch.float32"
        assert early["head"]["source_dtype"] == "torch.bfloat16"
        assert early["head"]["weight_shape"] == [152000, 4096]
        assert late["head"]["final_logit_softcapping"] == 30.0

    def test_from_registry_never_persists_credential_looking_keys(self):
        manifest = _build_manifest()

        for teacher in manifest.teachers:
            assert "token" not in teacher
            assert "use_auth_token" not in teacher

    def test_save_load_round_trip(self):
        manifest = _build_manifest()

        manifest.save(self.tmp_dir)
        loaded = TeacherManifest.load(self.tmp_dir)

        assert loaded == manifest

    def test_save_writes_allowlisted_keys_only_and_no_credentials(self):
        manifest = _build_manifest()
        manifest.save(self.tmp_dir)

        with open(f"{self.tmp_dir}/teacher_manifest.json", encoding="utf-8") as f:
            raw = f.read()

        assert "hf_should_never_be_persisted" not in raw
        assert "hf_should_also_never_be_persisted" not in raw
        assert '"token"' not in raw
        assert '"use_auth_token"' not in raw

    def test_check_compatible_accepts_identical_manifest(self):
        manifest = _build_manifest()
        manifest.check_compatible(_build_manifest())  # must not raise

    def test_check_compatible_rejects_reordered_teacher_ids(self):
        manifest = _build_manifest()
        other_registry = _registry_manifest()
        other_registry["teachers"] = list(reversed(other_registry["teachers"]))
        other = TeacherManifest.from_registry(
            other_registry, student_tokenizer_fingerprint="student-fp", beta=1.0, temperature=1.0, chunk_size=256
        )

        with pytest.raises(ValueError, match="teacher ids/order"):
            manifest.check_compatible(other)

    def test_check_compatible_reports_differing_field_by_name(self):
        manifest = _build_manifest()
        other_registry = _registry_manifest()
        other_registry["teachers"][1]["tokenizer_fingerprint"] = "fp-late-CHANGED"
        other = TeacherManifest.from_registry(
            other_registry, student_tokenizer_fingerprint="student-fp", beta=1.0, temperature=1.0, chunk_size=256
        )

        with pytest.raises(ValueError, match="late") as exc_info:
            manifest.check_compatible(other)
        assert "tokenizer_fingerprint" in str(exc_info.value)

    def test_check_compatible_reports_multiple_differences(self):
        manifest = _build_manifest()
        other = TeacherManifest.from_registry(
            _registry_manifest(), student_tokenizer_fingerprint="student-fp", beta=0.5, temperature=2.0, chunk_size=256
        )

        with pytest.raises(ValueError) as exc_info:
            manifest.check_compatible(other)
        message = str(exc_info.value)
        assert "beta" in message
        assert "temperature" in message

    def test_check_compatible_reports_head_field_mismatch(self):
        manifest = _build_manifest()
        other_registry = _registry_manifest()
        other_registry["teachers"][0]["head"]["has_bias"] = True
        other = TeacherManifest.from_registry(
            other_registry, student_tokenizer_fingerprint="student-fp", beta=1.0, temperature=1.0, chunk_size=256
        )

        with pytest.raises(ValueError, match=r"head\.has_bias"):
            manifest.check_compatible(other)


class TestTeacherMetricKey(TrlTestCase):
    def test_simple_id(self):
        assert teacher_metric_key("teacher_jsd", "math") == "teacher_jsd/math"

    def test_slash_in_id_does_not_collide(self):
        key_a = teacher_metric_key("teacher_jsd", "a/b")
        key_b = teacher_metric_key("teacher_jsd", "a")  # would collide with "a/b" scored under prefix "b" otherwise

        assert key_a != "teacher_jsd/a/b"
        assert key_a != key_b
        assert "/" not in key_a[len("teacher_jsd/") :]

    def test_whitespace_in_id_is_encoded(self):
        key = teacher_metric_key("teacher_jsd", "math teacher")

        assert " " not in key


class TestTeacherMetricsFromStats(TrlTestCase):
    def test_all_zero_stats_returns_empty_dict(self):
        stats = torch.zeros(3, 2)

        assert teacher_metrics_from_stats(stats, ["math", "code"]) == {}

    def test_shapes_and_values(self):
        # rows: divergence_sum, entropy_sum, count
        stats = torch.tensor(
            [
                [4.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
            ]
        )

        metrics = teacher_metrics_from_stats(stats, ["math", "code"])

        assert metrics[teacher_metric_key("teacher_jsd", "math")] == pytest.approx(2.0)
        assert metrics[teacher_metric_key("teacher_entropy", "math")] == pytest.approx(1.0)
        assert metrics[teacher_metric_key("teacher_token_frac", "math")] == pytest.approx(1.0)
        # `code` had zero valid tokens: means are omitted, but token_frac is still reported as 0.0.
        assert teacher_metric_key("teacher_jsd", "code") not in metrics
        assert teacher_metric_key("teacher_entropy", "code") not in metrics
        assert metrics[teacher_metric_key("teacher_token_frac", "code")] == pytest.approx(0.0)

    def test_absent_teacher_omits_means_but_keeps_token_frac(self):
        stats = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        metrics = teacher_metrics_from_stats(stats, ["only", "unused_a", "unused_b"])

        assert set(metrics) == {
            teacher_metric_key("teacher_jsd", "only"),
            teacher_metric_key("teacher_entropy", "only"),
            teacher_metric_key("teacher_token_frac", "only"),
            teacher_metric_key("teacher_token_frac", "unused_a"),
            teacher_metric_key("teacher_token_frac", "unused_b"),
        }
