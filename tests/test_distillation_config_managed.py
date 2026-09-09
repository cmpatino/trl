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

"""Tests for `DistillationConfig`'s managed multi-teacher (MOPD) fields and validation."""

import pytest

from trl import DistillationConfig

from .testing_utils import TrlTestCase


class TestDistillationConfigManagedDefaults(TrlTestCase):
    def test_defaults(self):
        config = DistillationConfig(output_dir=self.tmp_dir)

        assert config.teacher_model_init_kwargs_by_teacher is None
        assert config.teacher_target_cache_bytes == 1 << 30
        assert config.teacher_scoring_batch_size == 1
        assert config.teacher_cpu_weight_budget_bytes is None
        assert config.teacher_gpu_weight_budget_bytes is None

    def test_teacher_model_init_kwargs_by_teacher_dict(self):
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs_by_teacher={"early": {"revision": "commit-a"}, "late": {"revision": "commit-b"}},
        )

        assert config.teacher_model_init_kwargs_by_teacher == {
            "early": {"revision": "commit-a"},
            "late": {"revision": "commit-b"},
        }

    def test_teacher_model_init_kwargs_by_teacher_json_string_is_parsed(self):
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs_by_teacher='{"early": {"revision": "commit-a"}, "late": {"revision": "commit-b"}}',
        )

        assert config.teacher_model_init_kwargs_by_teacher == {
            "early": {"revision": "commit-a"},
            "late": {"revision": "commit-b"},
        }
        # Nested values go through the same str -> bool/int/float coercion as every other `_VALID_DICT_FIELDS` entry.
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs_by_teacher='{"early": {"trust_remote_code": "true", "revision": "3"}}',
        )
        assert config.teacher_model_init_kwargs_by_teacher == {"early": {"trust_remote_code": True, "revision": 3}}

    def test_custom_budgets_and_batch_size(self):
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_target_cache_bytes=2048,
            teacher_scoring_batch_size=4,
            teacher_cpu_weight_budget_bytes=1024,
            teacher_gpu_weight_budget_bytes=512,
        )

        assert config.teacher_target_cache_bytes == 2048
        assert config.teacher_scoring_batch_size == 4
        assert config.teacher_cpu_weight_budget_bytes == 1024
        assert config.teacher_gpu_weight_budget_bytes == 512


class TestDistillationConfigManagedValidation(TrlTestCase):
    def test_non_positive_teacher_target_cache_bytes_raises(self):
        with pytest.raises(ValueError, match="teacher_target_cache_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_target_cache_bytes=0)
        with pytest.raises(ValueError, match="teacher_target_cache_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_target_cache_bytes=-1)

    def test_non_positive_teacher_scoring_batch_size_raises(self):
        with pytest.raises(ValueError, match="teacher_scoring_batch_size must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_scoring_batch_size=0)
        with pytest.raises(ValueError, match="teacher_scoring_batch_size must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_scoring_batch_size=-1)

    def test_non_positive_teacher_cpu_weight_budget_bytes_raises(self):
        with pytest.raises(ValueError, match="teacher_cpu_weight_budget_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_cpu_weight_budget_bytes=0)
        with pytest.raises(ValueError, match="teacher_cpu_weight_budget_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_cpu_weight_budget_bytes=-1)

    def test_non_positive_teacher_gpu_weight_budget_bytes_raises(self):
        with pytest.raises(ValueError, match="teacher_gpu_weight_budget_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_gpu_weight_budget_bytes=0)
        with pytest.raises(ValueError, match="teacher_gpu_weight_budget_bytes must be positive"):
            DistillationConfig(output_dir=self.tmp_dir, teacher_gpu_weight_budget_bytes=-1)

    def test_non_dict_per_teacher_kwargs_raises(self):
        with pytest.raises(ValueError, match="teacher_model_init_kwargs_by_teacher must map each teacher ID"):
            DistillationConfig(
                output_dir=self.tmp_dir,
                teacher_model_init_kwargs_by_teacher={"early": "commit-a"},
            )
        with pytest.raises(ValueError, match="teacher_model_init_kwargs_by_teacher must map each teacher ID"):
            DistillationConfig(
                output_dir=self.tmp_dir,
                teacher_model_init_kwargs_by_teacher={"early": ["revision", "commit-a"]},
            )
        with pytest.raises(ValueError, match="teacher_model_init_kwargs_by_teacher must map each teacher ID"):
            DistillationConfig(
                output_dir=self.tmp_dir,
                teacher_model_init_kwargs_by_teacher={"early": None},
            )


class TestDistillationConfigExistingBehaviorUnchanged(TrlTestCase):
    def test_beta_bounds_still_enforced(self):
        DistillationConfig(output_dir=self.tmp_dir, beta=0.0)
        DistillationConfig(output_dir=self.tmp_dir, beta=1.0)
        with pytest.raises(ValueError, match=r"beta must be in \[0.0, 1.0\]"):
            DistillationConfig(output_dir=self.tmp_dir, beta=-0.1)
        with pytest.raises(ValueError, match=r"beta must be in \[0.0, 1.0\]"):
            DistillationConfig(output_dir=self.tmp_dir, beta=1.1)

    def test_defaults_unrelated_to_managed_fields(self):
        config = DistillationConfig(output_dir=self.tmp_dir)

        assert config.teacher_model_name_or_path is None
        assert config.teacher_model_revision is None
        assert config.teacher_model_init_kwargs is None
        assert config.disable_dropout is False
