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

"""Integration tests for the managed multi-teacher (`teacher_models`) path of `DistillationTrainer`."""

import json
import os
from unittest.mock import patch

import pytest
import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from trl import DistillationConfig, DistillationTrainer
from trl.trainer._distillation_teacher import (
    _TARGET_BLOCK_OVERHEAD_BYTES,
    TeacherExecutor,
    WindowStore,
    _as_cpu,
    _microbatch_fingerprint,
)

from .testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
# Registered teachers are materialized in float32 (the registry's default), and the tiny model's hidden width is 8.
TEACHER_HIDDEN_SIZE = 8
TEACHER_TARGET_ITEMSIZE = 4


def _teacher_checkpoint(path: str, scale: float | None = None) -> str:
    """Save a standalone tiny checkpoint, optionally with every weight scaled so its targets really differ."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    if scale is not None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale)
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    """Two local teacher checkpoints with deliberately different weights, shared by every test in this module."""
    directory = tmp_path_factory.mktemp("managed-teachers")
    return {
        "a": _teacher_checkpoint(str(directory / "a")),
        "b": _teacher_checkpoint(str(directory / "b"), scale=1.05),
    }


def _prompts(count: int) -> list[str]:
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
    prompts = list(dataset["prompt"])
    return [prompts[index % len(prompts)] for index in range(count)]


def _routed_dataset(teacher_ids: list[str]) -> Dataset:
    return Dataset.from_dict({"prompt": _prompts(len(teacher_ids)), "teacher_id": teacher_ids})


def _one_microbatch_budget(rows: int) -> int:
    """Target budget that admits exactly one microbatch of `rows` completion positions and never two."""
    return _TARGET_BLOCK_OVERHEAD_BYTES + rows * TEACHER_HIDDEN_SIZE * TEACHER_TARGET_ITEMSIZE


class TestManagedTraining(TrlTestCase):
    def test_two_teachers_train_and_log_per_teacher_metrics(self, teachers):
        # Item 1: two distinct local checkpoints routed by a `teacher_id` column train end to end, the per-teacher
        # metrics are logged under both IDs, and the teacher manifest is written next to the checkpoint.
        dataset = _routed_dataset(["a", "b"] * 6)
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            max_completion_length=8,
            max_steps=2,
            save_strategy="steps",
            save_steps=2,
            logging_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID, args=training_args, train_dataset=dataset, teacher_models=teachers
        )
        previous_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        step_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
        assert step_logs, "no training step was logged"
        for entry in step_logs:
            assert not torch.isnan(torch.tensor(entry["loss"]))
            for key in ("teacher_jsd/a", "teacher_jsd/b", "teacher_token_frac/a", "teacher_token_frac/b"):
                assert key in entry, f"{key} missing from {sorted(entry)}"
            assert entry["teacher_jsd/a"] != entry["teacher_jsd/b"], "the two teachers produced the same divergence"
            frac_sum = entry["teacher_token_frac/a"] + entry["teacher_token_frac/b"]
            assert frac_sum == pytest.approx(1.0)

        for name, param in previous_params.items():
            assert not torch.equal(param, trainer.model.get_parameter(name)), f"Parameter {name} has not changed."

        manifest_path = os.path.join(self.tmp_dir, "checkpoint-2", "teacher_manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        assert [teacher["id"] for teacher in manifest["teachers"]] == ["a", "b"]
        assert manifest["teachers"][0]["source_key"] != manifest["teachers"][1]["source_key"]
        # No credentials or loading options leak into the manifest.
        assert "loading" not in manifest["teachers"][0]
        # Nothing is left live once training returns.
        assert trainer._teacher_store.live_keys == []
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        assert trainer._teacher_executor.stats.live_device_weight_bytes == 0

    def test_single_managed_teacher_matches_the_legacy_trainer(self, teachers):
        # Item 2: with one registered teacher and no `teacher_id` column, the managed path must reproduce the legacy
        # single-teacher path exactly — same losses and same parameters after two optimizer steps.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def run(**trainer_kwargs):
            training_args = DistillationConfig(
                output_dir=self.tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                max_completion_length=8,
                max_steps=2,
                logging_steps=1,
                seed=42,
                report_to="none",
                # One teacher forward per microbatch, exactly like the legacy path, so the matmul shapes match and
                # parity can be asserted bitwise rather than within a tolerance.
                teacher_scoring_batch_size=3,
            )
            trainer = DistillationTrainer(model=MODEL_ID, args=training_args, train_dataset=dataset, **trainer_kwargs)
            set_seed(42)
            trainer.train()
            losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
            return losses, {name: param.detach().clone() for name, param in trainer.model.named_parameters()}

        legacy_losses, legacy_params = run(teacher_model=teachers["a"])
        managed_losses, managed_params = run(teacher_models={"only": teachers["a"]})

        assert managed_losses == legacy_losses
        assert sorted(managed_params) == sorted(legacy_params)
        for name, legacy_param in legacy_params.items():
            assert torch.equal(managed_params[name], legacy_param), f"Parameter {name} differs from the legacy run."

    @pytest.mark.parametrize("gradient_accumulation_steps", [1, 2, 4])
    def test_small_scoring_windows_give_the_same_update(self, teachers, gradient_accumulation_steps):
        # Item 5: a target budget that admits only one microbatch forces one scoring window per microbatch. The
        # update must be identical to a run whose window covers the whole accumulation.
        dataset = _routed_dataset(["a", "b"] * 12)
        completion_length = 4

        def run(target_cache_bytes):
            training_args = DistillationConfig(
                output_dir=self.tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_completion_length=completion_length,
                max_steps=2,
                logging_steps=1,
                seed=7,
                report_to="none",
                teacher_target_cache_bytes=target_cache_bytes,
            )
            trainer = DistillationTrainer(
                model=MODEL_ID, args=training_args, train_dataset=dataset, teacher_models=teachers
            )
            plans = []
            plan_window = trainer._teacher_store.plan_window

            def counting_plan_window(*args, **kwargs):
                plan = plan_window(*args, **kwargs)
                plans.append(plan)
                return plan

            trainer._teacher_store.plan_window = counting_plan_window
            set_seed(7)
            trainer.train()
            losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
            params = {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
            return losses, params, plans

        wide_losses, wide_params, wide_plans = run(1 << 30)
        narrow_losses, narrow_params, narrow_plans = run(_one_microbatch_budget(completion_length))

        # The wide run scores each generation batch in one window; the narrow one needs a window per microbatch.
        training_plans = [plan for plan in wide_plans if plan.generation_id > 0]
        assert all(len(plan.microbatch_indices) == gradient_accumulation_steps for plan in training_plans)
        narrow_training_plans = [plan for plan in narrow_plans if plan.generation_id > 0]
        assert all(len(plan.microbatch_indices) == 1 for plan in narrow_training_plans)
        assert len(narrow_training_plans) == gradient_accumulation_steps * len(training_plans)

        assert narrow_losses == wide_losses
        for name, wide_param in wide_params.items():
            assert torch.equal(narrow_params[name], wide_param), f"Parameter {name} depends on the window size."


class TestManagedRouting(TrlTestCase):
    def test_missing_teacher_id_with_two_teachers_raises(self, teachers):
        # Item 3: with more than one teacher registered every row must name one.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID, args=training_args, train_dataset=dataset, teacher_models=teachers
        )
        with pytest.raises(ValueError, match="no `teacher_id`"):
            trainer.train()

    def test_unknown_teacher_id_raises(self, teachers):
        dataset = _routed_dataset(["a", "nope"] * 3)
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID, args=training_args, train_dataset=dataset, teacher_models=teachers
        )
        with pytest.raises(ValueError, match="Unknown teacher ID"):
            trainer.train()

    def test_teacher_models_and_teacher_model_together_raise(self, teachers):
        with pytest.raises(ValueError, match="Pass only one"):
            DistillationTrainer(
                model=MODEL_ID,
                teacher_model=teachers["a"],
                teacher_models=teachers,
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
            )

    def test_teacher_models_and_config_name_or_path_together_raise(self, teachers):
        args = DistillationConfig(output_dir=self.tmp_dir, teacher_model_name_or_path=teachers["a"], report_to="none")
        with pytest.raises(ValueError, match="Pass only one"):
            DistillationTrainer(model=MODEL_ID, args=args, teacher_models=teachers)

    def test_singular_revision_with_teacher_models_raises(self, teachers):
        args = DistillationConfig(output_dir=self.tmp_dir, teacher_model_revision="main", report_to="none")
        with pytest.raises(ValueError, match="ambiguous"):
            DistillationTrainer(model=MODEL_ID, args=args, teacher_models=teachers)

    def test_managed_only_arguments_rejected_on_the_legacy_path(self, teachers):
        args = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs_by_teacher={"a": {"revision": "main"}},
            report_to="none",
        )
        with pytest.raises(ValueError, match="only applies to managed"):
            DistillationTrainer(model=MODEL_ID, teacher_model=teachers["a"], args=args)
        with pytest.raises(ValueError, match="only applies to managed"):
            DistillationTrainer(
                model=MODEL_ID,
                teacher_model=teachers["a"],
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
                teacher_tokenizers={"a": AutoTokenizer.from_pretrained(MODEL_ID)},
            )


class TestManagedBackendChecks(TrlTestCase):
    """The unsupported-mode rejections. Only the checks reachable without an accelerator are exercised here."""

    def _trainer(self, teachers):
        args = DistillationConfig(
            output_dir=self.tmp_dir, per_device_train_batch_size=1, max_completion_length=2, report_to="none"
        )
        return DistillationTrainer(
            model=MODEL_ID, args=args, train_dataset=_routed_dataset(["a"]), teacher_models={"a": teachers["a"]}
        )

    def test_quantized_teacher_kwargs_are_rejected_before_loading(self, teachers):
        args = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs={"load_in_4bit": True},
            report_to="none",
        )
        with pytest.raises(ValueError, match="load_in_4bit"):
            DistillationTrainer(model=MODEL_ID, args=args, teacher_models={"a": teachers["a"]})

    def test_per_teacher_quantized_kwargs_are_rejected(self, teachers):
        args = DistillationConfig(
            output_dir=self.tmp_dir,
            teacher_model_init_kwargs_by_teacher={"a": {"device_map": "auto"}},
            report_to="none",
        )
        with pytest.raises(ValueError, match="device_map"):
            DistillationTrainer(model=MODEL_ID, args=args, teacher_models={"a": teachers["a"]})

    def test_zero3_fsdp1_and_tensor_parallel_are_rejected(self, teachers):
        # `_check_managed_backend` is re-run against patched backend state: constructing a real ZeRO-3 / FSDP1 /
        # tensor-parallel accelerator needs an accelerator this environment does not have.
        trainer = self._trainer(teachers)
        trainer._dist.zero_stage = 3
        with pytest.raises(ValueError, match="ZeRO-3"):
            trainer._check_managed_backend({})
        trainer._dist.zero_stage = 0
        trainer._dist.fsdp_version = 1
        with pytest.raises(ValueError, match="FSDP version 1"):
            trainer._check_managed_backend({})
        trainer._dist.fsdp_version = None

        class _TensorParallel:
            tp_enabled = True
            cp_enabled = False
            sp_enabled = False

        trainer.args.parallelism_config = _TensorParallel()
        with pytest.raises(ValueError, match="tensor / context / sequence parallelism"):
            trainer._check_managed_backend({})
        trainer.args.parallelism_config = None

        trainer._is_vlm = True
        with pytest.raises(ValueError, match="vision-language"):
            trainer._check_managed_backend({})


class TestManagedMasks(TrlTestCase):
    def test_all_masked_and_partially_masked_batches_still_step(self, teachers):
        # Item 4: a nonempty microbatch whose loss mask is entirely zero still runs the student forward/backward with
        # a finite zero, and a partially masked one trains normally.
        class MaskingTrainer(DistillationTrainer):
            def __init__(self, *args, mask_mode, **kwargs):
                self.mask_mode = mask_mode
                super().__init__(*args, **kwargs)

            def _generate_and_score_completions(self, inputs):
                output = super()._generate_and_score_completions(inputs)
                mask = output["completion_mask"]
                if self.mask_mode == "all":
                    output["tool_mask"] = torch.zeros_like(mask)
                else:
                    tool_mask = torch.ones_like(mask)
                    tool_mask[::2] = 0
                    output["tool_mask"] = tool_mask
                return output

        for mask_mode in ("all", "half"):
            training_args = DistillationConfig(
                output_dir=self.tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=2,
                max_completion_length=4,
                max_steps=2,
                logging_steps=1,
                report_to="none",
            )
            trainer = MaskingTrainer(
                model=MODEL_ID,
                args=training_args,
                train_dataset=_routed_dataset(["a", "b"] * 4),
                teacher_models=teachers,
                mask_mode=mask_mode,
            )
            trainer.train()
            losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
            assert losses, f"no step logged for mask_mode={mask_mode}"
            assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)
            if mask_mode == "all":
                assert all(loss == 0.0 for loss in losses)
            assert trainer._teacher_store.live_keys == []

    def test_zero_row_microbatch_is_a_scheduling_error(self, teachers):
        # Item 4: a zero-row microbatch is rejected rather than silently skipped.
        training_args = DistillationConfig(
            output_dir=self.tmp_dir, per_device_train_batch_size=1, max_completion_length=2, report_to="none"
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a"]),
            teacher_models={"a": teachers["a"]},
        )
        store = WindowStore(trainer._teacher_registry)
        empty = {
            "prompt_ids": torch.zeros((0, 3), dtype=torch.long),
            "prompt_mask": torch.zeros((0, 3), dtype=torch.long),
            "completion_ids": torch.zeros((0, 2), dtype=torch.long),
            "completion_mask": torch.zeros((0, 2), dtype=torch.long),
            "teacher_index": torch.zeros((0,), dtype=torch.int64),
        }
        with pytest.raises(ValueError, match="no rows"):
            store.plan_window([empty], target_cache_bytes=1 << 20)


class TestManagedDeviceBoundaries(TrlTestCase):
    """
    The generation payload lives on the accelerator; the window store's bookkeeping is host-side by contract.

    A device payload tensor combined with a host one silently works on CPU and raises `RuntimeError: Expected all
    tensors to be on the same device` on an accelerator, which is how the first GPU run of `mopd_parity.py --mode
    ddp` failed inside `WindowStore._microbatch_groups`. These tests pin the two halves of the contract that *are*
    observable without an accelerator: every payload tensor shares one device, and everything the store hands back is
    on the host. The cross-device behaviour itself can only be covered by the GPU job (gate AB/DDP in
    `implementation/gpu/run_gate.sh`).
    """

    def test_managed_bookkeeping_stays_on_the_host(self, teachers):
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
        )
        trainer.model.train()
        batch = next(iter(trainer.get_train_dataloader()))
        inputs = trainer._prepare_inputs(batch)

        # Every routing/payload tensor sits on the accelerator device, next to the tokens and the masks: a host
        # `teacher_index` beside a device `completion_mask` is exactly what broke on GPU. Compared by device *type*:
        # Accelerate's `device` carries no index (`cuda`), while a tensor's does (`cuda:0`).
        payload_device = inputs["completion_mask"].device
        assert payload_device.type == trainer.accelerator.device.type
        for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "teacher_index"):
            assert inputs[key].device == payload_device, f"{key} is on {inputs[key].device}, not {payload_device}"
        assert inputs["teacher_index"].dtype == torch.int64
        assert inputs["teacher_index"].shape == (inputs["completion_ids"].shape[0],)

        # Everything the store produced is host-side: the loss moves the positions and the target rows itself.
        groups = trainer._teacher_store.targets_for(inputs["_teacher_targets_key"], inputs)
        assert groups, "no target groups were scored"
        for group in groups:
            assert group.positions.device.type == "cpu"
            assert group.positions.dtype == torch.int64
            assert group.hidden.device.type == "cpu"
        # And the fingerprint the store compares is a host-side digest of the payload values, not device tensors.
        key = inputs["_teacher_targets_key"]
        fingerprint = _microbatch_fingerprint(inputs, *key)
        assert isinstance(fingerprint, str) and len(fingerprint) == 64
        assert fingerprint == _microbatch_fingerprint(inputs, *key)

        trainer._teacher_store.release(inputs["_teacher_targets_key"])
        trainer.close_teachers()

    def test_scoring_autocast_dtype_follows_the_loss_context_then_deepspeed(self, teachers):
        # The teachers must be scored in the precision the student's loss really computes in. Two sources disagree
        # and both matter: `Trainer.compute_loss_context_manager()` is authoritative when it autocasts, but under
        # DeepSpeed it is a null context while the engine still computes in bf16, so the plugin's dtype has to be
        # used there. Getting this wrong made the ZeRO gates score fp32 teachers against a bf16 student.
        training_args = DistillationConfig(
            output_dir=self.tmp_dir, per_device_train_batch_size=1, max_completion_length=2, report_to="none"
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a"]),
            teacher_models={"a": teachers["a"]},
        )
        # This CPU environment autocasts nowhere and is not DeepSpeed, so no autocast: what the executor was built
        # with, and what makes managed-vs-legacy parity bitwise here.
        assert trainer._scoring_autocast_dtype() is None
        assert trainer._teacher_executor.autocast_dtype is None

        # A null loss context plus a DeepSpeed engine: take the dtype from the plugin. `mixed_precision` is a
        # read-only property, so it is patched on the class for the duration of each check.
        accelerator_type = type(trainer.accelerator)
        trainer.is_deepspeed_enabled = True
        for precision, expected in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("no", None)):
            with patch.object(accelerator_type, "mixed_precision", precision):
                assert trainer._scoring_autocast_dtype() is expected, precision

        # An autocasting loss context wins over the plugin, whichever backend is active.
        with patch.object(accelerator_type, "mixed_precision", "bf16"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                assert trainer._scoring_autocast_dtype() is torch.bfloat16
            trainer.is_deepspeed_enabled = False
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                assert trainer._scoring_autocast_dtype() is torch.bfloat16
            # Not DeepSpeed and no autocast: no autocast for scoring either.
            assert trainer._scoring_autocast_dtype() is None

    def test_as_cpu_converts_only_non_host_tensors(self):
        host = torch.zeros(3, dtype=torch.int64)
        assert _as_cpu(host) is host, "an already-host tensor must not be copied"

        class _OffHost:
            """Minimal stand-in for a device tensor: `_as_cpu` reads `.device.type` and calls `.cpu()`."""

            def __init__(self, replacement):
                self.device = torch.device("cuda", 0)
                self.replacement = replacement
                self.calls = 0

            def cpu(self):
                self.calls += 1
                return self.replacement

        replacement = torch.ones(2, dtype=torch.int64)
        off_host = _OffHost(replacement)
        assert _as_cpu(off_host) is replacement
        assert off_host.calls == 1


class TestManagedEvaluation(TrlTestCase):
    def test_evaluation_on_start_nested_and_every_step_with_a_tight_budget(self, teachers):
        # Item 6: `eval_on_start`, an evaluation nested in every training accumulation, and a target budget that
        # admits only one microbatch (so every window boundary replans) all coexist.
        completion_length = 4
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_completion_length=completion_length,
            max_steps=2,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            eval_on_start=True,
            report_to="none",
            teacher_target_cache_bytes=_one_microbatch_budget(completion_length),
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 6),
            eval_dataset=_routed_dataset(["a", "b"]),
            teacher_models=teachers,
        )
        trainer.train()

        eval_logs = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        assert len(eval_logs) >= 3, f"expected eval_on_start plus one per step, got {len(eval_logs)}"
        for entry in eval_logs:
            assert not torch.isnan(torch.tensor(entry["eval_loss"]))
        assert any("eval_teacher_jsd/a" in entry or "eval_teacher_jsd/b" in entry for entry in eval_logs)
        assert trainer._teacher_store.live_keys == []
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0

    def test_nested_evaluation_mid_accumulation_drops_and_rescores_the_training_window(self, teachers):
        # Item 6, the nested-evaluation policy: an evaluation that starts while a training scoring window is still
        # partly unconsumed must hand the target budget over, keep the buffered tokens and the consumption position,
        # and let the next training microbatch rescore what is left. `Trainer` only evaluates at optimizer-step
        # boundaries, where the window is always fully consumed, so the callback evaluates on a *substep* end.
        observations = []

        class EvaluatingCallback(TrainerCallback):
            def on_substep_end(self, args, state, control, **kwargs):
                trainer = self.trainer
                observations.append(
                    {
                        "when": "before_eval",
                        "live": list(trainer._teacher_store.live_keys),
                        "window": sorted(trainer._teacher_window),
                        "step": trainer._step,
                        "buffered": [id(batch["completion_ids"]) for batch in trainer._buffered_inputs],
                    }
                )
                trainer.evaluate()
                observations.append(
                    {
                        "when": "after_eval",
                        "live": list(trainer._teacher_store.live_keys),
                        "window": sorted(trainer._teacher_window),
                        "step": trainer._step,
                        "buffered": [id(batch["completion_ids"]) for batch in trainer._buffered_inputs],
                    }
                )

        callback = EvaluatingCallback()
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=2,
            logging_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 6),
            eval_dataset=_routed_dataset(["a", "b"]),
            teacher_models=teachers,
            callbacks=[callback],
        )
        callback.trainer = trainer
        plans = []
        plan_window = trainer._teacher_store.plan_window

        def counting_plan_window(*args, **kwargs):
            plan = plan_window(*args, **kwargs)
            plans.append(plan)
            return plan

        trainer._teacher_store.plan_window = counting_plan_window
        previous_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train()

        assert observations, "the callback never ran on a substep end"
        before = observations[0]
        after = observations[1]
        # The window really was partly unconsumed when the evaluation started, and the evaluation dropped it.
        assert before["window"], "nothing was scored when the nested evaluation started"
        assert after["window"] == [] and after["live"] == []
        # Buffered tokens and the training consumption position survive the evaluation untouched.
        assert after["buffered"] == before["buffered"]
        assert after["step"] == before["step"]
        # The dropped microbatch was rescored: the same training generation batch was planned more than once.
        training_plans = [plan for plan in plans if plan.generation_id > 0]
        rescored = [plan for plan in training_plans if plan.microbatch_indices and plan.microbatch_indices[0] > 0]
        assert rescored, f"the remaining window was never rescored; plans: {[p.microbatch_indices for p in plans]}"
        # And training still completed and moved the student.
        assert trainer.state.log_history[-1]["train_loss"] is not None
        for name, param in previous_params.items():
            assert not torch.equal(param, trainer.model.get_parameter(name)), f"Parameter {name} has not changed."
        assert trainer._teacher_store.live_keys == []

    def test_evaluation_batch_that_does_not_fit_the_budget_raises(self, teachers):
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=4,
            max_completion_length=8,
            report_to="none",
            teacher_target_cache_bytes=_TARGET_BLOCK_OVERHEAD_BYTES + 8,
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a"] * 4),
            eval_dataset=_routed_dataset(["a"] * 4),
            teacher_models={"a": teachers["a"]},
        )
        with pytest.raises(ValueError, match="teacher_target_cache_bytes"):
            trainer.evaluate()
        # The failed evaluation left nothing behind.
        assert trainer._teacher_store.live_keys == []

    def test_close_teachers_then_train_again_reopens(self, teachers):
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=1,
            logging_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
        )
        trainer.train()
        trainer.close_teachers()
        trainer.close_teachers()  # idempotent
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        # Registry metadata survives closure, so the next run reloads from the pinned snapshots.
        assert trainer._teacher_registry.teacher_ids == ["a", "b"]
        trainer.state.max_steps = 2
        trainer.args.max_steps = 2
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_close_teachers_while_targets_are_live_raises(self, teachers):
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
        )
        # Drive one microbatch through scoring without running the loss, so its targets stay live.
        trainer.model.train()
        batch = next(iter(trainer.get_train_dataloader()))
        inputs = trainer._prepare_inputs(batch)
        assert trainer._teacher_store.live_keys == [inputs["_teacher_targets_key"]]
        with pytest.raises(RuntimeError, match="still live"):
            trainer.close_teachers()
        trainer._teacher_store.release(inputs["_teacher_targets_key"])
        trainer.close_teachers()


class TestManagedResume(TrlTestCase):
    def _args(self, max_steps):
        return DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=max_steps,
            save_strategy="steps",
            save_steps=1,
            logging_steps=1,
            seed=3,
            report_to="none",
        )

    def test_resume_with_a_matching_manifest(self, teachers):
        dataset = _routed_dataset(["a"] * 8)
        trainer = DistillationTrainer(
            model=MODEL_ID, args=self._args(1), train_dataset=dataset, teacher_models={"a": teachers["a"]}
        )
        trainer.train()
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-1")
        assert os.path.exists(os.path.join(checkpoint, "teacher_manifest.json"))

        resumed = DistillationTrainer(
            model=MODEL_ID, args=self._args(2), train_dataset=dataset, teacher_models={"a": teachers["a"]}
        )
        resumed.train(resume_from_checkpoint=checkpoint)
        assert resumed.state.global_step == 2

    def test_resume_with_a_different_teacher_raises(self, teachers):
        dataset = _routed_dataset(["a"] * 8)
        trainer = DistillationTrainer(
            model=MODEL_ID, args=self._args(1), train_dataset=dataset, teacher_models={"a": teachers["a"]}
        )
        trainer.train()
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-1")

        # Same routing ID, different checkpoint content: the manifest's source key must catch it.
        swapped = DistillationTrainer(
            model=MODEL_ID, args=self._args(2), train_dataset=dataset, teacher_models={"a": teachers["b"]}
        )
        with pytest.raises(ValueError, match="incompatible with the checkpoint"):
            swapped.train(resume_from_checkpoint=checkpoint)


class TestManagedFailureCleanup(TrlTestCase):
    def test_callback_failure_releases_every_teacher_resource(self, teachers):
        class Boom(RuntimeError):
            pass

        class ExplodingCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                raise Boom("callback failure")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=2,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
            callbacks=[ExplodingCallback()],
        )
        with pytest.raises(Boom):
            trainer.train()

        assert trainer._teacher_store.live_keys == []
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        assert trainer._teacher_executor.stats.live_device_weight_bytes == 0
        # Nothing is retained, so closing is accepted.
        trainer.close_teachers()

    def test_a_failing_backward_clears_the_partial_gradients(self, teachers):
        class Boom(RuntimeError):
            pass

        class FailingLossTrainer(DistillationTrainer):
            def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
                # Run the real managed loss first, so the failure happens with targets consumed and a graph built.
                super()._compute_loss(unwrapped_student, inputs, num_items_in_batch)
                raise Boom("loss failure")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = FailingLossTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
        )
        with pytest.raises(Boom):
            trainer.train()
        assert all(param.grad is None for param in trainer.model.parameters())
        assert trainer._step == 0
        assert trainer._teacher_store.live_keys == []

    @pytest.mark.parametrize("fail_at", [1, 2])
    def test_a_scoring_failure_leaves_no_teacher_resources(self, teachers, fail_at):
        """A first-score failure (`fail_at=1`) and a partial-window failure (`fail_at=2`) must own nothing after."""

        class Boom(RuntimeError):
            pass

        real_score = TeacherExecutor.score
        calls = {"count": 0}

        def failing_score(executor, request, writer):
            calls["count"] += 1
            if calls["count"] == fail_at:
                raise Boom("scoring failure")
            return real_score(executor, request, writer)

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 8),
            teacher_models=teachers,
        )
        with patch.object(TeacherExecutor, "score", failing_score), pytest.raises(Boom):
            trainer.train()

        assert trainer._teacher_store.live_keys == []
        # The regression: a head retained for a group that then failed to score used to survive every cleanup path.
        assert trainer._teacher_executor._head_refcounts == {}
        assert trainer._teacher_executor._head_sources == {}
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        assert trainer._teacher_executor.stats.live_device_weight_bytes == 0
        assert all(param.grad is None for param in trainer.model.parameters())
        assert (trainer.state.global_step, trainer._step) == (0, 0)
        trainer.close_teachers()
        # Reopening from registry metadata and training again works.
        trainer.train()
        assert trainer.state.global_step == 1

    def test_a_backward_failure_after_an_accumulated_microbatch_clears_the_gradients(self, teachers):
        class Boom(RuntimeError):
            pass

        class _RaiseInBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, grad_output):
                raise Boom("backward failure")

        class FailingBackwardTrainer(DistillationTrainer):
            def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
                loss, entropy_sum, n_valid, teacher_stats = super()._compute_loss(
                    unwrapped_student, inputs, num_items_in_batch
                )
                if self._step == 1:  # the second microbatch of the accumulation window
                    loss = _RaiseInBackward.apply(loss)
                return loss, entropy_sum, n_valid, teacher_stats

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = FailingBackwardTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 8),
            teacher_models=teachers,
        )
        with pytest.raises(Boom):
            trainer.train()

        # The first microbatch's backward completed, so this really tests accumulated gradients being invalidated.
        assert trainer._step == 1
        assert all(param.grad is None for param in trainer.model.parameters())
        assert trainer.state.global_step == 0
        assert trainer._teacher_store.live_keys == []
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        trainer.close_teachers()

    def test_a_window_preparation_failure_propagates_its_own_error(self, teachers):
        """The trainer's cleanup scope calls `reset()`; a partially prepared window must not turn it into a KeyError."""

        def failing_fingerprint(microbatch, generation_id, microbatch_index):
            raise MemoryError("host copy failed")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 8),
            teacher_models=teachers,
        )
        with patch("trl.trainer._distillation_teacher._microbatch_fingerprint", failing_fingerprint):
            with pytest.raises(MemoryError):
                trainer.train()

        assert trainer._teacher_store.live_keys == []
        assert trainer._teacher_executor._head_refcounts == {}
        assert trainer._teacher_head_cache.stats.live_head_bytes == 0
        assert (trainer.state.global_step, trainer._step) == (0, 0)
        trainer.close_teachers()
        # The patch is scoped to the failing run, so a normal window still prepares, scores and steps.
        trainer.train()
        assert trainer.state.global_step == 1

    def test_compute_loss_without_prepared_targets_raises(self, teachers):
        training_args = DistillationConfig(
            output_dir=self.tmp_dir, per_device_train_batch_size=2, max_completion_length=4, report_to="none"
        )
        trainer = DistillationTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=_routed_dataset(["a", "b"] * 4),
            teacher_models=teachers,
        )
        trainer.model.train()
        batch = next(iter(trainer.get_train_dataloader()))
        inputs = trainer._prepare_inputs(batch)
        trainer._teacher_store.release(inputs["_teacher_targets_key"])
        del inputs["_teacher_targets_key"]
        with pytest.raises(RuntimeError, match="no teacher-target key"):
            trainer.compute_loss(trainer.model, inputs)
