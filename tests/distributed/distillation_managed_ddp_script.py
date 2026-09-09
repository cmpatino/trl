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

"""
Two-rank CPU worker for the managed multi-teacher distillation collective/normalization check.

Launched by `tests/distributed/test_distillation_managed_ddp.py`, twice: once under
`torchrun --standalone --nproc_per_node=2` (gloo, `--mode ddp`) and once as a plain process with the same *global*
batch (`--mode reference`). Completions are replaced by a deterministic function of the prompt tokens, so both runs
train on exactly the same (prompt, completion) pairs and the comparison isolates the distributed reduction from
sampling. Rank 0 writes a JSON summary and the final parameters.

This is normalization and collective-order evidence only. It says nothing about device memory: there is no
accelerator here.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl import DistillationConfig, DistillationTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
# Routed so that rank 0 only ever sees teacher "a" and rank 1 only teacher "b": with `shuffle_dataset=False` the
# repeat sampler yields index chunks of the global generation batch, and Accelerate hands each rank a contiguous
# slice of one chunk.
TRAIN_TEACHER_IDS = ["a", "a", "b", "b", "a", "a", "b", "b"]
# Three rows over two ranks with one row per rank per batch: the last evaluation batch is partial.
EVAL_TEACHER_IDS = ["a", "b", "a"]
PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The largest planet is",
    "Photosynthesis happens in",
    "The speed of light is",
    "Mount Everest is in",
    "The Pacific Ocean is",
    "An atom contains",
]


class FixedCompletionTrainer(DistillationTrainer):
    """Replaces sampled completions with a deterministic function of the prompt, so the two runs see equal tokens."""

    def _generate(self, prompts):
        prompt_ids, completion_ids, tool_mask, images, tool_images = super()._generate(prompts)
        fixed = []
        for ids in prompt_ids:
            offset = sum(int(token) for token in ids) % 997
            fixed.append([(offset + position * 7) % 100 + 1 for position in range(self.max_completion_length)])
        return prompt_ids, fixed, tool_mask, images, tool_images


def build_dataset(teacher_ids):
    return Dataset.from_dict(
        {"prompt": [PROMPTS[index % len(PROMPTS)] for index in range(len(teacher_ids))], "teacher_id": teacher_ids}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ddp", "reference"], required=True)
    parser.add_argument("--teacher-a", required=True)
    parser.add_argument("--teacher-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Same *global* batch in both modes: 4 rows per optimizer step, split 2 ranks x 1 row x 2 accumulation steps
    # under DDP and 1 process x 2 rows x 2 accumulation steps in the reference.
    per_device_train_batch_size = 1 if args.mode == "ddp" else 2
    training_args = DistillationConfig(
        output_dir=args.output_dir,
        learning_rate=0.01,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        max_completion_length=4,
        max_steps=2,
        logging_steps=1,
        eval_strategy="no",
        shuffle_dataset=False,
        seed=11,
        report_to="none",
        use_cpu=True,
        # One teacher forward per microbatch keeps the scoring shapes stable across the two modes.
        teacher_scoring_batch_size=per_device_train_batch_size,
    )
    trainer = FixedCompletionTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=build_dataset(TRAIN_TEACHER_IDS),
        eval_dataset=build_dataset(EVAL_TEACHER_IDS),
        teacher_models={"a": args.teacher_a, "b": args.teacher_b},
        teacher_tokenizers=None,
    )
    accelerator = trainer.accelerator
    world_size = accelerator.num_processes
    if args.mode == "ddp":
        # The whole point of this worker: a real two-rank run. Never let a single-process fallback be recorded as
        # distributed evidence.
        assert world_size == 2, f"expected world_size == 2 under torchrun, got {world_size}"
        assert torch.distributed.is_initialized(), "torch.distributed was not initialized"
        assert torch.distributed.get_backend() == "gloo", f"expected gloo, got {torch.distributed.get_backend()}"
        assert str(accelerator.distributed_type) == "MULTI_CPU", f"got {accelerator.distributed_type}"
    else:
        assert world_size == 1, f"the reference must be single-process, got {world_size}"

    # Record which teachers this rank actually routed to, to prove the ranks used disjoint teacher sets.
    routed = set()
    generate_and_score = trainer._generate_and_score_completions

    def recording_generate_and_score(inputs):
        output = generate_and_score(inputs)
        routed.update(int(index) for index in output["teacher_index"])
        return output

    trainer._generate_and_score_completions = recording_generate_and_score

    trainer.train()
    evaluation = trainer.evaluate()

    step_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    summary = {
        "mode": args.mode,
        "world_size": world_size,
        "process_index": accelerator.process_index,
        "per_device_train_batch_size": per_device_train_batch_size,
        "optimizer_steps": trainer.state.global_step,
        "teacher_ids": trainer._teacher_registry.teacher_ids,
        "routed_teacher_indices": sorted(routed),
        "train_losses": [entry["loss"] for entry in step_logs],
        "num_tokens": [entry["num_tokens"] for entry in step_logs],
        "teacher_token_frac": [
            {key: entry[key] for key in sorted(entry) if key.startswith("teacher_token_frac/")} for entry in step_logs
        ],
        "teacher_jsd": [
            {key: entry[key] for key in sorted(entry) if key.startswith("teacher_jsd/")} for entry in step_logs
        ],
        "eval_loss": evaluation["eval_loss"],
        "eval_teacher_token_frac": {
            key: value for key, value in sorted(evaluation.items()) if "teacher_token_frac" in key
        },
        "live_target_keys": trainer._teacher_store.live_keys,
        "live_head_bytes": trainer._teacher_head_cache.stats.live_head_bytes,
        "live_device_weight_bytes": trainer._teacher_executor.stats.live_device_weight_bytes,
        "head_cache_peak_head_bytes": trainer._teacher_head_cache.stats.peak_head_bytes,
        "executor_body_loads": trainer._teacher_executor.stats.body_loads,
        "executor_cpu_reloads": trainer._teacher_executor.stats.cpu_reloads,
    }

    # Every rank's routing set is needed for the disjointness check, and the collective doubles as a liveness check.
    if world_size > 1:
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, sorted(routed))
        summary["routed_teacher_indices_per_rank"] = gathered
    else:
        summary["routed_teacher_indices_per_rank"] = [sorted(routed)]

    trainer.close_teachers()

    if accelerator.is_main_process:
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        parameters = {name: param.detach().cpu() for name, param in trainer.model.named_parameters()}
        torch.save(parameters, os.path.splitext(args.out)[0] + "-params.pt")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    # `AutoTokenizer` is imported so the worker fails fast if the tiny model is not cached, rather than inside the
    # trainer where the traceback is harder to read from a torchrun log.
    AutoTokenizer.from_pretrained(MODEL_ID)
    main()
