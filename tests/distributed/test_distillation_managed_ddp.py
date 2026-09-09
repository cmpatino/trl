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
Two-rank CPU check of the managed multi-teacher update: `torchrun --standalone --nproc_per_node=2` with gloo.

What is compared against a single-process reference that trains on the same global batch and the same tokens:

* exactly, because they are integer/counting quantities the reduction must preserve — the number of optimizer steps,
  the global valid-token count per step, and the per-teacher *training* token fractions the `[3, num_teachers]`
  statistics reduce to;
* within a tolerance, because two ranks sum a step's gradients in a different order and over different microbatch
  shapes than one process does — the final parameters (`atol=1e-7`, `rtol=1e-5`; the observed maximum absolute
  difference is 3.7e-9), with the worst parameter reported in the failure message. The worker trains with plain SGD
  so the update is proportional to the gradient; AdamW would normalize by a near-zero second moment here and turn a
  last-bit gradient difference into a large parameter difference.

The last *evaluation* batch is deliberately partial (three evaluation rows over two ranks), which is what exercises
the fixed-shape `accelerator.reduce` of the teacher statistics. Accelerate pads that batch by repeating a row, and a
fixed-shape sum-reduce counts the repeat (unlike `gather_for_metrics`, which cannot be used here because it would
trim the statistics tensor's first dimension, the three statistic rows, as if they were examples). The evaluation
token fractions are therefore expected to differ from the single-process run — 0.75/0.25 versus 0.667/0.333 — and
this test asserts only that the reduction stayed coherent, never that the two agree.

This is normalization and collective-order evidence. It is not memory evidence: there is no accelerator here.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distillation_managed_ddp_script.py")
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# `torchrun`'s own entry point, addressed through the running interpreter so the test does not depend on the console
# script being on `PATH`.
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
PROBE = """
import torch.distributed as dist

dist.init_process_group("gloo")
assert dist.get_world_size() == 2
assert dist.get_backend() == "gloo"
dist.destroy_process_group()
"""


def _torchrun_available() -> bool:
    """Whether `torchrun` can start two gloo processes here, which is what makes this test meaningful."""
    with tempfile.TemporaryDirectory() as directory:
        probe = os.path.join(directory, "probe.py")
        with open(probe, "w") as handle:
            handle.write(PROBE)
        result = subprocess.run(
            [*TORCHRUN, probe],
            capture_output=True,
            text=True,
            timeout=300,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            cwd=REPOSITORY_ROOT,
        )
    return result.returncode == 0


def _teacher_checkpoint(path: str, scale: float | None = None) -> str:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    if scale is not None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale)
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(path)
    return path


@pytest.mark.slow
class TestManagedDistillationTwoRankCpu(TrlTestCase):
    def test_two_rank_update_matches_a_single_process_reference(self):
        # Probed here rather than in a `skipif`, which would spawn two processes during *collection* of any run that
        # touches this file. A skip is a skip: it is never reported as a pass.
        if not _torchrun_available():
            pytest.skip("torchrun cannot start two gloo processes here")
        teacher_a = _teacher_checkpoint(os.path.join(self.tmp_dir, "teacher-a"))
        teacher_b = _teacher_checkpoint(os.path.join(self.tmp_dir, "teacher-b"), scale=1.05)
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONPATH=REPOSITORY_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
            TOKENIZERS_PARALLELISM="false",
        )

        def run(command, mode):
            output = os.path.join(self.tmp_dir, f"{mode}.json")
            arguments = [
                *command,
                SCRIPT,
                "--mode",
                mode,
                "--teacher-a",
                teacher_a,
                "--teacher-b",
                teacher_b,
                "--out",
                output,
                "--output-dir",
                os.path.join(self.tmp_dir, f"out-{mode}"),
            ]
            result = subprocess.run(
                arguments, capture_output=True, text=True, timeout=1800, env=environment, cwd=REPOSITORY_ROOT
            )
            assert result.returncode == 0, f"{mode} run failed:\n{result.stdout[-4000:]}\n{result.stderr[-8000:]}"
            with open(output) as handle:
                summary = json.load(handle)
            parameters = torch.load(os.path.splitext(output)[0] + "-params.pt", weights_only=True)
            return summary, parameters

        ddp, ddp_parameters = run(TORCHRUN, "ddp")
        reference, reference_parameters = run([sys.executable], "reference")

        # The worker asserts this itself; assert it again on the evidence so a single-process fallback can never be
        # read as a two-rank result.
        assert ddp["world_size"] == 2
        assert reference["world_size"] == 1
        # Disjoint teacher sets per rank: rank 0 routed only to teacher 0, rank 1 only to teacher 1.
        assert ddp["routed_teacher_indices_per_rank"] == [[0], [1]], ddp["routed_teacher_indices_per_rank"]
        assert reference["routed_teacher_indices_per_rank"] == [[0, 1]]

        # Exact: counting quantities the cross-rank reduction must preserve.
        assert ddp["optimizer_steps"] == reference["optimizer_steps"] == 2
        assert ddp["num_tokens"] == reference["num_tokens"]
        assert ddp["teacher_token_frac"] == reference["teacher_token_frac"]

        # The partial evaluation batch: the reduction must stay coherent (one fixed-shape tensor, fractions summing
        # to one), but it counts Accelerate's repeated padding row, so it is not compared to the reference. See the
        # module docstring.
        for summary in (ddp, reference):
            assert sum(summary["eval_teacher_token_frac"].values()) == pytest.approx(1.0)
            assert not torch.isnan(torch.tensor(summary["eval_loss"]))
        assert sorted(ddp["eval_teacher_token_frac"]) == sorted(reference["eval_teacher_token_frac"])

        # Nothing retained on either side.
        assert ddp["live_target_keys"] == [] and reference["live_target_keys"] == []
        assert ddp["live_head_bytes"] == 0 and reference["live_head_bytes"] == 0
        assert ddp["live_device_weight_bytes"] == 0 and reference["live_device_weight_bytes"] == 0

        # Within tolerance: the same global update summed in a different order.
        assert sorted(ddp_parameters) == sorted(reference_parameters)
        differences = {
            name: (ddp_parameters[name] - reference_parameters[name]).abs().max().item()
            for name in reference_parameters
        }
        worst = max(differences, key=differences.get)
        for name, reference_parameter in reference_parameters.items():
            assert torch.allclose(ddp_parameters[name], reference_parameter, atol=1e-7, rtol=1e-5), (
                f"two-rank update differs from the single-process reference: parameter {name} by "
                f"{differences[name]:.3e} (worst is {worst} by {differences[worst]:.3e})"
            )
