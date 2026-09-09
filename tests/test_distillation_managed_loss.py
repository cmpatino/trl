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

"""Gate A tests: the managed checkpointed divergence loss and the one-slot device head cache.

Every test here runs on CPU. The properties CPU cannot observe (pinned staging tiles, the one-head device memory
bound) have `require_torch_accelerator` variants, which are skipped — not validated — without an accelerator.
"""

import contextlib
import gc
import threading
import time
import weakref
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl.trainer import _distillation_loss
from trl.trainer._distillation_heads import HeadIdentity, HeadSource, TargetGroup, TeacherHeadCache
from trl.trainer._distillation_loss import managed_chunked_divergence_loss
from trl.trainer.distillation_trainer import _chunked_divergence_loss

from .testing_utils import TrlTestCase, require_peft, require_torch_accelerator


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@dataclass
class _Spec:
    """One synthetic teacher: full `(B, K, H_t)` hidden states plus the head and the rows it owns.

    The full-grid `hidden`/`mask` form feeds the legacy loss; `group()` packs the same values into the managed
    loss's `(N, H_t)` CPU target group.
    """

    teacher_id: str
    hidden: torch.Tensor
    weight: torch.Tensor
    mask: torch.Tensor
    index: int = 0
    bias: torch.Tensor | None = None
    logit_scale: float = 1.0
    softcap: float | None = None

    @property
    def identity(self) -> HeadIdentity:
        return HeadIdentity(
            teacher_id=self.teacher_id,
            source_key=f"{self.teacher_id}@rev0",
            weight_shape=tuple(self.weight.shape),
            has_bias=self.bias is not None,
            source_dtype=self.weight.dtype,
            logit_scale=self.logit_scale,
            final_logit_softcapping=self.softcap,
        )

    @property
    def source(self) -> HeadSource:
        return HeadSource(identity=self.identity, weight=self.weight, bias=self.bias)

    def group(self) -> TargetGroup:
        positions = (self.mask.reshape(-1) != 0).nonzero(as_tuple=True)[0]
        flat = self.hidden.reshape(-1, self.hidden.size(-1))
        return TargetGroup(
            identity=self.identity,
            teacher_index=self.index,
            hidden=flat.index_select(0, positions).contiguous(),
            positions=positions,
        )


@dataclass
class _Case:
    """A student microbatch and the teachers scoring it, on both the managed and the legacy calling convention."""

    hidden: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    mask: torch.Tensor
    specs: list[_Spec] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return int(self.mask.sum().item())


def _case(B=2, K=6, H_s=8, V=17, n_masked=3, seed=0, bias=True, widths=(8,), scales=None, softcaps=None):
    """Build a student microbatch plus one teacher per entry of `widths`, splitting the valid rows between them."""
    g = torch.Generator().manual_seed(seed)
    student_hidden = torch.randn(B, K, H_s, generator=g)
    student_w = torch.randn(V, H_s, generator=g)
    student_b = torch.randn(V, generator=g) if bias else None
    mask = torch.ones(B, K)
    # Mask a few scattered positions so the managed loss must honour `positions` rather than the full grid.
    mask.reshape(-1)[torch.randperm(B * K, generator=g)[:n_masked]] = 0

    valid = (mask.reshape(-1) != 0).nonzero(as_tuple=True)[0]
    scales = scales or [1.0] * len(widths)
    softcaps = softcaps or [None] * len(widths)
    specs = []
    for index, (width, scale, softcap) in enumerate(zip(widths, scales, softcaps)):
        # Round-robin the valid rows over the teachers: disjoint groups whose union is the loss mask.
        owned = valid[index :: len(widths)]
        spec_mask = torch.zeros_like(mask).reshape(-1)
        spec_mask[owned] = 1
        specs.append(
            _Spec(
                teacher_id=f"teacher-{index}",
                hidden=torch.randn(B, K, width, generator=g),
                weight=torch.randn(V, width, generator=g),
                mask=spec_mask.reshape_as(mask),
                index=index,
                bias=torch.randn(V, generator=g) if bias else None,
                logit_scale=scale,
                softcap=softcap,
            )
        )
    return _Case(hidden=student_hidden, weight=student_w, bias=student_b, mask=mask, specs=specs)


def _cache(case, device="cpu", staging_bytes=64 << 20):
    cache = TeacherHeadCache(torch.device(device), staging_bytes=staging_bytes)
    for spec in case.specs:
        cache.retain_head_source(spec.source)
    return cache


def _managed(case, cache, *, beta, chunk_size, backward=True, evict_before_backward=False, **kwargs):
    """Run the managed loss on fresh grad-requiring copies of the student tensors; return outputs and gradients."""
    kwargs.setdefault("num_teachers", len(case.specs) or 1)
    hidden = case.hidden.clone().requires_grad_(True)
    weight = case.weight.clone().requires_grad_(True)
    bias = None if case.bias is None else case.bias.clone().requires_grad_(True)
    outputs = managed_chunked_divergence_loss(
        hidden,
        weight,
        bias,
        case.mask,
        [spec.group() for spec in case.specs],
        cache,
        beta,
        chunk_size,
        **kwargs,
    )
    if evict_before_backward:
        cache.evict_idle_gpu()
    if backward:
        outputs[0].backward()
    grads = (hidden.grad, weight.grad, None if bias is None else bias.grad)
    return outputs, grads


def _legacy(case, *, beta, chunk_size, denom, backward=True, **kwargs):
    """Sum `_chunked_divergence_loss` over each teacher's own rows, then apply the managed normalization once."""
    hidden = case.hidden.clone().requires_grad_(True)
    weight = case.weight.clone().requires_grad_(True)
    bias = None if case.bias is None else case.bias.clone().requires_grad_(True)
    total = hidden.new_zeros((), dtype=torch.float32)
    entropy = hidden.new_zeros((), dtype=torch.float32)
    for spec in case.specs:
        part, part_entropy, _ = _chunked_divergence_loss(
            hidden,
            spec.hidden,
            weight,
            spec.weight,
            spec.mask,
            beta,
            chunk_size,
            num_items_in_batch=1,
            student_lm_head_bias=bias,
            teacher_lm_head_bias=spec.bias,
            teacher_logit_scale=spec.logit_scale,
            teacher_final_logit_softcapping=spec.softcap,
            **kwargs,
        )
        total = total + part
        entropy = entropy + part_entropy
    loss = total / denom
    if backward:
        loss.backward()
    grads = (hidden.grad, weight.grad, None if bias is None else bias.grad)
    return (loss, entropy), grads


def _assert_parity(managed, legacy, atol=1e-6, rtol=1e-5):
    (managed_outputs, managed_grads), (legacy_outputs, legacy_grads) = managed, legacy
    torch.testing.assert_close(managed_outputs[0], legacy_outputs[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(managed_outputs[1], legacy_outputs[1], atol=atol, rtol=rtol)
    for managed_grad, legacy_grad in zip(managed_grads, legacy_grads):
        assert (managed_grad is None) == (legacy_grad is None)
        if managed_grad is not None:
            torch.testing.assert_close(managed_grad, legacy_grad, atol=atol, rtol=rtol)


class TestManagedLossParity(TrlTestCase):
    """The managed loss must reproduce `_chunked_divergence_loss` in value, gradient and optimizer step."""

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("chunk_size", [3, 4, 100])  # divides / doesn't divide / exceeds the row count
    def test_parity_single_teacher(self, beta, temperature, chunk_size):
        case = _case()
        cache = _cache(case)
        managed = _managed(case, cache, beta=beta, chunk_size=chunk_size, temperature=temperature)
        legacy = _legacy(case, beta=beta, chunk_size=chunk_size, denom=case.n_valid, temperature=temperature)
        _assert_parity(managed, legacy)
        assert int(managed[0][2].item()) == case.n_valid
        cache.close()

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_parity_without_bias(self, beta):
        case = _case(bias=False)
        cache = _cache(case)
        _assert_parity(
            _managed(case, cache, beta=beta, chunk_size=4),
            _legacy(case, beta=beta, chunk_size=4, denom=case.n_valid),
        )
        cache.close()

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_parity_logit_scale_and_softcapping(self, beta):
        # The teacher's scale/softcap come from its `HeadIdentity`, the student's from the call.
        case = _case(scales=[1.3], softcaps=[30.0])
        cache = _cache(case)
        kwargs = {"student_logit_scale": 0.7, "student_final_logit_softcapping": 50.0}
        _assert_parity(
            _managed(case, cache, beta=beta, chunk_size=4, **kwargs),
            _legacy(case, beta=beta, chunk_size=4, denom=case.n_valid, **kwargs),
        )
        cache.close()

    def test_parity_heterogeneous_teacher_width(self):
        # Only the vocabulary is shared; the teacher may be wider or narrower than the student.
        for width in (5, 12):
            case = _case(widths=(width,))
            cache = _cache(case)
            _assert_parity(
                _managed(case, cache, beta=0.5, chunk_size=4),
                _legacy(case, beta=0.5, chunk_size=4, denom=case.n_valid),
            )
            cache.close()

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("chunk_size", [2, 3, 100])
    def test_parity_several_teachers_in_one_microbatch(self, beta, chunk_size):
        # Three disjoint teacher groups with different widths, scales and softcaps inside one student microbatch.
        case = _case(K=8, widths=(8, 5, 12), scales=(1.0, 1.3, 0.6), softcaps=(None, 30.0, 50.0))
        cache = _cache(case)
        _assert_parity(
            _managed(case, cache, beta=beta, chunk_size=chunk_size, temperature=2.0),
            _legacy(case, beta=beta, chunk_size=chunk_size, denom=case.n_valid, temperature=2.0),
        )
        cache.close()

    def test_parity_num_items_in_batch(self):
        # `num_items_in_batch` replaces the local valid-token denominator (gradient-accumulation-correct reduction).
        case = _case()
        cache = _cache(case)
        for denom in (7, torch.tensor(7.0)):
            _assert_parity(
                _managed(case, cache, beta=0.5, chunk_size=4, num_items_in_batch=denom),
                _legacy(case, beta=0.5, chunk_size=4, denom=7),
            )
        cache.close()

    def test_parity_tied_embedding_and_head(self):
        # One parameter used both to embed and to project: it must collect the sum of both gradients on either path.
        case = _case(V=17, H_s=8)
        cache = _cache(case)
        ids = torch.arange(case.mask.numel()).reshape_as(case.mask) % case.weight.size(0)

        grads = []
        for run_managed in (True, False):
            tied = case.weight.clone().requires_grad_(True)
            hidden = F.embedding(ids, tied)
            if run_managed:
                loss = managed_chunked_divergence_loss(
                    hidden, tied, None, case.mask, [s.group() for s in case.specs], cache, 0.5, 4
                )[0]
            else:
                loss = _chunked_divergence_loss(
                    hidden,
                    case.specs[0].hidden,
                    tied,
                    case.specs[0].weight,
                    case.mask,
                    0.5,
                    4,
                    teacher_lm_head_bias=case.specs[0].bias,
                )[0]
            loss.backward()
            grads.append(tied.grad)
        torch.testing.assert_close(grads[0], grads[1], atol=1e-6, rtol=1e-5)
        cache.close()

    def test_masked_positions_are_ignored(self):
        # Perturbing the student's masked rows must not move the loss: they belong to no teacher group.
        case = _case()
        cache = _cache(case)
        loss_a = _managed(case, cache, beta=0.5, chunk_size=4, backward=False)[0][0]
        masked = (case.mask.reshape(-1) == 0).nonzero(as_tuple=True)[0]
        perturbed = case.hidden.clone().reshape(-1, case.hidden.size(-1))
        perturbed[masked] += 5.0
        case.hidden = perturbed.reshape_as(case.hidden)
        loss_b = _managed(case, cache, beta=0.5, chunk_size=4, backward=False)[0][0]
        torch.testing.assert_close(loss_a, loss_b)
        cache.close()

    def test_masked_positions_receive_no_gradient(self):
        case = _case()
        cache = _cache(case)
        grads = _managed(case, cache, beta=0.5, chunk_size=4)[1]
        grad = grads[0].reshape(-1, case.hidden.size(-1))
        valid = case.mask.reshape(-1) != 0
        assert (grad[valid].abs().sum(dim=-1) > 0).all()
        assert torch.equal(grad[~valid], torch.zeros_like(grad[~valid]))
        cache.close()

    @pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.AdamW])
    def test_one_optimizer_step_matches_legacy(self, optimizer_class):
        # Value and gradient parity must survive into the update the trainer actually applies.
        case = _case(K=8, widths=(8, 5))
        cache = _cache(case)
        inputs = torch.randn(case.hidden.size(0), case.hidden.size(1), 6, generator=torch.Generator().manual_seed(9))

        def build_student():
            torch.manual_seed(3)
            return nn.Sequential(nn.Linear(6, case.hidden.size(-1)), nn.Linear(case.hidden.size(-1), 17))

        updated = []
        for run_managed in (True, False):
            student = build_student()
            backbone, head = student[0], student[1]
            optimizer = optimizer_class(student.parameters(), lr=0.1)
            hidden = backbone(inputs)
            if run_managed:
                loss = managed_chunked_divergence_loss(
                    hidden,
                    head.weight,
                    head.bias,
                    case.mask,
                    [s.group() for s in case.specs],
                    cache,
                    0.5,
                    4,
                    num_teachers=len(case.specs),
                )[0]
            else:
                loss = sum(
                    _chunked_divergence_loss(
                        hidden,
                        spec.hidden,
                        head.weight,
                        spec.weight,
                        spec.mask,
                        0.5,
                        4,
                        num_items_in_batch=1,
                        student_lm_head_bias=head.bias,
                        teacher_lm_head_bias=spec.bias,
                    )[0]
                    for spec in case.specs
                ) / case.n_valid
            loss.backward()
            optimizer.step()
            updated.append([p.detach().clone() for p in student.parameters()])
        for managed_param, legacy_param in zip(*updated):
            torch.testing.assert_close(managed_param, legacy_param, atol=1e-6, rtol=1e-5)
        cache.close()


class _ChunkFailure(Exception):
    """Application exception raised from inside the checkpointed chunk body."""


class _FailingFunctional:
    """`torch.nn.functional` stand-in that raises inside the chunk body after `after` `log_softmax` calls."""

    def __init__(self, after):
        self.after = after
        self.calls = 0

    def __getattr__(self, name):
        return getattr(F, name)

    def log_softmax(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self.after:
            raise _ChunkFailure("chunk body failed")
        return F.log_softmax(*args, **kwargs)


def _halves(group):
    """Split a target group in two so a group list can alternate between two heads chunk by chunk."""
    half = group.hidden.size(0) // 2
    return [
        TargetGroup(group.identity, group.teacher_index, group.hidden[:half].contiguous(), group.positions[:half]),
        TargetGroup(group.identity, group.teacher_index, group.hidden[half:].contiguous(), group.positions[half:]),
    ]


class TestManagedLossPrecision(TrlTestCase):
    """The cached head must sit in the dtype the matmul executes in, without changing the baseline's rounding."""

    @pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("autocast_dtype", [None, torch.bfloat16])
    def test_cached_head_dtype_is_the_execution_dtype(self, hidden_dtype, autocast_dtype):
        # The head source is float32 throughout; only the targets' dtype and the autocast context vary.
        case = _case()
        case.specs[0].hidden = case.specs[0].hidden.to(hidden_dtype)
        cache = _cache(case)
        context = contextlib.nullcontext() if autocast_dtype is None else torch.autocast("cpu", dtype=autocast_dtype)
        with context:
            _managed(case, cache, beta=0.5, chunk_size=4, backward=False)
        assert cache._weight.dtype == (hidden_dtype if autocast_dtype is None else autocast_dtype)
        # The bias is never rounded through the execution dtype.
        assert cache._bias.dtype == torch.float32
        cache.close()

    @pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_parity_under_bf16_autocast(self, hidden_dtype):
        # A float32 head source under bfloat16 autocast, with a float32 bias whose values bfloat16 cannot hold.
        # One teacher whose row count is a multiple of the chunk size, so the chunk boundaries match the legacy
        # loss's and parity must be bit-exact rather than merely close.
        case = _case(B=2, K=8, n_masked=0)
        spec = case.specs[0]
        spec.hidden = spec.hidden.to(hidden_dtype)
        spec.bias = torch.full_like(spec.bias, 1.0 + 2.0**-10)
        assert not torch.equal(spec.bias.to(torch.bfloat16).float(), spec.bias)
        cache = _cache(case)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _assert_parity(
                _managed(case, cache, beta=0.5, chunk_size=4),
                _legacy(case, beta=0.5, chunk_size=4, denom=case.n_valid),
                atol=0,
                rtol=0,
            )
        cache.close()

    def test_parity_under_bf16_autocast_with_several_teachers(self):
        case = _case(B=2, K=8, n_masked=2, widths=(8, 5, 12), scales=(1.0, 1.3, 0.6), softcaps=(None, 30.0, 50.0))
        for spec in case.specs:
            spec.hidden = spec.hidden.to(torch.bfloat16)
        cache = _cache(case)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _assert_parity(
                _managed(case, cache, beta=0.5, chunk_size=4, temperature=2.0),
                _legacy(case, beta=0.5, chunk_size=4, denom=case.n_valid, temperature=2.0),
                atol=1e-5,
                rtol=1e-4,
            )
        cache.close()

    def test_bias_keeps_its_source_dtype_and_values(self):
        case = _case()
        spec = case.specs[0]
        spec.bias = torch.full_like(spec.bias, 1.0 + 2.0**-10)  # not representable in bfloat16
        cache = _cache(case)
        with cache.projection_lease(spec.identity, torch.bfloat16, torch.float32) as head:
            assert head.weight.dtype == torch.bfloat16
            assert head.bias.dtype == torch.float32
            assert torch.equal(head.bias, spec.bias)
        cache.close()

    def test_operand_dtype_is_part_of_the_cache_key(self):
        # float32 -> float16 -> bfloat16 is not float32 -> bfloat16, so the two must not share a cache entry.
        case = _case()
        spec = case.specs[0]
        cache = _cache(case)
        with cache.projection_lease(spec.identity, torch.bfloat16, torch.float32) as head:
            direct = head.weight.clone()
        with cache.projection_lease(spec.identity, torch.bfloat16, torch.float16) as head:
            two_stage = head.weight.clone()
        assert cache.stats.misses == 2 and cache.stats.hits == 0
        torch.testing.assert_close(direct, spec.weight.to(torch.bfloat16), atol=0, rtol=0)
        torch.testing.assert_close(two_stage, spec.weight.to(torch.float16).to(torch.bfloat16), atol=0, rtol=0)
        assert not torch.equal(direct, two_stage)
        cache.close()
