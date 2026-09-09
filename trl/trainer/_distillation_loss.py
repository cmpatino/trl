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

"""Managed multi-teacher divergence loss: checkpointed chunks that lease a device teacher head per invocation.

The legacy `_chunked_divergence_loss` in `distillation_trainer.py` passes device teacher tensors into the
checkpointed body, which keeps teacher head storage alive across the backward pass and makes eviction unsafe. The
managed loss here passes CPU targets plus an immutable [`HeadIdentity`] instead, and acquires/releases the device head
inside every checkpoint invocation, including replay. See
`/data/workspaces/mopd/MOPD-implementation-design.md` for the design.
"""

import torch
import torch.nn.functional as F

from ._distillation_heads import TargetGroup, TeacherHeadCache
from .utils import maybe_gather_lm_head_ctx


def _execution_dtype(hidden_dtype: torch.dtype, device: torch.device) -> torch.dtype:
    """
    Dtype the teacher projection's matmul really executes in, so the cached head is materialized in exactly it.

    The baseline matches the head weight to the hidden states' dtype and lets autocast take it from there. Under
    autocast the matmul therefore runs in the autocast dtype, and a head cached in the hidden states' dtype would be
    converted implicitly on every matmul — a second full head next to the cached one, defeating the one-head bound.
    Casting the source straight to the value returned here reproduces the baseline's rounding: the intermediate
    `source -> hidden dtype` step only ever widens (backbone hidden states are either the autocast dtype itself or
    float32), so it never rounds away bits that the following cast would have kept.

    Args:
        hidden_dtype (`torch.dtype`):
            Dtype of the teacher's cached hidden targets.
        device (`torch.device`):
            Device the projection runs on; autocast is enabled per device type.

    Returns:
        `torch.dtype`: the autocast dtype when autocast is enabled for this device type, else `hidden_dtype`.
    """
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return hidden_dtype


def _managed_chunk(h_s, w_s, b_s, s_scale, s_softcap, h_t_cpu, identity, head_cache, beta, temperature, valid):
    # Same body as the legacy `_chunk`, except the teacher arrives as a CPU slice plus an identity: the device head is
    # leased here so neither the checkpoint's arguments nor its graph retain it, and eviction between forward and
    # backward only costs a re-upload during replay.
    with maybe_gather_lm_head_ctx(w_s, b_s):
        # Project in the compute dtype and upcast only for the softmax, like `"nll"` and `transformers`' own
        # `ForCausalLMLoss` do. Matching the weight to the hidden-states dtype keeps the matmul in that dtype, which
        # is what `lm_head`'s own forward computes in.
        student_logits = (h_s @ w_s.to(h_s.dtype).t()).float()
        if b_s is not None:
            student_logits = student_logits + b_s.float()
    if s_scale != 1.0:
        student_logits = student_logits * s_scale
    if s_softcap is not None:
        student_logits = s_softcap * torch.tanh(student_logits / s_softcap)

    # The teacher is a fixed target: `no_grad` (never inference mode, whose tensors cannot be saved by the student
    # loss) so the projection builds no autograd graph and the teacher head accumulates no gradients.
    with torch.no_grad():
        device = head_cache.device
        # The lease materializes the head in the dtype the matmul executes in, so the projection adds no cast of its
        # own; every use of the leased tensors finishes inside the block, which invalidates them on exit.
        with head_cache.projection_lease(identity, _execution_dtype(h_t_cpu.dtype, device)) as head:
            h_t = h_t_cpu.to(device)
            teacher_logits = (h_t @ head.weight.t()).float()
            if head.bias is not None:
                # The bias is in the source dtype: adding it after the upcast keeps an fp32 bias exact under a bf16
                # execution dtype, matching the legacy `b.float()`.
                teacher_logits = teacher_logits + head.bias.float()
        if identity.logit_scale != 1.0:
            teacher_logits = teacher_logits * identity.logit_scale
        if identity.final_logit_softcapping is not None:
            softcap = identity.final_logit_softcapping
            teacher_logits = softcap * torch.tanh(teacher_logits / softcap)

    # Distillation (softmax) temperature: soften both distributions before the divergence, applied after any
    # per-model scaling/softcapping (matching the model's full forward, then the loss's temperature).
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # beta: 0 = forward KL, 1 = reverse KL, else generalized JSD. `F.kl_div(input, target)` computes
    # `target * (log target - input)`, hence the swapped argument order relative to the KL written in the paper.
    if beta == 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    per_token_jsd = jsd.sum(dim=-1) * valid
    per_token_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1) * valid
    # Read off the teacher log-probs that were computed anyway, so replay adds no work and the metric is produced by
    # the forward pass's return value only.
    per_token_teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1) * valid
    return per_token_jsd.sum(), per_token_entropy.sum(), per_token_teacher_entropy.sum()


def _student_zero_chunk(h_s, w_s, b_s, s_scale, s_softcap):
    # A nonempty all-masked microbatch has no teacher targets at all, but its zero loss still has to reach every
    # trainable student parameter, or DDP/FSDP synchronization hangs at the all-reduce.
    with maybe_gather_lm_head_ctx(w_s, b_s):
        student_logits = (h_s @ w_s.to(h_s.dtype).t()).float()
        if b_s is not None:
            student_logits = student_logits + b_s.float()
    if s_scale != 1.0:
        student_logits = student_logits * s_scale
    if s_softcap is not None:
        student_logits = s_softcap * torch.tanh(student_logits / s_softcap)
    return F.log_softmax(student_logits, dim=-1).sum() * 0.0


def managed_chunked_divergence_loss(
    student_hidden_states: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    student_lm_head_bias: torch.Tensor | None,
    loss_mask: torch.Tensor,
    target_groups: list[TargetGroup],
    head_cache: TeacherHeadCache,
    beta: float,
    chunk_size: int,
    temperature: float = 1.0,
    num_items_in_batch: torch.Tensor | int | None = None,
    student_logit_scale: float = 1.0,
    student_final_logit_softcapping: float | None = None,
    num_teachers: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient generalized JSD of one student microbatch against per-teacher CPU hidden targets.

    Equivalent to `_chunked_divergence_loss` summed over the teachers present in the microbatch, but the teacher side
    enters the checkpointed chunk as a CPU slice plus an immutable [`HeadIdentity`]. Each invocation — including
    checkpoint replay during the backward pass — leases the device head from `head_cache`, copies its target chunk
    over, projects under `torch.no_grad()` and releases the lease, so no device teacher tensor is captured by the
    checkpoint and the head may be evicted between forward and backward. Only the student computation participates in
    autograd, and the full `(chunk, V)` logits live only inside their own chunk.

    Each group's rows are all valid, so chunking runs over the group's own positions; the divergences of all groups
    are summed before the single normalization, exactly as the legacy loss normalizes one teacher's sum.

    Args:
        student_hidden_states (`torch.Tensor`):
            Student backbone output of shape `(B, K, H_s)`, aligned to the completion tokens (before the `lm_head`).
        student_lm_head_weight (`torch.Tensor`):
            Student `lm_head` weight of shape `(V, H_s)`. Under FSDP2 the caller passes the full (gathered) view.
        student_lm_head_bias (`torch.Tensor`, *optional*):
            Student `lm_head` bias of shape `(V,)`, added to each chunk's logits when provided.
        loss_mask (`torch.Tensor`):
            Binary mask of shape `(B, K)`; `1` marks completion positions included in the loss. Its count is the
            returned `n_valid` and the fallback denominator.
        target_groups (`list[`[`TargetGroup`]`]`):
            One group per teacher present in this microbatch, holding that teacher's CPU hidden targets and their
            positions in the flattened `(B * K)` completion layout. Groups are disjoint; an empty list is a fully
            masked microbatch.
        head_cache ([`TeacherHeadCache`]):
            Cache the chunk bodies lease their device head from. Referencing it from a checkpoint is safe; the device
            head itself is never captured.
        beta (`float`):
            Interpolation coefficient. `0.0` = forward KL, `1.0` = reverse KL, else generalized JSD.
        chunk_size (`int`):
            Number of valid positions processed per chunk. Peak memory scales linearly with this.
        temperature (`float`, *optional*, defaults to `1.0`):
            Softmax temperature applied to both distributions before the divergence, after any scale/softcapping.
        num_items_in_batch (`torch.Tensor`, `int` or `None`, *optional*):
            Total number of valid tokens across the global batch. When provided, the loss is reduced as `sum /
            num_items_in_batch` (gradient-accumulation-correct); when `None`, reduction is `mean` over local valid
            positions.
        student_logit_scale (`float`, *optional*, defaults to `1.0`):
            Multiplier applied to the student's logits before the softmax (Cohere-style `logit_scale`). The teacher's
            scale and softcapping come from its [`HeadIdentity`].
        student_final_logit_softcapping (`float`, *optional*):
            If set, applies `softcap * tanh(logits / softcap)` to the student's logits (Gemma-style), after the scale.
        num_teachers (`int`, *optional*, defaults to `1`):
            Width of the returned per-teacher statistics, i.e. the number of registered teachers.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`: scalar loss, sum of per-token student entropy
        (in nats), number of valid completion positions, and a detached float32 `(3, num_teachers)` tensor whose rows
        are the per-teacher divergence sum, teacher entropy sum and valid token count (zero columns for teachers
        absent from this microbatch). Raw sums are returned so callers can reduce correctly across ranks.

    Raises:
        `ValueError`: if the microbatch has no rows; scheduling a zero-row microbatch is a caller error.
    """
    if student_hidden_states.size(0) == 0:
        raise ValueError("`student_hidden_states` has no rows: a zero-row microbatch must not be scheduled")

    h_s = student_hidden_states.reshape(-1, student_hidden_states.size(-1))
    device = h_s.device
    n_valid = (loss_mask.reshape(-1) != 0).sum()
    loss = h_s.new_zeros((), dtype=torch.float32)
    entropy_sum = h_s.new_zeros((), dtype=torch.float32)
    teacher_stats = torch.zeros(3, num_teachers, dtype=torch.float32, device=device)

    if sum(group.hidden.size(0) for group in target_groups) == 0:
        loss = loss + torch.utils.checkpoint.checkpoint(
            _student_zero_chunk,
            h_s[:chunk_size],
            student_lm_head_weight,
            student_lm_head_bias,
            student_logit_scale,
            student_final_logit_softcapping,
            use_reentrant=False,
        )

    for group in target_groups:
        for start in range(0, group.hidden.size(0), chunk_size):
            positions = group.positions[start : start + chunk_size]
            hidden = group.hidden[start : start + chunk_size]
            chunk_jsd, chunk_entropy, chunk_teacher_entropy = torch.utils.checkpoint.checkpoint(
                _managed_chunk,
                h_s.index_select(0, positions.to(device)),
                student_lm_head_weight,
                student_lm_head_bias,
                student_logit_scale,
                student_final_logit_softcapping,
                hidden,
                group.identity,
                head_cache,
                beta,
                temperature,
                torch.ones(hidden.size(0), dtype=torch.float32, device=device),
                use_reentrant=False,
            )
            loss = loss + chunk_jsd
            entropy_sum = entropy_sum + chunk_entropy
            # Accumulated from the forward pass's return values only, so replay leaves the metrics untouched.
            teacher_stats[0, group.teacher_index] += chunk_jsd.detach()
            teacher_stats[1, group.teacher_index] += chunk_teacher_entropy
            teacher_stats[2, group.teacher_index] += hidden.size(0)

    if num_items_in_batch is None:
        # Clamped so a fully-masked rank reduces to a finite zero rather than `0 / 0`.
        loss = loss / n_valid.clamp(min=1)
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, entropy_sum, n_valid, teacher_stats
