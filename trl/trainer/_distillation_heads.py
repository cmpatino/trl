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

"""Teacher head records and the one-slot device head cache used by the managed multi-teacher distillation loss.

Shared seam between the teacher registry/executor (`_distillation_teacher.py`), the managed loss
(`_distillation_loss.py`), and the manifest (`_distillation_identity.py`). See
`/data/workspaces/mopd/implementation/interfaces.md` for the contract.
"""

import threading
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeadIdentity:
    """Immutable identity of one teacher output head, including its logit transformations."""

    teacher_id: str
    source_key: str
    weight_shape: tuple[int, int]
    has_bias: bool
    source_dtype: torch.dtype
    logit_scale: float
    final_logit_softcapping: float | None
    transform_version: int = 1


@dataclass
class HeadSource:
    """Exact CPU head tensors retained while any target projected through this head is live."""

    identity: HeadIdentity
    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass
class TargetGroup:
    """One teacher's cached hidden targets inside one student microbatch.

    `hidden` holds only valid completion positions, `(N, H_teacher)` on CPU; `positions` maps each row onto the
    student's flattened `(B * K)` completion layout (`b * K + k`).
    """

    identity: HeadIdentity
    teacher_index: int
    hidden: torch.Tensor
    positions: torch.Tensor


@dataclass
class HeadCacheStats:
    """Counters for [`TeacherHeadCache`]; `bytes`/`seconds` cover the device transfers it performs.

    `hits`/`misses` count projection leases served by the resident slot and leases that had to (re)upload a head;
    checkpoint replay adds to them, so they are cache statistics rather than training metrics.
    """

    uploads: int = 0
    hits: int = 0
    misses: int = 0
    bytes_uploaded: int = 0
    upload_seconds: float = 0.0
    live_head_bytes: int = 0
    peak_head_bytes: int = 0


@dataclass
class LeasedHead:
    """Device head tensors valid for the duration of one [`TeacherHeadCache.projection_lease`] block.

    `weight` is materialized in the requested execution dtype, so the projection needs no further cast. `bias` stays
    in the source dtype: the loss adds `bias.float()` after upcasting the projection, so rounding it through the
    execution dtype would change the arithmetic. Both fields are set to `None` when the lease exits, so the bundle a
    `with ... as head` binding keeps alive past its block owns nothing and cannot pin an evicted head.
    """

    weight: torch.Tensor | None
    bias: torch.Tensor | None


class _ProjectionLease:
    """Context manager returned by [`TeacherHeadCache.projection_lease`]."""

    def __init__(self, cache: "TeacherHeadCache", key: tuple):
        self._cache = cache
        self._key = key
        self._head: LeasedHead | None = None

    def __enter__(self) -> LeasedHead:
        self._head = self._cache._acquire(self._key)
        return self._head

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # Clear the bundle before the slot becomes available again: a `with ... as head` binding outlives the block,
        # and a bundle that still owned the tensors would keep the evicted head alive. Callers must likewise not
        # extract the tensors into longer-lived names.
        self._head.weight = self._head.bias = None
        self._head = None
        self._cache._release()
        # Falsy: checkpoint early-stop and application exceptions must propagate out of the chunk body.
        return False


class TeacherHeadCache:
    """
    One device head slot serving every teacher identity registered on a rank.

    Head sources are exact CPU tensors retained by the executor; the device copy is disposable. A projection lease
    uploads the requested head into the single slot, keeping it for reuse until another identity is requested, so a
    replay of the checkpointed loss re-acquires the exact head it projected in the forward pass without a second head
    ever being resident. Switching identities finishes outstanding device work and drops the old storage before
    allocating the replacement. Transfers go through one reused pinned staging tile on CUDA and block until complete
    (`non_blocking=False`), so a tile is never rewritten before its copy finishes. All bookkeeping is serialized by one
    lock because autograd worker threads acquire leases during checkpoint replay; a request for a different identity
    waits until the active leases drain rather than breaking the one-head bound.

    On CPU the "upload" is a dtype-cast copy, so hit/miss/eviction accounting stays observable without an accelerator.

    Args:
        device (`torch.device`):
            Device the projections run on. One cache instance serves one device.
        staging_bytes (`int`, *optional*, defaults to `64 << 20`):
            Size of the pinned staging tile used for CUDA uploads. Heads larger than the tile are copied in several
            blocking tiles.
    """

    def __init__(self, device: torch.device, staging_bytes: int = 64 << 20):
        self.device = torch.device(device)
        self.stats = HeadCacheStats()
        self._staging_bytes = staging_bytes
        self._sources: dict[HeadIdentity, HeadSource] = {}
        self._key: tuple | None = None
        self._weight: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        self._leases = 0
        self._staging: torch.Tensor | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)

    def retain_head_source(self, source: HeadSource) -> None:
        """
        Register the exact CPU head tensors for an identity; idempotent per identity.

        Args:
            source ([`HeadSource`]):
                CPU weight/bias to project through while any target using this identity is live.
        """
        with self._lock:
            self._sources.setdefault(source.identity, source)

    def release_head_source(self, identity: HeadIdentity) -> None:
        """
        Drop the CPU head source, and the device slot when it holds this identity.

        Args:
            identity ([`HeadIdentity`]):
                Identity whose source is no longer needed by any live target.
        """
        with self._lock:
            if self._key is not None and self._key[0] == identity and self._leases:
                raise RuntimeError(f"cannot release head source {identity.teacher_id!r} while a lease is active")
            if self._key is not None and self._key[0] == identity:
                self._free_slot()
            self._sources.pop(identity, None)

    def projection_lease(
        self, identity: HeadIdentity, execution_dtype: torch.dtype, operand_dtype: torch.dtype | None = None
    ) -> _ProjectionLease:
        """
        Lease the device head for one projection.

        Args:
            identity ([`HeadIdentity`]):
                Teacher head to project through; its source must be retained.
            execution_dtype (`torch.dtype`):
                Dtype the weight is materialized in: the *effective* dtype the projection matmul executes in, which
                under autocast is the autocast dtype rather than the hidden states' own dtype. Materializing the head
                in it keeps exactly one head resident, since the matmul then needs no implicit conversion.
            operand_dtype (`torch.dtype`, *optional*):
                Dtype the baseline matches the head to before the matmul, i.e. the hidden states' dtype. When it
                differs from `execution_dtype` the two casts are applied in that order, tile by tile in the staging
                buffer, so the rounding sequence is the baseline's. Defaults to `execution_dtype`, meaning a single
                cast.

        Returns:
            `ContextManager[`[`LeasedHead`]`]`: yields the device tensors, which are invalidated on exit; the block
            must finish every use of them before it ends and must not store them elsewhere.
        """
        operand_dtype = execution_dtype if operand_dtype is None else operand_dtype
        # Both dtypes are part of the cache key: the head's values depend on the whole rounding sequence, so a
        # different operand dtype is a different head, not a reusable one.
        return _ProjectionLease(self, (identity, self.device, operand_dtype, execution_dtype))

    def evict_idle_gpu(self) -> None:
        """Free the device slot when no lease is active, after outstanding device work completes."""
        with self._lock:
            if self._leases == 0:
                self._free_slot()

    def close(self) -> None:
        """Free the device slot, the staging tile and all head sources; idempotent."""
        with self._lock:
            if self._leases:
                raise RuntimeError(f"cannot close the head cache while {self._leases} lease(s) are active")
            self._free_slot()
            self._sources.clear()
            self._staging = None
            self._closed = True

    def _acquire(self, key: tuple) -> LeasedHead:
        with self._idle:
            if self._closed:
                raise RuntimeError("the head cache is closed")
            # One slot: a different identity/dtype waits for the resident head's leases to drain instead of
            # allocating a second head.
            while self._leases and self._key != key:
                self._idle.wait()
            if self._key == key:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
                self._free_slot()
                self._upload(key)
            self._leases += 1
            return LeasedHead(self._weight, self._bias)

    def _release(self) -> None:
        with self._idle:
            self._leases -= 1
            self._idle.notify_all()

    def _free_slot(self) -> None:
        if self._weight is None:
            return
        if self.device.type == "cuda":
            # Finish the work reading the head before its storage is dropped.
            torch.cuda.synchronize(self.device)
        self.stats.live_head_bytes -= _head_bytes(self._weight, self._bias)
        self._key = self._weight = self._bias = None

    def _upload(self, key: tuple) -> None:
        identity, _, operand_dtype, execution_dtype = key
        source = self._sources[identity]
        start = time.perf_counter()
        # Detached: a caller-owned teacher's head may still carry `requires_grad`, and no gradient may ever reach it.
        weight = self._staged_copy(source.weight.detach(), operand_dtype, execution_dtype)
        # The bias keeps its source dtype and is vocabulary-sized, so it goes straight over without a staging tile.
        bias = None if source.bias is None else source.bias.detach().to(device=self.device, copy=True)
        self.stats.upload_seconds += time.perf_counter() - start
        self.stats.uploads += 1
        self.stats.bytes_uploaded += _head_bytes(weight, bias)
        self.stats.live_head_bytes += _head_bytes(weight, bias)
        self.stats.peak_head_bytes = max(self.stats.peak_head_bytes, self.stats.live_head_bytes)
        self._key, self._weight, self._bias = key, weight, bias

    def _staged_copy(
        self, source: torch.Tensor, operand_dtype: torch.dtype, execution_dtype: torch.dtype
    ) -> torch.Tensor:
        # The destination owns its storage even when no cast is needed, so eviction is observable on CPU too.
        out = torch.empty(source.shape, dtype=execution_dtype, device=self.device)
        flat_source, flat_out = source.reshape(-1), out.reshape(-1)
        tile = self._staging_tile(execution_dtype)
        for start in range(0, flat_source.numel(), tile.numel()):
            elements = min(tile.numel(), flat_source.numel() - start)
            # `source -> operand dtype -> execution dtype`, the baseline's rounding sequence, one tile at a time:
            # the intermediate is tile-sized, so a narrowing operand dtype never costs a second full head.
            tile[:elements].copy_(flat_source[start : start + elements].to(operand_dtype))
            flat_out[start : start + elements].copy_(tile[:elements], non_blocking=False)  # completes before reuse
        return out

    def _staging_tile(self, dtype: torch.dtype) -> torch.Tensor:
        if self._staging is None or self._staging.dtype != dtype:
            elements = max(1, self._staging_bytes // torch.empty((), dtype=dtype).element_size())
            self._staging = None  # drop the previous tile before allocating the replacement
            self._staging = torch.empty(elements, dtype=dtype, pin_memory=self.device.type == "cuda")
        return self._staging


def _head_bytes(weight: torch.Tensor, bias: torch.Tensor | None) -> int:
    total = weight.numel() * weight.element_size()
    if bias is not None:
        total += bias.numel() * bias.element_size()
    return total
