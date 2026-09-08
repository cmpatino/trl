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
