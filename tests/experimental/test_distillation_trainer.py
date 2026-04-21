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

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from trl.experimental.distillation.distillation_trainer import (
    _jsd_divergence,
    _RepeatBatchDataLoader,
    build_teacher_request_inputs,
)


class DummyStudentModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.logits)


class DummyTeacherModel:
    def __init__(self, logits):
        self.logits = logits
        self.eval_called = False
        self.last_kwargs = None

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(logits=self.logits)


class DummyTeacherClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
        self.calls.append(
            {
                "sequences": sequences,
                "prompt_lengths": prompt_lengths,
                "top_logprobs": top_logprobs,
                "temperature": temperature,
            }
        )
        return self.result


def _make_inputs(input_ids, labels):
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def _build_server_result(teacher_logits, inputs, temperature=1.0):
    prompt_length = int((inputs["attention_mask"].sum(dim=1) - (inputs["labels"] != -100).sum(dim=1)).min().item())
    teacher_logits = teacher_logits[:, prompt_length - 1 : -1, :]
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    completion_tokens = inputs["input_ids"][:, prompt_length:]
    teacher_top1_token_ids = teacher_logits.argmax(dim=-1, keepdim=True)
    teacher_top1_logprobs = teacher_log_probs.gather(dim=-1, index=teacher_top1_token_ids)
    actual_teacher_logprobs = teacher_log_probs.gather(dim=-1, index=completion_tokens.unsqueeze(-1))
    return {
        "actual_logprobs": actual_teacher_logprobs.tolist(),
        "logprobs": teacher_top1_logprobs.tolist(),
        "logprob_token_ids": teacher_top1_token_ids.tolist(),
    }


def _build_local_trainer(teacher_logits, beta, reverse_mode="sampled", loss_add_tail=True):
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = DummyTeacherModel(teacher_logits)
    trainer.teacher_client = None
    trainer.use_teacher_server = False
    trainer.teacher_model_server_url = None
    trainer.use_liger_loss = False
    trainer._local_teacher_tokenizer_matches_student = True
    trainer.beta = beta
    trainer.temperature = 1.0
    trainer.reverse_kl_top_1_mode = reverse_mode
    trainer.loss_top_k = 1
    trainer.loss_add_tail = loss_add_tail
    return trainer


def _build_server_trainer(server_result, beta, loss_add_tail=True):
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = None
    trainer.teacher_client = DummyTeacherClient(server_result)
    trainer.use_teacher_server = True
    trainer.teacher_model_server_url = "http://localhost:8000"
    trainer.use_liger_loss = False
    trainer._local_teacher_tokenizer_matches_student = True
    trainer.beta = beta
    trainer.temperature = 1.0
    trainer.reverse_kl_top_1_mode = "sampled"
    trainer.loss_top_k = 1
    trainer.loss_add_tail = loss_add_tail
    return trainer


def test_distillation_config_rejects_liger_with_teacher_server(tmp_path):
    with pytest.raises(ValueError, match="use_liger_kernel=True is not supported with use_teacher_server=True"):
        DistillationConfig(
            output_dir=str(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            use_liger_kernel=True,
            report_to="none",
        )


def test_distillation_config_rejects_invalid_reverse_kl_top_1_mode(tmp_path):
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode must be one of"):
        DistillationConfig(output_dir=str(tmp_path), reverse_kl_top_1_mode="invalid", report_to="none")


def test_distillation_config_rejects_teacher_server_with_reverse_kl_argmax(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="reverse_kl_top_1_mode='argmax' is not supported"):
            DistillationConfig(
                output_dir=str(tmp_path),
                use_teacher_server=True,
                teacher_model_server_url="http://localhost:8000",
                reverse_kl_top_1_mode="argmax",
                report_to="none",
            )

    assert caught == []


def test_distillation_config_rejects_teacher_server_mixed_loss_without_top_1(tmp_path):
    with pytest.raises(ValueError, match="loss_top_k must be 1 when using use_teacher_server=True with beta>0"):
        DistillationConfig(
            output_dir=str(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            beta=0.5,
            loss_top_k=2,
            report_to="none",
        )


def test_distillation_config_requires_teacher_server_url(tmp_path):
    with pytest.raises(ValueError, match="teacher_model_server_url must be set when use_teacher_server=True"):
        DistillationConfig(output_dir=str(tmp_path), use_teacher_server=True, beta=0.5, loss_top_k=1, report_to="none")


def test_get_teacher_logits_uses_local_teacher_model():
    expected_logits = torch.randn(2, 3, 4)

    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = DummyTeacherModel(expected_logits)
    trainer.teacher_client = None
    trainer.use_teacher_server = False
    trainer.teacher_model_server_url = None

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
    }

    logits = DistillationTrainer._get_teacher_logits(trainer, inputs)

    assert trainer.teacher_model.eval_called is True
    assert trainer.teacher_model.last_kwargs == inputs
    assert torch.equal(logits, expected_logits)


def test_get_teacher_logits_rejects_teacher_server():
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = None
    trainer.teacher_client = None
    trainer.use_teacher_server = True
    trainer.teacher_model_server_url = None

    with pytest.raises(NotImplementedError, match="Server-backed distillation only supports per-token logprobs"):
        DistillationTrainer._get_teacher_logits(
            trainer,
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        )


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_local_argmax_mode_matches_previous_local_top_1_behavior(beta):
    inputs = _make_inputs(input_ids=[9, 1, 2], labels=[-100, 1, 2])
    student_logits = torch.tensor(
        [
            [
                [0.1, 3.2, 0.3, 1.1],
                [2.4, 0.1, 0.2, 1.7],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [
            [
                [0.2, 2.7, 0.1, 1.4],
                [1.9, 0.3, 0.2, 2.2],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    trainer = _build_local_trainer(teacher_logits, beta=beta, reverse_mode="argmax")
    loss = DistillationTrainer.compute_loss(trainer, DummyStudentModel(student_logits), inputs)

    prompt_length = 1
    expected = DistillationTrainer.generalized_jsd_loss(
        student_logits=student_logits[:, prompt_length - 1 : -1, :],
        teacher_logits=teacher_logits[:, prompt_length - 1 : -1, :],
        labels=inputs["labels"][:, prompt_length:],
        beta=beta,
        temperature=1.0,
        top_k=1,
        add_tail=True,
    )

    torch.testing.assert_close(loss, expected)


def test_sampled_mode_matches_between_local_and_external_teachers():
    inputs = _make_inputs(input_ids=[9, 1, 2], labels=[-100, 1, 2])
    student_logits = torch.tensor(
        [
            [
                [0.2, 0.1, 3.5, 1.0],
                [1.0, 0.2, 0.3, 2.8],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [
            [
                [0.1, 0.2, 1.3, 3.4],
                [0.2, 0.1, 3.1, 2.9],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    local_trainer = _build_local_trainer(teacher_logits, beta=0.5, reverse_mode="sampled")
    local_loss = DistillationTrainer.compute_loss(local_trainer, DummyStudentModel(student_logits), inputs)

    server_result = _build_server_result(teacher_logits, inputs)
    server_trainer = _build_server_trainer(server_result, beta=0.5)
    server_loss = DistillationTrainer.compute_loss(server_trainer, DummyStudentModel(student_logits), inputs)

    assert server_trainer.teacher_client.calls[0]["top_logprobs"] == 1
    torch.testing.assert_close(local_loss, server_loss)


def test_sampled_mode_keeps_teacher_argmax_for_forward_support():
    inputs = _make_inputs(input_ids=[9, 1], labels=[-100, 1])
    student_logits = torch.tensor([[[0.4, 0.3, 2.0, 1.1], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[0.2, 0.1, 0.4, 2.5], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    server_result = _build_server_result(teacher_logits, inputs)
    server_trainer = _build_server_trainer(server_result, beta=0.5, loss_add_tail=False)
    loss = DistillationTrainer.compute_loss(server_trainer, DummyStudentModel(student_logits), inputs)

    prompt_length = 1
    student_log_probs = F.log_softmax(student_logits[:, prompt_length - 1 : -1, :], dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits[:, prompt_length - 1 : -1, :], dim=-1)
    support = torch.tensor([[[3, 1]]], dtype=torch.long)
    support_mask = torch.ones_like(support, dtype=torch.bool)
    student_support = student_log_probs.gather(-1, support)
    teacher_support = teacher_log_probs.gather(-1, support)
    student_support = student_support - torch.logsumexp(student_support, dim=-1, keepdim=True)
    teacher_support = teacher_support - torch.logsumexp(teacher_support, dim=-1, keepdim=True)
    expected = _jsd_divergence(student_support, teacher_support, beta=0.5, support_mask=support_mask).sum()

    torch.testing.assert_close(loss, expected)


def test_server_teacher_path_handles_variable_prompt_lengths():
    class VariablePromptTeacherClient(DummyTeacherClient):
        def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
            assert sequences == [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]]
            assert prompt_lengths == [3, 5]
            assert top_logprobs == 1
            return self.result

    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = None
    trainer.teacher_client = VariablePromptTeacherClient(
        {
            "actual_logprobs": [[[-0.1], [-0.2]], [[-0.3], [-0.4]]],
            "logprobs": [[[-0.1], [-0.2]], [[-0.3], [-0.4]]],
            "logprob_token_ids": [[[20], [21]], [[40], [41]]],
        }
    )
    trainer.use_teacher_server = True
    trainer.teacher_model_server_url = "http://localhost:8000"
    trainer.use_liger_loss = False
    trainer._local_teacher_tokenizer_matches_student = True
    trainer.beta = 1.0
    trainer.temperature = 1.0
    trainer.reverse_kl_top_1_mode = "sampled"
    trainer.loss_top_k = 1
    trainer.loss_add_tail = False

    inputs = {
        "input_ids": torch.tensor([[0, 0, 10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor(
            [[-100, -100, -100, -100, -100, 20, 21], [-100, -100, -100, -100, -100, 40, 41]],
            dtype=torch.long,
        ),
        "prompts": torch.tensor([[0, 0, 10, 11, 12], [30, 31, 32, 33, 34]], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long),
    }

    batch_size, sequence_length = inputs["input_ids"].shape
    loss = DistillationTrainer.compute_loss(
        trainer,
        DummyStudentModel(torch.zeros(batch_size, sequence_length, 64, dtype=torch.float32)),
        inputs,
    )

    assert torch.isfinite(loss)


def test_server_teacher_path_handles_on_policy_padded_completions():
    class OnPolicyTeacherClient(DummyTeacherClient):
        def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
            assert sequences == [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]]
            assert prompt_lengths == [3, 5]
            assert top_logprobs == 1
            return self.result

    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.teacher_model = None
    trainer.teacher_client = OnPolicyTeacherClient(
        {
            "actual_logprobs": [[[-0.1], [-0.2]], [[-0.3], [-0.4]]],
            "logprobs": [[[-0.1], [-0.2]], [[-0.3], [-0.4]]],
            "logprob_token_ids": [[[20], [21]], [[40], [41]]],
        }
    )
    trainer.use_teacher_server = True
    trainer.teacher_model_server_url = "http://localhost:8000"
    trainer.use_liger_loss = False
    trainer._local_teacher_tokenizer_matches_student = True
    trainer.beta = 1.0
    trainer.temperature = 1.0
    trainer.reverse_kl_top_1_mode = "sampled"
    trainer.loss_top_k = 1
    trainer.loss_add_tail = False

    inputs = {
        "input_ids": torch.tensor(
            [[0, 0, 10, 11, 12, 20, 21, 0, 0], [30, 31, 32, 33, 34, 40, 41, 0, 0]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[-100, -100, -100, -100, -100, 20, 21, -100, -100], [-100, -100, -100, -100, -100, 40, 41, -100, -100]],
            dtype=torch.long,
        ),
        "prompts": torch.tensor([[0, 0, 10, 11, 12], [30, 31, 32, 33, 34]], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long),
    }

    batch_size, sequence_length = inputs["input_ids"].shape
    loss = DistillationTrainer.compute_loss(
        trainer,
        DummyStudentModel(torch.zeros(batch_size, sequence_length, 64, dtype=torch.float32)),
        inputs,
    )

    assert torch.isfinite(loss)


def test_build_teacher_request_inputs_preserves_empty_completions():
    sequences, prompt_lengths, completion_lengths = build_teacher_request_inputs(
        input_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100]], dtype=torch.long),
    )

    assert sequences == [[10, 11, 12]]
    assert prompt_lengths == [3]
    assert completion_lengths == [0]


def test_repeat_batch_dataloader_delegates_set_epoch_via_getattr():
    class DummyDataLoader:
        def __init__(self):
            self.epoch = None

        def __iter__(self):
            yield {"x": 1}

        def __len__(self):
            return 1

        def set_epoch(self, epoch):
            self.epoch = epoch

    dataloader = DummyDataLoader()
    wrapper = _RepeatBatchDataLoader(dataloader, repeat_count=2)

    wrapper.set_epoch(7)

    assert dataloader.epoch == 7
