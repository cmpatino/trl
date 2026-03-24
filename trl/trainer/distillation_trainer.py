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

import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType, broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import (
    is_datasets_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from ..extras.profiling import profiling_decorator
from ..generation.vllm_generation import VLLMGeneration
from ..import_utils import is_vllm_available
from ..models import prepare_deepspeed
from ..models.utils import unwrap_model_for_generation
from .base_trainer import _BaseTrainer
from .distillation_config import DistillationConfig
from .utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    pad,
    split_tensor_dict,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def _print_completions_sample(prompts: list[str], completions: list[str], step: int, num_samples: int = None) -> None:
    """Print a sample of prompt-completion pairs using rich."""
    if not is_rich_available():
        return

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")

    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]

    for prompt, completion in zip(prompts, completions, strict=True):
        table.add_row(Text(prompt), Text(completion))
        table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class _DistillationCollator:
    """Data collator for the distillation trainer with independent prompt/completion budgets.

    Unlike ``DataCollatorForChatML``, this collator tokenizes prompts and completions separately so
    that long completions can never truncate the prompt to empty.  It also handles prompt-only data
    (no assistant completions) for pure on-policy distillation (``lmbda=1``).
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int,
        max_prompt_length: int,
        messages_key: str = "messages",
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.messages_key = messages_key
        self.ignore_index = ignore_index

        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_prompt_ids: list[list[int]] = []

        for example in examples:
            messages = example[self.messages_key]

            # Split: prompt = everything before the last assistant turn, completion = last assistant turn
            has_completion = len(messages) > 1 and messages[-1].get("role") == "assistant"
            prompt_messages = messages[:-1] if has_completion else messages

            # Tokenize prompt with its own budget (truncate from the left to keep recent context)
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            if has_completion:
                # Tokenize the full message (prompt + completion) without truncation first
                formatted_full = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                full_ids = self.tokenizer(
                    formatted_full, truncation=False, padding=False, add_special_tokens=False
                )["input_ids"]

                # Identify completion tokens: everything after the prompt in the full sequence.
                # Use the un-truncated prompt length as the split point.
                formatted_prompt_ids = self.tokenizer(
                    formatted_prompt, truncation=False, padding=False, add_special_tokens=False
                )["input_ids"]
                completion_ids = full_ids[len(formatted_prompt_ids):]

                # Trim completion so prompt + completion <= max_length
                max_comp = self.max_length - len(prompt_ids)
                if max_comp > 0 and len(completion_ids) > max_comp:
                    completion_ids = completion_ids[:max_comp]
                elif max_comp <= 0:
                    completion_ids = []

                input_ids = prompt_ids + completion_ids
                labels = [self.ignore_index] * len(prompt_ids) + list(completion_ids)
            else:
                # Prompt-only: no completion to train on (on-policy will generate one)
                input_ids = list(prompt_ids)
                labels = [self.ignore_index] * len(prompt_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_prompt_ids.append(list(prompt_ids))

        # Convert to tensors and left-pad
        pad_id = self.tokenizer.pad_token_id
        input_ids_t = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids],
            padding_side="left", padding_value=pad_id,
        )
        attention_mask_t = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in all_input_ids],
            padding_side="left", padding_value=0,
        )
        labels_t = pad(
            [torch.tensor(lab, dtype=torch.long) for lab in all_labels],
            padding_side="left", padding_value=self.ignore_index,
        )
        prompts_t = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in all_prompt_ids],
            padding_side="left", padding_value=pad_id,
        )
        prompt_mask_t = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in all_prompt_ids],
            padding_side="left", padding_value=0,
        )

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
            "prompts": prompts_t,
            "prompt_attention_mask": prompt_mask_t,
        }


class DistillationTrainer(_BaseTrainer):
    """
    Trainer for knowledge distillation from a teacher model to a student model.

    Supports:
    - Generalized JSD loss (forward KL, reverse KL, or interpolated JSD via `beta`)
    - On-policy / off-policy mixing via `lmbda` (buffered across gradient accumulation)
    - Local teacher model or external teacher via vLLM server
    - Student on-policy generation via vLLM or model.generate()
    - Liger kernel for memory-efficient fused JSD loss
    """

    _tag_names = ["trl", "distillation"]
    _name = "Distillation"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: DistillationConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            args = DistillationConfig(output_dir="tmp_distillation")

        # ── Student model loading ──
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model_init_kwargs, str):
            import json

            model_init_kwargs = json.loads(model_init_kwargs)
        teacher_model_init_kwargs = args.teacher_model_init_kwargs or {}
        if isinstance(teacher_model_init_kwargs, str):
            import json

            teacher_model_init_kwargs = json.loads(teacher_model_init_kwargs)
        if isinstance(model, str):
            model_name_or_path = model
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            model_name_or_path = model.config._name_or_path if model is not None else None

        # ── Processing class (tokenizer) ──
        if processing_class is None and model_name_or_path is not None:
            processing_class = AutoTokenizer.from_pretrained(model_name_or_path)
        if processing_class is not None:
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        # ── PEFT ──
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # ── Data collator ──
        if data_collator is None:
            data_collator = _DistillationCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
            )

        # ── Liger fused JSD loss ──
        self.use_liger_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            self.use_liger_loss = True

        # ── Teacher model setup ──
        self.teacher_client = None
        self._local_teacher_tokenizer_matches_student = True
        if args.teacher_model_server_url is not None:
            from ..generation.vllm_client import VLLMClient

            self.teacher_client = VLLMClient(base_url=args.teacher_model_server_url, connection_timeout=60.0)
            teacher_model = None
        elif teacher_model is not None:
            if args.teacher_model_init_kwargs is not None and not isinstance(teacher_model, str):
                raise ValueError(
                    "You passed teacher_model_init_kwargs to the config, but your teacher_model is already "
                    "instantiated."
                )

            teacher_model_name_or_path = (
                teacher_model
                if isinstance(teacher_model, str)
                else getattr(getattr(teacher_model, "config", None), "_name_or_path", None)
            )
            if teacher_model_name_or_path is None:
                raise ValueError(
                    "DistillationTrainer requires a local teacher model with `config._name_or_path` set so its "
                    "tokenizer can be validated against the student tokenizer."
                )

            teacher_tokenizer_kwargs = {}
            teacher_revision = teacher_model_init_kwargs.get("revision", args.teacher_model_revision)
            if teacher_revision is not None:
                teacher_tokenizer_kwargs["revision"] = teacher_revision
            if teacher_model_init_kwargs.get("trust_remote_code") is not None:
                teacher_tokenizer_kwargs["trust_remote_code"] = teacher_model_init_kwargs["trust_remote_code"]
            teacher_processing_class = AutoTokenizer.from_pretrained(
                teacher_model_name_or_path, **teacher_tokenizer_kwargs
            )
            if getattr(teacher_processing_class, "pad_token", None) is None:
                teacher_processing_class.pad_token = teacher_processing_class.eos_token
            self._local_teacher_tokenizer_matches_student = self._local_teacher_tokenizers_match(
                processing_class, teacher_processing_class
            )
            if not self._local_teacher_tokenizer_matches_student:
                warnings.warn(
                    "DistillationTrainer's built-in local-teacher loss assumes the student and teacher share the "
                    "same tokenizer. The loaded local teacher tokenizer does not match the student tokenizer, so "
                    "the teacher weights will be left unchanged for subclass overrides. Direct use of the base "
                    "DistillationTrainer with this local teacher remains unsupported.",
                    UserWarning,
                    stacklevel=2,
                )

            if isinstance(teacher_model, str):
                torch_dtype = teacher_model_init_kwargs.get("torch_dtype")
                teacher_model_init_kwargs["torch_dtype"] = (
                    torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)
                )

            if isinstance(teacher_model, str):
                init_kwargs = dict(teacher_model_init_kwargs)
                if args.teacher_model_revision is not None:
                    init_kwargs.setdefault("revision", args.teacher_model_revision)
                if "torch_dtype" in init_kwargs and "dtype" not in init_kwargs:
                    init_kwargs["dtype"] = init_kwargs.pop("torch_dtype")
                teacher_model = create_model_from_path(teacher_model, **init_kwargs)

        # Trainer does not need to remove unused columns — the collator handles raw data
        args.remove_unused_columns = False

        # ── Call _BaseTrainer.__init__ (which is transformers.Trainer.__init__) ──
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # ── Prepare teacher model (after super().__init__ so accelerator is ready) ──
        if teacher_model is not None:
            if self._local_teacher_tokenizer_matches_student:
                teacher_model.resize_token_embeddings(self.model.config.vocab_size)
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        else:
            self.teacher_model = None

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # ── Store config values ──
        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_generations = args.num_generations
        self.teacher_server_top_logprobs = args.teacher_server_top_logprobs

        # ── Buffer state ──
        self._buffered_inputs = None
        self._buffered_on_policy_flags = None
        self._buffered_text_logs = None
        self._buffer_step = 0

        # ── Loss tracking ──
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # ── Generation config ──
        generation_kwargs = {
            "max_new_tokens": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "top_k": args.top_k,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        self.generation_kwargs = generation_kwargs
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # ── Metrics & Logging ──
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completions_steps = args.log_completions_steps
        self.num_completions_to_print = args.num_completions_to_print

        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
        }

        # ── vLLM for student generation ──
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_completion_length=args.max_completion_length,
                logprobs=None,
            )
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

    @staticmethod
    def _local_teacher_tokenizers_match(
        student_processing_class: PreTrainedTokenizerBase,
        teacher_processing_class: PreTrainedTokenizerBase,
    ) -> bool:
        """Check whether the student and local teacher tokenizers share the same vocabulary."""
        return student_processing_class.get_vocab() == teacher_processing_class.get_vocab()

    def _raise_if_local_teacher_tokenizer_mismatch(self) -> None:
        """Guard the base local-teacher JSD path, while still allowing subclass overrides."""
        if self.teacher_model is not None and not self._local_teacher_tokenizer_matches_student:
            raise ValueError(
                "DistillationTrainer's built-in local-teacher loss only supports student/teacher pairs that use "
                "the same tokenizer. Use a same-tokenizer local teacher, use `teacher_model_server_url`, or "
                "override the local teacher loss path in a subclass."
            )

    def _get_completion_lengths(self, generated_tokens: torch.Tensor, prompt_width: int) -> torch.Tensor:
        """Infer per-sample completion lengths from generated tokens."""
        completion_tokens = generated_tokens[:, prompt_width:]
        pad_token_id = self.processing_class.pad_token_id
        eos_token_id = self.generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_ids = set()
        elif isinstance(eos_token_id, int):
            eos_token_ids = {eos_token_id}
        else:
            eos_token_ids = set(eos_token_id)
        pad_equals_eos = pad_token_id is not None and pad_token_id in eos_token_ids

        completion_lengths = []
        for row in completion_tokens.tolist():
            if pad_equals_eos and eos_token_ids:
                completion_length = len(row)
                for idx, token_id in enumerate(row):
                    if token_id in eos_token_ids:
                        completion_length = idx + 1
                        break
            elif pad_token_id is not None:
                completion_length = len(row)
                while completion_length > 0 and row[completion_length - 1] == pad_token_id:
                    completion_length -= 1
            else:
                completion_length = len(row)
            completion_lengths.append(completion_length)

        return torch.tensor(completion_lengths, device=generated_tokens.device, dtype=torch.long)

    # ──────────────────────────────────────────────────────────────────────
    #  Dataset / Dataloader
    # ──────────────────────────────────────────────────────────────────────

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        extra_columns = ["prompts", "prompt_attention_mask", "messages", "chat_template_kwargs", "tools"]
        if self._signature_columns is None:
            self._signature_columns = extra_columns
        else:
            for col in extra_columns:
                if col not in self._signature_columns:
                    self._signature_columns.append(col)

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size * self.accelerator.num_processes,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        """
        Override to load one generation batch per optimizer window.

        The dataloader yields batches of size `per_device_train_batch_size * gradient_accumulation_steps`.
        RepeatSampler ensures each generation batch is repeated `gradient_accumulation_steps` times so the
        Trainer's loop iterates the correct number of times.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # ──────────────────────────────────────────────────────────────────────
    #  Buffering: on/off-policy mixing across gradient accumulation steps
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if not self.model.training:
            return generation_batch

        buffer_steps = self.args.gradient_accumulation_steps
        if self._buffer_step % buffer_steps == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, buffer_steps)

        slice_idx = self._buffer_step % buffer_steps
        inputs = self._buffered_inputs[slice_idx]
        self._buffer_step += 1
        return inputs

    @profiling_decorator
    def _fill_buffer(self, generation_batch: dict[str, torch.Tensor | Any], buffer_steps: int):
        """Split batch into slices and decide which are on-policy (student-generated) vs off-policy."""
        slices = split_tensor_dict(generation_batch, buffer_steps)

        # Decide on-policy flags (synchronized across processes)
        if self.accelerator.is_main_process:
            on_policy_flags = [random.random() <= self.lmbda for _ in range(buffer_steps)]
        else:
            on_policy_flags = [False] * buffer_steps
        on_policy_flags = broadcast_object_list(on_policy_flags, from_process=0)

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_on_policy_flags = on_policy_flags
        self._buffered_text_logs = [None] * buffer_steps

        # Store off-policy slices directly
        on_policy_indices = []
        for i, is_on_policy in enumerate(on_policy_flags):
            if is_on_policy:
                on_policy_indices.append(i)
            else:
                self._buffered_inputs[i] = slices[i]

        # Generate student completions for on-policy slices
        if on_policy_indices:
            self._generate_student_completions(slices, on_policy_indices)

    @profiling_decorator
    def _generate_student_completions(
        self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]
    ):
        """Generate completions from the student model for on-policy training."""
        if not self.use_vllm:
            self._generate_with_model(slices, on_policy_indices)
            return

        # Collect all prompts across on-policy slices, stripping left-padding so vLLM
        # receives only real prompt token IDs (like GRPO trainer).
        local_prompts = []
        local_slice_indices = []
        pad_token_id = self.processing_class.pad_token_id
        for slice_idx in on_policy_indices:
            prompt_mask = slices[slice_idx].get("prompt_attention_mask")
            for i, prompt in enumerate(slices[slice_idx]["prompts"]):
                if prompt_mask is not None:
                    prompt = prompt[prompt_mask[i].bool()]
                elif pad_token_id is not None:
                    first_non_pad = (prompt != pad_token_id).nonzero(as_tuple=True)[0]
                    if len(first_non_pad) > 0:
                        prompt = prompt[first_non_pad[0] :]
                local_prompts.append(prompt)
                local_slice_indices.append(slice_idx)

        # Sync student weights to vLLM if needed
        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step % self.vllm_sync_frequency == 0
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        # Generate completions — pass token IDs directly, no text decoding
        prompt_ids_list = [p.tolist() for p in local_prompts]
        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids_list, images=None, num_generations=self.num_generations
        )

        # Process completions into the buffer
        self._store_completions_in_buffer(
            slices, on_policy_indices, local_slice_indices, local_prompts, completion_ids
        )

    def _generate_with_model(self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]):
        """Fallback generation using model.generate() (no vLLM)."""
        with unwrap_model_for_generation(
            self.model, self.accelerator, generation_kwargs=self.generation_kwargs
        ) as unwrapped_model:
            for slice_idx in on_policy_indices:
                slice_inputs = slices[slice_idx]
                generated_outputs = unwrapped_model.generate(
                    input_ids=slice_inputs["prompts"],
                    attention_mask=slice_inputs.get("prompt_attention_mask", None),
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                )
                generated_tokens = generated_outputs.sequences
                batch_size = generated_tokens.size(0)
                device = generated_tokens.device
                pad_token_id = self.processing_class.pad_token_id
                prompt_width = slice_inputs["prompts"].shape[1]
                prompt_mask = slice_inputs.get("prompt_attention_mask")
                if prompt_mask is not None:
                    prompt_token_lengths = prompt_mask.sum(dim=1)
                else:
                    prompt_token_lengths = torch.full((batch_size,), prompt_width, dtype=torch.long, device=device)
                completion_lengths = self._get_completion_lengths(generated_tokens, prompt_width)
                new_attention_mask, new_labels = self._build_sequence_batch(
                    generated_tokens, prompt_width, prompt_token_lengths, completion_lengths
                )

                # Decode for logging
                prompt_texts = []
                completion_texts = []
                for idx in range(batch_size):
                    prompt_tokens = slice_inputs["prompts"][idx]
                    if prompt_mask is not None:
                        prompt_tokens = prompt_tokens[prompt_mask[idx].bool()]
                    elif pad_token_id is not None:
                        prompt_tokens = prompt_tokens[prompt_tokens != pad_token_id]
                    prompt_texts.append(
                        self.processing_class.decode(prompt_tokens.tolist(), skip_special_tokens=False)
                    )
                    length = prompt_width
                    completion_length = int(completion_lengths[idx].item())
                    completion_texts.append(
                        self.processing_class.decode(
                            generated_tokens[idx, length : length + completion_length].tolist(),
                            skip_special_tokens=False,
                        )
                    )

                updated = dict(slice_inputs)
                updated["input_ids"] = generated_tokens
                updated["attention_mask"] = new_attention_mask
                updated["labels"] = new_labels

                self._buffered_inputs[slice_idx] = updated
                self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    def _store_completions_in_buffer(
        self,
        slices: list[dict[str, torch.Tensor | Any]],
        on_policy_indices: list[int],
        local_slice_indices: list[int],
        local_prompts: list[torch.Tensor],
        completion_ids: list,
    ):
        """Process vLLM completions and store them in the buffer.

        Uses original prompt token IDs directly (no decode/re-encode roundtrip), following the same
        approach as GRPOTrainer.
        """
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        max_completion_length = self.generation_config.max_new_tokens

        # Group completions and prompt token IDs by slice
        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_prompt_ids = {idx: [] for idx in on_policy_indices}
        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(completion_ids[i])
            slice_prompt_ids[slice_idx].append(local_prompts[i])

        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_id_tensors = slice_prompt_ids[slice_idx]
            prompt_width = max(len(p) for p in prompt_id_tensors)
            prompt_token_lengths = torch.tensor([len(p) for p in prompt_id_tensors], device=device, dtype=torch.long)
            prompt_attention_mask = (
                torch.arange(prompt_width, device=device).unsqueeze(0)
                >= (prompt_width - prompt_token_lengths).unsqueeze(1)
            ).long()

            # Left-pad prompt token IDs to the longest prompt in this slice
            prompt_ids = torch.stack([
                F.pad(p, (prompt_width - len(p), 0), value=pad_token_id)
                for p in prompt_id_tensors
            ]).to(device)

            # Pad/truncate completions (right-pad to max_completion_length)
            completion_tensors = []
            completion_ids_for_text = []
            completion_lengths = []
            for comp_ids in slice_completions[slice_idx]:
                t = torch.tensor(comp_ids, device=device)
                if len(t) > max_completion_length:
                    t = t[:max_completion_length]
                completion_ids_for_text.append(t.tolist())
                completion_lengths.append(len(t))
                if len(t) < max_completion_length:
                    padding = torch.full(
                        (max_completion_length - len(t),), pad_token_id, device=device, dtype=t.dtype
                    )
                    t = torch.cat([t, padding])
                completion_tensors.append(t)

            completion_ids_padded = torch.stack(completion_tensors)
            new_input_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
            completion_lengths = torch.tensor(completion_lengths, device=device, dtype=torch.long)
            new_attention_mask, new_labels = self._build_sequence_batch(
                new_input_ids, prompt_width, prompt_token_lengths, completion_lengths
            )

            # Decode for logging only
            prompt_texts = self.processing_class.batch_decode(
                prompt_id_tensors, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            completion_texts = self.processing_class.batch_decode(
                completion_ids_for_text, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            updated = dict(slice_inputs)
            updated["input_ids"] = new_input_ids
            updated["attention_mask"] = new_attention_mask
            updated["labels"] = new_labels
            # Update prompts to match the new padding width so prompt_length is consistent
            updated["prompts"] = prompt_ids
            updated["prompt_attention_mask"] = prompt_attention_mask

            self._buffered_inputs[slice_idx] = updated
            self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    @staticmethod
    def _build_sequence_batch(
        new_input_ids: torch.Tensor,
        prompt_width: int,
        prompt_token_lengths: torch.Tensor,
        completion_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build attention mask and labels from prompt/completion lengths."""
        prompt_token_lengths = prompt_token_lengths.to(device=new_input_ids.device, dtype=torch.long)
        completion_lengths = completion_lengths.to(device=new_input_ids.device, dtype=torch.long)
        positions = torch.arange(new_input_ids.shape[1], device=new_input_ids.device).unsqueeze(0)
        prompt_mask = (positions < prompt_width) & (
            positions >= (prompt_width - prompt_token_lengths).unsqueeze(1)
        )
        completion_mask = (positions >= prompt_width) & (
            positions < (prompt_width + completion_lengths).unsqueeze(1)
        )
        new_attention_mask = (prompt_mask | completion_mask).long()

        new_labels = torch.full_like(new_input_ids, -100)
        new_labels[completion_mask] = new_input_ids[completion_mask]

        return new_attention_mask, new_labels

    # ──────────────────────────────────────────────────────────────────────
    #  Loss computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size).
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size).
            labels: Tensor of shape (batch_size, sequence_length) with -100 for positions to ignore.
            beta: Interpolation coefficient. 0.0 = forward KL, 1.0 = reverse KL.
            temperature: Softmax temperature.
            reduction: 'batchmean', 'sum', 'mean', or 'none'.

        Returns:
            Scalar loss tensor.
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    @staticmethod
    def token_level_divergence_loss(
        student_logprobs,
        teacher_logprobs,
        labels=None,
        beta=0.5,
    ):
        """
        Compute a per-token approximation of the generalized JSD loss using only the sampled token's logprobs.

        This is used when the teacher is an external server and only top-1 logprobs are available.
        For each token position, we have log p_student(token) and log p_teacher(token) and compute:
            - beta=0 (forward KL):  exp(log_teacher) * (log_teacher - log_student)
            - beta=1 (reverse KL):  exp(log_student) * (log_student - log_teacher)
            - 0 < beta < 1 (JSD):   weighted combination of forward and reverse token-level KL terms

        Args:
            student_logprobs: Tensor of shape (batch_size, completion_length) — student's log-prob per token.
            teacher_logprobs: Tensor of shape (batch_size, completion_length) — teacher's log-prob per token.
            labels: Tensor of shape (batch_size, completion_length) with -100 for positions to ignore.
            beta: Interpolation coefficient. 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL.

        Returns:
            Scalar loss tensor.
        """
        if beta == 0:
            # Forward KL: p_teacher * (log_teacher - log_student)
            loss = torch.exp(teacher_logprobs) * (teacher_logprobs - student_logprobs)
        elif beta == 1:
            # Reverse KL: p_student * (log_student - log_teacher)
            loss = torch.exp(student_logprobs) * (student_logprobs - teacher_logprobs)
        else:
            # Token-level JSD approximation
            forward_kl = torch.exp(teacher_logprobs) * (teacher_logprobs - student_logprobs)
            reverse_kl = torch.exp(student_logprobs) * (student_logprobs - teacher_logprobs)
            loss = beta * forward_kl + (1 - beta) * reverse_kl

        if labels is not None:
            mask = labels != -100
            loss = loss[mask]
            return loss.sum() / mask.sum()

        return loss.mean()

    def _get_teacher_logits(self, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        """Get teacher logits — dispatches between local model and external server."""
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                return self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).logits
        elif self.teacher_client is not None:
            return self._get_teacher_logits_from_server(inputs)
        else:
            raise ValueError("No teacher model or teacher server configured.")

    def _get_teacher_token_logprobs_from_server(self, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        """Fetch per-token teacher logprobs from an external vLLM server.

        Returns a tensor of shape (batch_size, completion_length) containing the teacher's log-probability
        for the token present at each completion position in the input sequence.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        prompt_length = inputs["prompts"].shape[1]

        # Extract unpadded sequences
        sequences = []
        prompt_lengths = []
        for i in range(batch_size):
            valid_mask = attention_mask[i].bool()
            seq = input_ids[i][valid_mask].tolist()
            sequences.append(seq)
            prompt_lengths.append(prompt_length)

        # Request top-1 logprobs from the teacher server. The server returns the logprob of the token at each
        # position (i.e., the token in the sequence we sent), which is exactly what we need for all divergences.
        result = self.teacher_client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=1,
        )

        # Build a (batch_size, completion_length) tensor of teacher logprobs for the sequence tokens
        completion_length = max(len(lps) for lps in result["logprobs"])
        device = input_ids.device
        teacher_logprobs = torch.full(
            (batch_size, completion_length), float("-inf"), dtype=torch.float32, device=device
        )
        for i, seq_lps in enumerate(result["logprobs"]):
            for pos, lps in enumerate(seq_lps):
                if lps and lps[0] is not None:
                    teacher_logprobs[i, pos] = lps[0]

        return teacher_logprobs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._raise_if_local_teacher_tokenizer_mismatch()

        if self.use_liger_loss:
            loss = self._compute_liger_loss(model, inputs)
            return (loss, None) if return_outputs else loss

        # Student forward pass
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        prompt_length = inputs["prompts"].shape[1]
        labels = inputs["labels"][:, prompt_length:]

        if self.teacher_client is not None:
            # Server path: token-level divergence using top-1 logprobs
            teacher_token_logprobs = self._get_teacher_token_logprobs_from_server(inputs)

            # Extract student logprobs for the same tokens in the completion region
            student_logits = student_outputs.logits[:, prompt_length - 1 : -1, :]
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            completion_tokens = inputs["input_ids"][:, prompt_length:]
            # Trim to match completion length from server
            comp_len = teacher_token_logprobs.shape[1]
            completion_tokens = completion_tokens[:, :comp_len]
            student_token_logprobs = student_log_probs[:, :comp_len, :].gather(
                dim=-1, index=completion_tokens.unsqueeze(-1)
            ).squeeze(-1)

            loss = self.token_level_divergence_loss(
                student_logprobs=student_token_logprobs,
                teacher_logprobs=teacher_token_logprobs,
                labels=labels[:, :comp_len],
                beta=self.beta,
            )
        else:
            # Local teacher: full-vocabulary generalized JSD
            teacher_logits = self._get_teacher_logits(inputs)
            student_logits = student_outputs.logits[:, prompt_length - 1 : -1, :]
            teacher_logits = teacher_logits[:, prompt_length - 1 : -1, :]

            loss = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                beta=self.beta,
                temperature=self.temperature,
            )

        return (loss, student_outputs) if return_outputs else loss

    def _compute_liger_loss(self, model, inputs):
        """Memory-efficient JSD using Liger kernel (operates on hidden states, not full logits)."""
        unwrapped_student = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
            base_student = unwrapped_student.get_decoder()
        else:
            base_student = getattr(
                unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
            )

        student_outputs = base_student(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )

        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
            base_teacher = unwrapped_teacher.get_decoder()
        else:
            base_teacher = getattr(
                unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
            )
        with torch.no_grad():
            teacher_outputs = base_teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        student_hidden = student_outputs.last_hidden_state[:, :-1]
        teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]
        del student_outputs, teacher_outputs

        student_hidden = student_hidden.reshape(-1, student_hidden.shape[-1])
        teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])

        labels_mask = inputs["labels"] != -100
        masked_input_ids = torch.where(
            labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
        )
        true_labels = masked_input_ids[:, 1:].contiguous().reshape(-1)

        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()

        loss = self.liger_jsd_loss(
            student_input=student_hidden,
            student_weight=student_head.weight,
            teacher_input=teacher_hidden,
            teacher_weight=teacher_head.weight,
            true_labels=true_labels,
            student_bias=getattr(student_head, "bias", None),
            teacher_bias=getattr(teacher_head, "bias", None),
        )

        del student_hidden, teacher_hidden, true_labels
        return loss

    def _get_liger_zero3_lm_head_gather_ctx(self, model: nn.Module):
        """Context manager for gathering lm_head parameters under Liger + ZeRO-3."""
        if not self.use_liger_loss:
            return nullcontext()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 3:
            return nullcontext()

        import deepspeed

        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()
        params = [student_head.weight, teacher_head.weight]
        if student_head.bias is not None:
            params.append(student_head.bias)
        if teacher_head.bias is not None:
            params.append(teacher_head.bias)
        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)

    # ──────────────────────────────────────────────────────────────────────
    #  Training step & Logging
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """Training step with on/off-policy loss tracking and completion stats."""
        buffer_steps = self.args.gradient_accumulation_steps

        with self._get_liger_zero3_lm_head_gather_ctx(model):
            loss = super().training_step(model, inputs, num_items_in_batch)

        slice_idx = (self._buffer_step - 1) % buffer_steps

        # Track on-policy text logs
        is_on_policy = False
        if self._buffered_on_policy_flags is not None and slice_idx < len(self._buffered_on_policy_flags):
            is_on_policy = self._buffered_on_policy_flags[slice_idx]

        if is_on_policy and self._buffered_text_logs is not None and self._buffered_text_logs[slice_idx] is not None:
            prompt_texts, completion_texts = self._buffered_text_logs[slice_idx]
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        # Track completion length stats
        labels = inputs.get("labels")
        if labels is not None:
            completion_lengths = (labels != -100).sum(dim=1).float()
            gathered_lengths = self.accelerator.gather(completion_lengths)
            mode = "train"
            prefix = "on_policy" if is_on_policy else "off_policy"
            self._metrics[mode][f"completions/{prefix}_mean_length"].append(gathered_lengths.mean().item())
            self._metrics[mode][f"completions/{prefix}_max_length"].append(gathered_lengths.max().item())
            self._metrics[mode][f"completions/{prefix}_min_length"].append(gathered_lengths.min().item())

        # Track loss per policy type
        loss_scalar = float(loss.detach())
        step_equiv = 1.0 / self.args.gradient_accumulation_steps
        if is_on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv

        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "train":
            # Aggregate on/off-policy losses across distributed processes
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            on_sum, off_sum, on_eq, off_eq = vec.tolist()
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        # Log completions to console and wandb
        if (
            self.accelerator.is_main_process
            and self.log_completions
            and self.state.global_step % self.log_completions_steps == 0
        ):
            prompts = list(self._textual_logs["prompt"])
            completions = list(self._textual_logs["completion"])

            _print_completions_sample(prompts, completions, self.state.global_step, self.num_completions_to_print)

            # Log as a wandb Table
            if self.args.report_to and "wandb" in self.args.report_to:
                try:
                    import wandb

                    if wandb.run is not None:
                        import pandas as pd

                        table_data = {
                            "step": [str(self.state.global_step)] * len(prompts),
                            "prompt": prompts,
                            "completion": completions,
                        }
                        df = pd.DataFrame(table_data)
                        if self.num_completions_to_print and len(df) > self.num_completions_to_print:
                            df = df.sample(n=self.num_completions_to_print, random_state=42)
                        wandb.log({"completions": wandb.Table(dataframe=df)})
                except ImportError:
                    pass
