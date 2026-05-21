# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Optional, Sized, Union
import re
import PIL.Image
import torch
import torch.utils.data
import transformers
from accelerate.utils import is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from model.vlm_module import VLMBaseModule

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# Per-step diagnostic prints (sample indices, generated completions, etc.) are
# silenced by default. Set GRPO_MA_VERBOSE=1 to re-enable them.
_VERBOSE = os.environ.get("GRPO_MA_VERBOSE", "0").lower() not in ("", "0", "false", "no")


class SynchronizedSampler(Sampler):
    """
    Sampler that ensures all GPU processes iterate over the same samples.

    The same ``seed`` produces the same shuffle on every rank, so each rank
    yields identical indices. Each sample is repeated ``repeat_count`` times
    per PPO iteration and each PPO update is repeated ``num_iterations`` times.
    """

    def __init__(
        self,
        data_source: Sized,
        repeat_count: int = 8,
        num_iterations: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_iterations = num_iterations
        self.num_samples = len(data_source)
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)

        all_indices = torch.randperm(self.num_samples, generator=generator).tolist()
        if _VERBOSE:
            print(f"SynchronizedSampler: generated indices: {all_indices[:5]}...")

        for sample_idx in all_indices:
            for iteration in range(self.num_iterations):
                if _VERBOSE:
                    print(f"SynchronizedSampler: yielding sample {sample_idx} "
                          f"for iteration {iteration + 1}/{self.num_iterations} with {self.repeat_count} repeats")
                for _ in range(self.repeat_count):
                    yield sample_idx

    def __len__(self) -> int:
        return self.num_samples * self.num_iterations * self.repeat_count


class GRPOMATrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with hierarchical thinking support.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        enable_reward_debug: bool = True,
        **kwargs,
    ):
        # Args initialization
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Validate required parameters
        if vlm_module is None:
            raise ValueError("vlm_module is required for GRPOMATrainer")

        self.vlm_module = vlm_module

        # Model initialization
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype

        # Validate model parameter
        if not isinstance(model, str):
            raise ValueError("model must be a string in the current implementation")

        model_id = model
        ref_model_id = args.ref_model_id or model_id

        # Process torch_dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            try:
                torch_dtype = getattr(torch, torch_dtype)
            except AttributeError as exc:
                raise ValueError(f"Invalid torch_dtype: {torch_dtype}") from exc

        # Disable caching if gradient checkpointing is enabled
        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # Add this line to fix the DeepSpeed issue
        model_init_kwargs["_fast_init"] = False

        # Load model
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # Apply LoRA if configured
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            print("Applying LoRA...")
            target_modules = self._find_linear_modules(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules if requested
        if freeze_vision_modules:
            print("Freezing vision modules...")
            self._freeze_vision_modules(model)

        # Print trainable parameters info
        self._print_trainable_params(model)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Initialize reference model
        self.beta = args.beta
        self.ref_model = self._initialize_reference_model(model, model_cls, ref_model_id, model_init_kwargs)

        # Initialize processing class
        if processing_class is None:
            processing_class = self._initialize_processing_class(model_id, model_init_kwargs, kwargs)

        # Post-initialization setup
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Initialize reward functions and processing classes
        self.reward_funcs = self._initialize_reward_functions(reward_funcs, model_init_kwargs)
        self.reward_processing_classes = self._initialize_reward_processing_classes(reward_processing_classes, self.reward_funcs)

        # Training configuration
        self._setup_training_config(args, processing_class)

        # Multi-step configuration
        self.num_iterations = args.num_iterations
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # Initialize metrics
        self._metrics = defaultdict(list)

        # Data collator (no collation needed for GRPO)
        def data_collator(features):
            return features

        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # GRPO-MA specific validation
        self._validate_grpo_ma_config(args)

        # Set unique seed for each process
        set_seed(args.seed, device_specific=False)

        # Set model loss kwargs flag
        self.model_accepts_loss_kwargs = False

        # Prepare models for distributed training
        self._prepare_models_for_training()

        self.enable_reward_debug = enable_reward_debug

        if self.enable_reward_debug:
            self._reward_debug_metrics = defaultdict(list)

    def _find_linear_modules(self, model, multimodal_keywords):
        """Find all linear modules for LoRA, excluding vision modules."""
        cls = torch.nn.Linear
        lora_module_names = set()

        for name, module in model.named_modules():
            # Skip vision modules
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                lora_module_names.add(name)

        # Remove embed_tokens for 16-bit compatibility
        lora_module_names = {m for m in lora_module_names if "embed_tokens" not in m}
        return list(lora_module_names)

    def _freeze_vision_modules(self, model):
        """Freeze parameters belonging to vision modules."""
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in self.vision_modules_keywords):
                param.requires_grad = False

    def _print_trainable_params(self, model):
        """Print the number of trainable parameters."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params:,}")

    def _initialize_reference_model(self, model, model_cls, model_id, model_init_kwargs):
        """Initialize reference model based on configuration."""
        if self.beta == 0.0:
            return None
        elif is_deepspeed_zero3_enabled():
            return model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            return None
        else:
            return create_reference_model(model)

    def _initialize_processing_class(self, model_id, model_init_kwargs, kwargs):
        """Initialize processing class with custom keywords."""
        processing_cls = self.vlm_module.get_processing_class()
        processing_class = processing_cls.from_pretrained(
            model_id,
            trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
        )

        # Set custom processing keywords
        for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
            if processing_keyword in kwargs:
                processing_component = getattr(processing_class, component, processing_class)
                setattr(processing_component, processing_keyword, kwargs[processing_keyword])

        # Set token IDs
        if hasattr(processing_class, "tokenizer"):
            processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        else:
            if not isinstance(processing_class, PreTrainedTokenizerBase):
                raise ValueError("processing_class must be an instance of PreTrainedTokenizerBase")

        return processing_class

    def _initialize_reward_functions(self, reward_funcs, model_init_kwargs):
        """Initialize reward functions."""
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **model_init_kwargs,
                )

        return reward_funcs

    def _initialize_reward_processing_classes(self, reward_processing_classes, reward_funcs):
        """Initialize processing classes for reward models."""
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        return reward_processing_classes

    def _setup_training_config(self, args, processing_class):
        """Setup training configuration parameters."""
        self.max_prompt_length = args.max_prompt_length
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, falling back to None.")
            self.max_prompt_length = None

        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id,
        )

        if hasattr(self.vlm_module, "get_eos_token_id"):
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)

        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

    def _validate_grpo_ma_config(self, args):
        """Validate GRPO-MA specific configuration."""
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        num_think_samples = getattr(args, "num_think_samples", 8)

        if global_batch_size < 1:
            raise ValueError(f"Global batch size must be at least 1, got {global_batch_size}")

        if global_batch_size % num_think_samples != 0:
            warnings.warn(f"Global batch size ({global_batch_size}) is not evenly divisible by "
                          f"num_think_samples ({num_think_samples}). This may cause uneven sample distribution.",)

    def _prepare_models_for_training(self):
        """Prepare models for distributed training."""
        # Suppress FLOPs estimation warning
        if hasattr(self.model, "warnings_issued"):
            self.model.warnings_issued["estimate_tokens"] = True

        # Prepare reference model
        if self.ref_model is not None:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Prepare reward functions
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enable gradient checkpointing for the model."""
        model.config.use_cache = False

        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            if getattr(model, "language_model", None) is not None:
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                args.gradient_checkpointing = False
            else:
                model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = ("use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"])

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_key_from_inputs(self, x, key):
        element = x.get(key, None)
        assert element is not None, f"The key {key} is not found in the input"
        if isinstance(element, list):
            return [e for e in element]
        return [element]

    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _get_think_and_answer_entropy(self, model, input_ids, attention_mask, think_mask, answer_mask, **custom_multimodal_inputs):
        """
        Compute the entropy for think and answer segments and log them to metrics.
        """
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits

        aligned_logits = logits[:, :-1, :]

        per_token_entropy = Categorical(logits=aligned_logits).entropy()

        aligned_think_mask = think_mask.float()
        aligned_answer_mask = answer_mask.float()

        think_entropy_values = per_token_entropy * aligned_think_mask
        answer_entropy_values = per_token_entropy * aligned_answer_mask

        num_think_tokens = aligned_think_mask.sum(dim=1).clamp(min=1e-5)
        num_answer_tokens = aligned_answer_mask.sum(dim=1).clamp(min=1e-5)

        think_entropy_per_sequence = think_entropy_values.sum(dim=1) / num_think_tokens
        answer_entropy_per_sequence = answer_entropy_values.sum(dim=1) / num_answer_tokens

        think_entropy_mean = think_entropy_per_sequence.mean()
        answer_entropy_mean = answer_entropy_per_sequence.mean()

        if hasattr(self, "_metrics") and hasattr(self, "accelerator") and isinstance(self._metrics, dict):
            if "think_entropy" not in self._metrics:
                self._metrics["think_entropy"] = []
            if "answer_entropy" not in self._metrics:
                self._metrics["answer_entropy"] = []

            self._metrics["think_entropy"].append(self.accelerator.gather_for_metrics(think_entropy_mean).mean().item())
            self._metrics["answer_entropy"].append(self.accelerator.gather_for_metrics(answer_entropy_mean).mean().item())

        return {
            "think_entropy": think_entropy_mean,
            "answer_entropy": answer_entropy_mean,
            "think_entropy_per_sequence": think_entropy_per_sequence,
            "answer_entropy_per_sequence": answer_entropy_per_sequence,
        }

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        base_seed = self.args.seed if self.args.seed is not None else 42
        generation_seed = base_seed + self.accelerator.process_index + self.state.global_step

        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(generation_seed)

        pad_token_id_val = self.processing_class.pad_token_id
        eos_token_id_val = self.processing_class.eos_token_id

        prompts_text_for_thinking = self.vlm_module.prepare_prompt(self.processing_class, inputs)

        is_batch_conversational = is_conversational(inputs[0])

        images_per_prompt = []
        for x in inputs:
            current_prompt_images = []
            if "image" in x:
                imgs_data = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs_data = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            else:
                imgs_data = []

            for img_data in imgs_data:
                try:
                    img = img_data
                    if not isinstance(img, PIL.Image.Image):
                        continue

                    w, h = img.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w, new_h = 28, int(h * (28 / w))
                        else:
                            new_h, new_w = 28, int(w * (28 / h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                    current_prompt_images.append(img)
                except Exception as exc:
                    warnings.warn(f"Failed to process image: {exc}. Skipping this image.")
            images_per_prompt.append(current_prompt_images)
        question_types = [x.get("question_type", None) for x in inputs]
        data_modality = [x.get("data_modality", "text") for x in inputs]
        original_prompts_text_for_answers = deepcopy(prompts_text_for_thinking)

        flat_images_for_thinking_phase = [img for prompt_imgs in images_per_prompt for img in prompt_imgs]
        if not flat_images_for_thinking_phase and any(images_per_prompt):
            warnings.warn("All images failed to load/process for this batch.")

        prompt_inputs_for_thinking = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text_for_thinking,
            images_per_prompt,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            data_modality=data_modality,
        )
        prompt_inputs_for_thinking = super()._prepare_inputs(prompt_inputs_for_thinking)

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            thinking_generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length - 1,
                do_sample=True,
                temperature=self.args.temperature,
                pad_token_id=self.processing_class.pad_token_id,
                stop_strings=self.args.stop_strings,
            )
            if hasattr(self.vlm_module, "get_eos_token_id"):
                thinking_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)

            tokenizer_for_generate = getattr(self.processing_class, "tokenizer", self.processing_class)

            generate_kwargs_for_thinking = {k: v for k, v in prompt_inputs_for_thinking.items() if k not in self.vlm_module.get_non_generate_params()}
            thinking_outputs_tokens = unwrapped_model.generate(
                **generate_kwargs_for_thinking,
                generation_config=thinking_generation_config,
                tokenizer=tokenizer_for_generate,
            )

            initial_prompt_len_scalar = prompt_inputs_for_thinking["input_ids"].size(1)
            if not self.vlm_module.is_embeds_input():
                thinking_completion_ids = thinking_outputs_tokens[:, initial_prompt_len_scalar:]
            else:
                thinking_completion_ids = thinking_outputs_tokens

        thinking_texts = self.processing_class.batch_decode(thinking_completion_ids, skip_special_tokens=True)

        thinking_validity = []
        for think_text_sample in thinking_texts:
            normalized_think_text = think_text_sample.strip()
            starts_with_think_tag = normalized_think_text.startswith("<think>")
            ends_with_think_tag = normalized_think_text.endswith("</think>")
            thinking_validity.append(starts_with_think_tag and ends_with_think_tag)

        thinking_validity_tensor = torch.tensor(thinking_validity, device=device, dtype=torch.bool)
        if self.state.global_step < self.state.max_steps * self.args.warmup_ratio:
            # Use a single answer per thinking sample during the warmup phase
            num_answers_per_thinking = 1
        else:
            num_answers_per_thinking = getattr(self.args, "num_answers_per_thinking", 8)
        K_local = len(thinking_texts)

        all_complete_sequences = []
        all_complete_attention_masks = []
        all_segment_info = []
        all_full_completions_for_reward = []
        expanded_original_prompts_for_reward = []

        for i in range(K_local):
            current_original_prompt_text = original_prompts_text_for_answers[i]
            current_thinking_text = thinking_texts[i]

            current_prompt_image_list_for_m_answers = [images_per_prompt[i] if images_per_prompt[i] else None] * num_answers_per_thinking

            prompt_plus_valid_think_text = current_original_prompt_text + current_thinking_text

            valid_thinking_prompt_inputs = self.vlm_module.prepare_model_inputs(
                self.processing_class,
                [prompt_plus_valid_think_text] * num_answers_per_thinking,
                current_prompt_image_list_for_m_answers,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                data_modality=data_modality,
            )
            valid_thinking_prompt_inputs = super()._prepare_inputs(valid_thinking_prompt_inputs)
            prompt_plus_think_len = valid_thinking_prompt_inputs["input_ids"].size(1)
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model_ans:
                current_thinking_ids = thinking_completion_ids[i]
                # Remove padding tokens to measure the actual thinking length
                actual_thinking_tokens = (current_thinking_ids != pad_token_id_val).sum().item()

                # Adjust max_new_tokens dynamically so the combined length stays within the limit
                remaining_tokens = max(1, self.max_completion_length - actual_thinking_tokens)
                if remaining_tokens >= 1:
                    answer_generation_config = GenerationConfig(
                        max_new_tokens=remaining_tokens,
                        do_sample=True,
                        temperature=self.args.temperature,
                        pad_token_id=pad_token_id_val,
                    )
                    if hasattr(self.vlm_module, "get_eos_token_id"):
                        answer_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)

                    generate_kwargs_for_answers = {k: v for k, v in valid_thinking_prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}
                    answer_outputs_tokens = unwrapped_model_ans.generate(
                        **generate_kwargs_for_answers,
                        generation_config=answer_generation_config,
                        tokenizer=tokenizer_for_generate,
                    )
                else:
                    answer_outputs_tokens = torch.tensor([eos_token_id_val], dtype=torch.long, device=device)  # No answer generated, will use dummy answer below
                if not self.vlm_module.is_embeds_input():
                    current_answer_completion_ids = answer_outputs_tokens[:, prompt_plus_think_len:]
                else:
                    current_answer_completion_ids = answer_outputs_tokens

            for ans_idx in range(num_answers_per_thinking):
                # Concatenate token IDs directly instead of joining decoded text
                prompt_thinking_ids = valid_thinking_prompt_inputs["input_ids"][ans_idx]
                answer_ids = current_answer_completion_ids[ans_idx]

                # Build the full sequence from prompt, thinking, and answer tokens
                complete_ids = torch.cat([prompt_thinking_ids, answer_ids], dim=0)
                complete_attention_mask = (complete_ids != pad_token_id_val).long()

                # Derive segment boundaries based on known lengths
                prompt_len = initial_prompt_len_scalar
                thinking_len = (current_thinking_ids != pad_token_id_val).sum().item()
                answer_len = (answer_ids != pad_token_id_val).sum().item()

                segment_info = {
                    "prompt_start": 0,
                    "prompt_end": prompt_len,
                    "thinking_start": prompt_len,
                    "thinking_end": prompt_len + thinking_len,
                    "answer_start": prompt_len + thinking_len,
                    "answer_end": prompt_len + thinking_len + answer_len,
                    "is_valid_thinking": True,
                    "padding_offset": 0,
                }

                all_complete_sequences.append(complete_ids)
                all_complete_attention_masks.append(complete_attention_mask)
                all_segment_info.append(segment_info)

                # For reward computation
                decoded_answer = self.processing_class.decode(answer_ids, skip_special_tokens=True)
                full_completion_text = current_thinking_text + decoded_answer
                all_full_completions_for_reward.append(full_completion_text)
                expanded_original_prompts_for_reward.append(current_original_prompt_text)
                if _VERBOSE:
                    print(f"{self.accelerator.process_index} Generated completion: {full_completion_text}")

        max_seq_len = max(seq.size(0) for seq in all_complete_sequences)

        final_complete_sequences = []
        final_complete_attention_masks = []

        for i, (seq, attention_mask) in enumerate(zip(all_complete_sequences, all_complete_attention_masks)):
            if seq.size(0) < max_seq_len:
                # Use right padding to remain compatible with Flash Attention
                pad_len = max_seq_len - seq.size(0)
                padding_ids = torch.full((pad_len,), pad_token_id_val, dtype=torch.long, device=device)
                padding_mask = torch.zeros(pad_len, dtype=torch.long, device=device)

                padded_seq = torch.cat([seq, padding_ids], dim=0)  # Right padding
                padded_attention_mask = torch.cat([attention_mask, padding_mask], dim=0)  # Right padding

                # With right padding, segment positions don't need adjustment
                all_segment_info[i]["padding_offset"] = 0  # No offset for right padding
                # Original segment positions remain valid
            else:
                padded_seq = seq
                padded_attention_mask = attention_mask

            final_complete_sequences.append(padded_seq)
            final_complete_attention_masks.append(padded_attention_mask)

        reward_prompts_for_rm = expanded_original_prompts_for_reward

        if is_batch_conversational:
            reward_completions_for_rm = [[{"role": "assistant", "content": c}] for c in all_full_completions_for_reward]
        else:
            reward_completions_for_rm = all_full_completions_for_reward

        rewards_per_func = torch.zeros(len(reward_prompts_for_rm), len(self.reward_funcs), device=device)
        format_rewards = []  # Track rewards for format separately
        accuracy_rewards = []  # Track rewards for accuracy separately

        for i_rm, (reward_func, reward_processing_class_rm) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                # This part needs careful construction of texts for RM
                if is_batch_conversational:  # If original input is conversational
                    texts_for_rm = [p_str + c_str[0]["content"] for p_str, c_str in zip(reward_prompts_for_rm, reward_completions_for_rm)]
                else:  # If original input is plain text
                    texts_for_rm = [p_str + c_str for p_str, c_str in zip(reward_prompts_for_rm, reward_completions_for_rm)]

                reward_inputs = reward_processing_class_rm(
                    texts_for_rm,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False,
                )  # RM might need truncation
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i_rm] = reward_func(**reward_inputs).logits.squeeze(-1)  # Assuming RM outputs (B,1) or (B,)
            else:  # Custom reward function
                # reward_kwargs needs to be expanded K_local*M times
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion", "image", "image_path"]}
                for key_to_expand in reward_kwargs:
                    for k_idx_input in range(K_local):  # Iterate through original K_local inputs
                        # Expand M times for each K_local input
                        reward_kwargs[key_to_expand].extend([inputs[k_idx_input][key_to_expand]] * num_answers_per_thinking)

                # Expand question_types from K elements to K*M elements
                expanded_question_types = []
                for k_idx in range(K_local):
                    # Repeat each question_type M times
                    expanded_question_types.extend([question_types[k_idx]] * num_answers_per_thinking)

                output_reward_func = reward_func(
                    prompts=reward_prompts_for_rm,
                    completions=reward_completions_for_rm,
                    question_types=expanded_question_types,
                    **reward_kwargs,
                )

                if isinstance(output_reward_func, tuple) and len(output_reward_func) == 2:
                    current_rewards_list, debug_metrics = output_reward_func
                    self._add_reward_debug_metrics(debug_metrics, reward_func)
                else:
                    current_rewards_list = output_reward_func

                current_rewards = torch.tensor(current_rewards_list, dtype=torch.float32, device=device)
                rewards_per_func[:, i_rm] = current_rewards

                # Determine reward type based on function name and store separately
                reward_func_name = getattr(reward_func, "__name__", str(reward_func))
                if "format" in reward_func_name.lower():
                    format_rewards.extend(current_rewards_list)
                elif "accuracy" in reward_func_name.lower():
                    accuracy_rewards.extend(current_rewards_list)

        if format_rewards:
            format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.float32, device=device)
            format_rewards = format_rewards_tensor.tolist()
        if accuracy_rewards:
            accuracy_rewards_tensor = torch.tensor(accuracy_rewards, dtype=torch.float32, device=device)
            accuracy_rewards = accuracy_rewards_tensor.tolist()

        # --- Process rewards for thinking and answer advantages (on current device) ---
        local_rewards_flat = rewards_per_func.sum(dim=1)  # (K_local*M) # This is now the primary source for rewards
        local_rewards_grouped = local_rewards_flat.view(K_local, num_answers_per_thinking)  # (K_local, M)

        # Gather rewards from all processes for global statistics
        gathered_rewards_flat = self.accelerator.gather(local_rewards_flat)  # (Total_K * M across all processes)
        # Compute global statistics
        global_rewards_mean = gathered_rewards_flat.mean()
        global_rewards_std = gathered_rewards_flat.std().clamp(min=1e-4)

        # New answer advantage calculation logic with global normalization - use all samples
        answer_advantages_flat = torch.zeros_like(local_rewards_flat)  # Initialize all to 0

        # Use all samples instead of selecting top/bottom
        # Calculate advantages for all answers using global statistics
        answer_advantages_flat = (local_rewards_flat - global_rewards_mean) / global_rewards_std

        # All samples are selected for training
        selected_answer_indices = list(range(len(local_rewards_flat)))

        # Thinking advantages with global normalization
        thinking_values_local = local_rewards_grouped.mean(dim=1)  # (K_local,)

        # Gather thinking values for global statistics
        gathered_thinking_values = self.accelerator.gather(thinking_values_local)  # (Total_K across all processes)
        thinking_advantages_local = torch.zeros_like(thinking_values_local)
        negative_advantage_for_invalid_think = -1.0

        if any(thinking_validity):  # Avoid division by zero if no valid thinks
            # Use global statistics for thinking normalization
            global_thinking_mean = gathered_thinking_values.mean()
            global_thinking_std = gathered_thinking_values.std().clamp(min=1e-4)

            # Normalize using global statistics
            thinking_advantages_local = (thinking_values_local - global_thinking_mean)

            # Log global metrics
            self._metrics["value_mean"].append(global_thinking_mean.item())
            self._metrics["value_std"].append(global_thinking_std.item())
        else:
            # If there are no valid thinking processes, all thinking processes are invalid
            thinking_advantages_local[~thinking_validity_tensor] = negative_advantage_for_invalid_think

        # answer_advantages_flat = answer_advantages_grouped.view(-1)  # (K_local*M)

        # --- Log Probabilities for Old Policy and Reference Model (computed in compute_loss) ---
        raw_old_logps = None
        raw_ref_logps = None

        # Prepare multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs_expanded = {}  # K*M
        for kw in multimodal_keywords:
            if kw in prompt_inputs_for_thinking:
                original_mm_input = prompt_inputs_for_thinking[kw]

                # First reshape to (K_local, patches_per_image, feature_dim)
                K_local = len(thinking_texts)
                patches_per_image = original_mm_input.shape[0] // K_local
                feature_dim = original_mm_input.shape[1]

                reshaped_mm_input = original_mm_input.view(K_local, patches_per_image, feature_dim)

                # Repeat each image block M times along the batch dimension
                expanded_mm_input = reshaped_mm_input.repeat_interleave(num_answers_per_thinking, dim=0)

                # Flatten back to (K_local*M*patches_per_image, feature_dim)
                multimodal_inputs_expanded[kw] = expanded_mm_input.view(-1, feature_dim)
            else:
                multimodal_inputs_expanded[kw] = None

        # Compute old and ref logps if needed
        with torch.no_grad():
            if self.num_iterations > 1:
                complete_sequences_tensor = torch.stack(final_complete_sequences, dim=0)  # Convert list to tensor (K_local*M, max_seq_len)
                complete_attention_masks_tensor = torch.stack(final_complete_attention_masks, dim=0)  # Convert list to tensor (K_local*M, max_seq_len)
                raw_old_logps = self._get_per_token_logps(model, complete_sequences_tensor, complete_attention_masks_tensor, **multimodal_inputs_expanded)
            else:
                raw_old_logps = None

            if self.beta > 0:
                complete_sequences_tensor = torch.stack(final_complete_sequences, dim=0)  # Convert list to tensor (K_local*M, max_seq_len)
                complete_attention_masks_tensor = torch.stack(final_complete_attention_masks, dim=0)  # Convert list to tensor (K_local*M, max_seq_len)
                target_model_for_ref = self.ref_model if self.ref_model is not None else model
                with (self.accelerator.unwrap_model(model).disable_adapter() if self.ref_model is None else torch.no_grad()):
                    raw_ref_logps = self._get_per_token_logps(
                        target_model_for_ref,
                        complete_sequences_tensor,
                        complete_attention_masks_tensor,
                        **multimodal_inputs_expanded,
                    )
            else:
                raw_ref_logps = None

        # --- Logging Metrics ---
        # Gather global rewards for logging (original GRPO style for overall reward metric)
        gathered_rewards_flat = self.accelerator.gather(local_rewards_flat)  # (Total_K * M)

        # Log separate reward components
        if format_rewards:
            gathered_format_rewards = self.accelerator.gather_for_metrics(torch.tensor(format_rewards, device=device))
            self._metrics["format_reward"].append(gathered_format_rewards.mean().item() if gathered_format_rewards.numel() > 0 else 0.0)

        if accuracy_rewards:
            gathered_accuracy_rewards = self.accelerator.gather_for_metrics(torch.tensor(accuracy_rewards, device=device))
            self._metrics["accuracy_reward"].append(gathered_accuracy_rewards.mean().item() if gathered_accuracy_rewards.numel() > 0 else 0.0)

        # Calculate overall mean/std for logging
        global_samples_per_original_prompt = (self.args.num_think_samples * num_answers_per_thinking * (self.accelerator.num_processes if not isinstance(self.train_dataset, IterableDataset) else 1))

        if (gathered_rewards_flat.numel() > 0 and gathered_rewards_flat.numel() % global_samples_per_original_prompt == 0):
            try:
                mean_global_rewards_grouped = gathered_rewards_flat.view(-1, global_samples_per_original_prompt).mean(dim=1)
                std_global_rewards_grouped = gathered_rewards_flat.view(-1, global_samples_per_original_prompt).std(dim=1)
                self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_global_rewards_grouped).mean().item())
            except RuntimeError as e:
                warnings.warn(f"Could not reshape gathered_rewards_flat for logging global std: {e}")
                self._metrics["reward_std"].append(gathered_rewards_flat.std().item())  # Fallback
        elif gathered_rewards_flat.numel() > 0:
            self._metrics["reward_std"].append(gathered_rewards_flat.std().item())  # Fallback
        else:
            self._metrics["reward_std"].append(0.0)

        self._metrics["reward"].append(gathered_rewards_flat.mean().item() if gathered_rewards_flat.numel() > 0 else 0.0)

        # Log answer and thinking lengths using segment_info
        answer_lengths = []
        thinking_lengths = []
        for segment_info in all_segment_info:
            answer_len = segment_info["answer_end"] - segment_info["answer_start"]
            thinking_len = segment_info["thinking_end"] - segment_info["thinking_start"]
            answer_lengths.append(answer_len)
            thinking_lengths.append(thinking_len)

        if answer_lengths:
            answer_lengths_tensor = torch.tensor(answer_lengths, dtype=torch.float32, device=device)
            self._metrics["answer_length_mean"].append(self.accelerator.gather_for_metrics(answer_lengths_tensor).mean().item())
        else:
            self._metrics["answer_length_mean"].append(0.0)

        if thinking_lengths:
            thinking_lengths_tensor = torch.tensor(thinking_lengths, dtype=torch.float32, device=device)
            self._metrics["thinking_length_mean"].append(self.accelerator.gather_for_metrics(thinking_lengths_tensor).mean().item())
        else:
            self._metrics["thinking_length_mean"].append(0.0)

        return {
            "complete_sequences": final_complete_sequences,  # (K_local*M, max_seq_len)
            "complete_attention_masks": final_complete_attention_masks,  # (K_local*M, max_seq_len)
            "segment_info": all_segment_info,  # List of dicts with segment positions for each sequence
            "raw_old_logps": raw_old_logps,  # (K_local*M, max_seq_len-1) or None
            "raw_ref_logps": raw_ref_logps,  # (K_local*M, max_seq_len-1) or None
            "answer_advantages": answer_advantages_flat,  # (K_local*M,)
            "thinking_advantages_local": thinking_advantages_local,  # (K_local,)
            "thinking_validity_K_samples": thinking_validity_tensor,  # (K_local,) boolean
            "multimodal_inputs": multimodal_inputs_expanded,  # K*M
            "num_answers_per_thinking": num_answers_per_thinking,  # scalar
            "answer_indices": selected_answer_indices,
        }

    def compute_loss(self, model, inputs_from_dataloader, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOMATrainer does not support returning outputs")
        # `inputs_from_dataloader` are the K_local original prompts + images
        if self.state.global_step % self.num_iterations == 0:
            with torch.no_grad():
                gpu_id = self.accelerator.process_index
                if _VERBOSE:
                    print(f"[GPU {gpu_id}] len of inputs: {len(inputs_from_dataloader)}")
                    prompts_un = [x["prompt"] for x in inputs_from_dataloader]
                    for idx in range(len(prompts_un)):
                        print(f"[GPU {gpu_id}] first prompt: {prompts_un[idx]}")
                processed_inputs = self._generate_and_score_completions(inputs_from_dataloader, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = processed_inputs
        else:
            processed_inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Extract data from processed_inputs
        complete_sequences = processed_inputs["complete_sequences"]  # (K_local*M, max_seq_len)
        complete_attention_masks = processed_inputs["complete_attention_masks"]  # (K_local*M, max_seq_len)
        all_segment_info = processed_inputs["segment_info"]  # List of segment info dicts
        raw_old_logps = processed_inputs["raw_old_logps"]
        raw_ref_logps = processed_inputs["raw_ref_logps"]
        answer_advantages = processed_inputs["answer_advantages"]  # (K_local*M,)
        thinking_advantages_local = processed_inputs["thinking_advantages_local"]  # (K_local,)
        thinking_validity_K_samples = processed_inputs["thinking_validity_K_samples"]  # (K_local,)
        multimodal_inputs = processed_inputs["multimodal_inputs"]
        num_answers_per_thinking = processed_inputs["num_answers_per_thinking"]
        answer_indices = processed_inputs["answer_indices"]
        # Compute current logps for the complete sequences
        complete_sequences_tensor = torch.stack(complete_sequences, dim=0)  # (K_local*M, max_seq_len)
        complete_attention_masks_tensor = torch.stack(complete_attention_masks, dim=0)  # (K_local*M, max_seq_len)

        current_logps = self._get_per_token_logps(model, complete_sequences_tensor, complete_attention_masks_tensor, **multimodal_inputs)  # (K_local*M, max_seq_len-1)
        # Create thinking and answer masks directly on the full sequence
        batch_size_km, seq_len_minus_1 = current_logps.shape
        thinking_masks_full = torch.zeros_like(current_logps)  # (K_local*M, max_seq_len-1)
        answer_masks_full = torch.zeros_like(current_logps)  # (K_local*M, max_seq_len-1)

        # Fill masks based on segment info
        for i, segment_info in enumerate(all_segment_info):
            # Thinking mask (adjusted for logps shift)
            thinking_start = max(0, segment_info["thinking_start"] - 1)
            thinking_end = max(0, segment_info["thinking_end"] - 1)
            if thinking_end > thinking_start and thinking_end <= seq_len_minus_1:
                thinking_masks_full[i, thinking_start:thinking_end] = 1.0

            # Answer mask (adjusted for logps shift)
            answer_start = max(0, segment_info["answer_start"] - 1)
            answer_end = max(0, segment_info["answer_end"] - 1)
            if answer_end > answer_start and answer_end <= seq_len_minus_1:
                answer_masks_full[i, answer_start:answer_end] = 1.0

        # Get old and ref logps if needed
        if raw_old_logps is not None:
            old_logps = raw_old_logps
        else:
            old_logps = current_logps.detach()

        if raw_ref_logps is not None:
            ref_logps = raw_ref_logps
        else:
            ref_logps = None
        with torch.no_grad():
            self._get_think_and_answer_entropy(
                model,
                complete_sequences_tensor,
                complete_attention_masks_tensor,
                thinking_masks_full,
                answer_masks_full,
                **multimodal_inputs,
            )

        # --- Thinking PPO Loss ---
        think_indices = torch.arange(0, thinking_masks_full.shape[0], device=thinking_masks_full.device, step=num_answers_per_thinking)
        ratio_think = torch.exp(current_logps[think_indices] - old_logps[think_indices])
        clipped_ratio_think = torch.clamp(ratio_think, 1 - self.epsilon_low, 1 + self.epsilon_high)

        thinking_masks_full = thinking_masks_full[think_indices]
        # For thinking: repeat each thinking advantage M times to match the shape
        # thinking_advantages_expanded = thinking_advantages_local.repeat_interleave(num_answers_per_thinking)  # (K_local*M,)
        thinking_advantages_broadcasted = thinking_advantages_local.unsqueeze(1)

        loss1_think = ratio_think * thinking_advantages_broadcasted * thinking_masks_full
        loss2_think = clipped_ratio_think * thinking_advantages_broadcasted * thinking_masks_full
        ppo_loss_think_per_token = -torch.min(loss1_think, loss2_think)
        ppo_loss_think = ppo_loss_think_per_token.sum(dim=1) / thinking_masks_full.sum(dim=1).clamp(min=1e-5)
        ppo_loss_think_mean = ppo_loss_think.mean()
        # --- Answer PPO Loss ---

        ratio_answer = torch.exp(current_logps[answer_indices] - old_logps[answer_indices])
        clipped_ratio_answer = torch.clamp(ratio_answer, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Broadcast answer advantages: (K_local*M,) -> (K_local*M, 1) -> (K_local*M, max_seq_len-1)
        answer_advantages_broadcasted = answer_advantages[answer_indices].unsqueeze(1)
        answer_masks_full = answer_masks_full[answer_indices]

        loss1_answer = ratio_answer * answer_advantages_broadcasted * answer_masks_full
        loss2_answer = clipped_ratio_answer * answer_advantages_broadcasted * answer_masks_full
        ppo_loss_answer_per_token = -torch.min(loss1_answer, loss2_answer)
        ppo_loss_answer = ppo_loss_answer_per_token.sum(dim=1) / answer_masks_full.sum(dim=1).clamp(min=1e-5)
        ppo_loss_answer_mean = ppo_loss_answer.mean()

        final_ppo_loss_mean = ppo_loss_think_mean + ppo_loss_answer_mean
        inconsistent_count = self.get_inconsistent_adv(thinking_advantages_local, answer_advantages)

        self._metrics["sign_inconsistent_count"].append(self.accelerator.gather_for_metrics(torch.tensor(inconsistent_count, device=answer_advantages.device)).sum().item())
        total_answers = answer_advantages.shape[0]
        inconsistent_ratio = inconsistent_count / total_answers if total_answers > 0 else 0.0
        self._metrics["sign_inconsistent_ratio"].append(self.accelerator.gather_for_metrics(torch.tensor(inconsistent_ratio, device=answer_advantages.device)).mean().item())

        if _VERBOSE:
            print(f"[GPU {self.accelerator.process_index}] Sign inconsistent count: {inconsistent_count}, ratio: {inconsistent_ratio:.4f}")

        # --- KL Penalty ---
        total_kl_penalty = 0.0
        if self.beta > 0 and ref_logps is not None:
            # Thinking KL
            if thinking_masks_full.sum() > 0:
                thinking_advantages_expanded = thinking_advantages_local.repeat_interleave(num_answers_per_thinking)  # (K_local*M,)
                kl_div_think = (torch.exp(ref_logps[think_indices] - current_logps[think_indices]) - (ref_logps[think_indices] - current_logps[think_indices]) - 1) * thinking_masks_full
                thinking_token_counts_kl = thinking_masks_full.sum(dim=1).clamp(min=1e-5)
                mean_kl_think = (kl_div_think.sum(dim=1) / thinking_token_counts_kl).mean()
            else:
                mean_kl_think = torch.tensor(0.0, device=current_logps.device)

            # Answer KL
            if answer_masks_full.sum() > 0:
                kl_div_answer = (torch.exp(ref_logps[answer_indices] - current_logps[answer_indices]) - (ref_logps[answer_indices] - current_logps[answer_indices]) - 1) * answer_masks_full
                answer_token_counts_kl = answer_masks_full.sum(dim=1).clamp(min=1e-5)
                mean_kl_answer = (kl_div_answer.sum(dim=1) / answer_token_counts_kl).mean()
            else:
                mean_kl_answer = torch.tensor(0.0, device=current_logps.device)

            total_kl_penalty = self.beta * (mean_kl_think + mean_kl_answer)

            # Log metrics
            self._metrics["kl_think"].append(self.accelerator.gather_for_metrics(mean_kl_think).mean().item())
            self._metrics["kl_answer"].append(self.accelerator.gather_for_metrics(mean_kl_answer).mean().item())
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl_think + mean_kl_answer).mean().item())
        loss = final_ppo_loss_mean + total_kl_penalty

        # Log clip ratios
        if thinking_masks_full.sum() > 0:
            is_clipped_think = (loss1_think.abs() > loss2_think.abs()).float() * thinking_masks_full
            clip_ratio_think = is_clipped_think.sum() / thinking_masks_full.sum().clamp(min=1e-5)
        else:
            clip_ratio_think = torch.tensor(0.0, device=answer_masks_full.device)
        self._metrics["clip_ratio_think"].append(self.accelerator.gather_for_metrics(clip_ratio_think).mean().item())
        if answer_masks_full.sum() > 0:
            is_clipped_answer = (loss1_answer.abs() > loss2_answer.abs()).float() * answer_masks_full
            clip_ratio_answer = is_clipped_answer.sum() / answer_masks_full.sum().clamp(min=1e-5)
        else:
            clip_ratio_answer = torch.tensor(0.0, device=answer_masks_full.device)
        self._metrics["clip_ratio_answer"].append(self.accelerator.gather_for_metrics(clip_ratio_answer).mean().item())
        final_ppo_loss_think = ppo_loss_think_per_token.mean()
        final_ppo_loss_answer = ppo_loss_answer_per_token.mean()
        self._metrics["loss_ppo_think"].append(self.accelerator.gather_for_metrics(final_ppo_loss_think).mean().item())
        self._metrics["loss_ppo_answer"].append(self.accelerator.gather_for_metrics(final_ppo_loss_answer).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
            @article{grpo-ma,
                title={GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training},
                }
            """)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO-MA",
            trainer_citation=citation,
            paper_title="Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO",
            paper_id="2509.24494",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _add_reward_debug_metrics(self, debug_metrics, reward_func):
        """Add reward function debug metrics to ``self._metrics``."""

        if not self.accelerator.is_main_process:
            return

        reward_func_name = getattr(reward_func, "__name__", str(reward_func))

        for metric_key, metric_values in debug_metrics.items():
            if metric_key == "error" or isinstance(metric_values, str):
                continue
            full_key = f"reward_{reward_func_name}_{metric_key}"

            if isinstance(metric_values, list):
                if metric_values:
                    tensor_values = torch.tensor(metric_values, dtype=torch.float32, device=self.accelerator.device)
                    valid_values = tensor_values[tensor_values != -1]

                    if valid_values.numel() > 0:
                        self._metrics[full_key + "_mean"].append(valid_values.mean().item())
            else:
                if isinstance(metric_values, (int, float)):
                    if full_key not in self._metrics:
                        self._metrics[full_key] = []
                    tensor_value = torch.tensor(metric_values, dtype=torch.float32, device=self.accelerator.device)
                    valid_values = tensor_value[tensor_value != -1]

                    if valid_values.numel() > 0:
                        self._metrics[full_key].append(valid_values.mean().item())

    def get_inconsistent_adv(self, think_adv, answer_adv):
        """
        Count how many answer advantages have a sign opposite to their corresponding think advantage.

        Args:
            think_adv: Thinking advantages with shape (K,).
            answer_adv: Answer advantages with shape (K*M,).

        Returns:
            int: Number of sign mismatches.
        """
        K_local = think_adv.shape[0]
        num_answers_per_thinking = self.args.num_answers_per_thinking

        answer_adv_reshaped = answer_adv.view(K_local, num_answers_per_thinking)
        think_adv_expanded = think_adv.unsqueeze(1).expand(-1, num_answers_per_thinking)

        answer_signs = torch.sign(answer_adv_reshaped)
        thinking_signs = torch.sign(think_adv_expanded)

        sign_inconsistent_mask = (answer_signs * thinking_signs) < 0
        inconsistent_count = sign_inconsistent_mask.sum().item()

        return inconsistent_count

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = self._get_train_sampler(train_dataset)
        num_think_samples = getattr(self.args, "num_think_samples", 8)
        return DataLoader(
            train_dataset,
            batch_size=num_think_samples,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )

    def _get_train_sampler(self, train_dataset) -> Sampler:
        """Return a sampler that ensures synchronized sampling across GPUs with PPO iterations."""
        num_think_samples = getattr(self.args, "num_think_samples", 8)

        print("Using synchronized sampling with PPO iterations:")
        print(f"- Each GPU will process {num_think_samples} copies of the same sample per PPO iteration")
        print(f"- Each sample will undergo {self.num_iterations} PPO iterations")
        print(f"- Total training samples: {len(self.train_dataset)}")
        print(f"- Total training steps: {len(self.train_dataset) * self.num_iterations}")

        return SynchronizedSampler(
            data_source=self.train_dataset,
            repeat_count=num_think_samples,
            num_iterations=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Return the evaluation sampler."""
        num_think_samples = getattr(self.args, "num_think_samples", 8)

        return SynchronizedSampler(
            data_source=eval_dataset,
            repeat_count=num_think_samples,
            num_iterations=1,
            seed=self.args.seed,
        )
