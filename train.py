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

import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

# `sys.path` must be set before any first-party imports so they resolve from the repo root.
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from trl import TrlParser, get_peft_config

from config.model_config import GRPOModelConfig
from config.training_config import GRPOConfig, GRPOScriptArguments
from dataset.dataset_grpo import MultiMediaGRPODataset
from model.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)
from model.qwen_module import Qwen2VLModule
from trainer.grpo_ma_trainer import GRPOMATrainer


def get_vlm_module(model_name_or_path):
    """Get the corresponding VLM module based on model name"""
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    else:
        print(f"Warning: Unsupported model {model_name_or_path}, using Qwen2VL as default")
        return Qwen2VLModule


def main(script_args, training_args, model_args):
    # Set seed before any data loading
    if training_args.seed is not None:
        random.seed(training_args.seed)
        np.random.seed(training_args.seed)
        torch.manual_seed(training_args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Save all arguments to JSON file
    args_dict = {
        "script_args": asdict(script_args),
        "training_args": asdict(training_args),
        "model_args": asdict(model_args),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to JSON file
    args_file_path = os.path.join(training_args.output_dir, "training_args.json")
    with open(args_file_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"Training arguments saved to: {args_file_path}")

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print(f"Using VLM module: {vlm_module_cls.__name__}")
    # Load the reward functions - use improved reward function system
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward,
        "format": vlm_module_cls.format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(f"Loaded reward functions: {script_args.reward_funcs}")

    # Validate if reward functions are available
    for func_name in script_args.reward_funcs:
        if func_name in reward_funcs_registry:
            print(f"✓ Reward function '{func_name}' is available")
        else:
            print(f"✗ Warning: Reward function '{func_name}' not found")
            print(f"Available functions: {list(reward_funcs_registry.keys())}")

    # Set learning rate scheduler parameters
    training_args.lr_scheduler_type = "cosine"  # Use cosine annealing
    print(f"Learning rate scheduler: {training_args.lr_scheduler_type}")
    print(f"Warmup ratio: {training_args.warmup_ratio}")
    print(f"Learning rate: {training_args.learning_rate}")

    # Load the dataset
    print(f"Loading dataset from: {script_args.dataset_name}")

    dataset = MultiMediaGRPODataset(script_args.dataset_name, script_args)
    print(f"Loaded {len(dataset)} samples")

    if training_args.train_cls == "GRPO-MA":
        trainer_cls = GRPOMATrainer
    print(f"Using trainer: {training_args.train_cls}")

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Apply monkey patches
    monkey_patch_qwen2_5vl_flash_attn()

    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("Zero3 is used, applying qwen2_5vl forward and torch load monkey patches")
        monkey_patch_qwen2_5vl_forward()
        monkey_patch_torch_load()

    main(script_args, training_args, model_args)
