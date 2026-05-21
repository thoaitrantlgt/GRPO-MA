import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import PIL
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from trl.data_utils import maybe_apply_chat_template

try:
    from scipy.io import loadmat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    def loadmat(filename):
        raise ImportError("scipy is not available")


from transformers import AutoModelForCausalLM, AutoTokenizer

from model.task_configs import (
    get_question_template,
    normalize_task_type,
)
from model.vlm_module import VLMBaseModule

from .reward_func import accuracy_reward, format_reward


class Qwen2VLModule(VLMBaseModule):

    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):

        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2_5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            print(f"Model ID {model_id} does not match any known Qwen2-VL models. Defaulting to AutoModelForCausalLM.")
            model_cls = AutoModelForCausalLM
        return model_cls

    def post_model_init(self, model, processing_class):
        pass

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):
        return ["visual"]

    def get_custom_multimodal_keywords(self):
        return ["pixel_values", "image_grid_thw"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [("image_processor", "max_pixels"), ("image_processor", "min_pixels")]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        data_modality="image",
    ):
        if data_modality[0] == "image":
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
        elif data_modality[0] == "text":
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
        return prompt_inputs

    @staticmethod
    def get_question_template(task_type: str):

        normalized_task_type = normalize_task_type(task_type)

        return get_question_template(normalized_task_type)

    @staticmethod
    def accuracy_reward(completions, solution, question_types, **kwargs):
        return accuracy_reward(completions, solution, question_types, **kwargs)

    @staticmethod
    def format_reward(completions, solution, question_types, **kwargs):
        return format_reward(completions, solution, question_types, **kwargs)
