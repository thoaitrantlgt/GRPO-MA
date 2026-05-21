import json
import math
import os
import random
import re
from typing import Optional, List, Dict, Any, Union

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

from config.training_config import GRPOScriptArguments
from model.qwen_module import Qwen2VLModule
from model.task_configs import normalize_task_type
from task.task_loader import get_task_registry

# Optional scipy import for .mat files
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    def loadmat(filename):
        raise ImportError("scipy is not available")


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280) -> tuple:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class MultiMediaGRPODataset(Dataset):
    """
    Dataset class for GRPO tasks that supports multiple images and/or videos.
    Handles lazy loading, smart resizing, and various task types.
    """

    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(MultiMediaGRPODataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.task_registry = get_task_registry()

        # Set random seed for reproducibility across processes
        seed = getattr(script_args, 'seed', 42)
        random.seed(seed)

        # Load data from YAML configuration
        self._load_data_from_yaml(data_path)

        # Final shuffle with consistent seed
        random.shuffle(self.list_data_dict)

        # Print dataset statistics
        self._print_dataset_info()

    def _load_data_from_yaml(self, data_path: str):
        """Load and process data from YAML configuration file."""
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Only YAML configuration files are supported, got: {data_path}")

        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets", [])

        for data in datasets:
            json_path = data.get("json_path")
            sampling_strategy = data.get("sampling_strategy", "all")
            data_root = data.get("data_root", "")
            data_modality = data.get("data_modality", "image")

            # Load JSON data
            cur_data_dict = self._load_json_file(json_path)

            # Apply sampling strategy
            cur_data_dict = self._apply_sampling_strategy(cur_data_dict, sampling_strategy)

            # Add metadata to each item
            for item in cur_data_dict:
                if 'data_root' not in item:
                    item["data_root"] = data_root
                item["data_modality"] = data_modality

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            self.list_data_dict.extend(cur_data_dict)

    def _load_json_file(self, json_path: str) -> List[Dict]:
        """Load data from JSON or JSONL file."""
        if json_path.endswith(".jsonl"):
            data_dict = []
            with open(json_path, "r") as json_file:
                for line in json_file:
                    data_dict.append(json.loads(line.strip()))
        elif json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                data_dict = json.load(json_file)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")
        return data_dict

    def _apply_sampling_strategy(self, data_dict: List[Dict], sampling_strategy: str) -> List[Dict]:
        """Apply sampling strategy to data."""
        if ":" not in sampling_strategy:
            return data_dict

        strategy, sampling_number = sampling_strategy.split(":")

        # Parse sampling number (support percentage or absolute number)
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(data_dict) / 100)
        else:
            sampling_number = int(sampling_number)

        # Apply sampling strategy
        if strategy == "first":
            return data_dict[:sampling_number]
        elif strategy == "end":
            return data_dict[-sampling_number:]
        elif strategy == "random":
            random.shuffle(data_dict)
            return data_dict[:sampling_number]
        else:
            return data_dict

    def _print_dataset_info(self):
        """Print dataset statistics."""
        question_types = set()
        for example in self.list_data_dict:
            if "question_type" in example:
                question_types.add(example["question_type"])

        print(f"\n{'='*50}")
        print(f"Dataset loaded: {len(self.list_data_dict)} samples")
        print(f"Question types: {sorted(question_types)}")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        example = self.list_data_dict[i].copy()

        # Get question template based on task type
        question_type = example.get("question_type", "default")
        QUESTION_TEMPLATE = Qwen2VLModule.get_question_template(question_type)

        # Get image/video root
        image_root = example.get('data_root') or example.get('image_root') or self.script_args.image_root

        # Process media (images and videos)
        images, videos = self._process_media(example, image_root)

        # Process answer with smart resize
        processed_answer = self._process_answer(example, images)

        # Create conversation prompt
        prompt = self._create_conversation_prompt(example, QUESTION_TEMPLATE, images, videos)

        # Build item dictionary
        item = self._build_item_dict(example, images, videos, processed_answer, prompt, i, question_type)

        return item

    def _process_media(self, example: Dict, image_root: str) -> tuple:
        """Process images and videos from the example."""
        images = []
        videos = []

        # Process images
        if "image" in example:
            image_paths = example["image"] if isinstance(example["image"], list) else [example["image"]]

            for img_path in image_paths:
                full_image_path = os.path.join(image_root, img_path)

                # Handle missing images
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image {full_image_path} not found, using fallback")
                    continue

                # Load and resize image
                image = Image.open(full_image_path).convert("RGB")
                original_width, original_height = image.size

                # Apply smart resize
                target_height, target_width = smart_resize(height=original_height,
                                                           width=original_width,
                                                           factor=getattr(self.script_args, 'resize_factor', 28),
                                                           min_pixels=getattr(self.script_args, 'min_pixels', 56 * 56),
                                                           max_pixels=getattr(self.script_args, 'max_pixels', 14 * 14 * 4 * 1280))

                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                images.append({'image': image, 'original_size': (original_width, original_height), 'resized_size': (target_width, target_height)})

        # Process videos
        if "video" in example:
            video_paths = example["video"] if isinstance(example["video"], list) else [example["video"]]

            for video_path in video_paths:
                full_video_path = os.path.join(image_root, video_path)

                if not os.path.exists(full_video_path):
                    print(f"Warning: Video {full_video_path} not found")
                    continue

                videos.append(full_video_path)

        return images, videos

    def _process_answer(self, example: Dict, images: List[Dict]) -> Any:
        """Process answers via task-specific processors registered with the task registry."""
        answer = example.get("answer", "")
        question_type = example.get("question_type", "")
        normalized_type = normalize_task_type(question_type)

        processor = None
        if getattr(self, "task_registry", None):
            processor = self.task_registry.get_answer_processor(normalized_type)

        original_size = None
        target_size = None

        if images:
            first_image = images[0]
            original_size = first_image.get('original_size')
            target_size = first_image.get('resized_size')

        if processor:
            processed_answer = processor(
                answer,
                example=example,
                original_size=original_size,
                target_size=target_size,
                images=images,
            )

            if processed_answer is not None:
                return processed_answer

        return answer

    def _create_conversation_prompt(self, example: Dict, template: str, images: List[Dict], videos: List[str]) -> List[Dict]:
        """Create conversation prompt with proper media placeholders."""
        question_text = template.format(Question=example["question"])

        # Add default placeholder if none exists
        if "<image>" not in question_text and "<video>" not in question_text:
            question_text = "<image>\n" + question_text

        # Parse placeholders and build content
        content = []
        current_pos = 0
        image_count = 0
        video_count = 0

        # Find all placeholders
        placeholders = []
        for match in re.finditer(r'<(image|video)>', question_text):
            placeholders.append({'type': match.group(1), 'start': match.start(), 'end': match.end()})

        # Build content with text and media
        for placeholder in placeholders:
            # Add text before placeholder
            if placeholder['start'] > current_pos:
                text_part = question_text[current_pos:placeholder['start']]
                if text_part.strip():
                    content.append({"type": "text", "text": text_part})

            # Add media
            if placeholder['type'] == 'image' and image_count < len(images):
                content.append({"type": "image"})
                image_count += 1
            elif placeholder['type'] == 'video' and video_count < len(videos):
                content.append({"type": "video"})
                video_count += 1

            current_pos = placeholder['end']

        # Add remaining text
        if current_pos < len(question_text):
            remaining_text = question_text[current_pos:]
            if remaining_text.strip():
                content.append({"type": "text", "text": remaining_text})

        return [{
            "role": "user",
            "content": content,
        }]

    def _build_item_dict(self, example: Dict, images: List[Dict], videos: List[str], processed_answer: Any, prompt: List[Dict], idx: int, question_type: str) -> Dict[str, Any]:
        """Build the final item dictionary."""
        item = {
            "problem": example["question"],
            "solution": str(processed_answer),
            "prompt": prompt,
            "idx": str(idx),
            "question_type": question_type,
        }

        # Add image information
        if images:
            if len(images) == 1:
                item["image"] = images[0]['image']
                item["image_height"] = images[0]['resized_size'][1]
                item["image_width"] = images[0]['resized_size'][0]
            else:
                item["image"] = [img['image'] for img in images]
                item["image_count"] = len(images)
                item["image_height"] = images[0]['resized_size'][1]
                item["image_width"] = images[0]['resized_size'][0]

        # Add video information
        if videos:
            item["video"] = videos if len(videos) > 1 else videos[0]
            item["video_count"] = len(videos)

        # Preserve additional metadata
        metadata_keys = ["data_root", "data_modality", "affordance_label_id", "answer_path"]
        for key in metadata_keys:
            if key in example:
                item[key] = example[key]

        return item
