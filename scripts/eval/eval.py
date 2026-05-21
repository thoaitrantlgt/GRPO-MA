"""Multi-task evaluation driver for GRPO-MA.

Loads a (base or LoRA-finetuned) Qwen2.5-VL checkpoint, runs it on a test set
that follows the same JSON schema as training metadata, parses each model
output's ``<answer>...</answer>``, and dispatches scoring to the task's
``compute_metrics`` function.

Usage
-----
::

    python scripts/eval/eval.py \\
        --model_name_or_path pretrained_weights/Qwen2.5-VL-3B-Instruct \\
        --adapter_path output/<run>/checkpoint-XXXX \\
        --dataset metadata/grpo_sharerobot_trajectory_test.json \\
        --image_root data/ShareRobot/trajectory/images \\
        --output_dir output/eval/<run>

The driver is task-agnostic; each task module owns its metric implementation.
Without ``--adapter_path`` it evaluates the base model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from model.task_configs import get_question_template, normalize_task_type  # noqa: E402
from task import get_task_registry  # noqa: E402


# -----------------------------------------------------------------------------
# Image / prompt helpers (mirrors dataset/dataset_grpo.py so eval prompts match
# training prompts exactly).
# -----------------------------------------------------------------------------

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280) -> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
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


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_answer(text: str) -> Optional[str]:
    match = _ANSWER_RE.search(text)
    return match.group(1).strip() if match else None


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def _load_json(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: str, image_root_override: Optional[str]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p}")

    items: List[Dict[str, Any]] = []
    if p.suffix in (".json", ".jsonl"):
        for item in _load_json(p):
            item.setdefault("data_root", image_root_override or "")
            items.append(item)
        return items

    if p.suffix in (".yaml", ".yml"):
        with open(p, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        for entry in manifest.get("datasets", []):
            json_path = entry["json_path"]
            data_root = image_root_override or entry.get("data_root", "")
            for item in _load_json(Path(json_path)):
                item.setdefault("data_root", data_root)
                items.append(item)
        return items

    raise ValueError(f"Unsupported dataset format: {p.suffix}")


def load_image(item: Dict[str, Any], default_root: Optional[str]) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    image_field = item.get("image")
    if image_field is None:
        return None, None, None
    if isinstance(image_field, list):
        image_field = image_field[0]
    root = item.get("data_root") or default_root or ""
    path = os.path.join(root, image_field)
    if not os.path.exists(path):
        return None, None, None
    img = Image.open(path).convert("RGB")
    return img, img.size, None  # original_size = (W, H)


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def build_prompt(item: Dict[str, Any], template: str) -> List[Dict[str, Any]]:
    question_text = template.format(Question=item["question"])
    content: List[Dict[str, Any]] = []
    if "image" in item:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question_text})
    return [{"role": "user", "content": content}]


def _resize_for_qwen(image: Image.Image, max_pixels: int, min_pixels: int) -> Tuple[Image.Image, Tuple[int, int]]:
    w, h = image.size
    new_h, new_w = smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS), (new_w, new_h)


def load_model(model_name_or_path: str, adapter_path: Optional[str], torch_dtype: str, attn_implementation: Optional[str]):
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    kwargs: Dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        try:
            model = model.merge_and_unload()
        except Exception:  # noqa: BLE001
            pass
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


@torch.inference_mode()
def generate_batch(model, processor, conversations: List[List[Dict[str, Any]]], images: List[Optional[Image.Image]], gen_kwargs: Dict[str, Any]) -> List[str]:
    prompts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    valid_images = [img for img in images if img is not None]
    inputs = processor(
        text=prompts,
        images=valid_images if valid_images else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO-MA multi-task evaluation driver")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path or HF id (e.g. Qwen/Qwen2.5-VL-3B-Instruct).")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter checkpoint to apply on top of the base model.")
    parser.add_argument("--dataset", required=True, help="Test dataset: JSON, JSONL, or YAML manifest.")
    parser.add_argument("--image_root", default=None, help="Override `data_root` for image paths in the dataset.")
    parser.add_argument("--output_dir", required=True, help="Where to write predictions.jsonl and metrics.json.")
    parser.add_argument("--task_type", default=None, help="Override the task_type for every record (otherwise uses `question_type` field).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 enables greedy decoding (do_sample=False).")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    parser.add_argument("--min_pixels", type=int, default=56 * 56)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to the first N samples (debug).")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    items = load_dataset(args.dataset, args.image_root)
    if args.max_samples is not None:
        items = items[: args.max_samples]
    print(f"Loaded {len(items)} samples from {args.dataset}")

    # Load model
    print(f"Loading model from {args.model_name_or_path}" + (f" with adapter {args.adapter_path}" if args.adapter_path else ""))
    model, processor = load_model(args.model_name_or_path, args.adapter_path, args.torch_dtype, args.attn_implementation)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.max_pixels = args.max_pixels
        image_processor.min_pixels = args.min_pixels

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_completion_length,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    task_registry = get_task_registry()

    # Run inference
    records: List[Dict[str, Any]] = []
    pbar = tqdm(range(0, len(items), args.batch_size), desc="Generating", unit="batch")
    for start in pbar:
        batch_items = items[start: start + args.batch_size]
        conversations = []
        images = []
        for item in batch_items:
            question_type = args.task_type or item.get("question_type", "default")
            template = get_question_template(normalize_task_type(question_type)) or "{Question}"
            conv = build_prompt(item, template)
            conversations.append(conv)

            img, _, _ = load_image(item, args.image_root)
            if img is not None:
                img, _ = _resize_for_qwen(img, args.max_pixels, args.min_pixels)
            images.append(img)

        outputs = generate_batch(model, processor, conversations, images, gen_kwargs)

        for item, output in zip(batch_items, outputs):
            parsed = parse_answer(output)
            records.append({
                "id": item.get("id", str(start)),
                "question_type": args.task_type or item.get("question_type", "default"),
                "question": item.get("question"),
                "image": item.get("image"),
                "ground_truth": item.get("answer"),
                "prediction": parsed,
                "raw_output": output,
                "meta": {k: v for k, v in item.items() if k not in {"question", "answer", "image", "data_root"}},
            })

    # Save predictions
    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} predictions to {pred_path}")

    # Compute metrics, grouped by task_type
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[normalize_task_type(r["question_type"])].append(r)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for task_type, task_records in grouped.items():
        metrics_fn = task_registry.get_metrics_fn(task_type)
        if metrics_fn is None:
            print(f"Warning: task '{task_type}' has no compute_metrics; skipping.")
            all_metrics[task_type] = {"num_samples": len(task_records), "warning": "no compute_metrics registered"}
            continue
        all_metrics[task_type] = metrics_fn(task_records)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "metrics": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote metrics to {metrics_path}")

    # Pretty-print
    print("\n=== Evaluation Results ===")
    for task_type, m in all_metrics.items():
        print(f"\n[{task_type}]")
        for key, value in m.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
