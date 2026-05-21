"""Template task module — copy this file when adding a new task.

Quick start
-----------
1. ``cp task/TEMPLATE.py task/<your_task>.py``
2. Edit ``TASK_CONFIG["task_type"]`` to a unique lowercase-with-hyphens
   identifier. The same identifier must appear in your metadata JSON under
   ``question_type``.
3. Fill in ``grpo_template`` with the prompt shown to the model. The dataset
   loader replaces ``{Question}`` with the question text.
4. Implement ``format_reward`` (cheap structural check) and ``accuracy_reward``
   (real metric). Either may return either ``float`` or ``(float, dict)`` — the
   dict is logged as auxiliary metrics during training.
5. (Optional) Implement ``process_answer`` if your ground-truth answer is in
   pixel coordinates and needs to be rescaled when the image is resized. See
   ``task/processing_utils.py`` for ready-made helpers.
6. (Optional) Implement ``compute_metrics`` so the shared evaluation driver
   (``scripts/eval/eval.py``) can score a checkpoint on this task.
7. Leave ``register`` as-is — the task loader calls it automatically.

The module is discovered automatically on next import; no global registry
edits are needed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Task configuration
# =============================================================================

TASK_CONFIG: Dict[str, Any] = {
    # Unique identifier used everywhere (question_type, registries, logs).
    "task_type": "my-task",
    "description": "One-line description of the task.",
    "input_format": "image + instruction",  # image / text / video + instruction
    "output_format": "<think>reasoning</think><answer>...</answer>",
    # The prompt template. {Question} is replaced with the metadata question.
    "grpo_template": (
        "{Question} First output the thinking process in <think> </think> "
        "tags and then output the final answer in <answer> </answer> tags."
    ),
    "evaluation_metrics": ["format_compliance", "accuracy"],
    "format_requirements": {
        "requires_think_tag": True,
        "requires_answer_tag": True,
        "answer_format": "string",
    },
}


# =============================================================================
# Reward functions
# =============================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def format_reward(completion, sol, **kwargs) -> float:
    """Return 1.0 if the response has the expected ``<think>`` / ``<answer>`` shape."""
    content = completion[0]["content"]
    has_think = _THINK_RE.search(content) is not None
    has_answer = _ANSWER_RE.search(content) is not None
    return 1.0 if (has_think and has_answer) else 0.0


def accuracy_reward(completion, sol, **kwargs):
    """Score the answer content against the ground truth ``sol``.

    Return either ``float`` or ``(float, dict)``. The optional dict is logged
    as auxiliary metrics during training (e.g. component-wise errors).
    """
    content = completion[0]["content"]
    match = _ANSWER_RE.search(content)
    if not match:
        return 0.0
    predicted = match.group(1).strip()
    # TODO: replace with task-specific scoring (string match, IoU, distance, ...).
    is_correct = predicted == str(sol).strip()
    return 1.0 if is_correct else 0.0


# =============================================================================
# Optional: answer pre-processing (only needed for pixel-space answers)
# =============================================================================

def process_answer(
    answer: Any,
    *,
    original_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    **_: Any,
):
    """Rescale a pixel-space answer when the input image is resized.

    See ``task/processing_utils.py`` for ``scale_trajectory`` / ``scale_bbox_xyxy``
    / ``scale_points`` if your answer is a list of coordinates.
    """
    return answer


# =============================================================================
# Optional: evaluation metrics (used by scripts/eval/eval.py)
# =============================================================================

def compute_metrics(predictions: List[Dict[str, Any]], **kwargs) -> Dict[str, float]:
    """Aggregate task-specific metrics over a list of prediction records.

    Each record is ``{"prediction": str, "ground_truth": Any, "raw": str, "meta": dict}``.
    Return a flat ``dict`` of scalar metrics; the eval driver will write it to
    ``metrics.json`` and print it to stdout.
    """
    if not predictions:
        return {"num_samples": 0}
    correct = sum(1 for p in predictions if p.get("prediction") == str(p.get("ground_truth")).strip())
    return {
        "num_samples": len(predictions),
        "accuracy": correct / len(predictions),
    }


# =============================================================================
# Registration (do not edit)
# =============================================================================

def register(accuracy_registry, format_registry, answer_registry=None):
    name = TASK_CONFIG["task_type"]
    accuracy_registry[name] = accuracy_reward
    format_registry[name] = format_reward
    if answer_registry is not None:
        answer_registry[name] = process_answer
