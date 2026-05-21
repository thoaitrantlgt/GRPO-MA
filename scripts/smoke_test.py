"""Quick smoke test: verifies that the task registry, reward functions, and
metric routing are all wired up without needing a GPU or any model weights.

Usage::

    python scripts/smoke_test.py

Exits non-zero on the first failure. Designed for CI as well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    failures = []

    # --- Task registry ---------------------------------------------------
    from task import get_task_registry, list_registered_tasks  # noqa: E402

    registry = get_task_registry()
    tasks = list_registered_tasks()
    print(f"[ok] task registry loaded: {tasks}")
    if "trajectory-sharerobot" not in tasks:
        failures.append("trajectory-sharerobot is not in the task registry")

    # --- Reward routing --------------------------------------------------
    from model.reward_func import (  # noqa: E402
        accuracy_reward as router_accuracy,
        format_reward as router_format,
        get_supported_task_types,
        list_registered_functions,
    )

    print(f"[ok] reward function registry: {list_registered_functions()}")
    print(f"[ok] supported task types: {get_supported_task_types()['all']}")

    # --- Trajectory rewards on a synthetic example -----------------------
    completion = [{"content": "<think>x</think><answer>[[0,0],[10,10],[20,20]]</answer>"}]
    gt = json.dumps([[0, 0], [10, 10], [20, 20]])
    fmt_scores = router_format(completion, [gt], ["trajectory-sharerobot"], image_width=[640], image_height=[480])
    acc_scores, _ = router_accuracy(completion, [gt], ["trajectory-sharerobot"], image_width=[640], image_height=[480])

    assert fmt_scores == [1.0], f"format reward should be 1.0, got {fmt_scores}"
    assert acc_scores[0] > 0.0, f"accuracy reward should be positive, got {acc_scores}"
    print(f"[ok] trajectory rewards on identical pred/gt: format={fmt_scores[0]}, accuracy={acc_scores[0]:.4f}")

    # --- Multi-task eval metrics dispatch --------------------------------
    metrics_fn = registry.get_metrics_fn("trajectory-sharerobot")
    if metrics_fn is None:
        failures.append("trajectory-sharerobot has no compute_metrics registered")
    else:
        sample_records = [
            {
                "prediction": "[[0,0],[10,10],[20,20]]",
                "ground_truth": [[0, 0], [10, 10], [20, 20]],
                "raw": "",
                "meta": {"image_width": 640, "image_height": 480},
            },
            {
                "prediction": "[[5,5],[15,15],[25,25]]",
                "ground_truth": [[0, 0], [10, 10], [20, 20]],
                "raw": "",
                "meta": {"image_width": 640, "image_height": 480},
            },
        ]
        metrics = metrics_fn(sample_records)
        print(f"[ok] compute_metrics returned: {metrics}")
        for key in ("dfd", "hd", "rmse", "endpoint_distance", "format_compliance"):
            if key not in metrics:
                failures.append(f"metrics missing key: {key}")

    # --- Task config / template -----------------------------------------
    from model.task_configs import get_question_template, normalize_task_type  # noqa: E402

    template = get_question_template(normalize_task_type("trajectory-sharerobot"))
    if not template or template == "N/A":
        failures.append("trajectory-sharerobot grpo_template is missing")
    else:
        print(f"[ok] template head: {template[:80]}...")

    # --- TEMPLATE.py must NOT be auto-registered -------------------------
    if "my-task" in tasks:
        failures.append("TEMPLATE.py was incorrectly auto-registered as a task")

    # --- processing_utils import path ------------------------------------
    from task.processing_utils import scale_bbox_xyxy, scale_trajectory  # noqa: E402

    scaled = scale_trajectory([[10, 10], [20, 20]], original_size=(100, 100), target_size=(200, 200))
    assert scaled == [[20, 20], [40, 40]], scaled
    box = scale_bbox_xyxy([10, 20, 30, 40], original_size=(100, 100), target_size=(200, 200))
    assert box == [20, 40, 60, 80], box
    print("[ok] processing_utils scaling works")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
