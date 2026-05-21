"""Trajectory ShareRobot Task Module"""

import json
import re
from typing import Any, Optional, Tuple

import numpy as np

from .processing_utils import scale_trajectory

# ============= Task Configuration =============

TASK_CONFIG = {
    "task_type":
        "trajectory-sharerobot",
    "description":
        "Trajectory Prediction for ShareRobot",
    "input_format":
        "image + instruction",
    "output_format":
        "<think>reasoning</think><answer>[[x1,y1], [x2,y2], ...]</answer>",
    "grpo_template":
        """{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.
        Output the final answer in the following JSON format:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
        Where each coordinate pair represents a point in the image's pixel space and the center of the end effector needs to follow the coordinates to complete the task. Each hand trajectory includes unknow number of [x,y] coordinate pairs. DONOT OUTPUT ANY ANSWER OR CONCLUSION IN THE THINK TAGS.""",
    "evaluation_metrics": ["frechet_distance", "hausdorff_distance", "rmse", "format_compliance"],
    "format_requirements": {
        "requires_think_tag": True,
        "requires_answer_tag": True,
        "answer_format": "list of coordinate pairs",
    }
}

# ============= Utility Functions =============


def get_default_debug_info(error_msg: str = 'unknown_error') -> dict:
    """Return default debug info structure with zero/default values."""
    return {
        'dfd': -1,
        'hd': -1,
        'rmse': -1,
        'endpoint_dist': -1,
        'rdfd': -1,
        'rhd': -1,
        'rrmse': -1,
        'rendpoint': -1,
        'rstartpoint': -1,
        'pred_traj_len': -1,
        'gt_traj_len': -1,
        'pred_traj_length': -1,
        'gt_traj_length': -1,
        'delta_traj_length': -1,
        'reward': -1,
        'error': error_msg,
    }


def calculate_trajectory_reward(pred_traj, gt_traj, image_width, image_height):
    """Calculate trajectory reward using multiple distance metrics with exponential normalization."""

    def normalize_point(point, width, height):
        return [point[0] / width, point[1] / height]

    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def discrete_frechet_distance(traj1, traj2):
        n, m = len(traj1), len(traj2)
        dist_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dist_matrix[i, j] = euclidean_distance(traj1[i], traj2[j])

        dp = np.full((n, m), np.inf)
        dp[0, 0] = dist_matrix[0, 0]

        for j in range(1, m):
            dp[0, j] = max(dp[0, j - 1], dist_matrix[0, j])
        for i in range(1, n):
            dp[i, 0] = max(dp[i - 1, 0], dist_matrix[i, 0])

        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]), dist_matrix[i, j])

        return dp[n - 1, m - 1]

    def hausdorff_distance(traj1, traj2):

        def directed_hausdorff(t1, t2):
            max_min_dist = 0
            for p1 in t1:
                min_dist = min(euclidean_distance(p1, p2) for p2 in t2)
                max_min_dist = max(max_min_dist, min_dist)
            return max_min_dist

        return max(directed_hausdorff(traj1, traj2), directed_hausdorff(traj2, traj1))

    def rmse_distance(traj1, traj2):
        if len(traj1) != len(traj2):
            min_len = min(len(traj1), len(traj2))
            traj1 = traj1[:min_len]
            traj2 = traj2[:min_len]
        squared_errors = [euclidean_distance(p1, p2)**2 for p1, p2 in zip(traj1, traj2)]
        return np.sqrt(np.mean(squared_errors)) if squared_errors else float('inf')

    def endpoint_distance(traj1, traj2):
        if len(traj1) == 0 or len(traj2) == 0:
            return float('inf')
        return euclidean_distance(traj1[-1], traj2[-1])

    try:
        debug_info = get_default_debug_info('ok')

        if not (isinstance(pred_traj, list) and isinstance(gt_traj, list)):
            return 0.0, debug_info
        if len(pred_traj) == 0 or len(gt_traj) == 0:
            return 0.0, debug_info

        pred_normalized = [normalize_point(p, image_width, image_height) for p in pred_traj]
        gt_normalized = [normalize_point(p, image_width, image_height) for p in gt_traj]

        def trajectory_length(traj):
            if len(traj) < 2:
                return 0.0
            return sum(euclidean_distance(traj[i - 1], traj[i]) for i in range(1, len(traj)))

        pred_length = trajectory_length(pred_normalized)
        gt_length = trajectory_length(gt_normalized)

        dfd = discrete_frechet_distance(pred_normalized, gt_normalized)
        hd = hausdorff_distance(pred_normalized, gt_normalized)
        rmse = rmse_distance(pred_normalized, gt_normalized)
        endpoint_dist = endpoint_distance(pred_normalized, gt_normalized)
        startpoint_dist = euclidean_distance(pred_normalized[0], gt_normalized[0]) if pred_normalized and gt_normalized else float('inf')

        rdfd = np.exp(-dfd / 0.1) if np.isfinite(dfd) else 0.0
        rhd = np.exp(-hd / 0.1) if np.isfinite(hd) else 0.0
        rrmse = np.exp(-rmse / 0.1) if np.isfinite(rmse) else 0.0
        rendpoint = np.exp(-endpoint_dist / 0.12) if np.isfinite(endpoint_dist) else 0.0
        rstartpoint = np.exp(-startpoint_dist / 0.12) if np.isfinite(startpoint_dist) else 0.0

        combined_score = rdfd + rhd + rrmse + rendpoint + rstartpoint

        debug_info.update({
            'dfd': dfd,
            'hd': hd,
            'rmse': rmse,
            'endpoint_dist': endpoint_dist,
            'rdfd': rdfd,
            'rhd': rhd,
            'rrmse': rrmse,
            'rendpoint': rendpoint,
            'rstartpoint': rstartpoint,
            'pred_traj_len': len(pred_traj),
            'gt_traj_len': len(gt_traj),
            'pred_traj_length': pred_length,
            'gt_traj_length': gt_length,
            'delta_traj_length': abs(pred_length - gt_length),
            'reward': combined_score,
        })

        return combined_score, debug_info

    except Exception as exc:  # pragma: no cover - diagnostic path
        info = get_default_debug_info(str(exc))
        return 0.0, info


# ============= Reward Functions =============


def format_reward(completion, sol, **kwargs):
    """Format reward for trajectory prediction"""
    content = completion[0]["content"]

    # Check wrapping - must have both <think> and <answer> tags
    has_think = re.search(r"<think>.*?</think>", content, re.DOTALL) is not None
    has_answer = re.search(r"<answer>.*?</answer>", content, re.DOTALL) is not None

    if not (has_think and has_answer):
        return 0.0

    # Check trajectory format: [[x1,y1], [x2,y2], ...]
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        try:
            trajectory = json.loads(answer_text)
            if isinstance(trajectory, list) and len(trajectory) > 0:
                for point in trajectory:
                    if not isinstance(point, list) or len(point) != 2:
                        return 0.0
                    if not all(isinstance(x, (int, float)) for x in point):
                        return 0.0
                return 1.0
        except json.JSONDecodeError:
            return 0.0

    return 0.0


def accuracy_reward(completion, sol, **kwargs):
    """Time-ordered trajectory accuracy reward returning reward and debug info."""
    content = completion[0]["content"]
    answer_pattern = r"<answer>(.*?)</answer>"

    try:
        image_width = kwargs.get("image_width", 1280)
        image_height = kwargs.get("image_height", 720)

        gt_data = json.loads(sol) if isinstance(sol, str) else sol
        if not isinstance(gt_data, list):
            return 0.0, get_default_debug_info('invalid_gt_format')

        answer_match = re.search(answer_pattern, content, re.DOTALL)
        if not answer_match:
            return 0.0, get_default_debug_info('no_answer_match')

        pred_data = json.loads(answer_match.group(1).strip())
        if not isinstance(pred_data, list):
            return 0.0, get_default_debug_info('invalid_pred_format')

        reward, debug_info = calculate_trajectory_reward(pred_data, gt_data, image_width, image_height)
        return reward, debug_info

    except Exception as exc:
        return 0.0, get_default_debug_info(str(exc))


# ============= Evaluation Metrics =============


def _parse_trajectory(text: Any) -> Optional[list]:
    """Parse a trajectory answer from a model output or ground-truth string."""
    if text is None:
        return None
    if isinstance(text, list):
        return text
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _euclidean(p1, p2):
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _discrete_frechet(traj1, traj2):
    n, m = len(traj1), len(traj2)
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = _euclidean(traj1[i], traj2[j])
    dp = np.full((n, m), np.inf)
    dp[0, 0] = dist[0, 0]
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], dist[0, j])
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], dist[i, 0])
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]), dist[i, j])
    return float(dp[n - 1, m - 1])


def _hausdorff(traj1, traj2):
    def directed(a, b):
        return max(min(_euclidean(p1, p2) for p2 in b) for p1 in a)
    return max(directed(traj1, traj2), directed(traj2, traj1))


def _rmse(traj1, traj2):
    n = min(len(traj1), len(traj2))
    if n == 0:
        return float("inf")
    sq = [_euclidean(traj1[i], traj2[i]) ** 2 for i in range(n)]
    return float(np.sqrt(np.mean(sq)))


def compute_metrics(predictions, **_: Any) -> dict:
    """Aggregate trajectory metrics over a list of evaluation records.

    Each record is ``{"prediction": ..., "ground_truth": ..., "raw": str, "meta": dict}``.
    Distances are computed in the original image's pixel space; ``meta`` should
    contain ``image_width`` / ``image_height`` of the *original* image so
    predictions made on a resized image can be scaled back if needed.
    Returns averaged DFD / HD / RMSE / endpoint distance and format compliance.
    """
    if not predictions:
        return {"num_samples": 0}

    dfd_vals, hd_vals, rmse_vals, ep_vals, fmt_vals = [], [], [], [], []
    n_parse_fail = 0

    for record in predictions:
        gt = _parse_trajectory(record.get("ground_truth"))
        pred = _parse_trajectory(record.get("prediction"))
        fmt_ok = (
            pred is not None
            and isinstance(pred, list)
            and len(pred) > 0
            and all(isinstance(p, list) and len(p) == 2 for p in pred)
        )
        fmt_vals.append(1.0 if fmt_ok else 0.0)
        if not (fmt_ok and gt and len(gt) > 0):
            n_parse_fail += int(not fmt_ok)
            continue

        dfd_vals.append(_discrete_frechet(pred, gt))
        hd_vals.append(_hausdorff(pred, gt))
        rmse_vals.append(_rmse(pred, gt))
        ep_vals.append(_euclidean(pred[-1], gt[-1]))

    def _avg(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "num_samples": len(predictions),
        "num_format_failures": n_parse_fail,
        "format_compliance": _avg(fmt_vals),
        "dfd": _avg(dfd_vals),
        "hd": _avg(hd_vals),
        "rmse": _avg(rmse_vals),
        "endpoint_distance": _avg(ep_vals),
    }


# ============= Registration =============


def process_answer(answer, *, original_size: Optional[Tuple[int, int]] = None, target_size: Optional[Tuple[int, int]] = None, **_: Any):
    """Scale trajectory coordinate sequences to match resized image dimensions."""
    return scale_trajectory(answer, original_size, target_size)


def register(accuracy_registry, format_registry, answer_registry=None):
    """Register this task's reward functions"""
    accuracy_registry["trajectory-sharerobot"] = accuracy_reward
    format_registry["trajectory-sharerobot"] = format_reward
    if answer_registry is not None:
        answer_registry["trajectory-sharerobot"] = process_answer
