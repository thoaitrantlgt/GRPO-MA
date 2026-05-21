"""Coordinate-rescaling helpers shared by image-based tasks.

Qwen2.5-VL applies ``smart_resize`` to every input image, so pixel-space
ground-truth answers (trajectories, bounding boxes, key points) must be
rescaled to the resized image before they are passed to the reward function.
Each task module imports the helper that matches its answer schema.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Tuple


_Size = Optional[Tuple[int, int]]


def _parse(answer: Any) -> Any:
    if isinstance(answer, str):
        return json.loads(answer)
    return answer


def _identity_scaling(original_size: _Size, target_size: _Size) -> bool:
    return (
        original_size is None
        or target_size is None
        or tuple(original_size) == tuple(target_size)
    )


def _scale_point(point: Sequence[float], sx: float, sy: float) -> List[int]:
    x, y = point[0], point[1]
    return [int(round(x * sx)), int(round(y * sy))]


def scale_trajectory(answer: Any, original_size: _Size = None, target_size: _Size = None) -> Any:
    """Rescale a list of ``[x, y]`` waypoints from original to target image size."""
    if _identity_scaling(original_size, target_size):
        return answer
    traj = _parse(answer)
    if not isinstance(traj, list):
        return answer
    ow, oh = original_size  # type: ignore[misc]
    tw, th = target_size  # type: ignore[misc]
    sx, sy = tw / ow, th / oh
    return [_scale_point(pt, sx, sy) for pt in traj]


def scale_bbox_xyxy(answer: Any, original_size: _Size = None, target_size: _Size = None) -> Any:
    """Rescale a ``[x1, y1, x2, y2]`` box."""
    if _identity_scaling(original_size, target_size):
        return answer
    box = _parse(answer)
    if not (isinstance(box, list) and len(box) == 4):
        return answer
    ow, oh = original_size  # type: ignore[misc]
    tw, th = target_size  # type: ignore[misc]
    sx, sy = tw / ow, th / oh
    p1 = _scale_point(box[:2], sx, sy)
    p2 = _scale_point(box[2:], sx, sy)
    return [p1[0], p1[1], p2[0], p2[1]]


def scale_points(answer: Any, original_size: _Size = None, target_size: _Size = None) -> Any:
    """Rescale a list of standalone ``[x, y]`` points (same schema as a trajectory)."""
    return scale_trajectory(answer, original_size, target_size)
