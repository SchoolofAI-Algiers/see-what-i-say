"""
Bounding box math helpers for converting between pixel xyxy, xywh, and normalized [0, 1] space.
"""

from __future__ import annotations

import math

from stark_vision.contracts import BoundingBox


def _sanitize_val(v: float, default: float = 0.0) -> float:
    if math.isnan(v) or math.isinf(v):
        return default
    return float(v)


def xyxy_to_xywh(
    box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h] with degenerate box safety."""
    x1 = _sanitize_val(box[0])
    y1 = _sanitize_val(box[1])
    x2 = max(x1, _sanitize_val(box[2]))
    y2 = max(y1, _sanitize_val(box[3]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (x1, y1, w, h)


def xywh_to_xyxy(
    box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x = _sanitize_val(box[0])
    y = _sanitize_val(box[1])
    w = max(0.0, _sanitize_val(box[2]))
    h = max(0.0, _sanitize_val(box[3]))
    return (x, y, x + w, y + h)


def normalize_box(
    box: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Normalize pixel coordinates [x1, y1, x2, y2] to [0.0, 1.0]."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    x1 = _sanitize_val(box[0])
    y1 = _sanitize_val(box[1])
    x2 = max(x1, _sanitize_val(box[2]))
    y2 = max(y1, _sanitize_val(box[3]))
    return (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    )


def denormalize_box(
    box: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[int, int, int, int]:
    """Denormalize [0.0, 1.0] coordinates to integer pixel coordinates [x1, y1, x2, y2]."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    x1 = max(0.0, min(1.0, _sanitize_val(box[0])))
    y1 = max(0.0, min(1.0, _sanitize_val(box[1])))
    x2 = max(x1, min(1.0, _sanitize_val(box[2])))
    y2 = max(y1, min(1.0, _sanitize_val(box[3])))
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def clip_box(
    box: tuple[float, float, float, float],
    max_w: float,
    max_h: float,
) -> tuple[float, float, float, float]:
    """Clip [x1, y1, x2, y2] coordinates to stay within image canvas bounds [0, max_w] and [0, max_h]."""
    if max_w <= 0 or max_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1 = max(0.0, min(max_w, _sanitize_val(box[0])))
    y1 = max(0.0, min(max_h, _sanitize_val(box[1])))
    x2 = max(x1, min(max_w, _sanitize_val(box[2])))
    y2 = max(y1, min(max_h, _sanitize_val(box[3])))
    return (x1, y1, x2, y2)


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(_sanitize_val(box_a[0]), _sanitize_val(box_b[0]))
    y1 = max(_sanitize_val(box_a[1]), _sanitize_val(box_b[1]))
    x2 = min(_sanitize_val(box_a[2]), _sanitize_val(box_b[2]))
    y2 = min(_sanitize_val(box_a[3]), _sanitize_val(box_b[3]))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def box_center(
    box: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Compute centroid (cx, cy) of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(
    box: tuple[float, float, float, float]
) -> float:
    """Compute spatial area of [x1, y1, x2, y2]."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def normalized_to_pixel(
    box: BoundingBox | tuple[float, float, float, float],
    width: int,
    height: int,
) -> BoundingBox:
    """Convert normalized [0, 1] bounding box to image pixel space."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    b = box.xyxy if hasattr(box, "xyxy") else box
    bx1 = max(0.0, min(float(width), float(b[0]) * width))
    by1 = max(0.0, min(float(height), float(b[1]) * height))
    bx2 = max(bx1 + 1.0, min(float(width), float(b[2]) * width))
    by2 = max(by1 + 1.0, min(float(height), float(b[3]) * height))
    return BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2)


def pixel_to_normalized(
    box: BoundingBox | tuple[float, float, float, float],
    width: int,
    height: int,
) -> BoundingBox:
    """Convert image pixel bounding box coordinates to normalized [0, 1] space."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    b = box.xyxy if hasattr(box, "xyxy") else box
    bx1 = max(0.0, min(1.0, float(b[0]) / width))
    by1 = max(0.0, min(1.0, float(b[1]) / height))
    bx2 = max(bx1 + 1e-4, min(1.0, float(b[2]) / width))
    by2 = max(by1 + 1e-4, min(1.0, float(b[3]) / height))
    return BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2)

