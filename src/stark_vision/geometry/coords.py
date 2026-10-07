""" converte spatial coordinate from normalized bounding bozes to image pixel space ( standard rgb )
( aspect ratio transformations and mask geometry)"""

from __future__ import annotations
import numpy as np
from stark_vision.contracts import BoundingBox

#convert normalized [0, 1] bounding box coordinates to image pixel space
def normalized_to_pixel(box: BoundingBox, width: int, height: int) -> BoundingBox:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    return BoundingBox(
        x1=max(0.0, min(float(width), box.x1 * width)),
        y1=max(0.0, min(float(height), box.y1 * height)),
        x2=max(0.0, min(float(width), box.x2 * width)),
        y2=max(0.0, min(float(height), box.y2 * height)),
    )

# conv from pixel space bounding box coords to normalized [0,1]
def pixel_to_normalized(box: BoundingBox, width: int, height: int) -> BoundingBox:
    """Convert image pixel bounding box coordinates to normalized [0, 1] space."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    return BoundingBox(
        x1=max(0.0, min(1.0, box.x1 / width)),
        y1=max(0.0, min(1.0, box.y1 / height)),
        x2=max(0.0, min(1.0, box.x2 / width)),
        y2=max(0.0, min(1.0, box.y2 / height)),
    )

# mask area : pixel number in mask
def mask_area(mask: np.ndarray) -> int:
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Mask must be a 2D boolean numpy array.")
    return int(np.sum(mask))


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """
    calc center coords of mask
    returns (center_x, center_y) floats, or None if the mask is empty.
    """
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Mask must be a 2D boolean numpy array.")
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0:
        return None
    return float(np.mean(x_indices)), float(np.mean(y_indices))

# we prob won't need this but still
def mask_to_bbox(mask: np.ndarray) -> BoundingBox | None:
    """get bounding box from mask"""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Mask must be a 2D boolean numpy array.")
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0:
        return None
    return BoundingBox(
        x1=float(np.min(x_indices)),
        y1=float(np.min(y_indices)),
        x2=float(np.max(x_indices) + 1),
        y2=float(np.max(y_indices) + 1),
    )


def map_letterbox_box_to_frame(
    box: BoundingBox, frame_size: tuple[int, int], model_size: tuple[int, int]
) -> BoundingBox:
    """transform model-space bounding box back to original camera frame coordinates.
    Accounts for letterboxing/padding offsets and scale factors.
    frame_size: (frame_width, frame_height)
    model_size: (model_width, model_height)
    """
    frame_w, frame_h = frame_size
    model_w, model_h = model_size
    if frame_w <= 0 or frame_h <= 0 or model_w <= 0 or model_h <= 0:
        raise ValueError("Frame and model dimensions must be positive integers.")

    scale = min(model_w / frame_w, model_h / frame_h)
    scaled_w = frame_w *scale
    scaled_h = frame_h *scale

    pad_x = (model_w - scaled_w) /2.0
    pad_y = (model_h - scaled_h) /2.0

    unpadded_x1 = (box.x1 - pad_x) /scale
    unpadded_y1 = (box.y1 - pad_y) /scale
    unpadded_x2 = (box.x2 - pad_x) /scale
    unpadded_y2 = (box.y2 - pad_y) /scale
    return BoundingBox(
        x1=max(0.0, min(float(frame_w),unpadded_x1)),
        y1=max(0.0, min(float(frame_h),unpadded_y1)),
        x2=max(0.0, min(float(frame_w),unpadded_x2)),
        y2=max(0.0, min(float(frame_h),unpadded_y2)),
    )