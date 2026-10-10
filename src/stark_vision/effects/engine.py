"""Mask-based RGB effects with no dependency on the segmentation backend."""

from __future__ import annotations

import cv2
import numpy as np

from stark_vision.contracts import EffectMode, EffectSpec


class EffectEngine:
    def apply(self, frame_rgb: np.ndarray, mask: np.ndarray | None, effect: EffectSpec) -> np.ndarray:
        frame = _validate_frame(frame_rgb)
        output = frame.copy()
        if mask is None or effect.mode is EffectMode.NONE:
            return output
        if mask.dtype != np.bool_ or mask.shape != frame.shape[:2]:
            raise ValueError("effect mask must be bool HxW and match the RGB frame")

        if effect.mode is EffectMode.RECOLOR:
            if effect.color_rgb is None:
                raise ValueError("recolor effect requires color_rgb")
            output[mask] = np.asarray(effect.color_rgb, dtype=np.uint8)
        elif effect.mode is EffectMode.ISOLATE:
            luminance = np.round(frame @ np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8)
            output[:] = luminance[..., None]
            output[mask] = frame[mask]
        elif effect.mode is EffectMode.HIGHLIGHT:
            tint = np.array([255, 205, 35], dtype=np.float32)
            output[mask] = np.clip(frame[mask].astype(np.float32) * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)
            edge = _mask_edge(mask)
            thickness = max(1, round(min(frame.shape[:2]) / 360))  # ~3 px on 1080p
            if thickness > 1:
                kernel = np.ones((thickness, thickness), np.uint8)
                edge = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
            output[edge] = (0, 245, 255)
        elif effect.mode is EffectMode.BLUR:
            # Radius scales with resolution so the blur is visible on 1080p, not just on small frames.
            radius = max(4, round(min(frame.shape[:2]) * 0.025))
            ys, xs = np.nonzero(mask)
            y1, y2 = max(0, ys.min() - radius), min(frame.shape[0], ys.max() + radius + 1)
            x1, x2 = max(0, xs.min() - radius), min(frame.shape[1], xs.max() + radius + 1)
            blurred = _box_blur(frame[y1:y2, x1:x2], radius=radius)  # only the target region
            region = mask[y1:y2, x1:x2]
            output[y1:y2, x1:x2][region] = blurred[region]
        return output


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be RGB uint8 HxWx3")
    return frame


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _box_blur(frame: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    padded = np.pad(frame.astype(np.uint32), ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
    height, width = frame.shape[:2]
    sums = (
        integral[size:size + height, size:size + width]
        - integral[:height, size:size + width]
        - integral[size:size + height, :width]
        + integral[:height, :width]
    )
    return (sums / (size * size)).astype(np.uint8)
