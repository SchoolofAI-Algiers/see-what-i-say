"""
Temporal smoothing for bounding box tracking across frames.

Uses a 1-euro filter on centroid and log-scale dimensions to damp
high-frequency jitter when stationary while tracking quick motions without lag.
"""

from __future__ import annotations

import math
import time
from typing import Sequence


class LowPassFilter:
    # simple exponential lowpass
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.s: float | None = None

    def filter(self, value: float) -> float:
        if self.s is None or math.isnan(self.s):
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self) -> None:
        self.s = None


class OneEuroFilter:
    # single axis 1-euro filter adjusting cutoff based on speed

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: float | None = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.0
        return 1.0 / (1.0 + tau / te) if (tau + te) > 0 else 1.0

    def filter(self, x: float, timestamp: float | None = None) -> float:
        t = timestamp if timestamp is not None else time.perf_counter()
        if self.last_time is None:
            self.last_time = t
            return self.x_filt.filter(x)

        # Clamp dt to handle frame drops gracefully (between 0.1ms and 1.0s)
        dt = max(1e-4, min(1.0, t - self.last_time))
        rate = 1.0 / dt
        self.last_time = t

        prev_x = self.x_filt.s if self.x_filt.s is not None else x
        dx = (x - prev_x) * rate
        edx = self.dx_filt.filter(dx)
        self.dx_filt.alpha = self._alpha(rate, self.d_cutoff)

        cutoff = self.min_cutoff + self.beta * abs(edx)
        self.x_filt.alpha = self._alpha(rate, cutoff)
        return self.x_filt.filter(x)

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


class BoxTemporalFilter:
    # filters (center_x, center_y, log_w, log_h) to stop reticle jitter on video frames

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.02,
        cut_iou_threshold: float = 0.15,
        cut_center_dist_threshold: float = 0.35,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.cut_iou_threshold = cut_iou_threshold
        self.cut_center_dist_threshold = cut_center_dist_threshold

        # 4 synchronized filters: cx, cy, log_w, log_h
        self.filter_cx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_cy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_log_w = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_log_h = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        self._last_raw_box: tuple[float, float, float, float] | None = None
        self._last_filtered_box: tuple[float, float, float, float] | None = None

    @staticmethod
    def _box_to_centroid_log(
        box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1e-4, x2 - x1)
        h = max(1e-4, y2 - y1)
        return (cx, cy, math.log(w), math.log(h))

    @staticmethod
    def _centroid_log_to_box(
        state: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        cx, cy, log_w, log_h = state
        w = max(1e-4, math.exp(log_w))
        h = max(1e-4, math.exp(log_h))
        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)
        return (x1, y1, x2, y2)

    def _detect_discontinuity(
        self,
        new_box: tuple[float, float, float, float],
    ) -> bool:
        """Detect scene cut, target loss, or sudden re-identification."""
        if self._last_filtered_box is None:
            return False

        # Compute IoU with previous filtered box
        x1 = max(new_box[0], self._last_filtered_box[0])
        y1 = max(new_box[1], self._last_filtered_box[1])
        x2 = min(new_box[2], self._last_filtered_box[2])
        y2 = min(new_box[3], self._last_filtered_box[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area1 = max(1e-4, (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]))
        area2 = max(1e-4, (self._last_filtered_box[2] - self._last_filtered_box[0]) * (self._last_filtered_box[3] - self._last_filtered_box[1]))
        iou = intersection / (area1 + area2 - intersection)

        # Center Euclidean jump
        cx_new = (new_box[0] + new_box[2]) / 2.0
        cy_new = (new_box[1] + new_box[3]) / 2.0
        cx_old = (self._last_filtered_box[0] + self._last_filtered_box[2]) / 2.0
        cy_old = (self._last_filtered_box[1] + self._last_filtered_box[3]) / 2.0
        center_dist = math.hypot(cx_new - cx_old, cy_new - cy_old)

        # Jump is measured relative to the previous box diagonal so the threshold
        # is resolution-independent (the old code compared pixels to 0.35).
        diag = math.hypot(
            self._last_filtered_box[2] - self._last_filtered_box[0],
            self._last_filtered_box[3] - self._last_filtered_box[1],
        )
        rel_jump = center_dist / max(1e-4, diag)
        return iou < self.cut_iou_threshold and rel_jump > self.cut_center_dist_threshold

    def filter_box(
        self,
        box: tuple[float, float, float, float],
        timestamp: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Filter bounding box in centroid-log-dimension space.

        Args:
            box: (x1, y1, x2, y2) bounding box
            timestamp: monotonic timestamp in seconds (defaults to time.perf_counter())

        Returns:
            Filtered (x1, y1, x2, y2) coordinates
        """
        t = timestamp if timestamp is not None else time.perf_counter()

        # Sanitize box coordinates
        clean_box = (
            float(box[0]), float(box[1]),
            max(float(box[0]) + 1e-3, float(box[2])),
            max(float(box[1]) + 1e-3, float(box[3])),
        )

        # Check for target jump / scene cut
        if self._detect_discontinuity(clean_box):
            self.reset()

        cx, cy, log_w, log_h = self._box_to_centroid_log(clean_box)

        filt_cx = self.filter_cx.filter(cx, t)
        filt_cy = self.filter_cy.filter(cy, t)
        filt_log_w = self.filter_log_w.filter(log_w, t)
        filt_log_h = self.filter_log_h.filter(log_h, t)

        out_box = self._centroid_log_to_box((filt_cx, filt_cy, filt_log_w, filt_log_h))
        self._last_raw_box = clean_box
        self._last_filtered_box = out_box

        return (
            round(out_box[0], 2),
            round(out_box[1], 2),
            round(out_box[2], 2),
            round(out_box[3], 2),
        )

    def reset(self) -> None:
        """Reset internal filter states immediately."""
        self.filter_cx.reset()
        self.filter_cy.reset()
        self.filter_log_w.reset()
        self.filter_log_h.reset()
        self._last_raw_box = None
        self._last_filtered_box = None
