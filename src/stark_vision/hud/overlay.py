"""Simple RGB HUD overlay that only reads runtime state."""

from __future__ import annotations

import numpy as np

from stark_vision.contracts import RenderedFrame, RuntimeSnapshot, TrackingStatus


class HudRenderer:
    def render(self, frame_rgb: np.ndarray, snapshot: RuntimeSnapshot) -> RenderedFrame:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("HUD input must be RGB uint8 HxWx3")
        output = frame_rgb.copy()
        active = snapshot.active_target
        color = (30, 220, 70) if active and active.status is TrackingStatus.TRACKING else (235, 165, 35)
        output[: min(6, output.shape[0]), :, :] = color
        if active and active.box:
            self._draw_box(output, active.box, (30, 255, 90))
        if snapshot.pending_is_ambiguous:
            for candidate in snapshot.pending_candidates:
                self._draw_box(output, candidate.box, (255, 170, 25))

        target = f"#{active.target_id} {active.prompt}" if active else "none"
        lines = (
            f"frame={snapshot.frame_id}",
            f"target={target}",
            f"status={active.status.value if active else 'idle'}",
            f"effect={snapshot.effect.mode.value}",
            f"message={snapshot.message}",
        )
        return RenderedFrame(output, lines)

    @staticmethod
    def _draw_box(image: np.ndarray, box, color: tuple[int, int, int], thickness: int = 2) -> None:
        height, width = image.shape[:2]
        x1 = max(0, min(width, int(box.x1)))
        y1 = max(0, min(height, int(box.y1)))
        x2 = max(0, min(width, int(np.ceil(box.x2))))
        y2 = max(0, min(height, int(np.ceil(box.y2))))
        if x2 <= x1 or y2 <= y1:
            return
        for offset in range(thickness):
            if y1 + offset < y2:
                image[y1 + offset, x1:x2] = color
            if y2 - 1 - offset >= y1:
                image[y2 - 1 - offset, x1:x2] = color
            if x1 + offset < x2:
                image[y1:y2, x1 + offset] = color
            if x2 - 1 - offset >= x1:
                image[y1:y2, x2 - 1 - offset] = color