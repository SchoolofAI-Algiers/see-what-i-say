"""SAM 2 adapter implementing the Tracker protocol from docs/interfaces.md.

Uses only SAM 2's public video-predictor API. That API wants a directory of
frames, so live frames are buffered into a small sliding window: every
`window_size` frames the window is propagated through the real SAM 2 memory
tracker, then slid forward re-anchored on the last confident box.

Between propagations the last mask is held (so the mask lags by up to
window_size-1 frames). True per-frame streaming (as in Grounded-SAM-2's camera
demo, which patches the predictor) is the later performance upgrade.

Contract notes (see interfaces.md):
- masks are bool HxW aligned to the frame passed in
- a LOST result carries mask=None and the target is dropped
- images are RGB (Frame.image); conversion to what SAM 2 reads happens here
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stark_vision.contracts import BoundingBox, Frame, TrackingResult, TrackingStatus

logger = logging.getLogger(__name__)


@dataclass
class _TargetState:
    box: BoundingBox
    mask: np.ndarray
    confidence: float


def _mask_to_box(mask: np.ndarray) -> BoundingBox | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class SAM2Tracker:
    def __init__(
        self,
        config_path: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt",
        device: str | None = None,
        window_size: int = 4,
    ) -> None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=self.device)
        self.window_size = max(2, window_size)
        self._dir = Path(tempfile.mkdtemp(prefix="sam2_stream_"))
        self._targets: dict[int, _TargetState] = {}
        self._count = 0
        self._last_frame_id: int | None = None

    # ---- Tracker protocol -------------------------------------------------

    def initialize(self, frame: Frame, box: BoundingBox, target_id: int) -> TrackingResult:
        """Stage a target on `frame` and return its first mask.

        Existing targets are re-anchored on this frame with their last box.
        """
        self._reset_dir()
        self._write(frame)
        clipped = self._clip(box, frame)
        prompts = {tid: st.box for tid, st in self._targets.items()}
        prompts[target_id] = clipped
        out = self._run(prompts, propagate=False)

        for tid, (mask, conf) in out.items():
            if tid == target_id:
                continue
            new_box = _mask_to_box(mask)
            if new_box is not None:
                self._targets[tid] = _TargetState(new_box, mask, conf)

        mask, conf = out[target_id]
        new_box = _mask_to_box(mask)
        if new_box is None:
            logger.warning("SAM2 produced an empty mask for target %d", target_id)
            return TrackingResult(target_id, frame.id, None, None, 0.0, TrackingStatus.LOST)
        self._targets[target_id] = _TargetState(new_box, mask, conf)
        self._last_frame_id = frame.id
        return TrackingResult(target_id, frame.id, new_box, mask, conf, TrackingStatus.TRACKING)

    def update(self, frame: Frame) -> tuple[TrackingResult, ...]:
        if not self._targets:
            return ()
        self._write(frame)
        self._last_frame_id = frame.id

        if self._count < self.window_size:
            # Window still filling: hold the last masks (stale by < window_size frames).
            return tuple(
                TrackingResult(tid, frame.id, st.box, st.mask, st.confidence, TrackingStatus.TRACKING)
                for tid, st in self._targets.items()
            )

        prompts = {tid: st.box for tid, st in self._targets.items()}
        out = self._run(prompts, propagate=True)
        results: list[TrackingResult] = []
        for tid in list(self._targets):
            mask, conf = out.get(tid, (np.zeros(0, dtype=bool), 0.0))
            box = _mask_to_box(mask) if mask.ndim == 2 else None
            if box is None:
                results.append(TrackingResult(tid, frame.id, None, None, 0.0, TrackingStatus.LOST))
                del self._targets[tid]
                continue
            self._targets[tid] = _TargetState(box, mask, conf)
            results.append(TrackingResult(tid, frame.id, box, mask, conf, TrackingStatus.TRACKING))
        self._slide()
        return tuple(results)

    def remove(self, target_id: int) -> None:
        self._targets.pop(target_id, None)

    def reset(self) -> None:
        self._targets.clear()
        self._reset_dir()

    def close(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    # ---- internals --------------------------------------------------------

    def _autocast(self):
        if not self.device.startswith("cuda"):
            return contextlib.nullcontext()
        major = self._torch.cuda.get_device_properties(0).major
        dtype = self._torch.bfloat16 if major >= 8 else self._torch.float16
        return self._torch.autocast("cuda", dtype=dtype)

    def _run(self, prompts: dict[int, BoundingBox], propagate: bool) -> dict[int, tuple[np.ndarray, float]]:
        torch = self._torch
        last = None
        with torch.inference_mode(), self._autocast():
            state = self.predictor.init_state(video_path=str(self._dir))
            self.predictor.reset_state(state)
            for tid, box in prompts.items():
                last = self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(tid),
                    box=np.asarray(box.xyxy, dtype=np.float32),
                )
            if propagate:
                for _idx, obj_ids, logits in self.predictor.propagate_in_video(state):
                    last = (_idx, obj_ids, logits)
            _idx, obj_ids, logits = last
            out: dict[int, tuple[np.ndarray, float]] = {}
            for i, oid in enumerate(obj_ids):
                lg = logits[i, 0]
                mask = (lg > 0.0).cpu().numpy()
                conf = float(torch.sigmoid(lg[lg > 0.0]).mean().item()) if mask.any() else 0.0
                out[int(oid)] = (mask, conf)
            self.predictor.reset_state(state)
        return out

    def _slide(self) -> None:
        """Re-anchor the next window on the newest frame."""
        newest = sorted(self._dir.glob("*.jpg"))[-1]
        carry = self._dir.with_name(self._dir.name + "_carry.jpg")
        shutil.copy(newest, carry)
        self._reset_dir()
        shutil.move(str(carry), str(self._dir / "00000.jpg"))
        self._count = 1

    def _reset_dir(self) -> None:
        for p in self._dir.glob("*.jpg"):
            p.unlink()
        self._count = 0

    def _write(self, frame: Frame) -> None:
        bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self._dir / f"{self._count:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self._count += 1

    @staticmethod
    def _clip(box: BoundingBox, frame: Frame) -> BoundingBox:
        h, w = frame.image.shape[:2]
        x1 = max(0.0, min(w - 2.0, box.x1))
        y1 = max(0.0, min(h - 2.0, box.y1))
        x2 = max(x1 + 1.0, min(float(w), box.x2))
        y2 = max(y1 + 1.0, min(float(h), box.y2))
        return BoundingBox(x1, y1, x2, y2)
