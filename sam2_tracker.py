"""
SAM2-backed Tracker component.

Conforms to the `Tracker` Protocol defined in docs/interfaces.md:

    class Tracker(Protocol):
        def initialize(self, frame: Frame, box: BoundingBox, target_id: int) -> TrackingResult: ...
        def update(self, frame: Frame) -> tuple[TrackingResult, ...]: ...
        def remove(self, target_id: int) -> None: ...
        def reset(self) -> None: ...

Scope / known limitations (read before relying on this for integration)
-------------------------------------------------------------------------

1. SINGLE ACTIVE TARGET. This implementation tracks one target at a time.
   `update()` still returns a tuple (0 or 1 elements) to satisfy the
   Protocol's multi-target shape, but calling `initialize()` while a
   target is already active REPLACES it rather than running both
   concurrently. Per interfaces.md, TargetManager expects a pending
   (staged) target to track *alongside* the still-rendering active
   target until promotion. True concurrent staged+active tracking is
   NOT implemented here — it needs shared window/frame-buffer state
   across targets instead of resetting it on every initialize(), which
   is its own design problem. Flagged explicitly rather than shipped
   silently broken.

2. SLIDING-WINDOW PROPAGATION, NOT PER-FRAME STREAMING. The public SAM2
   video-predictor API needs a known frame set up front
   (`init_state(video_path=...)`); there's no "append one live frame"
   mode in the stable API. This implementation buffers incoming frames
   into a small sliding window on disk (WINDOW_SIZE frames), runs real
   SAM2 video propagation (genuine temporal memory, not re-detection)
   across each window, then slides forward re-anchored on the last
   confident mask. Real SAM2 tracking behavior, with up to WINDOW_SIZE
   frames of added latency per update as an explicit, documented
   trade-off. Grounded-SAM-2's
   grounded_sam2_tracking_camera_with_continuous_id.py gets true
   per-frame streaming by patching sam2_video_predictor.py itself;
   that's the intended upgrade once we're validating on GPU hardware
   for D-Day, not required for this MVP milestone.

   KNOWN DRIFT RISK: cross-window memory does NOT carry over — each new
   window starts a fresh `init_state`, re-seeded only from the previous
   window's final mask. If that mask was imperfect, the error has no
   way to self-correct and can persist indefinitely. Two mitigations
   are applied (see #2b, #2c below); a stronger fix (periodic
   re-grounding, or keeping a single persistent inference_state instead
   of rebuilding one per window) is future work if drift is still
   visible after these.

   2b. BOX + CENTER-POINT PROMPTING. A box-only prompt tends to
   under/over-segment irregular silhouettes (hair, shoulders, loose
   clothing) because nothing in the prompt says *where inside the box*
   the foreground actually is. Every prompt (initial and re-seed) now
   combines the box with a positive point, via two sequential calls to
   `add_new_points_or_box` for the same obj_id/frame_idx — that method
   is explicitly designed for iterative refinement (repeated calls
   accumulate prompts), so two calls is a safe, well-supported pattern
   regardless of whether a single combined box+points call is accepted
   by the installed SAM2 version.

   2c. CENTROID RE-SEEDING, NOT BBOX-CORNER RE-SEEDING. The refinement
   point used to re-seed each new window is the previous mask's
   centroid (center of mass of the actual mask pixels), not the
   midpoint of its bounding box. A centroid is far more robust to a
   stray misclassified pixel cluster dragging the box off-center.

3. CONFIDENCE IS UNSET. SAM2's video predictor doesn't expose a
   calibrated per-mask confidence through the stable API used here.
   `TrackingResult.confidence` is always `None`. `UNCERTAIN` status is
   instead derived heuristically from mask area (a mask that's shrunk
   to near-nothing is flagged UNCERTAIN rather than TRACKING) — a
   placeholder worth replacing if a better signal becomes available.

4. THREADING. Per interfaces.md, "All calls are serialized on the
   vision loop" — this class does no internal locking and expects the
   caller not to call into it concurrently. A background-worker
   wrapper exists only in test_sam2_tracker_manual.py, to keep a
   manual webcam demo responsive; it is a test-harness convenience,
   NOT part of this component, and shouldn't be treated as part of the
   Tracker contract.

5. SHARED TYPES ARE LOCAL COPIES. Frame / BoundingBox / TrackingResult /
   TrackingStatus below are written standalone against the published
   spec because the canonical types module (likely `stark_vision.types`
   or `stark_vision.ports`, per the `stark_vision.mocks.adapters`
   package path referenced in interfaces.md) isn't available in this
   working copy. If/when that module is pulled in, delete the block
   below and import from there instead — field names/shapes match the
   spec exactly, so the swap should be a no-op.

6. FILE LOCATION. This was written against a bare clone of
   facebookresearch/sam2, not the actual stark_vision application repo
   (whose layout is `src/stark_vision/...` per interfaces.md). Move
   this file to e.g. `src/stark_vision/vision/sam2_tracker.py` in the
   real repo and adjust the `sam2` import path / PYTHONPATH as needed.

Dependency / checkpoint policy (see interfaces.md "Project-wide
dependency policy"): this assumes `sam2` is installed per the project's
shared pyproject.toml/uv.lock, not a subsystem-local requirements.txt.
The checkpoint path passed to __init__ must point to a file that is NOT
committed to the repo — document its source/version (checkpoint release
date + hiera size) in the PR description instead.
"""
from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


# ---------------------------------------------------------------------------
# Shared data types — see docs/interfaces.md "Shared data types".
# See limitation #5 above: swap for canonical imports once available.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    id: int
    timestamp_ns: int
    image: np.ndarray  # RGB, uint8, HxWx3. Treat as read-only.


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float  # right/bottom edges are exclusive, per spec

    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("BoundingBox requires positive width and height")

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


class TrackingStatus(Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackingResult:
    target_id: int
    frame_id: int
    box: Optional[BoundingBox]
    mask: Optional[np.ndarray]       # bool HxW, aligned to frame_id; None iff LOST
    confidence: Optional[float]
    status: TrackingStatus


# ---------------------------------------------------------------------------
# Tracker implementation
# ---------------------------------------------------------------------------

class SAM2Tracker:
    """SAM2-backed implementation of the `Tracker` Protocol. See module
    docstring for scope and known limitations before integrating."""

    WINDOW_SIZE = 4
    UNCERTAIN_AREA_FRACTION = 1e-4  # mask covering less than this fraction -> UNCERTAIN

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        self._predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        self._buffer_dir = "sam2_stream_buffer"
        self._frame_count_in_window = 0

        self._target_id: Optional[int] = None
        self._box: Optional[BoundingBox] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_result: Optional[TrackingResult] = None

        self._reset_buffer_dir()

    # -- Protocol methods ----------------------------------------------

    def initialize(self, frame: Frame, box: BoundingBox, target_id: int) -> TrackingResult:
        """Stage `target_id` on the exact seed `frame`. Replaces any
        currently-tracked target (see limitation #1) and starts a fresh
        propagation window."""
        self._reset_buffer_dir()
        self._save_frame(frame.image)

        state = self._predictor.init_state(video_path=self._buffer_dir)
        self._predictor.reset_state(state)
        # No prior mask yet -> refine with the box's own center point.
        mask_logits = self._prompt_box_and_point(state, obj_id=1, box=box, point=box.center())

        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        result = self._build_result(target_id, frame.id, mask)

        self._target_id = target_id
        self._box = result.box or box
        self._last_mask = mask if result.status != TrackingStatus.LOST else None
        self._last_result = result
        return result

    def update(self, frame: Frame) -> tuple[TrackingResult, ...]:
        """Advance the current target by one live frame. Returns an empty
        tuple when no target is active."""
        if self._target_id is None:
            return ()

        self._save_frame(frame.image)

        if self._frame_count_in_window < self.WINDOW_SIZE:
            # Buffering: hold the last known result steady, re-tagged to
            # this frame_id. Smaller WINDOW_SIZE reduces this lag at the
            # cost of more frequent re-init overhead.
            held = self._retag(self._last_result, frame.id)
            self._last_result = held
            return (held,)

        result = self._propagate_and_slide(frame.id)
        return (result,)

    def remove(self, target_id: int) -> None:
        if self._target_id == target_id:
            self._target_id = None
            self._box = None
            self._last_mask = None
            self._last_result = None

    def reset(self) -> None:
        self._target_id = None
        self._box = None
        self._last_mask = None
        self._last_result = None
        self._reset_buffer_dir()

    # -- Internal helpers ------------------------------------------------

    def _propagate_and_slide(self, current_frame_id: int) -> TrackingResult:
        state = self._predictor.init_state(video_path=self._buffer_dir)
        self._predictor.reset_state(state)

        # Re-seed using the previous mask's CENTROID, not its bbox corners
        # (see limitation #2c) — robust to a stray pixel cluster skewing
        # the box, which is what let a bad mask lock in and persist.
        refine_point = (
            self._mask_centroid(self._last_mask)
            if self._last_mask is not None and self._last_mask.sum() > 0
            else self._box.center()
        )
        mask_logits = self._prompt_box_and_point(state, obj_id=1, box=self._box, point=refine_point)

        final_mask = None
        for _frame_idx, _obj_ids, mask_logits in self._predictor.propagate_in_video(state):
            final_mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

        result = self._build_result(self._target_id, current_frame_id, final_mask)
        self._last_result = result
        if result.status != TrackingStatus.LOST:
            self._box = result.box
            self._last_mask = final_mask
        else:
            self._last_mask = None

        self._slide_window_to_last_frame()
        return result

    def _prompt_box_and_point(self, state, obj_id: int, box: BoundingBox, point: tuple[float, float]):
        """Prompt frame_idx=0 with a box, then refine with a positive
        point. Two sequential calls, not one combined call — see
        limitation #2b for why this is deliberate, not a workaround."""
        self._predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=obj_id, box=box.as_xyxy(),
        )
        px, py = point
        _, _, mask_logits = self._predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=obj_id,
            points=np.array([[px, py]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        return mask_logits

    def _build_result(self, target_id: int, frame_id: int, mask: Optional[np.ndarray]) -> TrackingResult:
        if mask is None or mask.sum() == 0:
            return TrackingResult(target_id, frame_id, box=None, mask=None,
                                   confidence=None, status=TrackingStatus.LOST)

        area_fraction = mask.sum() / mask.size
        status = (TrackingStatus.UNCERTAIN if area_fraction < self.UNCERTAIN_AREA_FRACTION
                  else TrackingStatus.TRACKING)
        return TrackingResult(target_id, frame_id, box=self._mask_to_box(mask),
                               mask=mask.astype(bool), confidence=None, status=status)

    @staticmethod
    def _retag(result: TrackingResult, frame_id: int) -> TrackingResult:
        return TrackingResult(result.target_id, frame_id, result.box, result.mask,
                               result.confidence, result.status)

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[BoundingBox]:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())

    def _reset_buffer_dir(self):
        if os.path.exists(self._buffer_dir):
            shutil.rmtree(self._buffer_dir)
        os.makedirs(self._buffer_dir)
        self._frame_count_in_window = 0

    def _save_frame(self, image_rgb: np.ndarray):
        path = os.path.join(self._buffer_dir, f"{self._frame_count_in_window:05d}.jpg")
        Image.fromarray(image_rgb).save(path)
        self._frame_count_in_window += 1

    def _slide_window_to_last_frame(self):
        last_frame = sorted(glob.glob(os.path.join(self._buffer_dir, "*.jpg")))[-1]
        carry_over = self._buffer_dir + "_carry.jpg"
        shutil.copy(last_frame, carry_over)
        self._reset_buffer_dir()
        shutil.move(carry_over, os.path.join(self._buffer_dir, "00000.jpg"))
        self._frame_count_in_window = 1
