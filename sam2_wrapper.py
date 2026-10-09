import os
import glob
import shutil
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


class SAM2Wrapper:
    """
    Implements: segmenter.initialize(frame, prompt) -> target_state
                tracker.update(frame, target_state) -> mask, tracking_state

    Backed by REAL SAM2 video-predictor propagation (temporal memory),
    using only the public/stable API. Since that API requires a known
    frame set (init_state(video_path=...)) rather than one live frame
    at a time, frames are buffered into a sliding window; every
    WINDOW_SIZE frames we propagate through the real SAM2 tracker
    across that window, then slide forward re-anchored on the last
    confident mask.

    This is NOT the lowest-latency version. Grounded-SAM-2's own
    grounded_sam2_tracking_camera_with_continuous_id.py achieves true
    per-frame streaming by patching sam2_video_predictor.py itself
    (video_path=None + process_stream_frame). That's the Day-11
    performance upgrade once we're on GPU hardware — this gets us
    real SAM2 propagation today, correctly, on CPU.
    """

    WINDOW_SIZE = 4  # frames per window; smaller = less lag, more re-init overhead

    def __init__(self, config_path, checkpoint_path, device="cpu"):
        self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        self.buffer_dir = "sam2_stream_buffer"
        self.frame_count_in_window = 0
        self.last_mask = None
        self.last_bbox = None
        self._reset_buffer_dir()

    def _reset_buffer_dir(self):
        if os.path.exists(self.buffer_dir):
            shutil.rmtree(self.buffer_dir)
        os.makedirs(self.buffer_dir)
        self.frame_count_in_window = 0

    def _save_frame(self, frame_rgb):
        path = os.path.join(self.buffer_dir, f"{self.frame_count_in_window:05d}.jpg")
        Image.fromarray(frame_rgb).save(path)
        self.frame_count_in_window += 1

    def initialize(self, frame_rgb, prompt_or_box):
        """Start a brand-new target from a point [x, y] or box [x1, y1, x2, y2]."""
        self._reset_buffer_dir()
        self._save_frame(frame_rgb)

        state = self.predictor.init_state(video_path=self.buffer_dir)
        self.predictor.reset_state(state)

        if len(prompt_or_box) == 2:
            _, _, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=[prompt_or_box], labels=[1],
            )
        else:
            _, _, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                box=np.array(prompt_or_box),
            )

        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        self.last_mask = mask
        self.last_bbox = self._mask_to_bbox(mask)

        return {"id": 1, "mask": mask, "bbox": self.last_bbox, "tracking_status": "TRACKING"}

    def update(self, frame_rgb, target_state):
        """Advance the target by one live frame using real SAM2 propagation."""
        if self.last_bbox is None:
            return None, "LOST"

        self._save_frame(frame_rgb)

        if self.frame_count_in_window < self.WINDOW_SIZE:
            # Buffering — hold the last known mask steady until the window fills.
            # Smaller WINDOW_SIZE reduces this visible lag.
            return self.last_mask, target_state.get("tracking_status", "TRACKING")

        return self._propagate_and_slide(target_state)

    def _propagate_and_slide(self, target_state):
        state = self.predictor.init_state(video_path=self.buffer_dir)
        self.predictor.reset_state(state)
        self.predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=np.array(self.last_bbox),
        )

        final_mask = self.last_mask
        for _frame_idx, _obj_ids, mask_logits in self.predictor.propagate_in_video(state):
            final_mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

        ok = final_mask.sum() > 0
        status = "TRACKING" if ok else "RECOVERY"
        if ok:
            self.last_mask = final_mask
            self.last_bbox = self._mask_to_bbox(final_mask)
            target_state.update(mask=final_mask, bbox=self.last_bbox, tracking_status=status)

        # Copy the last frame OUT before wiping the buffer dir, not after
        last_frame = sorted(glob.glob(os.path.join(self.buffer_dir, "*.jpg")))[-1]
        carry_over = self.buffer_dir + "_carry.jpg"
        shutil.copy(last_frame, carry_over)

        self._reset_buffer_dir()
        shutil.move(carry_over, os.path.join(self.buffer_dir, "00000.jpg"))
        self.frame_count_in_window = 1

        return final_mask, status

    @staticmethod
    def _mask_to_bbox(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
