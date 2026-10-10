import numpy as np

from stark_vision.contracts import BoundingBox, Frame, TrackingStatus
from stark_vision.segmentation.sam2_wrapper import HoldTracker, _TargetState


def test_hold_tracker_returns_held_mask_without_models():
    t = object.__new__(HoldTracker)  # skip __init__ (needs torch/SAM 2)
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    t._targets = {1: _TargetState(BoundingBox(2, 2, 5, 5), mask, 0.9)}
    frame = Frame(3, 3, np.zeros((8, 8, 3), np.uint8))
    (r,) = t.update(frame)
    assert r.target_id == 1 and r.frame_id == 3 and r.status is TrackingStatus.TRACKING
    assert np.array_equal(r.mask, mask)
