"""test of run_video"""
import cv2
import numpy as np

from stark_vision.main import App
from stark_vision.mocks.adapters import MockGrounder, MockTracker
from stark_vision.contracts import BoundingBox, GroundingCandidate
from stark_vision.video import process_video


def _make_video(path, n=20):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 64))
    for i in range(n):
        img = np.zeros((64, 64, 3), np.uint8)
        cv2.rectangle(img, (10 + i, 14), (30 + i, 48), (0, 0, 200), -1)
        w.write(img)
    w.release()


def test_process_video_mock(tmp_path):
    src, dst = tmp_path / "in.avi", tmp_path / "out.avi"
    _make_video(src)
    grounder = MockGrounder((GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.9, "person"),))
    app = App(grounder, MockTracker())
    try:
        stats = process_video(src, dst, app, ["segment the person", "blur him"], ground_timeout=10)
    finally:
        app.close()
    assert stats["frames"] == 20
    assert stats["tracked_frames"] >= 15
    assert len(grounder.calls) == 1
    assert dst.exists() and dst.stat().st_size > 0
