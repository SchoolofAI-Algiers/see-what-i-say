"""
Manual/visual smoke test for SAM2Tracker — NOT part of the Tracker
component itself. Opens a local webcam; press 'i' to pause and drag a
box around your actual target with the mouse (via cv2.selectROI), then
confirm with Enter/Space. Shows the tracked mask live afterward.

Why drag-select instead of a fixed box: a hardcoded box size/position
(e.g. a fixed 120x120 box at frame center) has no relationship to how
large your actual subject is in frame, which produces a poor initial
prompt independent of anything in SAM2Tracker itself. Letting you draw
the real box is a test-harness fix, not a component change — the
Tracker Protocol only specifies the BoundingBox *type*, not how a
caller constructs one.

This harness wraps the (synchronous, per-spec) SAM2Tracker in a small
background-thread runner purely so the camera preview stays smooth while
SAM2 inference runs — on CPU a single update() can take several seconds.
That decoupling is a convenience of THIS test script only. interfaces.md
specifies "All calls are serialized on the vision loop," so the real
vision loop should call SAM2Tracker synchronously and manage pacing
itself (e.g. dropping frames to the latest available), not rely on this
threading trick.

Run from the sam2 repo root (adjust the import below once this file and
sam2_tracker.py live inside the actual stark_vision repo — see
limitation #6 in sam2_tracker.py's docstring):

    python test_sam2_tracker_manual.py
"""
import itertools
import sys
import threading
import time

import cv2
import numpy as np

from sam2_tracker import SAM2Tracker, Frame, BoundingBox

CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_tiny.pt"
TARGET_ID = 1


class _HarnessWorker:
    """Test-only background runner — see module docstring. Not part of
    the Tracker contract."""

    def __init__(self, tracker: SAM2Tracker):
        self.tracker = tracker
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_result = None
        self.pending_init_box = None  # BoundingBox set by main thread after selectROI
        self.running = False
        self._frame_ids = itertools.count()
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def submit(self, image_rgb: np.ndarray):
        frame = Frame(id=next(self._frame_ids), timestamp_ns=time.time_ns(), image=image_rgb)
        with self.lock:
            self.latest_frame = frame

    def request_initialize(self, box: BoundingBox):
        with self.lock:
            self.pending_init_box = box

    def get_display_result(self):
        with self.lock:
            return self.latest_result

    def _loop(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
                init_box = self.pending_init_box
                self.pending_init_box = None

            if frame is None:
                time.sleep(0.01)
                continue

            if init_box is not None:
                result = self.tracker.initialize(frame, init_box, TARGET_ID)
                with self.lock:
                    self.latest_result = result
                continue

            results = self.tracker.update(frame)
            if results:
                with self.lock:
                    self.latest_result = results[0]
            else:
                time.sleep(0.01)


def run():
    tracker = SAM2Tracker(CONFIG_PATH, CHECKPOINT_PATH, device="cpu")
    worker = _HarnessWorker(tracker)
    worker.start()

    # DirectShow is a Windows-only backend. Let OpenCV select its normal
    # backend on Linux/macOS (including remote Linux Studio environments).
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        worker.stop()
        cap.release()
        raise RuntimeError(
            "Could not open camera 0. This manual test requires a webcam device "
            "to be exposed to this process; a remote Studio may not have access "
            "to your laptop's webcam."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):  # let auto-exposure/white-balance settle
        ret, _ = cap.read()
        if not ret:
            worker.stop()
            cap.release()
            raise RuntimeError(
                "Camera 0 opened but did not return frames. Check that the "
                "camera is connected, permitted, and not in use by another process."
            )

    print("Press 'i' to pause and drag a box around your target (Enter/Space to confirm, "
          "'c' to cancel). Press 'q' to quit.")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Camera read failed; stopping the manual test.")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            worker.submit(frame_rgb)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                # Blocking GUI selection on the current frame. Runs in the
                # main thread; the worker thread keeps running independently.
                x, y, w, h = cv2.selectROI("Drag a box, then press Enter", frame_bgr,
                                            fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Drag a box, then press Enter")
                if w > 0 and h > 0:
                    box = BoundingBox(float(x), float(y), float(x + w), float(y + h))
                    worker.request_initialize(box)
                else:
                    print("Selection cancelled or empty — not initializing.")

            display = frame_bgr.copy()
            result = worker.get_display_result()
            status_text = "IDLE"
            if result is not None:
                status_text = result.status.value
                if result.mask is not None:
                    mask = result.mask
                    if mask.shape[:2] != display.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (display.shape[1], display.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    display[mask] = (0.5 * display[mask] + 0.5 * np.array([255, 144, 30])).astype(np.uint8)

            cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("SAM2Tracker manual test", display)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
