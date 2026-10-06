# test_webcam_loop.py
import cv2
import numpy as np
from async_tracker import AsyncTracker


def run_webcam_test(config_path, checkpoint_path, device="cpu"):
    tracker = AsyncTracker(config_path, checkpoint_path, device=device)
    tracker.start()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Press 'i' to initialize on a center point, 'q' to quit.")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tracker.submit_frame(frame_rgb)  # non-blocking, always fresh for the worker

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                tracker.request_initialize()

            mask, status = tracker.get_display_state()  # may be a few frames stale — that's fine
            display = frame_bgr.copy()
            if mask is not None:
                if mask.shape[:2] != display.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (display.shape[1], display.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                display[mask] = (0.5 * display[mask] + 0.5 * np.array([255, 144, 30])).astype(np.uint8)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("SAM2 Wrapper Test", display)
    finally:
        tracker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_test("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", device="cpu")