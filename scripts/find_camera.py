"""Find which camera index gives a real image (index 0 is often an IR/virtual camera on Windows).

    uv run python scripts/find_camera.py
Open the saved camera_N.jpg files to see which one shows you.
"""

import os

import cv2

backends = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)] if os.name == "nt" else [("default", cv2.CAP_ANY)]
for index in range(6):
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        frame = None
        for _ in range(20):  # let auto-exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            print(f"camera {index} [{name}]: opens but returns no frames")
            continue
        brightness = float(frame.mean())
        verdict = "BLACK (probably IR/virtual - skip)" if brightness < 3 else "looks live"
        path = f"camera_{index}_{name}.jpg"
        cv2.imwrite(path, frame)
        print(f"camera {index} [{name}]: {frame.shape[1]}x{frame.shape[0]}, brightness {brightness:.0f} -> {verdict}  ({path})")
