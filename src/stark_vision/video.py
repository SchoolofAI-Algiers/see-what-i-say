"""Run the full pipeline on a video file (no webcam/mic): demos, tests, Colab."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from stark_vision.contracts import Frame


def process_video(src, dst, app, commands=(), *, command_frame: int = 0,
                  max_frames: int | None = None, ground_timeout: float = 120.0) -> dict:
    """Feed `src` through `app`, issuing `commands` (text) at `command_frame`, writing `dst`."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step_ns = int(1e9 / fps)
    writer = None
    n = tracked = 0
    t0 = time.perf_counter()
    try:
        while max_frames is None or n < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = Frame(n, (n + 1) * step_ns, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if n == command_frame:
                app.frames.append(frame)
                for text in commands:
                    app.handle_command(app.parser.parse(text))
                    app.wait_for_grounding(ground_timeout)
                rendered = app.process_frame(frame)
            else:
                rendered = app.on_frame(frame)
            active = app.manager.active
            if active is not None and active.frame_id == frame.id and active.mask is not None:
                tracked += 1
            out = cv2.cvtColor(rendered.image, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = out.shape[:2]
                ext = Path(str(dst)).suffix.lower()
                fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "MJPG"))
                writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
            writer.write(out)
            n += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    secs = time.perf_counter() - t0
    return {"frames": n, "tracked_frames": tracked, "seconds": secs, "fps": n / secs if secs else 0.0}
