"""Simulate the LIVE app with a video file as the camera and a WAV file as the microphone.

Runs the real threads, Whisper, parser, grounding worker, tracker, effects and HUD in real time,
without a webcam/mic/window. Good for Colab GPU runs.

    python scripts/simulate_live.py in.mp4 --wav command.wav --then "blur him" --then "stop tracking" --out sim.mp4
    python scripts/simulate_live.py in.mp4 --text "segment the person in red" --then "make him blue"
    python scripts/simulate_live.py in.mp4 --mock --text "segment the person"     # harness test, no models

Make a test WAV with a phone/laptop recorder, or:  pip install gTTS ; gTTS("segment the person in red").save("c.mp3")
(convert to wav with ffmpeg: ffmpeg -i c.mp3 -ar 16000 -ac 1 command.wav)
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from stark_vision.camera.frame_buffer import FrameBuffer
from stark_vision.contracts import AudioBuffer
from stark_vision.main import App, draw_overlay
from stark_vision.utils.logging_setup import setup_logging
from stark_vision.video import VideoCamera


def load_wav(path):
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return AudioBuffer(data.astype(np.float32), int(sr))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--wav", help="spoken first command (goes through Whisper)")
    p.add_argument("--text", help="first command as text (skips Whisper)")
    p.add_argument("--then", action="append", default=[], help="follow-up text commands, one per --gap seconds")
    p.add_argument("--gap", type=float, default=5.0)
    p.add_argument("--first-at", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out", default="sim_live.mp4")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--tracker", choices=["sam2", "hold"], default="sam2")
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--whisper-model", default="small.en")
    a = p.parse_args()
    setup_logging()
    if not (a.wav or a.text):
        p.error("give --wav or --text")

    stt = None
    if a.wav:
        from stark_vision.speech.stt import SpeechToText

        stt = SpeechToText(a.whisper_model)
        audio = load_wav(a.wav)

    if a.mock:
        from stark_vision.contracts import BoundingBox, GroundingCandidate
        from stark_vision.mocks.adapters import MockGrounder, MockTracker

        grounder = MockGrounder((GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.9, "person"),))
        tracker = MockTracker()
    else:
        from stark_vision.grounding.grounder import Grounder
        from stark_vision.segmentation.sam2_wrapper import HoldTracker, SAM2Tracker

        grounder = Grounder(model_id=a.dino_model)
        cls = HoldTracker if a.tracker == "hold" else SAM2Tracker
        tracker = cls(a.sam_config, a.sam_checkpoint, device=a.device, window_size=a.window)

    frames = FrameBuffer()
    app = App(grounder, tracker, frames=frames)
    cam = VideoCamera(a.video, frames)

    events: list[tuple[float, str, str]] = []
    events.append((a.first_at, "speech" if a.wav else "text", a.text or ""))
    for i, text in enumerate(a.then):
        events.append((a.first_at + a.gap * (i + 1), "text", text))
    duration = a.duration or (events[-1][0] + a.gap + 5.0)

    heard: list[str] = []

    def speak():
        text = stt.transcribe(audio).text
        heard.append(text)
        app.commands.put(app.parser.parse(text))

    writer = None
    processed = tracked = 0
    first_target_at = None
    start = time.perf_counter()
    last_id = -1
    try:
        while (now := time.perf_counter() - start) < duration:
            while events and events[0][0] <= now:
                _, kind, payload = events.pop(0)
                if kind == "speech":
                    threading.Thread(target=speak, daemon=True).start()
                else:
                    app.commands.put(app.parser.parse(payload))
            frame = frames.latest
            if frame is None or frame.id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame.id
            rendered = app.process_frame(frame)
            out = draw_overlay(app, rendered, "simulated", None)
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (w, h))
            writer.write(out)
            processed += 1
            act = app.manager.active
            if act is not None and act.mask is not None and act.frame_id == frame.id:
                tracked += 1
                if first_target_at is None:
                    first_target_at = now
    finally:
        cam.close()
        app.close()
        if writer is not None:
            writer.release()

    print("\n===== SIMULATED LIVE RESULT =====")
    print(f"Whisper heard      : {heard[0]!r}" if heard else "Whisper heard      : (no speech used)")
    print(f"first target at    : {first_target_at:.1f}s" if first_target_at else "first target at    : NEVER")
    print(f"frames processed   : {processed} in {duration:.0f}s = {processed / duration:.1f} FPS")
    print(f"mask on frames     : {tracked}/{processed}")
    print(f"final message      : {app.manager.message!r}")
    print(f"command latency    : {app.latency_ms:.0f} ms (last target)")
    print(f"output video       : {a.out}")
    raise SystemExit(0 if first_target_at else 1)


if __name__ == "__main__":
    main()
