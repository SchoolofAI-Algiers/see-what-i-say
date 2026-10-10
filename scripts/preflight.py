"""Hardware/model preflight: run this FIRST on the demo machine and send back preflight_report.txt.

    uv run python scripts/preflight.py                # full check (stand in front of the camera, say something)
    uv run python scripts/preflight.py --skip-mic --skip-whisper
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:4}] {name}: {detail}", flush=True)


def check(name):
    def deco(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                record("FAIL", name, f"{type(e).__name__}: {e}")
                results.append(("TRACE", name, traceback.format_exc()))
                return None
        return wrapper
    return deco


@check("python/os")
def check_env():
    record("INFO", "python/os", f"{sys.version.split()[0]} on {platform.platform()}")


@check("torch/cuda")
def check_torch():
    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        record("PASS", "torch/cuda", f"torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.1f} GB VRAM")
        return True
    record("FAIL", "torch/cuda",
           f"torch {torch.__version__} cannot see a GPU (CPU-only wheel? On Windows install the CUDA torch build "
           "from the pytorch.org index, then re-run). Everything below will run on CPU = unusably slow.")
    return False


@check("sam2")
def check_sam2():
    from pathlib import Path

    import sam2  # noqa: F401

    ckpt = Path("checkpoints/sam2.1_hiera_tiny.pt")
    if not ckpt.exists() or ckpt.stat().st_size < 50_000_000:
        record("FAIL", "sam2", f"checkpoint missing/too small: {ckpt} -> run scripts/download_models.py")
        return False
    record("PASS", "sam2", f"package importable, checkpoint {ckpt.stat().st_size / 2**20:.0f} MB")
    return True


@check("camera")
def check_camera(index: int):
    import os

    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index} (try --camera 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frames, t0 = 0, time.perf_counter()
    last = None
    while frames < 30:
        ok, f = cap.read()
        if ok:
            frames += 1
            last = f
    fps = frames / (time.perf_counter() - t0)
    cap.release()
    cv2.imwrite("preflight_frame.jpg", last)
    record("PASS", "camera", f"{last.shape[1]}x{last.shape[0]} at ~{fps:.0f} FPS (snapshot: preflight_frame.jpg)")
    return True


@check("microphone")
def check_mic(device):
    import numpy as np
    import sounddevice as sd

    default = sd.query_devices(kind="input")
    print("  input devices:", [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0])
    print("  >>> speak now for 2 seconds (say: segment the person in red)", flush=True)
    audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32", device=device)
    sd.wait()
    rms = float(np.sqrt((audio**2).mean()))
    if rms < 1e-3:
        record("FAIL", "microphone", f"'{default['name']}' recorded silence (rms={rms:.5f}); pass --mic <index>")
        return None
    record("PASS", "microphone", f"'{default['name']}' rms={rms:.4f}")
    return audio[:, 0]


@check("whisper")
def check_whisper(model_name, audio):
    import whisper

    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    load_s = time.perf_counter() - t0
    if audio is None:
        import numpy as np

        audio = np.zeros(16000, dtype="float32")
    t0 = time.perf_counter()
    text = model.transcribe(audio.astype("float32"), language="en", fp16=False)["text"].strip()
    record("PASS", "whisper", f"{model_name}: load {load_s:.1f}s, transcribe {time.perf_counter() - t0:.1f}s, heard: {text!r}")


@check("full pipeline")
def check_pipeline(args):
    import cv2

    from stark_vision.contracts import Frame
    from stark_vision.main import App

    if args.mock:
        from stark_vision.mocks.adapters import MockGrounder, MockTracker

        from stark_vision.contracts import BoundingBox, GroundingCandidate

        grounder = MockGrounder((GroundingCandidate(BoundingBox(190, 100, 420, 440), 0.9, "person"),))
        tracker = MockTracker()
    else:
        from stark_vision.grounding.grounder import Grounder
        from stark_vision.segmentation.sam2_wrapper import SAM2Tracker

        t0 = time.perf_counter()
        grounder = Grounder(model_id=args.dino_model)
        tracker = SAM2Tracker(window_size=args.window, device=args.device)
        record("INFO", "model load", f"DINO + SAM2 in {time.perf_counter() - t0:.1f}s")

    cap = None
    if not args.skip_camera:
        import os

        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def next_frame(i):
        import numpy as np

        if cap is not None:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
        else:
            bgr = np.zeros((480, 640, 3), np.uint8)
            cv2.rectangle(bgr, (200 + i, 100), (400 + i, 440), (0, 0, 200), -1)
        return Frame(i, (i + 1) * 33_000_000, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    app = App(grounder, tracker)
    try:
        print("  >>> stand in front of the camera (a person must be visible)", flush=True)
        app.on_frame(next_frame(0))
        t0 = time.perf_counter()
        app.handle_command(app.parser.parse(f"segment {args.prompt}"))
        app.wait_for_grounding(120)
        cmd_s = time.perf_counter() - t0
        if app.manager.active is None:
            record("FAIL", "grounding->tracking", f"no target (message: {app.manager.message!r}); is a person visible? ambiguous?")
            return
        record("PASS", "grounding->tracking", f"target acquired in {cmd_s:.1f}s (command latency incl. first-run warmup)")

        # warmup + measure sustained tracking
        for i in range(1, 11):
            app.on_frame(next_frame(i))
        n, tracked, t0 = 40, 0, time.perf_counter()
        for i in range(11, 11 + n):
            app.on_frame(next_frame(i))
            a = app.manager.active
            tracked += int(a is not None and a.mask is not None and a.frame_id == i)
        fps = n / (time.perf_counter() - t0)
        status = "PASS" if fps >= 15 and tracked >= n * 0.8 else "WARN"
        record(status, "tracking throughput", f"{fps:.1f} FPS, mask present on {tracked}/{n} frames (want >=15 FPS)")
        for k, mode in enumerate(("blur him", "make him blue", "stop tracking")):
            app.handle_command(app.parser.parse(mode))
            app.on_frame(next_frame(11 + n + k))
        record("PASS", "commands", "blur / recolor / stop executed without re-grounding")
    finally:
        app.close()
        if cap is not None:
            cap.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mic", default=None)
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default="the person")
    p.add_argument("--mock", action="store_true", help="test the harness without models")
    p.add_argument("--skip-camera", action="store_true")
    p.add_argument("--skip-mic", action="store_true")
    p.add_argument("--skip-whisper", action="store_true")
    args = p.parse_args()

    check_env()
    if not args.mock:
        check_torch()
        check_sam2()
    if not args.skip_camera:
        check_camera(args.camera)
    audio = None
    if not args.skip_mic:
        mic = int(args.mic) if args.mic and str(args.mic).isdigit() else args.mic
        audio = check_mic(mic)
    if not args.skip_whisper:
        check_whisper(args.whisper_model, audio)
    check_pipeline(args)

    fails = [r for r in results if r[0] == "FAIL"]
    lines = [f"[{s}] {n}: {d}" for s, n, d in results]
    lines.append("")
    lines.append("RESULT: " + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED"))
    open("preflight_report.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n" + lines[-1] + "  (details saved to preflight_report.txt)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
