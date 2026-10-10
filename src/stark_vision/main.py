"""STARK VISION runtime: webcam + push-to-talk + grounding + SAM 2 + effects + HUD.

    uv run python -m stark_vision.main

Keys (OpenCV window):
    SPACE  start/stop recording a voice command (press, speak, press again)
    T      type a command instead (ENTER sends, ESC cancels)
    M      manual target selection (draw a box, ENTER/SPACE to confirm)
    1-4    highlight / recolor(blue) / blur / isolate
    R, C   stop tracking and clear
    ESC    quit

Threading rules (docs/interfaces.md): only the vision loop (App.process_frame /
App.handle_command) mutates TargetManager and the tracker. Workers only send
immutable requests/results through queues.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from stark_vision.camera.frame_buffer import FrameBuffer
from stark_vision.contracts import (
    BoundingBox,
    CommandEvent,
    CommandType,
    EffectMode,
    EffectSpec,
    Frame,
    GroundingCandidate,
    GroundingRequest,
    GroundingResult,
    RenderedFrame,
    RuntimeSnapshot,
    TrackingStatus,
)
from stark_vision.effects.engine import EffectEngine
from stark_vision.hud.overlay import HudRenderer
from stark_vision.speech.command_parser import CommandParser
from stark_vision.target.target_manager import TargetManager

logger = logging.getLogger("stark_vision.main")

WINDOW = "STARK VISION"
EFFECT_KEYS = {
    ord("1"): EffectSpec(EffectMode.HIGHLIGHT),
    ord("2"): EffectSpec(EffectMode.RECOLOR, (0, 80, 255)),
    ord("3"): EffectSpec(EffectMode.BLUR),
    ord("4"): EffectSpec(EffectMode.ISOLATE),
}


class GroundingWorker:
    """Runs the (slow) grounder off the vision loop."""

    def __init__(self, grounder) -> None:
        self._grounder = grounder
        self._requests: queue.Queue[GroundingRequest | None] = queue.Queue()
        self._results: queue.Queue[GroundingResult] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="grounding", daemon=True)
        self._thread.start()

    def submit(self, request: GroundingRequest) -> None:
        self._requests.put(request)

    def poll(self) -> GroundingResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._requests.put(None)

    def _run(self) -> None:
        while True:
            req = self._requests.get()
            if req is None:
                return
            try:
                candidates = tuple(self._grounder.find(req.frame, req.prompt))
            except Exception:  # noqa: BLE001 - keep the demo alive
                logger.exception("grounding failed for %r", req.prompt)
                candidates = ()
            self._results.put(GroundingResult(req.request_id, req.frame.id, candidates))


class App:
    """Vision-loop orchestration. Adapters are injected so it can run headless with mocks."""

    def __init__(self, grounder, tracker, *, frames: FrameBuffer | None = None) -> None:
        self.grounder = grounder
        self.tracker = tracker
        self.frames = frames or FrameBuffer()
        self.manager = TargetManager()
        self.parser = CommandParser()
        self.effects = EffectEngine()
        self.hud = HudRenderer()
        self.worker = GroundingWorker(grounder)
        self.commands: queue.Queue[CommandEvent] = queue.Queue()
        self.last_heard = ""
        self.latency_ms = 0.0
        self.fps = 0.0
        self._next_request_id = 1
        self._cmd_t0: float | None = None
        self._last_t = time.perf_counter()

    # ---- commands (vision loop only) --------------------------------------

    def handle_command(self, cmd: CommandEvent) -> None:
        if cmd.transcript:
            self.last_heard = cmd.transcript
        if cmd.type is CommandType.SELECT_TARGET and cmd.prompt:
            self._begin_selection(cmd.prompt)
        elif cmd.type is CommandType.RESOLVE_CANDIDATE:
            chosen = self.manager.resolve_candidate(cmd.selector)
            if chosen is not None:
                self._stage(chosen)
            else:
                self.manager.message = "Nothing to choose between"
        elif cmd.type is CommandType.SET_EFFECT and cmd.effect is not None:
            self.manager.set_effect(cmd.effect)
        elif cmd.type is CommandType.FOLLOW:
            active = self.manager.active
            self.manager.message = f"Following {active.prompt}" if active else "No active target to follow"
        elif cmd.type is CommandType.STOP_TRACKING:
            self.stop()
        else:
            heard = cmd.transcript.strip()
            self.manager.message = f"Unknown command: {heard}" if heard else "Didn't catch that"

    def stop(self) -> None:
        self.tracker.reset()
        self.manager.stop()

    def _begin_selection(self, prompt: str) -> None:
        latest = self.frames.latest
        if latest is None:
            self.manager.message = "No camera frame yet"
            return
        old_pending = self.manager.pending_target_id()
        if old_pending is not None:
            self.tracker.remove(old_pending)
        request_id = self._next_request_id
        self._next_request_id += 1
        self._cmd_t0 = time.perf_counter()
        self.manager.begin_acquisition(request_id, prompt, latest.id)
        self.worker.submit(GroundingRequest(request_id, prompt, latest))

    def select_manual_box(self, box: BoundingBox) -> None:
        """Operator fallback: same downstream tracker/effects, no speech or grounding."""
        latest = self.frames.latest
        if latest is None:
            return
        request_id = self._next_request_id
        self._next_request_id += 1
        self._cmd_t0 = time.perf_counter()
        self.manager.begin_acquisition(request_id, "manual selection", latest.id)
        self.manager.receive_candidates(request_id, [GroundingCandidate(box, 1.0, "manual")])
        selected = self.manager.pending.selected if self.manager.pending else None
        if selected is not None:
            self._stage(selected)

    def wait_for_grounding(self, timeout: float = 60.0) -> bool:
        """Block until the pending request resolves (offline/video use; never in the live loop)."""
        deadline = time.perf_counter() + timeout
        while self.manager.pending is not None and time.perf_counter() < deadline:
            self._poll_grounding()
            time.sleep(0.02)
        return self.manager.pending is None

    def _poll_grounding(self) -> None:
        while (result := self.worker.poll()) is not None:
            selected = self.manager.receive_grounding_result(result)
            if selected is not None:
                self._stage(selected)

    def _stage(self, candidate: GroundingCandidate) -> None:
        pending = self.manager.pending
        if pending is None:
            return
        seed = self.frames.get(pending.source_frame_id)
        buffered = self.frames.after(pending.source_frame_id)
        if seed is None or buffered is None:
            # Seed history expired: ground again on the newest frame.
            newest = self.frames.latest
            if newest is None:
                self.manager.abort_pending("No frame available to retry")
                return
            self.manager.update_seed_frame(pending.request_id, newest.id)
            self.worker.submit(GroundingRequest(pending.request_id, pending.prompt, newest))
            self.manager.message = "Re-acquiring on newest frame"
            return

        target_id = self.manager.pending_target_id()
        if target_id is None:
            return
        result = self.tracker.initialize(seed, candidate.box, target_id)
        if result.mask is None or result.status is TrackingStatus.LOST:
            self.manager.abort_pending("Tracker could not initialize the target")
            self.tracker.remove(target_id)
            return
        pending_result = result
        for frame in buffered:  # catch up from the seed to the present
            results = self.tracker.update(frame)
            self.manager.observe(results)
            pending_result = next((r for r in results if r.target_id == target_id), pending_result)
        old_id = self.manager.promote(pending_result)
        if self.manager.pending is not None:
            failed = self.manager.pending_target_id()
            self.manager.abort_pending("New target was lost before promotion")
            if failed is not None:
                self.tracker.remove(failed)
        if old_id is not None:
            self.tracker.remove(old_id)
        if self._cmd_t0 is not None:
            self.latency_ms = (time.perf_counter() - self._cmd_t0) * 1000.0

    # ---- per frame (vision loop only) ------------------------------------

    def on_frame(self, frame: Frame) -> RenderedFrame:
        self.frames.append(frame)
        return self.process_frame(frame)

    def process_frame(self, frame: Frame) -> RenderedFrame:
        while True:
            try:
                self.handle_command(self.commands.get_nowait())
            except queue.Empty:
                break
        self._poll_grounding()
        results = self.tracker.update(frame)
        self.manager.observe(results)

        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(1e-6, now - self._last_t)
        self._last_t = now

        active = self.manager.active
        mask = active.mask if active and active.frame_id == frame.id else None
        image = self.effects.apply(frame.image, mask, self.manager.effect)
        pending = self.manager.pending
        snapshot = RuntimeSnapshot(
            frame_id=frame.id,
            active_target=active,
            pending_prompt=pending.prompt if pending else None,
            pending_candidates=pending.candidates if pending else (),
            pending_is_ambiguous=bool(pending and pending.ambiguous),
            effect=self.manager.effect,
            message=self.manager.message,
            capture_fps=self.fps,
            tracking_fps=self.fps,
        )
        return self.hud.render(image, snapshot)

    def close(self) -> None:
        self.worker.close()


def draw_overlay(app: App, rendered: RenderedFrame, status: str, typing: str | None) -> np.ndarray:
    """Annotate the HUD frame (RGB in, BGR out for cv2.imshow)."""
    img = np.ascontiguousarray(rendered.image)
    pending = app.manager.pending
    if pending and pending.ambiguous:
        for i, cand in enumerate(pending.candidates, start=1):
            cv2.putText(img, f"#{i}", (int(cand.box.x1) + 4, int(cand.box.y1) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 25), 2, cv2.LINE_AA)
    h = img.shape[0]
    lines = [
        *rendered.telemetry[1:],  # skip frame id
        f"heard: {app.last_heard}" if app.last_heard else "heard: -",
        f"FPS {app.fps:4.1f} | command latency {app.latency_ms:5.0f} ms | {status}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 28 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 245, 255), 1, cv2.LINE_AA)
    footer = (f"> {typing}_" if typing is not None
              else "SPACE talk | T type | M manual | 1-4 effects | R reset | ESC quit")
    cv2.putText(img, footer, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class Camera:
    """Capture thread: converts BGR->RGB once, stamps ids/timestamps, fills the FrameBuffer."""

    def __init__(self, device: int, width: int, height: int, frames: FrameBuffer) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frames = frames
        self._running = True
        self._next_id = 0
        self._last_ts = 0
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue
            ts = max(time.monotonic_ns(), self._last_ts + 1)
            self._last_ts = ts
            frame = Frame(self._next_id, ts, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self._next_id += 1
            self.frames.append(frame)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def run(args: argparse.Namespace) -> None:
    from stark_vision.grounding.grounder import Grounder
    from stark_vision.segmentation.sam2_wrapper import SAM2Tracker

    logger.info("Loading Grounding DINO (%s)...", args.dino_model)
    grounder = Grounder(model_id=args.dino_model)
    logger.info("Loading SAM 2...")
    tracker = SAM2Tracker(args.sam_config, args.sam_checkpoint, device=args.device, window_size=args.window)

    recorder = stt = None
    if not args.no_speech:
        try:
            from stark_vision.speech.recorder import AudioRecorder
            from stark_vision.speech.stt import SpeechToText

            device = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
            recorder = AudioRecorder(device=device)
            logger.info("Loading Whisper (%s)...", args.whisper_model)
            stt = SpeechToText(args.whisper_model)
        except Exception:  # noqa: BLE001
            logger.exception("Speech unavailable; use T to type commands")
            recorder = stt = None

    frames = FrameBuffer()
    app = App(grounder, tracker, frames=frames)
    camera = Camera(args.camera, args.width, args.height, frames)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    status = "ready"
    typing: str | None = None
    transcribing = threading.Event()
    last_id = -1

    def transcribe(audio) -> None:
        try:
            text = stt.transcribe(audio).text
            app.commands.put(app.parser.parse(text))
        except Exception:  # noqa: BLE001
            logger.exception("transcription failed")
            app.commands.put(CommandEvent(CommandType.UNKNOWN, ""))
        finally:
            transcribing.clear()

    try:
        while True:
            frame = frames.latest
            if frame is None or frame.id == last_id:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 and typing is None:
                    break
                continue
            last_id = frame.id
            rendered = app.process_frame(frame)
            if recorder is not None and recorder.recording:
                status = "RECORDING"
            elif transcribing.is_set():
                status = "transcribing"
            else:
                status = "ready"
            cv2.imshow(WINDOW, draw_overlay(app, rendered, status, typing))
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if typing is not None:  # typing mode swallows keys
                if key == 13:
                    app.commands.put(app.parser.parse(typing))
                    typing = None
                elif key == 27:
                    typing = None
                elif key in (8, 127):
                    typing = typing[:-1]
                elif 32 <= key <= 126:
                    typing += chr(key)
                continue

            if key == 27:
                break
            elif key == 32 and recorder is not None and stt is not None:
                if recorder.recording:
                    audio = recorder.stop()
                    transcribing.set()
                    threading.Thread(target=transcribe, args=(audio,), daemon=True).start()
                elif not transcribing.is_set():
                    recorder.start()
            elif key in (ord("t"), ord("T")):
                typing = ""
            elif key in (ord("m"), ord("M")):
                latest = frames.latest
                if latest is not None:
                    bgr = cv2.cvtColor(latest.image, cv2.COLOR_RGB2BGR)
                    x, y, w, h = cv2.selectROI(WINDOW, bgr, showCrosshair=False)
                    if w > 4 and h > 4:
                        app.select_manual_box(BoundingBox(float(x), float(y), float(x + w), float(y + h)))
            elif key in EFFECT_KEYS:
                app.manager.set_effect(EFFECT_KEYS[key])
            elif key in (ord("r"), ord("R"), ord("c"), ord("C")):
                app.stop()
    finally:
        camera.close()
        app.close()
        tracker.close()
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="STARK VISION")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    p.add_argument("--window", type=int, default=4, help="SAM 2 propagation window (frames)")
    p.add_argument("--device", default=None)
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--mic", default=None, help="input device index or name")
    p.add_argument("--no-speech", action="store_true")
    args = p.parse_args()
    from stark_vision.utils.logging_setup import setup_logging

    setup_logging(log_file="logs/stark_vision.log")
    print("STARK VISION starting (models load first; the window appears when they are ready)...", flush=True)
    run(args)


if __name__ == "__main__":
    main()
