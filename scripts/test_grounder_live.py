#!/usr/bin/env python3
"""
STARK VISION — JarvisHUD Live Interactive Grounding Engine Runner.

Features 100% on-screen interactive HUD text input directly within the OpenCV
window (no terminal freezing), real-time threshold tuning, dynamic box clearing,
rotating Arc Reactor telemetry, and tactical Iron Man reticle overlays.

Controls (All inside the OpenCV Window):
    Type any text       : Updates search query live on-screen
    BACKSPACE           : Delete character
    ENTER               : Ground current frame with search query
    SPACE               : Rescan with active prompt (or add space while typing)
    C                   : Clear search buffer and clear all active detections
    + / UP ARROW        : Increase detection threshold (less sensitive)
    - / DOWN ARROW      : Decrease detection threshold (more sensitive / faint targets)
    S                   : Save annotated frame snapshot
    ESC                 : Quit application
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# Add src to path if executed standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stark_vision.grounding.grounder import Grounder, GroundingCandidate, is_ambiguous
from stark_vision.hud.overlay import JarvisHUD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stark_vision.live_demo")


def run_on_webcam(
    grounder: Grounder,
    camera_id: int,
    initial_prompt: str,
    box_threshold: float,
    text_threshold: float,
    skip_boot: bool = False,
) -> None:
    """
    Run interactive real-time grounding on webcam with direct on-screen keyboard input.
    """
    logger.info("Connecting to camera device %d...", camera_id)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Failed to open camera %d. Check /dev/video* permissions.", camera_id)
        return

    hud = JarvisHUD(enable_voice=not skip_boot)
    if skip_boot:
        hud.is_booted = True

    active_prompt = initial_prompt
    search_buffer = initial_prompt
    b_thresh = box_threshold
    t_thresh = text_threshold

    candidates: list[GroundingCandidate] = []
    last_latency_ms = 0.0
    ambiguous = False
    prev_time = time.perf_counter()
    fps = 0.0
    status_msg = "Ready"

    window_name = "STARK VISION — Interactive Grounding Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Initial acquisition
    ret, frame = cap.read()
    if ret and active_prompt:
        t0 = time.perf_counter()
        candidates = grounder.find(frame, active_prompt, b_thresh, t_thresh)
        last_latency_ms = (time.perf_counter() - t0) * 1000
        ambiguous = is_ambiguous(candidates)

    logger.info("JarvisHUD online. Type directly inside the OpenCV window.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - prev_time))
            prev_time = now

            display_frame = frame.copy()

            # 1. Boot Animation Sequence (Runs for first 2.5s on startup)
            if not hud.is_booted:
                boot_complete = hud.render_boot_sequence(display_frame)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Any key bypasses boot animation immediately
                    hud.is_booted = True
                continue

            # 2. Main High-Tech JarvisHUD Overlay
            hud.render(
                frame=display_frame,
                candidates=candidates,
                search_buffer=search_buffer,
                active_prompt=active_prompt,
                box_threshold=b_thresh,
                latency_ms=last_latency_ms,
                fps=fps,
                ambiguous=ambiguous,
                status_msg=status_msg,
            )

            cv2.imshow(window_name, display_frame)

            # Handle Keystrokes inside OpenCV Window (NO terminal input!)
            key = cv2.waitKey(1)
            if key == -1:
                continue

            key_code = key & 0xFF

            # ESC or 'q' when search is empty: Clean shutdown
            if key_code == 27:
                break
            elif key_code in (ord("q"), ord("Q")) and not search_buffer:
                break

            # ENTER: Trigger Grounding with search buffer
            elif key_code in (13, 10):
                active_prompt = search_buffer.strip()
                if active_prompt:
                    status_msg = f"Scanning '{active_prompt}'..."
                    logger.info("Grounding frame against prompt: '%s' (thresh: %.2f)...", active_prompt, b_thresh)
                    t0 = time.perf_counter()
                    candidates = grounder.find(frame, active_prompt, b_thresh, t_thresh)
                    last_latency_ms = (time.perf_counter() - t0) * 1000
                    ambiguous = is_ambiguous(candidates)
                    status_msg = f"Locked: {len(candidates)}" if candidates else "0 Targets Found"
                else:
                    candidates = []
                    status_msg = "Cleared"

            # BACKSPACE: Delete character from search buffer
            elif key_code in (8, 127, 255):
                if search_buffer:
                    search_buffer = search_buffer[:-1]

            # 'C' or 'c': Clear detections and search buffer
            elif key_code in (ord("c"), ord("C")) and len(search_buffer) == 0:
                candidates = []
                status_msg = "All Cleared"

            # Sensitivity threshold tuning (+ / - or UP / DOWN)
            elif key_code in (ord("+"), ord("=")):
                b_thresh = min(0.90, round(b_thresh + 0.05, 2))
                status_msg = f"Gain: {b_thresh:.2f}"
            elif key_code in (ord("-"), ord("_")):
                b_thresh = max(0.10, round(b_thresh - 0.05, 2))
                status_msg = f"Gain: {b_thresh:.2f}"

            # SPACE: If buffer empty, rescan; if typing, add space
            elif key_code == 32:
                if not search_buffer:
                    if active_prompt:
                        status_msg = f"Rescanning '{active_prompt}'..."
                        t0 = time.perf_counter()
                        candidates = grounder.find(frame, active_prompt, b_thresh, t_thresh)
                        last_latency_ms = (time.perf_counter() - t0) * 1000
                        ambiguous = is_ambiguous(candidates)
                        status_msg = f"Locked: {len(candidates)}" if candidates else "0 Targets Found"
                else:
                    search_buffer += " "

            # 'S': Save snapshot
            elif key_code in (ord("s"), ord("S")) and not search_buffer:
                save_dir = Path("tests/data/sample_frames")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"stark_hud_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                status_msg = "Snapshot Saved"
                logger.info("HUD snapshot saved to %s", save_path)

            # Printable characters (a-z, A-Z, 0-9, punctuation)
            elif 32 < key_code <= 126:
                search_buffer += chr(key_code)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Demo concluded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STARK VISION — Live Grounding Engine Smoke Test Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model ID (default: IDEA-Research/grounding-dino-base)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Initial referring expression (default: 'person')",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Bounding box presence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Text-visual alignment threshold (default: 0.20)",
    )
    parser.add_argument(
        "--skip-boot",
        action="store_true",
        help="Skip the 2.5-second JARVIS startup diagnostic animation",
    )
    args = parser.parse_args()

    logger.info("Initializing STARK VISION Grounder on model: %s...", args.model)
    grounder = Grounder(
        model_id=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    run_on_webcam(
        grounder=grounder,
        camera_id=args.camera,
        initial_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        skip_boot=args.skip_boot,
    )


if __name__ == "__main__":
    main()
