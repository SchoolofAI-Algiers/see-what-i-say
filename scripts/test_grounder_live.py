#!/usr/bin/env python3
"""
STARK VISION — Live Interactive Grounding Engine Runner.

Features 100% on-screen interactive HUD text input directly within the OpenCV
window (no terminal freezing), real-time threshold tuning, dynamic box clearing,
and sci-fi reticle overlay.

Controls (All inside the OpenCV Window):
    Type any text       : Updates search query live on-screen
    BACKSPACE           : Delete character
    ENTER or SPACE      : Ground current frame with search query
    C                   : Clear prompt and clear all detections
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
import numpy as np

# Add src to path if executed standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stark_vision.grounding.grounder import Grounder, GroundingCandidate, is_ambiguous

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stark_vision.live_demo")


def draw_hud_box(
    frame: np.ndarray,
    candidate: GroundingCandidate,
    index: int,
    color: tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
) -> None:
    """
    Draw a sci-fi HUD reticle bounding box and telemetry label.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(candidate.x1)))
    y1 = max(0, min(h - 1, int(candidate.y1)))
    x2 = max(0, min(w - 1, int(candidate.x2)))
    y2 = max(0, min(h - 1, int(candidate.y2)))

    if x2 <= x1 or y2 <= y1:
        return

    # 1. Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 2. Corner reticle brackets
    corner_len = min(22, max(6, int((x2 - x1) * 0.15)))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 2)

    # 3. Target center reticle
    cx, cy = int(candidate.center[0]), int(candidate.center[1])
    if 0 <= cx < w and 0 <= cy < h:
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 12, 1)

    # 4. Telemetry badge
    label_text = f"#{index} {candidate.label} [{candidate.score * 100:.1f}%]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

    badge_y1 = max(0, y1 - th - baseline - 8)
    badge_y2 = y1
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), (18, 18, 18), -1)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, 1)
    cv2.putText(
        frame,
        label_text,
        (x1 + 5, y1 - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_hud_dashboard(
    frame: np.ndarray,
    search_buffer: str,
    active_prompt: str,
    latency_ms: float,
    num_candidates: int,
    ambiguous: bool,
    box_threshold: float,
    fps: float,
    status_msg: str,
) -> None:
    """
    Render top telemetry strip and bottom on-screen interactive input search bar.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top Telemetry Header
    header = frame.copy()
    cv2.rectangle(header, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(header, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "STARK VISION // GROUNDING ENGINE", (15, 22), font, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    thresh_color = (0, 255, 200) if box_threshold <= 0.25 else (200, 200, 200)
    cv2.putText(
        frame,
        f"LOCKED TARGET: '{active_prompt}'  |  THRESH: {box_threshold:.2f} (+/- to tune)",
        (15, 44),
        font,
        0.44,
        thresh_color,
        1,
        cv2.LINE_AA,
    )

    info_str = f"Targets: {num_candidates} | Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(info_str, font, 0.44, 1)
    status_color = (0, 255, 128) if num_candidates > 0 else (120, 120, 120)
    cv2.putText(frame, info_str, (w - tw - 15, 24), font, 0.44, status_color, 1, cv2.LINE_AA)

    if ambiguous:
        ambig_str = "! AMBIGUOUS TARGETS (Score delta < 0.15) !"
        (aw, _), _ = cv2.getTextSize(ambig_str, font, 0.44, 1)
        cv2.putText(frame, ambig_str, (w - aw - 15, 45), font, 0.44, (0, 165, 255), 1, cv2.LINE_AA)

    # Bottom Search & Input Bar (Interactive on-screen HUD)
    bar_h = 60
    footer = frame.copy()
    cv2.rectangle(footer, (0, h - bar_h), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(footer, 0.85, frame, 0.15, 0, frame)

    # Search bar outline
    cursor_char = "_" if int(time.time() * 2) % 2 == 0 else " "
    search_display = f"> SEARCH: {search_buffer}{cursor_char}"
    cv2.rectangle(frame, (10, h - bar_h + 8), (w - 10, h - 28), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, h - bar_h + 8), (w - 10, h - 28), (0, 220, 255), 1)

    cv2.putText(frame, search_display, (20, h - bar_h + 28), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Hotkey hints line
    hint_text = "Type to edit search  |  [ENTER/SPACE] Ground  |  [C] Clear Detections  |  [+/-] Sensitivity  |  [ESC] Exit"
    cv2.putText(frame, hint_text, (15, h - 10), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    if status_msg:
        (sw, _), _ = cv2.getTextSize(status_msg, font, 0.42, 1)
        cv2.putText(frame, status_msg, (w - sw - 20, h - bar_h + 28), font, 0.42, (0, 200, 255), 1, cv2.LINE_AA)


def run_on_webcam(
    grounder: Grounder,
    camera_id: int,
    initial_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> None:
    """
    Run interactive real-time grounding on webcam with direct on-screen keyboard input.
    """
    logger.info("Connecting to camera device %d...", camera_id)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Failed to open camera %d. Check /dev/video* permissions.", camera_id)
        return

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

    logger.info("Interactive HUD ready. Type directly inside the OpenCV window.")

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

            # Render detected objects
            for idx, cand in enumerate(candidates):
                color = (0, 255, 128) if idx == 0 else (255, 180, 0)
                draw_hud_box(display_frame, cand, idx + 1, color=color)

            # Render Dashboard and on-screen search bar
            draw_hud_dashboard(
                display_frame,
                search_buffer=search_buffer,
                active_prompt=active_prompt,
                latency_ms=last_latency_ms,
                num_candidates=len(candidates),
                ambiguous=ambiguous,
                box_threshold=b_thresh,
                fps=fps,
                status_msg=status_msg,
            )

            cv2.imshow(window_name, display_frame)

            # Handle Keystrokes inside OpenCV Window (NO terminal input!)
            key = cv2.waitKey(1)
            if key == -1:
                continue

            key_code = key & 0xFF

            # 1. Exit on ESC or 'q' if search buffer is empty
            if key_code == 27:  # ESC
                break
            elif key_code in (ord("q"), ord("Q")) and not search_buffer:
                break

            # 2. Enter: Trigger Grounding with search buffer
            elif key_code in (13, 10):  # ENTER
                active_prompt = search_buffer.strip()
                if active_prompt:
                    status_msg = f"Scanning '{active_prompt}'..."
                    logger.info("Grounding frame against prompt: '%s' (thresh: %.2f)...", active_prompt, b_thresh)
                    t0 = time.perf_counter()
                    candidates = grounder.find(frame, active_prompt, b_thresh, t_thresh)
                    last_latency_ms = (time.perf_counter() - t0) * 1000
                    ambiguous = is_ambiguous(candidates)
                    status_msg = f"Found {len(candidates)}" if candidates else "0 Targets Found"
                else:
                    candidates = []
                    status_msg = "Cleared"

            # 3. Backspace: delete character from search buffer
            elif key_code in (8, 127, 255):  # BACKSPACE
                if search_buffer:
                    search_buffer = search_buffer[:-1]

            # 4. 'C' or 'c': Clear detections and search buffer
            elif key_code in (ord("c"), ord("C")) and len(search_buffer) == 0:
                candidates = []
                status_msg = "All Detections Cleared"

            # 5. Sensitivity threshold tuning (+ / - or UP / DOWN)
            elif key_code in (ord("+"), ord("=")):
                b_thresh = min(0.90, round(b_thresh + 0.05, 2))
                status_msg = f"Thresh: {b_thresh:.2f}"
            elif key_code in (ord("-"), ord("_")):
                b_thresh = max(0.10, round(b_thresh - 0.05, 2))
                status_msg = f"Thresh: {b_thresh:.2f}"

            # 6. Space: If buffer is empty, re-ground; if typing, add space
            elif key_code == 32:  # SPACE
                if not search_buffer:
                    if active_prompt:
                        t0 = time.perf_counter()
                        candidates = grounder.find(frame, active_prompt, b_thresh, t_thresh)
                        last_latency_ms = (time.perf_counter() - t0) * 1000
                        ambiguous = is_ambiguous(candidates)
                        status_msg = f"Found {len(candidates)}" if candidates else "0 Targets Found"
                else:
                    search_buffer += " "

            # 7. Printable characters (a-z, A-Z, 0-9, punctuation)
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
    )



if __name__ == "__main__":
    main()
