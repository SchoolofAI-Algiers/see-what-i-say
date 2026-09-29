#!/usr/bin/env python3
"""
STARK VISION — Live Grounding Engine Smoke Test Runner.

Demonstrates open-vocabulary referring expression grounding on live webcam
or static image files using Grounding DINO.

Usage:
    # 1. Run live on laptop webcam:
    uv run python scripts/test_grounder_live.py

    # 2. Run with custom initial prompt:
    uv run python scripts/test_grounder_live.py --prompt "cell phone"

    # 3. Run on a static test image:
    uv run python scripts/test_grounder_live.py --image path/to/image.jpg

Controls (Webcam Mode):
    SPACE : Trigger Grounding DINO on current frame
    P     : Change text prompt in terminal
    S     : Save current annotated snapshot
    Q/ESC : Exit
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
    Draw a sci-fi HUD-style bounding box and telemetry label on the frame.
    """
    x1, y1, x2, y2 = int(candidate.x1), int(candidate.y1), int(candidate.x2), int(candidate.y2)

    # 1. Main bounding rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 2. Corner reticle brackets (Iron Man style)
    corner_len = min(20, max(5, int((x2 - x1) * 0.15)))
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 2)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 2)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 2)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 2)

    # 3. Telemetry badge
    label_text = f"#{index} {candidate.label} [{candidate.score * 100:.1f}%]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

    badge_y1 = max(0, y1 - th - baseline - 8)
    badge_y2 = y1
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), (20, 20, 20), -1)
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


def draw_hud_header(
    frame: np.ndarray,
    prompt: str,
    latency_ms: float,
    num_candidates: int,
    ambiguous: bool,
    fps: float = 0.0,
) -> None:
    """
    Draw the top HUD telemetry dashboard banner.
    """
    h, w = frame.shape[:2]
    # Header background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Title & Prompt
    cv2.putText(
        frame,
        f"STARK VISION // GROUNDING ENGINE",
        (15, 22),
        font,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"QUERY: '{prompt}'",
        (15, 44),
        font,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Telemetry metrics
    status_color = (0, 255, 128) if num_candidates > 0 else (120, 120, 120)
    info_str = f"Found: {num_candidates} | Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(info_str, font, 0.45, 1)
    cv2.putText(
        frame,
        info_str,
        (w - tw - 15, 25),
        font,
        0.45,
        status_color,
        1,
        cv2.LINE_AA,
    )

    if ambiguous:
        ambig_str = "! AMBIGUOUS TARGETS (Score delta < 0.15) !"
        (aw, _), _ = cv2.getTextSize(ambig_str, font, 0.45, 1)
        cv2.putText(
            frame,
            ambig_str,
            (w - aw - 15, 45),
            font,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )


def run_on_image(
    grounder: Grounder,
    image_path: str,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    output_path: str | None = None,
) -> None:
    """
    Run Grounding on a single static image file and display/save result.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Could not load image from %s", image_path)
        return

    logger.info("Grounding prompt '%s' on image %s...", prompt, image_path)
    t0 = time.perf_counter()
    candidates = grounder.find(
        frame_bgr=frame,
        text_prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    ambiguous = is_ambiguous(candidates)
    logger.info(
        "Detected %d candidates in %.1f ms (Ambiguous: %s)",
        len(candidates),
        latency_ms,
        ambiguous,
    )

    annotated = frame.copy()
    for idx, cand in enumerate(candidates):
        color = (0, 255, 128) if idx == 0 else (255, 180, 0)
        draw_hud_box(annotated, cand, idx + 1, color=color)

    draw_hud_header(annotated, prompt, latency_ms, len(candidates), ambiguous)

    save_path = output_path or "grounding_result.jpg"
    cv2.imwrite(save_path, annotated)
    logger.info("Saved annotated result to %s", save_path)

    # Attempt GUI display if display server available
    try:
        cv2.imshow("STARK VISION — Static Grounding Test", annotated)
        logger.info("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        logger.info("Headless environment: display skipped, result saved to %s", save_path)


def run_on_webcam(
    grounder: Grounder,
    camera_id: int,
    initial_prompt: str,
    box_threshold: float,
    text_threshold: float,
    continuous: bool = False,
) -> None:
    """
    Run live interactive grounding on the webcam stream.
    """
    logger.info("Opening camera device %d...", camera_id)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Failed to open camera device %d. Check permissions or /dev/video*.", camera_id)
        return

    prompt = initial_prompt
    logger.info("Webcam online. Controls: [SPACE] Ground | [P] Change Prompt | [S] Save | [Q] Quit")

    candidates: list[GroundingCandidate] = []
    last_latency_ms = 0.0
    ambiguous = False
    prev_time = time.perf_counter()
    fps = 0.0

    # Run initial grounding on startup
    ret, frame = cap.read()
    if ret:
        t0 = time.perf_counter()
        candidates = grounder.find(frame, prompt, box_threshold, text_threshold)
        last_latency_ms = (time.perf_counter() - t0) * 1000
        ambiguous = is_ambiguous(candidates)

    window_name = "STARK VISION — Live Grounding Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed. Retrying...")
                time.sleep(0.03)
                continue

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - prev_time))
            prev_time = now

            if continuous:
                t0 = time.perf_counter()
                candidates = grounder.find(frame, prompt, box_threshold, text_threshold)
                last_latency_ms = (time.perf_counter() - t0) * 1000
                ambiguous = is_ambiguous(candidates)

            # Render HUD
            display_frame = frame.copy()
            for idx, cand in enumerate(candidates):
                color = (0, 255, 128) if idx == 0 else (255, 180, 0)
                draw_hud_box(display_frame, cand, idx + 1, color=color)

            draw_hud_header(display_frame, prompt, last_latency_ms, len(candidates), ambiguous, fps=fps)

            # Instructions footer
            footer_text = "[SPACE] Ground Now  |  [P] Change Prompt  |  [S] Save Frame  |  [Q] Exit"
            cv2.putText(
                display_frame,
                footer_text,
                (15, display_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key == ord(" "):  # SPACE: trigger manual grounding
                logger.info("Grounding frame against prompt: '%s'...", prompt)
                t0 = time.perf_counter()
                candidates = grounder.find(frame, prompt, box_threshold, text_threshold)
                last_latency_ms = (time.perf_counter() - t0) * 1000
                ambiguous = is_ambiguous(candidates)
                logger.info(
                    "Result: %d candidates in %.1f ms (Ambiguous: %s)",
                    len(candidates),
                    last_latency_ms,
                    ambiguous,
                )
            elif key in (ord("p"), ord("P")):  # P: prompt change
                logger.info("Current prompt: '%s'", prompt)
                new_prompt = input("\nEnter new referring expression: ").strip()
                if new_prompt:
                    prompt = new_prompt
                    logger.info("Prompt updated to: '%s'. Re-grounding...", prompt)
                    t0 = time.perf_counter()
                    candidates = grounder.find(frame, prompt, box_threshold, text_threshold)
                    last_latency_ms = (time.perf_counter() - t0) * 1000
                    ambiguous = is_ambiguous(candidates)
            elif key in (ord("s"), ord("S")):  # S: save snapshot
                save_dir = Path("tests/data/sample_frames")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"grounding_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                logger.info("Snapshot saved to %s", save_path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Demo concluded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STARK VISION — Live Grounding Engine Smoke Test Runner"
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
        "--image",
        type=str,
        default=None,
        help="Optional path to static image file to test instead of webcam",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        help="Bounding box presence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text-visual alignment threshold (default: 0.25)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run grounding continuously every frame (simulates single-loop bottleneck)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for saving static image result",
    )
    args = parser.parse_args()

    logger.info("Initializing STARK VISION Grounder on RTX 4060 GPU...")
    grounder = Grounder(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.image:
        run_on_image(
            grounder=grounder,
            image_path=args.image,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            output_path=args.output,
        )
    else:
        run_on_webcam(
            grounder=grounder,
            camera_id=args.camera,
            initial_prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            continuous=args.continuous,
        )


if __name__ == "__main__":
    main()
