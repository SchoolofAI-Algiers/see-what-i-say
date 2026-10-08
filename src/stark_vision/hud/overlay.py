"""
OpenCV HUD drawing routines for live camera feed.
Renders target bounding boxes, telemetry badges, and bottom prompt bar over frames.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import cv2
import numpy as np

from stark_vision.hud.geometry_3d import draw_3d_arc_reactor
from stark_vision.hud.typography import blit_text

# BGR color values for hud widgets
ARC_CYAN = (255, 240, 0)          # cyan accent
PLASMA_BLUE = (255, 144, 30)       # secondary blue
STARK_GOLD = (0, 191, 255)         # gold warning/secondary
CRIMSON_REPULSOR = (42, 42, 255)   # red alert
OBSIDIAN_GLASS = (20, 14, 10)      # dark panel bg
HUD_WHITE = (250, 248, 245)        # text color
HUD_GRAY = (160, 160, 160)         # muted text

try:
    from stark_vision.speech.jarvis_voice import play_startup_greeting
except ImportError:
    play_startup_greeting = None


class JarvisHUD:
    # renders cv2 graphic elements over raw webcam frames

    def __init__(
        self,
        target_resolution: tuple[int, int] = (640, 480),
        enable_voice: bool = True,
    ) -> None:
        self.default_w, self.default_h = target_resolution
        self.enable_voice = enable_voice
        self.voice_played = False
        self.frame_count = 0
        self.boot_start_time: float | None = None
        self.is_booted = False

    def reset_boot(self) -> None:
        # resets startup sequence state
        self.boot_start_time = None
        self.is_booted = False
        self.voice_played = False

    @staticmethod
    def _draw_chamfered_panel(
        frame: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        chamfer: int = 12,
        bg_color: tuple[int, int, int] = OBSIDIAN_GLASS,
        border_color: tuple[int, int, int] = ARC_CYAN,
        alpha: float = 0.75,
        border_thickness: int = 1,
    ) -> None:
        """
        Draw a high-tech polygon panel with 45-degree chamfered corners.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        ch = min(chamfer, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))

        pts = np.array(
            [
                [x1 + ch, y1],
                [x2 - ch, y1],
                [x2, y1 + ch],
                [x2, y2 - ch],
                [x2 - ch, y2],
                [x1 + ch, y2],
                [x1, y2 - ch],
                [x1, y1 + ch],
            ],
            dtype=np.int32,
        )

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], bg_color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        if border_thickness > 0:
            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=PLASMA_BLUE,
                thickness=border_thickness + 1,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=border_color,
                thickness=border_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_arc_reactor(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: float = 24.0,
    ) -> None:
        # spinning ring gyroscope animation
        draw_3d_arc_reactor(frame, center=center, frame_count=self.frame_count, base_radius=radius)

    def _draw_voice_waveform(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        is_active: bool = False,
    ) -> None:
        # vertical spectrum bars for audio status
        cx, cy = center
        num_bars = 13
        spacing = 5
        bar_w = 3
        max_h = 24 if is_active else 12
        t = self.frame_count * 0.16

        start_x = cx - ((num_bars * (bar_w + spacing)) // 2)

        for i in range(num_bars):
            norm_idx = i - (num_bars // 2)
            envelope = math.exp(-0.16 * (norm_idx**2))  # Bell curve
            pulse = math.sin(t + i * 0.65) * math.cos(t * 0.75 + i * 0.35)
            h = int(6 + abs(pulse) * max_h * envelope)

            bx = start_x + i * (bar_w + spacing)
            color = ARC_CYAN if (i % 2 == 0) else PLASMA_BLUE
            if is_active and abs(norm_idx) <= 1:
                color = STARK_GOLD

            cv2.line(frame, (bx, cy - h // 2), (bx, cy + h // 2), PLASMA_BLUE, bar_w + 2, cv2.LINE_AA)
            cv2.line(frame, (bx, cy - h // 2), (bx, cy + h // 2), color, bar_w, cv2.LINE_AA)

    def _draw_target_reticle(
        self,
        frame: np.ndarray,
        candidate,
        index: int,
        color: tuple[int, int, int] = ARC_CYAN,
    ) -> None:
        # draws corner brackets and label tag on tracked object
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(w_frame - 1, int(candidate.x1)))
        y1 = max(0, min(h_frame - 1, int(candidate.y1)))
        x2 = max(0, min(w_frame - 1, int(candidate.x2)))
        y2 = max(0, min(h_frame - 1, int(candidate.y2)))

        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return

        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / (w_frame * h_frame)
        is_large_target = area_ratio > 0.45

        # 1. Subtle Translucent Target Tint
        roi = frame[y1:y2, x1:x2]
        tint = np.full_like(roi, color, dtype=np.uint8)
        tint_alpha = 0.04 if is_large_target else 0.08
        cv2.addWeighted(tint, tint_alpha, roi, 1.0 - tint_alpha, 0, roi)

        # 2. Four Tactical Chamfered L-Brackets
        bracket_len = min(28, max(10, int(min(box_w, box_h) * 0.16)))

        # Top-Left Bracket
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y1 + 3), (x1 + bracket_len - 4, y1 + 3), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y1 + 3), (x1 + 3, y1 + bracket_len - 4), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Top-Right Bracket
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y1 + 3), (x2 - bracket_len + 4, y1 + 3), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y1 + 3), (x2 - 3, y1 + bracket_len - 4), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Bottom-Left Bracket
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y2 - 3), (x1 + bracket_len - 4, y2 - 3), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y2 - 3), (x1 + 3, y2 - bracket_len + 4), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Bottom-Right Bracket
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y2 - 3), (x2 - bracket_len + 4, y2 - 3), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y2 - 3), (x2 - 3, y2 - bracket_len + 4), PLASMA_BLUE, 1, cv2.LINE_AA)

        # 3. Center Target Lock Reticle (Only for focused objects, not full-room walls)
        cx, cy = int(candidate.center[0]), int(candidate.center[1])
        if not is_large_target and 0 <= cx < w_frame and 0 <= cy < h_frame:
            cv2.circle(frame, (cx, cy), 8, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, HUD_WHITE, -1, cv2.LINE_AA)
            cv2.line(frame, (cx - 12, cy), (cx - 4, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx + 4, cy), (cx + 12, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 12), (cx, cy - 4), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 4), (cx, cy + 12), color, 1, cv2.LINE_AA)

        # 4. Animated Scanning Laser Sweep Line
        scan_progress = ((self.frame_count * 2.5) % 100) / 100.0
        scan_y = int(y1 + scan_progress * box_h)
        cv2.line(frame, (x1 + 6, scan_y), (x2 - 6, scan_y), ARC_CYAN, 1, cv2.LINE_AA)

        # 5. Aerospace Telemetry Badge using Orbitron font
        dist_est = max(0.6, round(2.0 * (480 / max(1, box_h)), 1))
        badge_w = 210
        badge_h = 38
        badge_x1 = max(12, min(w_frame - badge_w - 12, x1))
        badge_y1 = max(55, y1 - badge_h - 4) if y1 > 60 else y1 + 10
        badge_y2 = badge_y1 + badge_h

        self._draw_chamfered_panel(
            frame,
            (badge_x1, badge_y1),
            (badge_x1 + badge_w, badge_y2),
            chamfer=6,
            bg_color=OBSIDIAN_GLASS,
            border_color=color,
            alpha=0.88,
            border_thickness=1,
        )

        # Blit TrueType Text directly
        blit_text(
            frame,
            f"TARGET #{index} // {candidate.label.upper()} [{candidate.score * 100:.1f}%]",
            (badge_x1 + 8, badge_y1 + 4),
            font_name="orbitron",
            size=11,
            bgr_color=HUD_WHITE,
        )
        blit_text(
            frame,
            f"LOCK: ACQUIRED  |  RANGE: ~{dist_est}m",
            (badge_x1 + 8, badge_y1 + 20),
            font_name="mono",
            size=10,
            bgr_color=STARK_GOLD,
        )

    def _draw_top_floating_widgets(
        self,
        frame: np.ndarray,
        active_prompt: str,
        latency_ms: float,
        num_candidates: int,
        ambiguous: bool,
        box_threshold: float,
        fps: float,
    ) -> None:
        # renders top widgets for framerate, latency and status
        h, w = frame.shape[:2]

        # --- LEFT CAPSULE: 3D Arc Reactor & Stark OS Branding ---
        left_w = 220
        self._draw_chamfered_panel(
            frame,
            (10, 8),
            (10 + left_w, 56),
            chamfer=10,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.82,
            border_thickness=1,
        )
        # 3D Rotating Arc Reactor
        self._draw_arc_reactor(frame, center=(34, 32), radius=18.0)

        # Orbitron Branding Text
        blit_text(
            frame,
            "STARK VISION",
            (62, 14),
            font_name="orbitron",
            size=13,
            bgr_color=ARC_CYAN,
        )
        target_display = f"LOCK: {active_prompt.upper()}" if active_prompt else "STATUS: STANDBY"
        target_color = STARK_GOLD if active_prompt else HUD_GRAY
        blit_text(
            frame,
            target_display,
            (62, 34),
            font_name="mono",
            size=11,
            bgr_color=target_color,
        )

        # --- CENTER CAPSULE: Acoustic Frequency Spectrum Visualizer ---
        is_searching = bool(active_prompt and num_candidates == 0)
        center_w = 110
        cx = w // 2
        self._draw_chamfered_panel(
            frame,
            (cx - center_w // 2, 8),
            (cx + center_w // 2, 48),
            chamfer=8,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.75,
            border_thickness=1,
        )
        self._draw_voice_waveform(frame, center=(cx, 28), is_active=is_searching)

        # --- RIGHT CAPSULE: High-Speed Flight Telemetry & Hardware ---
        right_w = 230
        rx1 = w - right_w - 10
        self._draw_chamfered_panel(
            frame,
            (rx1, 8),
            (w - 10, 56),
            chamfer=10,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.82,
            border_thickness=1,
        )
        status_color = ARC_CYAN if num_candidates > 0 else HUD_GRAY
        blit_text(
            frame,
            f"TARGETS: {num_candidates}  |  FPS: {fps:.1f}",
            (rx1 + 10, 14),
            font_name="mono",
            size=11,
            bgr_color=status_color,
        )
        blit_text(
            frame,
            f"LATENCY: {latency_ms:.1f}ms  |  GAIN: {box_threshold:.2f}",
            (rx1 + 10, 34),
            font_name="mono",
            size=10,
            bgr_color=STARK_GOLD,
        )

        # Ambiguity Banner
        if ambiguous:
            ambig_text = "ALERT: AMBIGUOUS TARGETS DETECTED (DELTA < 0.15)"
            aw = 360
            ax1 = (w - aw) // 2
            self._draw_chamfered_panel(
                frame,
                (ax1, 62),
                (ax1 + aw, 88),
                chamfer=6,
                bg_color=OBSIDIAN_GLASS,
                border_color=CRIMSON_REPULSOR,
                alpha=0.92,
                border_thickness=1,
            )
            blit_text(
                frame,
                ambig_text,
                (ax1 + 14, 68),
                font_name="orbitron",
                size=11,
                bgr_color=CRIMSON_REPULSOR,
            )

    def _draw_bottom_console(
        self,
        frame: np.ndarray,
        search_buffer: str,
        status_msg: str,
    ) -> None:
        # bottom prompt text bar with keyboard input display
        h, w = frame.shape[:2]
        bar_h = 62
        y_top = h - bar_h - 10
        y_bottom = h - 10

        # Main Console Dock Plate
        self._draw_chamfered_panel(
            frame,
            (10, y_top),
            (w - 10, y_bottom),
            chamfer=12,
            bg_color=OBSIDIAN_GLASS,
            border_color=ARC_CYAN,
            alpha=0.86,
            border_thickness=1,
        )

        # Inner Input Field
        cursor_char = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        display_query = f"> PROTOCOL INPUT: [ {search_buffer}{cursor_char} ]"
        inner_y1 = y_top + 8
        inner_y2 = y_top + 34
        cv2.rectangle(frame, (18, inner_y1), (w - 18, inner_y2), (28, 20, 15), -1)
        cv2.rectangle(frame, (18, inner_y1), (w - 18, inner_y2), PLASMA_BLUE, 1, cv2.LINE_AA)

        blit_text(
            frame,
            display_query,
            (26, inner_y1 + 5),
            font_name="orbitron",
            size=12,
            bgr_color=HUD_WHITE,
        )

        # Status Tag Chip (Right edge)
        if status_msg:
            chip_w = 120
            chip_x1 = w - chip_w - 24
            cv2.rectangle(frame, (chip_x1, inner_y1 + 2), (w - 22, inner_y2 - 2), (45, 35, 20), -1)
            blit_text(
                frame,
                status_msg.upper(),
                (chip_x1 + 8, inner_y1 + 5),
                font_name="mono",
                size=11,
                bgr_color=STARK_GOLD,
            )

        # Hotkey Action Chips Line
        hints = "[ENTER] ACQUIRE  |  [C] CLEAR  |  [+/-] GAIN  |  [SPACE] RESCAN  |  [ESC] POWER DOWN"
        blit_text(
            frame,
            hints,
            (22, y_bottom - 18),
            font_name="mono",
            size=10,
            bgr_color=HUD_GRAY,
        )

    def _draw_screen_visor_flourishes(self, frame: np.ndarray) -> None:
        # border edge brackets for visor look
        h, w = frame.shape[:2]
        c_len = 22

        # 4 Corner Visor Brackets
        for cx_base, cy_base, dx, dy in (
            (10, 10, 1, 1),
            (w - 10, 10, -1, 1),
            (10, h - 10, 1, -1),
            (w - 10, h - 10, -1, -1),
        ):
            cv2.line(frame, (cx_base, cy_base), (cx_base + dx * c_len, cy_base), ARC_CYAN, 2, cv2.LINE_AA)
            cv2.line(frame, (cx_base, cy_base), (cx_base, cy_base + dy * c_len), ARC_CYAN, 2, cv2.LINE_AA)
            cv2.line(
                frame,
                (cx_base + dx * 3, cy_base + dy * 3),
                (cx_base + dx * (c_len - 6), cy_base + dy * 3),
                PLASMA_BLUE,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (cx_base + dx * 3, cy_base + dy * 3),
                (cx_base + dx * 3, cy_base + dy * (c_len - 6)),
                PLASMA_BLUE,
                1,
                cv2.LINE_AA,
            )

        # Marginal Telemetry Rail: Pitch / Elevation Ladder
        cy = h // 2
        for val in (-20, -10, 0, 10, 20):
            y = cy - int(val * 2.2)
            if 65 < y < h - 75:
                is_zero = val == 0
                tick_w = 8 if is_zero else 5
                color = ARC_CYAN if is_zero else PLASMA_BLUE
                cv2.line(frame, (12, y), (12 + tick_w, y), color, 1, cv2.LINE_AA)
                if abs(val) in (0, 20):
                    blit_text(frame, f"{val:+d}", (22, y - 6), font_name="mono", size=9, bgr_color=HUD_GRAY)

        # Right Marginal Rail: RTX 4060 GPU Badge
        rx = w - 108
        ry = cy - 24
        self._draw_chamfered_panel(
            frame,
            (rx, ry),
            (w - 12, ry + 46),
            chamfer=6,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.68,
            border_thickness=1,
        )
        blit_text(frame, "RTX 4060", (rx + 8, ry + 4), font_name="orbitron", size=10, bgr_color=ARC_CYAN)
        blit_text(frame, "CUDA 13 // 8GB", (rx + 8, ry + 18), font_name="mono", size=9, bgr_color=HUD_GRAY)
        blit_text(frame, "SYNC: ACTIVE", (rx + 8, ry + 30), font_name="mono", size=9, bgr_color=STARK_GOLD)

    def render_boot_sequence(self, frame: np.ndarray) -> bool:
        # returns True when the 2.5s startup intro finishes playing
        if self.is_booted:
            return True

        if self.enable_voice and not self.voice_played:
            if play_startup_greeting is not None:
                play_startup_greeting()
            self.voice_played = True

        if self.boot_start_time is None:
            self.boot_start_time = time.time()

        elapsed = time.time() - self.boot_start_time
        duration = 2.5
        progress = min(1.0, elapsed / duration)
        h, w = frame.shape[:2]

        # Darken frame during boot
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (12, 8, 6), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Center 3D Arc Reactor Spin-Up
        cx, cy = w // 2, h // 2 - 32
        r_spin = float(28 + 14 * math.sin(progress * math.pi))
        draw_3d_arc_reactor(frame, center=(cx, cy), frame_count=self.frame_count * 2, base_radius=r_spin)

        lines = [
            ("STARK INDUSTRIES // MARK LXXXV INITIALIZATION", "orbitron", 14, ARC_CYAN),
            ("> NEURAL REASONING ENGINE: ONLINE", "mono", 11, HUD_WHITE),
            ("> VISION BACKBONE: SWIN TRANSFORMER ACTIVE", "mono", 11, HUD_WHITE),
            ("> SENSOR INTERFACE: /dev/video0 CALIBRATED", "mono", 11, HUD_WHITE),
            ("> PROTOCOL ARMED: AWAITING OPERATOR DIRECTIVE", "mono", 11, STARK_GOLD),
        ]

        num_visible = min(len(lines), int(progress * (len(lines) + 1)))
        y_start = cy + 62

        for idx in range(num_visible):
            text, fname, fsize, color = lines[idx]
            blit_text(
                frame,
                text,
                (cx - 180, y_start + idx * 22),
                font_name=fname,
                size=fsize,
                bgr_color=color,
            )

        # Progress Bar & Percentage
        bar_w = 280
        bar_h = 6
        bx1 = (w - bar_w) // 2
        by1 = y_start + len(lines) * 22 + 12
        cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w, by1 + bar_h), (45, 35, 25), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bx1, by1), (bx1 + fill_w, by1 + bar_h), ARC_CYAN, -1)

        pct_str = f"SYSTEM STATUS: {int(progress * 100)}%"
        blit_text(frame, pct_str, (cx - 65, by1 + 12), font_name="orbitron", size=10, bgr_color=STARK_GOLD)

        self.frame_count += 1

        if progress >= 1.0:
            self.is_booted = True
            return True

        return False

    def render(
        self,
        frame: np.ndarray,
        candidates: Sequence,
        search_buffer: str,
        active_prompt: str,
        box_threshold: float,
        latency_ms: float = 0.0,
        fps: float = 30.0,
        ambiguous: bool = False,
        status_msg: str = "Ready",
    ) -> np.ndarray:
        # main drawing pipeline per frame: reticles, widgets, and console
        self.frame_count += 1

        # 1. Visor Screen Flourishes & Marginal Rails
        self._draw_screen_visor_flourishes(frame)

        # 2. Render Target Reticles
        for idx, cand in enumerate(candidates):
            color = ARC_CYAN if idx == 0 else STARK_GOLD
            self._draw_target_reticle(frame, cand, index=idx + 1, color=color)

        # 3. Render Floating Top Telemetry Capsules
        self._draw_top_floating_widgets(
            frame,
            active_prompt=active_prompt,
            latency_ms=latency_ms,
            num_candidates=len(candidates),
            ambiguous=ambiguous,
            box_threshold=box_threshold,
            fps=fps,
        )

        # 4. Render Bottom Cyber Command Console Dock
        self._draw_bottom_console(
            frame,
            search_buffer=search_buffer,
            status_msg=status_msg,
        )

        return frame
