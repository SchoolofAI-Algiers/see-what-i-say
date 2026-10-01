"""
STARK VISION — JarvisHUD Engine.

Provides high-tech Marvel Iron Man / Mark LXXXV tactical augmented reality
overlays using pure OpenCV (cv2) and NumPy.

Includes:
- Rotating Arc Reactor gyroscopic telemetry gauge
- Dynamic JARVIS acoustic voice waveform visualizer
- Tactical target reticles with corner brackets and laser sweep
- Holographic cyber-chamfered command console dock
- Diagnostic boot and calibration sequence
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import cv2
import numpy as np

# BGR Design Tokens (Iron Man / Arc Reactor Palette)
ARC_CYAN = (255, 240, 0)         # Primary electric hologram cyan
PLASMA_BLUE = (255, 144, 30)      # Deep secondary hologram blue
STARK_GOLD = (0, 191, 255)        # Mark 85 armor gold
CRIMSON_REPULSOR = (42, 42, 255)  # Alert crimson red
OBSIDIAN_GLASS = (25, 15, 10)     # Dark translucent backing
HUD_WHITE = (245, 245, 245)       # High-contrast telemetry text
HUD_GRAY = (160, 160, 160)        # Subtle secondary text

try:
    from stark_vision.speech.jarvis_voice import play_startup_greeting
except ImportError:
    play_startup_greeting = None


class JarvisHUD:
    """
    High-performance, sci-fi augmented reality HUD overlay engine.
    """

    def __init__(
        self,
        target_resolution: tuple[int, int] = (640, 480),
        enable_voice: bool = True,
    ) -> None:
        """
        Initialize the HUD engine.

        Args:
            target_resolution: Default (width, height) for telemetry scaling.
            enable_voice: Whether to play the JARVIS voice greeting on boot.
        """
        self.default_w, self.default_h = target_resolution
        self.enable_voice = enable_voice
        self.voice_played = False
        self.frame_count = 0
        self.boot_start_time: float | None = None
        self.is_booted = False

    def reset_boot(self) -> None:
        """Reset the boot sequence timer."""
        self.boot_start_time = None
        self.is_booted = False
        self.voice_played = False

    @staticmethod
    def _draw_chamfered_panel(
        frame: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        chamfer: int = 15,
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
        ch = min(chamfer, (x2 - x1) // 3, (y2 - y1) // 3)

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
                color=border_color,
                thickness=border_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_arc_reactor(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int = 24,
    ) -> None:
        """
        Draw animated rotating Arc Reactor concentric rings with radial tick marks.
        """
        cx, cy = center
        t = self.frame_count

        # 1. Outer segmented ring (clockwise rotation)
        angle_cw = (t * 3) % 360
        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            angle_cw,
            0,
            110,
            STARK_GOLD,
            2,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            angle_cw + 180,
            0,
            110,
            STARK_GOLD,
            2,
            cv2.LINE_AA,
        )

        # 2. Middle segmented ring (counter-clockwise rotation)
        angle_ccw = (-t * 4) % 360
        r_mid = max(8, radius - 6)
        cv2.ellipse(
            frame,
            (cx, cy),
            (r_mid, r_mid),
            angle_ccw,
            20,
            160,
            ARC_CYAN,
            2,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            frame,
            (cx, cy),
            (r_mid, r_mid),
            angle_ccw + 180,
            20,
            160,
            ARC_CYAN,
            2,
            cv2.LINE_AA,
        )

        # 3. Inner core and center pip
        r_core = max(4, radius - 14)
        cv2.circle(frame, (cx, cy), r_core, PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 4. Eight radial tick marks (45 degrees apart)
        for i in range(8):
            deg = math.radians(i * 45 + angle_cw)
            x_in = int(cx + (radius + 2) * math.cos(deg))
            y_in = int(cy + (radius + 2) * math.sin(deg))
            x_out = int(cx + (radius + 6) * math.cos(deg))
            y_out = int(cy + (radius + 6) * math.sin(deg))
            cv2.line(frame, (x_in, y_in), (x_out, y_out), ARC_CYAN, 1, cv2.LINE_AA)

    def _draw_voice_waveform(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        is_active: bool = False,
    ) -> None:
        """
        Draw animated acoustic waveform equalizer bars (JARVIS voice pulse).
        """
        cx, cy = center
        num_bars = 11
        spacing = 5
        bar_w = 3
        max_h = 24 if is_active else 12
        t = self.frame_count * 0.15

        start_x = cx - ((num_bars * (bar_w + spacing)) // 2)

        for i in range(num_bars):
            # Harmonic sine wave combination
            norm_idx = i - (num_bars // 2)
            envelope = math.exp(-0.15 * (norm_idx**2))  # Bell curve envelope
            pulse = math.sin(t + i * 0.7) * math.cos(t * 0.8 + i * 0.3)
            h = int(6 + abs(pulse) * max_h * envelope)

            bx = start_x + i * (bar_w + spacing)
            color = ARC_CYAN if i % 2 == 0 else PLASMA_BLUE
            if is_active:
                color = STARK_GOLD if i == num_bars // 2 else color

            cv2.line(
                frame,
                (bx, cy - h // 2),
                (bx, cy + h // 2),
                color,
                bar_w,
                cv2.LINE_AA,
            )

    def _draw_target_reticle(
        self,
        frame: np.ndarray,
        candidate,
        index: int,
        color: tuple[int, int, int] = ARC_CYAN,
    ) -> None:
        """
        Draw tactical Iron Man target lock reticle, chamfered brackets,
        sweeping laser scanline, and telemetry badge.
        """
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(w_frame - 1, int(candidate.x1)))
        y1 = max(0, min(h_frame - 1, int(candidate.y1)))
        x2 = max(0, min(w_frame - 1, int(candidate.x2)))
        y2 = max(0, min(h_frame - 1, int(candidate.y2)))

        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return

        box_w = x2 - x1
        box_h = y2 - y1

        # 1. Subtle translucent target tint
        roi = frame[y1:y2, x1:x2]
        tint = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(tint, 0.08, roi, 0.92, 0, roi)

        # 2. Four Tactical Chamfered L-Brackets
        bracket_len = min(28, max(8, int(min(box_w, box_h) * 0.20)))
        thick = 2

        # Top-Left Bracket
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y1 + 3), (x1 + bracket_len - 4, y1 + 3), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Top-Right Bracket
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y1 + 3), (x2 - bracket_len + 4, y1 + 3), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Bottom-Left Bracket
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1 + 3, y2 - 3), (x1 + bracket_len - 4, y2 - 3), PLASMA_BLUE, 1, cv2.LINE_AA)

        # Bottom-Right Bracket
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - 3, y2 - 3), (x2 - bracket_len + 4, y2 - 3), PLASMA_BLUE, 1, cv2.LINE_AA)

        # 3. Center Target Lock Reticle
        cx, cy = int(candidate.center[0]), int(candidate.center[1])
        if 0 <= cx < w_frame and 0 <= cy < h_frame:
            cv2.circle(frame, (cx, cy), 6, color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 1, cv2.LINE_AA)

        # 4. Animated Scanning Laser Sweep Line
        scan_progress = ((self.frame_count * 3) % 100) / 100.0
        scan_y = int(y1 + scan_progress * box_h)
        cv2.line(frame, (x1 + 4, scan_y), (x2 - 4, scan_y), ARC_CYAN, 1, cv2.LINE_AA)

        # 5. Tactical Telemetry Badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"TARGET #{index} // {candidate.label.upper()} [{candidate.score * 100:.1f}%]"
        # Estimated distance based on vertical fraction
        dist_est = max(0.5, round(2.2 * (480 / max(1, box_h)), 1))
        sub_text = f"LOCK: ACQUIRED | RANGE: ~{dist_est}m"

        (tw1, th1), _ = cv2.getTextSize(label_text, font, 0.44, 1)
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 0.36, 1)
        badge_w = max(tw1, tw2) + 16
        badge_h = th1 + th2 + 16

        badge_y1 = max(55, y1 - badge_h - 6)
        badge_y2 = badge_y1 + badge_h

        self._draw_chamfered_panel(
            frame,
            (x1, badge_y1),
            (x1 + badge_w, badge_y2),
            chamfer=6,
            bg_color=OBSIDIAN_GLASS,
            border_color=color,
            alpha=0.82,
            border_thickness=1,
        )

        cv2.putText(
            frame,
            label_text,
            (x1 + 8, badge_y1 + th1 + 4),
            font,
            0.44,
            HUD_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            sub_text,
            (x1 + 8, badge_y2 - 5),
            font,
            0.36,
            STARK_GOLD,
            1,
            cv2.LINE_AA,
        )

    def _draw_top_telemetry(
        self,
        frame: np.ndarray,
        active_prompt: str,
        latency_ms: float,
        num_candidates: int,
        ambiguous: bool,
        box_threshold: float,
        fps: float,
    ) -> None:
        """
        Render the top holographic flight telemetry bar with Arc Reactor & Waveform.
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top Bar Panel
        self._draw_chamfered_panel(
            frame,
            (8, 6),
            (w - 8, 58),
            chamfer=12,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.80,
            border_thickness=1,
        )

        # 1. Arc Reactor Gyroscope (Top-Left)
        self._draw_arc_reactor(frame, center=(38, 32), radius=20)

        # 2. Title & Target Identifier
        cv2.putText(
            frame,
            "STARK VISION // MARK LXXXV OS",
            (68, 26),
            font,
            0.50,
            ARC_CYAN,
            1,
            cv2.LINE_AA,
        )
        target_display = f"TARGET: '{active_prompt.upper()}'" if active_prompt else "TARGET: [STANDBY]"
        target_color = STARK_GOLD if active_prompt else HUD_GRAY
        cv2.putText(
            frame,
            target_display,
            (68, 48),
            font,
            0.42,
            target_color,
            1,
            cv2.LINE_AA,
        )

        # 3. Acoustic Voice Waveform (Center)
        is_searching = bool(active_prompt and num_candidates == 0)
        self._draw_voice_waveform(frame, center=(w // 2, 32), is_active=is_searching)

        # 4. Telemetry Metrics (Right side)
        status_color = ARC_CYAN if num_candidates > 0 else HUD_GRAY
        telemetry_str = f"TARGETS: {num_candidates}  |  LATENCY: {latency_ms:.1f}ms  |  FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(telemetry_str, font, 0.42, 1)
        cv2.putText(
            frame,
            telemetry_str,
            (w - tw - 24, 26),
            font,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )

        thresh_str = f"GAIN/THRESH: {box_threshold:.2f} [+/-]"
        (thw, _), _ = cv2.getTextSize(thresh_str, font, 0.38, 1)
        cv2.putText(
            frame,
            thresh_str,
            (w - thw - 24, 48),
            font,
            0.38,
            PLASMA_BLUE,
            1,
            cv2.LINE_AA,
        )

        # 5. Ambiguity Banner if triggered
        if ambiguous:
            ambig_text = "! AMBIGUOUS TARGETS DETECTED (SCORE DELTA < 0.15) !"
            (aw, _), _ = cv2.getTextSize(ambig_text, font, 0.44, 1)
            ax = (w - aw) // 2
            self._draw_chamfered_panel(
                frame,
                (ax - 10, 64),
                (ax + aw + 10, 92),
                chamfer=6,
                bg_color=OBSIDIAN_GLASS,
                border_color=CRIMSON_REPULSOR,
                alpha=0.90,
                border_thickness=1,
            )
            cv2.putText(
                frame,
                ambig_text,
                (ax, 83),
                font,
                0.44,
                CRIMSON_REPULSOR,
                1,
                cv2.LINE_AA,
            )

    def _draw_bottom_console(
        self,
        frame: np.ndarray,
        search_buffer: str,
        status_msg: str,
    ) -> None:
        """
        Render bottom holographic search console dock with cyber chamfers.
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        bar_h = 66
        y_top = h - bar_h - 8
        y_bottom = h - 8

        # Main Console Dock
        self._draw_chamfered_panel(
            frame,
            (8, y_top),
            (w - 8, y_bottom),
            chamfer=14,
            bg_color=OBSIDIAN_GLASS,
            border_color=ARC_CYAN,
            alpha=0.85,
            border_thickness=1,
        )

        # Search Bar Inner Plate
        cursor_char = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        search_str = f"> QUERY PROTOCOL: {search_buffer}{cursor_char}"
        cv2.rectangle(frame, (18, y_top + 10), (w - 20, y_top + 38), (35, 25, 20), -1)
        cv2.rectangle(frame, (18, y_top + 10), (w - 20, y_top + 38), PLASMA_BLUE, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            search_str,
            (28, y_top + 29),
            font,
            0.50,
            HUD_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Status Tag / State Chip (Right edge of search bar)
        if status_msg:
            (sw, _), _ = cv2.getTextSize(status_msg, font, 0.40, 1)
            chip_x1 = w - sw - 36
            cv2.rectangle(frame, (chip_x1, y_top + 12), (w - 22, y_top + 36), (50, 40, 25), -1)
            cv2.putText(
                frame,
                status_msg.upper(),
                (chip_x1 + 6, y_top + 29),
                font,
                0.40,
                STARK_GOLD,
                1,
                cv2.LINE_AA,
            )

        # Command & Hotkey Chips
        hints = "[ENTER] ACQUIRE  |  [C] DISMISS ALL  |  [+/-] SENSITIVITY  |  [SPACE] RESCAN  |  [ESC] POWER DOWN"
        cv2.putText(
            frame,
            hints,
            (22, y_bottom - 8),
            font,
            0.36,
            HUD_GRAY,
            1,
            cv2.LINE_AA,
        )

    def _draw_screen_visor_flourishes(self, frame: np.ndarray) -> None:
        """
        Draw high-tech corner flourishes and subtle perimeter visor frame.
        """
        h, w = frame.shape[:2]
        c_len = 24

        # 1. Top-Left Corner Visor Bracket
        cv2.line(frame, (10, 10), (10 + c_len, 10), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (10, 10), (10, 10 + c_len), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (14, 14), (14 + c_len - 8, 14), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (14, 14), (14, 14 + c_len - 8), PLASMA_BLUE, 1, cv2.LINE_AA)

        # 2. Top-Right Corner Visor Bracket
        cv2.line(frame, (w - 10, 10), (w - 10 - c_len, 10), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 10, 10), (w - 10, 10 + c_len), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, 14), (w - 14 - c_len + 8, 14), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (w - 14, 14), (w - 14, 14 + c_len - 8), PLASMA_BLUE, 1, cv2.LINE_AA)

        # 3. Bottom-Left Corner Visor Bracket
        cv2.line(frame, (10, h - 10), (10 + c_len, h - 10), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (10, h - 10), (10, h - 10 - c_len), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (14, h - 14), (14 + c_len - 8, h - 14), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (14, h - 14), (14, h - 14 - c_len + 8), PLASMA_BLUE, 1, cv2.LINE_AA)

        # 4. Bottom-Right Corner Visor Bracket
        cv2.line(frame, (w - 10, h - 10), (w - 10 - c_len, h - 10), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - c_len), ARC_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (w - 14, h - 14), (w - 14 - c_len + 8, h - 14), PLASMA_BLUE, 1, cv2.LINE_AA)
        cv2.line(frame, (w - 14, h - 14), (w - 14, h - 14 - c_len + 8), PLASMA_BLUE, 1, cv2.LINE_AA)

    def _draw_marginal_telemetry_rails(self, frame: np.ndarray) -> None:
        """
        Draw subtle lateral aerospace telemetry rails (altitude/pitch ladder and compute stats).
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cy = h // 2

        # Left Rail: Pitch / Sensor Elevation Ladder
        ladder_ticks = [-30, -20, -10, 0, 10, 20, 30]
        for val in ladder_ticks:
            y = cy - int(val * 2.2)
            if 70 < y < h - 80:
                is_zero = val == 0
                tick_w = 10 if is_zero else 6
                color = ARC_CYAN if is_zero else HUD_GRAY
                cv2.line(frame, (14, y), (14 + tick_w, y), color, 1, cv2.LINE_AA)
                if abs(val) in (0, 20):
                    txt = f"{val:+d}" if val != 0 else " 0"
                    cv2.putText(frame, txt, (28, y + 4), font, 0.30, HUD_GRAY, 1, cv2.LINE_AA)

        # Right Rail: System Compute & Accelerator telemetry plate
        rx = w - 110
        ry = cy - 25
        self._draw_chamfered_panel(
            frame,
            (rx, ry),
            (w - 12, ry + 50),
            chamfer=6,
            bg_color=OBSIDIAN_GLASS,
            border_color=PLASMA_BLUE,
            alpha=0.65,
            border_thickness=1,
        )
        cv2.putText(frame, "RTX 4060", (rx + 8, ry + 16), font, 0.34, ARC_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "CUDA 13 // 8GB", (rx + 8, ry + 30), font, 0.28, HUD_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, "SYNC: NOMINAL", (rx + 8, ry + 44), font, 0.28, STARK_GOLD, 1, cv2.LINE_AA)

    def render_boot_sequence(self, frame: np.ndarray) -> bool:
        """
        Play animated 2.5-second JARVIS diagnostic startup sequence.

        Returns:
            True if boot is complete, False while boot animation is actively running.
        """
        if self.is_booted:
            return True

        # Trigger non-blocking JARVIS voice greeting on initial boot frame
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
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Darken frame during boot
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 8, 5), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Center Arc Reactor Spin-Up
        cx, cy = w // 2, h // 2 - 30
        r_spin = int(35 + 15 * math.sin(progress * math.pi))
        self._draw_arc_reactor(frame, center=(cx, cy), radius=r_spin)

        # System Diagnostic Text lines typing out
        lines = [
            "STARK INDUSTRIES // MARK LXXXV INITIALIZATION",
            "> NEURAL REASONING ENGINE: ONLINE",
            "> VISION BACKBONE: SWIN TRANSFORMER ACTIVE",
            "> SENSOR INTERFACE: /dev/video0 CALIBRATED",
            "> PROTOCOL ARMED: AWAITING TARGET EXPRESSION",
        ]

        num_visible = min(len(lines), int(progress * (len(lines) + 1)))
        y_start = cy + 70

        for idx in range(num_visible):
            color = ARC_CYAN if idx == 0 else (STARK_GOLD if idx == len(lines) - 1 else HUD_WHITE)
            (tw, _), _ = cv2.getTextSize(lines[idx], font, 0.44, 1)
            cv2.putText(
                frame,
                lines[idx],
                ((w - tw) // 2, y_start + idx * 24),
                font,
                0.44,
                color,
                1,
                cv2.LINE_AA,
            )

        # Progress Bar & Status
        bar_w = 260
        bar_h = 6
        bx1 = (w - bar_w) // 2
        by1 = y_start + len(lines) * 24 + 15
        cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w, by1 + bar_h), (50, 40, 30), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bx1, by1), (bx1 + fill_w, by1 + bar_h), ARC_CYAN, -1)

        pct_str = f"SYSTEM STATUS: {int(progress * 100)}%"
        (pw, _), _ = cv2.getTextSize(pct_str, font, 0.36, 1)
        cv2.putText(frame, pct_str, ((w - pw) // 2, by1 + 18), font, 0.36, STARK_GOLD, 1, cv2.LINE_AA)

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
        """
        Main public interface: Render the full tactical HUD over a camera frame.

        Args:
            frame: OpenCV BGR image array (modified in place).
            candidates: Sequence of GroundingCandidate instances.
            search_buffer: Live query text currently in the search bar.
            active_prompt: Locked prompt currently evaluated by the model.
            box_threshold: Detection sensitivity threshold.
            latency_ms: Inference forward pass latency in ms.
            fps: Live video frame rate.
            ambiguous: Whether candidate ambiguity alert is active.
            status_msg: Short status string for the console chip.

        Returns:
            Annotated frame.
        """
        self.frame_count += 1

        # 1. Visor Screen Flourishes & Marginal Telemetry Rails
        self._draw_screen_visor_flourishes(frame)
        self._draw_marginal_telemetry_rails(frame)

        # 2. Render Target Reticles
        for idx, cand in enumerate(candidates):
            color = ARC_CYAN if idx == 0 else STARK_GOLD
            self._draw_target_reticle(frame, cand, index=idx + 1, color=color)

        # 3. Render Top Flight Telemetry Bar
        self._draw_top_telemetry(
            frame,
            active_prompt=active_prompt,
            latency_ms=latency_ms,
            num_candidates=len(candidates),
            ambiguous=ambiguous,
            box_threshold=box_threshold,
            fps=fps,
        )

        # 4. Render Bottom Cyber Command Console
        self._draw_bottom_console(
            frame,
            search_buffer=search_buffer,
            status_msg=status_msg,
        )

        return frame
