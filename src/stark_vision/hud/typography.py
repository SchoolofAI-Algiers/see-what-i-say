"""
Draws cached TrueType fonts directly onto OpenCV BGR frames.
Pillow renders text chips once into an alpha mask so we avoid font rasterization every frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
ORBITRON_PATH = FONTS_DIR / "Orbitron-Bold.ttf"
JETBRAINS_PATH = FONTS_DIR / "JetBrainsMono-Bold.ttf"

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_CHIP_CACHE: dict[tuple[str, str, int, tuple[int, int, int]], tuple[np.ndarray, np.ndarray]] = {}


def get_font(
    font_name: Literal["orbitron", "mono"] = "orbitron",
    size: int = 14,
) -> ImageFont.FreeTypeFont:
    # returns loaded font handle or falls back to default
    key = (font_name, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    target_path = ORBITRON_PATH if font_name == "orbitron" else JETBRAINS_PATH
    if target_path.exists():
        try:
            font = ImageFont.truetype(str(target_path), size)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            pass

    fallback = ImageFont.load_default()
    return fallback  # type: ignore[return-value]


def _render_chip(
    text: str,
    font_name: Literal["orbitron", "mono"],
    size: int,
    bgr_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    # renders small RGBA text slice and cached alpha mask
    key = (text, font_name, size, bgr_color)
    if key in _CHIP_CACHE:
        return _CHIP_CACHE[key]

    # limit cache size to prevent memory creep
    if len(_CHIP_CACHE) > 512:
        _CHIP_CACHE.clear()

    font = get_font(font_name, size)
    bbox = font.getbbox(text)
    if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        mask = np.zeros((1, 1, 1), dtype=np.float32)
        _CHIP_CACHE[key] = (empty, mask)
        return empty, mask

    w = max(1, bbox[2] - bbox[0] + 6)
    h = max(1, bbox[3] - bbox[1] + 6)
    chip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(chip)
    b, g, r = bgr_color
    draw.text((-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=(r, g, b, 255))

    chip_np = np.asarray(chip)
    # convert RGB to BGR
    bgr_chip = chip_np[:, :, :3][:, :, ::-1].copy()
    alpha_chip = (chip_np[:, :, 3:] / 255.0).astype(np.float32)

    _CHIP_CACHE[key] = (bgr_chip, alpha_chip)
    return bgr_chip, alpha_chip


def blit_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_name: Literal["orbitron", "mono"] = "orbitron",
    size: int = 14,
    bgr_color: tuple[int, int, int] = (255, 240, 0),
) -> None:
    # blits rendered text chip over numpy frame slice
    if not text:
        return

    bgr_chip, alpha_chip = _render_chip(text, font_name, size, bgr_color)
    h_chip, w_chip = bgr_chip.shape[:2]
    x, y = pos
    h_frame, w_frame = frame.shape[:2]

    if x >= w_frame or y >= h_frame or x + w_chip <= 0 or y + h_chip <= 0:
        return

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w_chip)
    y2 = min(h_frame, y + h_chip)

    cx1 = x1 - x
    cy1 = y1 - y
    cx2 = cx1 + (x2 - x1)
    cy2 = cy1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    bgr_slice = bgr_chip[cy1:cy2, cx1:cx2].astype(np.float32)
    a_slice = alpha_chip[cy1:cy2, cx1:cx2]

    frame[y1:y2, x1:x2] = (roi * (1.0 - a_slice) + bgr_slice * a_slice).astype(np.uint8)
