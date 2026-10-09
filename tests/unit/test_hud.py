"""
Unit tests for STARK VISION JarvisHUD Overlay Engine.

Validates geometry rendering, cyber chamfers, Arc Reactor trigonometry,
voice waveform calculation, target reticle bounding box annotations,
boot sequence lifecycle, and frame throughput latency.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from stark_vision.grounding.grounder import GroundingCandidate
from stark_vision.hud.overlay import (
    ARC_CYAN,
    OBSIDIAN_GLASS,
    PLASMA_BLUE,
    STARK_GOLD,
    JarvisHUD,
)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Create a standard 640x480 3-channel BGR synthetic frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_candidates() -> list[GroundingCandidate]:
    """Create sample target candidates with known coordinates."""
    return [
        GroundingCandidate(
            box=(100.0, 80.0, 300.0, 380.0),
            score=0.942,
            label="person",
        ),
        GroundingCandidate(
            box=(350.0, 120.0, 480.0, 320.0),
            score=0.815,
            label="laptop",
        ),
    ]


def test_hud_initialization():
    """Verify default attributes and resolution assignment."""
    hud = JarvisHUD(target_resolution=(1280, 720), enable_voice=False)
    assert hud.default_w == 1280
    assert hud.default_h == 720
    assert not hud.is_booted
    assert hud.boot_start_time is None
    assert hud.frame_count == 0


def test_chamfered_panel_rendering(blank_frame: np.ndarray):
    """Verify chamfered polygon drawing modifies pixels within bounds."""
    frame = blank_frame.copy()
    JarvisHUD._draw_chamfered_panel(
        frame,
        pt1=(50, 50),
        pt2=(200, 150),
        chamfer=10,
        bg_color=OBSIDIAN_GLASS,
        border_color=ARC_CYAN,
        alpha=0.8,
        border_thickness=2,
    )
    # Check that region inside the panel has non-zero pixel values
    assert np.any(frame[60:140, 60:190] > 0)


def test_arc_reactor_animation(blank_frame: np.ndarray):
    """Verify rotating Arc Reactor rings render without errors across frames."""
    hud = JarvisHUD(enable_voice=False)
    frame1 = blank_frame.copy()
    hud.frame_count = 0
    hud._draw_arc_reactor(frame1, center=(100, 100), radius=30)
    assert np.any(frame1 > 0)

    # Frame count increments rotate angles
    frame2 = blank_frame.copy()
    hud.frame_count = 45
    hud._draw_arc_reactor(frame2, center=(100, 100), radius=30)
    assert np.any(frame2 > 0)
    # Pixel representations differ due to rotation
    assert not np.array_equal(frame1, frame2)


def test_voice_waveform_envelope(blank_frame: np.ndarray):
    """Verify acoustic waveform generates multi-bar visualizer."""
    hud = JarvisHUD(enable_voice=False)
    frame = blank_frame.copy()
    hud._draw_voice_waveform(frame, center=(320, 40), is_active=True)
    assert np.any(frame[20:60, 280:360] > 0)


def test_target_reticle_geometry(blank_frame: np.ndarray, dummy_candidates: list[GroundingCandidate]):
    """Verify target reticles render chamfered brackets, reticle crosshairs, and badges."""
    hud = JarvisHUD(enable_voice=False)
    frame = blank_frame.copy()

    for idx, cand in enumerate(dummy_candidates):
        color = ARC_CYAN if idx == 0 else STARK_GOLD
        hud._draw_target_reticle(frame, cand, index=idx + 1, color=color)

    # Center of target 1 should contain crosshair pixels
    cx, cy = int(dummy_candidates[0].center[0]), int(dummy_candidates[0].center[1])
    assert np.any(frame[cy - 5 : cy + 5, cx - 5 : cx + 5] > 0)


def test_boot_sequence_lifecycle(blank_frame: np.ndarray):
    """Verify the 2.5-second boot diagnostic progression and completion."""
    hud = JarvisHUD(enable_voice=False)
    frame = blank_frame.copy()

    # Step 1: Initial call sets start time and returns False
    is_done = hud.render_boot_sequence(frame)
    assert not is_done
    assert hud.boot_start_time is not None
    assert not hud.is_booted

    # Step 2: Simulate elapsed time past duration
    hud.boot_start_time = time.time() - 3.0
    is_done = hud.render_boot_sequence(frame)
    assert is_done
    assert hud.is_booted

    # Step 3: Subsequent call returns True immediately
    assert hud.render_boot_sequence(frame)


def test_full_render_throughput(blank_frame: np.ndarray, dummy_candidates: list[GroundingCandidate]):
    """Verify full HUD render completes well within the 33ms budget for 30 FPS."""
    hud = JarvisHUD(enable_voice=False)
    hud.is_booted = True
    frame = blank_frame.copy()

    # Warmup
    hud.render(
        frame=frame,
        candidates=dummy_candidates,
        search_buffer="person",
        active_prompt="person",
        box_threshold=0.25,
        latency_ms=22.5,
        fps=30.0,
        status_msg="Locked: 2",
    )

    # Benchmark 20 iterations
    t0 = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        hud.render(
            frame=frame,
            candidates=dummy_candidates,
            search_buffer="person",
            active_prompt="person",
            box_threshold=0.25,
            latency_ms=22.5,
            fps=30.0,
            status_msg="Locked: 2",
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iterations

    # Must render in under 15 ms per frame (budget is 33.3ms for 30 FPS)
    assert elapsed_ms < 15.0, f"HUD render took {elapsed_ms:.2f}ms, exceeding budget."
