"""
Unit tests for CameraManager.

Verifies thread-safe frame acquisition, synthetic test patterns,
JPEG serialization, and lifecycle management.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from stark_vision.api.camera_manager import CameraManager


def test_camera_manager_synthetic_fallback():
    """Verify synthetic frame generator when camera device is inactive."""
    cam = CameraManager(device_id=999, target_width=320, target_height=240, target_fps=15)
    cam.start()

    time.sleep(0.1)  # Allow worker thread to produce initial frame
    frame = cam.get_frame()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8

    status = cam.get_status()
    assert status["resolution"] == [320, 240]
    assert status["hardware_active"] is False

    cam.stop()


def test_camera_manager_jpeg_encoding():
    """Verify JPEG output format and streaming chunk headers."""
    cam = CameraManager(device_id=999, target_width=320, target_height=240, target_fps=20)
    cam.start()

    time.sleep(0.1)
    jpeg_bytes = cam.get_jpeg()

    assert jpeg_bytes is not None
    # JPEG magic bytes: FF D8
    assert jpeg_bytes.startswith(b"\xff\xd8")

    # Verify MJPEG chunk generator
    gen = cam.generate_mjpeg()
    chunk = next(gen)
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk

    cam.stop()


def test_camera_manager_start_stop_idempotence():
    """Verify start and stop methods are safe to call multiple times."""
    cam = CameraManager(device_id=999)
    assert cam.start() is True
    assert cam.start() is True  # Idempotent

    cam.stop()
    assert cam._running is False
