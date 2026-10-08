"""
Camera acquisition and frame streaming manager.

Runs threaded OpenCV hardware capture (buffer size 1) with fallback
synthetic frames to decouple video ingestion from model inference.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Threaded OpenCV video capture manager with ring buffer size 1."""

    def __init__(
        self,
        device_id: int = 0,
        target_width: int = 640,
        target_height: int = 480,
        target_fps: int = 30,
    ) -> None:
        self.device_id = device_id
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._cap: cv2.VideoCapture | None = None
        self._is_hardware_active = False

        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._fps_measured: float = 0.0
        self._frame_count: int = 0

        # Eagerly pre-populate initial frame buffer to prevent cold-start race conditions
        initial_frame = self._generate_synthetic_frame()
        self._latest_frame = initial_frame
        _, buf = cv2.imencode(".jpg", initial_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self._latest_jpeg = buf.tobytes()

    def start(self) -> bool:
        """Start the background acquisition thread."""
        if self._running:
            return True

        self._init_capture()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_worker,
            name="CameraCaptureWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "CameraManager started (hardware_active=%s, device_id=%d)",
            self._is_hardware_active,
            self.device_id,
        )
        return True

    def _init_capture(self) -> None:
        """Attempt to open hardware video capture device."""
        try:
            cap = cv2.VideoCapture(self.device_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap = cap
                self._is_hardware_active = True
                logger.info("Hardware camera %d acquired successfully", self.device_id)
            else:
                self._cap = None
                self._is_hardware_active = False
                logger.warning(
                    "Camera device %d unavailable; activating synthetic fallback feed",
                    self.device_id,
                )
        except Exception as e:
            self._cap = None
            self._is_hardware_active = False
            logger.warning("Error opening camera device %d: %s", self.device_id, e)

    def _capture_worker(self) -> None:
        """Dedicated background acquisition loop."""
        frame_interval = 1.0 / max(1, self.target_fps)
        t_prev = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()
            frame: np.ndarray | None = None

            if self._is_hardware_active and self._cap is not None:
                ret, raw_frame = self._cap.read()
                if ret and raw_frame is not None and raw_frame.size > 0:
                    frame = raw_frame
                else:
                    logger.warning("Hardware frame read failed; reinitializing capture")
                    self._is_hardware_active = False
                    if self._cap:
                        self._cap.release()
                        self._cap = None

            if frame is None:
                frame = self._generate_synthetic_frame()

            # Encode JPEG for direct web streaming
            success, enc_buf = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            jpeg_bytes = enc_buf.tobytes() if success else None

            now = time.perf_counter()
            dt = max(1e-5, now - t_prev)
            t_prev = now
            inst_fps = 1.0 / dt

            with self._lock:
                self._latest_frame = frame
                self._latest_jpeg = jpeg_bytes
                self._frame_count += 1
                self._fps_measured = 0.9 * self._fps_measured + 0.1 * inst_fps

            elapsed = time.perf_counter() - loop_start
            sleep_duration = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_duration)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate high-contrast tactical HUD test pattern if camera is absent."""
        h, w = self.target_height, self.target_width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (14, 18, 24)  # Obsidian dark background

        # Subtle grid lines
        grid_color = (30, 45, 55)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), grid_color, 1)

        # Crosshair and telemetry labels
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 245, 255), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 245, 255), 1)
        cv2.circle(frame, (cx, cy), 18, (0, 245, 255), 1)

        t_str = time.strftime("%H:%M:%S")
        cv2.putText(
            frame,
            f"STARK OPTICS - STANDBY [{t_str}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 245, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def get_frame(self) -> np.ndarray:
        """Get the latest BGR frame for Grounder model execution."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return self._generate_synthetic_frame()

    def get_jpeg(self) -> bytes | None:
        """Get the pre-encoded JPEG bytes for streaming."""
        with self._lock:
            return self._latest_jpeg

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Generator yielding MJPEG multipart chunks."""
        if not self._running:
            self.start()
        while self._running:
            jpeg = self.get_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(1.0 / max(1, self.target_fps))

    def get_status(self) -> dict:
        """Report camera hardware status and throughput."""
        with self._lock:
            return {
                "hardware_active": self._is_hardware_active,
                "fps": round(self._fps_measured, 1),
                "resolution": [self.target_width, self.target_height],
                "frames_captured": self._frame_count,
            }

    def stop(self) -> None:
        """Gracefully release camera resources and terminate worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_hardware_active = False
        logger.info("CameraManager terminated successfully")
