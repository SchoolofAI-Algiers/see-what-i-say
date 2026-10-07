"""Small timestamped frame history used to seed a delayed segmentation request."""

from __future__ import annotations

from collections import deque
from threading import RLock

from stark_vision.contracts import Frame


class FrameBuffer:
    def __init__(
        self,
        max_frames: int = 30,
        max_bytes: int = 64 * 1024 * 1024,
        max_age_ns: int = 2_000_000_000,
    ) -> None:
        if max_frames <= 0 or max_bytes <= 0 or max_age_ns <= 0:
            raise ValueError("frame buffer limits must be positive")
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.max_age_ns = max_age_ns
        self._frames: deque[Frame] = deque()
        self._bytes = 0
        self._lock = RLock()

    @property
    def latest(self) -> Frame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def append(self, frame: Frame) -> None:
        with self._lock:
            if self._frames:
                previous = self._frames[-1]
                if frame.id <= previous.id or frame.timestamp_ns <= previous.timestamp_ns:
                    raise ValueError("frames must be appended in strictly increasing order")
            self._frames.append(frame)
            self._bytes += frame.image.nbytes
            self._prune(frame.timestamp_ns)

    def get(self, frame_id: int) -> Frame | None:
        with self._lock:
            return next((frame for frame in self._frames if frame.id == frame_id), None)

    def after(self, frame_id: int) -> tuple[Frame, ...] | None:
        """Return contiguous later frames, or None when the seed/history has expired."""
        with self._lock:
            seed = next((frame for frame in self._frames if frame.id == frame_id), None)
            if seed is None:
                return None
            later = tuple(frame for frame in self._frames if frame.id > frame_id)
            expected_id = frame_id + 1
            for frame in later:
                if frame.id != expected_id:
                    return None
                expected_id += 1
            return later

    def _prune(self, newest_timestamp_ns: int) -> None:
        while self._frames and (
            len(self._frames) > self.max_frames
            or self._bytes > self.max_bytes
            or newest_timestamp_ns - self._frames[0].timestamp_ns > self.max_age_ns
        ):
            self._bytes -= self._frames.popleft().image.nbytes
