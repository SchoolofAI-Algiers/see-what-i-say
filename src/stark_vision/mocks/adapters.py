"""Predictable fakes for the hardware and model adapter contracts."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from stark_vision.contracts import (
    AudioBuffer,
    BoundingBox,
    Frame,
    GroundingCandidate,
    TrackingResult,
    TrackingStatus,
    Transcript,
)


class MockCamera:
    def __init__(self, frames: Iterable[Frame]) -> None:
        self._frames = iter(frames)
        self.closed = False

    def read(self) -> Frame | None:
        if self.closed:
            return None
        return next(self._frames, None)

    def close(self) -> None:
        self.closed = True


class MockAudioRecorder:
    def __init__(self, sample_rate_hz: int = 16_000) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.recording = False
        self.start_count = 0

    def start(self) -> None:
        if self.recording:
            raise RuntimeError("recording is already active")
        self.recording = True
        self.start_count += 1

    def stop(self) -> AudioBuffer:
        if not self.recording:
            raise RuntimeError("recording has not started")
        self.recording = False
        return AudioBuffer(np.zeros(self.sample_rate_hz // 4, dtype=np.float32), self.sample_rate_hz)


class MockSTT:
    def __init__(self, transcripts: Iterable[str]) -> None:
        self._transcripts = deque(transcripts)
        self.call_count = 0

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        if not self._transcripts:
            raise RuntimeError("mock transcript queue is empty")
        self.call_count += 1
        return Transcript(self._transcripts.popleft(), language="en")


class MockGrounder:
    def __init__(self, candidates: Iterable[GroundingCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.calls: list[tuple[int, str]] = []

    def find(self, frame: Frame, text_prompt: str) -> list[GroundingCandidate]:
        self.calls.append((frame.id, text_prompt))
        return list(self.candidates)


class MockTracker:
    """A small moving-box tracker whose mask has the agreed HxW boolean shape."""

    def __init__(self, fail_target_ids: Iterable[int] = ()) -> None:
        self._targets: dict[int, tuple[BoundingBox, int]] = {}
        self._fail_ids = set(fail_target_ids)
        self.initialize_count = 0
        self.update_count = 0
        self.removed_ids: list[int] = []
        self.reset_count = 0

    def initialize(self, frame: Frame, box: BoundingBox, target_id: int) -> TrackingResult:
        self.initialize_count += 1
        if target_id in self._fail_ids:
            return TrackingResult(target_id, frame.id, box, None, 0.0, TrackingStatus.LOST)
        self._targets[target_id] = (box, frame.id)
        return self._result(frame, target_id, box)

    def update(self, frame: Frame) -> tuple[TrackingResult, ...]:
        self.update_count += 1
        results = []
        for target_id, (initial_box, seed_id) in tuple(self._targets.items()):
            shift = max(0, frame.id - seed_id)
            width, height = frame.image.shape[1], frame.image.shape[0]
            x_shift = min(float(shift), max(0.0, width - initial_box.x2))
            y_shift = min(float(shift // 2), max(0.0, height - initial_box.y2))
            box = BoundingBox(
                initial_box.x1 + x_shift,
                initial_box.y1 + y_shift,
                initial_box.x2 + x_shift,
                initial_box.y2 + y_shift,
            )
            self._targets[target_id] = (initial_box, seed_id)
            results.append(self._result(frame, target_id, box))
        return tuple(results)

    def remove(self, target_id: int) -> None:
        self._targets.pop(target_id, None)
        self.removed_ids.append(target_id)

    def reset(self) -> None:
        self._targets.clear()
        self.reset_count += 1

    @staticmethod
    def _result(frame: Frame, target_id: int, box: BoundingBox) -> TrackingResult:
        height, width = frame.image.shape[:2]
        x1 = max(0, min(width, int(box.x1)))
        y1 = max(0, min(height, int(box.y1)))
        x2 = max(0, min(width, int(np.ceil(box.x2))))
        y2 = max(0, min(height, int(np.ceil(box.y2))))
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[y1:y2, x1:x2] = True
        return TrackingResult(target_id, frame.id, box, mask, 0.97, TrackingStatus.TRACKING)


def make_scene_frames(
    count: int,
    width: int = 96,
    height: int = 64,
    start_id: int = 0,
    step_ns: int = 33_333_333,
) -> list[Frame]:
    """Create deterministic RGB frames with two visible people-like regions."""
    frames: list[Frame] = []
    for frame_id in range(start_id, start_id + count):
        image = np.full((height, width, 3), (24, 28, 36), dtype=np.uint8)
        drift = min(frame_id - start_id, 4)
        image[14:48, 10 + drift:30 + drift] = (210, 45, 45)
        image[13:50, 58 - drift:82 - drift] = (180, 35, 35)
        frames.append(Frame(frame_id, (frame_id + 1) * step_ns, image))
    return frames
