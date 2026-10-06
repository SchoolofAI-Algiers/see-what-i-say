"""Application-level data contracts shared by STARK VISION subsystems.

Images crossing application boundaries are RGB. OpenCV/model-specific
representations belong inside their adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class Frame:
    id: int
    timestamp_ns: int
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame id and timestamp must be non-negative")
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("frame.image must be an RGB uint8 array with shape HxWx3")
        if self.image.shape[0] == 0 or self.image.shape[1] == 0:
            raise ValueError("frame.image cannot be empty")


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.samples.dtype != np.float32 or self.samples.ndim != 1:
            raise ValueError("audio samples must be a one-dimensional float32 array")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None


@dataclass(frozen=True)
class BoundingBox:
    """Continuous pixel-edge coordinates, origin at the top-left, x2/y2 exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class GroundingCandidate:
    box: BoundingBox
    score: float
    label: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be in [0, 1]")


@dataclass(frozen=True)
class GroundingRequest:
    request_id: int
    prompt: str
    frame: Frame


@dataclass(frozen=True)
class GroundingResult:
    request_id: int
    source_frame_id: int
    candidates: tuple[GroundingCandidate, ...]


class CommandType(str, Enum):
    SELECT_TARGET = "select_target"
    SET_EFFECT = "set_effect"
    RESOLVE_CANDIDATE = "resolve_candidate"
    STOP_TRACKING = "stop_tracking"
    UNKNOWN = "unknown"


class EffectMode(str, Enum):
    NONE = "none"
    HIGHLIGHT = "highlight"
    BLUR = "blur"
    ISOLATE = "isolate"
    RECOLOR = "recolor"


class CandidateSelectorKind(str, Enum):
    SIDE = "side"
    RANK = "rank"


@dataclass(frozen=True)
class CandidateSelector:
    kind: CandidateSelectorKind
    value: str | int


@dataclass(frozen=True)
class EffectSpec:
    mode: EffectMode
    color_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class CommandEvent:
    type: CommandType
    transcript: str
    prompt: str | None = None
    effect: EffectSpec | None = None
    selector: CandidateSelector | None = None


class TrackingStatus(str, Enum):
    TRACKING = "tracking"
    UNCERTAIN = "uncertain"
    LOST = "lost"


@dataclass(frozen=True)
class TrackingResult:
    target_id: int
    frame_id: int
    box: BoundingBox | None
    mask: np.ndarray | None
    confidence: float | None
    status: TrackingStatus

    def __post_init__(self) -> None:
        if self.mask is not None and (self.mask.dtype != np.bool_ or self.mask.ndim != 2):
            raise ValueError("tracking mask must be a two-dimensional bool array")
        if self.status is TrackingStatus.LOST and self.mask is not None:
            raise ValueError("lost tracking results must not carry a stale mask")


@dataclass(frozen=True)
class TargetSnapshot:
    target_id: int
    prompt: str
    box: BoundingBox | None
    mask: np.ndarray | None
    confidence: float | None
    status: TrackingStatus
    frame_id: int


@dataclass(frozen=True)
class RuntimeSnapshot:
    frame_id: int
    active_target: TargetSnapshot | None
    pending_prompt: str | None
    pending_candidates: tuple[GroundingCandidate, ...]
    pending_is_ambiguous: bool
    effect: EffectSpec
    message: str
    capture_fps: float = 0.0
    tracking_fps: float = 0.0


@dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray
    telemetry: tuple[str, ...]
