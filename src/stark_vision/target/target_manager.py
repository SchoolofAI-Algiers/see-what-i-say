"""
STARK VISION — Target Manager with Rate-Decoupled Scheduling.

Coordinates between asynchronous Grounding DINO (1-2 Hz / voice events)
and real-time SAM 2 mask tracking (30 FPS) within RTX 4060 GPU budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from stark_vision.target.state_machine import TargetStateMachine
from stark_vision.target.target_state import TargetState


@dataclass
class TrackedTarget:
    """Active target metadata and spatial state."""

    box: tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    label: str
    mask: object | None = None
    last_updated: float = 0.0


class TargetManager:
    """
    Orchestrates target tracking lifecycle with rate decoupling.

    Enforces:
    - SAM 2 runs at 30 FPS for high-rate continuous mask tracking.
    - Grounding DINO runs at 1-2 Hz (or on-demand upon voice command / recovery),
      preserving 8 GB VRAM and computational headroom on laptop GPUs.
    """

    def __init__(
        self,
        grounding_cadence_sec: float = 0.50,  # 2 Hz re-verification
        recovery_threshold: float = 0.40,
    ) -> None:
        self.grounding_cadence_sec = grounding_cadence_sec
        self.fsm = TargetStateMachine(recovery_confidence_threshold=recovery_threshold)
        self.current_target: TrackedTarget | None = None
        self._last_grounding_time: float = 0.0

    @property
    def state(self) -> TargetState:
        return self.fsm.state

    def set_query(self, prompt: str) -> None:
        """Triggered when a new voice or text referring expression arrives."""
        self.fsm.trigger_query(prompt)
        self.current_target = None
        # Force immediate grounding on next frame
        self._last_grounding_time = 0.0

    def should_run_grounding(self, current_time: float | None = None) -> bool:
        """
        Rate Decoupling Scheduler.

        Returns True if Grounding DINO should run on the current frame.
        Returns False if SAM 2 tracking alone is sufficient, saving GPU cycles.
        """
        now = current_time if current_time is not None else time.perf_counter()

        # 1. Searching or Recovery requires immediate grounding
        if self.state in (TargetState.SEARCHING, TargetState.RECOVERY):
            return True

        # 2. Ambiguous requires user clarification before new visual pass
        if self.state in (TargetState.AMBIGUOUS, TargetState.IDLE):
            return False

        # 3. In active TRACKING state, run grounding at low cadence (1-2 Hz) for re-verification
        if self.state == TargetState.TRACKING:
            return (now - self._last_grounding_time) >= self.grounding_cadence_sec

        return False

    def mark_grounding_executed(self, current_time: float | None = None) -> None:
        """Record timestamp of completed Grounding DINO pass."""
        self._last_grounding_time = current_time if current_time is not None else time.perf_counter()

    def update_target(
        self,
        box: tuple[float, float, float, float],
        confidence: float,
        label: str,
        current_time: float | None = None,
    ) -> None:
        """Update active target coordinates and confidence."""
        now = current_time if current_time is not None else time.perf_counter()
        self.current_target = TrackedTarget(
            box=box,
            confidence=confidence,
            label=label,
            last_updated=now,
        )
        self.fsm.on_tracking_started()

    def on_tracker_step(self, confidence: float, box: tuple[float, float, float, float]) -> None:
        """High-frequency per-frame update from SAM 2 (at 30 FPS)."""
        self.fsm.on_tracker_frame(confidence)
        if self.current_target is not None:
            self.current_target.confidence = confidence
            self.current_target.box = box
            self.current_target.last_updated = time.perf_counter()

    def reset(self) -> None:
        """Reset target manager to IDLE."""
        self.fsm.reset()
        self.current_target = None
        self._last_grounding_time = 0.0
