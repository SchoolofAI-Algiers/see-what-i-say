"""
Tracks active target lifecycle across SAM 2 and Grounding DINO.

Runs SAM 2 tracking each frame while rate limiting heavy Grounding DINO
passes so the laptop GPU doesn't choke on concurrent transformer inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from stark_vision.target.state_machine import TargetStateMachine
from stark_vision.target.target_state import TargetState


@dataclass
class TrackedTarget:
    # bounding box cooridnates [x1, y1, x2, y2] normalized
    box: tuple[float, float, float, float]
    confidence: float
    label: str
    mask: object | None = None
    last_updated: float = 0.0


class TargetManager:
    # keeps tracker state and throttles heavy DINO detection calls

    def __init__(
        self,
        grounding_cadence_sec: float = 0.50,  # default 2hz re-check cadance
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
        # resets target state when user gives a new speech or text prompt
        self.fsm.trigger_query(prompt)
        self.current_target = None
        self._last_grounding_time = 0.0

    def should_run_grounding(self, current_time: float | None = None) -> bool:
        # checks if we need a full detector pass or can stay on lightweight tracker
        now = current_time if current_time is not None else time.perf_counter()

        if self.state in (TargetState.SEARCHING, TargetState.RECOVERY):
            return True

        if self.state in (TargetState.AMBIGUOUS, TargetState.IDLE):
            return False

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
        # store bbox and start tracking transition
        now = current_time if current_time is not None else time.perf_counter()
        self.current_target = TrackedTarget(
            box=box,
            confidence=confidence,
            label=label,
            last_updated=now,
        )
        self.fsm.on_tracking_started()

    def on_tracker_step(self, confidence: float, box: tuple[float, float, float, float]) -> None:
        # called on each fast tracker frame to keep position current
        self.fsm.on_tracker_frame(confidence)
        if self.current_target is not None:
            self.current_target.confidence = confidence
            self.current_target.box = box
            self.current_target.last_updated = time.perf_counter()

    def reset(self) -> None:
        # drops active track and sets fsm back to IDLE
        self.fsm.reset()
        self.current_target = None
        self._last_grounding_time = 0.0
