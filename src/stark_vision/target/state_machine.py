"""
STARK VISION — Target State Machine.

Manages deterministic state transitions between voice grounding, SAM 2 tracking,
ambiguity resolution, and recovery.
"""

from __future__ import annotations

import logging
from typing import Sequence

from stark_vision.target.target_state import TargetState

logger = logging.getLogger(__name__)


class TargetStateMachine:
    """
    Deterministic finite state machine governing visual target tracking lifecycle.
    """

    def __init__(self, recovery_confidence_threshold: float = 0.40) -> None:
        self.state = TargetState.IDLE
        self.recovery_confidence_threshold = recovery_confidence_threshold
        self.active_prompt: str = ""
        self.clarification_message: str | None = None

    def trigger_query(self, text_prompt: str) -> TargetState:
        """New user voice/text query initiates search."""
        self.active_prompt = text_prompt.strip()
        self.clarification_message = None
        self.state = TargetState.SEARCHING
        logger.debug("State transition -> SEARCHING for query '%s'", self.active_prompt)
        return self.state

    def on_grounding_candidates(
        self,
        candidate_count: int,
        is_ambiguous: bool = False,
        clarification_msg: str | None = None,
    ) -> TargetState:
        """Handle candidates returned from Grounding DINO."""
        if candidate_count == 0:
            # Nothing detected yet, remain searching
            self.state = TargetState.SEARCHING
        elif is_ambiguous:
            self.state = TargetState.AMBIGUOUS
            self.clarification_message = clarification_msg
        else:
            self.state = TargetState.GROUNDED
            self.clarification_message = None

        logger.debug("State transition -> %s (count=%d)", self.state.value, candidate_count)
        return self.state

    def on_tracking_started(self) -> TargetState:
        """SAM 2 initialized with prompt box and started mask propagation."""
        if self.state in (TargetState.SEARCHING, TargetState.GROUNDED, TargetState.RECOVERY):
            self.state = TargetState.TRACKING
            logger.debug("State transition -> TRACKING")
        return self.state

    def on_tracker_frame(self, tracker_confidence: float) -> TargetState:
        """Per-frame update from SAM 2 tracking at 30 FPS."""
        if self.state == TargetState.TRACKING:
            if tracker_confidence < self.recovery_confidence_threshold:
                self.state = TargetState.RECOVERY
                logger.warning(
                    "Tracker confidence dropped to %.2f (threshold: %.2f) -> RECOVERY",
                    tracker_confidence,
                    self.recovery_confidence_threshold,
                )
        return self.state

    def on_target_lost(self) -> TargetState:
        """Target completely out of frame or lost."""
        self.state = TargetState.LOST
        logger.debug("State transition -> LOST")
        return self.state

    def reset(self) -> TargetState:
        """Reset state machine to IDLE."""
        self.state = TargetState.IDLE
        self.active_prompt = ""
        self.clarification_message = None
        return self.state
