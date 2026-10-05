"""
STARK VISION — Target State Representation.

Defines lifecycle states for referring video segmentation targets:
SEARCHING -> GROUNDED -> TRACKING -> AMBIGUOUS -> RECOVERY / LOST.
"""

from __future__ import annotations

from enum import Enum, auto


class TargetState(str, Enum):
    """Target lifecycle states in real-time referring tracking."""

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    GROUNDED = "GROUNDED"
    TRACKING = "TRACKING"
    AMBIGUOUS = "AMBIGUOUS"
    RECOVERY = "RECOVERY"
    LOST = "LOST"
