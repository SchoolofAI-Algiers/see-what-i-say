"""
States for refering object tracking pipeline.
Cycles between searching, grounded, active tracking, and recovery when lost.
"""

from __future__ import annotations

from enum import Enum, auto


class TargetState(str, Enum):
    # lifecycle states for target tracking

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    GROUNDED = "GROUNDED"
    TRACKING = "TRACKING"
    AMBIGUOUS = "AMBIGUOUS"
    RECOVERY = "RECOVERY"
    LOST = "LOST"
