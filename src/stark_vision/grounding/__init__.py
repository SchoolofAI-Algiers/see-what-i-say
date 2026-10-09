"""
Grounding package for STARK VISION.
"""

from stark_vision.contracts import BoundingBox, Frame
from stark_vision.grounding.grounder import Grounder, GroundingCandidate, is_ambiguous

__all__ = ["Grounder", "GroundingCandidate", "BoundingBox", "Frame", "is_ambiguous"]

