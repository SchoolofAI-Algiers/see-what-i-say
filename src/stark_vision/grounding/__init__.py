"""Grounding package for STARK VISION.

Heavy imports (torch / transformers) are lazy so the pure-Python modules
(adaptive_threshold, disambiguation, temporal_filter) work in a CPU/mock install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stark_vision.contracts import BoundingBox, Frame, GroundingCandidate

if TYPE_CHECKING:  # pragma: no cover
    from stark_vision.grounding.grounder import Grounder, is_ambiguous

__all__ = ["Grounder", "GroundingCandidate", "BoundingBox", "Frame", "is_ambiguous"]


def __getattr__(name: str):
    if name in ("Grounder", "is_ambiguous"):
        from stark_vision.grounding import grounder

        return getattr(grounder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
