"""
STARK VISION — Multi-Dimensional Ambiguity Resolver & Spatial Disambiguator.

Replaces basic delta heuristics with spatial intelligence:
1. Center-Distance + Containment (IoMin) clustering: Merges multiple proposal boxes
   wrapping the exact same object while preserving distinct adjacent targets in crowds.
2. Directional spatial reasoning: Resolves "left", "right", "nearest", "center", "top", "bottom".
3. Conversational clarification question generator for true multi-instance ambiguity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SpatialCandidate:
    """Internal candidate representation with normalized coordinates."""

    box: tuple[float, float, float, float]  # [x1, y1, x2, y2]
    score: float
    label: str

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def area(self) -> float:
        w = max(0.0, self.box[2] - self.box[0])
        h = max(0.0, self.box[3] - self.box[1])
        return w * h


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def compute_iomin(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Minimum Area (Containment Index)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1e-4, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-4, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return intersection / min(area_a, area_b)


class SpatialResolver:
    """
    Industrial spatial disambiguation engine for multimodal referring expressions.
    """

    SPATIAL_KEYWORDS = {
        "left": "left",
        "leftmost": "left",
        "on the left": "left",
        "to the left": "left",
        "right": "right",
        "rightmost": "right",
        "on the right": "right",
        "to the right": "right",
        "center": "center",
        "middle": "center",
        "in the center": "center",
        "in the middle": "center",
        "closest": "nearest",
        "nearest": "nearest",
        "largest": "nearest",
        "biggest": "nearest",
        "top": "top",
        "upper": "top",
        "highest": "top",
        "bottom": "bottom",
        "lower": "bottom",
        "lowest": "bottom",
    }

    @staticmethod
    def extract_spatial_cue(text_prompt: str) -> str | None:
        """Extract directional cue from natural language prompt if present."""
        if not text_prompt:
            return None
        prompt_lower = text_prompt.lower()
        for pattern, cue in SpatialResolver.SPATIAL_KEYWORDS.items():
            if re.search(r"\b" + re.escape(pattern) + r"\b", prompt_lower):
                return cue
        return None

    @staticmethod
    def is_same_object(
        cand_a: SpatialCandidate,
        cand_b: SpatialCandidate,
        iou_threshold: float = 0.55,
        containment_threshold: float = 0.70,
    ) -> bool:
        """
        Determine if two proposal boxes belong to the same physical object.

        Checks:
        1. High IoU (> 0.55), OR
        2. High Containment (IoMin > 0.70) with small center offset.
        This preserves adjacent distinct targets in crowds while merging duplicate proposals.
        """
        iou = compute_iou(cand_a.box, cand_b.box)
        if iou >= iou_threshold:
            return True

        iomin = compute_iomin(cand_a.box, cand_b.box)
        if iomin >= containment_threshold:
            # Check center distance
            dist = math.hypot(cand_a.center_x - cand_b.center_x, cand_a.center_y - cand_b.center_y)
            avg_diag = math.sqrt(cand_a.area + cand_b.area)
            if dist < 0.35 * avg_diag:
                return True

        return False

    @classmethod
    def deduplicate_candidates(
        cls,
        candidates: Sequence[SpatialCandidate],
        iou_threshold: float = 0.55,
    ) -> list[SpatialCandidate]:
        """Deduplicate candidates belonging to the same physical object."""
        if len(candidates) <= 1:
            return list(candidates)

        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        unique_cands: list[SpatialCandidate] = []

        for cand in sorted_cands:
            is_dup = False
            for kept in unique_cands:
                if cls.is_same_object(cand, kept, iou_threshold=iou_threshold):
                    is_dup = True
                    break
            if not is_dup:
                unique_cands.append(cand)

        return unique_cands

    def evaluate_ambiguity(
        self,
        candidates: Sequence[SpatialCandidate],
        text_prompt: str,
        delta_threshold: float = 0.15,
    ) -> tuple[bool, list[SpatialCandidate], str | None]:
        """
        Evaluate and optionally resolve candidate ambiguity using spatial reasoning.

        Returns:
            (is_ambiguous, resolved_candidates, clarification_message)
        """
        if not candidates:
            return False, [], None

        # Step 1: Cluster proposals of the same physical object
        distinct_cands = self.deduplicate_candidates(candidates)

        if len(distinct_cands) <= 1:
            return False, distinct_cands, None

        # Step 2: Auto-resolve if user provided a spatial cue
        spatial_cue = self.extract_spatial_cue(text_prompt)
        if spatial_cue:
            resolved = list(distinct_cands)
            if spatial_cue == "left":
                resolved.sort(key=lambda c: c.center_x)
                return False, [resolved[0]], None
            elif spatial_cue == "right":
                resolved.sort(key=lambda c: c.center_x, reverse=True)
                return False, [resolved[0]], None
            elif spatial_cue == "nearest":
                resolved.sort(key=lambda c: c.area, reverse=True)
                return False, [resolved[0]], None
            elif spatial_cue == "top":
                resolved.sort(key=lambda c: c.center_y)
                return False, [resolved[0]], None
            elif spatial_cue == "bottom":
                resolved.sort(key=lambda c: c.center_y, reverse=True)
                return False, [resolved[0]], None
            elif spatial_cue == "center":
                # Distance to image center (assuming normalized [0, 1] or pixel coords)
                avg_cx = sum(c.center_x for c in resolved) / len(resolved)
                avg_cy = sum(c.center_y for c in resolved) / len(resolved)
                resolved.sort(key=lambda c: math.hypot(c.center_x - avg_cx, c.center_y - avg_cy))
                return False, [resolved[0]], None

        # Step 3: Check score delta between the top two distinct objects
        score_gap = abs(distinct_cands[0].score - distinct_cands[1].score)
        if score_gap < delta_threshold:
            label = distinct_cands[0].label
            clarification = (
                f"Detected {len(distinct_cands)} matching instances of '{label}'. "
                f"Please specify (e.g. 'on the left' or 'on the right')."
            )
            return True, distinct_cands, clarification

        return False, distinct_cands, None
