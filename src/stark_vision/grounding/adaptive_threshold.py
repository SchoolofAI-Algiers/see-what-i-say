"""
STARK VISION — Dynamic Adaptive Thresholding (DAT) Engine.

Eliminates static hardcoded confidence thresholds (e.g., fixed 0.25).
Dynamically computes optimal detection cutoffs based on:
1. Linguistic query complexity (token count, modifiers, spatial descriptors)
2. Score distribution & relative drop-off ratio (gamma falloff)
3. Background noise floor estimation & safe range boundaries
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration boundaries for dynamic threshold calibration."""

    min_box_threshold: float = 0.15
    max_box_threshold: float = 0.65
    terse_base_threshold: float = 0.28   # Benchmark calibrated baseline for terse nouns ("table", "chair", "person")
    relative_dropoff_ratio: float = 0.55  # Keep candidates within 55% of top score
    text_threshold_ratio: float = 0.75    # Text alignment threshold derived relative to box


class DynamicThresholdCalculator:
    """
    Production-grade dynamic thresholding for open-vocabulary visual grounding.

    Adapts sensitivity automatically between terse queries ("person") and
    dense referring expressions ("person in black hoodie near the coffee table").
    """

    # Common descriptive modifiers in referring expressions
    MODIFIERS = frozenset({
        "in", "on", "with", "wearing", "holding", "near", "next", "to",
        "behind", "under", "front", "black", "white", "red", "blue",
        "green", "yellow", "dark", "light", "small", "large", "tall",
        "left", "right", "standing", "sitting", "walking"
    })

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()

    def estimate_query_complexity(self, text_prompt: str) -> float:
        """
        Estimate linguistic complexity penalty factor in [0.0, 1.0].

        Longer queries with multiple modifiers spread cross-attention
        activation across multiple tokens, reducing peak sigmoid confidence.
        """
        if not text_prompt or not text_prompt.strip():
            return 0.0

        words = re.findall(r"\b\w+\b", text_prompt.lower())
        word_count = len(words)
        if word_count <= 1:
            return 0.0

        modifier_count = sum(1 for w in words if w in self.MODIFIERS)

        # Complexity scales smoothly from 0.0 (single noun) to 1.0 (dense sentence)
        complexity = min(1.0, (max(0, word_count - 1) * 0.12) + (modifier_count * 0.10))
        return round(complexity, 3)

    def compute_baseline_thresholds(
        self, text_prompt: str
    ) -> tuple[float, float]:
        """
        Compute baseline (box_threshold, text_threshold) adapted to query complexity.

        Returns:
            (box_threshold, text_threshold) strictly within [min_box_threshold, max_box_threshold]
        """
        if not text_prompt or not text_prompt.strip():
            # Edge case: empty query defaults to safe conservative threshold
            return self.config.terse_base_threshold, round(self.config.terse_base_threshold * self.config.text_threshold_ratio, 3)

        complexity = self.estimate_query_complexity(text_prompt)

        # Terse queries anchor at 0.28 (benchmarked on Grounding DINO);
        # Complex expressions ("person in black hoodie near table") scale down to ~0.18.
        box_thresh = self.config.terse_base_threshold - (complexity * 0.10)
        box_thresh = max(
            self.config.min_box_threshold,
            min(self.config.max_box_threshold, box_thresh),
        )

        text_thresh = max(0.12, box_thresh * self.config.text_threshold_ratio)
        return round(box_thresh, 3), round(text_thresh, 3)

    def filter_by_relative_falloff(
        self,
        scores: Sequence[float],
        box_threshold: float,
    ) -> list[int]:
        """
        Filter candidate indices using relative score drop-off with noise-floor rejection.

        If the top candidate has high confidence (e.g. 0.90), candidates below 0.90 * gamma
        (0.54) are rejected as background noise, even if they exceed box_threshold.
        If the top candidate does not meet box_threshold, returns empty (absent referent).

        Args:
            scores: Sequence of candidate scores (must be non-empty and finite).
            box_threshold: Absolute baseline floor.

        Returns:
            List of accepted indices.
        """
        if not scores:
            return []

        # Sanitize scores and ensure descending order
        clean_scores = [float(s) if not (math.isnan(s) or math.isinf(s)) else 0.0 for s in scores]
        top_score = max(clean_scores)

        # Absent referent check: if even the top detection fails the baseline, reject all
        if top_score < box_threshold:
            return []

        # Dynamic floor is max of baseline and relative falloff from peak detection
        dynamic_floor = max(
            box_threshold,
            top_score * self.config.relative_dropoff_ratio,
        )

        accepted_indices: list[int] = []
        for idx, score in enumerate(clean_scores):
            if score >= dynamic_floor:
                accepted_indices.append(idx)

        return accepted_indices
