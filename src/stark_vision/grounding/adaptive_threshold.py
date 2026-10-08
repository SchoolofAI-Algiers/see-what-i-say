"""
Dynamic threshold calibration for visual grounding queries.

Grounding DINO confidence drops on longer multi-token phrases compared to
simple single words. We scale the base threshhold depending on query length
and strip low-scoring clutter relative to the peak match.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThresholdConfig:
    # bounds and ratios for cutoff calibration
    min_box_threshold: float = 0.15
    max_box_threshold: float = 0.65
    terse_base_threshold: float = 0.28   # baseline for short queries (table, chair)
    relative_dropoff_ratio: float = 0.55  # drop candidates scoring under 55% of best
    text_threshold_ratio: float = 0.75


class DynamicThresholdCalculator:
    # common prepositions and descriptive tokens that spread attention
    MODIFIERS = frozenset({
        "in", "on", "with", "wearing", "holding", "near", "next", "to",
        "behind", "under", "front", "black", "white", "red", "blue",
        "green", "yellow", "dark", "light", "small", "large", "tall",
        "left", "right", "standing", "sitting", "walking"
    })

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()

    def estimate_query_complexity(self, text_prompt: str) -> float:
        # rough heuristic: longer phrases with modifiers yield lower peak logits
        if not text_prompt or not text_prompt.strip():
            return 0.0

        words = re.findall(r"\b\w+\b", text_prompt.lower())
        word_count = len(words)
        if word_count <= 1:
            return 0.0

        modifier_count = sum(1 for w in words if w in self.MODIFIERS)
        complexity = min(1.0, (max(0, word_count - 1) * 0.12) + (modifier_count * 0.10))
        return round(complexity, 3)

    def compute_baseline_thresholds(
        self, text_prompt: str
    ) -> tuple[float, float]:
        # returns (box_thresh, text_thresh) adapted for the prompt length
        if not text_prompt or not text_prompt.strip():
            return self.config.terse_base_threshold, round(self.config.terse_base_threshold * self.config.text_threshold_ratio, 3)

        complexity = self.estimate_query_complexity(text_prompt)

        # short queries anchor around 0.28; longer descriptive prompts scale down toward ~0.18
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
        # keeps detections within a fraction of top score and rejects background clutter
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
