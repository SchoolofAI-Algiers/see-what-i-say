"""Unit tests for Dynamic Adaptive Thresholding (DAT) engine including edge cases."""

import pytest
from stark_vision.grounding.adaptive_threshold import (
    DynamicThresholdCalculator,
    ThresholdConfig,
)


def test_query_complexity_scaling():
    calc = DynamicThresholdCalculator()

    # Simple one-word query
    c_simple = calc.estimate_query_complexity("mug")
    assert c_simple == 0.0

    # Complex multi-token query with attributes and spatial prepositions
    c_complex = calc.estimate_query_complexity("the large red mug on the table near the laptop")
    assert c_complex > 0.5


def test_dynamic_threshold_calculation_adapts_to_prompt():
    calc = DynamicThresholdCalculator()

    box_simple, text_simple = calc.compute_baseline_thresholds("cup")
    box_complex, text_complex = calc.compute_baseline_thresholds(
        "the guy wearing a black hoodie and blue jeans holding a cup"
    )

    # Complex queries must lower the baseline threshold to prevent false negatives
    assert box_complex < box_simple
    # Ensure thresholds obey safety bounds
    assert box_complex >= 0.15
    assert box_simple <= 0.65


def test_relative_falloff_filter():
    calc = DynamicThresholdCalculator(ThresholdConfig(relative_dropoff_ratio=0.60))

    # Top score is 0.90, floor = max(0.20, 0.90 * 0.60) = 0.54
    # Candidates with score < 0.54 should be discarded even if > 0.20
    scores = [0.90, 0.85, 0.60, 0.40, 0.25]
    accepted = calc.filter_by_relative_falloff(scores, box_threshold=0.20)

    # 0.90, 0.85, 0.60 are >= 0.54
    assert accepted == [0, 1, 2]


def test_edge_cases_empty_and_absent_referents():
    calc = DynamicThresholdCalculator()

    # Edge Case 1: Empty and whitespace prompts
    b1, t1 = calc.compute_baseline_thresholds("")
    b2, t2 = calc.compute_baseline_thresholds("   ")
    assert b1 == b2 == calc.config.terse_base_threshold

    # Edge Case 2: Absent referent (all detection scores below threshold)
    scores = [0.18, 0.12, 0.05]
    accepted = calc.filter_by_relative_falloff(scores, box_threshold=0.25)
    assert accepted == []  # Rejects all false alarms

    # Edge Case 3: Empty candidates list
    assert calc.filter_by_relative_falloff([], box_threshold=0.25) == []

    # Edge Case 4: Non-descending scores (safely handles unordered input)
    unordered = [0.30, 0.80, 0.60]
    accepted_un = calc.filter_by_relative_falloff(unordered, box_threshold=0.20)
    # Top is 0.80 -> floor = max(0.20, 0.80 * 0.60) = 0.48 -> keeps 0.80 and 0.60
    assert 1 in accepted_un
    assert 2 in accepted_un
    assert 0 not in accepted_un
