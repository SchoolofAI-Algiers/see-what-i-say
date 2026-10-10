"""Unit tests for SpatialResolver, crowd preservation, and spatial disambiguation."""

import pytest
from stark_vision.grounding.disambiguation import (
    SpatialCandidate,
    SpatialResolver,
    compute_iomin,
    compute_iou,
)


def test_iou_and_containment_deduplication():
    # Two overlapping boxes for the same target (IoU ~ 0.70)
    cand1 = SpatialCandidate(box=(100.0, 100.0, 200.0, 200.0), score=0.88, label="person")
    cand2 = SpatialCandidate(box=(105.0, 105.0, 205.0, 205.0), score=0.82, label="person")

    # Distinct person on the other side
    cand3 = SpatialCandidate(box=(500.0, 100.0, 600.0, 200.0), score=0.85, label="person")

    resolver = SpatialResolver()
    deduped = resolver.deduplicate_candidates([cand1, cand2, cand3], iou_threshold=0.55)

    # Should keep cand1 and cand3, dropping duplicate cand2
    assert len(deduped) == 2
    assert deduped[0].score == 0.88
    assert deduped[1].score == 0.85


def test_crowd_preservation_distinct_adjacent_people():
    # Two distinct adjacent people standing next to each other
    # Person A: x in [100, 220], center = 160
    # Person B: x in [190, 310], center = 250 (30px overlap out of 120px width -> IoU ~ 0.14)
    person_a = SpatialCandidate(box=(100.0, 50.0, 220.0, 400.0), score=0.91, label="person")
    person_b = SpatialCandidate(box=(190.0, 50.0, 310.0, 400.0), score=0.89, label="person")

    resolver = SpatialResolver()
    deduped = resolver.deduplicate_candidates([person_a, person_b])

    # Both adjacent people MUST be preserved (NOT merged by aggressive NMS)
    assert len(deduped) == 2


def test_spatial_direction_resolution():
    cand_left = SpatialCandidate(box=(50.0, 100.0, 150.0, 200.0), score=0.85, label="cup")
    cand_right = SpatialCandidate(box=(600.0, 100.0, 700.0, 200.0), score=0.84, label="cup")

    resolver = SpatialResolver()

    # Query explicitly asks for "the cup on the left"
    is_ambiguous, resolved, msg = resolver.evaluate_ambiguity(
        [cand_left, cand_right], text_prompt="the cup on the left"
    )

    # Should automatically disambiguate without requiring clarification
    assert not is_ambiguous
    assert len(resolved) == 1
    assert resolved[0].box == cand_left.box


def test_center_and_nearest_cues():
    cand_center = SpatialCandidate(box=(250.0, 200.0, 350.0, 300.0), score=0.85, label="box")
    cand_edge = SpatialCandidate(box=(10.0, 10.0, 110.0, 110.0), score=0.84, label="box")

    resolver = SpatialResolver()
    _, resolved_center, _ = resolver.evaluate_ambiguity(
        [cand_center, cand_edge], text_prompt="box in the center"
    )
    assert resolved_center[0].box == cand_center.box


def test_ambiguity_clarification_trigger():
    cand_left = SpatialCandidate(box=(50.0, 100.0, 150.0, 200.0), score=0.85, label="bottle")
    cand_right = SpatialCandidate(box=(600.0, 100.0, 700.0, 200.0), score=0.83, label="bottle")

    resolver = SpatialResolver()

    # Query without spatial modifier should flag ambiguity and generate clarification
    is_ambiguous, resolved, msg = resolver.evaluate_ambiguity(
        [cand_left, cand_right], text_prompt="bottle"
    )

    assert is_ambiguous
    assert len(resolved) == 2
    assert "Detected 2 matching instances" in msg


def test_edge_cases_empty_and_single():
    resolver = SpatialResolver()
    assert resolver.evaluate_ambiguity([], "test") == (False, [], None)

    single = SpatialCandidate(box=(0, 0, 10, 10), score=0.9, label="cat")
    assert resolver.evaluate_ambiguity([single], "test") == (False, [single], None)
