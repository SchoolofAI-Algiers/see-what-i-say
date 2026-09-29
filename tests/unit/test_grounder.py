"""
Unit tests for STARK VISION Grounder engine.
"""

import numpy as np
import pytest

from stark_vision.grounding.grounder import Grounder, GroundingCandidate


@pytest.fixture(scope="module")
def grounder() -> Grounder:
    """Instantiate a shared Grounder engine on available device."""
    return Grounder(model_id="IDEA-Research/grounding-dino-tiny")


def test_grounding_candidate_geometry():
    """Verify geometry calculations on GroundingCandidate."""
    cand = GroundingCandidate(
        box=(100.0, 150.0, 300.0, 450.0),
        score=0.92,
        label="test object",
    )
    assert cand.x1 == 100.0
    assert cand.y1 == 150.0
    assert cand.x2 == 300.0
    assert cand.y2 == 450.0
    assert cand.width == 200.0
    assert cand.height == 300.0
    assert cand.center == (200.0, 300.0)
    assert cand.area == 60000.0


def test_normalize_prompt():
    """Verify text query period normalization."""
    assert Grounder._normalize_prompt("person wearing red") == "person wearing red ."
    assert Grounder._normalize_prompt("person wearing red .") == "person wearing red ."
    assert Grounder._normalize_prompt("   ") == "."


def test_find_candidates_synthetic_frame(grounder: Grounder):
    """Verify inference on a synthetic image with a prominent colored target."""
    # Create black canvas 480x640 with a red rectangle in the center
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # OpenCV uses BGR: red is (0, 0, 255)
    frame[100:300, 150:350] = (0, 0, 255)

    candidates = grounder.find_candidates(
        frame_bgr=frame,
        text_prompt="a red square",
        box_threshold=0.25,
        text_threshold=0.20,
    )

    assert len(candidates) >= 1
    top = candidates[0]
    assert top.score > 0.60
    # Coordinates should closely envelop the box [150, 100, 350, 300]
    assert 130 <= top.x1 <= 170
    assert 80 <= top.y1 <= 120
    assert 330 <= top.x2 <= 370
    assert 280 <= top.y2 <= 320


def test_empty_frame_raises_error(grounder: Grounder):
    """Verify error handling on invalid frames."""
    with pytest.raises(ValueError):
        grounder.find_candidates(np.array([]), "person")


def test_grounder_find_contract(grounder: Grounder):
    """Verify that grounder.find() works as an exact contract alias."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 150:350] = (0, 0, 255)
    candidates = grounder.find(frame, "red square")
    assert isinstance(candidates, list)


def test_is_ambiguous_helper():
    """Verify is_ambiguous correctly flags close scores and clears distant scores."""
    from stark_vision.grounding import is_ambiguous

    # Case 1: single candidate -> never ambiguous
    cand1 = GroundingCandidate(box=(0, 0, 10, 10), score=0.90, label="target 1")
    assert not is_ambiguous([cand1])

    # Case 2: two candidates with close scores (gap = 0.03 < 0.15) -> ambiguous!
    cand2 = GroundingCandidate(box=(20, 20, 40, 40), score=0.87, label="target 2")
    assert is_ambiguous([cand1, cand2], delta_threshold=0.15)

    # Case 3: two candidates with large gap (gap = 0.40 >= 0.15) -> not ambiguous!
    cand3 = GroundingCandidate(box=(20, 20, 40, 40), score=0.50, label="target 3")
    assert not is_ambiguous([cand1, cand3], delta_threshold=0.15)

