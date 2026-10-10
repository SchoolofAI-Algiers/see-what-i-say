"""
Contract compliance and edge case tests verifying full compatibility with
Rayan's subsystem contracts (Frame, BoundingBox, GroundingCandidate, HudRenderer).
"""

from __future__ import annotations

import numpy as np
import pytest

from stark_vision.contracts import (
    BoundingBox,
    EffectMode,
    EffectSpec,
    Frame,
    GroundingCandidate,
    RenderedFrame,
    RuntimeSnapshot,
    TargetSnapshot,
    TrackingStatus,
)
from stark_vision.geometry.coords import (
    normalize_box,
    normalized_to_pixel,
    pixel_to_normalized,
)
from stark_vision.grounding.grounder import Grounder, is_ambiguous
from stark_vision.hud.overlay import HudRenderer


@pytest.fixture(scope="module")
def grounder() -> Grounder:
    return Grounder(model_id="IDEA-Research/grounding-dino-tiny")


def test_bounding_box_contract_and_sequence_compatibility():
    box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)

    # Contract fields and properties
    assert box.x1 == 10.0
    assert box.y1 == 20.0
    assert box.x2 == 110.0
    assert box.y2 == 220.0
    assert box.center == (60.0, 120.0)
    assert box.center[0] == 60.0
    assert box.xyxy == (10.0, 20.0, 110.0, 220.0)

    # Sequence indexing and unpacking
    assert box[0] == 10.0
    assert box[1] == 20.0
    assert box[2] == 110.0
    assert box[3] == 220.0
    assert len(box) == 4

    x1, y1, x2, y2 = box
    assert (x1, y1, x2, y2) == (10.0, 20.0, 110.0, 220.0)


def test_bounding_box_validation_edge_cases():
    # Inverted or zero width/height must fail
    with pytest.raises(ValueError, match="positive width and height"):
        BoundingBox(x1=100.0, y1=20.0, x2=50.0, y2=220.0)

    with pytest.raises(ValueError, match="positive width and height"):
        BoundingBox(x1=10.0, y1=200.0, x2=110.0, y2=100.0)

    with pytest.raises(ValueError, match="positive width and height"):
        BoundingBox(x1=10.0, y1=20.0, x2=10.0, y2=20.0)


def test_grounding_candidate_interop():
    # Instantiation with BoundingBox
    b = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
    cand1 = GroundingCandidate(box=b, score=0.95, label="laptop")
    assert isinstance(cand1.box, BoundingBox)
    assert cand1.box.center[0] == 60.0
    assert cand1.x1 == 10.0
    assert cand1.width == 100.0
    assert cand1.height == 200.0
    assert cand1.area == 20000.0

    # Instantiation with tuple (auto-converted to BoundingBox)
    cand2 = GroundingCandidate(box=(50.0, 60.0, 150.0, 260.0), score=0.82, label="mug")
    assert isinstance(cand2.box, BoundingBox)
    assert cand2.box.center == (100.0, 160.0)
    assert cand2.box.center[0] == 100.0
    assert cand2.xyxy == (50.0, 60.0, 150.0, 260.0)

    # Score bounds validation
    with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
        GroundingCandidate(box=b, score=1.5, label="invalid")

    with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
        GroundingCandidate(box=b, score=-0.1, label="invalid")


def test_grounder_find_with_frame_contract(grounder: Grounder):
    # Construct an RGB frame matching Frame contract (RGB uint8)
    image_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # Paint a blue rectangle in RGB: (0, 120, 255)
    image_rgb[120:280, 160:360] = (0, 120, 255)

    frame = Frame(id=42, timestamp_ns=1_000_000_000, image=image_rgb)
    candidates = grounder.find(frame, "blue box")

    assert isinstance(candidates, list)
    if candidates:
        top = candidates[0]
        assert isinstance(top.box, BoundingBox)
        # Verify access pattern used in Rayan's TargetManager
        cx = top.box.center[0]
        assert isinstance(cx, float)
        assert 0.0 <= top.score <= 1.0


def test_grounder_find_edge_cases(grounder: Grounder):
    frame_arr = np.zeros((240, 320, 3), dtype=np.uint8)

    # Empty prompt returns empty candidate list
    assert grounder.find(frame_arr, "") == []
    assert grounder.find(frame_arr, "   ") == []

    # Invalid frame types raise ValueError
    with pytest.raises(ValueError):
        grounder.find(None, "person")

    with pytest.raises(ValueError):
        grounder.find(np.zeros((0, 0, 3), dtype=np.uint8), "person")

    with pytest.raises(ValueError):
        grounder.find("not_a_frame", "person")


def test_coords_interop_with_bounding_box():
    bbox = BoundingBox(x1=64.0, y1=48.0, x2=320.0, y2=240.0)

    # normalize_box supports BoundingBox via sequence protocol
    norm = normalize_box(bbox, width=640, height=480)
    assert norm == (0.1, 0.1, 0.5, 0.5)

    # pixel_to_normalized returns a valid BoundingBox in normalized coordinates
    norm_box = pixel_to_normalized(bbox, width=640, height=480)
    assert isinstance(norm_box, BoundingBox)
    assert norm_box.x1 == pytest.approx(0.1)
    assert norm_box.y1 == pytest.approx(0.1)
    assert norm_box.x2 == pytest.approx(0.5)
    assert norm_box.y2 == pytest.approx(0.5)

    # normalized_to_pixel converts back
    pixel_box = normalized_to_pixel(norm_box, width=640, height=480)
    assert isinstance(pixel_box, BoundingBox)
    assert pixel_box.x1 == pytest.approx(64.0)
    assert pixel_box.y1 == pytest.approx(48.0)
    assert pixel_box.x2 == pytest.approx(320.0)
    assert pixel_box.y2 == pytest.approx(240.0)


def test_hud_renderer_contract():
    renderer = HudRenderer()
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    target_box = BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0)
    active_target = TargetSnapshot(
        target_id=1,
        prompt="coffee cup",
        box=target_box,
        mask=None,
        confidence=0.88,
        status=TrackingStatus.TRACKING,
        frame_id=10,
    )
    snapshot = RuntimeSnapshot(
        frame_id=10,
        active_target=active_target,
        pending_prompt=None,
        pending_candidates=(),
        pending_is_ambiguous=False,
        effect=EffectSpec(mode=EffectMode.HIGHLIGHT),
        message="Tracking active",
    )

    rendered = renderer.render(frame_rgb, snapshot)
    assert isinstance(rendered, RenderedFrame)
    assert rendered.image.shape == (480, 640, 3)
    assert any("coffee cup" in line for line in rendered.telemetry)

    # Invalid input shapes/dtypes must raise ValueError
    with pytest.raises(ValueError, match="HUD input must be RGB uint8"):
        renderer.render(np.zeros((480, 640), dtype=np.uint8), snapshot)
