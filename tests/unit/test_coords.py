"""Unit tests for coordinate transformation, degenerate boxes, and boundary geometry."""

import math
import pytest
from stark_vision.geometry.coords import (
    box_area,
    box_center,
    clip_box,
    compute_iou,
    denormalize_box,
    normalize_box,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_xyxy_to_xywh_and_back():
    box = (100.0, 50.0, 300.0, 250.0)
    xywh = xyxy_to_xywh(box)
    assert xywh == (100.0, 50.0, 200.0, 200.0)

    reconstructed = xywh_to_xyxy(xywh)
    assert reconstructed == box


def test_normalize_and_denormalize():
    img_w, img_h = 1920, 1080
    box = (192.0, 108.0, 960.0, 540.0)

    norm = normalize_box(box, img_w, img_h)
    assert pytest.approx(norm[0], 0.001) == 0.1
    assert pytest.approx(norm[1], 0.001) == 0.1
    assert pytest.approx(norm[2], 0.001) == 0.5
    assert pytest.approx(norm[3], 0.001) == 0.5

    denorm = denormalize_box(norm, img_w, img_h)
    assert denorm == (192, 108, 960, 540)


def test_compute_iou():
    box1 = (0.0, 0.0, 100.0, 100.0)
    box2 = (50.0, 0.0, 150.0, 100.0)
    iou = compute_iou(box1, box2)
    assert pytest.approx(iou, 0.001) == 5000 / 15000  # 1/3

    # Disjoint boxes
    box3 = (200.0, 200.0, 300.0, 300.0)
    assert compute_iou(box1, box3) == 0.0


def test_clip_box():
    out_of_bounds = (-20.0, -10.0, 2000.0, 1200.0)
    clipped = clip_box(out_of_bounds, 1920.0, 1080.0)
    assert clipped == (0.0, 0.0, 1920.0, 1080.0)


def test_degenerate_and_nan_boxes():
    # Degenerate inverted box (x2 < x1)
    inv_box = (200.0, 200.0, 100.0, 100.0)
    xywh = xyxy_to_xywh(inv_box)
    assert xywh[2] == 0.0  # width clamped to 0
    assert xywh[3] == 0.0  # height clamped to 0

    # NaN coordinates sanitized
    nan_box = (float("nan"), 0.0, 100.0, 100.0)
    xywh_nan = xyxy_to_xywh(nan_box)
    assert not math.isnan(xywh_nan[0])

    # Invalid image dimensions raise ValueError
    with pytest.raises(ValueError):
        normalize_box((0, 0, 10, 10), width=0, height=100)
