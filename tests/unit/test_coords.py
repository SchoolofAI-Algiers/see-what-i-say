from __future__ import annotations

import unittest
import numpy as np

from stark_vision.contracts import BoundingBox
from stark_vision.geometry.coords import (
    map_letterbox_box_to_frame,
    mask_area,
    mask_centroid,
    mask_to_bbox,
    normalized_to_pixel,
    pixel_to_normalized,
)
from stark_vision.mocks.adapters import make_scene_frames


class GeometryCoordsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = make_scene_frames(count=2, width=96, height=64)
        self.frame_w = self.frames[0].image.shape[1]  # 96
        self.frame_h = self.frames[0].image.shape[0]  # 64

    def test_normalized_and_pixel_conversions_roundtrip(self) -> None:
        norm_box = BoundingBox(0.1, 0.2, 0.5, 0.8)
        pix_box = normalized_to_pixel(norm_box, self.frame_w, self.frame_h)

        self.assertAlmostEqual(pix_box.x1, 9.6)
        self.assertAlmostEqual(pix_box.y1, 12.8)
        self.assertAlmostEqual(pix_box.x2, 48.0)
        self.assertAlmostEqual(pix_box.y2, 51.2)

        roundtrip = pixel_to_normalized(pix_box, self.frame_w, self.frame_h)
        self.assertAlmostEqual(roundtrip.x1, norm_box.x1)
        self.assertAlmostEqual(roundtrip.y1, norm_box.y1)
        self.assertAlmostEqual(roundtrip.x2, norm_box.x2)
        self.assertAlmostEqual(roundtrip.y2, norm_box.y2)

    def test_mask_area_centroid_and_bbox(self) -> None:
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.bool_)
        mask[10:30, 20:50] = True

        self.assertEqual(mask_area(mask), 600)

        centroid = mask_centroid(mask)
        self.assertIsNotNone(centroid)
        self.assertAlmostEqual(centroid[0], 34.5)
        self.assertAlmostEqual(centroid[1], 19.5)

        derived_box = mask_to_bbox(mask)
        self.assertEqual(derived_box, BoundingBox(20.0, 10.0, 50.0, 30.0))

    def test_empty_mask_edge_cases(self) -> None:
        empty_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.bool_)
        self.assertEqual(mask_area(empty_mask), 0)
        self.assertIsNone(mask_centroid(empty_mask))
        self.assertIsNone(mask_to_bbox(empty_mask))

    def test_letterbox_mapping(self) -> None:
        model_box = BoundingBox(0, 140, 640, 500)
        unpadded = map_letterbox_box_to_frame(
            model_box, frame_size=(1920, 1080), model_size=(640, 640)
        )

        self.assertAlmostEqual(unpadded.x1, 0.0)
        self.assertAlmostEqual(unpadded.y1, 0.0)
        self.assertAlmostEqual(unpadded.x2, 1920.0)
        self.assertAlmostEqual(unpadded.y2, 1080.0)


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
        

if __name__ == "__main__":
    unittest.main()
