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


if __name__ == "__main__":
    unittest.main()