"""Unit tests for BoxTemporalFilter and OneEuroFilter in Centroid-Log-Scale space."""

import pytest
from stark_vision.grounding.temporal_filter import BoxTemporalFilter, OneEuroFilter


def test_one_euro_filter_smoothing():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)

    t0 = 1000.0
    # Feed stationary noisy signal around 100.0
    vals = [100.0, 102.0, 98.0, 101.5, 99.0]
    filtered_vals = []
    for i, v in enumerate(vals):
        filtered_vals.append(f.filter(v, timestamp=t0 + i * 0.033))

    # The filter must dampen the noise
    assert abs(filtered_vals[-1] - 100.0) < abs(vals[-1] - 100.0) + 1.0


def test_box_temporal_filter_preserves_dimensions():
    box_filter = BoxTemporalFilter()
    box_in = (100.0, 150.0, 300.0, 450.0)

    box_out = box_filter.filter_box(box_in, timestamp=1.0)
    assert len(box_out) == 4
    # First frame output matches input closely
    assert pytest.approx(box_out[0], 0.1) == box_in[0]
    assert pytest.approx(box_out[1], 0.1) == box_in[1]
    assert pytest.approx(box_out[2], 0.1) == box_in[2]
    assert pytest.approx(box_out[3], 0.1) == box_in[3]


def test_aspect_ratio_preservation_under_zoom():
    # Box initially 100x200 (aspect ratio 0.5)
    box_filter = BoxTemporalFilter()
    box_filter.filter_box((100.0, 100.0, 200.0, 300.0), timestamp=1.0)

    # Simulated zoom: box scales up smoothly to 200x400 (aspect ratio still 0.5)
    zoomed_box = box_filter.filter_box((50.0, 50.0, 250.0, 450.0), timestamp=1.033)
    w = zoomed_box[2] - zoomed_box[0]
    h = zoomed_box[3] - zoomed_box[1]
    aspect_ratio = w / h
    # Aspect ratio should stay close to 0.5 (no squishing/stretching from independent edge filters)
    assert pytest.approx(aspect_ratio, 0.05) == 0.5


def test_scene_cut_reset():
    box_filter = BoxTemporalFilter()
    # Tracking target on the left
    box_filter.filter_box((10.0, 10.0, 50.0, 50.0), timestamp=1.0)
    box_filter.filter_box((12.0, 11.0, 52.0, 51.0), timestamp=1.033)

    # Scene cut / sudden jump to completely different target on far right
    new_target = (800.0, 600.0, 850.0, 650.0)
    jumped = box_filter.filter_box(new_target, timestamp=1.066)

    # Filter must detect discontinuity and reset immediately, NOT blend the boxes across screen
    assert pytest.approx(jumped[0], 1.0) == new_target[0]
    assert pytest.approx(jumped[2], 1.0) == new_target[2]
