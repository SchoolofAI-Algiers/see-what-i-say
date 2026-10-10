import numpy as np

from stark_vision.contracts import EffectMode, EffectSpec
from stark_vision.effects.engine import EffectEngine

def _scene(h, w):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), bool)
    mask[h // 4: 3 * h // 4, w // 3: 2 * w // 3] = True
    return frame, mask

def test_blur_is_visible_on_1080p_and_stays_inside_mask():
    frame, mask = _scene(1080, 1920)
    out = EffectEngine().apply(frame, mask, EffectSpec(EffectMode.BLUR))
    assert np.array_equal(out[~mask], frame[~mask])           # outside untouched
    inside_before = frame[mask].astype(float).std()
    inside_after = out[mask].astype(float).std()
    assert inside_after < 0.5 * inside_before                 # noise strongly smoothed


def test_highlight_outline_thicker_on_large_frames():
    f_small, m_small = _scene(360, 640)
    f_big, m_big = _scene(1080, 1920)
    eng = EffectEngine()
    cyan = np.array([0, 245, 255])
    small = (eng.apply(f_small, m_small, EffectSpec(EffectMode.HIGHLIGHT)) == cyan).all(axis=2).sum()
    big = (eng.apply(f_big, m_big, EffectSpec(EffectMode.HIGHLIGHT)) == cyan).all(axis=2).sum()
    assert big / (1080 * 1920) > small / (360 * 640) * 0.9
