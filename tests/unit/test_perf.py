from stark_vision.utils.perf import RollingMean, Timer


def test_rolling_mean_window():
    r = RollingMean(3)
    for v in (1, 2, 3, 4):
        r.add(v)
    assert r.mean == 3.0 and r.count == 3


def test_timer_measures():
    with Timer() as t:
        sum(range(1000))
    assert t.ms >= 0.0
