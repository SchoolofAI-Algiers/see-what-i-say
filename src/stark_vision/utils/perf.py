"""Tiny profiling helpers for the performance overlay."""

from __future__ import annotations

import time
from collections import deque


class RollingMean:
    def __init__(self, size: int = 30) -> None:
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def count(self) -> int:
        return len(self._values)


class Timer:
    """with Timer() as t: ...; t.ms"""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
