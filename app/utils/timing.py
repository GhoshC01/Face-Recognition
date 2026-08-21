from __future__ import annotations

import time


class Stopwatch:
    """Elapsed-time helper for structured latency logging. `lap_ms()` returns
    milliseconds since the last `lap_ms()`/`reset()` call (or construction),
    so a caller can time a sequence of stages without juggling timestamps."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def lap_ms(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1000
        self._start = now
        return round(elapsed, 2)
