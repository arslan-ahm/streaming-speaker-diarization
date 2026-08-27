"""Wall-clock and memory measurement.

The build standard is explicit that *params and MACs are inputs; wall-clock is
the claim*, and that a sibling project once reported a tiny model as 5x slower
than reality because it warmed up for three iterations instead of eight. Both
lessons are baked into the defaults here: ``warmup=8``, ``repeats=25``, median
and IQR reported rather than a mean that one scheduling hiccup can dominate.

Peak memory uses ``tracemalloc``, which counts Python-side allocations only. It
does not see the torch allocator's arena or NumPy buffers allocated inside C
extensions, so it *undercounts* absolute RSS. It is used here only for
*relative* comparison between the streaming and offline pipelines, which
allocate through the same paths, and that limitation is restated in
``docs/RESULTS.md`` rather than being papered over.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class Timing:
    """Latency statistics in milliseconds."""

    median_ms: float
    iqr_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    repeats: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_times(samples_s: list[float] | np.ndarray) -> Timing:
    """Turn a vector of durations (seconds) into a :class:`Timing`."""
    a = np.asarray(list(samples_s), dtype=float) * 1000.0
    if a.size == 0:
        nan = float("nan")
        return Timing(nan, nan, nan, nan, nan, nan, nan, 0)
    q1, q3 = np.percentile(a, [25, 75])
    return Timing(
        median_ms=float(np.median(a)),
        iqr_ms=float(q3 - q1),
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        mean_ms=float(a.mean()),
        min_ms=float(a.min()),
        repeats=int(a.size),
    )


def measure_latency(
    fn: Callable[[], Any], warmup: int = 8, repeats: int = 25
) -> Timing:
    """Time ``fn`` after an adequate warm-up.

    ``warmup=8`` is not decoration. Torch selects and caches a convolution
    algorithm per input shape on first use; with three warm-up calls the first
    timed call still pays that cost and can dominate a 25-sample median for a
    small model.
    """
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return summarize_times(samples)


def measure_peak_memory(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, peak_python_heap_MiB)``.

    See the module docstring: this is a Python-heap measurement, useful for
    comparing two pipelines in the same process, not an RSS figure.
    """
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already:
            tracemalloc.stop()
    return result, peak / (1024.0 * 1024.0)


def count_parameters(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def real_time_factor(processing_s: float, audio_s: float) -> float:
    """RTF = compute time / audio duration. Below 1.0 is faster than real time."""
    if audio_s <= 0:
        return float("nan")
    return float(processing_s / audio_s)
