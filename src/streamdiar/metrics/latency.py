"""Emission latency: the axis this project actually competes on.

Throughput is the wrong measurement for a streaming diarizer and reporting it
instead of latency is the most common way the online/offline comparison gets
fudged. An offline system can have an excellent real-time factor — it processes
an hour of audio in ninety seconds — and still be unusable for live captioning,
because it emits its first label only after the last frame has arrived.

So there are two distinct quantities here and both are reported:

**Emission delay** (:func:`emission_latency`). For a single turn decision: the
wall-clock-equivalent time between the moment the audio a decision covers has
finished arriving, and the moment the label for it is emitted. This is *algorithmic*
delay, measured in audio time, not compute time — it is a property of the
algorithm's dependency structure, so it is exactly reproducible and does not
depend on how loaded the machine is. For the offline reference it is
``recording_end - window_end``, which grows without bound in recording length.
That unboundedness is the point.

**Real-time factor** (:func:`real_time_factor` in ``utils/bench.py``). Compute
time over audio duration. Both systems are far below 1.0 here; RTF is reported
so the reader can confirm neither is compute-bound, which is what makes the
emission-delay comparison a statement about algorithm design rather than about
this laptop.

A third quantity, **per-decision compute time**, is measured in
``pipelines/efficiency.py`` with the standard's warm-up floor. Total emission
latency in a deployed system is the sum of the algorithmic delay and the compute
time; the tables report both terms rather than a single conflated number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LatencyStats:
    """Emission-delay distribution over one recording's turn decisions, in ms."""

    median_ms: float
    mean_ms: float
    p95_ms: float
    max_ms: float
    iqr_ms: float
    n_decisions: int
    #: Delay of the very first emitted label — the "time to first caption".
    first_emission_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def emission_latency(
    window_end_frames: np.ndarray,
    emission_frames: np.ndarray,
    frame_rate: int,
) -> LatencyStats:
    """Summarise per-decision emission delay.

    Args:
        window_end_frames: For each decision, the frame index at which the audio
            it covers finished arriving.
        emission_frames: For each decision, the frame index at which its label
            became available to a consumer.
        frame_rate: Frames per second, to convert to milliseconds.

    Returns:
        A :class:`LatencyStats`. All-NaN with ``n_decisions=0`` for an empty
        input — an empty recording does not have a latency of zero.

    Raises:
        ValueError: If any emission precedes its own window's end, which would
            mean the system used information it did not have.
    """
    we = np.asarray(window_end_frames, dtype=np.float64)
    em = np.asarray(emission_frames, dtype=np.float64)
    if we.shape != em.shape:
        raise ValueError(f"shape mismatch: {we.shape} vs {em.shape}")
    if we.size == 0:
        nan = float("nan")
        return LatencyStats(nan, nan, nan, nan, nan, 0, nan)

    delay_frames = em - we
    if (delay_frames < -1e-9).any():
        worst = float(delay_frames.min())
        raise ValueError(
            f"negative emission delay ({worst:.3f} frames): a decision was emitted "
            "before its own audio arrived, which is a causality violation"
        )
    ms = delay_frames * 1000.0 / frame_rate
    q1, q3 = np.percentile(ms, [25, 75])
    return LatencyStats(
        median_ms=float(np.median(ms)),
        mean_ms=float(ms.mean()),
        p95_ms=float(np.percentile(ms, 95)),
        max_ms=float(ms.max()),
        iqr_ms=float(q3 - q1),
        n_decisions=int(ms.size),
        first_emission_ms=float(ms[int(np.argmin(we))]),
    )


def speaker_count_accuracy(true_counts: np.ndarray, pred_counts: np.ndarray) -> dict[str, float]:
    """Exact-match accuracy and error statistics for the estimated speaker count.

    Reported separately from DER because they fail in different ways: a system
    that finds 3 speakers instead of 4 has a bounded DER cost (one speaker's
    time), while a system that finds 12 has a large one. ``bias`` is signed on
    purpose — over- and under-clustering call for opposite fixes, and a mean
    absolute error hides which one is happening.
    """
    t = np.asarray(true_counts, dtype=np.float64)
    p = np.asarray(pred_counts, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        nan = float("nan")
        return {"accuracy": nan, "mae": nan, "bias": nan, "within_one": nan, "n": 0.0}
    err = p - t
    return {
        "accuracy": float((err == 0).mean()),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "within_one": float((np.abs(err) <= 1).mean()),
        "n": float(t.size),
    }
