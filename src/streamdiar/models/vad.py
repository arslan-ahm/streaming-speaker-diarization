"""Causal energy voice-activity detection.

Diarization papers routinely report DER "with oracle VAD", which hands the system
the reference speech/non-speech segmentation. That isolates the clustering
question, and this repository reports it as a secondary column for exactly that
reason — but it cannot be the headline, because an oracle VAD is future
information: knowing that frame ``t`` is speech generally requires having heard
frame ``t+1``. A system claiming bounded latency cannot use it.

So the primary numbers use this detector, which is deliberately simple and
strictly causal:

1. Frame energy is the mean of the standardised feature vector. In the
   generator's units, non-speech sits near ``log(background) = -2`` and speech
   near ``log(1 + background) = 0.13``, so a single threshold does most of the
   work. The threshold itself is **fitted on the dev split** (see
   :func:`fit_vad_threshold`) — not on test, and not analytically from the
   generator's constants, which would be leaking knowledge of the data-generating
   process into the model.
2. A **hangover** keeps the state in speech for a few frames after energy drops,
   which is causal (it only ever extends a decision forward in time) and removes
   the frame-level flicker inside a turn that would otherwise fragment windows.
3. A **minimum onset run** requires several consecutive above-threshold frames
   before declaring speech, suppressing single-frame noise spikes. This costs
   latency: an onset is detected ``onset_frames`` late, and that delay is real and
   included in the reported emission latency because it delays the frame labels,
   not merely the decision.

A hangover-and-onset state machine is the standard cheap VAD (see e.g. the ITU-T
G.729 Annex B and ETSI AMR VAD designs); nothing here is novel and it is not
meant to be. It is meant to be honest about what a streaming system can know.
"""

from __future__ import annotations

import numpy as np


def frame_energy(features: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``(T,)`` mean standardised energy per frame.

    Standardising by the *training* statistics (not the recording's own) is what
    keeps this causal; see ``models/embedder.py`` for the same argument.
    """
    x = np.asarray(features, dtype=np.float64)
    denom = np.maximum(np.asarray(sd, dtype=np.float64), 1e-6)
    z = (x - np.asarray(mean, dtype=np.float64)) / denom
    return z.mean(axis=1)


def energy_vad(
    energy: np.ndarray,
    threshold: float,
    hangover_frames: int = 12,
    onset_frames: int = 3,
) -> np.ndarray:
    """Causal speech/non-speech decision from a frame-energy contour.

    Args:
        energy: ``(T,)`` frame energies, e.g. from :func:`frame_energy`.
        threshold: Above this counts as active.
        hangover_frames: Frames of speech held after energy falls below threshold.
        onset_frames: Consecutive active frames required to enter speech.

    Returns:
        ``(T,)`` bool. Frame ``t``'s value depends only on ``energy[:t+1]``, which
        ``tests/test_causality.py::test_vad_is_causal`` asserts by perturbation.
    """
    e = np.asarray(energy, dtype=np.float64)
    n = e.shape[0]
    out = np.zeros(n, dtype=bool)
    above = e > float(threshold)
    run = 0
    hang = 0
    in_speech = False
    for t in range(n):
        if above[t]:
            run += 1
            hang = int(hangover_frames)
            if run >= max(1, int(onset_frames)):
                in_speech = True
        else:
            run = 0
            if in_speech:
                hang -= 1
                if hang <= 0:
                    in_speech = False
        out[t] = in_speech
    return out


def fit_vad_threshold(
    energies: list[np.ndarray],
    references: list[np.ndarray],
    hangover_frames: int = 12,
    onset_frames: int = 3,
    n_grid: int = 61,
) -> tuple[float, float]:
    """Grid-search the threshold that minimises frame-level speech/non-speech error.

    Args:
        energies: Per-recording frame energies from the **dev** split.
        references: Per-recording bool speech masks (``reference_matrix().any(1)``).
        hangover_frames: Passed through to :func:`energy_vad`.
        onset_frames: Passed through to :func:`energy_vad`.
        n_grid: Candidate thresholds, spanning the observed energy range.

    Returns:
        ``(threshold, frame_error_rate)`` at the optimum.

    One scalar fitted on held-out data is the smallest amount of supervision that
    makes the VAD honest. Fitting it on test would be cheating; deriving it from
    ``BACKGROUND_LOG_POWER`` would be worse, because it would not transfer to the
    real-data path at all.
    """
    if not energies:
        return 0.0, float("nan")
    pool = np.concatenate([np.asarray(e, dtype=np.float64) for e in energies])
    grid = np.linspace(float(np.percentile(pool, 2)), float(np.percentile(pool, 98)), n_grid)
    best_t, best_err = float(grid[0]), np.inf
    for t in grid:
        wrong = 0
        total = 0
        for e, ref in zip(energies, references, strict=True):
            pred = energy_vad(e, float(t), hangover_frames, onset_frames)
            wrong += int((pred != np.asarray(ref, dtype=bool)).sum())
            total += int(pred.size)
        err = wrong / max(total, 1)
        if err < best_err:
            best_t, best_err = float(t), float(err)
    return best_t, best_err


def windows_are_speech(
    speech: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    min_fraction: float = 0.5,
) -> np.ndarray:
    """Whether each window's *responsibility region* is mostly speech.

    The decision uses the region ``[region_start, region_end)`` — the slice of the
    recording this window is responsible for labelling — rather than the whole
    embedding window. Using the whole window would mark a window as speech
    because of audio a previous window already accounted for, which produces a
    false-alarm tail of ``window - hop`` frames after every turn.
    """
    sp = np.asarray(speech, dtype=bool)
    out = np.zeros(len(region_starts), dtype=bool)
    for k, (s, e) in enumerate(zip(region_starts, region_ends, strict=True)):
        if e > s:
            out[k] = sp[int(s) : int(e)].mean() >= min_fraction
    return out
