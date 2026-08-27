"""The inference loop: features -> VAD -> embeddings -> a diarization method -> metrics.

One function, :func:`run_diarizer`, dispatches all four mechanisms so that every
comparison in the repository shares the same feature extraction, the same VAD,
and the same scoring code. If the offline baseline had its own inference path it
would be impossible to tell whether a DER gap came from the clustering mechanism
or from an accidental difference in windowing, and that is the most common way an
"our method wins" table turns out to be wrong.

Scoring conventions, all of them consequential
----------------------------------------------
* **Primary DER** uses ``collar = 0`` and scores overlapped regions. That is
  stricter than the NIST convention (250 ms collar, overlap often excluded) and
  it is the primary number because the generated reference has no annotation slop
  to forgive. The two flattering variants are computed too and reported as
  clearly-labelled secondary columns, never as the headline.
* **Speech/non-speech** comes from the causal VAD by default. ``oracle_vad=True``
  substitutes the reference, which is future information and therefore not a
  valid streaming configuration — it exists only to separate "the VAD is wrong"
  from "the clustering is wrong".
* **Overlap is an irreducible floor.** Every mechanism here emits at most one
  speaker per window, so a frame with two simultaneous reference speakers costs
  one miss no matter what. The floor is measured and reported
  (``overlap_miss_floor``) so the reader can subtract it; it is identical for all
  methods, so it cancels in every comparison.
* **Turn-decision correctness** for calibration is defined through the DER
  mapping, on windows whose region contains exactly one reference speaker.
  Windows with none (non-speech) or several (overlap) are excluded and counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..data.generator import Recording
from ..metrics.der import DERResult, der, labels_to_matrix
from ..metrics.latency import emission_latency
from ..models.offline import diarize_offline
from ..models.online import DiarizationOutput, diarize_naive_online, diarize_online
from ..models.vad import energy_vad, frame_energy, windows_are_speech
from .train import TrainedModel

#: Methods dispatched by :func:`run_diarizer`.
METHODS = ("online", "naive_online", "offline_ahc", "offline_spectral")


@dataclass
class RecordingResult:
    """Everything measured for one (recording, method) pair."""

    recording: str
    method: str
    latency_budget_ms: float
    metrics: dict[str, float]
    output: DiarizationOutput
    der_result: DERResult
    #: Per-turn-decision ``(confidence, margin, correct)`` for the calibration study.
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    margin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    correct: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    n_excluded_decisions: int = 0


def fit_vad(model: TrainedModel, cfg: Config, dev_recordings: list[Recording]) -> TrainedModel:
    """Fit the VAD threshold on the dev split and store it on the model.

    Mutates and returns ``model``. Called once after training; every evaluation
    then reuses the same scalar, so the VAD is not re-tuned per experiment (which
    would make the ablations incomparable).
    """
    from ..models.vad import fit_vad_threshold

    energies = [frame_energy(r.features, model.feature_mean, model.feature_sd)
                for r in dev_recordings]
    refs = [r.reference_matrix().any(axis=1) for r in dev_recordings]
    threshold, err = fit_vad_threshold(
        energies, refs, cfg.diarizer.vad_hangover_frames, cfg.diarizer.vad_onset_frames
    )
    model.vad_threshold = float(threshold)
    model.vad_frame_error = float(err)
    return model


def run_diarizer(
    model: TrainedModel,
    cfg: Config,
    recording: Recording,
    method: str = "online",
    latency_budget_ms: float | None = None,
    oracle_vad: bool = False,
    oracle_count: bool = False,
) -> tuple[DiarizationOutput, dict[str, float]]:
    """Diarize one recording. Returns ``(output, timing in seconds)``.

    Args:
        model: Trained embedder with its fitted VAD threshold.
        cfg: Full config; ``cfg.diarizer`` supplies the mechanism switches.
        recording: The recording to process.
        method: One of :data:`METHODS`.
        latency_budget_ms: Overrides ``cfg.diarizer.latency_budget_ms``. This is
            the swept axis, so it is a parameter rather than only a config field.
        oracle_vad: Use the reference speech mask instead of the causal VAD.
        oracle_count: Supply the true speaker count to the clustering.

    The timing dict separates ``embed_s`` from ``cluster_s`` because they scale
    differently: embedding is linear in recording length for every method, while
    the offline clustering step is the quadratic-to-cubic one. Reporting only a
    total would hide exactly the term the efficiency claim is about.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    features = recording.features
    n_frames = int(features.shape[0])
    win = cfg.frames_per_window
    hop = cfg.frames_per_hop

    t0 = time.perf_counter()
    embeddings, starts, ends = model.embedder.embed_recording(features, win, hop)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    region_starts = np.maximum(0, ends - hop)
    if oracle_vad:
        speech_frames = recording.reference_matrix().any(axis=1)
    else:
        energy = frame_energy(features, model.feature_mean, model.feature_sd)
        speech_frames = energy_vad(
            energy, model.vad_threshold,
            cfg.diarizer.vad_hangover_frames, cfg.diarizer.vad_onset_frames,
        )
    is_speech = windows_are_speech(speech_frames, region_starts, ends)
    vad_s = time.perf_counter() - t0

    budget_ms = cfg.diarizer.latency_budget_ms if latency_budget_ms is None else latency_budget_ms
    budget_windows = int(float(budget_ms) // cfg.diarizer.hop_ms)
    oracle_k = int(recording.n_speakers) if oracle_count else None

    t0 = time.perf_counter()
    if method == "online":
        out = diarize_online(
            embeddings, starts, ends, is_speech, cfg.diarizer, budget_windows,
            n_frames, hop, oracle_n_speakers=oracle_k,
        )
    elif method == "naive_online":
        out = diarize_naive_online(
            embeddings, starts, ends, is_speech, cfg.diarizer, n_frames, hop
        )
    else:
        out = diarize_offline(
            embeddings, starts, ends, is_speech, cfg.diarizer, n_frames, hop,
            method=method, seed=cfg.seed, oracle_n_speakers=oracle_k,
        )
    cluster_s = time.perf_counter() - t0

    timing = {
        "embed_s": float(embed_s),
        "vad_s": float(vad_s),
        "cluster_s": float(cluster_s),
        "total_s": float(embed_s + vad_s + cluster_s),
        "audio_s": float(recording.duration_s),
    }
    return out, timing


def _region_reference_speaker(ref: np.ndarray, start: int, end: int) -> int:
    """The single reference speaker active over ``[start, end)``, or a sentinel.

    Returns the speaker index, ``-1`` when the region has no reference speech, and
    ``-2`` when two or more distinct reference speakers are active. Both sentinels
    exclude the window from the calibration set; conflating them would hide how
    much of the excluded mass is overlap rather than silence.
    """
    if end <= start:
        return -1
    active = np.flatnonzero(ref[start:end].any(axis=0))
    if active.size == 0:
        return -1
    if active.size > 1:
        return -2
    return int(active[0])


def score_output(
    recording: Recording,
    out: DiarizationOutput,
    cfg: Config,
    timing: dict[str, float] | None = None,
) -> RecordingResult:
    """Score one hypothesis: DER family, latency, speaker count, calibration records."""
    ref = recording.reference_matrix()
    n_frames = ref.shape[0]
    frame_labels = out.frame_labels()
    hyp = labels_to_matrix(frame_labels, n_frames)

    primary = der(ref, hyp, collar_frames=0, score_overlap=True)
    no_overlap = der(ref, hyp, collar_frames=0, score_overlap=False)
    collar_frames = int(round(cfg.eval.collar_s * recording.frame_rate)) or int(
        round(0.25 * recording.frame_rate)
    )
    with_collar = der(ref, hyp, collar_frames=collar_frames, score_overlap=True)

    # Latency over *turn decisions* only: non-speech windows are not decisions
    # about a speaker and their zero delay would flatter every system.
    speech_windows = out.labels >= 0
    lat = emission_latency(
        out.window_ends[speech_windows], out.emission_frames[speech_windows],
        recording.frame_rate,
    )

    counts = ref.sum(axis=1)
    speech = counts > 0
    overlap_floor = (
        float(np.maximum(0, counts[speech] - 1).sum() / max(counts.sum(), 1))
        if speech.any()
        else 0.0
    )

    # Calibration: correctness of each turn decision, through the DER mapping.
    used = list(dict.fromkeys(frame_labels[frame_labels >= 0].tolist()))
    remap = {orig: idx for idx, orig in enumerate(used)}
    conf_list: list[float] = []
    margin_list: list[float] = []
    correct_list: list[bool] = []
    n_excluded = 0
    hop = out.hop_frames
    for k in np.flatnonzero(speech_windows):
        end = int(out.window_ends[k])
        spk = _region_reference_speaker(ref, max(0, end - hop), end)
        if spk < 0:
            n_excluded += 1
            continue
        label = int(out.labels[k])
        conf_list.append(float(out.confidence[k]))
        margin_list.append(float(out.margin[k]))
        correct_list.append(primary.mapping.get(spk, -1) == remap.get(label, -999))

    metrics: dict[str, float] = {
        "der": primary.der,
        "miss": primary.miss,
        "false_alarm": primary.false_alarm,
        "confusion": primary.confusion,
        "jer": primary.jer,
        "der_no_overlap": no_overlap.der,
        "der_collar250ms": with_collar.der,
        "overlap_miss_floor": overlap_floor,
        "n_speakers_true": float(recording.n_speakers),
        "n_speakers_pred": float(out.n_speakers),
        "speaker_count_error": float(out.n_speakers - recording.n_speakers),
        "latency_median_ms": lat.median_ms,
        "latency_p95_ms": lat.p95_ms,
        "latency_max_ms": lat.max_ms,
        "latency_first_ms": lat.first_emission_ms,
        "n_decisions": float(lat.n_decisions),
        "turn_accuracy": float(np.mean(correct_list)) if correct_list else float("nan"),
        "duration_s": recording.duration_s,
        "overlap_fraction": recording.overlap_fraction(),
    }
    if timing:
        metrics.update(
            {
                "embed_s": timing["embed_s"],
                "cluster_s": timing["cluster_s"],
                "total_s": timing["total_s"],
                "rtf": timing["total_s"] / max(timing["audio_s"], 1e-9),
            }
        )
    metrics.update({f"extra_{k}": v for k, v in out.extra.items()})

    return RecordingResult(
        recording=recording.name,
        method=out.method,
        latency_budget_ms=float(cfg.diarizer.latency_budget_ms),
        metrics=metrics,
        output=out,
        der_result=primary,
        confidence=np.asarray(conf_list, dtype=np.float64),
        margin=np.asarray(margin_list, dtype=np.float64),
        correct=np.asarray(correct_list, dtype=bool),
        n_excluded_decisions=n_excluded,
    )


def evaluate(
    model: TrainedModel,
    cfg: Config,
    recordings: list[Recording],
    method: str = "online",
    latency_budget_ms: float | None = None,
    oracle_vad: bool = False,
    oracle_count: bool = False,
) -> list[RecordingResult]:
    """Run and score one method over a list of recordings.

    Returns one :class:`RecordingResult` per recording, in input order, so every
    downstream paired test can rely on the pairing without re-sorting.
    """
    results = []
    for rec in recordings:
        out, timing = run_diarizer(
            model, cfg, rec, method=method, latency_budget_ms=latency_budget_ms,
            oracle_vad=oracle_vad, oracle_count=oracle_count,
        )
        res = score_output(rec, out, cfg, timing)
        if latency_budget_ms is not None:
            res.latency_budget_ms = float(latency_budget_ms)
        results.append(res)
    return results


def aggregate(results: list[RecordingResult]) -> dict[str, float]:
    """Mean of each metric across recordings, ignoring NaN, plus the contributing count.

    ``n_contributing_<metric>`` is written for every metric whose NaN count is
    non-zero. The build standard's rule is that undefined is NaN and that the
    number of items behind each mean must be visible; a DER averaged over 22 of
    24 recordings is a different number from one averaged over 24 and the table
    has to say which it is.
    """
    if not results:
        return {}
    keys = sorted({k for r in results for k in r.metrics})
    out: dict[str, float] = {}
    for k in keys:
        vals = np.asarray([r.metrics.get(k, np.nan) for r in results], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[k] = float(finite.mean()) if finite.size else float("nan")
        if finite.size != vals.size:
            out[f"n_contributing_{k}"] = float(finite.size)
    out["n_recordings"] = float(len(results))
    return out


def per_recording_metric(results: list[RecordingResult], metric: str) -> np.ndarray:
    """``(n_recordings,)`` vector of one metric, for the paired statistical tests."""
    return np.asarray([r.metrics.get(metric, np.nan) for r in results], dtype=np.float64)


def pooled_calibration(
    results: list[RecordingResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Concatenate turn-decision records across recordings.

    Pooling across recordings is the right unit here — a calibration curve is a
    statement about decisions, not about recordings — but it does mean a long
    recording contributes more decisions than a short one. All recordings are the
    same length in the shipped config, so the weighting is uniform in practice;
    the caveat is restated in ``docs/RESULTS.md`` because it would not be on real
    data.
    """
    if not results:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), 0
    conf = np.concatenate([r.confidence for r in results]) if results else np.zeros(0)
    margin = np.concatenate([r.margin for r in results]) if results else np.zeros(0)
    correct = np.concatenate([r.correct for r in results]) if results else np.zeros(0, dtype=bool)
    excluded = int(sum(r.n_excluded_decisions for r in results))
    return conf, margin, correct, excluded
