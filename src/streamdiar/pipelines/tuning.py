"""Threshold tuning on the dev split. Run once, frozen into ``configs/base.yaml``.

Every clustering mechanism here has one or two scalar thresholds, and they are not
guessable. The measured cosine-similarity distributions on dev (seed 0) are:

* same speaker, different window: median **0.968**, 5th percentile 0.604
* different speakers:             median **0.641**, 95th percentile 0.956

so the embedding space is a narrow cone rather than an isotropic ball, and the
best single same/different threshold sits near **0.85**, not near the 0.55 a
cosine-similarity intuition suggests. The first version of this project shipped
``spawn_threshold=0.55``, which meant a new speaker was essentially never created:
the diarizer found 1.6 speakers where there were 3.5, and DER was 0.50. That is
recorded here rather than quietly fixed, because it is the reason this module
exists.

Methodology, stated because it is the thing that makes the numbers admissible:

* Tuning uses the **dev** split only — 8 recordings from the test *speaker pool*
  but disjoint recording indices from the test split.
* Tuning uses **seed 0 only**. The chosen values are then frozen and used for
  every seed and every experiment, so seeds 1 and 2 are genuinely held out with
  respect to hyperparameter choice. Re-tuning per seed would leak dev information
  into the seed-variance estimate and shrink it.
* The grid is coarse on purpose. A fine grid over 8 dev recordings would fit dev
  noise, and the resulting values would not transfer.
* Embeddings are computed once per recording and reused across the whole grid.
  That is what makes a 28-point search take seconds instead of minutes, and it is
  exact — the embedder does not depend on any threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, replace
from ..data.generator import Recording
from ..engine import TrainedModel, score_output
from ..models.offline import diarize_offline
from ..models.online import diarize_online
from ..models.vad import energy_vad, frame_energy, windows_are_speech
from .common import write_table


@dataclass
class CachedRecording:
    """A recording with its embeddings and VAD decisions already computed."""

    recording: Recording
    embeddings: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    is_speech: np.ndarray


def precompute(model: TrainedModel, cfg: Config, recordings: list[Recording]) -> list[CachedRecording]:
    """Embed and run the VAD once per recording, so a grid search only re-clusters."""
    win, hop = cfg.frames_per_window, cfg.frames_per_hop
    cached = []
    for rec in recordings:
        emb, starts, ends = model.embedder.embed_recording(rec.features, win, hop)
        energy = frame_energy(rec.features, model.feature_mean, model.feature_sd)
        speech = energy_vad(
            energy, model.vad_threshold,
            cfg.diarizer.vad_hangover_frames, cfg.diarizer.vad_onset_frames,
        )
        is_speech = windows_are_speech(speech, np.maximum(0, ends - hop), ends)
        cached.append(CachedRecording(rec, emb, starts, ends, is_speech))
    return cached


def similarity_distributions(
    cached: list[CachedRecording], hop_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(same_speaker_sims, different_speaker_sims)`` over single-speaker windows.

    Windows whose region contains zero or several reference speakers are skipped:
    the first has no speaker and the second has no single right answer, so
    including either would blur the very distributions being measured.
    """
    same: list[np.ndarray] = []
    diff: list[np.ndarray] = []
    for c in cached:
        ref = c.recording.reference_matrix()
        labels = np.full(len(c.ends), -1, dtype=np.int64)
        for k, end in enumerate(c.ends):
            active = np.flatnonzero(ref[max(0, int(end) - hop_frames) : int(end)].any(axis=0))
            if active.size == 1:
                labels[k] = int(active[0])
        keep = labels >= 0
        emb, lab = c.embeddings[keep], labels[keep]
        if lab.size < 2:
            continue
        sims = emb @ emb.T
        iu = np.triu_indices(lab.size, k=1)
        pair_same = lab[iu[0]] == lab[iu[1]]
        same.append(sims[iu][pair_same])
        diff.append(sims[iu][~pair_same])
    return (
        np.concatenate(same) if same else np.zeros(0),
        np.concatenate(diff) if diff else np.zeros(0),
    )


def _score_cached(cfg: Config, cached: list[CachedRecording], outputs: list) -> dict[str, float]:
    """Mean DER / confusion / speaker count over a set of hypotheses."""
    ders, confs, counts, jers = [], [], [], []
    for c, out in zip(cached, outputs):
        res = score_output(c.recording, out, cfg)
        ders.append(res.metrics["der"])
        confs.append(res.metrics["confusion"])
        jers.append(res.metrics["jer"])
        counts.append(res.metrics["n_speakers_pred"] - res.metrics["n_speakers_true"])
    return {
        "der": float(np.nanmean(ders)),
        "confusion": float(np.nanmean(confs)),
        "jer": float(np.nanmean(jers)),
        "speaker_count_bias": float(np.nanmean(counts)),
    }



def select_best(df: pd.DataFrame, keys: tuple[str, ...], tol: float = 0.01) -> dict[str, float]:
    """Pick a grid row by DER, breaking near-ties on speaker-count bias.

    Selecting purely on dev DER is wrong here, and measurably so. On the online
    grid the best three settings span a DER range of 0.003 — far inside the
    per-recording spread over 8 dev recordings — while their speaker-count bias
    ranges from +0.75 to +2.5 speakers. Taking the DER argmin would freeze a
    badly over-clustering configuration on the strength of a difference that is
    not there.

    So: among all rows within ``tol`` *relative* DER of the best, choose the one
    with the smallest ``|speaker_count_bias|``. This is a documented,
    dev-only, pre-registered rule rather than a look at the test set, and it is
    applied identically to the online and both offline grids.
    """
    best_der = float(df["der"].min())
    cutoff = best_der * (1.0 + float(tol))
    near = df[df["der"] <= cutoff].copy()
    near["abs_bias"] = near["speaker_count_bias"].abs()
    row = near.sort_values(["abs_bias", "der"]).iloc[0]
    out = {k: float(row[k]) for k in keys}
    out["der"] = float(row["der"])
    out["speaker_count_bias"] = float(row["speaker_count_bias"])
    out["n_near_ties"] = float(len(near))
    return out


def tune_online(
    cfg: Config,
    cached: list[CachedRecording],
    spawn_grid: tuple[float, ...] = (0.70, 0.75, 0.80, 0.83, 0.86, 0.89, 0.92),
    micro_grid: tuple[float, ...] = (0.85, 0.90, 0.93, 0.96),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search ``(spawn_threshold, micro_cluster_threshold)`` on dev DER.

    Only combinations with ``micro >= spawn`` are considered: a micro-cluster
    threshold below the spawn threshold would admit two speakers into one
    micro-cluster that the tracker would then refuse to treat as the same
    speaker, which is incoherent rather than merely suboptimal.
    """
    hop = cfg.frames_per_hop
    rows = []
    for spawn in spawn_grid:
        for micro in micro_grid:
            if micro < spawn:
                continue
            dcfg = replace(cfg.diarizer, spawn_threshold=spawn, micro_cluster_threshold=micro)
            outputs = [
                diarize_online(
                    c.embeddings, c.starts, c.ends, c.is_speech, dcfg,
                    cfg.budget_windows, c.recording.n_frames, hop,
                )
                for c in cached
            ]
            scores = _score_cached(cfg, cached, outputs)
            rows.append({"spawn_threshold": spawn, "micro_cluster_threshold": micro, **scores})
    df = pd.DataFrame(rows).sort_values("der").reset_index(drop=True)
    return select_best(df, ("spawn_threshold", "micro_cluster_threshold")), df


def tune_offline_ahc(
    cfg: Config,
    cached: list[CachedRecording],
    grid: tuple[float, ...] = (0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26, 0.32, 0.40),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search the AHC stopping threshold (cosine *distance*) on dev DER."""
    hop = cfg.frames_per_hop
    rows = []
    for thr in grid:
        dcfg = replace(cfg.diarizer, offline_ahc_threshold=thr)
        outputs = [
            diarize_offline(
                c.embeddings, c.starts, c.ends, c.is_speech, dcfg,
                c.recording.n_frames, hop, method="offline_ahc", seed=cfg.seed,
            )
            for c in cached
        ]
        rows.append({"offline_ahc_threshold": thr, **_score_cached(cfg, cached, outputs)})
    df = pd.DataFrame(rows).sort_values("der").reset_index(drop=True)
    return select_best(df, ("offline_ahc_threshold",)), df


def tune_offline_spectral(
    cfg: Config,
    cached: list[CachedRecording],
    grid: tuple[float, ...] = (0.0, 0.50, 0.70, 0.80, 0.90, 0.95),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search the spectral affinity-refinement percentile on dev DER."""
    hop = cfg.frames_per_hop
    rows = []
    for pct in grid:
        dcfg = replace(cfg.diarizer, offline_spectral_percentile=pct)
        outputs = [
            diarize_offline(
                c.embeddings, c.starts, c.ends, c.is_speech, dcfg,
                c.recording.n_frames, hop, method="offline_spectral", seed=cfg.seed,
            )
            for c in cached
        ]
        rows.append({"offline_spectral_percentile": pct, **_score_cached(cfg, cached, outputs)})
    df = pd.DataFrame(rows).sort_values("der").reset_index(drop=True)
    return select_best(df, ("offline_spectral_percentile",)), df


def write_tuning_tables(
    tables: dict[str, pd.DataFrame], out_dir: str | Path = "results/tables"
) -> None:
    """Write each grid to ``threshold_tuning_<name>.csv`` — the evidence for the choices."""
    for name, df in tables.items():
        write_table(Path(out_dir) / f"threshold_tuning_{name}.csv", df)
