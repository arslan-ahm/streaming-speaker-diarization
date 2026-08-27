"""Efficiency: wall-clock, real-time factor, and per-decision latency, measured.

The build standard is blunt about this: params and MACs are inputs, wall-clock is
the claim. It also records a specific prior failure — three warm-up iterations
made a small model look 5x slower than reality — so the defaults here are the
standard's floor of **8 warm-up iterations and 25 repeats**, with median and IQR
rather than a mean that one scheduler hiccup dominates.

Three separate quantities, because conflating them is how efficiency claims go
wrong:

**Algorithmic emission delay** — the project's actual axis. Audio time between a
window's audio arriving and its label being emitted. A property of the algorithm's
dependency structure, so it is exact and machine-independent. The online system's
is its budget; the offline system's is ``n_frames - window_end``, which grows with
recording length.

**Per-decision compute time** — how long the clustering step takes per window.
Small for both, and reported so the reader can confirm that the emission-delay
comparison is a statement about algorithm design and not about this laptop.

**Clustering wall-clock scaling** — the term where the two designs genuinely
diverge in *compute*. Offline global clustering builds an ``N x N`` affinity and
runs an ``O(N^3)`` eigendecomposition or merge loop in the number of windows; the
online tracker is ``O(max_speakers)`` per window. :func:`scaling_study` measures
this at several recording lengths, because a claim of "bounded per-frame cost" is
a claim about a *slope*, and one length cannot show a slope.

The embedding forward pass is shared by every method and is deliberately excluded
from the clustering measurements, then reported separately. Including it would
dilute the very term the comparison is about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, replace
from ..data.generator import generate_recording
from ..engine import run_diarizer
from ..metrics.latency import emission_latency
from ..models.offline import diarize_offline
from ..models.online import diarize_online
from ..models.vad import energy_vad, frame_energy, windows_are_speech
from ..utils.bench import count_parameters, measure_latency, real_time_factor
from .common import load_or_train, test_split, write_table

#: Recording lengths for the scaling study, in seconds.
SCALING_DURATIONS_S = (15.0, 30.0, 60.0, 120.0, 240.0)


def run_efficiency(
    cfg: Config,
    seed: int = 0,
    methods: tuple[str, ...] = ("online", "naive_online", "offline_ahc", "offline_spectral"),
    out_dir: str | Path = "results/tables",
    verbose: bool = True,
) -> pd.DataFrame:
    """Wall-clock, RTF and emission latency per method on the standard test split."""
    model = load_or_train(cfg, seed, verbose=verbose)
    recordings = test_split(cfg, seed)
    probe = recordings[0]
    rows: list[dict[str, object]] = []

    for method in methods:
        # One representative recording, benchmarked properly, rather than an
        # average over the split: the standard's warm-up and repeat floors are
        # about one measurement being trustworthy.
        timing = measure_latency(
            lambda m=method: run_diarizer(model, cfg, probe, method=m),
            warmup=cfg.eval.bench_warmup,
            repeats=cfg.eval.bench_repeats,
        )
        out, split_timing = run_diarizer(model, cfg, probe, method=method)
        speech = out.labels >= 0
        lat = emission_latency(
            out.window_ends[speech], out.emission_frames[speech], probe.frame_rate
        )
        n_decisions = int(speech.sum())
        rows.append(
            {
                "method": method,
                "n_parameters": count_parameters(model.embedder),
                "total_median_ms": timing.median_ms,
                "total_iqr_ms": timing.iqr_ms,
                "total_p95_ms": timing.p95_ms,
                "embed_ms": split_timing["embed_s"] * 1000.0,
                "vad_ms": split_timing["vad_s"] * 1000.0,
                "cluster_ms": split_timing["cluster_s"] * 1000.0,
                "cluster_ms_per_decision": (
                    split_timing["cluster_s"] * 1000.0 / max(n_decisions, 1)
                ),
                "rtf": real_time_factor(timing.median_ms / 1000.0, probe.duration_s),
                "audio_s": probe.duration_s,
                "n_decisions": n_decisions,
                "emission_latency_median_ms": lat.median_ms,
                "emission_latency_p95_ms": lat.p95_ms,
                "emission_latency_max_ms": lat.max_ms,
                "emission_latency_first_ms": lat.first_emission_ms,
                "repeats": timing.repeats,
                "warmup": cfg.eval.bench_warmup,
            }
        )
        if verbose:
            r = rows[-1]
            print(
                f"  {method:18s} total {r['total_median_ms']:7.1f} ms "
                f"(IQR {r['total_iqr_ms']:.1f})  cluster {r['cluster_ms']:6.1f} ms  "
                f"RTF {r['rtf']:.4f}  emission latency median "
                f"{r['emission_latency_median_ms']:8.1f} ms"
            )

    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "efficiency.csv", df)
    return df


def scaling_study(
    cfg: Config,
    seed: int = 0,
    durations_s: tuple[float, ...] = SCALING_DURATIONS_S,
    out_dir: str | Path = "results/tables",
    verbose: bool = True,
) -> pd.DataFrame:
    """Clustering wall-clock versus recording length, for the online and offline paths.

    Measures the *clustering step only*, on pre-computed embeddings, so the shared
    linear-time embedding pass does not mask the difference in scaling. The online
    tracker should be linear in the number of windows; the offline methods should
    be super-linear.
    """
    model = load_or_train(cfg, seed, verbose=verbose)
    rows: list[dict[str, object]] = []

    for duration in durations_s:
        dcfg = replace(cfg.data, duration_s=float(duration))
        rec = generate_recording(dcfg, seed, 9000, pool="test")
        win, hop = cfg.frames_per_window, cfg.frames_per_hop
        embeddings, starts, ends = model.embedder.embed_recording(rec.features, win, hop)
        energy = frame_energy(rec.features, model.feature_mean, model.feature_sd)
        speech_frames = energy_vad(
            energy, model.vad_threshold,
            cfg.diarizer.vad_hangover_frames, cfg.diarizer.vad_onset_frames,
        )
        is_speech = windows_are_speech(speech_frames, np.maximum(0, ends - hop), ends)
        n_frames = rec.n_frames
        budget = cfg.budget_windows

        # Loop variables are bound as keyword defaults rather than captured:
        # a late-binding closure inside a loop is the classic way a timing
        # harness ends up benchmarking the last iteration's data every time.
        def _online(e=embeddings, s=starts, n=ends, sp=is_speech,
                    b=budget, nf=n_frames, h=hop):
            return diarize_online(e, s, n, sp, cfg.diarizer, b, nf, h)

        def _ahc(e=embeddings, s=starts, n=ends, sp=is_speech,
                 nf=n_frames, h=hop):
            return diarize_offline(e, s, n, sp, cfg.diarizer, nf, h,
                                  method="offline_ahc", seed=seed)

        def _spectral(e=embeddings, s=starts, n=ends, sp=is_speech,
                      nf=n_frames, h=hop):
            return diarize_offline(e, s, n, sp, cfg.diarizer, nf, h,
                                  method="offline_spectral", seed=seed)

        jobs = {
            "online": _online,
            "offline_ahc": _ahc,
            "offline_spectral": _spectral,
        }
        for name, fn in jobs.items():
            t = measure_latency(fn, warmup=max(2, cfg.eval.bench_warmup // 2), repeats=9)
            rows.append(
                {
                    "method": name,
                    "duration_s": float(duration),
                    "n_windows": int(embeddings.shape[0]),
                    "n_speech_windows": int(is_speech.sum()),
                    "cluster_median_ms": t.median_ms,
                    "cluster_iqr_ms": t.iqr_ms,
                    "ms_per_window": t.median_ms / max(int(is_speech.sum()), 1),
                }
            )
        if verbose:
            trio = {r["method"]: r["cluster_median_ms"] for r in rows[-3:]}
            print(
                f"  {duration:6.0f} s ({int(embeddings.shape[0])} windows): "
                + "  ".join(f"{k} {v:.1f} ms" for k, v in trio.items())
            )

    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "scaling.csv", df)
    return df
