"""Ablations: one config switch each, with significance tests rather than raw deltas.

The rule the build standard sets is that a variant which changes three things
attributes nothing. So each row here flips exactly one switch relative to the full
system, and each is reported with its paired test *and* its position relative to
the run-to-run noise scale.

===================== ================================= ==============================
ablation              switch                            question it answers
===================== ================================= ==============================
``full``              —                                 the reference point
``no_bounded_window`` ``bounded_window: false``         does the within-budget
                                                        micro-cluster do the work, or
                                                        is having a buffer enough?
``no_spawn``          ``spawn_enabled: false``          does creating speakers after
                                                        the warm-up prefix matter?
``no_momentum``       ``centroid_momentum: 1.0``        is a centroid with memory
                                                        better than the last
                                                        embedding assigned to it?
``noncausal``         ``embedder.causal: false``        what does causality cost?
===================== ================================= ==============================

``noncausal`` is the one ablation that needs its **own trained embedder**, because
it changes the architecture rather than the inference mechanism. It therefore gets
its own checkpoint tag, and ``build_embedder`` is seeded locally so the causal and
non-causal models start from the same initialisation — otherwise the ablation
would confound padding with initialisation.

``noncausal`` is also **not a valid online system**: with centred padding each
frame's representation depends on ~150 ms of future audio, so its reported
emission latency understates its true latency by that amount. It is a diagnostic,
labelled as such, and ``tests/test_causality.py`` asserts it fails the bit-identity
check the causal model passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, replace
from ..engine import aggregate, evaluate
from ..metrics.stats import compare, holm_bonferroni, noise_scale_from_seeds
from .common import (
    METRIC_FAMILY,
    append_rows,
    load_or_train,
    rows_from_results,
    test_split,
    write_table,
)

#: ``(label, diarizer overrides, embedder overrides, checkpoint tag)``.
ABLATIONS: tuple[tuple[str, dict[str, object], dict[str, object], str | None], ...] = (
    ("full", {}, {}, None),
    ("no_bounded_window", {"bounded_window": False}, {}, None),
    ("no_spawn", {"spawn_enabled": False}, {}, None),
    ("no_momentum", {"centroid_momentum": 1.0}, {}, None),
    ("noncausal", {}, {"causal": False}, "noncausal"),
)


def run_ablations(
    cfg: Config,
    seeds: tuple[int, ...] = (0, 1, 2),
    ablations: tuple[tuple[str, dict[str, object], dict[str, object], str | None], ...] = ABLATIONS,
    out_dir: str | Path = "results/tables",
    per_recording_csv: str = "ablation_per_recording.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every ablation on every seed, appending per-recording rows as it goes."""
    csv_path = Path(out_dir) / per_recording_csv
    if csv_path.exists():
        csv_path.unlink()

    for seed in seeds:
        if verbose:
            print(f"[ablation] seed {seed}")
        for label, diar_over, emb_over, tag in ablations:
            variant_cfg = replace(
                cfg,
                diarizer=replace(cfg.diarizer, **diar_over) if diar_over else cfg.diarizer,
                embedder=replace(cfg.embedder, **emb_over) if emb_over else cfg.embedder,
            )
            model = load_or_train(variant_cfg, seed, tag=tag, verbose=verbose)
            recordings = test_split(variant_cfg, seed)
            results = evaluate(
                model, variant_cfg, recordings, method="online",
                latency_budget_ms=variant_cfg.diarizer.latency_budget_ms,
            )
            rows = rows_from_results(results, ablation=label, seed=seed)
            append_rows(csv_path, rows)
            if verbose:
                agg = aggregate(results)
                print(
                    f"  {label:20s} DER {agg['der']:.4f}  conf {agg['confusion']:.4f}  "
                    f"spk {agg['n_speakers_pred']:.2f}  "
                    f"query {agg.get('extra_mean_query_size', float('nan')):.2f}"
                )
    return pd.read_csv(csv_path)


def summarise_ablations(
    table: pd.DataFrame,
    metrics: tuple[str, ...] = METRIC_FAMILY,
    extra_metrics: tuple[str, ...] = ("miss", "false_alarm", "n_speakers_pred",
                                      "extra_mean_query_size", "latency_median_ms"),
    out_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """One row per ablation: mean, across-seed sd, noise scale, contributing count."""
    all_metrics = tuple(dict.fromkeys(metrics + extra_metrics))
    rows = []
    for label, sub in table.groupby("ablation", sort=False):
        row: dict[str, object] = {"ablation": label, "n_seeds": int(sub["seed"].nunique())}
        for metric in all_metrics:
            if metric not in sub.columns:
                continue
            per_seed = sub.groupby("seed")[metric].mean().to_numpy(dtype=np.float64)
            vals = np.asarray(sub[metric], dtype=np.float64)
            row[metric] = float(np.nanmean(vals)) if vals.size else float("nan")
            row[f"{metric}_seed_sd"] = (
                float(per_seed.std(ddof=1)) if np.isfinite(per_seed).sum() > 1 else float("nan")
            )
            row[f"{metric}_noise_scale"] = noise_scale_from_seeds(per_seed)
            row[f"{metric}_n_contributing"] = int(np.isfinite(vals).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "ablation_summary.csv", df)
    return df


def ablation_tests(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    reference: str = "full",
    metrics: tuple[str, ...] = METRIC_FAMILY,
    reference_seed: int | None = None,
    n_resamples: int = 2000,
    out_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Paired tests of each ablation against ``full``, Holm-corrected per ablation.

    Same unit-of-analysis caveat as the method comparison: the pairing is over
    recordings within one seed, so the test is about one pair of trained models,
    and the noise-scale column is what decides whether a mechanism's contribution
    is claimable.
    """
    seeds = sorted(table["seed"].unique())
    seed = int(seeds[0]) if reference_seed is None else int(reference_seed)
    sub = table[table["seed"] == seed]

    noise: dict[tuple[str, str], float] = {}
    for _, row in summary.iterrows():
        for metric in metrics:
            key = f"{metric}_noise_scale"
            if key in row:
                noise[(str(row["ablation"]), metric)] = float(row[key])

    ref_rows = sub[sub["ablation"] == reference].sort_values("recording")
    if ref_rows.empty:
        raise ValueError(f"reference ablation {reference!r} absent from the table")

    out_rows: list[dict[str, object]] = []
    for label in [v for v in sub["ablation"].unique() if v != reference]:
        other = sub[sub["ablation"] == label].sort_values("recording")
        comparisons = []
        for metric in metrics:
            if metric not in sub.columns:
                continue
            ns = max(
                noise.get((reference, metric), float("nan")),
                noise.get((label, metric), float("nan")),
            )
            comparisons.append(
                compare(
                    ref_rows[metric].to_numpy(dtype=np.float64),
                    other[metric].to_numpy(dtype=np.float64),
                    reference, label, metric=metric,
                    n_resamples=n_resamples, seed=seed, noise_scale=ns,
                )
            )
        holm_bonferroni(comparisons)
        for c in comparisons:
            d = c.to_dict()
            d["seed"] = seed
            d["family_size"] = len(comparisons)
            out_rows.append(d)

    df = pd.DataFrame(out_rows)
    write_table(Path(out_dir) / "ablation_tests.csv", df)
    return df
