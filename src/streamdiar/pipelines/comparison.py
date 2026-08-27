"""The method comparison: proposed online system against every baseline, with statistics.

What is compared, and why each one is here
------------------------------------------
============================  =============================================
variant                       what it isolates
============================  =============================================
``online``                    the proposed bounded-latency mechanism
``naive_online``              greedy nearest-centroid; the obvious online thing
``offline_ahc``               the reference approach, agglomerative
``offline_spectral``          the reference approach, spectral + eigengap
``online_oracle_count``       online, told the true speaker count
``offline_ahc_oracle_count``  offline, told the true speaker count
``online_oracle_vad``         online with reference speech/non-speech
``offline_ahc_oracle_vad``    offline with reference speech/non-speech
============================  =============================================

The two oracle-count rows separate "clustering is hard" from "counting speakers
is hard", on *both* sides of the online/offline divide — doing it only for the
online system would let a large oracle-count gain be mistaken for a property of
online clustering when it is a property of the task. The two oracle-VAD rows do
the same for speech detection. None of the oracle rows is a valid deployable
system; they are diagnostics and are labelled as such in every table.

Every variant shares the same trained embedder, the same recordings, the same
seed and the same scoring code. The only thing that differs is the mechanism
under test.

Statistical procedure
---------------------
1. Per-recording metrics for every variant, all seeds, to
   ``results/tables/method_comparison.csv``.
2. Per-seed means, then the across-seed standard deviation, giving the run-to-run
   noise scale ``sqrt(2) * sd`` for every (variant, metric).
3. Paired Wilcoxon signed-rank plus a paired bootstrap CI over the 24 test
   recordings of the reference seed, Holm-Bonferroni corrected across the
   five-metric family, effect size reported, to
   ``results/tables/statistical_tests.csv``.
4. Each difference graded by :func:`streamdiar.metrics.stats.verdict`, which
   puts the noise scale *ahead* of the p-value: a paired test over recordings
   conditions on one trained model per method and can be arbitrarily significant
   about a difference a different seed would erase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
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

#: ``(label, method, oracle_count, oracle_vad)``.
VARIANTS: tuple[tuple[str, str, bool, bool], ...] = (
    ("online", "online", False, False),
    ("naive_online", "naive_online", False, False),
    ("offline_ahc", "offline_ahc", False, False),
    ("offline_spectral", "offline_spectral", False, False),
    ("online_oracle_count", "online", True, False),
    ("offline_ahc_oracle_count", "offline_ahc", True, False),
    ("online_oracle_vad", "online", False, True),
    ("offline_ahc_oracle_vad", "offline_ahc", False, True),
)

#: The variant every other variant is compared against.
REFERENCE_VARIANT = "online"


def run_comparison(
    cfg: Config,
    seeds: tuple[int, ...] = (0, 1, 2),
    variants: tuple[tuple[str, str, bool, bool], ...] = VARIANTS,
    out_dir: str | Path = "results/tables",
    per_recording_csv: str = "method_comparison.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate every variant on every seed, appending per-recording rows as it goes.

    Returns the full per-recording table. Rows land on disk after each
    ``(seed, variant)`` pair, so an interrupted run keeps everything it finished.
    """
    out = Path(out_dir)
    csv_path = out / per_recording_csv
    if csv_path.exists():
        csv_path.unlink()  # a fresh matrix, not an append to a stale one

    for seed in seeds:
        if verbose:
            print(f"[comparison] seed {seed}")
        model = load_or_train(cfg, seed, verbose=verbose)
        recordings = test_split(cfg, seed)
        for label, method, oracle_count, oracle_vad in variants:
            results = evaluate(
                model, cfg, recordings, method=method,
                latency_budget_ms=cfg.diarizer.latency_budget_ms,
                oracle_vad=oracle_vad, oracle_count=oracle_count,
            )
            rows = rows_from_results(results, variant=label, seed=seed,
                                     oracle_count=oracle_count, oracle_vad=oracle_vad)
            append_rows(csv_path, rows)
            if verbose:
                agg = aggregate(results)
                print(
                    f"  {label:26s} DER {agg['der']:.4f}  conf {agg['confusion']:.4f}  "
                    f"lat {agg['latency_median_ms']:7.1f} ms  "
                    f"spk {agg['n_speakers_pred']:.2f}/{agg['n_speakers_true']:.2f}"
                )
    return pd.read_csv(csv_path)


def summarise(
    table: pd.DataFrame,
    metrics: tuple[str, ...] = METRIC_FAMILY,
    extra_metrics: tuple[str, ...] = (
        "miss", "false_alarm", "der_no_overlap", "der_collar250ms",
        "latency_median_ms", "latency_p95_ms", "n_speakers_pred", "rtf",
        "overlap_miss_floor",
    ),
    out_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Per-variant summary: per-seed means, across-seed sd, and the noise scale.

    ``n_contributing`` counts the recordings whose metric was finite, because a
    DER averaged over 22 of 24 recordings is a different number from one averaged
    over 24 and the table has to say which it is.
    """
    all_metrics = tuple(dict.fromkeys(metrics + extra_metrics))
    rows = []
    for variant, sub in table.groupby("variant", sort=False):
        row: dict[str, object] = {"variant": variant}
        for metric in all_metrics:
            if metric not in sub.columns:
                continue
            per_seed = sub.groupby("seed")[metric].mean()
            values = per_seed.to_numpy(dtype=np.float64)
            finite = np.asarray(sub[metric], dtype=np.float64)
            row[metric] = float(np.nanmean(finite)) if finite.size else float("nan")
            row[f"{metric}_seed_sd"] = (
                float(values.std(ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
            )
            row[f"{metric}_noise_scale"] = noise_scale_from_seeds(values)
            row[f"{metric}_n_contributing"] = int(np.isfinite(finite).sum())
        row["n_seeds"] = int(sub["seed"].nunique())
        rows.append(row)
    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "method_summary.csv", df)
    return df


def statistical_tests(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    reference: str = REFERENCE_VARIANT,
    metrics: tuple[str, ...] = METRIC_FAMILY,
    reference_seed: int | None = None,
    n_resamples: int = 2000,
    out_dir: str | Path = "results/tables",
    filename: str = "statistical_tests.csv",
) -> pd.DataFrame:
    """Paired tests of every variant against ``reference``, Holm-corrected per pair.

    Args:
        table: The per-recording table from :func:`run_comparison`.
        summary: The output of :func:`summarise`, for the noise scales.
        reference: The variant everything is compared against.
        metrics: The metric family Holm corrects across.
        reference_seed: Which seed's recordings the paired test uses. Defaults to
            the smallest seed present.
        n_resamples: Bootstrap replicates; the standard's floor is 2000.

    Returns and writes the table of comparisons.

    **The unit of analysis is a recording, not a training run.** The pairing is
    over the 24 test recordings of a single seed, so each test answers "do these
    two *sets of weights and mechanisms* differ on this test set", not "is this
    mechanism better in general". A method-level claim needs the training run as
    the sampling unit, and with three seeds there is not enough power for that —
    which is exactly why the noise-scale column, not the p-value, decides the
    verdict.

    The paired test must stay *within* one seed because the test recordings
    themselves are generated per seed; pooling seeds would break the pairing.
    """
    seeds = sorted(table["seed"].unique())
    seed = int(seeds[0]) if reference_seed is None else int(reference_seed)
    sub = table[table["seed"] == seed]

    noise: dict[tuple[str, str], float] = {}
    for _, row in summary.iterrows():
        for metric in metrics:
            key = f"{metric}_noise_scale"
            if key in row:
                noise[(str(row["variant"]), metric)] = float(row[key])

    ref_rows = sub[sub["variant"] == reference].sort_values("recording")
    if ref_rows.empty:
        raise ValueError(f"reference variant {reference!r} absent from the table")

    out_rows: list[dict[str, object]] = []
    for variant in [v for v in sub["variant"].unique() if v != reference]:
        other = sub[sub["variant"] == variant].sort_values("recording")
        if len(other) != len(ref_rows):
            raise ValueError(
                f"unpaired data: {reference} has {len(ref_rows)} recordings, "
                f"{variant} has {len(other)}"
            )
        comparisons = []
        for metric in metrics:
            if metric not in sub.columns:
                continue
            a = ref_rows[metric].to_numpy(dtype=np.float64)
            b = other[metric].to_numpy(dtype=np.float64)
            # Noise scale of a *difference* uses the larger of the two variants'
            # run-to-run scales: the conservative choice when they differ.
            ns = max(
                noise.get((reference, metric), float("nan")),
                noise.get((variant, metric), float("nan")),
            )
            comparisons.append(
                compare(a, b, reference, variant, metric=metric,
                        n_resamples=n_resamples, seed=seed, noise_scale=ns)
            )
        holm_bonferroni(comparisons)
        for c in comparisons:
            d = c.to_dict()
            d["seed"] = seed
            d["family_size"] = len(comparisons)
            out_rows.append(d)

    df = pd.DataFrame(out_rows)
    write_table(Path(out_dir) / filename, df)
    return df


def headline_table(
    summary: pd.DataFrame,
    metrics: tuple[str, ...] = (
        "der", "miss", "false_alarm", "confusion", "jer",
        "latency_median_ms", "n_speakers_pred", "rtf",
    ),
) -> pd.DataFrame:
    """The README's table: point estimate and run-to-run noise scale per variant."""
    cols: dict[str, list[object]] = {"variant": list(summary["variant"])}
    for m in metrics:
        if m in summary.columns:
            cols[m] = [float(v) for v in summary[m]]
        ns = f"{m}_noise_scale"
        if ns in summary.columns:
            cols[f"{m}_noise"] = [float(v) for v in summary[ns]]
    return pd.DataFrame(cols)
