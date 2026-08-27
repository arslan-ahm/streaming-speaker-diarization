"""The headline experiment: DER as a function of the latency budget.

This is the artifact the project exists to produce. The offline reference is not
a competitor to be beaten — it is the horizontal asymptote at infinite latency,
and the deliverable is the *shape of the approach to it*: how much DER a bounded
emission delay costs, at each budget, with the noise scale attached so the reader
can see which parts of the curve are real structure and which are wobble.

Design notes that matter for reading the curve
----------------------------------------------
**The curve is flat below one hop, by construction.** A budget of ``B`` ms buys
``floor(B / hop_ms)`` hops of lookahead, so at ``hop = 250 ms`` every budget in
``[0, 250)`` buys nothing — the next window does not exist yet. The sweep
includes 0 ms and 125 ms specifically to make that step visible rather than
letting a reader assume the curve is smooth. It is the honest conversion:
pretending a 125 ms budget helps would require partial windows the streaming
grid does not produce.

**Chunked one budget per invocation.** ``scripts/latency_sweep.py`` runs a single
budget and appends its rows, so an interrupted sweep costs one budget rather than
the whole curve, and a completed budget is never recomputed. The function checks
what is already in the CSV before running.

**The offline line is measured, not assumed.** Its latency is
``n_frames - window_end`` per decision, which for 60 s recordings is a median of
tens of seconds; it appears in the figure as a point at its own measured latency,
not at "infinity".

**Budgets are compared within a seed.** The test recordings are generated per
seed, so a paired test across budgets is valid only within one seed; the
across-seed spread is reported separately as the noise scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..engine import aggregate, evaluate
from ..metrics.stats import noise_scale_from_seeds
from .common import append_rows, load_or_train, rows_from_results, test_split, write_table

#: The swept budgets in milliseconds. 0 and 125 both buy zero lookahead at the
#: default 250 ms hop; that flat step is a feature of the measurement, not noise.
DEFAULT_BUDGETS_MS: tuple[float, ...] = (0.0, 125.0, 250.0, 500.0, 1000.0, 2000.0,
                                        4000.0, 8000.0, 16000.0)

#: Offline systems evaluated once each to draw the reference line.
OFFLINE_METHODS = ("offline_ahc", "offline_spectral")


def already_done(csv_path: str | Path, budget_ms: float, seed: int, method: str) -> bool:
    """Whether this ``(budget, seed, method)`` cell is already in the CSV.

    The brief is explicit about checking for landed output before re-running an
    experiment; this is that check, and it is what makes the chunked sweep
    resumable rather than merely restartable.
    """
    p = Path(csv_path)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError):
        return False
    if not {"latency_budget_ms", "seed", "method"} <= set(df.columns):
        return False
    hit = (
        np.isclose(df["latency_budget_ms"].to_numpy(dtype=np.float64), float(budget_ms))
        & (df["seed"].to_numpy() == int(seed))
        & (df["method"].to_numpy() == method)
    )
    return bool(hit.any())


def run_latency_point(
    cfg: Config,
    seed: int,
    budget_ms: float,
    out_csv: str | Path = "results/tables/latency_sweep.csv",
    method: str = "online",
    force: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate one ``(budget, seed)`` cell and append its per-recording rows.

    Returns the aggregate metrics for the cell. Skips and returns the stored
    aggregate when the cell is already present, unless ``force``.
    """
    if not force and already_done(out_csv, budget_ms, seed, method):
        if verbose:
            print(f"  [skip] {method} budget {budget_ms:.0f} ms seed {seed} already present")
        df = pd.read_csv(out_csv)
        sel = (
            np.isclose(df["latency_budget_ms"], float(budget_ms))
            & (df["seed"] == int(seed))
            & (df["method"] == method)
        )
        return {"der": float(df.loc[sel, "der"].mean())}

    model = load_or_train(cfg, seed, verbose=verbose)
    recordings = test_split(cfg, seed)
    results = evaluate(model, cfg, recordings, method=method, latency_budget_ms=budget_ms)
    rows = rows_from_results(
        results, seed=seed,
        budget_windows=int(budget_ms // cfg.diarizer.hop_ms),
    )
    for row in rows:
        row["latency_budget_ms"] = float(budget_ms)
    append_rows(out_csv, rows)
    agg = aggregate(results)
    if verbose:
        print(
            f"  {method} budget {budget_ms:7.0f} ms "
            f"(={int(budget_ms // cfg.diarizer.hop_ms)} hops) "
            f"seed {seed}: DER {agg['der']:.4f}  conf {agg['confusion']:.4f}  "
            f"measured latency {agg['latency_median_ms']:.0f} ms  "
            f"query size {agg.get('extra_mean_query_size', float('nan')):.2f}"
        )
    return agg


def run_latency_sweep(
    cfg: Config,
    seeds: tuple[int, ...] = (0, 1, 2),
    budgets_ms: tuple[float, ...] = DEFAULT_BUDGETS_MS,
    out_csv: str | Path = "results/tables/latency_sweep.csv",
    include_offline: bool = True,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the whole sweep in one process. Prefer the chunked script for long runs."""
    for seed in seeds:
        for budget in budgets_ms:
            run_latency_point(cfg, seed, budget, out_csv, "online", force, verbose)
        if include_offline:
            for method in OFFLINE_METHODS:
                run_offline_reference(cfg, seed, out_csv, method, force, verbose)
    return pd.read_csv(out_csv)


def run_offline_reference(
    cfg: Config,
    seed: int,
    out_csv: str | Path = "results/tables/latency_sweep.csv",
    method: str = "offline_ahc",
    force: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate an offline method into the same CSV, at its own measured latency.

    Stored with ``latency_budget_ms = inf`` because that is what the algorithm's
    budget actually is; its *measured* median emission delay is in the
    ``latency_median_ms`` column and is what the figure plots it at.
    """
    if not force and already_done(out_csv, float("inf"), seed, method):
        if verbose:
            print(f"  [skip] {method} seed {seed} already present")
        return {}
    model = load_or_train(cfg, seed, verbose=verbose)
    recordings = test_split(cfg, seed)
    results = evaluate(model, cfg, recordings, method=method)
    rows = rows_from_results(results, seed=seed, budget_windows=-1)
    for row in rows:
        row["latency_budget_ms"] = float("inf")
    append_rows(out_csv, rows)
    agg = aggregate(results)
    if verbose:
        print(
            f"  {method} (offline) seed {seed}: DER {agg['der']:.4f}  "
            f"measured latency {agg['latency_median_ms']:.0f} ms"
        )
    return agg


def summarise_curve(
    table: pd.DataFrame,
    metrics: tuple[str, ...] = ("der", "miss", "false_alarm", "confusion", "jer",
                               "turn_accuracy", "n_speakers_pred", "latency_median_ms",
                               "latency_p95_ms", "extra_mean_query_size"),
    out_dir: str | Path = "results/tables",
    filename: str = "latency_curve.csv",
) -> pd.DataFrame:
    """Collapse the sweep to one row per ``(method, budget)`` with the noise scale.

    ``*_noise_scale`` is ``sqrt(2) * sd`` of the per-seed means — the scale on
    which to judge whether two adjacent points on the curve differ at all.
    """
    rows = []
    for (method, budget), sub in table.groupby(["method", "latency_budget_ms"], sort=True):
        row: dict[str, object] = {
            "method": method,
            "latency_budget_ms": float(budget),
            "n_seeds": int(sub["seed"].nunique()),
            "n_recordings": int(len(sub)),
        }
        for metric in metrics:
            if metric not in sub.columns:
                continue
            per_seed = sub.groupby("seed")[metric].mean().to_numpy(dtype=np.float64)
            vals = np.asarray(sub[metric], dtype=np.float64)
            row[metric] = float(np.nanmean(vals)) if vals.size else float("nan")
            row[f"{metric}_noise_scale"] = noise_scale_from_seeds(per_seed)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["method", "latency_budget_ms"]).reset_index(drop=True)
    write_table(Path(out_dir) / filename, df)
    return df
