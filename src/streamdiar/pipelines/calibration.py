"""Calibration and abstention over turn assignments.

The practical payoff of the online design is here. A bounded-latency diarizer is
*less accurate* than an offline one; if it also knows **which** of its turn
decisions are probably wrong, a deployment can defer that minority to a slower
offline pass or a human and recover most of the gap at a fraction of the cost.
So this pipeline measures two separate things:

1. **Is the confidence calibrated?** ECE (equal-width), ACE (equal-mass), MCE,
   Brier, NLL, and the signed overconfidence. Reported before *and after* a
   single temperature fitted on the **dev** split — never on test, which is the
   easiest way to accidentally cheat at this metric.
2. **Is the confidence useful for abstention?** Risk-coverage curve, AURC against
   the oracle AURC floor, error-detection AUROC, and the concrete operating
   points: DER-equivalent error over the retained turns at 90%, 80% and 70%
   coverage.

The confidence being scored is the assignment margin defined in
``models/online.py``: a softmax over ``{existing centroids} + {spawn}`` at the
configured temperature. For the offline systems the analogous quantity is a *post
hoc* similarity to the assigned cluster centroid — it was not available at
decision time because there was no decision time — and that asymmetry is stated
wherever the offline calibration numbers appear rather than being glossed.

Two exclusions, both counted rather than silent: non-speech windows have no
correct speaker, and overlapped windows cannot be right for a single-label
system. ``n_excluded`` is written to the table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..engine import evaluate, pooled_calibration
from ..metrics.calibration import (
    apply_temperature,
    area_under_risk_coverage,
    evaluate_calibration,
    fit_temperature,
    risk_coverage_curve,
)
from .common import dev_split, load_or_train, test_split, write_table

#: Coverage levels reported as concrete abstention operating points.
COVERAGE_POINTS = (1.0, 0.9, 0.8, 0.7, 0.5)


def run_calibration(
    cfg: Config,
    seeds: tuple[int, ...] = (0, 1, 2),
    methods: tuple[str, ...] = ("online", "naive_online", "offline_ahc"),
    out_dir: str | Path = "results/tables",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calibration and selective-prediction metrics per ``(method, seed)``.

    The temperature is fitted per ``(method, seed)`` on that seed's dev split, so
    the recalibrated numbers never touch test data. Fitting one shared
    temperature across seeds would be a defensible alternative but would mix
    information across runs and make the seed sd meaningless.
    """
    rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for seed in seeds:
        model = load_or_train(cfg, seed, verbose=verbose)
        dev = dev_split(cfg, seed)
        test = test_split(cfg, seed)
        for method in methods:
            dev_res = evaluate(model, cfg, dev, method=method,
                               latency_budget_ms=cfg.diarizer.latency_budget_ms)
            _, dev_margin, dev_correct, _ = pooled_calibration(dev_res)
            temperature = fit_temperature(dev_margin, dev_correct)

            test_res = evaluate(model, cfg, test, method=method,
                                latency_budget_ms=cfg.diarizer.latency_budget_ms)
            conf, margin, correct, excluded = pooled_calibration(test_res)
            raw = evaluate_calibration(conf, correct, cfg.eval.n_calibration_bins, excluded)
            scaled_conf = apply_temperature(margin, temperature)
            scaled = evaluate_calibration(
                scaled_conf, correct, cfg.eval.n_calibration_bins, excluded
            )

            row: dict[str, object] = {"method": method, "seed": seed,
                                      "temperature": float(temperature)}
            row.update({f"raw_{k}": v for k, v in raw.to_dict().items()})
            row.update({f"cal_{k}": v for k, v in scaled.to_dict().items()})
            row.update(_coverage_points(conf, correct))
            rows.append(row)

            cov, risk = risk_coverage_curve(conf, correct)
            if cov.size:
                # Subsample to ~200 points: the full curve is one row per
                # decision, which is 5000+ rows per method and unreadable.
                idx = np.unique(np.linspace(0, cov.size - 1, min(200, cov.size)).astype(int))
                for i in idx:
                    curves.append({"method": method, "seed": seed,
                                   "coverage": float(cov[i]), "risk": float(risk[i])})
            if verbose:
                print(
                    f"  {method:16s} seed {seed}: turn acc {raw.accuracy:.4f}  "
                    f"ECE {raw.ece:.4f} -> {scaled.ece:.4f} (T={temperature:.2f})  "
                    f"AUROC {raw.error_detection_auroc:.4f}  "
                    f"AURC {raw.aurc:.4f} (oracle {raw.aurc_oracle:.4f})"
                )

    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "calibration.csv", df)
    write_table(Path(out_dir) / "risk_coverage.csv", pd.DataFrame(curves))
    return df


def _coverage_points(conf: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Error rate over the most-confident fraction, at :data:`COVERAGE_POINTS`.

    This is the number a deployment actually asks for: "if I send the least
    confident 20% of turns to the offline pass, how wrong is the rest?"
    """
    out: dict[str, float] = {}
    cov, risk = risk_coverage_curve(conf, correct)
    for c in COVERAGE_POINTS:
        if cov.size == 0:
            out[f"risk_at_coverage_{int(c * 100)}"] = float("nan")
            continue
        k = max(0, min(cov.size - 1, int(round(c * cov.size)) - 1))
        out[f"risk_at_coverage_{int(c * 100)}"] = float(risk[k])
    out["aurc"] = area_under_risk_coverage(conf, correct)
    return out


def summarise_calibration(
    table: pd.DataFrame,
    out_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Mean and across-seed sd of every calibration column, per method."""
    from ..metrics.stats import noise_scale_from_seeds

    numeric = [c for c in table.columns if c not in ("method", "seed")]
    rows = []
    for method, sub in table.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "n_seeds": int(sub["seed"].nunique())}
        for col in numeric:
            vals = np.asarray(sub[col], dtype=np.float64)
            row[col] = float(np.nanmean(vals)) if vals.size else float("nan")
            row[f"{col}_seed_sd"] = (
                float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
            )
            row[f"{col}_noise_scale"] = noise_scale_from_seeds(vals)
        rows.append(row)
    df = pd.DataFrame(rows)
    write_table(Path(out_dir) / "calibration_summary.csv", df)
    return df
