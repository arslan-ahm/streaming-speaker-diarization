"""Print every headline number from the committed CSVs, in one place.

    uv run python scripts/report.py

Exists so that the numbers in README.md and docs/*.md can be cross-checked
against their source in a single pass rather than by eye. The build standard is
explicit that every number in every document must be traceable to a committed
artifact; this is the tool that makes that check cheap, and it is why it prints
the source filename beside every block.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("results/tables")


def _load(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    if not p.exists():
        print(f"[missing] {p}")
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def section(title: str, source: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n  source: results/tables/{source}\n{'=' * 78}")


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 80)

    curve = _load("latency_curve.csv")
    if curve is not None:
        section("DER vs latency budget (the headline)", "latency_curve.csv")
        online = curve[(curve.method == "online") & np.isfinite(curve.latency_budget_ms)]
        cols = ["latency_budget_ms", "der", "der_noise_scale", "confusion", "jer",
                "n_speakers_pred", "latency_median_ms", "extra_mean_query_size", "n_seeds"]
        print(online[[c for c in cols if c in online.columns]]
              .sort_values("latency_budget_ms").round(4).to_string(index=False))
        der = online.sort_values("latency_budget_ms").der.to_numpy(float)
        ns = float(np.nanmax(online.der_noise_scale.to_numpy(float)))
        best_b = online.loc[online.der.idxmin(), "latency_budget_ms"]
        worst_b = online.loc[online.der.idxmax(), "latency_budget_ms"]
        print(f"\n  best budget       : {best_b:.0f} ms  (DER {der.min():.4f})")
        print(f"  worst budget      : {worst_b:.0f} ms  (DER {der.max():.4f})")
        print(f"  range across budgets: {der.max() - der.min():.4f}")
        print(f"  noise scale sqrt(2)*sd (max over budgets): {ns:.4f}")
        print(f"  range / noise scale : {(der.max() - der.min()) / ns:.2f}"
              "   <- below 1.0 means the whole curve is inside run-to-run noise")

        off = curve[curve.method.str.startswith("offline")]
        if len(off):
            print()
            print(off[["method", "der", "der_noise_scale", "confusion", "jer",
                       "n_speakers_pred", "latency_median_ms"]].round(4).to_string(index=False))

    summary = _load("method_summary.csv")
    if summary is not None:
        section("Method comparison (3 seeds)", "method_summary.csv")
        cols = ["variant", "der", "der_seed_sd", "der_noise_scale", "miss", "false_alarm",
                "confusion", "jer", "n_speakers_pred", "latency_median_ms", "rtf",
                "overlap_miss_floor"]
        print(summary[[c for c in cols if c in summary.columns]].round(4).to_string(index=False))

    tests = _load("statistical_tests.csv")
    if tests is not None:
        section("Statistical verdicts vs online", "statistical_tests.csv")
        for metric in ["der", "confusion", "jer", "turn_accuracy", "speaker_count_error"]:
            sub = tests[tests.metric == metric]
            if not len(sub):
                continue
            print(f"\n-- {metric}")
            print(sub[["name_b", "mean_a", "mean_b", "delta", "ci_lower", "ci_upper",
                       "p_value", "p_adjusted", "noise_ratio", "verdict"]]
                  .round(5).to_string(index=False))

    abl = _load("ablation_summary.csv")
    if abl is not None:
        section("Ablations", "ablation_summary.csv")
        cols = ["ablation", "der", "der_seed_sd", "der_noise_scale", "confusion",
                "n_speakers_pred", "extra_mean_query_size"]
        print(abl[[c for c in cols if c in abl.columns]].round(4).to_string(index=False))
    abl_t = _load("ablation_tests.csv")
    if abl_t is not None:
        section("Ablation verdicts vs full", "ablation_tests.csv")
        sub = abl_t[abl_t.metric == "der"]
        print(sub[["name_b", "mean_a", "mean_b", "delta", "ci_lower", "ci_upper",
                   "p_adjusted", "noise_ratio", "verdict"]].round(5).to_string(index=False))

    cal = _load("calibration_summary.csv")
    if cal is not None:
        section("Calibration and abstention", "calibration_summary.csv")
        cols = ["method", "temperature", "raw_accuracy", "raw_mean_confidence",
                "raw_overconfidence", "raw_ece", "raw_ece_noise_scale", "cal_ece",
                "raw_ace", "raw_mce", "raw_brier", "raw_nll",
                "raw_error_detection_auroc", "raw_error_detection_auroc_noise_scale",
                "raw_aurc", "raw_aurc_oracle", "raw_n", "raw_n_excluded",
                "risk_at_coverage_100", "risk_at_coverage_80", "risk_at_coverage_50"]
        print(cal[[c for c in cols if c in cal.columns]].round(4).to_string(index=False))

    eff = _load("efficiency.csv")
    if eff is not None:
        section("Efficiency", "efficiency.csv")
        cols = ["method", "n_parameters", "total_median_ms", "total_iqr_ms", "embed_ms",
                "cluster_ms", "cluster_ms_per_decision", "rtf", "audio_s", "n_decisions",
                "emission_latency_median_ms", "emission_latency_p95_ms",
                "emission_latency_max_ms", "warmup", "repeats"]
        print(eff[[c for c in cols if c in eff.columns]].round(4).to_string(index=False))
        try:
            on = eff[eff.method == "online"].iloc[0]
            ahc = eff[eff.method == "offline_ahc"].iloc[0]
            factor = ahc.emission_latency_median_ms / max(on.emission_latency_median_ms, 1e-9)
            print(f"\n  emission-latency factor, offline_ahc / online: {factor:.1f}x")
            print(f"  clustering wall-clock factor, offline_ahc / online: "
                  f"{ahc.cluster_ms / max(on.cluster_ms, 1e-9):.2f}x")
        except (IndexError, KeyError):
            pass

    scal = _load("scaling.csv")
    if scal is not None:
        section("Clustering cost vs recording length", "scaling.csv")
        piv = scal.pivot(index="n_speech_windows", columns="method",
                         values="cluster_median_ms")
        print(piv.round(2).to_string())
        print("\n  ms per window:")
        print(scal.pivot(index="n_speech_windows", columns="method",
                         values="ms_per_window").round(4).to_string())

    mom = _load("momentum_tuning.csv")
    if mom is not None:
        section("Dev-split momentum sweep (why 0.10, not 0.35)", "momentum_tuning.csv")
        print(mom.pivot(index="centroid_momentum", columns="latency_budget_ms",
                        values="der").round(4).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
