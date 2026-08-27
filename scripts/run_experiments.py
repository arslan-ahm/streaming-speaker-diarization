"""Method comparison, ablations, calibration and efficiency. One stage per invocation.

    uv run python scripts/run_experiments.py --stage comparison
    uv run python scripts/run_experiments.py --stage ablations
    uv run python scripts/run_experiments.py --stage calibration
    uv run python scripts/run_experiments.py --stage efficiency
    uv run python scripts/run_experiments.py --stage all

Each stage writes its own CSVs under ``results/tables/`` and is independently
re-runnable; the comparison and ablation stages reuse checkpoints rather than
retraining. Staged rather than monolithic for the same reason the latency sweep is
chunked: a kill should cost one stage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config  # noqa: E402
from streamdiar.pipelines.ablation import (  # noqa: E402
    ablation_tests,
    run_ablations,
    summarise_ablations,
)
from streamdiar.pipelines.calibration import run_calibration, summarise_calibration  # noqa: E402
from streamdiar.pipelines.comparison import (  # noqa: E402
    run_comparison,
    statistical_tests,
    summarise,
)
from streamdiar.pipelines.efficiency import run_efficiency, scaling_study  # noqa: E402
from streamdiar.utils.seeding import limit_threads  # noqa: E402

STAGES = ("comparison", "ablations", "calibration", "efficiency")


def stage_comparison(cfg, seeds):
    table = run_comparison(cfg, seeds=seeds)
    summary = summarise(table)
    tests = statistical_tests(table, summary)
    print("\n--- method summary (DER, mean over all seeds) ---")
    cols = [c for c in ["variant", "der", "der_seed_sd", "der_noise_scale", "confusion",
                        "jer", "latency_median_ms", "n_speakers_pred"] if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print("\n--- statistical tests vs online (DER only) ---")
    sub = tests[tests["metric"] == "der"]
    print(sub[["name_b", "mean_a", "mean_b", "delta", "ci_lower", "ci_upper",
               "p_adjusted", "noise_ratio", "verdict"]].to_string(index=False))
    return table


def stage_ablations(cfg, seeds):
    table = run_ablations(cfg, seeds=seeds)
    summary = summarise_ablations(table)
    tests = ablation_tests(table, summary)
    print("\n--- ablation summary ---")
    cols = [c for c in ["ablation", "der", "der_seed_sd", "der_noise_scale", "confusion",
                        "n_speakers_pred", "extra_mean_query_size"] if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print("\n--- ablation tests vs full (DER only) ---")
    sub = tests[tests["metric"] == "der"]
    print(sub[["name_b", "mean_a", "mean_b", "delta", "p_adjusted", "noise_ratio",
               "verdict"]].to_string(index=False))
    return table


def stage_calibration(cfg, seeds):
    table = run_calibration(cfg, seeds=seeds)
    summary = summarise_calibration(table)
    cols = [c for c in ["method", "raw_accuracy", "raw_ece", "cal_ece", "raw_ace",
                        "raw_brier", "raw_error_detection_auroc", "raw_aurc",
                        "aurc", "risk_at_coverage_100", "risk_at_coverage_80",
                        "risk_at_coverage_50"] if c in summary.columns]
    print("\n--- calibration summary ---")
    print(summary[cols].to_string(index=False))
    return table


def stage_efficiency(cfg, seed):
    eff = run_efficiency(cfg, seed=seed)
    scal = scaling_study(cfg, seed=seed)
    print("\n--- efficiency ---")
    print(eff[["method", "total_median_ms", "total_iqr_ms", "cluster_ms", "rtf",
               "emission_latency_median_ms", "emission_latency_p95_ms"]].to_string(index=False))
    print("\n--- clustering scaling ---")
    print(scal.to_string(index=False))
    return eff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage", default="all", choices=(*STAGES, "all"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg.train.threads)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        t0 = time.perf_counter()
        print(f"\n================ stage: {stage} ================", flush=True)
        if stage == "comparison":
            stage_comparison(cfg, seeds)
        elif stage == "ablations":
            stage_ablations(cfg, seeds)
        elif stage == "calibration":
            stage_calibration(cfg, seeds)
        elif stage == "efficiency":
            stage_efficiency(cfg, seeds[0])
        print(f"[{stage} done in {time.perf_counter() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
