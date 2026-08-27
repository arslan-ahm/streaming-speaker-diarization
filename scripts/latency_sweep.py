"""The headline experiment: one latency budget per invocation, appended to a shared CSV.

    # one cell
    uv run python scripts/latency_sweep.py --budget-ms 500 --seed 0
    # every budget for one seed
    uv run python scripts/latency_sweep.py --all-budgets --seed 0
    # the offline reference line
    uv run python scripts/latency_sweep.py --offline --seed 0

Chunked deliberately. A sweep of 9 budgets x 3 seeds is the longest job in the
repository, and a crash partway through must cost one cell rather than the whole
curve, so every cell appends its per-recording rows to
``results/tables/latency_sweep.csv`` before the next begins, and a cell already
present in that CSV is skipped rather than recomputed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config  # noqa: E402
from streamdiar.pipelines.latency import (  # noqa: E402
    DEFAULT_BUDGETS_MS,
    OFFLINE_METHODS,
    run_latency_point,
    run_offline_reference,
    summarise_curve,
)
from streamdiar.utils.seeding import limit_threads  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-ms", type=float, default=None,
                    help="Run a single budget (milliseconds).")
    ap.add_argument("--all-budgets", action="store_true",
                    help="Run every budget in DEFAULT_BUDGETS_MS for this seed.")
    ap.add_argument("--offline", action="store_true",
                    help="Run the offline reference methods for this seed.")
    ap.add_argument("--summarise", action="store_true",
                    help="Collapse the CSV to results/tables/latency_curve.csv and exit.")
    ap.add_argument("--csv", default="results/tables/latency_sweep.csv")
    ap.add_argument("--force", action="store_true", help="Recompute cells already present.")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg.train.threads)

    if args.summarise:
        import pandas as pd

        table = pd.read_csv(args.csv)
        curve = summarise_curve(table)
        print(curve.to_string(index=False))
        return 0

    if args.offline:
        for method in OFFLINE_METHODS:
            run_offline_reference(cfg, args.seed, args.csv, method, args.force)
        return 0

    budgets = DEFAULT_BUDGETS_MS if args.all_budgets else (
        (args.budget_ms,) if args.budget_ms is not None else (cfg.diarizer.latency_budget_ms,)
    )
    for budget in budgets:
        run_latency_point(cfg, args.seed, float(budget), args.csv, "online", args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
