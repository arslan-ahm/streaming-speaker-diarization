"""Tune ``centroid_momentum`` on the dev split, at several latency budgets.

    uv run python scripts/tune_momentum.py

Why this exists, stated plainly. The first experiment pass tuned the three
clustering *thresholds* on dev but left ``centroid_momentum`` at a guessed 0.35.
The comparison then showed the naive baseline (which uses an exact running mean)
beating the proposed system at its own optimum, and the online system at ``B = 0``
losing badly (DER 0.378 vs 0.292). That is a symptom of an untuned parameter, not
of a broken mechanism, and leaving it untuned would mean comparing a tuned
baseline against an untuned proposal.

So the momentum is tuned here on **dev, seed 0 only**, by the same rule used for
the thresholds: minimum dev DER, near-ties within 1% relative DER broken on the
smallest absolute speaker-count bias. It is swept jointly with the budget because
the two interact -- a sticky centroid needs less lookahead and vice versa -- and
reporting only the marginal best of each would hide that.

Writes results/tables/momentum_tuning.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config, replace  # noqa: E402
from streamdiar.models.online import diarize_online  # noqa: E402
from streamdiar.pipelines.common import dev_split, load_or_train, write_table  # noqa: E402
from streamdiar.pipelines.tuning import _score_cached, precompute, select_best  # noqa: E402
from streamdiar.utils.seeding import limit_threads  # noqa: E402

MOMENTUM_GRID = (0.05, 0.10, 0.20, 0.35, 0.60, 1.00)
BUDGET_GRID_MS = (0.0, 250.0, 500.0, 1000.0, 2000.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg.train.threads)
    model = load_or_train(cfg, args.seed)
    cached = precompute(model, cfg, dev_split(cfg, args.seed))
    hop = cfg.frames_per_hop

    rows = []
    for momentum in MOMENTUM_GRID:
        for budget in BUDGET_GRID_MS:
            dcfg = replace(cfg.diarizer, centroid_momentum=momentum)
            windows = int(budget // cfg.diarizer.hop_ms)
            outputs = [
                diarize_online(c.embeddings, c.starts, c.ends, c.is_speech, dcfg,
                               windows, c.recording.n_frames, hop)
                for c in cached
            ]
            scores = _score_cached(cfg, cached, outputs)
            rows.append({"centroid_momentum": momentum, "latency_budget_ms": budget,
                         **scores})

    df = pd.DataFrame(rows)
    write_table("results/tables/momentum_tuning.csv", df)

    pivot = df.pivot(index="centroid_momentum", columns="latency_budget_ms", values="der")
    print("\ndev DER by (centroid_momentum, latency budget ms):")
    print(pivot.round(4).to_string())

    best = select_best(df.sort_values("der"),
                       ("centroid_momentum", "latency_budget_ms"))
    print("\n--- freeze into configs/base.yaml (tuned on dev, seed 0) ---")
    print(f"  centroid_momentum: {best['centroid_momentum']}")
    print(f"  (best joint budget on dev was {best['latency_budget_ms']:.0f} ms, "
          f"dev DER {best['der']:.4f}, speaker-count bias "
          f"{best['speaker_count_bias']:+.2f}, {int(best['n_near_ties'])} near-ties)")

    # The marginal best momentum at the shipped default budget, for reference.
    at_default = df[df.latency_budget_ms == cfg.diarizer.latency_budget_ms]
    if len(at_default):
        row = at_default.sort_values("der").iloc[0]
        print(f"  at the shipped {cfg.diarizer.latency_budget_ms:.0f} ms budget the best "
              f"momentum is {row['centroid_momentum']} (dev DER {row['der']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
