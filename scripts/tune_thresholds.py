"""Tune the clustering thresholds on the dev split (seed 0) and report the grids.

    uv run python scripts/tune_thresholds.py --config configs/base.yaml

Writes results/tables/threshold_tuning_*.csv and prints the values to freeze into
configs/base.yaml. Run once; the chosen values are then held fixed for every seed
and every experiment so that seeds 1+ are held out with respect to hyperparameter
choice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config  # noqa: E402
from streamdiar.pipelines.common import dev_split, load_or_train  # noqa: E402
from streamdiar.pipelines.tuning import (  # noqa: E402
    precompute,
    similarity_distributions,
    tune_offline_ahc,
    tune_offline_spectral,
    tune_online,
    write_tuning_tables,
)
from streamdiar.utils.seeding import limit_threads  # noqa: E402


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

    same, diff = similarity_distributions(cached, cfg.frames_per_hop)
    pct = [5, 25, 50, 75, 95]
    print(f"dev cosine similarity, same speaker      : median {np.median(same):.3f}  "
          f"pct{pct} {np.round(np.percentile(same, pct), 3).tolist()}")
    print(f"dev cosine similarity, different speakers: median {np.median(diff):.3f}  "
          f"pct{pct} {np.round(np.percentile(diff, pct), 3).tolist()}")

    print("\ntuning online (spawn_threshold, micro_cluster_threshold)...")
    best_online, tbl_online = tune_online(cfg, cached)
    print(tbl_online.head(6).to_string(index=False))

    print("\ntuning offline_ahc threshold...")
    best_ahc, tbl_ahc = tune_offline_ahc(cfg, cached)
    print(tbl_ahc.to_string(index=False))

    print("\ntuning offline_spectral percentile...")
    best_spec, tbl_spec = tune_offline_spectral(cfg, cached)
    print(tbl_spec.to_string(index=False))

    write_tuning_tables(
        {"online": tbl_online, "offline_ahc": tbl_ahc, "offline_spectral": tbl_spec}
    )
    print("\n--- freeze these into configs/base.yaml (tuned on dev, seed 0) ---")
    print(f"  micro_cluster_threshold: {best_online['micro_cluster_threshold']}")
    print(f"  spawn_threshold: {best_online['spawn_threshold']}")
    print(f"  offline_ahc_threshold: {best_ahc['offline_ahc_threshold']}")
    print(f"  offline_spectral_percentile: {best_spec['offline_spectral_percentile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
