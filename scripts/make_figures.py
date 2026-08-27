"""Render every figure from the committed CSVs. No numbers are recomputed here.

    uv run python scripts/make_figures.py

Each panel is skipped with a printed note if its CSV is missing, so this is safe
to run against a partially completed experiment matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config  # noqa: E402
from streamdiar.pipelines import figures as F  # noqa: E402
from streamdiar.utils.seeding import limit_threads  # noqa: E402

TABLES = Path("results/tables")


def _load(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    if not path.exists():
        print(f"  [skip] {path} missing")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [skip] {path} is empty")
        return None
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    limit_threads(cfg.train.threads)
    written: list[Path] = []

    curve = _load("latency_curve.csv")
    if curve is not None:
        written.append(F.der_vs_latency(curve))

    summary = _load("method_summary.csv")
    if summary is not None:
        written.append(F.der_decomposition(summary))

    abl_summary = _load("ablation_summary.csv")
    abl_tests = _load("ablation_tests.csv")
    if abl_summary is not None and abl_tests is not None:
        written.append(F.ablation_deltas(abl_summary, abl_tests))

    calib = _load("calibration.csv")
    rc = _load("risk_coverage.csv")
    if calib is not None and rc is not None:
        written.append(F.calibration_panels(calib, rc))

    eff = _load("efficiency.csv")
    scal = _load("scaling.csv")
    if eff is not None and scal is not None:
        written.append(F.efficiency_panels(eff, scal))

    # The two figures that need the data rather than a table.
    from streamdiar.data.generator import generate_recording
    from streamdiar.pipelines.common import dev_split, load_or_train
    from streamdiar.pipelines.tuning import precompute, similarity_distributions

    written.append(F.data_overview(generate_recording(cfg.data, args.seed, 1000, "test")))

    ckpt = Path("checkpoints") / f"{cfg.name}_seed{args.seed}.pt"
    if ckpt.exists():
        model = load_or_train(cfg, args.seed, verbose=False)
        cached = precompute(model, cfg, dev_split(cfg, args.seed))
        same, diff = similarity_distributions(cached, cfg.frames_per_hop)
        written.append(F.similarity_histogram(same, diff))
    else:
        print(f"  [skip] {ckpt} missing, no separability histogram")

    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
