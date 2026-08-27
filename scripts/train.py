"""Train the causal speaker embedder and fit the VAD threshold on dev.

    uv run python scripts/train.py --config configs/base.yaml --seed 0

Writes ``results/runs/<name>_seed<seed>/{config.yaml,history.jsonl,summary.json}``
and a checkpoint under ``checkpoints/``. Thin by design: all logic lives in
``streamdiar.engine`` so the tests can reach it without a subprocess.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamdiar.config import load_config, save_config  # noqa: E402
from streamdiar.data.generator import generate_split  # noqa: E402
from streamdiar.engine import fit_vad, save_model, train_embedder  # noqa: E402
from streamdiar.utils.io import write_json  # noqa: E402
from streamdiar.utils.seeding import limit_threads  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override a config field, e.g. --set train.steps=100")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    limit_threads(cfg.train.threads)

    run = Path(cfg.out_root) / "runs" / f"{cfg.name}_seed{cfg.seed}"
    run.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")

    t0 = time.perf_counter()
    model = train_embedder(cfg, seed=cfg.seed, history_path=str(run / "history.jsonl"),
                           progress=not args.quiet)
    dev = generate_split(cfg.data, cfg.seed, "dev")
    fit_vad(model, cfg, dev)
    wall = time.perf_counter() - t0

    ckpt = Path(args.checkpoint or f"checkpoints/{cfg.name}_seed{cfg.seed}.pt")
    save_model(model, ckpt)

    summary = {
        "name": cfg.name,
        "seed": cfg.seed,
        "n_parameters": model.n_parameters,
        "train_seconds": model.train_seconds,
        "wall_seconds": wall,
        "final_loss": model.history[-1]["loss"] if model.history else None,
        "final_batch_accuracy": model.history[-1]["batch_accuracy"] if model.history else None,
        "vad_threshold": model.vad_threshold,
        "vad_frame_error": model.vad_frame_error,
        "checkpoint": str(ckpt),
    }
    write_json(run / "summary.json", summary)
    if not args.quiet:
        print(f"\ntrained {cfg.name} seed {cfg.seed} in {wall:.0f}s")
        print(f"  final loss {summary['final_loss']:.4f}  "
              f"batch acc {summary['final_batch_accuracy']:.3f}")
        print(f"  VAD threshold {model.vad_threshold:.3f} "
              f"(dev frame error {model.vad_frame_error:.4f})")
        print(f"  checkpoint -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
