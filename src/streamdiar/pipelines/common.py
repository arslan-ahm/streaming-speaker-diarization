"""Shared plumbing for the experiment pipelines.

Two things here exist because of specific failure modes rather than tidiness:

**Checkpoint reuse.** :func:`load_or_train` checks for an existing checkpoint
before training. The full experiment matrix trains four embedders and then
evaluates them under a dozen configurations; without reuse, a re-run of one
ablation would retrain from scratch, and a sweep interrupted at budget 6 of 8
would throw away all six. The build brief is explicit: check whether the output
already landed before re-running.

**Incremental CSV append.** :func:`append_rows` writes a header on first call and
appends afterwards, with LF endings. The latency sweep is chunked one invocation
per budget precisely so that a kill costs one budget rather than the whole curve,
and that only works if each budget's rows are on disk before the next starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.generator import Recording, generate_split
from ..engine import (
    RecordingResult,
    TrainedModel,
    fit_vad,
    load_model,
    save_model,
    train_embedder,
)


def checkpoint_path(cfg: Config, seed: int, tag: str | None = None) -> Path:
    """Where a trained model for ``(cfg.name or tag, seed)`` lives."""
    name = tag or cfg.name
    return Path("checkpoints") / f"{name}_seed{seed}.pt"


def load_or_train(
    cfg: Config,
    seed: int,
    tag: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> TrainedModel:
    """Return a trained model, reusing a checkpoint when one exists.

    The dev split used to fit the VAD threshold is generated from the *same* seed
    as the training run, so a reused checkpoint carries the same threshold it was
    saved with and no evaluation silently changes underneath a cached model.
    """
    path = checkpoint_path(cfg, seed, tag)
    if path.exists() and not force:
        if verbose:
            print(f"  reusing checkpoint {path}")
        return load_model(path, cfg)
    if verbose:
        print(f"  training {path.stem} ({cfg.train.steps} steps)...", flush=True)
    model = train_embedder(cfg, seed=seed)
    fit_vad(model, cfg, generate_split(cfg.data, seed, "dev"))
    save_model(model, path)
    return model


def dev_split(cfg: Config, seed: int) -> list[Recording]:
    return generate_split(cfg.data, seed, "dev")


def test_split(cfg: Config, seed: int) -> list[Recording]:
    """The evaluation set.

    Note that the test *recordings* depend on the seed as well as the model. That
    is deliberate for the seed study — it varies the whole pipeline, which is the
    honest run-to-run scale — and it means a per-recording paired test is only
    valid *within* one seed. Every paired comparison in this repository is run
    within a seed for that reason, and the caveat is restated in
    ``docs/RESULTS.md``.
    """
    return generate_split(cfg.data, seed, "test")


def rows_from_results(
    results: list[RecordingResult], **extra: Any
) -> list[dict[str, Any]]:
    """One flat dict per recording, ready for a CSV row."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "recording": r.recording,
            "method": r.method,
            "latency_budget_ms": r.latency_budget_ms,
        }
        row.update(extra)
        row.update({k: v for k, v in r.metrics.items()})
        rows.append(row)
    return rows


def append_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Append rows to a CSV, writing the header only when creating the file.

    Columns are unioned with whatever is already on disk, so a later chunk that
    reports an extra metric does not silently shift every column. LF endings
    throughout, per the build standard.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(p, index=False, lineterminator="\n")
    return p


def write_table(path: str | Path, rows: list[dict[str, Any]] | pd.DataFrame) -> Path:
    """Overwrite a CSV table with LF endings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(p, index=False, lineterminator="\n")
    return p


#: The metric family that every statistical comparison corrects across with
#: Holm-Bonferroni. Fixed here rather than per-experiment so the family size
#: cannot be quietly reduced to make a p-value survive.
METRIC_FAMILY = ("der", "confusion", "jer", "turn_accuracy", "speaker_count_error")

#: Metrics where lower is better, for rendering the sign of a delta correctly.
LOWER_IS_BETTER = {
    "der", "miss", "false_alarm", "confusion", "jer", "der_no_overlap",
    "der_collar250ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms",
    "rtf", "ece", "ace", "mce", "brier", "nll", "aurc",
}


def finite_mean(values: np.ndarray) -> tuple[float, int]:
    """``(mean over finite entries, count of them)``. NaN mean for an empty set."""
    v = np.asarray(values, dtype=np.float64)
    f = v[np.isfinite(v)]
    return (float(f.mean()) if f.size else float("nan")), int(f.size)
