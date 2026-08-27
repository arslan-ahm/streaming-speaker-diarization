"""Checkpoint save/load for :class:`~streamdiar.engine.train.TrainedModel`.

Checkpoints are *not* committed (``.gitignore`` excludes ``*.pt``): they are
reproducible from a seed in about four minutes, so committing 200 KB of weights
would add nothing a reader could not regenerate. What *is* committed is the
config, the history JSONL and the per-recording CSVs — the evidence, not the
intermediate.

The one non-obvious requirement is that the feature statistics and the fitted VAD
threshold travel with the weights. They are learned quantities; a checkpoint that
restored only ``state_dict`` would load without error and score several DER points
worse, which is the worst kind of bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..models.embedder import build_embedder
from .train import TrainedModel


def save_model(model: TrainedModel, path: str | Path) -> Path:
    """Serialise weights, feature statistics, VAD threshold and provenance."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.embedder.state_dict(),
            "feature_mean": np.asarray(model.feature_mean),
            "feature_sd": np.asarray(model.feature_sd),
            "vad_threshold": float(model.vad_threshold),
            "vad_frame_error": float(model.vad_frame_error),
            "seed": int(model.seed),
            "history": model.history,
            "train_seconds": float(model.train_seconds),
            "n_parameters": int(model.n_parameters),
        },
        p,
    )
    return p


def load_model(path: str | Path, cfg: Config) -> TrainedModel:
    """Rebuild a :class:`TrainedModel` from ``path``.

    ``weights_only=False`` is required because the payload carries NumPy arrays
    and the history list. The file is one this repository wrote, in a directory it
    owns, so that is safe here — it would not be for an untrusted checkpoint.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    embedder = build_embedder(
        cfg.embedder, cfg.data.n_features, blob["feature_mean"], blob["feature_sd"]
    )
    embedder.load_state_dict(blob["state_dict"])
    embedder.eval()
    return TrainedModel(
        embedder=embedder,
        feature_mean=blob["feature_mean"],
        feature_sd=blob["feature_sd"],
        vad_threshold=float(blob["vad_threshold"]),
        vad_frame_error=float(blob.get("vad_frame_error", float("nan"))),
        seed=int(blob.get("seed", 0)),
        history=list(blob.get("history", [])),
        train_seconds=float(blob.get("train_seconds", 0.0)),
        n_parameters=int(blob.get("n_parameters", 0)),
    )
