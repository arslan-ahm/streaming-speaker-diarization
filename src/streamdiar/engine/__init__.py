"""Engine: training, checkpointing, and the diarization/evaluation loop."""

from __future__ import annotations

from .checkpoint import load_model, save_model
from .diarize import (
    METHODS,
    RecordingResult,
    aggregate,
    evaluate,
    fit_vad,
    per_recording_metric,
    pooled_calibration,
    run_diarizer,
    score_output,
)
from .train import PrototypicalLoss, TrainedModel, train_embedder

__all__ = [
    "METHODS",
    "PrototypicalLoss",
    "RecordingResult",
    "TrainedModel",
    "aggregate",
    "evaluate",
    "fit_vad",
    "load_model",
    "per_recording_metric",
    "pooled_calibration",
    "run_diarizer",
    "save_model",
    "score_output",
    "train_embedder",
]
