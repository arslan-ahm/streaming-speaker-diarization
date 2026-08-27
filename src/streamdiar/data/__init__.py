"""Data: the procedural conversation generator and an optional real-dataset loader."""

from __future__ import annotations

from .generator import (
    Recording,
    Turn,
    World,
    dataset_stats,
    feature_statistics,
    generate_recording,
    generate_split,
    get_world,
    sample_segment,
    sample_training_batch,
    sample_turns,
)

__all__ = [
    "Recording",
    "Turn",
    "World",
    "dataset_stats",
    "feature_statistics",
    "generate_recording",
    "generate_split",
    "get_world",
    "sample_segment",
    "sample_training_batch",
    "sample_turns",
]
