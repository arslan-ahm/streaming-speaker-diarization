"""Models: the causal embedder and the four diarization mechanisms."""

from __future__ import annotations

from .embedder import (
    CausalConvBlock,
    SpeakerEmbedder,
    build_embedder,
    region_bounds,
    window_grid,
)

__all__ = [
    "CausalConvBlock",
    "SpeakerEmbedder",
    "build_embedder",
    "region_bounds",
    "window_grid",
]
