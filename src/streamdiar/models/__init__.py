"""Models: the causal embedder, the VAD, and the four diarization mechanisms."""

from __future__ import annotations

from .embedder import (
    CausalConvBlock,
    SpeakerEmbedder,
    build_embedder,
    region_bounds,
    window_grid,
)
from .offline import (
    agglomerative_average_linkage,
    cosine_distance_matrix,
    diarize_offline,
    estimate_n_speakers_eigengap,
    kmeans,
    refine_affinity,
    spectral_cluster,
)
from .online import (
    DiarizationOutput,
    SpeakerTracker,
    diarize_naive_online,
    diarize_online,
)
from .vad import energy_vad, fit_vad_threshold, frame_energy, windows_are_speech

__all__ = [
    "CausalConvBlock",
    "DiarizationOutput",
    "SpeakerEmbedder",
    "SpeakerTracker",
    "agglomerative_average_linkage",
    "build_embedder",
    "cosine_distance_matrix",
    "diarize_naive_online",
    "diarize_offline",
    "diarize_online",
    "energy_vad",
    "estimate_n_speakers_eigengap",
    "fit_vad_threshold",
    "frame_energy",
    "kmeans",
    "refine_affinity",
    "region_bounds",
    "spectral_cluster",
    "window_grid",
    "windows_are_speech",
]
