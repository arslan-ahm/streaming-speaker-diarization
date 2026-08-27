"""Shared fixtures. Everything here is deliberately tiny so the suite stays fast.

No fixture trains a model beyond a handful of steps. The tests assert *behaviour*
— invariants, closed forms, causality, edge cases — and those do not need a
converged embedder. The two tests that do need training are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from streamdiar.config import Config, DataConfig, EmbedderConfig, TrainConfig
from streamdiar.data.generator import feature_statistics, generate_recording, generate_split
from streamdiar.engine.train import TrainedModel
from streamdiar.models.embedder import build_embedder

torch.set_num_threads(1)


@pytest.fixture(scope="session", autouse=True)
def _threads() -> None:
    """Other agents share this machine; never let a test spawn a thread pool."""
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_data_cfg() -> DataConfig:
    """A 4-second, 12-channel world with a small speaker pool."""
    return DataConfig(
        n_features=12,
        frame_rate=100,
        duration_s=4.0,
        n_speakers_range=(2, 3),
        turn_median_s=0.8,
        n_train_speakers=16,
        n_test_speakers=8,
        n_dev_recordings=2,
        n_test_recordings=3,
        n_phones=8,
    )


@pytest.fixture(scope="session")
def tiny_cfg(tiny_data_cfg: DataConfig) -> Config:
    """A full config sized for tests: 250 ms hop, 500 ms window, 250 ms budget."""
    cfg = Config(name="test", seed=0, data=tiny_data_cfg)
    cfg.embedder = EmbedderConfig(channels=16, dilations=(1, 2), embed_dim=8, dropout=0.0)
    cfg.train = TrainConfig(
        steps=3, n_speakers_per_batch=4, n_segments_per_speaker=2, segment_frames=40,
        log_every=1, threads=1,
    )
    cfg.diarizer.window_ms = 500.0
    cfg.diarizer.hop_ms = 250.0
    cfg.diarizer.latency_budget_ms = 250.0
    cfg.diarizer.max_speakers = 6
    return cfg


@pytest.fixture(scope="session")
def feature_stats(tiny_data_cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    return feature_statistics(tiny_data_cfg, 0, n_segments=8)


@pytest.fixture(scope="session")
def untrained_model(tiny_cfg: Config, feature_stats) -> TrainedModel:
    """An untrained embedder with real feature statistics and a plausible VAD threshold.

    Untrained is fine and is the point: causality, shape, determinism and
    edge-case behaviour are properties of the architecture, not of the weights.
    A test that needed converged weights to pass would be testing the data.
    """
    mean, sd = feature_stats
    embedder = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
    embedder.eval()
    return TrainedModel(
        embedder=embedder,
        feature_mean=mean,
        feature_sd=sd,
        vad_threshold=-0.55,
        seed=0,
        n_parameters=sum(p.numel() for p in embedder.parameters()),
    )


@pytest.fixture(scope="session")
def tiny_recordings(tiny_data_cfg: DataConfig):
    return generate_split(tiny_data_cfg, 0, "test")


@pytest.fixture(scope="session")
def one_recording(tiny_data_cfg: DataConfig):
    return generate_recording(tiny_data_cfg, 0, 0, pool="test")


@pytest.fixture
def two_speaker_reference() -> np.ndarray:
    """``(10, 2)`` reference: speaker 0 on frames 0-4, speaker 1 on frames 5-9."""
    ref = np.zeros((10, 2), dtype=bool)
    ref[:5, 0] = True
    ref[5:, 1] = True
    return ref


@pytest.fixture
def overlapping_reference() -> np.ndarray:
    """``(10, 2)``: speaker 0 on 0-5, speaker 1 on 4-9, so frames 4 and 5 overlap."""
    ref = np.zeros((10, 2), dtype=bool)
    ref[:6, 0] = True
    ref[4:, 1] = True
    return ref


def unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows; used by the clustering tests to build clean inputs."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@pytest.fixture
def three_clusters() -> tuple[np.ndarray, np.ndarray]:
    """240 unit vectors in 3 well-separated clusters, with their true labels."""
    rng = np.random.default_rng(0)
    centres = unit_rows(rng.normal(size=(3, 16)))
    pts = np.concatenate([c + 0.10 * rng.normal(size=(80, 16)) for c in centres])
    return unit_rows(pts), np.repeat([0, 1, 2], 80)
