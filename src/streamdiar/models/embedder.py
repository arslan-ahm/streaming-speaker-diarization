"""Causal dilated-convolution speaker embedder.

This is the front end for every method in the repository — online, naive online,
and the offline reference all consume embeddings from the *same* trained model,
which is what makes the DER-versus-latency curve a statement about the
clustering/assignment mechanism rather than about two different feature
extractors.

Causality is a structural property here, not a convention
---------------------------------------------------------
Three things in a normal speaker-embedding stack quietly look at the future, and
all three are removed:

1. **Convolution padding.** A ``Conv1d`` with ``padding=(k-1)*d//2`` is centred:
   frame ``t``'s output depends on frames up to ``t + (k-1)*d/2``. Here the input
   is left-padded by the full ``(k-1)*d`` and the convolution is unpadded, so
   output ``t`` depends on inputs ``<= t`` exactly. ``causal=False`` restores
   centred padding and is the non-causal ablation.
2. **Normalisation over time.** ``BatchNorm1d`` and ``GroupNorm`` on a
   ``(B, C, T)`` tensor both reduce over ``T``, so one late frame shifts every
   earlier activation. The norm used here is a ``LayerNorm`` over the *channel*
   axis at each timestep independently — no time reduction at all.
3. **Input feature standardisation.** Normalising a recording by its own mean and
   variance is the most common causality leak in diarization code, and it is
   invisible because it improves offline numbers. The per-channel statistics here
   are estimated once from *training* segments and frozen into buffers.

``tests/test_causality.py`` asserts the consequence directly: replace every frame
after ``t0`` with noise, and the embeddings of all windows ending at or before
``t0`` are **bit-identical**, not merely close.

Window pooling in O(1) per window
---------------------------------
Statistics pooling over a window needs the mean and standard deviation of the
frame features inside it. Computed naively that is O(window) per window and
O(T * window / hop) per recording. Prefix sums of the features and their squares
make each window O(C) regardless of window length, which is what lets the
streaming path claim per-frame cost independent of window size. The prefix sums
are accumulated in float64 because the float32 catastrophic cancellation in
``E[x^2] - E[x]^2`` is real at 6000 frames: it produced small negative variances
during development, which is why :meth:`SpeakerEmbedder.pool_windows` clamps
before the square root and why the clamp is documented rather than silent.

References
----------
Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling*, 2018 — the dilated causal TCN block.
Snyder et al., *X-Vectors: Robust DNN Embeddings for Speaker Recognition*,
ICASSP 2018 — statistics pooling to a fixed-size speaker embedding.
Wan et al., *Generalized End-to-End Loss for Speaker Verification*, ICASSP 2018.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import EmbedderConfig


class CausalConvBlock(nn.Module):
    """One dilated convolution block: channel-LayerNorm, conv, GELU, dropout, residual.

    The residual connection is what makes a four-block stack trainable in 900
    steps on a CPU; without it the same stack needed roughly three times the
    steps to reach the same training loss during development.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        #: Total padding the kernel needs; all of it on the left when causal.
        self.pad = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # A 1x1 projection only when the residual needs reshaping, so the
        # identity path stays exactly an identity in the common case.
        self.project = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, C_in) -> (B, T, C_out)``. Length is preserved."""
        h = self.norm(x).transpose(1, 2)  # (B, C, T)
        if self.causal:
            h = nn.functional.pad(h, (self.pad, 0))
        else:
            left = self.pad // 2
            h = nn.functional.pad(h, (left, self.pad - left))
        h = self.drop(self.act(self.conv(h)))
        res = x.transpose(1, 2)
        if self.project is not None:
            res = self.project(res)
        return (h + res).transpose(1, 2)


class SpeakerEmbedder(nn.Module):
    """Frame encoder plus statistics pooling to an L2-normalised speaker embedding.

    Args:
        cfg: Architecture switches, including ``causal``.
        n_features: Input feature dimension.
        feature_mean: ``(n_features,)`` per-channel mean from *training* data.
        feature_sd: ``(n_features,)`` per-channel sd from *training* data.

    The feature statistics are ``register_buffer``\\ s, so they travel with the
    checkpoint. A model loaded without them would standardise by zero-mean /
    unit-sd and silently lose several points of DER, which is exactly the kind of
    failure that looks like a bad idea rather than a bug.
    """

    def __init__(
        self,
        cfg: EmbedderConfig,
        n_features: int,
        feature_mean: np.ndarray | None = None,
        feature_sd: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = int(n_features)

        mean = np.zeros(n_features) if feature_mean is None else np.asarray(feature_mean)
        sd = np.ones(n_features) if feature_sd is None else np.asarray(feature_sd)
        self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_sd", torch.tensor(np.maximum(sd, 1e-3), dtype=torch.float32))

        blocks: list[nn.Module] = []
        in_ch = self.n_features
        for dilation in cfg.dilations:
            blocks.append(
                CausalConvBlock(
                    in_ch, cfg.channels, cfg.kernel_size, dilation, cfg.dropout, cfg.causal
                )
            )
            in_ch = cfg.channels
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(cfg.channels)

        pool_dim = cfg.channels * (2 if cfg.use_std_pooling else 1)
        self.head = nn.Linear(pool_dim, cfg.embed_dim)

    @property
    def receptive_field(self) -> int:
        """Frames of history each output frame depends on, inclusive of itself."""
        return 1 + sum((self.cfg.kernel_size - 1) * d for d in self.cfg.dilations)

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen training-set statistics. Causal by construction."""
        return (x - self.feature_mean) / self.feature_sd

    def frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, F) -> (B, T, C)`` frame encodings.

        When ``cfg.causal``, output frame ``t`` is a function of input frames
        ``0..t`` only. Nothing downstream re-reduces over time except the
        window pooling, which is bounded to a window that has already arrived.
        """
        h = self.standardise(x)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)

    # ------------------------------------------------------------------ #
    # Statistics pooling
    # ------------------------------------------------------------------ #
    def _prefix_sums(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exclusive prefix sums of ``frames`` and ``frames**2``, in float64.

        float64 is not caution for its own sake. With float32 and 6000 frames the
        cancellation in ``E[x^2] - E[x]^2`` produced variances of about ``-1e-4``
        during development — negative, so ``sqrt`` returned NaN and every
        downstream embedding became NaN. Widening the accumulator removes the
        cause; the clamp in :meth:`pool_windows` covers the residual.
        """
        f = frames.double()
        zero = torch.zeros(1, f.shape[-1], dtype=torch.float64, device=f.device)
        cs = torch.cat([zero, f.cumsum(0)], dim=0)
        css = torch.cat([zero, (f * f).cumsum(0)], dim=0)
        return cs, css

    def pool_windows(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """Pool ``(T, C)`` frame features over ``[start, end)`` windows to ``(N, D)``.

        O(C) per window regardless of window length, via the prefix sums. The
        returned embeddings are L2-normalised, so every similarity downstream is
        a cosine and the thresholds in :class:`~streamdiar.config.DiarizerConfig`
        are comparable across configurations.
        """
        if frames.ndim != 2:
            raise ValueError(f"frames must be (T, C), got {tuple(frames.shape)}")
        cs, css = self._prefix_sums(frames)
        s = starts.to(torch.long)
        e = ends.to(torch.long)
        n = (e - s).clamp(min=1).unsqueeze(-1).double()
        mean = (cs[e] - cs[s]) / n
        if self.cfg.use_std_pooling:
            var = (css[e] - css[s]) / n - mean * mean
            sd = var.clamp(min=1e-8).sqrt()
            pooled = torch.cat([mean, sd], dim=-1)
        else:
            pooled = mean
        emb = self.head(pooled.to(frames.dtype))
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, F) -> (B, embed_dim)``: one embedding per input segment.

        This is the training-time path — each item in the batch is a fixed-length
        single-speaker segment pooled over its whole extent. Inference uses
        :meth:`embed_recording`, which shares every parameter but pools over a
        sliding window instead.
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {tuple(x.shape)}")
        feats = self.frame_features(x)
        mean = feats.mean(dim=1)
        if self.cfg.use_std_pooling:
            sd = feats.var(dim=1, unbiased=False).clamp(min=1e-8).sqrt()
            pooled = torch.cat([mean, sd], dim=-1)
        else:
            pooled = mean
        return nn.functional.normalize(self.head(pooled), dim=-1)

    @torch.no_grad()
    def embed_recording(
        self,
        features: np.ndarray,
        window_frames: int,
        hop_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed a whole recording on the streaming window grid.

        Returns:
            ``(embeddings (K, D) float32, starts (K,), ends (K,))``. Row ``k`` is
            the embedding of frames ``[starts[k], ends[k])``, and ``ends[k]`` is
            the frame at which that window's audio has finished arriving — the
            reference point for every latency number in the project.

        Called once per recording rather than per window. That is a compute
        optimisation only: because the encoder is causal, running it over the
        whole sequence gives *bit-identical* frame features to feeding it frame by
        frame, which ``tests/test_causality.py`` checks. The genuinely streaming
        cost is measured separately in ``pipelines/efficiency.py``.
        """
        self.eval()
        starts, ends = window_grid(features.shape[0], window_frames, hop_frames)
        if starts.size == 0:
            return (
                np.zeros((0, self.cfg.embed_dim), dtype=np.float32),
                starts,
                ends,
            )
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).unsqueeze(0)
        frames = self.frame_features(x)[0]
        emb = self.pool_windows(
            frames, torch.from_numpy(starts), torch.from_numpy(ends)
        )
        return emb.numpy().astype(np.float32), starts, ends


def window_grid(
    n_frames: int, window_frames: int, hop_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """The streaming decision grid: one window per hop, ends clipped to ``n_frames``.

    Window ``k`` ends at ``min(n_frames, (k+1) * hop)`` and starts
    ``window_frames`` earlier, floored at 0. Two consequences, both deliberate:

    * The grid starts at ``hop`` rather than at ``window_frames``, so the first
      decision is available after one hop instead of one window. Early windows
      are therefore *shorter* than ``window_frames`` and their embeddings are
      noisier — that is honest streaming behaviour, and it is visible as the
      early-recording error in ``notebooks/04``.
    * The regions ``[k*hop, ends[k])`` tile the recording exactly once, so
      converting window decisions to frame labels needs no tie-breaking.
    """
    if window_frames <= 0 or hop_frames <= 0:
        raise ValueError(f"window and hop must be positive, got {window_frames}, {hop_frames}")
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n_windows = int(np.ceil(n_frames / hop_frames))
    ends = np.minimum(n_frames, (np.arange(n_windows) + 1) * hop_frames).astype(np.int64)
    starts = np.maximum(0, ends - window_frames).astype(np.int64)
    return starts, ends


def region_bounds(n_frames: int, hop_frames: int, ends: np.ndarray) -> np.ndarray:
    """Start frame of the region each window is responsible for labelling."""
    _ = n_frames
    return np.maximum(0, ends - hop_frames).astype(np.int64)


def build_embedder(
    cfg: EmbedderConfig,
    n_features: int,
    feature_mean: np.ndarray | None = None,
    feature_sd: np.ndarray | None = None,
    seed: int | None = None,
) -> SpeakerEmbedder:
    """Construct an embedder with a reproducible initialisation.

    Seeding here and not only globally matters because the ablations build a
    second model in the same process; without a local seed the non-causal variant
    would start from a different initialisation than the causal one and the
    ablation would confound architecture with initialisation.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    return SpeakerEmbedder(cfg, n_features, feature_mean, feature_sd)
