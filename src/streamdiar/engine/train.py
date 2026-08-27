"""Metric-learning training for the causal embedder.

The diarizer has no classifier — at inference time it must decide whether two
windows are the same person, from a pool of speakers it has never heard. So the
objective is a metric one: pull segments of the same speaker together on the unit
sphere and push different speakers apart, with the *cosine* geometry that the
clustering mechanisms downstream actually use.

The loss is the prototypical / GE2E form (Wan et al., 2018; Snell et al., 2017).
A batch is ``N`` speakers x ``M`` segments. For segment ``(n, m)`` the similarity
to speaker ``k``'s prototype is a scaled cosine, and the target is ``n``:

    S[(n,m), k] = w * cos(e_nm, c_k) + b,      c_k = mean_{m'} e_km'

with one detail that is not optional: when ``k == n`` the prototype **excludes**
``e_nm`` itself,

    c_n^{-(nm)} = (1 / (M - 1)) * sum_{m' != m} e_nm'

Without the exclusion, ``e_nm`` appears on both sides of its own positive
similarity, the loss can be driven down by inflating the norm of a single
embedding, and training collapses to a state where every embedding of a speaker
is identical to its own batch-mate mean while different speakers are not
separated at all. This is the single most common bug in GE2E reimplementations
and it produces a training curve that looks *better* than the correct one.

``w`` and ``b`` are learnable, initialised at 10 and -5. They matter because
cosine similarities live in ``[-1, 1]`` and a softmax over that range is nearly
uniform: without a learned scale the gradient signal is tiny and the model
underfits at any learning rate.

Why not an angular-margin softmax (AAM/ArcFace). It needs a weight matrix with
one column per training speaker, which ties model size to the speaker pool and
gives nothing to a 160-speaker pool at this scale. The prototypical form is also
what makes the objective match the test-time operation — comparing an embedding
to a centroid — exactly.

References
----------
Wan et al., *Generalized End-to-End Loss for Speaker Verification*, ICASSP 2018.
Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..data.generator import feature_statistics, sample_training_batch
from ..models.embedder import SpeakerEmbedder, build_embedder
from ..utils.io import append_jsonl
from ..utils.seeding import limit_threads, seed_everything


class PrototypicalLoss(nn.Module):
    """Scaled-cosine prototypical loss with the positive prototype self-excluded."""

    def __init__(self, init_scale: float = 10.0, init_bias: float = -5.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(max(init_scale, 1e-3)))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, embeddings: torch.Tensor, n_speakers: int, n_segments: int) -> torch.Tensor:
        """``(N*M, D)`` embeddings, grouped speaker-major, to a scalar loss.

        Raises:
            ValueError: If ``n_segments < 2``. With one segment per speaker the
                self-excluded prototype is undefined, and silently falling back to
                the inclusive prototype is exactly the bug the docstring warns
                about — so it raises instead.
        """
        if n_segments < 2:
            raise ValueError(
                f"prototypical loss needs n_segments >= 2, got {n_segments}; with one "
                "segment per speaker the self-excluded prototype does not exist"
            )
        e = embeddings.view(n_speakers, n_segments, -1)
        totals = e.sum(dim=1, keepdim=True)  # (N, 1, D)

        # Inclusive prototypes for the negatives, self-excluded for the positive.
        proto_all = nn.functional.normalize(totals.squeeze(1), dim=-1)  # (N, D)
        proto_excl = nn.functional.normalize((totals - e) / (n_segments - 1), dim=-1)  # (N, M, D)

        flat = e.reshape(n_speakers * n_segments, -1)
        sims = flat @ proto_all.t()  # (N*M, N)
        own = (e * proto_excl).sum(dim=-1).reshape(-1)  # (N*M,)
        idx = torch.arange(n_speakers, device=e.device).repeat_interleave(n_segments)
        sims = sims.clone()
        sims[torch.arange(sims.shape[0], device=e.device), idx] = own

        logits = self.log_scale.exp() * sims + self.bias
        return nn.functional.cross_entropy(logits, idx)


@dataclass
class TrainedModel:
    """A trained embedder plus everything inference needs to be causal.

    ``vad_threshold`` is fitted on the **dev** split after training, never on
    test. Bundling it here rather than in the config is deliberate: it is a
    *learned* parameter, and a checkpoint that does not carry it would silently
    fall back to the config default and change every DER in the repository.
    """

    embedder: SpeakerEmbedder
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    vad_threshold: float = 0.0
    vad_frame_error: float = float("nan")
    seed: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    n_parameters: int = 0

    def state(self) -> dict[str, object]:
        return {
            "state_dict": self.embedder.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_sd": self.feature_sd,
            "vad_threshold": self.vad_threshold,
            "vad_frame_error": self.vad_frame_error,
            "seed": self.seed,
            "n_parameters": self.n_parameters,
        }


def train_embedder(
    cfg: Config,
    seed: int | None = None,
    history_path: str | None = None,
    progress: bool = False,
) -> TrainedModel:
    """Train the embedder with the prototypical objective. Returns a :class:`TrainedModel`.

    Args:
        cfg: Full config; uses ``cfg.train``, ``cfg.embedder``, ``cfg.data``.
        seed: Overrides ``cfg.seed``. This is the *training-run* seed — the unit of
            analysis for every method-level claim in ``docs/RESULTS.md``.
        history_path: JSONL file appended per log step, so a killed run keeps its
            curve.
        progress: Print a line per log step.

    The batch is generated on the fly rather than pre-materialised. At 48 segments
    of 100 frames that is ~0.1 s per step against ~0.25 s of forward+backward, so
    caching would buy under a third of the wall-clock at the cost of the
    per-step-independent RNG streams that make the data reproducible.
    """
    limit_threads(cfg.train.threads)
    run_seed = int(cfg.seed if seed is None else seed)
    seed_everything(run_seed)

    mean, sd = feature_statistics(cfg.data, run_seed)
    embedder = build_embedder(cfg.embedder, cfg.data.n_features, mean, sd, seed=run_seed)
    loss_fn = PrototypicalLoss(cfg.train.init_logit_scale, cfg.train.init_logit_bias)
    params = list(embedder.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=max(1, cfg.train.steps), pct_start=0.15
    )

    history: list[dict[str, float]] = []
    embedder.train()
    t0 = time.perf_counter()
    for step in range(cfg.train.steps):
        feats, labels = sample_training_batch(
            cfg.data,
            run_seed,
            step,
            cfg.train.n_speakers_per_batch,
            cfg.train.n_segments_per_speaker,
            cfg.train.segment_frames,
        )
        x = torch.from_numpy(feats)
        emb = embedder(x)
        loss = loss_fn(emb, cfg.train.n_speakers_per_batch, cfg.train.n_segments_per_speaker)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.train.log_every == 0 or step == cfg.train.steps - 1:
            with torch.no_grad():
                acc = _batch_accuracy(emb, cfg.train.n_speakers_per_batch,
                                     cfg.train.n_segments_per_speaker)
            record = {
                "step": float(step),
                "loss": float(loss.item()),
                "batch_accuracy": float(acc),
                "grad_norm": float(grad_norm),
                "lr": float(sched.get_last_lr()[0]),
                "logit_scale": float(loss_fn.log_scale.exp().item()),
                "elapsed_s": float(time.perf_counter() - t0),
            }
            history.append(record)
            if history_path:
                append_jsonl(history_path, record)
            if progress:
                print(
                    f"  step {step:4d}  loss {record['loss']:.4f}  "
                    f"acc {record['batch_accuracy']:.3f}  {record['elapsed_s']:.0f}s",
                    flush=True,
                )
        _ = labels

    embedder.eval()
    return TrainedModel(
        embedder=embedder,
        feature_mean=mean,
        feature_sd=sd,
        seed=run_seed,
        history=history,
        train_seconds=float(time.perf_counter() - t0),
        n_parameters=int(sum(p.numel() for p in embedder.parameters())),
    )


@torch.no_grad()
def _batch_accuracy(embeddings: torch.Tensor, n_speakers: int, n_segments: int) -> float:
    """Fraction of segments whose nearest self-excluded prototype is their own speaker.

    A far more legible progress signal than the loss, because it is on a fixed
    scale: chance is ``1 / n_speakers`` (0.083 at the default N=12), so a curve
    that sits at 0.08 says "not learning" without needing a reference loss value.
    """
    e = embeddings.view(n_speakers, n_segments, -1)
    totals = e.sum(dim=1, keepdim=True)
    proto_all = nn.functional.normalize(totals.squeeze(1), dim=-1)
    proto_excl = nn.functional.normalize((totals - e) / max(n_segments - 1, 1), dim=-1)
    flat = e.reshape(n_speakers * n_segments, -1)
    sims = flat @ proto_all.t()
    own = (e * proto_excl).sum(dim=-1).reshape(-1)
    idx = torch.arange(n_speakers, device=e.device).repeat_interleave(n_segments)
    sims[torch.arange(sims.shape[0], device=e.device), idx] = own
    return float((sims.argmax(dim=1) == idx).double().mean().item())
