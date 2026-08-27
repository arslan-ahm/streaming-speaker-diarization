"""Bounded-latency online diarization, and the naive online baseline.

The claim this file implements
------------------------------
An offline diarizer sees every embedding in the recording before it assigns any
label. That is what makes global clustering accurate, and it is also what makes
it unusable for live captioning: the first label arrives after the last frame.
The mechanism here assigns a label to each window within a **fixed emission
delay** of that window's audio arriving, and never stores more than a bounded
number of speaker centroids — so per-decision cost and resident state are O(1) in
recording length.

The interesting question is not whether that costs accuracy (it does) but *how
much, at each budget*. So the budget is a single config number and the shipped
result is a curve.

What the budget actually buys
----------------------------
A latency budget of ``B`` milliseconds means a window's label may be withheld for
up to ``B`` ms after its audio has arrived, i.e. for ``floor(B / hop)`` further
hops. During that grace period the *later* windows have arrived, so the decision
for the oldest window can be made from a **local micro-cluster** rather than from
one noisy embedding:

1. Starting at the oldest pending window, walk forward in time while each
   successive embedding still matches the running micro-cluster mean at cosine
   similarity ``>= micro_cluster_threshold``. Stop at the first mismatch — a
   speaker turn is contiguous, and a later re-match after a gap belongs to a
   different turn as far as *this* decision is concerned.
2. Use the micro-cluster mean as the query against the tracked centroids.

That is the whole mechanism, and it is a variance-reduction argument: averaging
``m`` window embeddings of the same speaker cuts the query's noise by roughly
``sqrt(m)``, and the assignment threshold is what turns reduced query noise into
fewer label errors. Setting ``bounded_window=False`` removes step 1 and 2 and
queries with the single embedding — that is the ablation that isolates this
mechanism from the mere fact of having a buffer.

**Emission timing is exact, not modelled.** Window ``k`` is emitted when window
``k + budget_windows`` arrives, so its emission frame is ``ends[k + budget]``
and its delay is exactly the budget — except in the recording's tail, where the
remaining windows flush at the final frame and their delay is *smaller*. Latency
is therefore measured, not assumed, by ``metrics/latency.py``.

Two mechanisms are separately switchable because they fail differently:
``spawn_enabled`` (can a new speaker appear after the warm-up prefix?) and
``oracle_n_speakers`` (is the true count supplied?). Conflating them would make
"clustering is hard" and "counting speakers is hard" inseparable, which is the
single most common confound in the online-diarization literature.

References
----------
Zhang et al., *Fully Supervised Speaker Diarization* (UIS-RNN), ICASSP 2019 — the
online sequential-assignment framing.
Coria et al., *Overlap-Aware Low-Latency Online Speaker Diarization*, 2021 — the
latency-budget axis and incremental centroid updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import DiarizerConfig


@dataclass
class DiarizationOutput:
    """One recording's hypothesis, with everything the metrics need.

    ``emission_frames`` is the load-bearing field: it is what makes the latency
    claim falsifiable, and :func:`streamdiar.metrics.emission_latency` raises if
    any entry precedes its own window's end.
    """

    #: ``(K,)`` speaker id per window; ``-1`` is non-speech.
    labels: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    #: ``(K,)`` frame index at which each label became available to a consumer.
    emission_frames: np.ndarray
    #: ``(K,)`` probability in ``[0, 1]`` that the emitted label is correct.
    confidence: np.ndarray
    #: ``(K,)`` raw assignment margin; the logit temperature scaling calibrates.
    margin: np.ndarray
    #: ``(K,)`` True where the decision created a new speaker.
    spawned: np.ndarray
    #: ``(K,)`` size of the micro-cluster used as the query (1 when disabled).
    query_size: np.ndarray
    n_frames: int
    hop_frames: int
    #: Distinct speaker ids actually used, excluding non-speech.
    n_speakers: int = 0
    method: str = "online"
    extra: dict[str, float] = field(default_factory=dict)

    def frame_labels(self) -> np.ndarray:
        """``(n_frames,)`` per-frame label, ``-1`` for non-speech.

        Each window labels the region ``[end - hop, end)`` it is responsible for.
        Those regions tile the recording exactly once (see
        ``models/embedder.window_grid``), so there is no overlap to resolve and no
        frame left unassigned.
        """
        out = np.full(self.n_frames, -1, dtype=np.int64)
        for lab, end in zip(self.labels.tolist(), self.window_ends.tolist()):
            start = max(0, int(end) - self.hop_frames)
            out[start : int(end)] = int(lab)
        return out


class SpeakerTracker:
    """A bounded set of L2-normalised speaker centroids with momentum updates.

    Memory is ``max_speakers * embed_dim`` floats and never grows with recording
    length — that is the "bounded memory" half of the claim, and the cap is a hard
    one: once ``max_speakers`` are tracked, no further speaker can be created and
    the assignment is forced. That failure mode is visible in the results as a
    negative speaker-count bias on high-speaker-count recordings.

    ``centroid_momentum`` is an exponential update rather than a running mean on
    purpose. A running mean makes a centroid increasingly immovable, so a speaker
    whose voice drifts (or whose first few windows were mis-assigned) can never be
    recovered; an EMA keeps every centroid responsive at constant cost. The price
    is that the centroid is not the mean of its members, so it is not a k-means
    fixed point — which matters only if one wanted to claim optimality, and none
    is claimed.
    """

    def __init__(self, embed_dim: int, max_speakers: int, momentum: float) -> None:
        self.embed_dim = int(embed_dim)
        self.max_speakers = int(max_speakers)
        self.momentum = float(momentum)
        self.centroids = np.zeros((self.max_speakers, self.embed_dim), dtype=np.float64)
        self.counts = np.zeros(self.max_speakers, dtype=np.int64)
        self.n_active = 0

    @property
    def full(self) -> bool:
        return self.n_active >= self.max_speakers

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """``(n_active,)`` cosine similarities. Empty when no speaker exists yet."""
        if self.n_active == 0:
            return np.zeros(0, dtype=np.float64)
        return self.centroids[: self.n_active] @ np.asarray(query, dtype=np.float64)

    def spawn(self, embedding: np.ndarray) -> int:
        """Create a speaker seeded at ``embedding``. Returns its id, or ``-1`` if full."""
        if self.full:
            return -1
        idx = self.n_active
        self.centroids[idx] = _unit(embedding)
        self.counts[idx] = 1
        self.n_active += 1
        return idx

    def update(self, label: int, embedding: np.ndarray) -> None:
        """Move centroid ``label`` toward ``embedding`` by ``momentum``, re-normalised."""
        if not 0 <= label < self.n_active:
            raise IndexError(f"speaker {label} is not active (n_active={self.n_active})")
        a = self.momentum
        self.centroids[label] = _unit((1.0 - a) * self.centroids[label] + a * _unit(embedding))
        self.counts[label] += 1


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise, mapping the zero vector to itself rather than to NaN."""
    x = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def _softmax_confidence(
    sims: np.ndarray, chosen: int, spawn_score: float, temperature: float
) -> tuple[float, float]:
    """Confidence and margin for a decision over ``{existing centroids} + {spawn}``.

    The competitor set includes a virtual "spawn" option scored at
    ``spawn_threshold``, which is what makes an assign decision and a spawn
    decision comparable on one confidence scale. Without it, every decision taken
    while only one centroid exists would report confidence 1.0 by vacuity, and
    those are a large fraction of a short recording's decisions. The genuinely
    first decision of a recording still reports 1.0 — there is no alternative to
    compare against — and that handful of windows is a known artefact in the
    calibration numbers rather than a hidden one.

    Returns ``(confidence in (0, 1), margin)`` where margin is
    ``chosen_score - best_alternative_score``, the logit that
    ``metrics.calibration.fit_temperature`` recalibrates.
    """
    scores = np.concatenate([np.asarray(sims, dtype=np.float64), [float(spawn_score)]])
    idx = int(chosen) if chosen >= 0 else scores.size - 1
    t = max(float(temperature), 1e-6)
    shifted = (scores - scores.max()) / t
    exp = np.exp(shifted)
    conf = float(exp[idx] / exp.sum())
    others = np.delete(scores, idx)
    margin = float(scores[idx] - others.max()) if others.size else float(scores[idx])
    return conf, margin


def _micro_cluster(
    pending: list[tuple[int, np.ndarray]], threshold: float
) -> tuple[np.ndarray, int]:
    """Grow a contiguous micro-cluster forward from the oldest pending window.

    Walks forward in time, accumulating while each next embedding matches the
    running mean at cosine ``>= threshold``, and stops at the first mismatch. The
    stop is what makes this a *turn* estimate rather than a local clustering: if
    the speaker changes and later changes back, the second run is a different
    turn and must not be averaged into this decision's query, or a short
    interjection would drag the query toward the wrong speaker.

    Returns ``(unit-norm query, number of windows averaged)``.
    """
    acc = np.asarray(pending[0][1], dtype=np.float64).copy()
    size = 1
    for _, emb in pending[1:]:
        e = np.asarray(emb, dtype=np.float64)
        if float(_unit(acc) @ _unit(e)) < threshold:
            break
        acc = acc + e
        size += 1
    return _unit(acc), size


def diarize_online(
    embeddings: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    is_speech: np.ndarray,
    cfg: DiarizerConfig,
    budget_windows: int,
    n_frames: int,
    hop_frames: int,
    oracle_n_speakers: int | None = None,
) -> DiarizationOutput:
    """The proposed bounded-latency online diarizer.

    Args:
        embeddings: ``(K, D)`` L2-normalised window embeddings.
        window_starts: ``(K,)`` first frame of each embedding window.
        window_ends: ``(K,)`` frame at which each window's audio finished arriving.
        is_speech: ``(K,)`` bool from the causal VAD.
        cfg: Mechanism and ablation switches.
        budget_windows: Lookahead in whole hops, ``floor(budget_ms / hop_ms)``.
        n_frames: Recording length, for the frame-label expansion.
        hop_frames: Hop in frames, for the frame-label expansion.
        oracle_n_speakers: True speaker count, or ``None``. When given, spawning
            is capped at that count instead of by ``spawn_threshold`` alone.

    Returns:
        A :class:`DiarizationOutput` whose ``emission_frames`` are exact: window
        ``k`` is emitted when window ``k + budget_windows`` arrives, or at the
        final frame for the tail that never gets that far.

    The loop below is the honest streaming structure — one pass, forward only,
    with a deadline check that runs on *every* window including non-speech ones.
    Checking only on speech windows would let a pending decision drift past its
    deadline across a silence, quietly buying unbudgeted lookahead; that was a
    real bug during development and it made the ``B=0`` curve point impossible.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    n_windows = emb.shape[0]
    budget = max(0, int(budget_windows))

    labels = np.full(n_windows, -1, dtype=np.int64)
    emission = np.zeros(n_windows, dtype=np.int64)
    confidence = np.full(n_windows, np.nan, dtype=np.float64)
    margin = np.full(n_windows, np.nan, dtype=np.float64)
    spawned_flags = np.zeros(n_windows, dtype=bool)
    query_size = np.zeros(n_windows, dtype=np.int64)

    if n_windows == 0:
        return DiarizationOutput(
            labels, np.asarray(window_starts), np.asarray(window_ends), emission,
            confidence, margin, spawned_flags, query_size, int(n_frames),
            int(hop_frames), 0, "online",
        )

    tracker = SpeakerTracker(emb.shape[1], cfg.max_speakers, cfg.centroid_momentum)
    pending: deque[tuple[int, np.ndarray]] = deque()

    def emit(arrival_frame: int) -> None:
        j, e0 = pending[0]
        if cfg.bounded_window and len(pending) > 1:
            query, size = _micro_cluster(list(pending), cfg.micro_cluster_threshold)
        else:
            query, size = _unit(e0), 1

        sims = tracker.similarities(query)
        best = int(np.argmax(sims)) if sims.size else -1
        best_sim = float(sims[best]) if sims.size else -np.inf

        if oracle_n_speakers is not None:
            may_spawn = tracker.n_active < int(oracle_n_speakers)
        else:
            may_spawn = (cfg.spawn_enabled or j < cfg.warmup_windows) and not tracker.full

        chosen_is_spawn = sims.size == 0 or (best_sim < cfg.spawn_threshold and may_spawn)
        if chosen_is_spawn:
            label = tracker.spawn(e0)
            if label < 0:  # cap reached between the check and the spawn
                label, chosen_is_spawn = best, False
                tracker.update(label, e0)
        else:
            label = best
            tracker.update(label, e0)

        conf, mrg = _softmax_confidence(
            sims, -1 if chosen_is_spawn else label, cfg.spawn_threshold,
            cfg.confidence_temperature,
        )
        labels[j] = label
        emission[j] = int(arrival_frame)
        confidence[j] = conf
        margin[j] = mrg
        spawned_flags[j] = chosen_is_spawn
        query_size[j] = size
        pending.popleft()

    speech = np.asarray(is_speech, dtype=bool)
    ends = np.asarray(window_ends, dtype=np.int64)
    for k in range(n_windows):
        if speech[k]:
            pending.append((k, emb[k]))
        # Deadline check on every window, speech or not. See the docstring.
        while pending and pending[0][0] + budget <= k:
            emit(int(ends[k]))
        if not speech[k]:
            emission[k] = int(ends[k])
    final = int(ends[-1])
    while pending:
        emit(final)

    used = {int(v) for v in labels.tolist() if v >= 0}
    return DiarizationOutput(
        labels=labels,
        window_starts=np.asarray(window_starts, dtype=np.int64),
        window_ends=ends,
        emission_frames=emission,
        confidence=confidence,
        margin=margin,
        spawned=spawned_flags,
        query_size=query_size,
        n_frames=int(n_frames),
        hop_frames=int(hop_frames),
        n_speakers=len(used),
        method="online",
        extra={"mean_query_size": float(query_size[labels >= 0].mean()) if used else float("nan")},
    )


def diarize_naive_online(
    embeddings: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    is_speech: np.ndarray,
    cfg: DiarizerConfig,
    n_frames: int,
    hop_frames: int,
) -> DiarizationOutput:
    """Greedy nearest-centroid online baseline: zero latency, no buffer, running mean.

    This is the obvious thing to write, and it is a *baseline rather than an
    ablation* because it differs from the proposed method in centroid update rule
    as well as in lookahead: centroids here are the exact running mean of their
    assigned embeddings, the textbook incremental approach. Comparing against it
    answers "is the proposed mechanism better than the naive online thing", while
    the ``bounded_window=False`` ablation answers the narrower "is the lookahead
    the reason". Both questions are reported, and they have different answers.

    Emission delay is exactly zero for every decision, which is its one genuine
    advantage and is reported as such rather than hidden.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    n_windows = emb.shape[0]
    labels = np.full(n_windows, -1, dtype=np.int64)
    emission = np.asarray(window_ends, dtype=np.int64).copy()
    confidence = np.full(n_windows, np.nan, dtype=np.float64)
    margin = np.full(n_windows, np.nan, dtype=np.float64)
    spawned_flags = np.zeros(n_windows, dtype=bool)
    query_size = np.ones(n_windows, dtype=np.int64)

    centroids = np.zeros((cfg.max_speakers, emb.shape[1] if n_windows else 1), dtype=np.float64)
    counts = np.zeros(cfg.max_speakers, dtype=np.int64)
    n_active = 0
    speech = np.asarray(is_speech, dtype=bool)

    for k in range(n_windows):
        if not speech[k]:
            continue
        e = _unit(emb[k])
        sims = centroids[:n_active] @ e if n_active else np.zeros(0)
        best = int(np.argmax(sims)) if sims.size else -1
        best_sim = float(sims[best]) if sims.size else -np.inf
        if sims.size == 0 or (best_sim < cfg.spawn_threshold and n_active < cfg.max_speakers):
            centroids[n_active] = e
            counts[n_active] = 1
            labels[k] = n_active
            spawned_flags[k] = True
            n_active += 1
            chosen = -1
        else:
            n = counts[best]
            centroids[best] = _unit((centroids[best] * n + e) / (n + 1))
            counts[best] = n + 1
            labels[k] = best
            chosen = best
        confidence[k], margin[k] = _softmax_confidence(
            sims, chosen, cfg.spawn_threshold, cfg.confidence_temperature
        )

    used = {int(v) for v in labels.tolist() if v >= 0}
    return DiarizationOutput(
        labels=labels,
        window_starts=np.asarray(window_starts, dtype=np.int64),
        window_ends=np.asarray(window_ends, dtype=np.int64),
        emission_frames=emission,
        confidence=confidence,
        margin=margin,
        spawned=spawned_flags,
        query_size=query_size,
        n_frames=int(n_frames),
        hop_frames=int(hop_frames),
        n_speakers=len(used),
        method="naive_online",
    )
