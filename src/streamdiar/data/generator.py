"""Procedural multi-speaker conversation generator with ground truth by construction.

Why generate instead of downloading. Every number this repository ships has to be
reproducible on a laptop with no network and no dataset licence, and — more
importantly — the diarization *reference* has to be exact. AMI and VoxConverse
references are human annotations with real boundary error, typically tens of
milliseconds, which is the same order as the differences this project is trying
to resolve on the latency axis. Here the turn list is sampled *first* and the
features are rendered *from* it, so the reference is the generative truth. An
optional real-data path exists (:mod:`streamdiar.data.real`) but is never on the
critical path for tests or shipped results.

The generative model, and what it forces the embedder to learn
--------------------------------------------------------------
A frame is a log-power spectrum-like vector in ``R^F``. Three additive terms
live in that space, and only the first carries speaker identity:

* **timbre** ``B_spk @ c_s`` — a smooth low-frequency envelope. ``c_s in R^8`` is
  the speaker's identity; ``B_spk`` is a *global* cosine basis shared by all
  speakers, so identities live on an 8-dimensional manifold rather than being
  ``F`` independent numbers. This is what makes the task learnable at all.
* **phonetic content** ``B_phn @ d_p`` — one of ``n_phones`` higher-frequency
  templates, resampled every 8-20 frames, drawn from a pool *shared across
  speakers*. It is pure nuisance: an embedder that does not average it out will
  cluster by phone instead of by speaker.
* **channel** a per-segment offset plus a slow energy contour, again pure
  nuisance. Training segments get *independent* channels even within a speaker,
  specifically so channel cannot be used as a shortcut for identity; within a
  test recording all speakers share one channel, which is the easier direction.

Overlap is summed in the **linear power domain** and then log-compressed, which
is what actually happens to a mel filterbank when two people talk at once:
``log(exp(a) + exp(b))``, not ``a + b``. Getting this wrong would make overlap
trivially detectable as an amplitude spike.

References
----------
Wan et al., *Generalized End-to-End Loss for Speaker Verification*, ICASSP 2018 —
the metric-learning objective these segments feed.
Fujita et al., *End-to-End Neural Speaker Diarization with Permutation-Free
Objectives*, Interspeech 2019 — the overlap-as-multi-label formulation used by
the reference matrix here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DataConfig
from ..utils.seeding import rng_for

#: Log-power of the non-speech background. Speech sits near 0, so this is a
#: clearly quieter floor without being a silence that trivially segments.
BACKGROUND_LOG_POWER = -2.0

#: Dimension of the speaker-identity manifold.
N_SPEAKER_BASIS = 8
#: Dimension of the phonetic-content manifold.
N_PHONE_BASIS = 10


@dataclass(frozen=True)
class Turn:
    """A speech region. ``end`` is exclusive, in frames."""

    start: int
    end: int
    speaker: int

    @property
    def n_frames(self) -> int:
        return max(0, self.end - self.start)

    def seconds(self, frame_rate: int) -> tuple[float, float]:
        return self.start / frame_rate, self.end / frame_rate


@dataclass
class Recording:
    """One synthetic conversation with its exact reference."""

    name: str
    #: ``(T, F)`` float32 features, one row per frame.
    features: np.ndarray
    turns: list[Turn]
    #: Number of speakers actually present. Never passed to the diarizer unless
    #: the oracle-count variant is explicitly enabled.
    n_speakers: int
    frame_rate: int
    #: Indices into the global speaker pool, for provenance.
    pool_ids: list[int] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.frame_rate

    def reference_matrix(self) -> np.ndarray:
        """``(T, n_speakers)`` bool activity. Multi-label: overlap sets two columns."""
        ref = np.zeros((self.n_frames, self.n_speakers), dtype=bool)
        for t in self.turns:
            ref[t.start : t.end, t.speaker] = True
        return ref

    def overlap_fraction(self) -> float:
        """Fraction of *speech* frames with two or more simultaneous speakers."""
        counts = self.reference_matrix().sum(axis=1)
        speech = counts > 0
        return float((counts[speech] >= 2).mean()) if speech.any() else 0.0

    def speech_fraction(self) -> float:
        return float((self.reference_matrix().sum(axis=1) > 0).mean())

    def to_rttm(self) -> str:
        """NIST RTTM text, so the reference can be scored by external tools too."""
        lines = []
        for t in sorted(self.turns, key=lambda x: x.start):
            start, end = t.seconds(self.frame_rate)
            lines.append(
                f"SPEAKER {self.name} 1 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> spk{t.speaker:02d} <NA> <NA>"
            )
        return "\n".join(lines) + "\n"


def _cosine_basis(n_features: int, n_basis: int, low: int, seed_stream: str) -> np.ndarray:
    """``(n_features, n_basis)`` smooth basis of cosines starting at frequency ``low``.

    Deterministic given ``seed_stream``: the basis is a property of the *world*,
    not of a particular dataset split, so train and test speakers inhabit the
    same feature geometry. A random projection would work too, but a cosine
    basis makes the "timbre is smooth, phonetics is not" distinction visible in
    ``notebooks/01``.
    """
    x = (np.arange(n_features) + 0.5) / n_features
    cols = [np.cos(np.pi * (low + k) * x) for k in range(n_basis)]
    basis = np.stack(cols, axis=1)
    # A fixed random rotation removes the accidental orthogonality-to-the-axes
    # that would let a single feature channel identify a basis coefficient.
    rot = rng_for(0, seed_stream).standard_normal((n_basis, n_basis)) / np.sqrt(n_basis)
    return (basis @ rot).astype(np.float64)


class World:
    """The shared, split-independent generative constants.

    Built once per ``(n_features, n_phones)`` so that every recording in every
    split draws speakers from the same manifold. Cheap enough to rebuild, but
    caching it keeps the generator from dominating the profile of a 24-recording
    evaluation.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.speaker_basis = _cosine_basis(cfg.n_features, N_SPEAKER_BASIS, 1, "speaker_basis")
        self.phone_basis = _cosine_basis(cfg.n_features, N_PHONE_BASIS, 4, "phone_basis")
        pr = rng_for(0, "phones")
        coeffs = pr.standard_normal((cfg.n_phones, N_PHONE_BASIS))
        #: ``(n_phones, n_features)`` content templates, shared across speakers.
        self.phone_templates = (coeffs @ self.phone_basis.T) * 0.9

    def speaker_timbre(self, pool: str, pool_id: int) -> np.ndarray:
        """Speaker ``pool_id``'s envelope. Addressed by name, so it is stable.

        ``rng_for(0, pool, pool_id)`` means speaker 7 of the test pool has the
        same voice no matter how many recordings preceded it, which is what lets
        an ablation change recording counts without perturbing identities.
        """
        c = rng_for(0, pool, int(pool_id)).standard_normal(N_SPEAKER_BASIS)
        return (c @ self.speaker_basis.T) * self.cfg.speaker_scale


_WORLD_CACHE: dict[tuple[int, int, float], World] = {}


def get_world(cfg: DataConfig) -> World:
    key = (cfg.n_features, cfg.n_phones, cfg.speaker_scale)
    if key not in _WORLD_CACHE:
        _WORLD_CACHE[key] = World(cfg)
    return _WORLD_CACHE[key]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_speech(
    world: World,
    timbre: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    channel: np.ndarray,
) -> np.ndarray:
    """Render ``n_frames`` of one speaker's log-power features.

    The phone sequence is piecewise-constant with 8-20 frame runs (80-200 ms,
    the real range for a phone) and a linear cross-fade of 3 frames at each
    boundary. Without the cross-fade the feature sequence has step
    discontinuities that a dilated convolution can latch onto as a free
    segmentation cue, which would make the task easier than speech is.
    """
    cfg = world.cfg
    out = np.empty((n_frames, cfg.n_features), dtype=np.float64)
    pos = 0
    prev = world.phone_templates[rng.integers(cfg.n_phones)]
    while pos < n_frames:
        run = int(rng.integers(8, 21))
        cur = world.phone_templates[rng.integers(cfg.n_phones)]
        stop = min(n_frames, pos + run)
        block = np.repeat(cur[None, :], stop - pos, axis=0)
        fade = min(3, stop - pos)
        if fade > 0:
            w = np.linspace(0.0, 1.0, fade + 1)[1:][:, None]
            block[:fade] = w * cur[None, :] + (1.0 - w) * prev[None, :]
        out[pos:stop] = block
        prev = cur
        pos = stop

    # A slow energy contour: broadcast over channels, so it is a pure nuisance
    # direction that carries no spectral shape and hence no identity.
    contour = _smooth_noise(n_frames, rng, scale=0.55, tau=25.0)
    return out + timbre[None, :] + channel[None, :] + contour[:, None]


def _smooth_noise(n: int, rng: np.random.Generator, scale: float, tau: float) -> np.ndarray:
    """An OU-like smooth random walk of length ``n``, sd ``scale``, timescale ``tau``."""
    if n <= 0:
        return np.zeros(0)
    a = float(np.exp(-1.0 / max(tau, 1e-6)))
    innov = rng.standard_normal(n) * scale * np.sqrt(1.0 - a * a)
    out = np.empty(n)
    acc = rng.standard_normal() * scale
    for i in range(n):
        acc = a * acc + innov[i]
        out[i] = acc
    return out


def sample_turns(cfg: DataConfig, n_speakers: int, rng: np.random.Generator) -> list[Turn]:
    """Sample the turn structure. This *is* the ground truth.

    Turn durations are log-normal, which is the right family for conversational
    turns (Jefferson, 1989; and every corpus since). Overlap is generated only
    against the immediately preceding turn: three-way simultaneous speech is
    vanishingly rare in real meetings, and pretending otherwise would let the
    generator manufacture an overlap regime no system could ever face.

    Speaker choice is a first-order Markov chain that forbids self-succession
    (two consecutive turns by the same speaker are one turn), which means the
    marginal turn-length distribution is exactly the sampled log-normal.
    """
    total = int(round(cfg.duration_s * cfg.frame_rate))
    mu = np.log(max(cfg.turn_median_s, 1e-3))
    min_frames = max(2, int(0.35 * cfg.frame_rate))
    max_frames = max(min_frames + 1, int(12.0 * cfg.frame_rate))

    turns: list[Turn] = []
    cursor = int(rng.integers(0, max(1, int(0.4 * cfg.frame_rate))))
    prev_speaker = -1
    prev_end = cursor
    prev_len = 0

    while cursor < total:
        choices = [s for s in range(n_speakers) if s != prev_speaker] or list(range(n_speakers))
        speaker = int(rng.choice(choices))
        dur = int(np.clip(np.exp(mu + cfg.turn_sigma * rng.standard_normal()) * cfg.frame_rate,
                          min_frames, max_frames))

        if turns and rng.random() < cfg.overlap_prob:
            back = int(rng.random() * cfg.overlap_frac_max * min(dur, max(prev_len, 1)))
            start = max(0, prev_end - back)
        else:
            pause_s = rng.uniform(*cfg.pause_range_s)
            start = prev_end + int(pause_s * cfg.frame_rate)

        end = min(total, start + dur)
        if start >= total:
            break
        if end - start >= min_frames // 2:
            turns.append(Turn(start=int(start), end=int(end), speaker=speaker))
            prev_speaker, prev_end, prev_len = speaker, end, end - start
        else:
            prev_end = end
        cursor = max(cursor + 1, prev_end)

    if not turns:  # degenerate config (duration shorter than one turn)
        turns = [Turn(0, min(total, min_frames), 0)]
    return turns


def generate_recording(
    cfg: DataConfig,
    seed: int,
    index: int,
    pool: str = "test",
    n_speakers: int | None = None,
) -> Recording:
    """Generate recording ``index`` of a split. Deterministic in ``(seed, index, pool)``.

    Note the two independent RNG streams: ``("turns", index)`` drives the
    reference and ``("render", index)`` drives the acoustics. Changing the noise
    level therefore leaves the turn structure bit-identical, which is what makes
    a difficulty sweep a controlled experiment instead of a new dataset.
    """
    world = get_world(cfg)
    turn_rng = rng_for(seed, pool, "turns", index)
    render_rng = rng_for(seed, pool, "render", index)

    lo, hi = cfg.n_speakers_range
    k = int(n_speakers) if n_speakers is not None else int(turn_rng.integers(lo, hi + 1))
    k = max(1, k)

    n_pool = cfg.n_train_speakers if pool == "train" else cfg.n_test_speakers
    # Sample *without* replacement so two columns of the reference are never the
    # same voice, which would make a "confusion" error unfalsifiable.
    pool_ids = sorted(int(v) for v in turn_rng.choice(n_pool, size=min(k, n_pool), replace=False))
    timbres = [world.speaker_timbre(pool, pid) for pid in pool_ids]

    turns = sample_turns(cfg, k, turn_rng)
    total = int(round(cfg.duration_s * cfg.frame_rate))
    channel = render_rng.standard_normal(cfg.n_features) * cfg.channel_sd

    # Accumulate in linear power, then compress. See the module docstring.
    power = np.full((total, cfg.n_features), np.exp(BACKGROUND_LOG_POWER), dtype=np.float64)
    for t in turns:
        seg = _render_speech(world, timbres[t.speaker % len(timbres)], t.n_frames,
                             render_rng, channel)
        power[t.start : t.end] += np.exp(seg)

    features = np.log(power) + render_rng.standard_normal(power.shape) * cfg.noise_sd
    return Recording(
        name=f"{pool}_{seed:03d}_{index:03d}",
        features=features.astype(np.float32),
        turns=turns,
        n_speakers=k,
        frame_rate=cfg.frame_rate,
        pool_ids=pool_ids,
    )


def generate_split(cfg: DataConfig, seed: int, split: str) -> list[Recording]:
    """Generate the ``dev`` or ``test`` split.

    Both draw from the *test* speaker pool, disjoint from the training pool, so
    temperature scaling fitted on dev never sees a training voice either. The
    split index offset keeps dev and test recordings distinct.
    """
    if split == "dev":
        n, offset = cfg.n_dev_recordings, 0
    elif split == "test":
        n, offset = cfg.n_test_recordings, 1000
    else:
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    return [generate_recording(cfg, seed, offset + i, pool="test") for i in range(n)]


# --------------------------------------------------------------------------- #
# Training segments for metric learning
# --------------------------------------------------------------------------- #
def sample_segment(
    cfg: DataConfig,
    seed: int,
    pool: str,
    pool_id: int,
    variant: int,
    n_frames: int,
) -> np.ndarray:
    """One single-speaker segment, rendered through the *same* pipeline as a recording.

    Two deliberate choices:

    * The channel offset is drawn per ``(pool_id, variant)``, so two segments of
      the same speaker in the same batch have *different* channels. If they
      shared one, the cheapest solution to the metric-learning objective would
      be to encode the channel, and the embedder would collapse on real
      recordings where all speakers share it.
    * Background power and observation noise are added exactly as in
      :func:`generate_recording`, so a training segment and a single-speaker
      test window are draws from the same distribution. Overlapped windows are
      *not* represented in training — nothing sensible can be learned for them
      with a single-label objective, and ``docs/RESULTS.md`` reports the
      resulting overlap-region error separately rather than hiding it.
    """
    world = get_world(cfg)
    rng = rng_for(seed, pool, "segment", int(pool_id), int(variant))
    channel = rng.standard_normal(cfg.n_features) * cfg.channel_sd
    seg = _render_speech(world, world.speaker_timbre(pool, pool_id), n_frames, rng, channel)
    power = np.exp(BACKGROUND_LOG_POWER) + np.exp(seg)
    return (np.log(power) + rng.standard_normal(seg.shape) * cfg.noise_sd).astype(np.float32)


def sample_training_batch(
    cfg: DataConfig,
    seed: int,
    step: int,
    n_speakers: int,
    n_segments: int,
    segment_frames: int,
    pool: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """A prototypical-loss batch: ``(N*M, L, F)`` features and ``(N*M,)`` labels.

    Labels are batch-local ``0..N-1``; the identity of the underlying pool
    speaker is irrelevant to a metric-learning objective and keeping labels
    local means the loss never has to size itself to the pool.
    """
    n_pool = cfg.n_train_speakers if pool == "train" else cfg.n_test_speakers
    rng = rng_for(seed, "batch", int(step))
    chosen = rng.choice(n_pool, size=min(n_speakers, n_pool), replace=False)

    feats: list[np.ndarray] = []
    labels: list[int] = []
    for local, pid in enumerate(chosen):
        for _ in range(n_segments):
            variant = int(rng.integers(0, 1_000_000))
            feats.append(sample_segment(cfg, seed, pool, int(pid), variant, segment_frames))
            labels.append(local)
    return np.stack(feats, axis=0), np.asarray(labels, dtype=np.int64)


def feature_statistics(
    cfg: DataConfig, seed: int, n_segments: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and sd, estimated from *training* segments only.

    These are frozen into the model at construction time. This is the causal
    choice: normalising a test recording by its own statistics would make every
    frame depend on the whole recording, silently destroying the online claim
    that ``tests/test_causality.py`` asserts. It is also why the ``channel_sd``
    nuisance is not simply cancelled by normalisation.
    """
    feats, _ = sample_training_batch(
        cfg, seed, step=-1, n_speakers=min(n_segments, cfg.n_train_speakers),
        n_segments=1, segment_frames=100, pool="train",
    )
    flat = feats.reshape(-1, feats.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    sd = np.maximum(flat.std(axis=0), 1e-3)
    return mean.astype(np.float32), sd.astype(np.float32)


def dataset_stats(recordings: list[Recording]) -> dict[str, float]:
    """Descriptive statistics for the split, reported in ``docs/RESULTS.md``."""
    if not recordings:
        return {}
    turn_lengths = [t.n_frames / r.frame_rate for r in recordings for t in r.turns]
    return {
        "n_recordings": float(len(recordings)),
        "total_hours": float(sum(r.duration_s for r in recordings) / 3600.0),
        "mean_speakers": float(np.mean([r.n_speakers for r in recordings])),
        "min_speakers": float(min(r.n_speakers for r in recordings)),
        "max_speakers": float(max(r.n_speakers for r in recordings)),
        "mean_turns_per_recording": float(np.mean([len(r.turns) for r in recordings])),
        "median_turn_s": float(np.median(turn_lengths)),
        "mean_overlap_fraction": float(np.mean([r.overlap_fraction() for r in recordings])),
        "mean_speech_fraction": float(np.mean([r.speech_fraction() for r in recordings])),
    }
