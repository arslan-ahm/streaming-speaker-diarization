"""Typed configuration: dataclasses, YAML with ``_base_`` inheritance, ``--set`` overrides.

Three properties are non-negotiable for a repository whose claims must be
traceable:

1. **Every run's effective config is serialisable.** ``Config.to_dict`` round-trips
   through YAML, and every run directory gets a ``config.yaml`` written from the
   *resolved* object, not from the file the user passed. A config file that
   inherits from three bases is not evidence of what ran; the resolved dump is.
2. **Unknown keys are errors.** A typo in ``diarizer.laetncy_budget_ms`` that is
   silently ignored produces a run that looks like an ablation and is not one.
   :func:`from_dict` raises instead.
3. **Overrides are typed by the target field, not by the string.** ``--set
   train.lr=1e-3`` must become a float because ``TrainConfig.lr`` is a float,
   and ``--set diarizer.spawn_enabled=false`` must become ``False`` and not the
   truthy string ``"false"`` — the classic silent-ablation-failure bug.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml


@dataclass
class DataConfig:
    """The procedural conversation generator.

    Ground truth is exact *by construction*: turns are sampled first, features
    are rendered from them, so there is no annotation error anywhere in the
    evaluation. That is the whole reason this project generates data rather than
    downloading it (see ``docs/METHOD.md`` §2).
    """

    #: Feature dimension (log-mel-like channels).
    n_features: int = 40
    #: Frames per second. 100 fps = a 10 ms hop, the usual speech convention.
    frame_rate: int = 100
    #: Recording length in seconds.
    duration_s: float = 60.0
    #: Inclusive range for the number of speakers per recording. The diarizer is
    #: never told which value was drawn.
    n_speakers_range: tuple[int, int] = (2, 5)
    #: Log-normal turn length: median seconds and sigma in log space.
    turn_median_s: float = 2.4
    turn_sigma: float = 0.75
    #: Probability that a turn overlaps its predecessor rather than following a pause.
    overlap_prob: float = 0.45
    #: Overlap extent as a fraction of the shorter of the two turns.
    overlap_frac_max: float = 0.6
    #: Inter-turn silence, uniform in this range (seconds).
    pause_range_s: tuple[float, float] = (0.05, 0.7)
    #: Speaker-timbre separability, i.e. the scale of the 8-dimensional identity
    #: envelope. Calibrated so the task is genuinely learnable but not trivial:
    #: at these three values an *untrained* embedder scores 0.281 on the 12-way
    #: batch task (chance 0.083) and a trained one 0.708 after 300 steps. At the
    #: original speaker_scale=1.0 / channel_sd=0.35 an untrained network already
    #: scored 0.868, which would have made the whole experiment a measurement of
    #: random projections rather than of learned embeddings.
    speaker_scale: float = 0.5
    #: Additive observation noise sd in feature units.
    noise_sd: float = 0.9
    #: Per-segment channel offset sd. Isotropic in R^F while identity lives in an
    #: 8-dimensional subspace, so the embedder must *learn* to project identity
    #: out of the channel rather than reading it off the raw features.
    channel_sd: float = 1.5
    #: Number of shared phonetic templates. Content varies within a speaker.
    n_phones: int = 24
    #: Disjoint speaker-identity pools, so test speakers are never trained on.
    n_train_speakers: int = 160
    n_test_speakers: int = 60
    #: Recording counts.
    n_train_recordings: int = 0  # training samples segments directly, not recordings
    n_dev_recordings: int = 8
    n_test_recordings: int = 24
    #: Optional real-dataset directory (RTTM + features). ``None`` = generated data.
    real_root: str | None = None


@dataclass
class EmbedderConfig:
    """Causal dilated-convolution speaker embedder.

    ``causal=True`` is the mechanism under test, not a detail: it is what makes
    a frame's representation independent of every later frame, which
    ``tests/test_causality.py`` asserts bit-identically. Setting it False is the
    non-causal ablation and makes the "online" claim false by construction.
    """

    channels: int = 64
    kernel_size: int = 3
    #: One dilated block per entry. Receptive field = 1 + 2*sum(dilations) frames
    #: for kernel 3, i.e. 31 frames = 310 ms at the default 100 fps.
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    embed_dim: int = 32
    dropout: float = 0.1
    causal: bool = True
    #: Statistics pooling over a window uses mean and sd when True, mean only otherwise.
    use_std_pooling: bool = True


@dataclass
class TrainConfig:
    """Prototypical (GE2E-style) metric learning over generated segments."""

    steps: int = 900
    #: Speakers per batch (N) and segments per speaker (M). Loss is N*M-way.
    n_speakers_per_batch: int = 12
    n_segments_per_speaker: int = 4
    #: Segment length in frames used during training (1.0 s at 100 fps).
    segment_frames: int = 100
    lr: float = 3e-3
    weight_decay: float = 1e-4
    #: Learnable scale/bias on the cosine logits, as in GE2E (Wan et al., 2018).
    init_logit_scale: float = 10.0
    init_logit_bias: float = -5.0
    grad_clip: float = 5.0
    log_every: int = 50
    threads: int = 2


@dataclass
class DiarizerConfig:
    """The online mechanism and its switches.

    ``latency_budget_ms`` is the axis of this project. It is the maximum time a
    window's label may be withheld after that window's audio has arrived. Every
    other field is either a mechanism that the budget enables
    (``bounded_window``) or an ablation switch.
    """

    #: Embedding window length and hop, in milliseconds.
    window_ms: float = 1000.0
    hop_ms: float = 250.0
    #: Emission delay budget. 0 = decide the instant the window closes.
    latency_budget_ms: float = 500.0
    #: Use the within-budget lookahead to form a local micro-cluster before
    #: assigning. Off = decide from the single window embedding (the ablation).
    bounded_window: bool = True
    #: Cosine-similarity threshold for local micro-cluster agglomeration.
    micro_cluster_threshold: float = 0.85
    #: Cosine similarity below which no existing speaker matches and a new one
    #: is created.
    spawn_threshold: float = 0.83
    #: Allow new speakers after the warm-up prefix. Off = ablation.
    spawn_enabled: bool = True
    #: Windows in the warm-up prefix during which spawning is always allowed.
    warmup_windows: int = 8
    #: Hard cap on tracked speakers — this is what makes memory O(1) in length.
    max_speakers: int = 12
    #: Centroid update rate. 0 < alpha <= 1; 1.0 = replace, small = sticky.
    centroid_momentum: float = 0.10
    #: If given, the true speaker count is supplied (oracle-count variant).
    oracle_n_speakers: bool = False
    #: Softmax temperature on cosine similarities for the confidence score.
    confidence_temperature: float = 0.12
    #: Method selector: online | naive_online | offline_ahc | offline_spectral.
    method: str = "online"
    #: Offline agglomerative stopping threshold (cosine distance).
    offline_ahc_threshold: float = 0.22
    #: Maximum K considered by the spectral eigengap search.
    offline_spectral_max_k: int = 10
    #: Row-wise affinity thresholding percentile for the spectral refinement of
    #: Wang et al. (2018). 0 disables it. This makes the offline baseline
    #: *stronger*, which is the point of a fair reference.
    offline_spectral_percentile: float = 0.80
    #: k-means restarts inside spectral clustering. Deterministic given the seed.
    offline_kmeans_restarts: int = 5
    #: Frames of causal hangover and onset run for the energy VAD.
    vad_hangover_frames: int = 12
    vad_onset_frames: int = 3
    #: Fitted on the dev split by scripts/train.py and stored in the checkpoint;
    #: this default is only used if a diarizer is built without a fitted model.
    vad_threshold: float = 0.0


@dataclass
class EvalConfig:
    """Scoring options and statistics."""

    #: Forgiveness collar in seconds applied around every reference boundary.
    #: 0.0 scores every frame, which is stricter than the NIST convention and is
    #: this project's primary number.
    collar_s: float = 0.0
    #: Score overlapped regions. Excluding them is common and flatters every
    #: system; this project reports both.
    score_overlap: bool = True
    bootstrap_resamples: int = 2000
    ci_level: float = 0.95
    #: Calibration binning.
    n_calibration_bins: int = 15
    #: Efficiency benchmark: the standard's warm-up floor is 8 and 25 repeats.
    bench_warmup: int = 8
    bench_repeats: int = 25


@dataclass
class Config:
    """The whole experiment."""

    name: str = "default"
    seed: int = 0
    out_root: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return _asdict_plain(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @property
    def frames_per_window(self) -> int:
        return max(1, int(round(self.diarizer.window_ms * self.data.frame_rate / 1000.0)))

    @property
    def frames_per_hop(self) -> int:
        return max(1, int(round(self.diarizer.hop_ms * self.data.frame_rate / 1000.0)))

    @property
    def budget_windows(self) -> int:
        """Latency budget expressed in whole hops of lookahead.

        A budget shorter than one hop buys nothing: the next window does not
        exist yet, so ``floor`` here is the honest conversion and the reason the
        DER-vs-latency curve is flat below one hop.
        """
        return int(self.diarizer.latency_budget_ms // self.diarizer.hop_ms)


def _asdict_plain(obj: Any) -> Any:
    """``dataclasses.asdict`` but tuples become lists so YAML stays clean."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict_plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #
def _resolved_type(cls: type, name: str) -> Any:
    """Field annotations are strings under ``from __future__ import annotations``."""
    import typing

    return typing.get_type_hints(cls).get(name, Any)


def _nested_dataclass_type(cls: type, name: str) -> type | None:
    t = _resolved_type(cls, name)
    return t if isinstance(t, type) and is_dataclass(t) else None


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce ``value`` to the annotated ``target_type``.

    Handles the annotation forms this project actually uses: ``int``, ``float``,
    ``bool``, ``str``, ``tuple[...]``, ``list[...]`` and ``X | None``. Anything
    else passes through, because guessing is worse than a clear failure later.
    """
    origin = get_origin(target_type)

    if origin is not None and type(None) in get_args(target_type):
        if value is None or (isinstance(value, str) and value.lower() in {"none", "null", ""}):
            return None
        inner = [a for a in get_args(target_type) if a is not type(None)]
        return _coerce(value, inner[0]) if len(inner) == 1 else value

    if origin in (tuple, list):
        args = get_args(target_type)
        if isinstance(value, str):
            value = [p for p in value.strip("[]() ").split(",") if p.strip() != ""]
        seq = list(value)
        if args and args[-1] is Ellipsis:
            seq = [_coerce(v, args[0]) for v in seq]
        elif args:
            seq = [_coerce(v, args[i]) if i < len(args) else v for i, v in enumerate(seq)]
        return tuple(seq) if origin is tuple else seq

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes", "on"}:
                return True
            if low in {"false", "0", "no", "off"}:
                return False
            raise ValueError(f"Cannot read {value!r} as a bool")
        return bool(value)

    if target_type is int:
        # float() first so "1e3" and 1.0 both work; reject a real 1.5 -> 1 truncation.
        f = float(value)
        if f != int(f):
            raise ValueError(f"{value!r} is not an integer")
        return int(f)

    if target_type is float:
        return float(value)

    if target_type is str:
        return str(value)

    return value


def from_dict(payload: dict[str, Any], cls: type = Config) -> Any:
    """Build a dataclass from a nested mapping, rejecting unknown keys.

    Rejecting unknown keys is the point. A typo in an ablation switch that is
    silently dropped produces a run that *looks* like an ablation, is not one,
    and is indistinguishable from the baseline in the results table.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) for {cls.__name__}: {unknown}; known: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in payload:
            continue
        nested = _nested_dataclass_type(cls, name)
        if nested is not None:
            kwargs[name] = from_dict(payload[name] or {}, nested)
        else:
            kwargs[name] = _coerce(payload[name], _resolved_type(cls, name))
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# YAML with _base_ inheritance
# --------------------------------------------------------------------------- #
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive mapping merge; ``override`` wins. Neither input is mutated."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_tree(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config, resolving ``_base_`` (a path or list of paths) first.

    Bases resolve relative to the *including* file so ``configs/`` can be moved
    wholesale. A cycle raises with the full chain rather than recursing into a
    stack overflow, because a stack overflow is a bad error message.
    """
    p = Path(path).resolve()
    if p in _seen:
        chain = " -> ".join(str(s) for s in (*_seen, p))
        raise ValueError(f"Cyclic _base_ inheritance: {chain}")
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config {p} must be a mapping at top level, got {type(raw).__name__}")
    raw = dict(raw)

    bases = raw.pop("_base_", None)
    merged: dict[str, Any] = {}
    if bases:
        for b in [bases] if isinstance(bases, str) else list(bases):
            merged = deep_merge(merged, load_yaml_tree(p.parent / b, (*_seen, p)))
    return deep_merge(merged, raw)


def parse_set_override(expr: str) -> tuple[list[str], str]:
    """Split ``a.b.c=value``. The value stays a string; :func:`_coerce` types it."""
    if "=" not in expr:
        raise ValueError(f"--set expects key=value, got {expr!r}")
    key, _, value = expr.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--set expects a non-empty key, got {expr!r}")
    return key.split("."), value.strip()


def apply_overrides(payload: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``--set`` expressions onto a nested mapping, returning a new mapping."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in payload.items()}
    for expr in overrides or []:
        parts, value = parse_set_override(expr)
        node = out
        for part in parts[:-1]:
            nxt = node.get(part)
            node[part] = dict(nxt) if isinstance(nxt, dict) else {}
            node = node[part]
        node[parts[-1]] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    **kwargs: Any,
) -> Config:
    """The single entry point every script uses.

    Precedence, lowest first: dataclass defaults < ``_base_`` chain < the named
    YAML file < ``--set`` overrides < explicit ``kwargs``. Asserted in
    ``tests/test_config.py::test_override_precedence_chain``.
    """
    payload = load_yaml_tree(path) if path is not None else {}
    payload = apply_overrides(payload, overrides)
    for k, v in kwargs.items():
        payload[k] = v
    return from_dict(payload, Config)


def save_config(cfg: Config, path: str | Path) -> Path:
    """Dump the *resolved* config with LF endings alongside a run's artefacts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cfg.to_yaml())
    return p


def replace(cfg: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **changes)
