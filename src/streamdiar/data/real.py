"""Optional real-dataset path: WAV + RTTM in, :class:`Recording` out.

**Nothing in this module is required.** No test imports it, no shipped result
depends on it, and it is never reached unless ``data.real_root`` is set. It
exists so the claim "this pipeline is not synthetic-only by construction" can be
checked by a reader with a copy of AMI or VoxConverse, and so the feature
front-end is written down rather than hand-waved.

Everything here is stdlib plus NumPy on purpose. ``pyannote.audio``,
``speechbrain`` and ``librosa`` would each be a heavier install than the rest of
this repository combined, and a reader who cannot install them cannot read the
code either. The log-mel front-end below is 40 lines and matches the standard
recipe: 25 ms Hann frames, 10 ms hop, 40 triangular mel filters, natural log.

Caveat that must be stated wherever real-data numbers appear: an RTTM reference
is a *human annotation* with boundary error on the order of tens of
milliseconds, so DER differences smaller than that are not resolvable here. The
generated data has no such floor, which is why it carries the shipped results.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .generator import Recording, Turn


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """O'Shaughnessy's mel scale, the one every toolkit uses."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float = 20.0, fmax: float | None = None
) -> np.ndarray:
    """``(n_mels, n_fft // 2 + 1)`` triangular filterbank, area-normalised rows."""
    fmax = float(fmax if fmax is not None else sample_rate / 2)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * np.asarray(edges) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        if mid > lo:
            fb[m, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
        total = fb[m].sum()
        if total > 0:
            fb[m] /= total
    return fb


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_mels: int = 40,
    frame_rate: int = 100,
    win_ms: float = 25.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """``(T, n_mels)`` natural-log mel features. Same units as the generator's output.

    ``np.log`` and not ``10*log10``: the generator renders in natural log-power
    because overlap is then exactly ``logaddexp``, and the two front-ends must
    agree or a model trained on one cannot be evaluated on the other.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = max(1, int(round(sample_rate / frame_rate)))
    win = max(hop, int(round(sample_rate * win_ms / 1000.0)))
    n_fft = 1 << (win - 1).bit_length()

    n_frames = 1 + max(0, (len(x) - win) // hop)
    if n_frames <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = spec @ mel_filterbank(n_mels, n_fft, sample_rate).T
    return np.log(mel + eps).astype(np.float32)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV with the stdlib. Returns ``(samples in [-1, 1], rate)``."""
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        n_channels = fh.getnchannels()
        width = fh.getsampwidth()
        raw = fh.readframes(fh.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate


def parse_rttm(text: str) -> dict[str, list[tuple[float, float, str]]]:
    """Parse RTTM ``SPEAKER`` lines into ``{recording: [(start_s, end_s, label)]}``.

    Malformed lines are skipped rather than raising: real RTTM files in the wild
    carry comments and non-SPEAKER record types, and a loader that dies on the
    first ``SPKR-INFO`` line is useless.
    """
    out: dict[str, list[tuple[float, float, str]]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            start, dur = float(parts[3]), float(parts[4])
        except ValueError:
            continue
        if dur <= 0:
            continue
        out.setdefault(parts[1], []).append((start, start + dur, parts[7]))
    return out


def load_real_recording(
    wav_path: str | Path,
    rttm_path: str | Path,
    n_mels: int = 40,
    frame_rate: int = 100,
) -> Recording:
    """Build a :class:`Recording` from a WAV and its RTTM reference.

    Speaker labels are remapped to dense ``0..K-1`` in order of first
    appearance, matching the generator's convention so every metric and every
    diarizer works unchanged on both paths.
    """
    name = Path(wav_path).stem
    samples, rate = read_wav(wav_path)
    features = log_mel_spectrogram(samples, rate, n_mels=n_mels, frame_rate=frame_rate)

    segments = parse_rttm(Path(rttm_path).read_text(encoding="utf-8")).get(name)
    if segments is None:
        merged = parse_rttm(Path(rttm_path).read_text(encoding="utf-8"))
        if len(merged) != 1:
            raise KeyError(f"{name!r} not in {rttm_path} (found {sorted(merged)})")
        segments = next(iter(merged.values()))

    order: dict[str, int] = {}
    turns: list[Turn] = []
    for start, end, label in sorted(segments):
        if label not in order:
            order[label] = len(order)
        s = int(round(start * frame_rate))
        e = min(features.shape[0], int(round(end * frame_rate)))
        if e > s:
            turns.append(Turn(s, e, order[label]))

    return Recording(
        name=name,
        features=features,
        turns=turns,
        n_speakers=max(1, len(order)),
        frame_rate=frame_rate,
        pool_ids=[],
    )


def load_real_split(root: str | Path, frame_rate: int = 100, n_mels: int = 40) -> list[Recording]:
    """Load every ``*.wav`` under ``root`` that has a sibling ``*.rttm``."""
    root = Path(root)
    recs = []
    for wav in sorted(root.rglob("*.wav")):
        rttm = wav.with_suffix(".rttm")
        if rttm.exists():
            recs.append(load_real_recording(wav, rttm, n_mels=n_mels, frame_rate=frame_rate))
    if not recs:
        raise FileNotFoundError(
            f"No wav/rttm pairs under {root}. See scripts/download_real_data.py."
        )
    return recs
