"""Diarization Error Rate, decomposed, over an optimal one-to-one speaker mapping.

DER is a *time*-weighted error rate with overlap counted with multiplicity, which
is easy to get subtly wrong. The definition implemented here is NIST md-eval's,
expressed on a frame grid. At frame ``t`` let ``R_t`` be the number of active
reference speakers, ``H_t`` the number of active hypothesis speakers, and ``C_t``
the number of reference speakers whose *mapped* hypothesis label is also active:

    miss_t      = max(0, R_t - H_t)
    falarm_t    = max(0, H_t - R_t)
    confusion_t = min(R_t, H_t) - C_t
    DER         = sum_t (miss_t + falarm_t + confusion_t) / sum_t R_t

Three things worth stating because they are where implementations diverge:

**The denominator is reference speaker-time, not wall-clock.** A frame with two
simultaneous speakers contributes 2. This is why DER can exceed 1.0 and why an
overlap-blind system has an irreducible floor equal to the overlap fraction.

**The mapping is global and optimal, and is computed once per recording.** Not
per segment, not greedily. ``metrics/assignment.py`` explains why greedy inflates
DER, with a worked 2x2 counterexample. Note the mapping is chosen to maximise
``C_t``, i.e. to minimise confusion only — that is md-eval's choice, and it is
not the same as minimising total DER, because miss and false alarm do not depend
on the mapping at all. They don't, so the two objectives coincide here; the
argument is spelled out in ``docs/METHOD.md`` §4.

**Undefined is NaN.** A recording with no reference speech has no DER. Returning
0.0 there would silently pull down the mean of a sweep.

Also implemented: **JER** (Jaccard Error Rate, DIHARD II), which averages a
per-reference-speaker error and so weights a speaker who says three words the
same as one who talks for a minute. It disagrees with DER on purpose and is
reported alongside it.

References
----------
NIST, *Rich Transcription 2009 Evaluation Plan* — md-eval DER.
Ryant et al., *The Second DIHARD Diarization Challenge*, Interspeech 2019 — JER.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .assignment import linear_sum_assignment


@dataclass
class DERResult:
    """DER and its three components, all as rates over the same denominator."""

    der: float
    miss: float
    false_alarm: float
    confusion: float
    #: ``sum_t R_t`` over scored frames — the denominator. 0 means DER is NaN.
    total_ref_frames: int
    #: Number of frames that passed the collar/overlap scoring mask.
    scored_frames: int
    n_ref_speakers: int
    n_hyp_speakers: int
    #: Optimal ``reference label -> hypothesis label``. Unmapped refs are absent.
    mapping: dict[int, int] = field(default_factory=dict)
    jer: float = float("nan")
    #: Signed error in the speaker count, ``hyp - ref``.
    speaker_count_error: int = 0

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["mapping"] = {int(k): int(v) for k, v in self.mapping.items()}
        return d

    def __str__(self) -> str:
        return (
            f"DER={self.der:.4f} (miss={self.miss:.4f} fa={self.false_alarm:.4f} "
            f"conf={self.confusion:.4f}) ref={self.n_ref_speakers} hyp={self.n_hyp_speakers}"
        )


def _boundary_collar_mask(ref: np.ndarray, collar_frames: int) -> np.ndarray:
    """Frames to *keep*: those at least ``collar_frames`` from a reference boundary.

    md-eval's collar forgives annotation slop around every reference segment
    boundary. This project's primary number uses ``collar_frames=0`` because the
    generated reference has no slop to forgive — a collar there would only hide
    real boundary error, and boundary error is precisely what a latency budget
    trades against. The collar path exists so numbers on real RTTM data (where
    slop is genuine) stay comparable to the literature.
    """
    keep = np.ones(ref.shape[0], dtype=bool)
    if collar_frames <= 0:
        return keep
    active = ref.any(axis=1).astype(np.int8)
    change = np.flatnonzero(np.diff(np.concatenate(([0], active, [0]))) != 0)
    # Per-speaker boundaries too: a speaker change inside continuous speech is a
    # boundary even though total activity never drops to zero.
    for s in range(ref.shape[1]):
        col = ref[:, s].astype(np.int8)
        change = np.union1d(change, np.flatnonzero(np.diff(np.concatenate(([0], col, [0]))) != 0))
    n = ref.shape[0]
    for b in change:
        keep[max(0, b - collar_frames) : min(n, b + collar_frames)] = False
    return keep


def der(
    ref: np.ndarray,
    hyp: np.ndarray,
    collar_frames: int = 0,
    score_overlap: bool = True,
) -> DERResult:
    """Score a hypothesis against a reference.

    Args:
        ref: ``(T, R)`` bool reference activity, multi-label for overlap.
        hyp: ``(T, H)`` bool hypothesis activity. ``H`` need not equal ``R``.
        collar_frames: Forgiveness collar, in frames, around every reference
            boundary. 0 scores every frame.
        score_overlap: When False, frames with two or more reference speakers are
            excluded. This flatters every system and is reported as a secondary
            column, never as the headline.

    Returns:
        A :class:`DERResult`. ``der`` is NaN when no reference speech is scored.
    """
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)
    if ref.ndim != 2 or hyp.ndim != 2:
        raise ValueError(f"ref and hyp must be 2-D, got {ref.shape} and {hyp.shape}")
    if ref.shape[0] != hyp.shape[0]:
        raise ValueError(f"frame count mismatch: ref {ref.shape[0]} vs hyp {hyp.shape[0]}")

    n_ref, n_hyp = ref.shape[1], hyp.shape[1]
    keep = _boundary_collar_mask(ref, collar_frames)
    if not score_overlap:
        keep &= ref.sum(axis=1) <= 1
    r, h = ref[keep], hyp[keep]

    counts_r = r.sum(axis=1).astype(np.int64)
    counts_h = h.sum(axis=1).astype(np.int64)
    total = int(counts_r.sum())

    # Co-occurrence in frames; maximise it to fix the label permutation.
    overlap = (r.astype(np.int64).T @ h.astype(np.int64)) if (n_ref and n_hyp) else np.zeros(
        (n_ref, n_hyp), dtype=np.int64
    )
    mapping: dict[int, int] = {}
    if n_ref and n_hyp and overlap.size:
        rows, cols = linear_sum_assignment(overlap.astype(np.float64), maximize=True)
        mapping = {int(i): int(j) for i, j in zip(rows, cols, strict=True)}

    correct = np.zeros(r.shape[0], dtype=np.int64)
    for i, j in mapping.items():
        correct += (r[:, i] & h[:, j]).astype(np.int64)

    miss = int(np.maximum(0, counts_r - counts_h).sum())
    falarm = int(np.maximum(0, counts_h - counts_r).sum())
    conf = int((np.minimum(counts_r, counts_h) - correct).sum())

    if total == 0:
        nan = float("nan")
        return DERResult(nan, nan, nan, nan, 0, int(keep.sum()), n_ref, n_hyp, mapping,
                         nan, n_hyp - n_ref)

    return DERResult(
        der=(miss + falarm + conf) / total,
        miss=miss / total,
        false_alarm=falarm / total,
        confusion=conf / total,
        total_ref_frames=total,
        scored_frames=int(keep.sum()),
        n_ref_speakers=n_ref,
        n_hyp_speakers=n_hyp,
        mapping=mapping,
        jer=jer(r, h, mapping),
        speaker_count_error=n_hyp - n_ref,
    )


def jer(ref: np.ndarray, hyp: np.ndarray, mapping: dict[int, int]) -> float:
    """Jaccard Error Rate over the *same* mapping DER used.

    Per DIHARD II: for each reference speaker ``i`` mapped to hypothesis ``j``,
    the error is ``(miss_i + falarm_i) / total_i`` where ``total_i`` is reference
    speaker ``i``'s own speech time. An unmapped reference speaker scores 1.0.
    Averaging over speakers rather than over time is the point — it stops one
    dominant talker from hiding a completely missed one.
    """
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)
    if ref.shape[1] == 0:
        return float("nan")
    errs = []
    for i in range(ref.shape[1]):
        total = int(ref[:, i].sum())
        if total == 0:
            continue  # a reference speaker with no speech has no JER (standard §8)
        j = mapping.get(i)
        if j is None:
            errs.append(1.0)
            continue
        inter = int((ref[:, i] & hyp[:, j]).sum())
        miss = total - inter
        falarm = int(hyp[:, j].sum()) - inter
        errs.append((miss + falarm) / total)
    return float(np.mean(errs)) if errs else float("nan")


def labels_to_matrix(
    labels: np.ndarray, n_frames: int, n_speakers: int | None = None
) -> np.ndarray:
    """Frame labels (``-1`` = non-speech) to a ``(T, H)`` one-hot activity matrix.

    Hypothesis labels are densified to ``0..H-1`` in order of first appearance so
    that ``H`` is the number of speakers the system actually *used*, not the size
    of the id space it allocated from. That distinction matters: a diarizer that
    spawns speaker 9 and then never uses it should not be charged with a
    speaker-count error.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] != n_frames:
        raise ValueError(f"labels length {labels.shape[0]} != n_frames {n_frames}")
    used = [int(v) for v in dict.fromkeys(labels[labels >= 0].tolist())]
    remap = {v: k for k, v in enumerate(used)}
    h = int(n_speakers) if n_speakers is not None else len(used)
    out = np.zeros((n_frames, h), dtype=bool)
    for t, lab in enumerate(labels.tolist()):
        if lab >= 0 and remap[lab] < h:
            out[t, remap[lab]] = True
    return out
