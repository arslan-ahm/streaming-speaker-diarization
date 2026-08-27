"""Calibration and selective prediction over turn assignments.

The useful question for a live diarizer is not only "how often is the label
wrong" but "does the system know which labels are wrong". If it does, a
deployment can route the uncertain turns to a slower offline pass or to a human,
and buy back most of the accuracy the latency budget cost. That is the practical
payoff of the whole online design, so it is measured rather than asserted.

The unit of analysis is one **turn decision**: the online diarizer emits a label
for each window along with a confidence, and the label is *correct* when, under
the recording's optimal global speaker mapping, it matches a reference speaker
active in that window. Note two consequences that are stated wherever these
numbers appear:

* Correctness is defined *through the mapping*, which is itself estimated. A
  system that consistently swaps two labels is scored as wrong even though its
  partition is right. That is the same convention DER uses.
* Non-speech windows and windows the reference marks as overlap are excluded from
  the calibration set — for the first there is no correct speaker, and for the
  second a single-label system cannot be right in the sense being measured.
  Excluded counts are reported (``n_excluded``) so this is not a silent filter.

Metrics
-------
* **ECE** — equal-*width* binning, the original Guo et al. (2017) form.
* **ACE** — equal-*mass* binning. With a confident model almost every sample
  lands in the top width-bin, so ECE is dominated by one bin and understates
  miscalibration; equal-mass bins fix that and usually read higher.
* **MCE** — the worst bin, which is what a risk-averse deployment cares about.
* **Brier** and **NLL** — proper scoring rules, so they cannot be gamed by a
  model that only sharpens.
* **Risk-coverage / AURC** — abstention: sort by confidence, and plot error rate
  over the retained fraction. AURC is the area, lower is better.
* **Error-detection AUROC** — can the confidence rank the errors above the
  correct ones? 0.5 is useless, 1.0 is a perfect error detector.

References
----------
Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 — ECE,
temperature scaling.
Nixon et al., *Measuring Calibration in Deep Learning*, CVPRW 2019 — adaptive
(equal-mass) binning.
Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*,
NeurIPS 2017 — risk-coverage and AURC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    """Calibration and selective-prediction summary for one system."""

    ece: float
    ace: float
    mce: float
    brier: float
    nll: float
    accuracy: float
    mean_confidence: float
    #: ``mean_confidence - accuracy``. Positive = overconfident.
    overconfidence: float
    aurc: float
    #: AURC of an oracle that abstains on exactly the errors — the achievable floor.
    aurc_oracle: float
    error_detection_auroc: float
    n: int
    n_excluded: int = 0
    #: ``(bin_centre, bin_accuracy, bin_confidence, bin_count)`` for the reliability plot.
    bins: list[tuple[float, float, float, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.pop("bins", None)
        return d


def _clean(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct).astype(np.float64)
    if c.shape != y.shape:
        raise ValueError(f"shape mismatch: {c.shape} vs {y.shape}")
    keep = np.isfinite(c) & np.isfinite(y)
    return np.clip(c[keep], 0.0, 1.0), y[keep]


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, float, list[tuple[float, float, float, int]]]:
    """Equal-width ECE and MCE, plus the reliability-diagram bins.

    Empty bins contribute nothing and are omitted from ``bins`` rather than
    being reported as perfectly calibrated with zero samples.
    """
    c, y = _clean(confidence, correct)
    if c.size == 0:
        return float("nan"), float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    bins: list[tuple[float, float, float, int]] = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        acc = float(y[sel].mean())
        conf = float(c[sel].mean())
        gap = abs(acc - conf)
        ece += (n / c.size) * gap
        mce = max(mce, gap)
        bins.append((float(0.5 * (edges[b] + edges[b + 1])), acc, conf, n))
    return float(ece), float(mce), bins


def adaptive_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-mass (adaptive) calibration error.

    Bins are quantiles of the confidence distribution, so each holds roughly the
    same number of samples. For a sharp model this is the more informative of the
    two: equal-width ECE puts 90% of the samples in one bin and reports its
    average gap, which is a much weaker statement.
    """
    c, y = _clean(confidence, correct)
    if c.size == 0:
        return float("nan")
    n_bins = max(1, min(n_bins, c.size))
    order = np.argsort(c, kind="stable")
    splits = np.array_split(order, n_bins)
    total = 0.0
    for part in splits:
        if part.size == 0:
            continue
        total += (part.size / c.size) * abs(float(y[part].mean()) - float(c[part].mean()))
    return float(total)


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(coverage, risk)`` for abstention in descending-confidence order.

    ``risk[k]`` is the error rate over the ``k`` most-confident decisions. This is
    the curve a deployment reads to pick "defer the least-confident 20% of turns
    to the offline pass".
    """
    c, y = _clean(confidence, correct)
    if c.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(-c, kind="stable")
    errors = 1.0 - y[order]
    k = np.arange(1, c.size + 1)
    return k / c.size, np.cumsum(errors) / k


def area_under_risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AURC: mean risk over all coverage levels. Lower is better."""
    cov, risk = risk_coverage_curve(confidence, correct)
    return float(risk.mean()) if risk.size else float("nan")


def error_detection_auroc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for using ``-confidence`` to detect errors, via the Mann-Whitney identity.

    Computed as ``P(conf_correct > conf_error) + 0.5 P(equal)`` from rank sums,
    which handles ties exactly and needs no threshold sweep. NaN when one class
    is absent — an AUROC over a single class is undefined, not 0.5.
    """
    c, y = _clean(confidence, correct)
    pos = c[y > 0.5]  # correct decisions
    neg = c[y <= 0.5]  # errors
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < order.size:  # average ranks within ties
        j = i
        while j + 1 < order.size and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error of the confidence as a probability of being correct."""
    c, y = _clean(confidence, correct)
    return float(((c - y) ** 2).mean()) if c.size else float("nan")


def negative_log_likelihood(
    confidence: np.ndarray, correct: np.ndarray, eps: float = 1e-12
) -> float:
    """Mean NLL of the binary correctness event under the reported confidence.

    ``eps`` clipping is not cosmetic: an unclipped confidence of exactly 1.0 on a
    single wrong decision sends NLL to infinity and destroys the mean, which
    would make the metric a detector of one outlier rather than of calibration.
    """
    c, y = _clean(confidence, correct)
    if c.size == 0:
        return float("nan")
    c = np.clip(c, eps, 1.0 - eps)
    return float(-(y * np.log(c) + (1.0 - y) * np.log(1.0 - c)).mean())


def evaluate_calibration(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    n_excluded: int = 0,
) -> CalibrationResult:
    """All calibration and selective-prediction metrics in one pass."""
    c, y = _clean(confidence, correct)
    ece, mce, bins = expected_calibration_error(c, y, n_bins)
    acc = float(y.mean()) if c.size else float("nan")
    conf = float(c.mean()) if c.size else float("nan")
    return CalibrationResult(
        ece=ece,
        ace=adaptive_calibration_error(c, y, n_bins),
        mce=mce,
        brier=brier_score(c, y),
        nll=negative_log_likelihood(c, y),
        accuracy=acc,
        mean_confidence=conf,
        overconfidence=float(conf - acc) if c.size else float("nan"),
        aurc=area_under_risk_coverage(c, y),
        aurc_oracle=area_under_risk_coverage(y, y),
        error_detection_auroc=error_detection_auroc(c, y),
        n=int(c.size),
        n_excluded=int(n_excluded),
        bins=bins,
    )


def fit_temperature(
    logits: np.ndarray,
    correct: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Fit a single temperature by grid search on held-out NLL.

    The knob here is the temperature applied to the *similarity margin* before
    the sigmoid that produces a confidence, so it is a one-parameter recalibration
    exactly as in Guo et al. (2017). Grid search rather than LBFGS because the
    objective is one-dimensional and smooth: 121 evaluations of a closed form
    beat an optimiser dependency, and the result is deterministic.

    Fitted on the **dev** split only. Fitting on test would make every
    calibration number here meaningless, and it is the single easiest way to
    accidentally cheat at this metric.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(correct).astype(np.float64)
    keep = np.isfinite(z) & np.isfinite(y)
    z, y = z[keep], y[keep]
    if z.size == 0:
        return 1.0
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        nll = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Sigmoid of ``logits / temperature``, clipped into the open unit interval."""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1.0 - 1e-12)
