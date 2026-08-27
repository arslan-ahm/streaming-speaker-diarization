"""Statistical comparison, with no SciPy: bootstrap CIs, Wilcoxon, Holm, noise scale.

Reporting that the online system scored DER 0.184 and the offline reference
0.171 is not a result. It becomes one only when the gap is placed against two
different scales:

* **Per-recording spread**, addressed by a paired test over test recordings.
  This conditions on *one* trained embedder per method: it answers "do these two
  sets of weights differ on this test set", not "is this method better". The
  build standard is explicit about that distinction and so is every table here.
* **Run-to-run spread**, addressed by the seed study. Two runs of the same config
  differ by a random amount with standard deviation ``sqrt(2) * sd`` where ``sd``
  is the across-seed standard deviation of a single run. Any difference smaller
  than that is noise, whatever its p-value. :func:`verdict` is the function that
  refuses to let a claim past this gate.

Everything is hand-rolled because this environment's SciPy is broken (see
``metrics/assignment.py``), and because a 30-line Wilcoxon whose tie handling you
can read is worth more here than an import. The exact null distribution is used
whenever it is valid, which for the shipped 24-recording test set it is.

References
----------
Wilcoxon, *Individual comparisons by ranking methods*, Biometrics Bulletin 1945.
Holm, *A simple sequentially rejective multiple test procedure*, Scand. J.
Statist. 1979.
Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 — percentile method.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

#: Above this n the exact signed-rank distribution is skipped for the normal
#: approximation. 25 keeps the DP under a few thousand cells and the
#: approximation is excellent by then.
EXACT_WILCOXON_MAX_N = 25


@dataclass
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lower) and np.isfinite(self.upper)) and (
            self.lower > 0.0 or self.upper < 0.0
        )


@dataclass
class Comparison:
    """A paired comparison of two methods on the same recordings."""

    name_a: str
    name_b: str
    metric: str
    mean_a: float
    mean_b: float
    #: ``mean_a - mean_b`` with a paired bootstrap interval.
    difference: Interval
    #: Two-sided Wilcoxon signed-rank p-value on the paired differences.
    p_value: float
    #: Paired Cohen's d: mean difference over the sd of the differences.
    effect_size: float
    n: int
    p_adjusted: float | None = None
    #: ``sqrt(2) * seed_sd`` for this metric, if a seed study is available.
    noise_scale: float | None = None

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return bool(np.isfinite(p) and p < 0.05)

    @property
    def noise_ratio(self) -> float:
        """``|difference| / noise_scale``. NaN when no seed study is attached."""
        if not self.noise_scale or not np.isfinite(self.noise_scale) or self.noise_scale <= 0:
            return float("nan")
        return abs(self.difference.estimate) / self.noise_scale

    def verdict(self) -> str:
        return verdict(self.difference.estimate, self.noise_scale, self.p_value, self.p_adjusted)

    def to_dict(self) -> dict[str, object]:
        d = {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.difference.estimate,
            "ci_lower": self.difference.lower,
            "ci_upper": self.difference.upper,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "n": self.n,
            "noise_scale": self.noise_scale,
            "noise_ratio": self.noise_ratio,
            "verdict": self.verdict(),
        }
        return d


def verdict(
    delta: float,
    noise_scale: float | None,
    p_value: float,
    p_adjusted: float | None = None,
) -> str:
    """Turn a difference into one of three honest words.

    The gate is deliberately conservative and the noise scale outranks the
    p-value, because a paired per-recording test can be arbitrarily significant
    about a difference that a different random seed would erase.

    * ``inside noise`` — below the run-to-run scale, whatever the p-value. This
      test comes first on purpose.
    * ``survives`` — corrected p < 0.05 **and** ``|delta| >= 2 * sqrt(2) * sd``.
    * ``suggestive`` — corrected p < 0.05 but only ``1-2x`` the noise scale.
    * ``not significant`` — above the noise scale but the paired test does not
      reject. Usually means the per-recording spread is large.
    * ``untested`` — no seed study attached, so the claim cannot be graded.
    """
    p = p_value if p_adjusted is None else p_adjusted
    sig = bool(np.isfinite(p) and p < 0.05)
    if noise_scale is None or not np.isfinite(noise_scale) or noise_scale <= 0:
        return "untested (no seed study)"
    ratio = abs(delta) / noise_scale
    if ratio < 1.0:
        return "inside noise"
    if not sig:
        return "not significant"
    return "survives" if ratio >= 2.0 else "suggestive"


def noise_scale_from_seeds(values: np.ndarray) -> float:
    """``sqrt(2) * sd`` — the run-to-run scale of a *difference* between two runs.

    Two independent runs each with variance ``s^2`` differ with variance
    ``2 s^2``. Comparing a between-method delta against a single-run ``sd``
    instead of ``sqrt(2) sd`` understates the noise by 41%, which is enough to
    turn "inside noise" into "significant" on its own.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(math.sqrt(2.0) * v.std(ddof=1))


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval for a summary of ``values``.

    Non-parametric on purpose: per-recording DER is right-skewed (a floor at 0
    and a tail of recordings where two speakers got merged), so a t-interval
    would be wrong in exactly the cases that matter. ``NaN`` entries are dropped
    and counted in ``Interval.n``, per the standard's "count the items that
    contributed" rule.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    reduce = np.mean if statistic == "mean" else np.median
    if data.size == 0:
        nan = float("nan")
        return Interval(nan, nan, nan, level, 0)
    point = float(reduce(data))
    if data.size < 2:
        return Interval(point, point, point, level, int(data.size))

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(n_resamples, data.size))
    replicates = reduce(data[picks], axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(replicates, [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, int(data.size))


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples *recording indices*, not the two arrays independently, which is
    what preserves the pairing and earns the tighter interval that a paired
    design is for.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must match in shape: {x.shape} vs {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    return bootstrap_ci((x - y)[valid], n_resamples, level, seed, statistic="mean")


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank, hand-rolled
# --------------------------------------------------------------------------- #
def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties averaged. (``scipy.stats.rankdata`` equivalent.)"""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _signed_rank_exact_p(w_plus: float, ranks: np.ndarray) -> float:
    """Two-sided exact p-value by enumerating the null sign distribution.

    Valid only when the absolute differences are distinct (integer ranks). The
    DP counts, for every achievable value of ``W+``, how many of the ``2^n`` sign
    assignments produce it — the standard exact signed-rank construction.
    """
    ints = np.rint(ranks).astype(np.int64)
    total = int(ints.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in ints:
        # Out-of-place: ``counts[r:] += counts[:-r]`` aliases its own source and
        # would double-count subsets when r < len(counts) / 2.
        nxt = counts.copy()
        nxt[r:] += counts[:-r]
        counts = nxt
    n_total = counts.sum()
    w = int(round(w_plus))
    p_low = counts[: w + 1].sum() / n_total
    p_high = counts[w:].sum() / n_total
    return float(min(1.0, 2.0 * min(p_low, p_high)))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray | None = None) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test. Returns ``(statistic W+, p-value)``.

    Zero differences are dropped (``zero_method="wilcox"``, the usual
    convention). The exact null distribution is used when ``n <=
    EXACT_WILCOXON_MAX_N`` and the non-zero absolute differences are all
    distinct; otherwise a normal approximation with the standard tie correction

        var = n(n+1)(2n+1)/24 - sum(t^3 - t)/48

    and a continuity correction of 0.5. With ties present the exact test is not
    valid, and silently using it anyway is a real and common bug.

    The p-value is ``NaN`` when every difference is zero — two identical systems
    are not "significantly the same", the test is simply undefined.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.zeros_like(x) if b is None else np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must match in shape: {x.shape} vs {y.shape}")
    d = (x - y)[np.isfinite(x) & np.isfinite(y)]
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(d[0] > 0), 1.0

    absd = np.abs(d)
    ranks = _rankdata_average(absd)
    w_plus = float(ranks[d > 0].sum())

    distinct = np.unique(absd).size == n
    if n <= EXACT_WILCOXON_MAX_N and distinct:
        return w_plus, _signed_rank_exact_p(w_plus, ranks)

    mean_w = n * (n + 1) / 4.0
    _, tie_counts = np.unique(absd, return_counts=True)
    tie_term = float(((tie_counts.astype(np.float64) ** 3) - tie_counts).sum())
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return w_plus, float("nan")
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    return w_plus, float(min(1.0, 2.0 * (1.0 - _normal_cdf(max(z, 0.0)))))


# --------------------------------------------------------------------------- #
# Comparison and multiplicity
# --------------------------------------------------------------------------- #
def compare(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    metric: str = "metric",
    n_resamples: int = 2000,
    seed: int = 0,
    noise_scale: float | None = None,
) -> Comparison:
    """Full paired comparison of two methods over the same recordings."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y
    _, p = wilcoxon_signed_rank(x, y)
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    effect = float(diff.mean() / sd) if sd > 0 else 0.0
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        mean_a=float(x.mean()) if x.size else float("nan"),
        mean_b=float(y.mean()) if y.size else float("nan"),
        difference=paired_bootstrap_difference(x, y, n_resamples, seed=seed),
        p_value=p,
        effect_size=effect,
        n=int(diff.size),
        noise_scale=noise_scale,
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Holm-Bonferroni step-down correction, applied in place and returned.

    Sorts raw p-values ascending and scales the ``i``-th by ``m - i``, enforcing
    monotonicity so a corrected p-value never decreases down the list. Uniformly
    more powerful than plain Bonferroni at the same family-wise error rate.
    ``NaN`` p-values are excluded from ``m`` — an undefined test does not consume
    family budget.
    """
    testable = [c for c in comparisons if np.isfinite(c.p_value)]
    m = len(testable)
    for c in comparisons:
        if not np.isfinite(c.p_value):
            c.p_adjusted = float("nan")
    order = sorted(range(m), key=lambda i: testable[i].p_value)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * testable[idx].p_value)
        running = max(running, adj)
        testable[idx].p_adjusted = running
    _ = alpha
    return comparisons


def summarize_seed_study(per_seed: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """``{metric: [value per seed]}`` to ``{metric: {mean, sd, noise_scale, n_seeds}}``."""
    out: dict[str, dict[str, float]] = {}
    for metric, values in per_seed.items():
        v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
        out[metric] = {
            "mean": float(v.mean()) if v.size else float("nan"),
            "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "noise_scale": noise_scale_from_seeds(v),
            "n_seeds": float(v.size),
        }
    return out
