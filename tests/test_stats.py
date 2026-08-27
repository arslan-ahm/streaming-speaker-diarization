"""Statistics: exact Wilcoxon against enumeration and a published value, bootstrap, Holm.

The exact signed-rank p-value is checked two ways: against a full enumeration of
all ``2^n`` sign patterns, and against the value R's ``wilcox.test`` reports for
the standard paired example (V = 40, p = 0.03906). Getting a Wilcoxon
implementation subtly wrong — using the exact distribution when ties are present,
or forgetting the continuity correction — produces p-values that are plausible
and wrong, so this is not a formality.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from streamdiar.metrics.stats import (
    Comparison,
    Interval,
    _rankdata_average,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    noise_scale_from_seeds,
    paired_bootstrap_difference,
    summarize_seed_study,
    verdict,
    wilcoxon_signed_rank,
)


def enumerate_exact_p(d: np.ndarray) -> float:
    """Two-sided exact signed-rank p by enumerating every sign assignment."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = d.size
    ranks = _rankdata_average(np.abs(d))
    w = ranks[d > 0].sum()
    totals = np.array(
        [sum(ranks[i] for i in range(n) if signs[i]) for signs in itertools.product([0, 1], repeat=n)]
    )
    return float(min(1.0, 2.0 * min((totals <= w).mean(), (totals >= w).mean())))


class TestRankData:
    def test_ranks_start_at_one(self):
        assert _rankdata_average(np.array([5.0, 1.0, 3.0])).tolist() == [3.0, 1.0, 2.0]

    def test_ties_are_averaged(self):
        assert _rankdata_average(np.array([1.0, 1.0, 3.0])).tolist() == [1.5, 1.5, 3.0]

    def test_all_ties(self):
        assert _rankdata_average(np.ones(4)).tolist() == [2.5, 2.5, 2.5, 2.5]

    def test_sum_of_ranks_is_n_times_n_plus_one_over_two(self):
        rng = np.random.default_rng(0)
        for n in [3, 7, 20]:
            r = _rankdata_average(rng.normal(size=n))
            assert r.sum() == pytest.approx(n * (n + 1) / 2)


class TestWilcoxon:
    def test_matches_r_published_value(self):
        """R: wilcox.test(x, y, paired=TRUE) gives V = 40, p-value = 0.03906."""
        x = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
        y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
        w, p = wilcoxon_signed_rank(x, y)
        assert w == 40.0
        assert p == pytest.approx(0.03906, abs=1e-5)

    @pytest.mark.parametrize("trial", range(25))
    def test_exact_p_matches_full_enumeration(self, trial):
        rng = np.random.default_rng(trial)
        n = int(rng.integers(3, 11))
        # Distinct absolute values, so the exact test is valid.
        d = rng.choice(np.arange(1, 40), size=n, replace=False) * rng.choice([-1, 1], size=n)
        _, p = wilcoxon_signed_rank(d.astype(float))
        assert p == pytest.approx(enumerate_exact_p(d.astype(float)))

    def test_all_zero_differences_is_nan_not_significant(self):
        """Two identical systems are not 'significantly the same' — the test is undefined."""
        w, p = wilcoxon_signed_rank(np.ones(5), np.ones(5))
        assert math.isnan(w) and math.isnan(p)

    def test_single_nonzero_difference_gives_p_one(self):
        _, p = wilcoxon_signed_rank(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        assert p == 1.0

    def test_ties_fall_back_to_the_normal_approximation(self):
        """With repeated |d| the exact distribution is invalid; the result must still
        be a usable p-value rather than a wrong exact one."""
        d = np.array([1.0, 1.0, -1.0, 2.0, 3.0, -3.0, 4.0, 5.0, 6.0, 7.0])
        _, p = wilcoxon_signed_rank(d)
        assert 0.0 < p < 1.0
        assert p != pytest.approx(enumerate_exact_p(d))

    def test_all_positive_differences_gives_small_p(self):
        _, p = wilcoxon_signed_rank(np.arange(1.0, 13.0))
        assert p < 0.001

    def test_symmetric_data_gives_large_p(self):
        d = np.array([1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 0.5, -0.75])
        _, p = wilcoxon_signed_rank(d)
        assert p > 0.3

    def test_sign_flip_leaves_p_unchanged(self):
        rng = np.random.default_rng(3)
        d = rng.normal(size=15)
        _, p1 = wilcoxon_signed_rank(d)
        _, p2 = wilcoxon_signed_rank(-d)
        assert p1 == pytest.approx(p2)

    def test_zero_differences_are_dropped(self):
        d = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        _, p_with = wilcoxon_signed_rank(d)
        _, p_without = wilcoxon_signed_rank(d[2:])
        assert p_with == pytest.approx(p_without)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="match in shape"):
            wilcoxon_signed_rank(np.zeros(3), np.zeros(4))

    def test_nan_pairs_are_dropped(self):
        d = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
        _, p1 = wilcoxon_signed_rank(d)
        _, p2 = wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert p1 == pytest.approx(p2)

    def test_p_is_always_in_the_unit_interval(self):
        rng = np.random.default_rng(4)
        for _ in range(40):
            n = int(rng.integers(2, 40))
            _, p = wilcoxon_signed_rank(rng.normal(size=n) + rng.normal())
            assert math.isnan(p) or 0.0 <= p <= 1.0

    def test_large_n_uses_the_approximation_without_error(self):
        rng = np.random.default_rng(5)
        _, p = wilcoxon_signed_rank(rng.normal(size=200) + 0.4)
        assert 0.0 <= p <= 1.0


class TestBootstrap:
    def test_point_estimate_is_the_sample_mean(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert bootstrap_ci(v).estimate == pytest.approx(3.0)

    def test_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(rng.normal(size=200))
        assert ci.lower <= ci.estimate <= ci.upper

    def test_median_statistic_is_supported(self):
        v = np.array([1.0, 2.0, 100.0])
        assert bootstrap_ci(v, statistic="median").estimate == 2.0

    def test_nan_entries_are_dropped_and_counted(self):
        ci = bootstrap_ci(np.array([1.0, np.nan, 3.0]))
        assert ci.n == 2
        assert ci.estimate == pytest.approx(2.0)

    def test_empty_input_is_nan_with_zero_n(self):
        ci = bootstrap_ci(np.array([]))
        assert math.isnan(ci.estimate) and ci.n == 0

    def test_single_value_collapses_to_a_point(self):
        ci = bootstrap_ci(np.array([7.0]))
        assert (ci.estimate, ci.lower, ci.upper) == (7.0, 7.0, 7.0)

    def test_is_deterministic_given_the_seed(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=50)
        a, b = bootstrap_ci(v, seed=3), bootstrap_ci(v, seed=3)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_different_seeds_give_different_intervals(self):
        rng = np.random.default_rng(2)
        v = rng.normal(size=50)
        assert bootstrap_ci(v, seed=1).lower != bootstrap_ci(v, seed=2).lower

    def test_wider_level_gives_wider_interval(self):
        rng = np.random.default_rng(3)
        v = rng.normal(size=200)
        narrow = bootstrap_ci(v, level=0.80)
        wide = bootstrap_ci(v, level=0.99)
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

    def test_paired_difference_uses_the_pairing(self):
        """Perfectly correlated arrays with a constant offset have a zero-width CI."""
        rng = np.random.default_rng(4)
        a = rng.normal(size=100)
        ci = paired_bootstrap_difference(a + 0.5, a)
        assert ci.estimate == pytest.approx(0.5)
        assert ci.upper - ci.lower < 1e-9

    def test_paired_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="match in shape"):
            paired_bootstrap_difference(np.zeros(3), np.zeros(4))

    def test_excludes_zero_property(self):
        assert Interval(0.5, 0.2, 0.8).excludes_zero is True
        assert Interval(0.5, -0.2, 0.8).excludes_zero is False
        assert Interval(-0.5, -0.8, -0.2).excludes_zero is True

    def test_interval_str_is_readable(self):
        assert str(Interval(0.5, 0.25, 0.75)) == "0.5000 [0.2500, 0.7500]"


class TestNoiseScale:
    def test_is_sqrt_two_times_sd(self):
        """Two independent runs each with variance s^2 differ with variance 2 s^2."""
        v = np.array([1.0, 2.0, 3.0])
        assert noise_scale_from_seeds(v) == pytest.approx(math.sqrt(2.0) * 1.0)

    def test_identical_seeds_give_zero(self):
        assert noise_scale_from_seeds(np.array([2.0, 2.0, 2.0])) == 0.0

    def test_fewer_than_two_seeds_is_nan(self):
        assert math.isnan(noise_scale_from_seeds(np.array([1.0])))
        assert math.isnan(noise_scale_from_seeds(np.array([])))

    def test_nan_entries_are_dropped(self):
        assert noise_scale_from_seeds(np.array([1.0, np.nan, 3.0])) == pytest.approx(
            math.sqrt(2.0) * np.std([1.0, 3.0], ddof=1)
        )

    def test_using_sd_instead_of_sqrt2_sd_would_understate_by_29_percent(self):
        """sqrt(2) ~= 1.414, so a bare sd understates the difference scale by 1 - 1/sqrt(2)."""
        v = np.array([0.10, 0.12, 0.14])
        sd = float(np.std(v, ddof=1))
        assert noise_scale_from_seeds(v) / sd == pytest.approx(math.sqrt(2.0))


class TestVerdict:
    def test_below_noise_is_inside_noise_even_when_significant(self):
        """The noise scale outranks the p-value. This is the whole point of the gate."""
        assert verdict(0.005, 0.01, 1e-9) == "inside noise"

    def test_significant_and_two_times_noise_survives(self):
        assert verdict(0.05, 0.01, 0.001) == "survives"

    def test_significant_but_only_one_point_five_times_noise_is_suggestive(self):
        assert verdict(0.015, 0.01, 0.001) == "suggestive"

    def test_large_but_not_significant_is_not_significant(self):
        assert verdict(0.5, 0.1, 0.6) == "not significant"

    def test_no_seed_study_is_untested(self):
        assert verdict(0.05, None, 0.001).startswith("untested")

    def test_nan_noise_scale_is_untested(self):
        assert verdict(0.05, float("nan"), 0.001).startswith("untested")

    def test_zero_noise_scale_is_untested(self):
        assert verdict(0.05, 0.0, 0.001).startswith("untested")

    def test_nan_p_value_is_never_significant(self):
        assert verdict(0.05, 0.01, float("nan")) == "not significant"

    def test_adjusted_p_takes_precedence(self):
        assert verdict(0.05, 0.01, 0.001, p_adjusted=0.9) == "not significant"

    def test_sign_does_not_matter(self):
        assert verdict(-0.05, 0.01, 0.001) == verdict(0.05, 0.01, 0.001)

    def test_exactly_at_the_noise_scale_is_not_inside_noise(self):
        assert verdict(0.01, 0.01, 0.001) != "inside noise"


class TestCompare:
    def test_means_are_reported(self):
        c = compare(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        assert c.mean_a == pytest.approx(2.0)
        assert c.mean_b == pytest.approx(1.0)

    def test_difference_estimate_is_a_minus_b(self):
        c = compare(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        assert c.difference.estimate == pytest.approx(1.0)

    def test_effect_size_is_zero_when_the_difference_is_constant(self):
        """A constant difference has zero sd, so paired Cohen's d is undefined;
        reporting 0.0 rather than inf is the deliberate choice."""
        c = compare(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        assert c.effect_size == 0.0

    def test_effect_size_sign_follows_the_difference(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=40)
        assert compare(base + 1.0 + 0.1 * rng.normal(size=40), base).effect_size > 0
        assert compare(base - 1.0 + 0.1 * rng.normal(size=40), base).effect_size < 0

    def test_nan_pairs_are_dropped_and_n_reflects_it(self):
        a = np.array([1.0, np.nan, 3.0, 4.0])
        b = np.array([0.0, 1.0, np.nan, 2.0])
        assert compare(a, b).n == 2

    def test_identical_inputs_give_nan_p(self):
        v = np.arange(10.0)
        assert math.isnan(compare(v, v).p_value)

    def test_noise_ratio_is_delta_over_noise_scale(self):
        c = compare(np.array([1.0, 1.0]), np.array([0.0, 0.0]), noise_scale=0.25)
        assert c.noise_ratio == pytest.approx(4.0)

    def test_noise_ratio_is_nan_without_a_seed_study(self):
        c = compare(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        assert math.isnan(c.noise_ratio)

    def test_to_dict_carries_the_verdict(self):
        d = compare(np.array([1.0, 2.0]), np.array([0.0, 1.0]), noise_scale=0.1).to_dict()
        assert "verdict" in d and "noise_ratio" in d and "p_adjusted" in d

    def test_significant_property_uses_adjusted_p_when_present(self):
        c = Comparison("a", "b", "m", 1.0, 0.0, Interval(1.0, 0.5, 1.5), 0.01, 1.0, 10)
        assert c.significant is True
        c.p_adjusted = 0.9
        assert c.significant is False


class TestHolm:
    def test_smallest_p_is_scaled_by_the_family_size(self):
        cs = [
            Comparison("a", "b", f"m{i}", 0, 0, Interval(0, 0, 0), p, 0, 10)
            for i, p in enumerate([0.01, 0.02, 0.03, 0.04])
        ]
        holm_bonferroni(cs)
        assert cs[0].p_adjusted == pytest.approx(0.04)

    def test_adjusted_p_values_are_monotone(self):
        cs = [
            Comparison("a", "b", f"m{i}", 0, 0, Interval(0, 0, 0), p, 0, 10)
            for i, p in enumerate([0.001, 0.20, 0.30, 0.90])
        ]
        holm_bonferroni(cs)
        adj = [c.p_adjusted for c in cs]
        assert adj == sorted(adj)

    def test_adjusted_never_below_raw(self):
        rng = np.random.default_rng(0)
        cs = [
            Comparison("a", "b", f"m{i}", 0, 0, Interval(0, 0, 0), float(p), 0, 10)
            for i, p in enumerate(rng.uniform(size=8))
        ]
        holm_bonferroni(cs)
        assert all(c.p_adjusted >= c.p_value - 1e-12 for c in cs)

    def test_adjusted_is_capped_at_one(self):
        cs = [
            Comparison("a", "b", f"m{i}", 0, 0, Interval(0, 0, 0), 0.9, 0, 10)
            for i in range(5)
        ]
        holm_bonferroni(cs)
        assert all(c.p_adjusted == 1.0 for c in cs)

    def test_nan_p_values_do_not_consume_family_budget(self):
        cs = [
            Comparison("a", "b", "m0", 0, 0, Interval(0, 0, 0), 0.01, 0, 10),
            Comparison("a", "b", "m1", 0, 0, Interval(0, 0, 0), float("nan"), 0, 10),
        ]
        holm_bonferroni(cs)
        assert cs[0].p_adjusted == pytest.approx(0.01)  # family size 1, not 2
        assert math.isnan(cs[1].p_adjusted)

    def test_single_comparison_is_unchanged(self):
        cs = [Comparison("a", "b", "m", 0, 0, Interval(0, 0, 0), 0.03, 0, 10)]
        holm_bonferroni(cs)
        assert cs[0].p_adjusted == pytest.approx(0.03)

    def test_is_less_conservative_than_bonferroni(self):
        ps = [0.01, 0.04, 0.045, 0.048]
        cs = [
            Comparison("a", "b", f"m{i}", 0, 0, Interval(0, 0, 0), p, 0, 10)
            for i, p in enumerate(ps)
        ]
        holm_bonferroni(cs)
        # Bonferroni would multiply every p by 4; Holm multiplies the k-th by 4-k.
        assert cs[1].p_adjusted < 0.04 * 4

    def test_empty_list_is_fine(self):
        assert holm_bonferroni([]) == []


class TestSeedStudySummary:
    def test_reports_mean_sd_noise_and_count(self):
        out = summarize_seed_study({"der": [0.30, 0.32, 0.34]})
        assert out["der"]["mean"] == pytest.approx(0.32)
        assert out["der"]["n_seeds"] == 3.0
        assert out["der"]["noise_scale"] == pytest.approx(math.sqrt(2.0) * 0.02)

    def test_single_seed_gives_nan_sd(self):
        out = summarize_seed_study({"der": [0.30]})
        assert math.isnan(out["der"]["sd"])
        assert out["der"]["n_seeds"] == 1.0

    def test_nan_values_are_excluded_from_the_count(self):
        out = summarize_seed_study({"der": [0.3, float("nan"), 0.5]})
        assert out["der"]["n_seeds"] == 2.0

    def test_empty_metric_is_nan(self):
        out = summarize_seed_study({"der": []})
        assert math.isnan(out["der"]["mean"])
