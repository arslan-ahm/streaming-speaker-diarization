"""Calibration and selective prediction: closed forms, limits, and degenerate inputs.

Each metric is checked against a case where the right answer is known analytically
— a perfectly calibrated stream, a perfectly overconfident one, a perfect error
detector — rather than against a stored number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from streamdiar.metrics.calibration import (
    adaptive_calibration_error,
    apply_temperature,
    area_under_risk_coverage,
    brier_score,
    error_detection_auroc,
    evaluate_calibration,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    risk_coverage_curve,
)


@pytest.fixture(scope="module")
def calibrated_stream() -> tuple[np.ndarray, np.ndarray]:
    """20000 decisions whose confidence *is* the probability of being correct."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 20000)
    return p, rng.uniform(size=20000) < p


class TestEce:
    def test_well_calibrated_stream_has_small_ece(self, calibrated_stream):
        conf, correct = calibrated_stream
        ece, _, _ = expected_calibration_error(conf, correct)
        assert ece < 0.02

    def test_perfect_overconfidence_gives_ece_one(self):
        """Confidence 1.0, accuracy 0.0: the gap is 1.0 in the only occupied bin."""
        ece, mce, _ = expected_calibration_error(np.ones(100), np.zeros(100))
        assert ece == pytest.approx(1.0)
        assert mce == pytest.approx(1.0)

    def test_perfect_underconfidence_gives_ece_one(self):
        ece, _, _ = expected_calibration_error(np.zeros(100), np.ones(100))
        assert ece == pytest.approx(1.0)

    def test_mce_is_at_least_ece(self, calibrated_stream):
        conf, correct = calibrated_stream
        ece, mce, _ = expected_calibration_error(conf, correct)
        assert mce >= ece - 1e-12

    def test_empty_bins_are_omitted_not_reported_as_perfect(self):
        """Two tight clusters of confidence occupy 2 of 15 bins; only those appear."""
        conf = np.concatenate([np.full(50, 0.05), np.full(50, 0.95)])
        correct = np.concatenate([np.zeros(50), np.ones(50)])
        _, _, bins = expected_calibration_error(conf, correct, n_bins=15)
        assert len(bins) == 2

    def test_bin_counts_sum_to_n(self, calibrated_stream):
        conf, correct = calibrated_stream
        _, _, bins = expected_calibration_error(conf, correct)
        assert sum(b[3] for b in bins) == conf.size

    def test_empty_input_is_nan(self):
        ece, mce, bins = expected_calibration_error(np.array([]), np.array([]))
        assert math.isnan(ece) and math.isnan(mce) and bins == []

    def test_confidence_is_clipped_into_the_unit_interval(self):
        ece, _, _ = expected_calibration_error(np.array([1.5, -0.5]), np.array([1, 0]))
        assert ece == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            expected_calibration_error(np.zeros(3), np.zeros(4))


class TestAce:
    def test_agrees_with_ece_on_a_uniform_stream(self, calibrated_stream):
        """With confidence uniform on [0,1], equal-width and equal-mass bins coincide."""
        conf, correct = calibrated_stream
        ece, _, _ = expected_calibration_error(conf, correct)
        assert adaptive_calibration_error(conf, correct) == pytest.approx(ece, abs=0.01)

    def test_exceeds_ece_for_a_sharp_model(self):
        """A model whose confidence piles into the top width-bin: equal-width ECE
        averages one bin's gap and understates; equal-mass bins resolve it."""
        rng = np.random.default_rng(1)
        conf = 0.90 + 0.10 * rng.uniform(size=4000)
        correct = rng.uniform(size=4000) < 0.60  # badly overconfident
        ece, _, _ = expected_calibration_error(conf, correct, n_bins=15)
        ace = adaptive_calibration_error(conf, correct, n_bins=15)
        assert ace >= ece - 1e-9

    def test_empty_input_is_nan(self):
        assert math.isnan(adaptive_calibration_error(np.array([]), np.array([])))

    def test_more_bins_than_samples_is_handled(self):
        assert adaptive_calibration_error(np.array([0.5, 0.5]), np.array([1, 0]), 15) >= 0.0


class TestBrierAndNll:
    def test_brier_of_a_perfect_predictor_is_zero(self):
        assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0

    def test_brier_of_the_worst_predictor_is_one(self):
        assert brier_score(np.array([1.0, 0.0]), np.array([0, 1])) == 1.0

    def test_brier_of_always_half_is_a_quarter(self):
        assert brier_score(np.full(100, 0.5), np.random.default_rng(0).integers(0, 2, 100)) == 0.25

    def test_nll_of_a_perfect_predictor_is_near_zero(self):
        assert negative_log_likelihood(np.array([1.0, 0.0]), np.array([1, 0])) < 1e-9

    def test_nll_is_clipped_rather_than_infinite(self):
        """One confident mistake must not send the mean to inf and make the metric
        a detector of a single outlier."""
        nll = negative_log_likelihood(np.array([1.0, 1.0]), np.array([1, 0]))
        assert np.isfinite(nll)

    def test_nll_of_always_half_is_log_two(self):
        assert negative_log_likelihood(
            np.full(100, 0.5), np.random.default_rng(0).integers(0, 2, 100)
        ) == pytest.approx(math.log(2.0))

    def test_empty_inputs_are_nan(self):
        assert math.isnan(brier_score(np.array([]), np.array([])))
        assert math.isnan(negative_log_likelihood(np.array([]), np.array([])))


class TestRiskCoverage:
    def test_perfect_ranking_gives_zero_risk_at_low_coverage(self):
        conf = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        cov, risk = risk_coverage_curve(conf, correct)
        assert risk[0] == 0.0
        assert risk[49] == 0.0

    def test_full_coverage_risk_is_the_overall_error_rate(self):
        conf = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        _, risk = risk_coverage_curve(conf, correct)
        assert risk[-1] == pytest.approx(0.5)

    def test_coverage_ends_at_one(self):
        cov, _ = risk_coverage_curve(np.array([0.9, 0.1]), np.array([1, 0]))
        assert cov[-1] == pytest.approx(1.0)

    def test_curve_length_equals_n(self):
        cov, risk = risk_coverage_curve(np.arange(10) / 10.0, np.ones(10))
        assert cov.size == risk.size == 10

    def test_aurc_of_a_perfect_predictor_is_zero(self):
        assert area_under_risk_coverage(np.ones(10), np.ones(10)) == 0.0

    def test_aurc_equals_oracle_for_a_perfect_ranking(self):
        conf = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        assert area_under_risk_coverage(conf, correct) == pytest.approx(
            area_under_risk_coverage(correct, correct)
        )

    def test_aurc_is_worse_than_oracle_for_a_random_ranking(self):
        rng = np.random.default_rng(2)
        correct = rng.uniform(size=500) < 0.7
        conf = rng.uniform(size=500)
        assert area_under_risk_coverage(conf, correct) > area_under_risk_coverage(
            correct.astype(float), correct
        )

    def test_empty_input_is_nan(self):
        assert math.isnan(area_under_risk_coverage(np.array([]), np.array([])))


class TestErrorDetectionAuroc:
    def test_perfect_separation_is_one(self):
        conf = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        assert error_detection_auroc(conf, correct) == 1.0

    def test_inverted_separation_is_zero(self):
        conf = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        assert error_detection_auroc(conf, correct) == 0.0

    def test_all_ties_is_one_half(self):
        conf = np.full(100, 0.5)
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        assert error_detection_auroc(conf, correct) == pytest.approx(0.5)

    def test_single_class_is_nan_not_one_half(self):
        """An AUROC over one class is undefined; 0.5 would be a fabricated number."""
        assert math.isnan(error_detection_auroc(np.random.rand(10), np.ones(10)))
        assert math.isnan(error_detection_auroc(np.random.rand(10), np.zeros(10)))

    def test_agrees_with_the_rank_definition(self):
        rng = np.random.default_rng(3)
        conf = rng.uniform(size=200)
        correct = rng.uniform(size=200) < 0.6
        pos, neg = conf[correct], conf[~correct]
        brute = float(
            (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        )
        assert error_detection_auroc(conf, correct) == pytest.approx(brute)

    def test_monotone_transform_of_confidence_does_not_change_auroc(self):
        rng = np.random.default_rng(4)
        conf = rng.uniform(0.01, 0.99, size=300)
        correct = rng.uniform(size=300) < conf
        a = error_detection_auroc(conf, correct)
        b = error_detection_auroc(conf**3, correct)
        assert a == pytest.approx(b)


class TestTemperature:
    def test_recovers_a_known_temperature(self):
        """Generate correctness from sigmoid(z / 2.0); the fit must find T near 2."""
        rng = np.random.default_rng(0)
        z = rng.normal(size=8000) * 3.0
        y = rng.uniform(size=8000) < 1.0 / (1.0 + np.exp(-z / 2.0))
        assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.15)

    def test_recovers_temperature_one(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=8000) * 3.0
        y = rng.uniform(size=8000) < 1.0 / (1.0 + np.exp(-z))
        assert fit_temperature(z, y) == pytest.approx(1.0, rel=0.20)

    def test_empty_input_returns_one(self):
        assert fit_temperature(np.array([]), np.array([])) == 1.0

    def test_is_deterministic(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=500)
        y = rng.uniform(size=500) < 0.5
        assert fit_temperature(z, y) == fit_temperature(z, y)

    def test_nan_pairs_are_dropped(self):
        z = np.array([1.0, np.nan, -1.0, 2.0])
        y = np.array([1, 1, 0, 1])
        assert np.isfinite(fit_temperature(z, y))

    def test_apply_temperature_is_a_sigmoid(self):
        assert apply_temperature(np.array([0.0]), 1.0)[0] == pytest.approx(0.5)

    def test_apply_temperature_is_monotone_in_the_logit(self):
        out = apply_temperature(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 1.0)
        assert np.all(np.diff(out) > 0)

    def test_lower_temperature_sharpens(self):
        z = np.array([1.0])
        assert apply_temperature(z, 0.5)[0] > apply_temperature(z, 2.0)[0]

    def test_output_is_strictly_inside_the_unit_interval(self):
        out = apply_temperature(np.array([-1e6, 1e6]), 1.0)
        assert out[0] > 0.0 and out[1] < 1.0

    def test_zero_temperature_does_not_divide_by_zero(self):
        assert np.all(np.isfinite(apply_temperature(np.array([1.0, -1.0]), 0.0)))

    def test_fitting_reduces_nll_on_a_miscalibrated_stream(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=5000) * 4.0
        y = rng.uniform(size=5000) < 1.0 / (1.0 + np.exp(-z / 3.0))
        t = fit_temperature(z, y)
        before = negative_log_likelihood(apply_temperature(z, 1.0), y)
        after = negative_log_likelihood(apply_temperature(z, t), y)
        assert after <= before


class TestEvaluateCalibration:
    def test_reports_every_field(self, calibrated_stream):
        conf, correct = calibrated_stream
        r = evaluate_calibration(conf, correct)
        for field in ["ece", "ace", "mce", "brier", "nll", "accuracy", "mean_confidence",
                      "overconfidence", "aurc", "aurc_oracle", "error_detection_auroc", "n"]:
            assert hasattr(r, field)

    def test_overconfidence_is_confidence_minus_accuracy(self):
        r = evaluate_calibration(np.full(100, 0.9), np.concatenate([np.ones(50), np.zeros(50)]))
        assert r.overconfidence == pytest.approx(0.9 - 0.5)

    def test_negative_overconfidence_means_underconfident(self):
        r = evaluate_calibration(np.full(100, 0.5), np.ones(100))
        assert r.overconfidence == pytest.approx(-0.5)

    def test_accuracy_matches_the_correctness_mean(self, calibrated_stream):
        conf, correct = calibrated_stream
        assert evaluate_calibration(conf, correct).accuracy == pytest.approx(correct.mean())

    def test_n_excluded_is_carried_through(self):
        r = evaluate_calibration(np.array([0.5]), np.array([1]), n_excluded=17)
        assert r.n_excluded == 17

    def test_to_dict_drops_the_bins(self, calibrated_stream):
        conf, correct = calibrated_stream
        assert "bins" not in evaluate_calibration(conf, correct).to_dict()

    def test_oracle_aurc_is_never_worse_than_the_real_one(self):
        rng = np.random.default_rng(5)
        correct = rng.uniform(size=400) < 0.7
        r = evaluate_calibration(rng.uniform(size=400), correct)
        assert r.aurc_oracle <= r.aurc + 1e-12

    def test_empty_input_gives_nan_metrics_and_zero_n(self):
        r = evaluate_calibration(np.array([]), np.array([]))
        assert r.n == 0
        assert math.isnan(r.ece) and math.isnan(r.brier)

    def test_all_correct_gives_nan_auroc(self):
        r = evaluate_calibration(np.full(10, 0.9), np.ones(10))
        assert math.isnan(r.error_detection_auroc)
        assert r.accuracy == 1.0
