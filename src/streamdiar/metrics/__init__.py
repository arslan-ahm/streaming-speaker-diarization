"""Metrics: DER with an optimal speaker mapping, latency, calibration, statistics."""

from __future__ import annotations

from .assignment import (
    assignment_cost,
    greedy_assignment,
    linear_sum_assignment,
)
from .calibration import (
    CalibrationResult,
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
from .der import DERResult, der, jer, labels_to_matrix
from .latency import LatencyStats, emission_latency, speaker_count_accuracy
from .stats import (
    Comparison,
    Interval,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    noise_scale_from_seeds,
    paired_bootstrap_difference,
    summarize_seed_study,
    verdict,
    wilcoxon_signed_rank,
)

__all__ = [
    "CalibrationResult",
    "Comparison",
    "DERResult",
    "Interval",
    "LatencyStats",
    "adaptive_calibration_error",
    "apply_temperature",
    "area_under_risk_coverage",
    "assignment_cost",
    "bootstrap_ci",
    "brier_score",
    "compare",
    "der",
    "emission_latency",
    "error_detection_auroc",
    "evaluate_calibration",
    "expected_calibration_error",
    "fit_temperature",
    "greedy_assignment",
    "holm_bonferroni",
    "jer",
    "labels_to_matrix",
    "linear_sum_assignment",
    "negative_log_likelihood",
    "noise_scale_from_seeds",
    "paired_bootstrap_difference",
    "risk_coverage_curve",
    "speaker_count_accuracy",
    "summarize_seed_study",
    "verdict",
    "wilcoxon_signed_rank",
]
