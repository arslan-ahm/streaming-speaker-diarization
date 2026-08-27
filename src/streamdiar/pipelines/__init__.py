"""Experiment orchestration. Scripts are thin wrappers over these functions."""

from __future__ import annotations

from .ablation import ABLATIONS, ablation_tests, run_ablations, summarise_ablations
from .calibration import run_calibration, summarise_calibration
from .common import (
    LOWER_IS_BETTER,
    METRIC_FAMILY,
    append_rows,
    load_or_train,
    test_split,
    write_table,
)
from .comparison import (
    VARIANTS,
    headline_table,
    run_comparison,
    statistical_tests,
    summarise,
)
from .efficiency import run_efficiency, scaling_study
from .latency import (
    DEFAULT_BUDGETS_MS,
    run_latency_point,
    run_latency_sweep,
    run_offline_reference,
    summarise_curve,
)

__all__ = [
    "ABLATIONS",
    "DEFAULT_BUDGETS_MS",
    "LOWER_IS_BETTER",
    "METRIC_FAMILY",
    "VARIANTS",
    "ablation_tests",
    "append_rows",
    "headline_table",
    "load_or_train",
    "run_ablations",
    "run_calibration",
    "run_comparison",
    "run_efficiency",
    "run_latency_point",
    "run_latency_sweep",
    "run_offline_reference",
    "scaling_study",
    "statistical_tests",
    "summarise",
    "summarise_ablations",
    "summarise_calibration",
    "summarise_curve",
    "test_split",
    "write_table",
]
