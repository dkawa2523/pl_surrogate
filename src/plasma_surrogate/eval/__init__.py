"""Evaluation utilities."""

from plasma_surrogate.eval.core_metrics import build_benchmark_eval_row, build_region_metrics
from plasma_surrogate.eval.metrics import (
    mae,
    r2,
    r2_by_var,
    r2_masked,
    rmse,
    rmse_by_var,
    rmse_masked,
    uniformity,
)
from plasma_surrogate.eval.payloads import build_eval_metrics_payload, build_viz_tables_payload
from plasma_surrogate.eval.physics_metrics import build_single_case_physics_metrics

__all__ = [
    "rmse",
    "mae",
    "r2",
    "r2_by_var",
    "r2_masked",
    "rmse_by_var",
    "rmse_masked",
    "uniformity",
    "build_single_case_physics_metrics",
    "build_region_metrics",
    "build_benchmark_eval_row",
    "build_eval_metrics_payload",
    "build_viz_tables_payload",
]
