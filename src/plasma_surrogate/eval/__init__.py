"""Evaluation utilities."""

from plasma_surrogate.eval.metrics_builder import (
    build_benchmark_eval_row,
    build_eval_metrics_payload,
    build_region_metrics,
    build_single_case_physics_metrics,
    build_viz_tables_payload,
)
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
