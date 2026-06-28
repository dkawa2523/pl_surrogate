"""Evaluation payload builders for CLI outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.eval.metrics import (
    finite_pair_stats,
    poisson_residual_norm,
    r2_by_var,
    r2_masked,
    rmse_by_var,
    rmse_masked,
)
from plasma_surrogate.eval.sanity_checks import build_metric_validity_flags
from plasma_surrogate.train.losses import poisson_residual_loss


def build_eval_metrics_payload(
    *,
    true_eval: dict[str, np.ndarray],
    pred_eval: dict[str, np.ndarray],
    eps: np.ndarray | None = None,
    mask_plasma: np.ndarray | None = None,
) -> dict[str, Any]:
    keys = [k for k in true_eval.keys() if k in pred_eval]
    rmse = rmse_by_var({k: true_eval[k] for k in keys}, {k: pred_eval[k] for k in keys})
    r2 = r2_by_var({k: true_eval[k] for k in keys}, {k: pred_eval[k] for k in keys})
    rmse_plasma: dict[str, float] = {}
    r2_plasma: dict[str, float] = {}
    validity_flags: dict[str, Any] = {
        "plasma_mask_available": mask_plasma is not None,
        "target_metrics_valid": True,
        "invalid_target_vars": [],
        "invalid_reasons": {},
    }
    if mask_plasma is not None:
        finite_stats_plasma: dict[str, dict[str, float]] = {}
        for k in keys:
            finite_stats_plasma[k] = finite_pair_stats(true_eval[k], pred_eval[k], mask_plasma)
            rmse_plasma[k] = float(rmse_masked(true_eval[k], pred_eval[k], mask_plasma))
            r2_plasma[k] = float(r2_masked(true_eval[k], pred_eval[k], mask_plasma))
        validity_flags = build_metric_validity_flags(
            target_vars=keys,
            metrics_plasma=rmse_plasma,
            r2_plasma=r2_plasma,
            finite_stats_plasma=finite_stats_plasma,
            mask_plasma=mask_plasma,
            quality_components={"surrogate_quality_score": 0.0},
        )
    if "phi" in pred_eval:
        rmse["phi_poisson_residual"] = float(poisson_residual_loss(pred_eval["phi"][:, 0]))
        rmse["phi_poisson_residual_norm"] = float(poisson_residual_norm(pred_eval["phi"][:, 0], eps=eps))
    return {
        "rmse": rmse,
        "r2": r2,
        "rmse_plasma": rmse_plasma,
        "r2_plasma": r2_plasma,
        "validity_flags": validity_flags,
    }


def build_viz_tables_payload(
    *,
    diag_rows: list[dict[str, float | str]],
    region_rows: list[dict[str, float | str]],
) -> dict[str, Any]:
    preferred_diag = [
        "case_key",
        "physics_diagnostics_available",
        "poisson_residual_norm",
        "bc_potential_mae",
        "uniformity",
    ]
    diag_keys = {str(key) for row in diag_rows for key in row}
    diag_header = [key for key in preferred_diag if key in diag_keys]
    diag_header.extend(sorted(key for key in diag_keys if key not in set(diag_header)))
    region_header = [
        "case_key",
        "density_key",
        "plasma_mean_density",
        "bulk_mean_density",
        "boundary_mean_density",
    ]
    return {
        "diag_header": diag_header,
        "diag_rows": [[row.get(h, 0.0) for h in diag_header] for row in diag_rows],
        "region_header": region_header,
        "region_rows": [[row.get(h, 0.0) for h in region_header] for row in region_rows],
    }


__all__ = ["build_eval_metrics_payload", "build_viz_tables_payload"]
