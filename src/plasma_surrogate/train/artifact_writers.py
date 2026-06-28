"""Training artifact writers shared by NumPy and Torch trainers."""

from __future__ import annotations

import math
from typing import Any

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.train.diagnostics_writer import save_optimization_diagnostics


NUMPY_DIAGNOSTIC_PRIORITY = [
    "epoch",
    "grad_l2_total",
    "active_weight_ratio",
    "step_rel_hidden_mean",
    "step_rel_output",
    "step_rel_output_to_hidden",
    "grad_scale_applied",
    "grad_norm_pre_scale",
    "grad_norm_post_scale",
    "grad_norm_post_clip",
    "clip_ratio",
    "effective_clip_flag",
    "stagnation_flag",
    "distance_transform_mode",
    "selection_valid_flag",
    "selected_epoch_score",
    "selected_epoch_flag",
]

TORCH_DIAGNOSTIC_PRIORITY = [
    "epoch",
    "grad_l2_total",
    "active_weight_ratio",
    "step_rel_hidden_mean",
    "step_rel_output",
    "step_rel_output_to_hidden",
    "grad_scale_applied",
    "grad_norm_pre_scale",
    "grad_norm_post_scale",
    "grad_norm_post_clip",
    "clip_ratio",
    "effective_clip_flag",
    "stagnation_flag",
    "head_refresh_applied",
    "head_design_cond",
]


def save_physics_terms(
    *,
    store: ArtifactStore,
    history: list[dict[str, float]],
    include_stage: bool = False,
) -> None:
    if not history:
        return
    if include_stage:
        header = [
            "epoch",
            "stage",
            "train_phys_loss",
            "train_poisson_loss",
            "train_boundary_operator_loss",
            "train_boundary_loss",
            "train_rho_loss",
        ]
    else:
        header = [
            "epoch",
            "train_phys_loss",
            "train_poisson_loss",
            "train_boundary_loss",
            "train_boundary_operator_loss",
            "train_rho_loss",
        ]
    rows = [[row.get(key, 0.0) for key in header] for row in history]
    store.save_csv("scalars/physics_terms.csv", header, rows)


def save_resolved_physics(
    *,
    store: ArtifactStore,
    physics_cfg: dict[str, Any] | None,
    boundary_operator_default_mode: str,
) -> None:
    cfg = dict(physics_cfg or {})
    boundary = dict(cfg.get("boundary_operator", {}) or {})
    store.save_json(
        "resolved_physics.json",
        {
            "enabled": bool(cfg.get("enabled", False)),
            "resolved_terms": list(cfg.get("resolved_terms", [])),
            "boundary_operator": {
                "enabled": bool(boundary.get("enabled", False)),
                "weight": float(boundary.get("weight", 0.0)),
                "mode": str(boundary.get("mode", boundary_operator_default_mode)),
                "primary_qoi_key": str(boundary.get("primary_qoi_key", "Gamma_i")),
            },
        },
    )


def save_training_progress(
    *,
    store: ArtifactStore,
    history: list[dict[str, float]],
    diagnostics_rows: list[dict[str, float]],
    total_epochs: int,
    elapsed_seconds: float,
    best_epoch: int | None = None,
    best_score: float | None = None,
) -> None:
    if not history:
        return
    header = list(history[0].keys())
    rows = [[row.get(key, "") for key in header] for row in history]
    store.save_csv("scalars/metrics_partial.csv", header, rows)

    latest = dict(history[-1])
    latest["epoch_completed"] = int(latest.get("epoch", len(history) - 1))
    latest["epochs_completed"] = int(len(history))
    latest["total_epochs"] = int(total_epochs)
    latest["elapsed_seconds"] = float(elapsed_seconds)
    if best_epoch is not None:
        latest["best_epoch_so_far"] = int(best_epoch)
    if best_score is not None and math.isfinite(float(best_score)):
        latest["best_score_so_far"] = float(best_score)
    store.save_json("scalars/progress_latest.json", latest)

    if diagnostics_rows:
        diag_header = list(diagnostics_rows[0].keys())
        diag_rows = [[row.get(key, "") for key in diag_header] for row in diagnostics_rows]
        store.save_csv("scalars/optimization_diagnostics_partial.csv", diag_header, diag_rows)


def save_numpy_optimization_diagnostics(*, store: ArtifactStore, rows: list[dict[str, Any]]) -> None:
    save_optimization_diagnostics(store=store, rows=rows, priority=NUMPY_DIAGNOSTIC_PRIORITY)


def save_torch_optimization_diagnostics(*, store: ArtifactStore, rows: list[dict[str, Any]]) -> None:
    save_optimization_diagnostics(store=store, rows=rows, priority=TORCH_DIAGNOSTIC_PRIORITY)


__all__ = [
    "save_numpy_optimization_diagnostics",
    "save_physics_terms",
    "save_resolved_physics",
    "save_torch_optimization_diagnostics",
    "save_training_progress",
]
