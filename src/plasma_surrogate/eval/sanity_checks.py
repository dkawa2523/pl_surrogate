"""Simulation sanity checks for surrogate evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvalResult:
    """Structured evaluation output with diagnostics separated from core metrics."""

    core_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    spatial_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _invalid_target_reasons(
    *,
    name: str,
    metrics_plasma: dict[str, float],
    r2_plasma: dict[str, float],
    finite_stats_plasma: dict[str, dict[str, float]],
    mask_plasma: np.ndarray | None,
) -> list[str]:
    if mask_plasma is None:
        return []
    reasons: list[str] = []
    stats = finite_stats_plasma.get(name)
    if not stats:
        reasons.append("missing_plasma_stats")
    else:
        if float(stats.get("n_active", 0.0)) <= 0.0:
            reasons.append("empty_plasma_mask")
        if float(stats.get("n_nonfinite", 0.0)) > 0.0:
            reasons.append("nonfinite_values")
    for metric_name, src in (("rmse", metrics_plasma), ("r2", r2_plasma)):
        if name not in src:
            reasons.append(f"missing_{metric_name}")
        elif not np.isfinite(float(src[name])):
            reasons.append(f"nonfinite_{metric_name}")
    return reasons


def build_metric_validity_flags(
    *,
    target_vars: list[str],
    metrics_plasma: dict[str, float],
    r2_plasma: dict[str, float],
    finite_stats_plasma: dict[str, dict[str, float]],
    mask_plasma: np.ndarray | None,
    quality_components: dict[str, float],
) -> dict[str, Any]:
    invalid_vars: list[str] = []
    reasons: dict[str, list[str]] = {}

    for name in [str(v) for v in target_vars]:
        target_reasons = _invalid_target_reasons(
            name=name,
            metrics_plasma=metrics_plasma,
            r2_plasma=r2_plasma,
            finite_stats_plasma=finite_stats_plasma,
            mask_plasma=mask_plasma,
        )
        if target_reasons:
            invalid_vars.append(name)
            reasons[name] = target_reasons

    quality_score = float(quality_components.get("surrogate_quality_score", float("nan")))
    finite_ratios = [
        float(stats.get("finite_ratio", 0.0))
        for stats in finite_stats_plasma.values()
        if stats is not None
    ]
    active_counts = [
        float(stats.get("n_active", 0.0))
        for stats in finite_stats_plasma.values()
        if stats is not None
    ]
    return {
        "target_metrics_valid": len(invalid_vars) == 0,
        "invalid_target_vars": invalid_vars,
        "invalid_reasons": reasons,
        "quality_score_valid": bool(np.isfinite(quality_score)),
        "plasma_mask_available": mask_plasma is not None,
        "min_active_count_plasma": min(active_counts) if active_counts else None,
        "min_finite_ratio_plasma": min(finite_ratios) if finite_ratios else None,
    }


__all__ = [
    "EvalResult",
    "build_metric_validity_flags",
]
