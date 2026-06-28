"""Group-wise evaluation metric columns."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups


def _schema_has_targets(target_role_schema: dict[str, Any] | None) -> bool:
    targets = dict(target_role_schema or {}).get("targets", [])
    return isinstance(targets, list) and len(targets) > 0


def _resolve_target_groups_for_metrics(
    *,
    output_vars: list[str] | None,
    target_role_schema: dict[str, Any] | None,
) -> dict[str, TargetGroup]:
    if output_vars is None or not _schema_has_targets(target_role_schema):
        return {}
    return resolve_target_groups(
        output_vars=[str(v) for v in output_vars],
        target_role_schema=dict(target_role_schema or {}),
    )


def _finite_group_mean(src: dict[str, float], targets: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for target in targets:
        if target not in src:
            continue
        value = float(src[target])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def build_target_group_metric_columns(
    *,
    output_vars: list[str] | None,
    target_role_schema: dict[str, Any] | None,
    metrics: dict[str, float],
    r2_scores: dict[str, float] | None = None,
    metrics_plasma: dict[str, float] | None = None,
    r2_plasma: dict[str, float] | None = None,
    metrics_boundary_band: dict[str, float] | None = None,
    metrics_deep_plasma: dict[str, float] | None = None,
    metrics_outside: dict[str, float] | None = None,
    positive_violation_rate: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build group-wise metric columns from existing target-wise metric maps."""

    groups = _resolve_target_groups_for_metrics(
        output_vars=output_vars,
        target_role_schema=target_role_schema,
    )
    if not groups:
        return {}
    metric_sources = {
        "test_rmse_group_{group}": dict(metrics or {}),
        "test_r2_group_{group}": dict(r2_scores or {}),
        "test_rmse_group_{group}_plasma": dict(metrics_plasma or {}),
        "test_r2_group_{group}_plasma": dict(r2_plasma or {}),
        "test_rmse_group_{group}_boundary_band": dict(metrics_boundary_band or {}),
        "test_rmse_group_{group}_deep_plasma": dict(metrics_deep_plasma or {}),
        "test_rmse_group_{group}_outside": dict(metrics_outside or {}),
        "positive_violation_rate_group_{group}": dict(positive_violation_rate or {}),
    }
    out: dict[str, float] = {}
    for group in groups.values():
        if not group.targets:
            continue
        for key_template, src in metric_sources.items():
            value = _finite_group_mean(src, group.targets)
            if value is None:
                continue
            out[key_template.format(group=group.name)] = value
    return out


__all__ = ["build_target_group_metric_columns"]
