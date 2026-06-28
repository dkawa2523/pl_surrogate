"""Positive-target evaluation diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def positive_targets_from_schema(target_role_schema: dict[str, Any] | None) -> set[str]:
    schema = dict(target_role_schema or {})
    positive = schema.get("positive_targets", [])
    if isinstance(positive, list):
        return {str(v) for v in positive if str(v).strip()}
    targets = schema.get("targets", [])
    if not isinstance(targets, list):
        return set()
    out: set[str] = set()
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        target_id = str(entry.get("id", "")).strip()
        if target_id and entry.get("positive") is True:
            out.add(target_id)
    return out


def positive_diagnostic_columns(
    *,
    pred_eval: dict[str, np.ndarray],
    target_role_schema: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    violation_rate: dict[str, float] = {}
    negative_min: dict[str, float] = {}
    columns: dict[str, float] = {}
    for name in sorted(positive_targets_from_schema(target_role_schema)):
        if name not in pred_eval:
            continue
        arr = np.asarray(pred_eval[name], dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        vals = arr[finite]
        rate = float(np.mean((vals < 0.0).astype(np.float64)))
        min_value = float(np.min(vals))
        violation_rate[name] = rate
        negative_min[name] = min_value
        columns[f"positive_violation_rate_{name}"] = rate
        columns[f"negative_min_{name}"] = min_value
    return columns, violation_rate, negative_min


__all__ = ["positive_diagnostic_columns", "positive_targets_from_schema"]
