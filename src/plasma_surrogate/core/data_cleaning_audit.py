"""Synthetic data cleaning audit helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _resolve_required_outputs(
    cases: list[dict[str, Any]],
    required_outputs: list[str] | None,
) -> list[str]:
    if required_outputs:
        return [str(v) for v in required_outputs]
    if not cases:
        return []
    y0 = cases[0].get("y", {})
    if isinstance(y0, dict) and len(y0) > 0:
        return [str(k) for k in y0.keys()]
    return []


def _count_missing_cases(
    cases: list[dict[str, Any]],
    cond_order: list[str],
    *,
    required_outputs: list[str],
) -> dict[str, int]:
    missing = {
        "case_id": 0,
        "cond": 0,
        "axis": 0,
        "y": 0,
        "y_missing_any": 0,
    }
    for key in required_outputs:
        missing[f"y_{key}"] = 0
    for case in cases:
        if "case_id" not in case:
            missing["case_id"] += 1
        cond = case.get("cond")
        if not isinstance(cond, dict):
            missing["cond"] += 1
            continue
        for key in cond_order:
            if key not in cond:
                missing["cond"] += 1
                break
        if "axis" not in case:
            missing["axis"] += 1
        y = case.get("y")
        if not isinstance(y, dict):
            missing["y"] += 1
            continue
        missing_any = False
        for key in required_outputs:
            if key not in y:
                missing[f"y_{key}"] += 1
                missing_any = True
        if missing_any:
            missing["y_missing_any"] += 1
    return missing


def _range_violations(cases: list[dict[str, Any]], cond_order: list[str], axis_mode: str) -> dict[str, int]:
    violations = {"cond_non_finite": 0, "cond_out_of_unit_interval": 0, "axis_non_finite": 0, "axis_out_of_range": 0}
    for case in cases:
        cond = case.get("cond", {})
        vals: list[float] = []
        if isinstance(cond, dict):
            for key in cond_order:
                if key in cond:
                    vals.append(float(cond[key]))
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                violations["cond_non_finite"] += 1
            if np.any((arr < 0.0) | (arr > 1.0)):
                violations["cond_out_of_unit_interval"] += 1
        axis_val = float(case.get("axis", 0.0))
        if not np.isfinite(axis_val):
            violations["axis_non_finite"] += 1
            continue
        if axis_mode == "steady":
            if abs(axis_val) > 1e-7:
                violations["axis_out_of_range"] += 1
        elif axis_mode in {"time", "phase_sincos"}:
            if (axis_val < 0.0) or (axis_val >= 1.0):
                violations["axis_out_of_range"] += 1
    return violations


def _axis_hist(cases: list[dict[str, Any]], axis_mode: str) -> dict[str, Any]:
    axis_vals = np.asarray([float(c.get("axis", 0.0)) for c in cases], dtype=np.float64)
    if axis_vals.size == 0:
        return {"mode": axis_mode, "bins": [], "counts": []}
    if axis_mode == "steady":
        return {"mode": axis_mode, "bins": [0.0, 1.0], "counts": [int(axis_vals.size)]}
    if axis_mode in {"time", "phase_sincos"}:
        hist, edges = np.histogram(axis_vals, bins=10, range=(0.0, 1.0))
        return {"mode": axis_mode, "bins": [float(v) for v in edges.tolist()], "counts": [int(v) for v in hist.tolist()]}
    hist, edges = np.histogram(axis_vals, bins=10)
    return {"mode": axis_mode, "bins": [float(v) for v in edges.tolist()], "counts": [int(v) for v in hist.tolist()]}


def _duplicate_case_keys(cases: list[dict[str, Any]], cond_order: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for case in cases:
        cond = case.get("cond", {})
        cond_payload = {}
        if isinstance(cond, dict):
            for key in cond_order:
                if key in cond:
                    try:
                        cond_payload[key] = float(cond[key])
                    except (TypeError, ValueError):
                        cond_payload[key] = str(cond[key])
        key_payload = {
            "cond": cond_payload,
            "axis": float(case.get("axis", 0.0)),
        }
        key = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        counts[key] = counts.get(key, 0) + 1
    return sorted([key for key, count in counts.items() if count > 1])


def _cond_range_summary(cases: list[dict[str, Any]], cond_order: list[str]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for key in cond_order:
        vals: list[float] = []
        for case in cases:
            cond = case.get("cond", {})
            if isinstance(cond, dict) and key in cond:
                try:
                    val = float(cond[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    vals.append(val)
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[key] = {"min": float(np.min(arr)), "max": float(np.max(arr))}
        else:
            summary[key] = {"min": None, "max": None}
    return summary


def run_data_audit(
    cases: list[dict[str, Any]],
    cond_order: list[str],
    axis_mode: str,
    required_outputs: list[str] | None = None,
) -> dict[str, Any]:
    req_outputs = _resolve_required_outputs(cases, required_outputs)
    return {
        "n_cases": int(len(cases)),
        "missing_counts": _count_missing_cases(cases, cond_order, required_outputs=req_outputs),
        "range_violations": _range_violations(cases, cond_order, axis_mode),
        "axis_hist": _axis_hist(cases, axis_mode),
        "duplicate_case_keys": _duplicate_case_keys(cases, cond_order),
        "cond_range_summary": _cond_range_summary(cases, cond_order),
        "required_outputs": req_outputs,
    }


def run_synthetic_data_audit(
    cases: list[dict[str, Any]],
    cond_order: list[str],
    axis_mode: str,
    required_outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper kept for existing callers/tests."""

    return run_data_audit(
        cases=cases,
        cond_order=cond_order,
        axis_mode=axis_mode,
        required_outputs=required_outputs,
    )
