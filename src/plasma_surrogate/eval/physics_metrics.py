"""Scalar physics diagnostic metric builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _diag_float(diag: dict[str, Any], key: str) -> float:
    if key not in diag:
        return float("nan")
    try:
        value = float(diag[key])
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def build_single_case_physics_metrics(case_dir: Path) -> dict[str, float | str]:
    diag = _load_json(case_dir / "diagnostics.json")
    qoi = _load_json(case_dir / "qoi.json")
    physics_available = bool(diag.get("physics_diagnostics_available", "poisson_residual_norm" in diag))
    row: dict[str, float | str] = {
        "case_key": case_dir.name,
        "physics_diagnostics_available": physics_available,
        "poisson_residual_norm": _diag_float(diag, "poisson_residual_norm"),
        "bc_potential_mae": _diag_float(diag, "bc_potential_mae"),
        "uniformity": float(qoi.get("uniformity", float("nan"))),
    }
    if "poisson_residual_map_l2" in diag:
        row["poisson_residual_map_l2"] = _diag_float(diag, "poisson_residual_map_l2")
    if "boundary_operator_residual_map_l2" in diag:
        row["boundary_residual_map_l2"] = _diag_float(diag, "boundary_operator_residual_map_l2")
    return row


__all__ = ["build_single_case_physics_metrics"]
