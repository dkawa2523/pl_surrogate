"""Shared DeepONet contract helpers for train/benchmark workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_deeponet_task_artifacts(
    pre_dir: Path,
    task: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    root = pre_dir / "sampling" / "deeponet" / str(task)
    idx_path = root / "sensor_query_index.json"
    meta_path = root / "index_meta.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"DeepONet task artifacts missing for task='{task}'")
    with idx_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    sensor_indices = np.asarray(idx.get("sensor_indices", []), dtype=np.int64)
    query_indices = np.asarray(idx.get("query_indices", []), dtype=np.int64)
    return sensor_indices, query_indices, idx, meta


def validate_deeponet_task_meta(
    task: str,
    meta: dict[str, Any],
    expected_shape: tuple[int, int],
    expected_order: str | None = None,
) -> None:
    got_shape = tuple(meta.get("grid_shape", list(expected_shape)))
    if got_shape != tuple(expected_shape):
        raise ValueError(f"deeponet task '{task}' grid_shape mismatch: expected={expected_shape}, got={got_shape}")
    if expected_order is not None:
        got_order = str(meta.get("flatten_order", "C"))
        if got_order != str(expected_order):
            raise ValueError(
                f"deeponet task '{task}' flatten_order mismatch: "
                f"expected={expected_order}, got={got_order}"
            )


def resolve_sample_idx_source(
    sample_idx_source: Any,
    *,
    boundary_sensor_idx: np.ndarray,
    boundary_query_idx: np.ndarray,
    poisson_sensor_idx: np.ndarray,
    poisson_query_idx: np.ndarray,
) -> np.ndarray | None:
    if sample_idx_source is None:
        return None
    if isinstance(sample_idx_source, (list, tuple, np.ndarray)):
        idx = np.asarray(sample_idx_source, dtype=np.int64).reshape(-1)
        return idx if idx.size > 0 else None
    src = str(sample_idx_source).strip()
    if src in {"", "none", "dense_mask"}:
        return None
    if src in {"boundary_operator.query_indices", "deeponet_task:boundary_operator.query_indices"}:
        return np.asarray(boundary_query_idx, dtype=np.int64).reshape(-1)
    if src in {"boundary_operator.sensor_indices", "deeponet_task:boundary_operator.sensor_indices"}:
        return np.asarray(boundary_sensor_idx, dtype=np.int64).reshape(-1)
    if src in {"poisson_head.query_indices", "deeponet_task:poisson_head.query_indices"}:
        return np.asarray(poisson_query_idx, dtype=np.int64).reshape(-1)
    if src in {"poisson_head.sensor_indices", "deeponet_task:poisson_head.sensor_indices"}:
        return np.asarray(poisson_sensor_idx, dtype=np.int64).reshape(-1)
    raise ValueError(f"Unsupported physics.boundary_operator.sample_idx_source: {src}")


def load_supervised_boundary_targets(
    target_path: Any,
    *,
    primary_qoi_key: str,
    expected_grid_shape: tuple[int, int],
    base_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    if target_path is None:
        return None
    path = Path(str(target_path))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"supervised boundary target file not found: {path}")
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        key = primary_qoi_key if primary_qoi_key in payload else (payload.files[0] if payload.files else None)
        if key is None:
            raise ValueError(f"supervised boundary target npz has no arrays: {path}")
        arr = np.asarray(payload[key], dtype=np.float32)
    elif path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
    else:
        raise ValueError("supervised boundary target must be .npz or .npy")
    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        arr = arr[:, None, ...]
    elif arr.ndim != 4:
        raise ValueError(f"unsupported supervised boundary target shape: {arr.shape}")
    if tuple(arr.shape[-2:]) != tuple(expected_grid_shape):
        raise ValueError(
            f"supervised boundary target grid mismatch: expected={expected_grid_shape}, actual={tuple(arr.shape[-2:])}"
        )
    return {str(primary_qoi_key): arr.astype(np.float32)}


def masked_r2_score(y_true: np.ndarray, y_pred: np.ndarray, mask_bhw: np.ndarray) -> float:
    m = np.asarray(mask_bhw, dtype=bool).reshape(-1)
    if not np.any(m):
        return float("nan")
    t = np.asarray(y_true, dtype=np.float32).reshape(-1)[m]
    p = np.asarray(y_pred, dtype=np.float32).reshape(-1)[m]
    denom = float(np.sum((t - float(np.mean(t))) ** 2))
    if denom <= 0.0:
        return 0.0
    numer = float(np.sum((t - p) ** 2))
    return float(1.0 - numer / max(denom, 1.0e-12))


def allvars_plasma_balance_score(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    y_vars: list[str],
    weights: dict[str, float],
    plasma_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    parts: dict[str, float] = {}
    score_sum = 0.0
    score_weight = 0.0
    mask_base = np.asarray(plasma_mask, dtype=bool)
    for idx, name in enumerate(y_vars):
        r2_val = masked_r2_score(target[:, idx], pred[:, idx], mask_base)
        parts[f"r2_{name}_plasma"] = float(r2_val)
        w = float(weights.get(name, 0.0))
        if w > 0.0 and np.isfinite(r2_val):
            score_sum += w * float(r2_val)
            score_weight += w
    if score_weight <= 0.0:
        return float("nan"), parts
    return float(score_sum / score_weight), parts
