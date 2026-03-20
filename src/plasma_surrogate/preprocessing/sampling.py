"""Sampling artifacts for point-based and patch-based training."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_point_pools(
    mask_plasma: np.ndarray,
    distance_any: np.ndarray,
    delta_edge: float = 1.0,
    delta_bulk: float = 3.0,
    wafer_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mask = (mask_plasma > 0.5)
    pools: dict[str, np.ndarray] = {}

    idx_all = np.flatnonzero(mask)
    pools["idx_plasma"] = idx_all.astype(np.int64)

    idx_boundary = np.flatnonzero(mask & (distance_any <= delta_edge))
    pools["idx_boundary_band"] = idx_boundary.astype(np.int64)

    idx_bulk = np.flatnonzero(mask & (distance_any >= delta_bulk))
    if len(idx_bulk) == 0:
        idx_bulk = idx_all
    pools["idx_bulk"] = idx_bulk.astype(np.int64)

    if wafer_mask is not None:
        idx_wafer = np.flatnonzero((wafer_mask > 0.5) & mask)
        pools["idx_wafer_band"] = idx_wafer.astype(np.int64)

    return pools


def sample_fixed_indices(
    pool: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    if len(pool) == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    replace = len(pool) < n
    return rng.choice(pool, size=n, replace=replace).astype(np.int64)


def build_patch_index(
    shape: tuple[int, int],
    patch_hw: tuple[int, int],
    stride_hw: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    ph, pw = patch_hw
    sh, sw = stride_hw
    out: list[tuple[int, int]] = []
    for i in range(0, max(1, h - ph + 1), max(1, sh)):
        for j in range(0, max(1, w - pw + 1), max(1, sw)):
            out.append((i, j))
    if not out:
        out = [(0, 0)]
    return np.array(out, dtype=np.int64)


def build_phase_wrap_pairs(
    sample_ids: list[int],
    axis_values: list[float] | np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if len(sample_ids) < 2:
        return []
    if axis_values is None:
        return [(sample_ids[0], sample_ids[-1])]
    vals = np.asarray(axis_values, dtype=np.float32).reshape(-1)
    if vals.shape[0] != len(sample_ids):
        raise ValueError(
            "build_phase_wrap_pairs axis_values length mismatch: "
            f"expected={len(sample_ids)} got={vals.shape[0]}"
        )
    vals = np.mod(vals, 1.0)
    order = np.argsort(vals)
    lo = int(order[0])
    hi = int(order[-1])
    if lo == hi:
        return []
    return [(int(sample_ids[lo]), int(sample_ids[hi]))]


def build_time_adjacent_pairs(
    sample_ids: list[int],
    axis_values: list[float] | np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if len(sample_ids) < 2:
        return []
    if axis_values is None:
        return [(sample_ids[i], sample_ids[i + 1]) for i in range(len(sample_ids) - 1)]
    vals = np.asarray(axis_values, dtype=np.float32).reshape(-1)
    if vals.shape[0] != len(sample_ids):
        raise ValueError(
            "build_time_adjacent_pairs axis_values length mismatch: "
            f"expected={len(sample_ids)} got={vals.shape[0]}"
        )
    order = np.argsort(vals)
    ids = [int(sample_ids[int(i)]) for i in order]
    return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]


def build_deeponet_indices(
    n_points: int,
    n_sensors: int,
    n_queries: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if int(n_points) <= 0:
        raise ValueError("n_points must be > 0")
    if int(n_sensors) <= 0:
        raise ValueError("n_sensors must be > 0")
    if int(n_queries) <= 0:
        raise ValueError("n_queries must be > 0")

    pool = np.arange(int(n_points), dtype=np.int64)
    sensors = sample_fixed_indices(pool=pool, n=int(n_sensors), seed=int(seed))
    queries = sample_fixed_indices(pool=pool, n=int(n_queries), seed=int(seed) + 1)
    return {
        "sensor_indices": sensors.astype(np.int64),
        "query_indices": queries.astype(np.int64),
    }


def build_flattened_coords(
    coord_grid: np.ndarray,
    order: str = "C",
) -> np.ndarray:
    cg = np.asarray(coord_grid, dtype=np.float32)
    if cg.ndim != 3 or cg.shape[0] != 2:
        raise ValueError("coord_grid must be [2,H,W]")
    flat_x = cg[0].reshape(-1, order=order)
    flat_y = cg[1].reshape(-1, order=order)
    return np.stack([flat_x, flat_y], axis=1).astype(np.float32)


def save_sampling_artifacts(path: str, payload: dict[str, Any]) -> None:
    np.savez_compressed(path, **payload)
