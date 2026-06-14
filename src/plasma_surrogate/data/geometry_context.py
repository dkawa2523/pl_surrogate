"""Geometry context and distance fallback helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GeometryContext:
    mask_plasma: np.ndarray
    distance_any: np.ndarray
    dist0: np.ndarray
    eps: np.ndarray
    coord_grid: np.ndarray
    distance_signed: np.ndarray | None = None
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    ops: dict[str, float] = field(default_factory=lambda: {"hx": 1.0, "hy": 1.0})
    bc_dir_mask: np.ndarray | None = None
    bc_dir_value: np.ndarray | None = None
    coord_source: str = "normalized_fallback"


def _boundary_mask(mask_plasma: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_plasma > 0.5, dtype=bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (m & (~up | ~down | ~left | ~right)).astype(np.uint8)


def _distance_from_seeds(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask > 0, dtype=bool)
    seeds = np.asarray(seed_mask > 0, dtype=bool) & active
    h, w = active.shape
    if not np.any(seeds):
        return np.zeros((h, w), dtype=np.float32)

    inf = np.float32(h + w + 1)
    dist = np.where(seeds, np.float32(0.0), inf).astype(np.float32)

    for j in range(1, w):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j - 1] + np.float32(1.0))
    for j in range(w - 2, -1, -1):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j + 1] + np.float32(1.0))
    for i in range(1, h):
        dist[i, :] = np.minimum(dist[i, :], dist[i - 1, :] + np.float32(1.0))
    for i in range(h - 2, -1, -1):
        dist[i, :] = np.minimum(dist[i, :], dist[i + 1, :] + np.float32(1.0))

    dist[~active] = 0.0
    dist[dist >= inf] = 0.0
    return dist.astype(np.float32)


def _outside_boundary_mask(mask_plasma: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_plasma > 0.5, dtype=bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (~m & (up | down | left | right)).astype(np.uint8)


def build_signed_distance_fields(mask_plasma: np.ndarray) -> np.ndarray:
    """
    Build signed Manhattan distance on a 2D mask.

    Returns:
      - positive distances in plasma region
      - negative distances in chamber region
      - zero on the plasma boundary interface
    """

    m = (mask_plasma > 0.5).astype(np.uint8)
    inside_boundary = _boundary_mask(m)
    outside_boundary = _outside_boundary_mask(m)
    inside = _distance_from_seeds(m, inside_boundary)
    outside = _distance_from_seeds((1 - m).astype(np.uint8), outside_boundary)
    if np.any(m == 0):
        outside = np.where(m > 0, outside, outside + np.float32(1.0)).astype(np.float32)
    signed = np.where(m > 0, inside, -outside).astype(np.float32)
    return signed


def build_distance_fields(mask_plasma: np.ndarray, bc_dir_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build boundary distance fields for SDF-less datasets."""

    m = (mask_plasma > 0.5).astype(np.uint8)
    signed = build_signed_distance_fields(m)
    distance_any = np.abs(signed).astype(np.float32)
    boundary = _boundary_mask(m)
    dist_in = _distance_from_seeds(m, boundary)

    if bc_dir_mask is None:
        dist0 = dist_in.copy()
    else:
        dist0 = _distance_from_seeds(m, (bc_dir_mask > 0.5).astype(np.uint8))

    return distance_any.astype(np.float32), dist0.astype(np.float32)


def build_coord_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return coordinate channels [2, H, W] in [0,1]."""

    h, w = shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    return np.stack([xv, yv], axis=0)


def to_dict(ctx: GeometryContext) -> dict[str, Any]:
    return {
        "mask_plasma": ctx.mask_plasma,
        "distance_any": ctx.distance_any,
        "dist0": ctx.dist0,
        "distance_signed": ctx.distance_signed,
        "eps": ctx.eps,
        "coord_grid": ctx.coord_grid,
        "regions": ctx.regions,
        "ops": ctx.ops,
        "bc_dir_mask": ctx.bc_dir_mask,
        "bc_dir_value": ctx.bc_dir_value,
        "coord_source": ctx.coord_source,
    }
