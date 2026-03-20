"""Geometry context and distance fallback helpers."""

from __future__ import annotations

from collections import deque
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
    m = (mask_plasma > 0.5).astype(np.uint8)
    h, w = m.shape
    boundary = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            if m[i, j] == 0:
                continue
            neighbors = (
                m[i - 1, j] if i > 0 else 0,
                m[i + 1, j] if i < h - 1 else 0,
                m[i, j - 1] if j > 0 else 0,
                m[i, j + 1] if j < w - 1 else 0,
            )
            if any(v == 0 for v in neighbors):
                boundary[i, j] = 1
    return boundary


def _distance_from_seeds(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inf = float("inf")
    dist = np.full((h, w), inf, dtype=np.float32)
    q: deque[tuple[int, int]] = deque()

    for i in range(h):
        for j in range(w):
            if seed_mask[i, j] > 0 and mask[i, j] > 0:
                dist[i, j] = 0.0
                q.append((i, j))

    while q:
        i, j = q.popleft()
        base = dist[i, j]
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                continue
            if mask[ni, nj] <= 0:
                continue
            nd = base + 1.0
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                q.append((ni, nj))

    dist[mask <= 0] = 0.0
    dist[np.isinf(dist)] = 0.0
    return dist


def _outside_boundary_mask(mask_plasma: np.ndarray) -> np.ndarray:
    m = (mask_plasma > 0.5).astype(np.uint8)
    h, w = m.shape
    outside = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            if m[i, j] > 0:
                continue
            neighbors = (
                m[i - 1, j] if i > 0 else 0,
                m[i + 1, j] if i < h - 1 else 0,
                m[i, j - 1] if j > 0 else 0,
                m[i, j + 1] if j < w - 1 else 0,
            )
            if any(v > 0 for v in neighbors):
                outside[i, j] = 1
    return outside


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
