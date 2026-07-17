"""Physical-coordinate profile extraction for inference and assimilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext


@dataclass(frozen=True)
class RadialProfile:
    """A radial field profile sampled at one physical height."""

    r_m: np.ndarray
    z_m: float
    values: np.ndarray
    valid: np.ndarray


def _field_hw(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or tuple(arr.shape) != tuple(shape):
        raise ValueError(f"profile field must reduce to shape={shape}; got={arr.shape}")
    return arr


def _rectilinear_coords(geom_ctx: GeometryContext) -> tuple[np.ndarray, np.ndarray]:
    coord = np.asarray(geom_ctx.coord_grid, dtype=np.float64)
    shape = tuple(np.asarray(geom_ctx.mask_plasma).shape)
    if coord.shape != (2, *shape):
        raise ValueError(f"coord_grid must have shape={(2, *shape)}; got={coord.shape}")
    r = np.asarray(coord[0, 0, :], dtype=np.float64)
    z = np.asarray(coord[1, :, 0], dtype=np.float64)
    if not np.allclose(coord[0], r[None, :], rtol=0.0, atol=1.0e-9):
        raise ValueError("radial profile extraction requires a rectilinear r coordinate grid")
    if not np.allclose(coord[1], z[:, None], rtol=0.0, atol=1.0e-9):
        raise ValueError("radial profile extraction requires a rectilinear z coordinate grid")
    if np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("physical r/z coordinates must be strictly increasing")
    return r, z


def _bracket(coords: np.ndarray, value: float, *, name: str) -> tuple[int, int, float]:
    target = float(value)
    if not np.isfinite(target) or target < float(coords[0]) or target > float(coords[-1]):
        raise ValueError(f"{name}={target!r} is outside [{float(coords[0])}, {float(coords[-1])}]")
    hi = int(np.searchsorted(coords, target, side="left"))
    if hi < len(coords) and np.isclose(float(coords[hi]), target, rtol=0.0, atol=1.0e-12):
        return hi, hi, 0.0
    if hi == 0 or hi >= len(coords):
        raise ValueError(f"cannot bracket {name}={target!r}")
    lo = hi - 1
    weight = (target - float(coords[lo])) / (float(coords[hi]) - float(coords[lo]))
    return lo, hi, float(weight)


def extract_radial_profile(
    field: np.ndarray,
    geom_ctx: GeometryContext,
    *,
    z_m: float,
    r_m: np.ndarray | list[float] | tuple[float, ...],
    require_plasma: bool = True,
) -> RadialProfile:
    """Bilinearly sample a 2-D field at physical ``(r, z_m)`` locations.

    A sample is valid only when its interpolation stencil is finite and, when
    requested, every stencil corner belongs to the plasma mask. Invalid values
    are returned as NaN so callers cannot silently treat them as measurements.
    """

    shape = tuple(np.asarray(geom_ctx.mask_plasma).shape)
    values_2d = _field_hw(field, shape)
    mask = np.asarray(geom_ctx.mask_plasma, dtype=np.float64) > 0.5
    r_coords, z_coords = _rectilinear_coords(geom_ctx)
    requested_r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    if requested_r.size == 0:
        raise ValueError("r_m must contain at least one sample")

    z0, z1, wz = _bracket(z_coords, float(z_m), name="z_m")
    out = np.full(requested_r.shape, np.nan, dtype=np.float64)
    valid = np.zeros(requested_r.shape, dtype=bool)
    for idx, radius in enumerate(requested_r):
        r0, r1, wr = _bracket(r_coords, float(radius), name="r_m")
        corners = {(z0, r0), (z0, r1), (z1, r0), (z1, r1)}
        if require_plasma and not all(bool(mask[row, col]) for row, col in corners):
            continue
        if not all(np.isfinite(values_2d[row, col]) for row, col in corners):
            continue
        low = (1.0 - wr) * values_2d[z0, r0] + wr * values_2d[z0, r1]
        high = (1.0 - wr) * values_2d[z1, r0] + wr * values_2d[z1, r1]
        out[idx] = (1.0 - wz) * low + wz * high
        valid[idx] = True

    return RadialProfile(r_m=requested_r, z_m=float(z_m), values=out, valid=valid)


__all__ = ["RadialProfile", "extract_radial_profile"]
