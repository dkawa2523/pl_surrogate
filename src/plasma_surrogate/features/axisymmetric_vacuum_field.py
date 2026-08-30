"""Deterministic axisymmetric vacuum-field features for circular coil layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


MU0 = 4.0e-7 * np.pi


def complete_elliptic_ke(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return complete elliptic K(m), E(m) using the AGM iteration."""

    m = np.clip(np.asarray(parameter, dtype=np.float64), 0.0, 1.0 - 1.0e-14)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    c0 = np.sqrt(m)
    correction = 0.5 * c0 * c0
    coefficient = 1.0
    for _ in range(16):
        c = 0.5 * (a - b)
        correction += coefficient * c * c
        next_a = 0.5 * (a + b)
        b = np.sqrt(a * b)
        a = next_a
        coefficient *= 2.0
        if float(np.max(np.abs(c))) < 1.0e-13:
            break
    k = np.pi / (2.0 * a)
    e = k * (1.0 - correction)
    return k, e


def circular_loop_vacuum_field(
    radial: np.ndarray,
    axial: np.ndarray,
    *,
    loop_radius: float,
    loop_axial: float,
    current: float = 1.0,
    softening: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Aphi, Br, Bz`` from one circular filament in SI units."""

    r = np.asarray(radial, dtype=np.float64)
    z = np.asarray(axial, dtype=np.float64)
    if r.shape != z.shape:
        raise ValueError(f"radial/axial shape mismatch: {r.shape} != {z.shape}")
    a = float(loop_radius)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("loop_radius must be finite and > 0")
    dz = z - float(loop_axial)
    eps2 = max(float(softening), 0.0) ** 2
    d2 = (a + r) ** 2 + dz**2 + eps2
    q2 = (a - r) ** 2 + dz**2 + eps2
    d = np.sqrt(np.maximum(d2, 1.0e-30))
    q2 = np.maximum(q2, 1.0e-30)
    m = np.clip(4.0 * a * np.maximum(r, 0.0) / d2, 0.0, 1.0 - 1.0e-14)
    k, e = complete_elliptic_ke(m)

    ratio = np.empty_like(m)
    small = m < 1.0e-7
    ratio[small] = np.pi * m[small] / 16.0
    ratio[~small] = ((2.0 - m[~small]) * k[~small] - 2.0 * e[~small]) / m[~small]
    aphi = MU0 * float(current) * a * ratio / (np.pi * d)

    bz = MU0 * float(current) * (k + ((a * a - r * r - dz * dz - eps2) / q2) * e) / (2.0 * np.pi * d)
    br = np.zeros_like(bz)
    off_axis = r > 1.0e-12
    br[off_axis] = (
        MU0
        * float(current)
        * dz[off_axis]
        * (-k[off_axis] + ((a * a + r[off_axis] ** 2 + dz[off_axis] ** 2 + eps2) / q2[off_axis]) * e[off_axis])
        / (2.0 * np.pi * r[off_axis] * d[off_axis])
    )
    if np.any(~off_axis):
        axis_d2 = a * a + dz[~off_axis] ** 2 + eps2
        bz[~off_axis] = MU0 * float(current) * a * a / (2.0 * axis_d2 ** 1.5)
        aphi[~off_axis] = 0.0
    return aphi.astype(np.float32), br.astype(np.float32), bz.astype(np.float32)


def coil_layout_vacuum_field(
    *,
    r_coords: np.ndarray,
    z_coords: np.ndarray,
    coils: Sequence[Mapping[str, float]],
    length_unit_to_m: float = 1.0e-2,
    quadrature_order: int = 3,
) -> dict[str, np.ndarray]:
    """Build smooth unit-current fields for a rectangular circular-coil layout."""

    if int(quadrature_order) not in {1, 3}:
        raise ValueError("quadrature_order must be 1 or 3")
    scale = float(length_unit_to_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("length_unit_to_m must be finite and > 0")
    r1 = np.asarray(r_coords, dtype=np.float64).reshape(-1) * scale
    z1 = np.asarray(z_coords, dtype=np.float64).reshape(-1) * scale
    rr, zz = np.meshgrid(r1, z1)
    aphi = np.zeros_like(rr, dtype=np.float64)
    br = np.zeros_like(rr, dtype=np.float64)
    bz = np.zeros_like(rr, dtype=np.float64)
    offsets = np.asarray([0.0] if int(quadrature_order) == 1 else [-1.0 / 3.0, 0.0, 1.0 / 3.0])
    n_filaments = float(offsets.size * offsets.size)
    grid_step = min(float(np.min(np.diff(r1))), float(np.min(np.diff(z1))))
    active = 0
    for raw in coils:
        if int(float(raw.get("active", 1.0))) <= 0:
            continue
        radius = float(raw["r_center"]) * scale
        axial = float(raw["z_center"]) * scale
        width = max(float(raw.get("width", 0.0)) * scale, grid_step)
        height = max(float(raw.get("height", 0.0)) * scale, grid_step)
        softening = max(0.2 * min(width, height), 0.5 * grid_step)
        for dr in offsets:
            for dz in offsets:
                ai, bri, bzi = circular_loop_vacuum_field(
                    rr,
                    zz,
                    loop_radius=radius + float(dr) * width,
                    loop_axial=axial + float(dz) * height,
                    current=1.0 / n_filaments,
                    softening=softening,
                )
                aphi += ai
                br += bri
                bz += bzi
        active += 1
    if active <= 0:
        raise ValueError("coil layout must contain at least one active coil")
    bmag = np.sqrt(br * br + bz * bz)
    payload = {
        "vacuum_aphi_unit": aphi.astype(np.float32),
        "vacuum_br_unit": br.astype(np.float32),
        "vacuum_bz_unit": bz.astype(np.float32),
        "vacuum_bmag_unit": bmag.astype(np.float32),
    }
    if any(not np.all(np.isfinite(value)) for value in payload.values()):
        raise ValueError("vacuum field generation produced non-finite values")
    return payload


__all__ = [
    "MU0",
    "circular_loop_vacuum_field",
    "coil_layout_vacuum_field",
    "complete_elliptic_ke",
]
