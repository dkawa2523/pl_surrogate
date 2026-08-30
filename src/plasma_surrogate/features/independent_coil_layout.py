"""Canonical geometry contract for independently positioned ICP coils.

The same :class:`IndependentCoilLayout` is used to create both competing
representations: a padded dimension vector and a permutation-invariant union
signed-distance field.  Keeping geometry generation here prevents either
surrogate from receiving a different physical layout by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


MAX_COILS = 6
MIN_COILS = 2
DIMENSION_CHANNELS = (
    *(f"coil_active_{i:02d}" for i in range(1, MAX_COILS + 1)),
    *(f"coil_r_center_{i:02d}" for i in range(1, MAX_COILS + 1)),
    "coil_width",
    "coil_z_center",
)


@dataclass(frozen=True)
class IndependentCoilLayout:
    """Manufacturable row of equal-section coils with independent radii.

    ``r_centers`` are sorted radial centre positions.  All active coils share
    the same square cross-section and height, so only their radial placement
    differs in the first comparison experiment.
    """

    r_centers: tuple[float, ...]
    width: float
    z_center: float
    r_bounds: tuple[float, float] = (2.0, 30.0)
    z_bounds: tuple[float, float] = (14.0, 22.0)
    min_gap_fraction: float = 0.2

    def __post_init__(self) -> None:
        centers = tuple(float(value) for value in self.r_centers)
        object.__setattr__(self, "r_centers", centers)
        values = np.asarray((*centers, self.width, self.z_center, *self.r_bounds, *self.z_bounds), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("coil layout values must be finite")
        if not MIN_COILS <= len(centers) <= MAX_COILS:
            raise ValueError(f"coil count must be in [{MIN_COILS}, {MAX_COILS}]")
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("r_centers must be strictly increasing")
        if self.width <= 0.0:
            raise ValueError("width must be positive")
        if self.min_gap_fraction < 0.0:
            raise ValueError("min_gap_fraction must be non-negative")
        half = 0.5 * self.width
        if centers[0] - half < self.r_bounds[0] or centers[-1] + half > self.r_bounds[1]:
            raise ValueError("coil radial bounds exceed the allowed chamber interval")
        if self.z_center - half < self.z_bounds[0] or self.z_center + half > self.z_bounds[1]:
            raise ValueError("coil axial bounds exceed the allowed chamber interval")
        required_pitch = self.width * (1.0 + self.min_gap_fraction)
        if any(right - left < required_pitch for left, right in zip(centers, centers[1:])):
            raise ValueError("adjacent coils violate the minimum edge gap")

    @property
    def coil_count(self) -> int:
        return len(self.r_centers)

    def rows(self, *, case_id: str = "") -> list[dict[str, float | int | str]]:
        """Return the established six-slot ``coil_layout.csv`` representation."""

        rows: list[dict[str, float | int | str]] = []
        for index in range(1, MAX_COILS + 1):
            if index <= self.coil_count:
                center = self.r_centers[index - 1]
                half = 0.5 * self.width
                rows.append(
                    {
                        "case_id": case_id,
                        "coil_index": index,
                        "active": 1,
                        "order": index,
                        "r_min": center - half,
                        "r_max": center + half,
                        "z_min": self.z_center - half,
                        "z_max": self.z_center + half,
                        "r_center": center,
                        "z_center": self.z_center,
                        "width": self.width,
                        "height": self.width,
                        "pitch": "",
                    }
                )
            else:
                rows.append(
                    {
                        "case_id": case_id,
                        "coil_index": index,
                        "active": 0,
                        "order": index,
                        "r_min": "",
                        "r_max": "",
                        "z_min": "",
                        "z_max": "",
                        "r_center": "",
                        "z_center": "",
                        "width": "",
                        "height": "",
                        "pitch": "",
                    }
                )
        return rows

    def dimension_vector(self) -> np.ndarray:
        """Return active flags, padded centres, width and axial centre."""

        active = np.zeros(MAX_COILS, dtype=np.float32)
        centers = np.zeros(MAX_COILS, dtype=np.float32)
        active[: self.coil_count] = 1.0
        centers[: self.coil_count] = np.asarray(self.r_centers, dtype=np.float32)
        return np.concatenate(
            (active, centers, np.asarray([self.width, self.z_center], dtype=np.float32))
        )

    def mask(self, r_coords: Iterable[float], z_coords: Iterable[float]) -> np.ndarray:
        """Rasterize the union of closed coil rectangles."""

        rr, zz = _coordinate_mesh(r_coords, z_coords)
        result = np.zeros(rr.shape, dtype=bool)
        half = 0.5 * self.width
        for center in self.r_centers:
            result |= (np.abs(rr - center) <= half) & (np.abs(zz - self.z_center) <= half)
        return result.astype(np.float32)

    def union_sdf(self, r_coords: Iterable[float], z_coords: Iterable[float]) -> np.ndarray:
        """Analytic Euclidean union SDF in the same physical units as the grid.

        Values are negative inside a coil, zero at its boundary and positive
        outside.  Taking the minimum over rectangles makes the representation
        independent of coil labels or input ordering.
        """

        rr, zz = _coordinate_mesh(r_coords, z_coords)
        half = 0.5 * self.width
        fields = []
        for center in self.r_centers:
            q_r = np.abs(rr - center) - half
            q_z = np.abs(zz - self.z_center) - half
            outside = np.sqrt(np.maximum(q_r, 0.0) ** 2 + np.maximum(q_z, 0.0) ** 2)
            inside = np.minimum(np.maximum(q_r, q_z), 0.0)
            fields.append(outside + inside)
        return np.min(np.stack(fields, axis=0), axis=0).astype(np.float32)


def regular_layout(
    *,
    coil_count: int,
    r_min: float,
    r_max: float,
    width: float,
    z_center: float,
    **kwargs: object,
) -> IndependentCoilLayout:
    """Build the equal-spacing anchor used by the legacy ICP data."""

    centers = tuple(float(value) for value in np.linspace(r_min, r_max, int(coil_count)))
    return IndependentCoilLayout(centers, width=float(width), z_center=float(z_center), **kwargs)


def _coordinate_mesh(
    r_coords: Iterable[float], z_coords: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(tuple(r_coords), dtype=np.float64)
    z = np.asarray(tuple(z_coords), dtype=np.float64)
    if r.ndim != 1 or z.ndim != 1 or r.size == 0 or z.size == 0:
        raise ValueError("r_coords and z_coords must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)):
        raise ValueError("coordinates must be finite")
    rr, zz = np.meshgrid(r, z, indexing="xy")
    return rr, zz
