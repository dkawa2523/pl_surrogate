"""Synthetic dataset builder for mainline end-to-end commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SyntheticDataset:
    cases: list[dict[str, Any]]
    cond_order: list[str]
    geometry_root: Path
    shape: tuple[int, int]
    structure_root: Path | None = None
    target_metadata: list[dict[str, Any]] = field(default_factory=list)


def synthetic_target_metadata() -> list[dict[str, Any]]:
    """Metadata for the built-in synthetic fixture."""

    return [
        {
            "id": "ne",
            "role": "density_electron",
            "positive": True,
            "field_family": "density",
            "default_region": "plasma_only",
        },
        {
            "id": "ni",
            "role": "density_ion",
            "positive": True,
            "field_family": "density",
            "default_region": "plasma_only",
        },
        {
            "id": "Te",
            "role": "temperature_electron",
            "positive": True,
            "field_family": "temperature",
            "default_region": "plasma_only",
        },
        {
            "id": "phi",
            "role": "potential",
            "positive": False,
            "field_family": "electrostatic",
            "default_region": "all",
        },
    ]


def build_synthetic_dataset(cfg: dict[str, Any], output_root: str | Path) -> SyntheticDataset:
    """Build a deterministic synthetic dataset and geometry files."""

    n_cases = int(cfg.get("n_cases", 12))
    h = int(cfg.get("height", 8))
    w = int(cfg.get("width", 8))
    cond_dim = int(cfg.get("cond_dim", 3))
    seed = int(cfg.get("seed", 0))
    axis_mode = str(cfg.get("axis_mode", cfg.get("axis", {}).get("mode", "steady")))

    rng = np.random.default_rng(seed)
    cond_order = [f"c{i}" for i in range(cond_dim)]

    cond = rng.uniform(0.0, 1.0, size=(n_cases, cond_dim)).astype(np.float32)

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")

    y = np.zeros((n_cases, 4, h, w), dtype=np.float32)
    for i in range(n_cases):
        c = cond[i]
        y[i, 0] = c[0] + 0.3 * xv + 0.2 * yv
        y[i, 1] = c[min(0, cond_dim - 1)] + 0.25 * xv + 0.15 * yv
        y[i, 2] = c[min(1, cond_dim - 1)] + 0.1 * np.sin(2 * np.pi * xv)
        y[i, 3] = c[min(2, cond_dim - 1)] + 0.2 * np.cos(2 * np.pi * yv)

    if axis_mode == "steady":
        axis_values = np.zeros((n_cases,), dtype=np.float32)
    elif axis_mode in {"time", "phase_sincos"}:
        axis_values = np.linspace(0.0, 1.0, n_cases, endpoint=False, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported synthetic axis_mode: {axis_mode}")

    out_root = Path(output_root)
    geom_root = out_root / "dataset"
    geom_dir = geom_root / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)

    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0
    mask[-1, 0:2] = 0
    eps = np.ones((h, w), dtype=np.float32)
    wafer = np.zeros((h, w), dtype=np.float32)
    wafer[-2:, :] = mask[-2:, :]

    np.save(geom_dir / "mask_plasma.npy", mask)
    np.save(geom_dir / "eps.npy", eps)
    np.save(geom_dir / "wafer_mask.npy", wafer)

    cases: list[dict[str, Any]] = []
    for i in range(n_cases):
        cond_dict = {k: float(cond[i, j]) for j, k in enumerate(cond_order)}
        cases.append(
            {
                "case_id": f"case_{i:03d}",
                "cond": cond_dict,
                "axis": float(axis_values[i]),
                "y": {
                    "ne": y[i, 0],
                    "ni": y[i, 1],
                    "Te": y[i, 2],
                    "phi": y[i, 3],
                },
            }
        )

    return SyntheticDataset(
        cases=cases,
        cond_order=cond_order,
        geometry_root=geom_root,
        shape=(h, w),
        target_metadata=synthetic_target_metadata(),
    )
