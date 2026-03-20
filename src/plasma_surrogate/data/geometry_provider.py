"""Load fixed geometry data from dataset folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_surrogate.data.geometry_context import (
    GeometryContext,
    build_coord_grid,
    build_distance_fields,
    build_signed_distance_fields,
)


class FixedGeometryProvider:
    """v1 fixed geometry provider with SDF-less fallback distance generation."""

    def __init__(self, dataset_root: str | Path, *, coord_grid_source: str = "coord_grid"):
        self.dataset_root = Path(dataset_root)
        self.geometry_root = self.dataset_root / "geometry"
        self.coord_grid_source = str(coord_grid_source).strip().lower()
        if self.coord_grid_source not in {"coord_grid", "rz_linear", "normalized_fallback"}:
            raise ValueError(
                "coord_grid_source must be one of: coord_grid, rz_linear, normalized_fallback"
            )
        self._ctx: GeometryContext | None = None

    def _load_array(self, rel_path: str, default: np.ndarray | None = None) -> np.ndarray | None:
        path = self.geometry_root / rel_path
        if path.exists():
            return np.load(path)
        return default

    def _build(self) -> GeometryContext:
        mask = self._load_array("mask_plasma.npy")
        if mask is None:
            raise FileNotFoundError("geometry/mask_plasma.npy is required")
        mask = mask.astype(np.float32)

        eps = self._load_array("eps.npy", default=np.ones_like(mask, dtype=np.float32))
        if eps is None:
            eps = np.ones_like(mask, dtype=np.float32)

        bc_dir_mask = self._load_array("bc_dir_mask.npy", default=np.zeros_like(mask, dtype=np.float32))
        if bc_dir_mask is None:
            bc_dir_mask = np.zeros_like(mask, dtype=np.float32)

        bc_dir_value = self._load_array("bc_dir_value.npy", default=np.zeros_like(mask, dtype=np.float32))
        if bc_dir_value is None:
            bc_dir_value = np.zeros_like(mask, dtype=np.float32)

        distance_any = self._load_array("distance_any.npy")
        dist0 = self._load_array("dist0.npy")
        distance_signed = self._load_array("distance_signed.npy")

        if distance_signed is None:
            if distance_any is not None:
                distance_signed = np.where(mask > 0.5, distance_any, -distance_any).astype(np.float32)
            else:
                distance_signed = build_signed_distance_fields(mask).astype(np.float32)
        else:
            distance_signed = np.asarray(distance_signed, dtype=np.float32)
            if distance_signed.shape != mask.shape:
                raise ValueError(
                    f"geometry/distance_signed.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(distance_signed.shape)}"
                )

        if distance_any is None:
            distance_any = np.abs(distance_signed).astype(np.float32)
        else:
            distance_any = np.asarray(distance_any, dtype=np.float32)
            if distance_any.shape != mask.shape:
                raise ValueError(
                    f"geometry/distance_any.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(distance_any.shape)}"
                )

        if dist0 is None:
            _, dist0 = build_distance_fields(mask, bc_dir_mask)
        else:
            dist0 = np.asarray(dist0, dtype=np.float32)
            if dist0.shape != mask.shape:
                raise ValueError(
                    f"geometry/dist0.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(dist0.shape)}"
                )

        coord_source = "normalized_fallback"
        coord = None
        if self.coord_grid_source == "coord_grid":
            coord = self._load_array("coord_grid.npy")
            if coord is not None:
                coord_source = "coord_grid"
        if coord is None and self.coord_grid_source in {"coord_grid", "rz_linear"}:
            r_coords = self._load_array("r_coords.npy")
            z_coords = self._load_array("z_coords.npy")
            if r_coords is not None and z_coords is not None:
                r = np.asarray(r_coords, dtype=np.float32).reshape(-1)
                z = np.asarray(z_coords, dtype=np.float32).reshape(-1)
                if int(r.shape[0]) == int(mask.shape[1]) and int(z.shape[0]) == int(mask.shape[0]):
                    rr, zz = np.meshgrid(r, z, indexing="xy")
                    coord = np.stack([rr, zz], axis=0).astype(np.float32)
                    coord_source = "rz_linear"
        if coord is None:
            coord = build_coord_grid(mask.shape)
            coord_source = "normalized_fallback"

        regions = {}
        wafer = self._load_array("wafer_mask.npy")
        if wafer is not None:
            regions["wafer_mask"] = wafer.astype(np.float32)

        return GeometryContext(
            mask_plasma=mask,
            distance_any=distance_any.astype(np.float32),
            dist0=dist0.astype(np.float32),
            eps=eps.astype(np.float32),
            coord_grid=coord.astype(np.float32),
            distance_signed=distance_signed.astype(np.float32),
            regions=regions,
            ops={"hx": 1.0, "hy": 1.0},
            bc_dir_mask=bc_dir_mask.astype(np.float32),
            bc_dir_value=bc_dir_value.astype(np.float32),
            coord_source=coord_source,
        )

    def get(self, geom_ref: dict[str, object] | None = None) -> GeometryContext:
        if geom_ref is not None and not isinstance(geom_ref, dict):
            raise TypeError("geom_ref must be dict when provided")
        if geom_ref is not None and "geom_param" in geom_ref and not isinstance(geom_ref["geom_param"], dict):
            raise TypeError("geom_param must be dict")
        if self._ctx is None:
            self._ctx = self._build()
        return self._ctx
