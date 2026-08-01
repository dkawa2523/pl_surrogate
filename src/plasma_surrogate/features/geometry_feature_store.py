"""Featurization cache for deterministic geometry-derived artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.data.geometry_provider import GeometryProviderLike


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hash_bytes(a.tobytes())


def hash_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hash_bytes(canonical.encode("utf-8"))


def _harmonic_extension(mask: np.ndarray, value: np.ndarray, active: np.ndarray, n_iters: int = 64) -> np.ndarray:
    phi = np.zeros_like(value, dtype=np.float32)
    phi = (1.0 - mask) * phi + mask * value
    for _ in range(max(1, int(n_iters))):
        p = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
        neigh = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
        phi = active * ((1.0 - mask) * neigh + mask * value)
    return phi.astype(np.float32)


class GeometryFeatureStore:
    def __init__(self, root: str | Path, delta_bulk: float = 3.0, strict_hash_check: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.delta_bulk = float(delta_bulk)
        self.strict_hash_check = bool(strict_hash_check)
        self.cache_root = self.root / "geometry_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def geom_key(geom_ref: dict[str, Any]) -> str:
        if "geom_param" in geom_ref:
            params = {k: float(v) for k, v in sorted(dict(geom_ref["geom_param"]).items())}
            digest = hash_json({"geom_id": geom_ref.get("geom_id", "default"), "geom_param": params})
            return f"param_{digest[:16]}"
        return str(geom_ref.get("geom_id", "default"))

    def _dir(self, geom_ref: dict[str, Any]) -> Path:
        d = self.cache_root / self.geom_key(geom_ref)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, geom_ref: dict[str, Any]) -> Path:
        return self._dir(geom_ref) / "meta.json"

    def _build_payload(self, ctx: GeometryContext, axis_value: float, axis_mode: str) -> dict[str, np.ndarray]:
        bc_mask = np.zeros_like(ctx.mask_plasma, dtype=np.float32) if ctx.bc_dir_mask is None else np.asarray(ctx.bc_dir_mask, dtype=np.float32)
        bc_value = np.zeros_like(ctx.mask_plasma, dtype=np.float32) if ctx.bc_dir_value is None else np.asarray(ctx.bc_dir_value, dtype=np.float32)
        active = (ctx.mask_plasma > 0.5).astype(np.float32)
        mask_bulk = ((ctx.distance_any > self.delta_bulk) & (ctx.mask_plasma > 0.5)).astype(np.float32)
        if float(np.sum(mask_bulk)) <= 0:
            mask_bulk = active.astype(np.float32)

        # v1: one-group basis cache. This keeps the interface stable for future group expansion.
        basis_mask = np.where(bc_mask > 0.5, 1.0, 0.0).astype(np.float32)
        bc_basis = _harmonic_extension(mask=basis_mask, value=np.ones_like(basis_mask), active=active, n_iters=64)[None, ...]
        bc_coeffs = np.array([1.0], dtype=np.float32)
        phi_bc_ext = _harmonic_extension(mask=bc_mask, value=bc_value, active=active, n_iters=64)

        channel_names = np.array(
            ["coord_x", "coord_y", "mask_plasma", "distance_any", "dist0", "eps", "phi_bc_ext"],
            dtype=object,
        )
        x_grid = np.stack(
            [
                ctx.coord_grid[0],
                ctx.coord_grid[1],
                ctx.mask_plasma,
                ctx.distance_any,
                ctx.dist0,
                ctx.eps,
                phi_bc_ext,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "distance_any": ctx.distance_any.astype(np.float32),
            "dist0": ctx.dist0.astype(np.float32),
            "mask_bulk": mask_bulk.astype(np.float32),
            "bc_dir_mask": bc_mask.astype(np.float32),
            "bc_dir_value": bc_value.astype(np.float32),
            "eps": ctx.eps.astype(np.float32),
            "bc_basis": bc_basis.astype(np.float32),
            "bc_coeffs": bc_coeffs.astype(np.float32),
            "phi_bc_ext": phi_bc_ext.astype(np.float32),
            "x_grid": x_grid,
            "channel_names": channel_names,
            "axis_value": np.array([float(axis_value)], dtype=np.float32),
            "axis_mode_code": np.array([0.0 if axis_mode == "steady" else 1.0], dtype=np.float32),
        }

    def prepare(
        self,
        geom_ref: dict[str, Any],
        axis_value: float,
        axis_mode: str,
        geometry_provider: GeometryProviderLike,
        force: bool = False,
    ) -> dict[str, Any]:
        cache_dir = self._dir(geom_ref)
        meta_path = self._meta_path(geom_ref)
        if meta_path.exists() and not force:
            with meta_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        ctx = geometry_provider.get(geom_ref)
        payload = self._build_payload(ctx, axis_value=axis_value, axis_mode=axis_mode)
        hashes: dict[str, str] = {}
        for name, arr in payload.items():
            path = cache_dir / f"{name}.npy"
            np.save(path, arr)
            hashes[name] = hash_array(arr)

        meta = {
            "geom_ref": geom_ref,
            "axis_mode": axis_mode,
            "axis_value": float(axis_value),
            "hashes": hashes,
            "feature_hash": hash_json(hashes),
            "delta_bulk": self.delta_bulk,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return meta

    def load_meta(self, geom_ref: dict[str, Any]) -> dict[str, Any]:
        with self._meta_path(geom_ref).open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_feature(self, geom_ref: dict[str, Any], name: str) -> np.ndarray:
        return np.load(self._dir(geom_ref) / f"{name}.npy")

    def get_context(
        self,
        geom_ref: dict[str, Any],
        axis_value: float,
        axis_mode: str,
        geometry_provider: GeometryProviderLike,
    ) -> GeometryContext:
        meta = self.prepare(geom_ref, axis_value=axis_value, axis_mode=axis_mode, geometry_provider=geometry_provider)
        ctx = geometry_provider.get(geom_ref)
        regions = dict(ctx.regions)
        for name in ["mask_bulk", "bc_basis", "bc_coeffs", "phi_bc_ext"]:
            arr = self.load_feature(geom_ref, name)
            if self.strict_hash_check:
                digest = hash_array(arr)
                if digest != meta["hashes"].get(name):
                    raise ValueError(f"Feature hash mismatch for {name}: {digest} != {meta['hashes'].get(name)}")
            regions[name] = arr.astype(np.float32)
        ctx.regions = regions
        return ctx
