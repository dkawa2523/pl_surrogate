from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.data.geometry_provider import (
    FixedGeometryProvider,
    ParametricPartsGeometryProvider,
    build_geometry_provider,
)


def _write_base_geometry(dataset_root: Path, *, h: int = 8, w: int = 8) -> None:
    g = dataset_root / "geometry"
    g.mkdir(parents=True, exist_ok=True)
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    np.save(g / "mask_plasma.npy", mask)
    np.save(g / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(g / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))


def _write_parametric_artifacts(dataset_root: Path) -> None:
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.5, "max": 0.5},
            "part.p0.ty": {"default": 0.0, "min": -0.5, "max": 0.5},
            "part.p0.scale_x": {"default": 1.0, "min": 0.5, "max": 1.5},
            "part.p0.scale_y": {"default": 1.0, "min": 0.5, "max": 1.5},
            "part.p0.rotation_deg": {"default": 0.0, "min": -30.0, "max": 30.0},
            "part.p0.fillet": {"default": 0.0, "min": -0.2, "max": 0.2},
        }
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(
        g / "parts_pack.npz",
        part_ids=np.array(["p0"], dtype=object),
        mask_stack=mask_stack.astype(np.float32),
    )


def test_fixed_provider_rejects_geom_param(tmp_path: Path) -> None:
    _write_base_geometry(tmp_path / "dataset")
    provider = FixedGeometryProvider(tmp_path / "dataset")
    with pytest.raises(ValueError, match="provider_mode=fixed"):
        provider.get({"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}})


def test_parametric_provider_requires_manifest_and_pack(tmp_path: Path) -> None:
    _write_base_geometry(tmp_path / "dataset_missing_parts")
    provider = ParametricPartsGeometryProvider(tmp_path / "dataset_missing_parts")
    with pytest.raises(FileNotFoundError, match="parts_manifest.json"):
        provider.get({"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}})


def test_parametric_provider_rebuilds_geometry_context(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_parts"
    _write_parametric_artifacts(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")

    base = provider.get({"geom_id": "default"})
    moved = provider.get({"geom_id": "default", "geom_param": {"part.p0.tx": 0.2}})
    assert base.mask_plasma.shape == moved.mask_plasma.shape
    assert not np.array_equal(base.mask_plasma, moved.mask_plasma)
    assert not np.array_equal(base.distance_signed, moved.distance_signed)


def test_parametric_provider_rejects_manifest_min_gt_max(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_bad_bounds"
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": 0.2, "max": -0.2},
        }
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["p0"], dtype=object), mask_stack=mask_stack.astype(np.float32))

    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")
    with pytest.raises(ValueError, match="requires min <= max"):
        provider.get({"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}})


def test_parametric_provider_rejects_manifest_unknown_part(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_unknown_part"
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "param_specs": {
            "part.p9.tx": {"default": 0.0, "min": -0.5, "max": 0.5},
        }
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["p0"], dtype=object), mask_stack=mask_stack.astype(np.float32))

    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")
    with pytest.raises(ValueError, match="unknown part id"):
        provider.get({"geom_id": "default", "geom_param": {"part.p9.tx": 0.0}})


def test_parametric_provider_rejects_manifest_pack_part_id_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_part_id_mismatch"
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "part_ids": ["p0"],
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.5, "max": 0.5},
        },
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["p1"], dtype=object), mask_stack=mask_stack.astype(np.float32))

    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")
    with pytest.raises(ValueError, match="part_ids and parts_pack.npz part_ids must match"):
        provider.get({"geom_id": "default", "geom_param": {"part.p0.tx": 0.0}})
