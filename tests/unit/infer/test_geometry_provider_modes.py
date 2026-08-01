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


def _write_layout_artifacts(dataset_root: Path) -> None:
    h, w = 23, 31
    _write_base_geometry(dataset_root, h=h, w=w)
    g = dataset_root / "geometry"
    np.save(g / "r_coords.npy", np.linspace(0.0, 30.0, w, dtype=np.float32))
    np.save(g / "z_coords.npy", np.linspace(0.0, 22.0, h, dtype=np.float32))
    part_ids = ["coil_01", "coil_02"]
    manifest = {
        "part_ids": part_ids,
        "plasma_mode": "preserve",
        "param_specs": {
            "layout.coil_01.r_center": {"default": 5.0, "min": 0.0, "max": 30.0},
            "layout.coil_01.z_center": {"default": 15.0, "min": 0.0, "max": 22.0},
            "layout.coil_01.width": {"default": 2.0, "min": 0.0, "max": 6.0},
            "layout.coil_01.height": {"default": 2.0, "min": 0.0, "max": 6.0},
            "layout.coil_02.r_center": {"default": 10.0, "min": 0.0, "max": 30.0},
            "layout.coil_02.z_center": {"default": 15.0, "min": 0.0, "max": 22.0},
            "layout.coil_02.width": {"default": 0.0, "min": 0.0, "max": 6.0},
            "layout.coil_02.height": {"default": 0.0, "min": 0.0, "max": 6.0},
        },
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez(
        g / "parts_pack.npz",
        part_ids=np.array(part_ids, dtype=object),
        mask_stack=np.zeros((2, h, w), dtype=np.float32),
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


def test_parametric_provider_builds_layout_rectangles_and_allows_inactive_slots(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_layout"
    _write_layout_artifacts(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")

    base = provider.get({"geom_id": "default", "geom_param": {}})
    stack = np.asarray(base.regions["part_mask_stack"], dtype=np.float32)
    assert stack.shape == (2, 23, 31)
    assert [bool(np.sum(stack[idx]) > 0.0) for idx in range(2)] == [True, False]

    changed = provider.get(
        {
            "geom_id": "default",
            "geom_param": {
                "layout.coil_02.width": 2.0,
                "layout.coil_02.height": 2.0,
            },
        }
    )
    changed_stack = np.asarray(changed.regions["part_mask_stack"], dtype=np.float32)
    assert [bool(np.sum(changed_stack[idx]) > 0.0) for idx in range(2)] == [True, True]


def test_parametric_provider_rejects_negative_layout_size(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_layout_bad"
    _write_layout_artifacts(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")

    with pytest.raises(ValueError, match="must be >="):
        provider.get(
            {
                "geom_id": "default",
                "geom_param": {
                    "layout.coil_01.width": -1.0,
                },
            }
        )


def test_parametric_provider_selects_one_alternative_case_pack_without_union(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_alternatives"
    _write_base_geometry(dataset_root)
    geometry = dataset_root / "geometry"
    structure_root = dataset_root / "structure_features"
    structure_root.mkdir(parents=True)
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, :] = 0.0
    base2_stack = np.zeros((2, 8, 8), dtype=np.float32)
    base2_stack[0, 2, 1:5] = 1.0
    base2_stack[1, 3:6, 5] = 1.0
    base3_stack = np.zeros((2, 8, 8), dtype=np.float32)
    base3_stack[0, 5, 2:7] = 1.0
    base3_stack[1, 1:4, 2] = 1.0
    for base_name, stack in (("base2", base2_stack), ("base3", base3_stack)):
        np.savez_compressed(
            structure_root / f"{base_name}.npz",
            mask_plasma=mask,
            valid_field_mask=mask,
            outside_mask=1.0 - mask,
            part_mask_stack=stack,
            part_ids=np.asarray(["boundary_00", "boundary_01"], dtype=object),
        )
    np.savez_compressed(
        geometry / "parts_pack.npz",
        part_ids=np.asarray(["base2", "base3"], dtype=object),
        mask_stack=np.stack(
            [np.maximum.reduce(base2_stack, axis=0), np.maximum.reduce(base3_stack, axis=0)],
            axis=0,
        ),
    )
    (geometry / "parts_manifest.json").write_text(
        json.dumps(
            {
                "part_semantics": "alternatives",
                "default_alternative_id": "base2",
                "plasma_mode": "preserve",
                "part_ids": ["base2", "base3"],
                "structure_npz_by_alternative": {
                    "base2": "structure_features/base2.npz",
                    "base3": "structure_features/base3.npz",
                },
                "param_specs": {
                    "part.base2.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
                    "part.base3.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")

    base2 = provider.get({"geom_id": "base2"})
    base3 = provider.get({"geom_id": "base3"})

    assert base2.regions["part_ids"].tolist() == ["boundary_00", "boundary_01"]
    assert base3.regions["part_ids"].tolist() == ["boundary_00", "boundary_01"]
    assert np.array_equal(base2.regions["part_mask_stack"], base2_stack)
    assert np.array_equal(base3.regions["part_mask_stack"], base3_stack)
    assert not np.array_equal(base2.regions["solid_union_mask"], base3.regions["solid_union_mask"])
    assert np.array_equal(base2.mask_plasma, mask)
    assert np.array_equal(base3.mask_plasma, mask)
