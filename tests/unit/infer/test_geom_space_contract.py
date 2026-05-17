from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.cases import parse_batch_cases
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def _write_base_geometry(dataset_root: Path, *, h: int = 8, w: int = 8) -> None:
    g = dataset_root / "geometry"
    g.mkdir(parents=True, exist_ok=True)
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    np.save(g / "mask_plasma.npy", mask)
    np.save(g / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(g / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))


def _write_parametric_parts(dataset_root: Path) -> None:
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.5, "max": 0.5},
        }
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["p0"], dtype=object), mask_stack=mask_stack.astype(np.float32))


def _write_icp_part_sdf_parts(dataset_root: Path) -> None:
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    part_ids = [f"coil_{i:02d}" for i in range(1, 7)]
    param_specs = {}
    for part_id in part_ids:
        param_specs[f"part.{part_id}.tx"] = {"default": 0.0, "min": -0.2, "max": 0.2}
        param_specs[f"part.{part_id}.ty"] = {"default": 0.0, "min": -0.2, "max": 0.2}
        param_specs[f"part.{part_id}.scale_x"] = {"default": 1.0, "min": 0.8, "max": 1.2}
        param_specs[f"part.{part_id}.scale_y"] = {"default": 1.0, "min": 0.8, "max": 1.2}
    manifest = {
        "part_ids": part_ids,
        "plasma_mode": "preserve",
        "param_specs": param_specs,
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mask_stack = np.zeros((6, 8, 8), dtype=np.float32)
    for i in range(6):
        mask_stack[i, 2:4, min(1 + i, 7)] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(part_ids, dtype=object), mask_stack=mask_stack)


def _write_layout_parts(dataset_root: Path) -> None:
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "part_ids": ["coil_01"],
        "plasma_mode": "preserve",
        "param_specs": {
            "layout.coil_01.r_center": {"default": 0.4, "min": 0.2, "max": 0.6},
            "layout.coil_01.z_center": {"default": 0.5, "min": 0.3, "max": 0.7},
            "layout.coil_01.width": {"default": 0.2, "min": 0.1, "max": 0.3},
            "layout.coil_01.height": {"default": 0.2, "min": 0.1, "max": 0.3},
        },
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["coil_01"], dtype=object), mask_stack=mask_stack)


def _build_engine(tmp_path: Path, provider_mode: str, *, input_mode: str) -> InferenceEngine:
    dataset_root = tmp_path / f"dataset_{provider_mode}_{input_mode}"
    if provider_mode == "parametric_parts":
        _write_parametric_parts(dataset_root)
    else:
        _write_base_geometry(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode=provider_mode)
    return InferenceEngine(
        model=GlobalMLP(input_dim=2, grid_shape=(8, 8), seed=3),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=provider,
        output_dir=tmp_path / f"infer_{provider_mode}_{input_mode}",
        input_mode=input_mode,
    )


def test_table_only_rejects_geom_space(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "fixed", input_mode="table_only")
    with pytest.raises(ValueError, match="geom_space"):
        engine.optimize_run(
            space={"c0": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=3,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_table_plus_structure_fixed_provider_rejects_geom_space(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "fixed", input_mode="table_plus_structure")
    with pytest.raises(ValueError, match="provider_mode=parametric_parts"):
        engine.optimize_run(
            space={"c0": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=3,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_table_plus_structure_parametric_provider_joint_optimize_smoke(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "parametric_parts", input_mode="table_plus_structure")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = engine.optimize_run(
            space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=4,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
            seed=5,
            backend="random",
        )
    assert "best_geom_param" in out
    assert "part.p0.tx" in out["best_geom_param"]
    assert out["invalid_trial_count"] >= 0
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    summary = json.loads((engine.store.root / "optimize" / "summary.json").read_text(encoding="utf-8"))
    assert summary["geom_space_enabled_effective"] is True
    assert summary["geom_param_keys_effective"] == ["part.p0.tx"]
    assert "invalid_trial_count" in summary


def test_parametric_batch_cases_run_through_engine(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "parametric_parts", input_mode="table_plus_structure")
    cases = parse_batch_cases(
        {
            "cases": [
                {
                    "case_id": "base",
                    "cond": {"c0": 0.1, "c1": 0.2},
                    "geom": {"geom_id": "default"},
                },
                {
                    "case_id": "shifted",
                    "cond": {"c0": 0.3, "c1": 0.4},
                    "geom": {"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}},
                },
            ]
        },
        cond_schema=engine.cond_schema,
        axis_schema=engine.axis_schema,
    )

    for case in cases:
        result = engine.single_run_aggregated(cond=case.cond, geom=case.geom, axis=case.axis)
        assert "uniformity" in result.qoi
        assert np.isfinite(result.fields_model["phi"]).all()

    single_dirs = [p for p in (engine.store.root / "single").iterdir() if p.is_dir()]
    assert len(single_dirs) == 2


def test_icp_part_sdf_lite_runtime_features_change_with_geom_param(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_icp_part_sdf"
    _write_icp_part_sdf_parts(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=2, grid_shape=(8, 8), seed=3),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=provider,
        output_dir=tmp_path / "infer_icp_part_sdf",
        input_mode="table_plus_structure",
        grid_input_features_cfg={"mode": "geom_feature_pack", "require_pack": "off"},
    )
    channels = [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
        "distance_coil",
        "coil_proximity",
        "sdf_coil_01",
        "sdf_coil_02",
        "sdf_coil_03",
        "sdf_coil_04",
        "sdf_coil_05",
        "sdf_coil_06",
    ]
    base = provider.get({"geom_id": "default", "geom_param": {}})
    shifted = provider.get({"geom_id": "default", "geom_param": {"part.coil_01.tx": 0.2}})
    rows_base = engine._build_grid_feature_rows(base, channels)
    rows_shifted = engine._build_grid_feature_rows(shifted, channels)
    assert rows_base.shape == (64, 14)
    assert np.allclose(rows_base[:, channels.index("mask_plasma")], rows_shifted[:, channels.index("mask_plasma")])
    assert not np.allclose(rows_base[:, channels.index("mask_coil")], rows_shifted[:, channels.index("mask_coil")])
    assert not np.allclose(rows_base[:, channels.index("sdf_coil_01")], rows_shifted[:, channels.index("sdf_coil_01")])
    assert np.all(np.isfinite(rows_shifted))


def test_layout_geom_space_rebuilds_rectangular_structure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_layout_parts"
    _write_layout_parts(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode="parametric_parts")
    base = provider.get({"geom_id": "default", "geom_param": {}})
    changed = provider.get({"geom_id": "default", "geom_param": {"layout.coil_01.r_center": 0.6}})

    assert np.allclose(base.mask_plasma, changed.mask_plasma)
    assert not np.allclose(base.regions["solid_union_mask"], changed.regions["solid_union_mask"])


def test_uniformity_values_for_plasma_mid_height_region() -> None:
    mask = np.zeros((5, 6), dtype=np.float32)
    mask[1:4, 1:5] = 1.0
    target = np.arange(30, dtype=np.float32).reshape(5, 6)

    vals, meta = InferenceEngine._uniformity_values_for_region(
        target,
        mask_plasma=mask,
        wafer_mask=None,
        region="plasma_mid_height",
        mid_height_band_px=0,
    )

    assert meta["uniformity_region"] == "plasma_mid_height"
    assert meta["uniformity_mid_height_row"] == 2.0
    assert meta["uniformity_sample_count"] == 4.0
    np.testing.assert_allclose(vals, target[2, 1:5])


def test_uniformity_values_for_plasma_mean_height_region() -> None:
    mask = np.zeros((6, 6), dtype=np.float32)
    mask[1, 1:3] = 1.0
    mask[4, 1:5] = 1.0
    target = np.arange(36, dtype=np.float32).reshape(6, 6)

    vals, meta = InferenceEngine._uniformity_values_for_region(
        target,
        mask_plasma=mask,
        wafer_mask=None,
        region="plasma_mean_height",
        mid_height_band_px=0,
    )

    assert meta["uniformity_region"] == "plasma_mean_height"
    assert meta["uniformity_mid_height_row"] == 4.0
    assert meta["uniformity_sample_count"] == 4.0
    np.testing.assert_allclose(vals, target[4, 1:5])


def test_cv_over_density_gain_discourages_low_or_negative_density() -> None:
    good_score, good_meta = InferenceEngine._cv_over_density_gain_score(
        np.asarray([9.0, 10.0, 11.0], dtype=np.float32),
        relative_uniformity=0.1,
        mean_density=10.0,
        density_ref=10.0,
    )
    low_score, _ = InferenceEngine._cv_over_density_gain_score(
        np.asarray([0.9, 1.0, 1.1], dtype=np.float32),
        relative_uniformity=0.1,
        mean_density=1.0,
        density_ref=10.0,
    )
    negative_score, negative_meta = InferenceEngine._cv_over_density_gain_score(
        np.asarray([-9.0, -10.0, -11.0], dtype=np.float32),
        relative_uniformity=0.1,
        mean_density=-10.0,
        density_ref=10.0,
    )

    assert good_meta["uniformity_density_gain"] == 1.0
    assert low_score > good_score
    assert negative_score > low_score
    assert negative_meta["uniformity_negative_penalty"] > 0.0
