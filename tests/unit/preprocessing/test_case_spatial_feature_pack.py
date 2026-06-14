from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.preprocessing.runner import PreprocessRunner


def _write_geometry(root: Path, shape: tuple[int, int]) -> Path:
    geom = root / "geometry"
    geom.mkdir(parents=True)
    mask = np.ones(shape, dtype=np.float32)
    mask[0, :] = 0.0
    np.save(geom / "mask_plasma.npy", mask)
    np.save(geom / "eps.npy", np.ones(shape, dtype=np.float32))
    np.save(geom / "wafer_mask.npy", mask)
    return root


def _write_structure(path: Path, shape: tuple[int, int], *, coil_col: int) -> None:
    mask_plasma = np.ones(shape, dtype=np.float32)
    mask_plasma[0, :] = 0.0
    mask_coil = np.zeros(shape, dtype=np.float32)
    mask_coil[1:, coil_col] = 1.0
    np.savez_compressed(
        path,
        mask_plasma=mask_plasma,
        mask_coil=mask_coil,
        valid_field_mask=np.ones(shape, dtype=np.float32),
        outside_mask=1.0 - mask_plasma,
    )


def _write_part_sdf_structure(path: Path, shape: tuple[int, int], *, coil_col: int, active_slots: int = 2) -> None:
    mask_plasma = np.ones(shape, dtype=np.float32)
    mask_plasma[0, :] = 0.0
    mask_coil = np.zeros(shape, dtype=np.float32)
    mask_coil[1:, coil_col] = 1.0
    payload = {
        "mask_plasma": mask_plasma,
        "mask_coil": mask_coil,
        "valid_field_mask": np.ones(shape, dtype=np.float32),
        "outside_mask": 1.0 - mask_plasma,
    }
    for idx in range(1, 7):
        sdf = np.full(shape, float(max(shape)), dtype=np.float32)
        if idx <= active_slots:
            sdf[:, :] = np.abs(np.arange(shape[1], dtype=np.float32).reshape(1, -1) - float(coil_col + idx - 1))
            sdf[1:, min(coil_col + idx - 1, shape[1] - 1)] *= -1.0
        payload[f"sdf_coil_{idx:02d}"] = sdf
    np.savez_compressed(path, **payload)


def test_preprocess_writes_case_spatial_pack_and_process_only_cond(tmp_path: Path) -> None:
    shape = (5, 6)
    dataset_root = _write_geometry(tmp_path / "dataset", shape)
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    cases = []
    for i in range(4):
        structure_path = structure_dir / f"case_{i}.npz"
        _write_structure(structure_path, shape, coil_col=1 + (i % 3))
        cases.append(
            {
                "case_id": f"case_{i}",
                "split_group": f"group_{i}",
                "cond": {"pp": float(i + 1), "pp0": float(i + 0.5)},
                "axis": 0.0,
                "structure_npz": str(structure_path),
                "y": {"phi": np.full(shape, float(i), dtype=np.float32)},
            }
        )

    runtime = {
        "input_mode": "table_plus_structure",
        "structure": {
            "feature_profile": "icp_struct_spatial_v1",
            "descriptor_profile": "none",
            "latent_profile": "none",
            "adapter_mode": "auto",
            "provider_mode": "fixed",
        },
    }
    cfg = {
        "split": {"seed": 1, "ratios": [0.5, 0.25, 0.25]},
        "cond_order": ["pp", "pp0"],
        "axis_schema": {"mode": "steady"},
        "scalers": {
            "fit_split": "extrap",
            "cond": "robust",
            "target_transforms": {
                "phi": {
                    "value_transform": "identity",
                    "scaler": "zscore",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "none"},
                }
            }
        },
        "coord_features": {
            "enabled": True,
            "channels_from_profile": "icp_struct_spatial_v1",
            "distance_transform_stats": {"enabled": True, "fit_scope": "train_split"},
            "scaling": {
                "enabled": True,
                "mode": "zscore",
                "fit_scope": "train_split",
                "mask_scope": "plasma_plus_band",
            },
        },
    }
    out = tmp_path / "preprocess"
    PreprocessRunner(cfg, out, runtime_cfg=runtime).run(cases=cases, geometry_root=dataset_root)

    with np.load(out / "features" / "static_spatial_feature_pack.npz", allow_pickle=True) as pack:
        assert pack["data"].shape == (5, *shape)
        assert [str(v) for v in pack["channels"].tolist()] == [
            "x",
            "y",
            "mask_plasma",
            "distance_signed",
            "distance_any",
        ]
        assert np.all(np.isfinite(pack["data"]))

    with np.load(out / "features" / "case_structure_feature_pack.npz", allow_pickle=True) as pack:
        assert pack["data"].shape == (4, 3, *shape)
        assert [str(v) for v in pack["channels"].tolist()] == [
            "mask_coil",
            "distance_coil",
            "coil_proximity",
        ]
        assert np.all(np.isfinite(pack["data"]))

    report = json.loads((out / "validation" / "report.json").read_text(encoding="utf-8"))
    assert report["case_spatial_pack_used"] is True
    assert report["scaler_fit_split"] == "extrap"
    assert report["case_spatial_feature_shape"] == [4, 8, *shape]
    assert report["static_spatial_feature_shape"] == [5, *shape]
    assert report["case_structure_feature_shape"] == [4, 3, *shape]
    assert report["coord_feature_pack_path"] == ""
    assert report["case_spatial_feature_pack_path"] == ""

    cond_schema = json.loads((out / "schema" / "cond_schema.json").read_text(encoding="utf-8"))
    assert cond_schema["order"] == ["pp", "pp0"]
    cond_scaler = json.loads((out / "scalers" / "cond_scaler.json").read_text(encoding="utf-8"))
    assert cond_scaler["cond_dim"] == 2
    assert cond_scaler["fit_split"] == "extrap"
    assert cond_scaler["type"] == "robust"

    structure_holdout = json.loads((out / "split" / "split_structure_holdout_v1.json").read_text(encoding="utf-8"))
    assert set(structure_holdout) == {"train", "val", "test"}

    coord_scaler = json.loads((out / "scalers" / "coord_feature_scaler.json").read_text(encoding="utf-8"))
    assert coord_scaler["channels"]["distance_coil"]["type"] == "zscore"
    assert coord_scaler["channels"]["mask_coil"]["type"] == "none"
    assert coord_scaler["channels"]["coil_proximity"]["type"] == "none"


def test_preprocess_writes_part_sdf_lite_case_pack(tmp_path: Path) -> None:
    shape = (5, 6)
    dataset_root = _write_geometry(tmp_path / "dataset", shape)
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    cases = []
    for i in range(4):
        structure_path = structure_dir / f"case_{i}.npz"
        _write_part_sdf_structure(structure_path, shape, coil_col=1 + (i % 2), active_slots=1 + (i % 3))
        cases.append(
            {
                "case_id": f"case_{i}",
                "split_group": f"group_{i}",
                "cond": {"pp": float(i + 1), "pp0": float(i + 0.5)},
                "axis": 0.0,
                "structure_npz": str(structure_path),
                "y": {"phi": np.full(shape, float(i), dtype=np.float32)},
            }
        )

    runtime = {
        "input_mode": "table_plus_structure",
        "structure": {
            "feature_profile": "icp_part_sdf_lite_v1",
            "descriptor_profile": "none",
            "latent_profile": "none",
            "adapter_mode": "auto",
            "provider_mode": "parametric_parts",
        },
    }
    cfg = {
        "split": {"seed": 1, "ratios": [0.5, 0.25, 0.25]},
        "cond_order": ["pp", "pp0"],
        "axis_schema": {"mode": "steady"},
        "scalers": {
            "fit_split": "extrap",
            "target_transforms": {
                "phi": {
                    "value_transform": "identity",
                    "scaler": "zscore",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "none"},
                }
            }
        },
        "coord_features": {
            "enabled": True,
            "channels_from_profile": "icp_part_sdf_lite_v1",
            "distance_transform_stats": {"enabled": True, "fit_scope": "train_split"},
            "scaling": {
                "enabled": True,
                "mode": "zscore",
                "fit_scope": "train_split",
                "mask_scope": "plasma_plus_band",
            },
        },
    }
    out = tmp_path / "preprocess"
    PreprocessRunner(cfg, out, runtime_cfg=runtime).run(cases=cases, geometry_root=dataset_root)

    with np.load(out / "features" / "case_structure_feature_pack.npz", allow_pickle=True) as pack:
        assert pack["data"].shape == (4, 9, *shape)
        assert [str(v) for v in pack["channels"].tolist()] == [
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
        assert np.all(np.isfinite(pack["data"]))

    report = json.loads((out / "validation" / "report.json").read_text(encoding="utf-8"))
    assert report["case_spatial_feature_shape"] == [4, 14, *shape]
    assert report["case_structure_feature_shape"] == [4, 9, *shape]

    coord_scaler = json.loads((out / "scalers" / "coord_feature_scaler.json").read_text(encoding="utf-8"))
    assert coord_scaler["channels"]["sdf_coil_01"]["type"] == "zscore"
    assert coord_scaler["channels"]["coil_proximity"]["type"] == "none"
    distance_stats = json.loads((out / "scalers" / "distance_transform_stats.json").read_text(encoding="utf-8"))
    assert float(distance_stats["coil_proximity_tau"]) > 0.0


def test_preprocess_writes_part_lite_case_pack(tmp_path: Path) -> None:
    shape = (5, 6)
    dataset_root = _write_geometry(tmp_path / "dataset", shape)
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    cases = []
    mask_plasma = np.ones(shape, dtype=np.float32)
    mask_plasma[0, :] = 0.0
    for i in range(4):
        structure_path = structure_dir / f"case_{i}.npz"
        part_col = 1 + (i % 3)
        part_stack = np.zeros((2, *shape), dtype=np.float32)
        part_stack[0, 1:, part_col] = 1.0
        part_stack[1, 2:4, min(part_col + 2, shape[1] - 1)] = 1.0
        np.savez_compressed(
            structure_path,
            mask_plasma=mask_plasma,
            valid_field_mask=np.ones(shape, dtype=np.float32),
            outside_mask=1.0 - mask_plasma,
            part_mask_stack=part_stack,
        )
        cases.append(
            {
                "case_id": f"case_{i}",
                "split_group": f"group_{i}",
                "cond": {"pp": float(i + 1), "pp0": float(i + 0.5)},
                "axis": 0.0,
                "structure_npz": str(structure_path),
                "y": {"phi": np.full(shape, float(i), dtype=np.float32)},
            }
        )

    runtime = {
        "input_mode": "table_plus_structure",
        "structure": {
            "feature_profile": "part_lite_v1",
            "descriptor_profile": "none",
            "latent_profile": "none",
            "adapter_mode": "auto",
            "provider_mode": "parametric_parts",
        },
    }
    cfg = {
        "split": {"seed": 1, "ratios": [0.5, 0.25, 0.25]},
        "cond_order": ["pp", "pp0"],
        "axis_schema": {"mode": "steady"},
        "scalers": {
            "fit_split": "extrap",
            "target_transforms": {
                "phi": {
                    "value_transform": "identity",
                    "scaler": "zscore",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "none"},
                }
            },
        },
        "coord_features": {
            "enabled": True,
            "channels_from_profile": "part_lite_v1",
            "distance_transform_stats": {"enabled": True, "fit_scope": "train_split"},
            "scaling": {
                "enabled": True,
                "mode": "zscore",
                "fit_scope": "train_split",
                "mask_scope": "plasma_plus_band",
            },
        },
    }
    out = tmp_path / "preprocess"
    PreprocessRunner(cfg, out, runtime_cfg=runtime).run(cases=cases, geometry_root=dataset_root)

    with np.load(out / "features" / "static_spatial_feature_pack.npz", allow_pickle=True) as pack:
        assert pack["data"].shape == (9, *shape)
        assert [str(v) for v in pack["channels"].tolist()] == [
            "x",
            "y",
            "mask_plasma",
            "distance_signed",
            "distance_any",
            "normal_x",
            "normal_y",
            "curvature_proxy",
            "boundary_band",
        ]
        assert np.all(np.isfinite(pack["data"]))

    with np.load(out / "features" / "case_structure_feature_pack.npz", allow_pickle=True) as pack:
        assert pack["data"].shape == (4, 4, *shape)
        assert [str(v) for v in pack["channels"].tolist()] == [
            "part_sdf_nearest",
            "part_sdf_second",
            "part_gap_proxy",
            "solid_proximity",
        ]
        assert np.all(np.isfinite(pack["data"]))

    report = json.loads((out / "validation" / "report.json").read_text(encoding="utf-8"))
    assert report["case_spatial_feature_shape"] == [4, 13, *shape]
    assert report["static_spatial_feature_shape"] == [9, *shape]
    assert report["case_structure_feature_shape"] == [4, 4, *shape]

    coord_scaler = json.loads((out / "scalers" / "coord_feature_scaler.json").read_text(encoding="utf-8"))
    assert coord_scaler["channels"]["boundary_band"]["type"] == "zscore"
    assert coord_scaler["channels"]["part_sdf_nearest"]["type"] == "zscore"
