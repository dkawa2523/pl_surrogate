from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.preprocessing.runner import PreprocessRunner


def _write_parametric_geometry_root(root: Path) -> Path:
    geom = root / "geometry"
    geom.mkdir(parents=True, exist_ok=True)
    h, w = 8, 8
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    np.save(geom / "mask_plasma.npy", mask)
    np.save(geom / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(geom / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))
    part_stack = np.zeros((2, h, w), dtype=np.float32)
    part_stack[0, 2:4, 2:4] = 1.0
    part_stack[1, 4:6, 5:7] = 1.0
    np.savez_compressed(
        geom / "parts_pack.npz",
        mask_stack=part_stack,
        part_ids=np.asarray(["p0", "p1"], dtype=object),
    )
    manifest = {
        "part_ids": ["p0", "p1"],
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p0.ty": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p1.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p1.ty": {"default": 0.0, "min": -0.2, "max": 0.2},
        },
    }
    (geom / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _cases(n: int = 6, h: int = 8, w: int = 8) -> list[dict[str, object]]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for i in range(n):
        y_map = {
            "ne": rng.normal(size=(h, w)).astype(np.float32),
            "ni": rng.normal(size=(h, w)).astype(np.float32),
            "Te": rng.normal(size=(h, w)).astype(np.float32),
            "phi": rng.normal(size=(h, w)).astype(np.float32),
        }
        rows.append(
            {
                "case_id": f"case_{i:03d}",
                "cond": {"c0": float(i) / float(max(n - 1, 1)), "c1": float(i % 3) / 2.0},
                "axis": 0.0,
                "y": y_map,
            }
        )
    return rows


def _attach_case_structure_npz(
    cases: list[dict[str, object]],
    root: Path,
    *,
    h: int = 8,
    w: int = 8,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mask_plasma = np.ones((h, w), dtype=np.float32)
    mask_plasma[0, :] = 0.0
    mask_plasma[:, 0] = 0.0
    for i, case in enumerate(cases):
        part_stack = np.zeros((2, h, w), dtype=np.float32)
        part_stack[0, 2:4, 2:4] = 1.0
        if i % 2 == 0:
            part_stack[1, 4:6, 5:7] = 1.0
        path = root / f"{case['case_id']}.npz"
        np.savez_compressed(
            path,
            mask_plasma=mask_plasma,
            valid_field_mask=np.ones((h, w), dtype=np.float32),
            outside_mask=(mask_plasma <= 0.5).astype(np.float32),
            part_mask_stack=part_stack,
        )
        case["structure_npz"] = str(path)


def _preprocess_cfg() -> dict[str, object]:
    return {
        "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
        "axis_schema": {"mode": "steady", "harmonics": 1},
        "coord_features": {"enabled": False},
        "scalers": {
            "target_transforms": {
                "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
            }
        },
    }


def test_table_plus_descriptor_profile_writes_descriptor_artifacts(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    runner = PreprocessRunner(
        cfg=_preprocess_cfg(),
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "struct_desc_v1",
                "latent_profile": "none",
                "adapter_mode": "descriptor_branch",
                "provider_mode": "parametric_parts",
            },
        },
    )
    runner.run(_cases(), geometry_root)
    report_path = tmp_path / "pre" / "validation" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["structure_descriptor_pack_path"] == "features/structure_descriptor_pack.npz"
    assert int(report["structure_descriptor_dim"]) > 0
    assert (tmp_path / "pre" / "features" / "structure_descriptor_pack.npz").exists()
    assert (tmp_path / "pre" / "features" / "structure_descriptor_pack_meta.json").exists()


def test_case_structure_npz_writes_aligned_descriptor_matrix(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    cases = _cases()
    _attach_case_structure_npz(cases, tmp_path / "case_structures")
    runner = PreprocessRunner(
        cfg=_preprocess_cfg(),
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "struct_desc_v2",
                "latent_profile": "none",
                "adapter_mode": "descriptor_branch",
                "provider_mode": "parametric_parts",
            },
        },
    )

    runner.run(cases, geometry_root)

    pack_path = tmp_path / "pre" / "features" / "structure_descriptor_pack.npz"
    with np.load(pack_path, allow_pickle=True) as pack:
        assert pack["vectors"].shape == (len(cases), 6 + 2 * 9)
        assert pack["case_ids"].tolist() == [str(case["case_id"]) for case in cases]
        assert pack["n_parts_by_case"].tolist() == [2, 1, 2, 1, 2, 1]
        assert pack["n_part_slots_by_case"].tolist() == [2] * len(cases)
        assert not np.array_equal(pack["vectors"][0], pack["vectors"][1])
    meta = json.loads(
        (tmp_path / "pre" / "features" / "structure_descriptor_pack_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["case_specific"] is True
    assert meta["case_ids"] == [str(case["case_id"]) for case in cases]
    report = json.loads((tmp_path / "pre" / "validation" / "report.json").read_text(encoding="utf-8"))
    assert report["structure_descriptor_case_specific"] is True
    assert report["structure_descriptor_case_count"] == len(cases)


def test_case_structure_descriptor_uses_first_case_as_schema_reference_with_fixed_provider(
    tmp_path: Path,
) -> None:
    geometry_root = tmp_path / "dataset"
    geom = geometry_root / "geometry"
    geom.mkdir(parents=True)
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    np.save(geom / "mask_plasma.npy", mask)
    np.save(geom / "eps.npy", np.ones_like(mask))
    np.save(geom / "wafer_mask.npy", np.zeros_like(mask))
    cases = _cases()
    _attach_case_structure_npz(cases, tmp_path / "case_structures")
    runner = PreprocessRunner(
        cfg=_preprocess_cfg(),
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "struct_desc_v2",
                "latent_profile": "none",
                "adapter_mode": "descriptor_branch",
                "provider_mode": "fixed",
            },
        },
    )

    runner.run(cases, geometry_root)

    with np.load(tmp_path / "pre" / "features" / "structure_descriptor_pack.npz", allow_pickle=True) as pack:
        assert pack["vectors"].shape[0] == len(cases)
        assert pack["case_ids"].tolist() == [str(case["case_id"]) for case in cases]


def test_table_only_does_not_write_descriptor_artifacts(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    runner = PreprocessRunner(
        cfg=_preprocess_cfg(),
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_only",
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        },
    )
    runner.run(_cases(), geometry_root)
    report = json.loads((tmp_path / "pre" / "validation" / "report.json").read_text(encoding="utf-8"))
    assert report["structure_descriptor_pack_path"] == ""
    assert report["structure_descriptor_dim"] == 0
    assert not (tmp_path / "pre" / "features" / "structure_descriptor_pack.npz").exists()


def test_table_plus_latent_profile_requires_external_artifact(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    runner = PreprocessRunner(
        cfg=_preprocess_cfg(),
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "none",
                "latent_profile": "shape_ae_v1",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        },
    )
    with pytest.raises(ValueError, match="latent artifact is missing"):
        runner.run(_cases(), geometry_root)
