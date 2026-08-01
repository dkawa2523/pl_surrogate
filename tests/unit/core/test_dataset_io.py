from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.core.dataset_io import load_csv_npz_dataset, load_dataset
from tests._config_presets import csv_npz_targets_four_field_example


def _write_geometry(root: Path, h: int = 6, w: int = 6) -> None:
    g = root / "geometry"
    g.mkdir(parents=True, exist_ok=True)
    np.save(g / "mask_plasma.npy", np.ones((h, w), dtype=np.float32))
    np.save(g / "eps.npy", np.ones((h, w), dtype=np.float32))
    np.save(g / "wafer_mask.npy", np.ones((h, w), dtype=np.float32))


def _write_case_npz(path: Path, h: int = 6, w: int = 6, *, include_phi: bool = True) -> None:
    payload = {
        "ne": np.ones((h, w), dtype=np.float32),
        "ni": np.ones((h, w), dtype=np.float32) * 1.1,
        "Te": np.ones((h, w), dtype=np.float32),
    }
    if include_phi:
        payload["phi"] = np.full((h, w), 2.0, dtype=np.float32)
    np.savez_compressed(path, **payload)


def _csv_npz_cfg(root: Path) -> dict[str, object]:
    return {
        "type": "csv_npz",
        "root": str(root),
        "index_csv": "index.csv",
        "cond_columns": ["c0", "c1", "c2"],
        "axis_column": "axis",
        "fields_npz_column": "fields_npz",
        "case_id_column": "case_id",
        "geometry_root": "geometry",
        "targets": csv_npz_targets_four_field_example(),
    }


def test_load_csv_npz_dataset_builds_cases_contract(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    _write_case_npz(root / "b.npz")

    index = root / "index.csv"
    index.write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,a.npz\n"
        "case_b,0.5,0.4,0.5,0.6,b.npz\n",
        encoding="utf-8",
    )

    ds = load_csv_npz_dataset(_csv_npz_cfg(root), run_dir=tmp_path)

    assert ds.shape == (6, 6)
    assert ds.cond_order == ["c0", "c1", "c2"]
    assert len(ds.cases) == 2
    assert ds.cases[0]["case_id"] == "case_a"
    assert np.isclose(float(ds.cases[1]["axis"]), 0.5)
    assert ds.geometry_root == root


def test_load_csv_npz_dataset_missing_fields_key_fails_fast(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "bad.npz", include_phi=False)

    index = root / "index.csv"
    index.write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_bad,0.0,0.1,0.2,0.3,bad.npz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing keys"):
        load_csv_npz_dataset(
            _csv_npz_cfg(root),
            run_dir=tmp_path,
        )


@pytest.mark.parametrize("legacy_key", ["output_vars", "output_key_map", "output_value_transform"])
def test_load_csv_npz_dataset_rejects_removed_legacy_target_keys(tmp_path: Path, legacy_key: str):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,a.npz\n",
        encoding="utf-8",
    )

    cfg = _csv_npz_cfg(root)
    cfg[legacy_key] = ["ne"]

    with pytest.raises(ValueError, match=f"dataset.{legacy_key} is removed"):
        load_csv_npz_dataset(cfg, run_dir=tmp_path)


def test_load_csv_npz_dataset_rejects_non_identity_target_transform_in_mainline(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    np.savez_compressed(
        root / "a.npz",
        log_ne=np.zeros((6, 6), dtype=np.float32),
        ni=np.ones((6, 6), dtype=np.float32),
        Te=np.ones((6, 6), dtype=np.float32),
        phi=np.ones((6, 6), dtype=np.float32),
    )
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,a.npz\n",
        encoding="utf-8",
    )
    cfg = _csv_npz_cfg(root)
    cfg["targets"] = [
        {"id": "ne", "source_key": "log_ne", "value_transform": "pow10"},
        {"id": "ni", "source_key": "ni", "value_transform": "identity"},
        {"id": "Te", "source_key": "Te", "value_transform": "identity"},
        {"id": "phi", "source_key": "phi", "value_transform": "identity"},
    ]

    with pytest.raises(ValueError, match="value_transform='pow10' is removed"):
        load_csv_npz_dataset(cfg, run_dir=tmp_path)


def test_load_dataset_dispatches_csv_npz(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,a.npz\n",
        encoding="utf-8",
    )
    out = load_dataset(
        {
            "dataset": {
                **_csv_npz_cfg(root),
            }
        },
        run_dir=tmp_path,
    )
    assert len(out.cases) == 1


def test_load_csv_npz_dataset_reads_split_group_and_base_case_id(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a0.npz")
    _write_case_npz(root / "a1.npz")
    (root / "index.csv").write_text(
        "case_id,base_case_id,split_group,axis,c0,c1,c2,fields_npz\n"
        "case_a__t000,case_a,case_a,0.0,0.1,0.2,0.3,a0.npz\n"
        "case_a__t001,case_a,case_a,0.1,0.1,0.2,0.3,a1.npz\n",
        encoding="utf-8",
    )
    ds = load_csv_npz_dataset(
        {
            **_csv_npz_cfg(root),
            "base_case_id_column": "base_case_id",
            "split_group_column": "split_group",
        },
        run_dir=tmp_path,
    )
    assert ds.cases[0]["base_case_id"] == "case_a"
    assert ds.cases[0]["split_group"] == "case_a"
    assert ds.cases[1]["split_group"] == "case_a"


def test_load_csv_npz_dataset_duplicate_case_id_fails_fast(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    _write_case_npz(root / "b.npz")
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_dup,0.0,0.1,0.2,0.3,a.npz\n"
        "case_dup,0.1,0.1,0.2,0.3,b.npz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicated case_id"):
        load_csv_npz_dataset(
            _csv_npz_cfg(root),
            run_dir=tmp_path,
        )


def test_load_csv_npz_dataset_reads_case_structure_reference_and_metadata(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    structure_dir = root / "structure_features"
    structure_dir.mkdir()
    mask = np.ones((6, 6), dtype=np.float32)
    part_stack = np.zeros((2, 6, 6), dtype=np.float32)
    part_stack[0, 1, :] = 1.0
    part_stack[1, 4, :] = 1.0
    np.savez_compressed(
        structure_dir / "base2.npz",
        mask_plasma=mask,
        valid_field_mask=mask,
        outside_mask=np.zeros_like(mask),
        part_mask_stack=part_stack,
    )
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,base_name,structure_output_key,structure_relative_path,structure_npz,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,base2,structure:model_mphtxt,structure/base2/model.mphtxt,"
        "structure_features/base2.npz,a.npz\n",
        encoding="utf-8",
    )
    cfg = {
        **_csv_npz_cfg(root),
        "structure_npz_column": "structure_npz",
        "structure_base_name_column": "base_name",
        "structure_output_key_column": "structure_output_key",
        "structure_source_path_column": "structure_relative_path",
    }

    ds = load_csv_npz_dataset(cfg, run_dir=tmp_path)

    case = ds.cases[0]
    assert case["structure_npz"] == str(structure_dir / "base2.npz")
    assert case["base_name"] == "base2"
    assert case["geom_id"] == "base2"
    assert case["structure_output_key"] == "structure:model_mphtxt"
    assert case["structure_relative_path"] == "structure/base2/model.mphtxt"


def test_load_csv_npz_dataset_validation_rejects_missing_pa_condition(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\ncase_a,0.0,0.1,0.2,0.3,a.npz\n",
        encoding="utf-8",
    )
    cfg = _csv_npz_cfg(root)
    cfg["validation"] = {"required_condition_columns": ["PA"]}

    with pytest.raises(ValueError, match="missing from dataset.cond_columns.*PA"):
        load_csv_npz_dataset(cfg, run_dir=tmp_path)


def test_load_csv_npz_dataset_validation_rejects_same_input_different_output(tmp_path: Path):
    root = tmp_path / "csv_ds"
    root.mkdir(parents=True)
    _write_geometry(root)
    _write_case_npz(root / "a.npz")
    _write_case_npz(root / "b.npz")
    with np.load(root / "b.npz") as payload:
        changed = {key: np.asarray(payload[key]).copy() for key in payload.files}
    changed["ne"][2, 2] = 9.0
    np.savez_compressed(root / "b.npz", **changed)
    (root / "index.csv").write_text(
        "case_id,axis,c0,c1,c2,fields_npz\n"
        "case_a,0.0,0.1,0.2,0.3,a.npz\n"
        "case_b,0.0,0.1,0.2,0.3,b.npz\n",
        encoding="utf-8",
    )
    cfg = _csv_npz_cfg(root)
    cfg["validation"] = {"reject_same_input_different_output": True}

    with pytest.raises(ValueError, match="identical model inputs with different outputs.*PA"):
        load_csv_npz_dataset(cfg, run_dir=tmp_path)
