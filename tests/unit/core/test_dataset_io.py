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
