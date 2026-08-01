from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_icp_stage4_part_lite_dataset import prepare_part_lite_dataset


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, duplicate: bool = False) -> tuple[Path, Path]:
    src = tmp_path / "src"
    (src / "fields").mkdir(parents=True)
    (src / "structure_features").mkdir(parents=True)
    np.savez_compressed(src / "fields" / "case_a.npz", ne=np.ones((2, 3), dtype=np.float32))
    np.savez_compressed(
        src / "structure_features" / "case_a.npz",
        r_coords=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        z_coords=np.asarray([0.0, 1.0], dtype=np.float32),
        mask_plasma=np.ones((2, 3), dtype=np.float32),
    )
    _write_csv(
        src / "index.csv",
        [
            {
                "case_id": "case_a",
                "nncoil": 2,
                "fields_npz": "fields/case_a.npz",
                "structure_npz": "structure_features/case_a.npz",
            }
        ],
    )
    (src / "conversion_summary.json").write_text(json.dumps({"n_cases": 1}), encoding="utf-8")
    layouts = tmp_path / "layouts"
    rows = [
        {"coil_index": 1, "active": 1, "r_min": 0, "r_max": 0, "z_min": 0, "z_max": 0},
        {"coil_index": 1 if duplicate else 2, "active": 1, "r_min": 2, "r_max": 2, "z_min": 1, "z_max": 1},
    ]
    _write_csv(layouts / "case_a__coil_layout.csv", rows)
    return src, layouts


def test_prepare_part_lite_dataset_writes_active_only_stack(tmp_path: Path) -> None:
    src, layouts = _fixture(tmp_path)
    dst = tmp_path / "dst"
    summary = prepare_part_lite_dataset(src_root=src, dst_root=dst, coil_layout_root=layouts)

    with np.load(dst / "structure_features" / "case_a.npz", allow_pickle=False) as data:
        assert data["part_mask_stack"].shape == (6, 2, 3)
        assert not np.any(data["part_mask_stack"][2:])
        assert data["part_ids"].tolist() == [f"coil_{idx:02d}" for idx in range(1, 7)]
        assert data["active_part_ids"].tolist() == ["coil_01", "coil_02"]
    assert summary["part_lite_v1_ready"] is True
    assert summary["part_count_distribution"] == {"2": 1}
    assert (dst / "fields" / "case_a.npz").exists()


def test_prepare_part_lite_dataset_rejects_duplicate_indices(tmp_path: Path) -> None:
    src, layouts = _fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate coil_index"):
        prepare_part_lite_dataset(src_root=src, dst_root=tmp_path / "dst", coil_layout_root=layouts)
