from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "convert_outputs_icp_stage4_enriched_to_csv_npz_core4.py"
SPEC = importlib.util.spec_from_file_location("convert_outputs_icp_stage4_enriched_core4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDNAMES = ["case_id", "coil_index", "active", "r_min", "r_max", "z_min", "z_max"]


def _write_layout(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_read_coil_layout_preserves_fixed_slots_and_audits_missing_rows(tmp_path: Path) -> None:
    path = _write_layout(
        tmp_path / "layout.csv",
        [
            {
                "case_id": "case_0",
                "coil_index": 1,
                "active": 1,
                "r_min": 0.0,
                "r_max": 1.0,
                "z_min": 0.0,
                "z_max": 1.0,
            },
            {"case_id": "case_0", "coil_index": 2, "active": 0},
        ],
    )

    out = MODULE._read_coil_layout(
        path,
        r_coords=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        z_coords=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
    )

    assert out["mask_stack"].shape == (6, 3, 3)
    assert out["active_count"] == 1
    assert out["layout_row_count"] == 2
    assert out["missing_slot_indices"] == [3, 4, 5, 6]


def test_read_coil_layout_rejects_duplicate_coil_index(tmp_path: Path) -> None:
    path = _write_layout(
        tmp_path / "duplicate.csv",
        [
            {"case_id": "case_0", "coil_index": 1, "active": 0},
            {"case_id": "case_0", "coil_index": 1, "active": 0},
        ],
    )

    with pytest.raises(ValueError, match="duplicate coil_index"):
        MODULE._read_coil_layout(
            path,
            r_coords=np.asarray([0.0, 1.0], dtype=np.float32),
            z_coords=np.asarray([0.0, 1.0], dtype=np.float32),
        )

def test_read_coil_layout_rejects_index_above_fixed_six_slots(tmp_path: Path) -> None:
    path = _write_layout(
        tmp_path / "overflow.csv",
        [{"case_id": "case_0", "coil_index": 7, "active": 0}],
    )

    with pytest.raises(ValueError, match="allowed=1..6"):
        MODULE._read_coil_layout(
            path,
            r_coords=np.asarray([0.0, 1.0], dtype=np.float32),
            z_coords=np.asarray([0.0, 1.0], dtype=np.float32),
        )
