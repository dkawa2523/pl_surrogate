from __future__ import annotations

import csv
import runpy
from pathlib import Path


SCRIPT = Path("experiments/icp_stage4/scripts/summarize_icp_stage4_spatial_learning.py")


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summary_marks_missing_variants_incomplete(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="spatial_learning_summary_test")
    baseline = []
    for target in module["TARGETS"]:
        baseline.append(
            {
                "model": "U-Net compact",
                "target": target,
                "relative_l2": 1.0,
                "gradient_relative_l2": 1.0,
            }
        )
    baseline_path = tmp_path / "baseline.csv"
    _write(baseline_path, baseline)
    summary, acceptance = module["build_summary"](run_root=tmp_path / "runs", baseline_csv=baseline_path)
    assert len(summary) == 4
    assert len(acceptance) == len(module["VARIANTS"])
    assert all(row["complete"] is False for row in acceptance)


def test_write_csv_accepts_sparse_row_schema(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="spatial_learning_summary_sparse_test")
    path = tmp_path / "summary.csv"

    module["_write_csv"](path, [{"variant": "missing"}, {"variant": "done", "score": 1.0}])

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"variant": "missing", "score": ""},
        {"variant": "done", "score": "1.0"},
    ]
