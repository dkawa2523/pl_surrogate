from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "compare_selected_models.py"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_compare_selected_models_supports_auto_primary_and_dynamic_headers(tmp_path: Path) -> None:
    global_csv = tmp_path / "global" / "leaderboard.csv"
    ffno_csv = tmp_path / "ffno" / "leaderboard.csv"
    header = [
        "model_id",
        "input_mode_effective",
        "primary_metric",
        "primary_metric_value",
        "target_family_effective",
        "test_rmse_ne",
        "test_rmse_Te",
        "score_total_dual",
        "test_r2_plasma_mean_dual",
    ]
    _write_csv(
        global_csv,
        header,
        [
            ["global_mlp", "table_plus_structure", "score_total_dual", 0.42, "allvars", 0.11, 0.22, 0.42, 0.0],
        ],
    )
    _write_csv(
        ffno_csv,
        header,
        [
            ["ffno", "table_plus_structure", "test_r2_plasma_mean_dual", 0.71, "allvars", 0.09, 0.18, 0.0, 0.71],
        ],
    )
    cfg = {
        "compare": {
            "output_dir": str(tmp_path / "compare"),
            "global_reference_mode": "frozen",
            "objective_metric": "auto_primary",
            "objective_mode": "max",
            "rows": [
                {"name": "global", "model_id": "global_mlp", "leaderboard_csv": str(global_csv)},
                {"name": "ffno", "model_id": "ffno", "leaderboard_csv": str(ffno_csv)},
            ],
        }
    }
    cfg_path = tmp_path / "compare.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg_path)],
        cwd=str(ROOT),
        check=True,
    )

    out_csv = tmp_path / "compare" / "selected_models_comparison.csv"
    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["model_id"] for row in rows] == ["ffno", "global_mlp"]
    assert "test_rmse_ne" in rows[0]
    assert "test_rmse_Te" in rows[0]
    assert rows[0]["primary_metric"] == "test_r2_plasma_mean_dual"
    assert rows[1]["reference_type"] == "frozen"


def test_compare_selected_models_frozen_requires_global_row(tmp_path: Path) -> None:
    ffno_csv = tmp_path / "ffno" / "leaderboard.csv"
    _write_csv(
        ffno_csv,
        ["model_id", "input_mode_effective", "primary_metric", "primary_metric_value"],
        [["ffno", "table_only", "score_total_dual", 0.5]],
    )
    cfg = {
        "compare": {
            "output_dir": str(tmp_path / "compare"),
            "global_reference_mode": "frozen",
            "objective_metric": "auto_primary",
            "rows": [
                {"name": "ffno", "model_id": "ffno", "leaderboard_csv": str(ffno_csv)},
            ],
        }
    }
    cfg_path = tmp_path / "compare_invalid.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "global_reference_mode=frozen requires at least one row with model_id=global_mlp" in proc.stderr
