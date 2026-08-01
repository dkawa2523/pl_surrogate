from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "summarize_gec_ccp_trustworthy.py"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_summarize_gec_ccp_trustworthy_outputs_required_tables(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "gec_ccp_trustworthy_v1"
    out_dir = tmp_path / "reports" / "gec_ccp_trustworthy_v1"
    header = [
        "model_id",
        "input_mode_effective",
        "surrogate_quality_score",
        "primary_metric",
        "primary_metric_value",
        "primary_metric_reliable",
        "test_r2_plasma_mean_dual",
        "test_r2_plasma_mean_interp",
        "test_r2_plasma_mean_extrap",
        "score_nrmse_component",
        "score_boundary_component",
        "score_continuity_component",
        "score_physics_component",
        "score_sign_component",
        "sdf_boundary_to_deep_rmse_ratio_mean",
        "positive_target_negative_ratio_penalty",
    ]
    _write_csv(
        run_root / "n27" / "global_mlp" / "leaderboard.csv",
        header,
        [
            [
                "global_mlp",
                "table_only",
                0.42,
                "surrogate_quality_score",
                0.42,
                "True",
                0.81,
                0.86,
                0.76,
                0.30,
                0.04,
                0.02,
                0.01,
                0.00,
                1.25,
                0.00,
            ]
        ],
    )
    _write_csv(
        run_root / "n27" / "coord_mlp_pod_residual" / "leaderboard.csv",
        header,
        [
            [
                "coord_mlp_pod_residual",
                "table_plus_structure",
                0.31,
                "surrogate_quality_score",
                0.31,
                "True",
                0.88,
                0.90,
                0.84,
                0.21,
                0.03,
                0.02,
                0.01,
                0.00,
                1.10,
                0.00,
            ]
        ],
    )
    _write_csv(
        run_root / "run_status.csv",
        ["dataset_size", "model_id", "status", "log"],
        [[27, "fno", "failed_runtime", "runs/gec_ccp_trustworthy_v1/logs/n27_fno.log"]],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--sizes",
            "27",
            "--models",
            "global_mlp",
            "coord_mlp_pod_residual",
            "fno",
        ],
        cwd=str(ROOT),
        check=True,
    )

    summary_csv = out_dir / "summary_trustworthy_27_54_78.csv"
    core_csv = out_dir / "summary_trustworthy_core_27_54_78.csv"
    report_md = out_dir / "report_trustworthy_27_54_78.md"
    assert summary_csv.exists()
    assert core_csv.exists()
    assert report_md.exists()

    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert set(rows[0]) >= {
        "dataset_size",
        "model_id",
        "input_mode_effective",
        "surrogate_quality_score",
        "test_r2_plasma_mean_dual",
        "test_r2_plasma_mean_interp",
        "test_r2_plasma_mean_extrap",
        "score_nrmse_component",
        "sdf_boundary_to_deep_rmse_ratio_mean",
        "positive_target_negative_ratio_penalty",
        "leaderboard_csv",
    }

    by_model = {row["model_id"]: row for row in rows}
    assert by_model["global_mlp"]["status"] == "leaderboard_found"
    assert by_model["global_mlp"]["failure_mode"] != "leaderboard_found"
    assert by_model["coord_mlp_pod_residual"]["surrogate_quality_score"] == "0.31"
    assert by_model["fno"]["status"] == "failed_runtime"
    assert by_model["fno"]["failure_mode"] == "failed_runtime"

    report = report_md.read_text(encoding="utf-8")
    assert "## Main Reliability Ranking" in report
    assert "## Auxiliary Plasma R2 Ranking" in report
    assert "## Data Size Dependence" in report
    assert "failed_runtime" in report


def test_summarize_gec_ccp_trustworthy_default_output_stays_under_run_root() -> None:
    from scripts.summarize_gec_ccp_trustworthy import _parse_args

    old_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT)]
        args = _parse_args()
    finally:
        sys.argv = old_argv

    assert args.out_dir == "runs/gec_ccp_trustworthy_v1/summary"
