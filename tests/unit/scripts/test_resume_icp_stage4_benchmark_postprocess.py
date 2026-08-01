from __future__ import annotations

import csv
import runpy
from pathlib import Path


SCRIPT = Path("experiments/icp_stage4/scripts/resume_icp_stage4_benchmark_postprocess.py")


def test_read_history_preserves_contract_strings(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="resume_icp_postprocess_test")
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "val_loss", "selection_mode_effective"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 3,
                "val_loss": 0.25,
                "selection_mode_effective": "best_val_spatial_objective",
            }
        )

    rows = module["read_history"](path)

    assert rows == [
        {
            "epoch": 3.0,
            "val_loss": 0.25,
            "selection_mode_effective": "best_val_spatial_objective",
        }
    ]
