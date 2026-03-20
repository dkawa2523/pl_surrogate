from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_surrogate.cli.main import main


def test_cli_time_phase_pipeline_smoke(tmp_path: Path):
    run_dir = tmp_path / "time_phase_run"
    cfg = {
        "run_dir": str(run_dir),
        "dataset": {
            "type": "synthetic",
            "n_cases": 12,
            "height": 8,
            "width": 8,
            "cond_dim": 3,
            "seed": 7,
            "axis_mode": "phase_sincos",
        },
        "preprocessing": {
            "split": {"seed": 0, "ratios": [0.7, 0.15, 0.15]},
            "axis_schema": {"mode": "phase_sincos", "harmonics": 1},
        },
        "model": {"name": "global_mlp"},
        "train": {"epochs": 3, "lr": 0.01},
        "inference": {
            "single": {
                "enabled": True,
                "cond": {"c0": 0.2, "c1": 0.5, "c2": 0.8},
                "axis": {"mode": "phase_sincos", "value": 0.25},
            }
        },
    }
    cfg_path = tmp_path / "time_phase.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    with (run_dir / "preprocessing" / "schema" / "axis_schema.json").open("r", encoding="utf-8") as f:
        axis_schema = json.load(f)
    assert axis_schema["mode"] == "phase_sincos"

    with (run_dir / "preprocessing" / "scalers" / "cond_scaler.json").open("r", encoding="utf-8") as f:
        cond_scaler = json.load(f)
    assert len(cond_scaler["mean"]) == 5  # cond(3) + phase sin/cos(2)

    with (run_dir / "preprocessing" / "sampling" / "pairs" / "phase_wrap_pairs.json").open("r", encoding="utf-8") as f:
        wrap_pairs = json.load(f)
    assert len(wrap_pairs["pairs"]) == 1

