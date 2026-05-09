from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from plasma_surrogate.cli.main import main


def test_optimize_csv_backend_smoke(tmp_path: Path):
    run_dir = tmp_path / "pipeline_csv_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": {
            "input_mode": "table_only",
            "strict_input_mode": "error",
            "allow_mode_fallback": False,
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        },
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 2, "seed": 5},
        "preprocessing": {
            "split": {"seed": 0, "ratios": [0.7, 0.15, 0.15]},
            "axis_schema": {"mode": "steady"},
            "coord_features": {"enabled": False},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "global_mlp"},
        "train": {"epochs": 3, "lr": 0.01},
        "inference": {
            "optimize": {
                "enabled": True,
                "backend": "csv",
                "n_trials": 3,
                "space": {"c0": [0.0, 1.0], "c1": [0.0, 1.0]},
                "backend_cfg": {"csv_path": "candidates.csv", "deduplicate": True},
            }
        },
    }
    cfg_path = tmp_path / "pipeline_opt_csv.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with (tmp_path / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["c0", "c1"])
        writer.writeheader()
        writer.writerow({"c0": "0.1", "c1": "0.2"})
        writer.writerow({"c0": "0.3", "c1": "0.4"})
        writer.writerow({"c0": "0.3", "c1": "0.4"})  # duplicate
        writer.writerow({"c0": "0.8", "c1": "0.9"})

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    summary_path = run_dir / "inference" / "optimize" / "summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["backend"] == "csv"
    assert summary["backend_cfg"]["deduplicate"] is True
    assert int(summary["n_trials"]) == 3

    trials_path = run_dir / "inference" / "optimize" / "trials.csv"
    with trials_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 1 <= len(rows) <= 3
