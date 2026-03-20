from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner


def test_benchmark_runner_smoke(tmp_path: Path):
    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 0},
            "profile": "m7_global_frozen_ref",
            "seed": 7,
            "split": {"seed": 9, "ratios": [0.6, 0.2, 0.2]},
            "preprocessing": {
                "scalers": {
                    "target_transforms": {
                        "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    }
                }
            },
        }
    }
    cfg_path = tmp_path / "bench.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    runner = BenchmarkRunner.from_yaml(cfg_path)
    result = runner.run()
    assert result.leaderboard_path.exists()
    with (tmp_path / "bench" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == "m7_global_frozen_ref"
    assert resolved["split"]["seed"] == 9
    assert resolved["split"]["ratios"] == [0.6, 0.2, 0.2]
    assert "artifact_hashes" in resolved
    assert "feature_hash" in resolved["artifact_hashes"]
    assert "sampling_hash" in resolved["artifact_hashes"]
