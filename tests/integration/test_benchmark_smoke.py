from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.core.input_modes import input_mode_metadata_keys
from tests._config_presets import runtime_table_only


def test_benchmark_runner_smoke(tmp_path: Path):
    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench"),
            "runtime": runtime_table_only(),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 0},
            "profile": "m7_global_frozen_ref",
            "seed": 7,
            "split": {"seed": 9, "ratios": [0.6, 0.2, 0.2]},
            "inference": {"qoi": {"uniformity": {"target": "ne"}}},
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
    with (tmp_path / "bench" / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["artifacts"]["split_config"]["seed"] == 9
    assert manifest["artifacts"]["split_config"]["ratios"] == [0.6, 0.2, 0.2]
    assert set(manifest["artifacts"]["split"]) == {"train", "val", "test"}
    assert "artifact_hashes" in manifest["artifacts"]
    assert "feature_hash" in manifest["artifacts"]["artifact_hashes"]
    assert "sampling_hash" in manifest["artifacts"]["artifact_hashes"]
    assert not (tmp_path / "bench" / "resolved_benchmark.json").exists()
    for key in input_mode_metadata_keys():
        assert key in manifest["input_mode"]
