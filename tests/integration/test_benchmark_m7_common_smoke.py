from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.core.torch_backend import torch_runtime_available


def test_benchmark_runner_m7_fno_isolated_smoke(tmp_path: Path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")

    cfg = {
            "benchmark": {
                "output_dir": str(tmp_path / "bench_m7"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_fno_isolated",
            "phi_mode": "direct",
            "split": {"seed": 7, "ratios": [0.7, 0.15, 0.15]},
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
            "eval_protocol": {"mode": "dual_axis", "scope": "fno_isolated", "primary_split": "interp"},
            "guardrails": {
                "enabled": True,
                "mode": "warn",
                "checks": {
                    "effective_steps_floor": True,
                },
            },
            "effective_steps_floor": {
                "fno": 2,
            },
            "train": {
                "optimizer_contract": {
                    "grad_clip": {"mode": "global", "norm": 0.5, "adaptive_by_dim": False},
                    "diagnostics": {"enabled": True},
                },
                "fno": {
                    "epochs": 1,
                    "lr": 0.01,
                    "target_family": "allvars",
                    "target_vars": ["ne", "ni", "Te", "phi"],
                    "selection": {
                        "mode": "best_val_allvars_balance",
                        "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                    },
                    "input_features": {
                        "mode": "geom_feature_pack",
                        "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    },
                    "model_cfg": {"backend": "torch"},
                },
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_m7.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    assert result.leaderboard_path.exists()
    with result.leaderboard_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["model_id"] for r in rows}
    assert {"fno"}.issubset(ids)

    with (tmp_path / "bench_m7" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == "m7_fno_isolated"
    assert "optimizer_contract" in resolved
    assert "resolved_train_per_model" in resolved
    assert set(resolved["resolved_train_per_model"].keys()) >= {
        "fno",
    }
    assert "effective_steps_per_model" in resolved
    assert set(resolved["effective_steps_per_model"].keys()) >= {
        "fno",
    }
    assert "guardrail_warnings" in resolved
    assert isinstance(resolved["guardrail_warnings"], list)
    assert any("effective_steps_floor" in w for w in resolved["guardrail_warnings"])
    assert "interp_overlap_status" in resolved
    assert "fno_contract_effective" in resolved
    assert "input_features_mode" in resolved["fno_contract_effective"]
    assert "input_feature_channels" in resolved["fno_contract_effective"]
    assert "target_vars_for_score_effective" in resolved["eval"]
    assert "target_family_for_score_effective" in resolved["eval"]
    assert "primary_metric_effective" in resolved["eval"]
    spatial_summary_interp = tmp_path / "bench_m7" / "models" / "fno" / "eval_protocol" / "interp" / "eval" / "spatial_error_summary.csv"
    spatial_by_case_interp = tmp_path / "bench_m7" / "models" / "fno" / "eval_protocol" / "interp" / "eval" / "spatial_error_by_case.csv"
    assert spatial_summary_interp.exists()
    assert spatial_by_case_interp.exists()


def test_benchmark_runner_scope_invalid_raises(tmp_path: Path):
    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench_scope_mismatch"),
            "dataset": {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_global_frozen_ref",
            "phi_mode": "direct",
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
            "eval_protocol": {"mode": "dual_axis", "scope": "coord_isolated"},
            "train": {"global_mlp": {"epochs": 1}},
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_scope_mismatch.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="scope must be one of"):
        BenchmarkRunner.from_yaml(cfg_path).run()


def test_benchmark_runner_scope_unet_profile_mismatch_raises(tmp_path: Path):
    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench_scope_mismatch_unet"),
            "dataset": {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_global_frozen_ref",
            "phi_mode": "direct",
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
            "eval_protocol": {"mode": "dual_axis", "scope": "unet_isolated"},
            "train": {"global_mlp": {"epochs": 1}},
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_scope_mismatch_unet.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="scope=unet_isolated"):
        BenchmarkRunner.from_yaml(cfg_path).run()


def test_benchmark_runner_unet_target_family_score_family_mismatch_raises(tmp_path: Path):
    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench_unet_family_mismatch"),
            "dataset": {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_unet_isolated",
            "phi_mode": "direct",
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
            "eval": {
                "target_family_for_score": "field",
                "target_vars_for_score": ["Te", "phi"],
            },
            "eval_protocol": {"mode": "dual_axis", "scope": "unet_isolated"},
            "train": {
                "unet": {
                    "epochs": 1,
                    "target_family": "allvars",
                    "model_cfg": {"backend": "numpy"},
                }
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_unet_family_mismatch.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="train.unet.target_family must match"):
        BenchmarkRunner.from_yaml(cfg_path).run()
