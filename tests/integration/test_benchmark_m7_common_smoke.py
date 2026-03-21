from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.core.torch_backend import torch_runtime_available


def test_benchmark_runner_m7_fno_isolated_smoke(tmp_path: Path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
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


def test_benchmark_runner_m7_ffno_isolated_smoke(tmp_path: Path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    cfg = {
            "benchmark": {
                "output_dir": str(tmp_path / "bench_m7_ffno"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_ffno_isolated",
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
            "eval_protocol": {"mode": "dual_axis", "scope": "ffno_isolated", "primary_split": "interp"},
            "guardrails": {
                "enabled": True,
                "mode": "warn",
                "checks": {
                    "effective_steps_floor": True,
                },
            },
            "effective_steps_floor": {
                "ffno": 2,
            },
            "train": {
                "optimizer_contract": {
                    "grad_clip": {"mode": "global", "norm": 0.5, "adaptive_by_dim": False},
                    "diagnostics": {"enabled": True},
                },
                "ffno": {
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
                    "model_cfg": {
                        "backend": "torch",
                        "n_modes": 2,
                        "spectral_cfg": {
                            "width": 16,
                            "n_layers": 2,
                            "dealias_ratio": 0.67,
                            "taper_alpha": 4.0,
                            "skip_filter": "match_spectral",
                            "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
                        },
                    },
                },
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_m7_ffno.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    assert result.leaderboard_path.exists()
    with result.leaderboard_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["model_id"] for r in rows}
    assert {"ffno"}.issubset(ids)

    with (tmp_path / "bench_m7_ffno" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == "m7_ffno_isolated"
    assert "ffno_contract_effective" in resolved
    assert "input_features_mode" in resolved["ffno_contract_effective"]
    assert "input_feature_channels" in resolved["ffno_contract_effective"]
    assert set(resolved["resolved_train_per_model"].keys()) >= {"ffno"}
    assert set(resolved["effective_steps_per_model"].keys()) >= {"ffno"}
    spatial_summary_interp = tmp_path / "bench_m7_ffno" / "models" / "ffno" / "eval_protocol" / "interp" / "eval" / "spatial_error_summary.csv"
    spatial_by_case_interp = tmp_path / "bench_m7_ffno" / "models" / "ffno" / "eval_protocol" / "interp" / "eval" / "spatial_error_by_case.csv"
    assert spatial_summary_interp.exists()
    assert spatial_by_case_interp.exists()


def test_benchmark_runner_m7_unetpp_attn_isolated_smoke(tmp_path: Path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench_m7_unetpp_attn"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_unetpp_attn_isolated",
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
            "eval_protocol": {"mode": "dual_axis", "scope": "unetpp_attn_isolated", "primary_split": "interp"},
            "train": {
                "optimizer_contract": {
                    "grad_clip": {"mode": "global", "norm": 0.5, "adaptive_by_dim": False},
                    "diagnostics": {"enabled": True},
                },
                "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
                "unetpp_attn": {
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
                    "model_cfg": {
                        "backend": "torch",
                        "conv_cfg": {
                            "base_channels": 8,
                            "depth": 2,
                            "upsample_mode": "bilinear",
                            "nested_skip": True,
                            "attention_cfg": {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"},
                        },
                        "output_heads": {"mode": "shared"},
                    },
                },
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_m7_unetpp_attn.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    assert result.leaderboard_path.exists()
    with result.leaderboard_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["model_id"] for r in rows}
    assert {"unetpp_attn"}.issubset(ids)

    with (tmp_path / "bench_m7_unetpp_attn" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == "m7_unetpp_attn_isolated"
    assert "unet_contract_effective" in resolved
    assert set(resolved["resolved_train_per_model"].keys()) >= {"unetpp_attn"}
    assert set(resolved["effective_steps_per_model"].keys()) >= {"unetpp_attn"}
    spatial_summary_interp = (
        tmp_path / "bench_m7_unetpp_attn" / "models" / "unetpp_attn" / "eval_protocol" / "interp" / "eval" / "spatial_error_summary.csv"
    )
    spatial_by_case_interp = (
        tmp_path / "bench_m7_unetpp_attn" / "models" / "unetpp_attn" / "eval_protocol" / "interp" / "eval" / "spatial_error_by_case.csv"
    )
    assert spatial_summary_interp.exists()
    assert spatial_by_case_interp.exists()


@pytest.mark.parametrize(
    ("model_name", "profile_name", "model_cfg"),
    [
        (
            "coord_mlp_fourier",
            "m7_coord_mlp_fourier_experimental",
            {
                "embedding": {"type": "fourier", "n_frequencies": 4, "include_raw": True, "frequency_scale": 10.0},
                "cond_hidden": [16, 16],
                "latent_dim": 12,
                "decoder_hidden": [24, 24],
                "decoder_activation": "gelu",
            },
        ),
        (
            "coord_mlp_siren",
            "m7_coord_mlp_siren_experimental",
            {
                "embedding": {"type": "none"},
                "cond_hidden": [16, 16],
                "latent_dim": 12,
                "decoder_hidden": [24, 24],
                "siren": {"enabled": True, "w0_initial": 30.0, "w0_hidden": 1.0},
            },
        ),
    ],
)
def test_benchmark_runner_coord_mlp_experimental_smoke(
    tmp_path: Path,
    model_name: str,
    profile_name: str,
    model_cfg: dict[str, object],
):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / f"bench_{model_name}"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": profile_name,
            "phi_mode": "direct",
            "split": {"seed": 7, "ratios": [0.7, 0.15, 0.15]},
            "preprocessing": {
                "coord_features": {
                    "enabled": True,
                    "channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
                "scalers": {
                    "target_transforms": {
                        "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    }
                },
            },
            "eval_protocol": {"mode": "dual_axis", "scope": "common", "primary_split": "interp"},
            "train": {
                "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
                model_name: {
                    "epochs": 1,
                    "lr": 0.01,
                    "target_family": "allvars",
                    "target_vars": ["ne", "ni", "Te", "phi"],
                    "input_features": {
                        "mode": "geom_feature_pack",
                        "require_pack": "error",
                        "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                        "distance_transform": {"mode": "raw"},
                    },
                    "model_cfg": model_cfg,
                },
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / f"bench_{model_name}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    assert result.leaderboard_path.exists()
    with (tmp_path / f"bench_{model_name}" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == profile_name
    assert set(resolved["resolved_train_per_model"].keys()) >= {model_name}
    assert "coord_mlp_contract_effective" in resolved
    assert resolved["coord_mlp_contract_effective"]["input_features_mode"] == "geom_feature_pack"
    assert resolved["coord_mlp_contract_effective"]["model_type_effective"] == model_name
    if model_name == "coord_mlp_siren":
        assert resolved["coord_mlp_contract_effective"]["embedding"] == {"type": "none"}
        assert resolved["coord_mlp_contract_effective"]["siren"]["enabled"] is True
    else:
        assert resolved["coord_mlp_contract_effective"]["embedding"]["type"] == "fourier"
        assert resolved["coord_mlp_contract_effective"]["siren"]["enabled"] is False


def test_benchmark_runner_deeponet_pod_experimental_smoke(tmp_path: Path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    cfg = {
        "benchmark": {
            "output_dir": str(tmp_path / "bench_deeponet_pod"),
            "dataset": {"n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 21},
            "profile": "m7_deeponet_pod_experimental",
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
            "eval_protocol": {"mode": "dual_axis", "scope": "common", "primary_split": "interp"},
            "train": {
                "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
                "deeponet_pod": {
                    "epochs": 1,
                    "lr": 0.01,
                    "target_family": "allvars",
                    "target_vars": ["ne", "ni", "Te", "phi"],
                    "selection": {"mode": "best_val_allvars_balance"},
                    "model_cfg": {
                        "hidden_dim": 16,
                        "latent_dim": 12,
                        "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True, "center": True},
                    },
                },
            },
            "physics": {"enabled": False},
        }
    }
    cfg_path = tmp_path / "bench_deeponet_pod.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    assert result.leaderboard_path.exists()
    with (tmp_path / "bench_deeponet_pod" / "resolved_benchmark.json").open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["profile"] == "m7_deeponet_pod_experimental"
    assert set(resolved["resolved_train_per_model"].keys()) >= {"deeponet_pod"}
    assert "deeponet_pod_contract_effective" in resolved
    assert resolved["deeponet_pod_contract_effective"]["basis_fit_scope_effective"] == "train_only"


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
