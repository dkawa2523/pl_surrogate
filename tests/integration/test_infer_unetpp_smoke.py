from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.cli.main import main
from plasma_surrogate.core.torch_backend import torch_runtime_available


def test_infer_unetpp_smoke(tmp_path: Path) -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")

    run_dir = tmp_path / "unetpp_run"
    cfg = {
        "run_dir": str(run_dir),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 5},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                },
            },
            "coord_features": {
                "enabled": True,
                "channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                "distance_transform_stats": {"enabled": True, "fit_scope": "train_split", "mask_scope": "plasma_plus_band"},
            },
        },
        "model": {
            "name": "unetpp",
            "phi_mode": "direct",
            "backend": "torch",
            "conv_cfg": {"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
            "output_heads": {"mode": "shared"},
        },
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {"enabled": False},
            "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
            "unetpp": {
                "target_family": "allvars",
                "input_features": {
                    "mode": "geom_feature_pack",
                    "require_pack": "error",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {
                    "backend": "torch",
                    "conv_cfg": {"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
                    "output_heads": {"mode": "shared"},
                },
            },
        },
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / "unetpp.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
