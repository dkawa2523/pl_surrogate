from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.cli.main import main
from plasma_surrogate.core.torch_backend import torch_runtime_available


def test_train_pod_deeponet_smoke(tmp_path: Path) -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    run_dir = tmp_path / "pod_deeponet_train"
    cfg = {
        "run_dir": str(run_dir),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 9},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "deeponet_pod", "phi_mode": "direct"},
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
            "deeponet_pod": {
                "epochs": 2,
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
    }
    cfg_path = tmp_path / "train_pod_deeponet.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert (run_dir / "checkpoints" / "meta.json").exists()

