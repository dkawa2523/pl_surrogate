from __future__ import annotations

import pytest
from pathlib import Path


import yaml

from plasma_surrogate.cli.main import main
from tests._config_presets import runtime_table_only
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def test_train_pod_deeponet_smoke(tmp_path: Path) -> None:
    require_torch_runtime()

    run_dir = tmp_path / "pod_deeponet_train"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
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
