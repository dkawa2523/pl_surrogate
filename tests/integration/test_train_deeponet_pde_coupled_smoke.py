from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.cli.main import main
from plasma_surrogate.core.torch_backend import torch_runtime_available


def test_train_deeponet_pde_coupled_smoke(tmp_path: Path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    run_dir = tmp_path / "deeponet_pde_train"
    cfg = {
        "run_dir": str(run_dir),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 9},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "sampling": {
                "deeponet": {
                    "tasks": {
                        "poisson_head": {"n_sensors": 8, "n_queries": 64, "seed": 5},
                        "boundary_operator": {"n_sensors": 12, "n_queries": 16, "seed": 6, "primary_qoi_key": "Gamma_i"},
                    }
                }
            },
        },
        "model": {"name": "deeponet_plasma", "phi_mode": "deeponet_poisson"},
        "train": {
            "epochs": 4,
            "lr": 0.01,
            "deeponet": {
                "stage1": {"epochs": 2, "lr": 0.01, "freeze_poisson_head": True, "freeze_boundary_operator": True},
                "stage2": {"epochs": 2, "lr": 0.005, "freeze_poisson_head": False, "freeze_boundary_operator": False},
            },
            "physics": {
                "enabled": True,
                "lambda_poisson": 0.02,
                "boundary_operator": {
                    "enabled": True,
                    "lambda": 0.01,
                    "mode": "operator_prior",
                    "primary_qoi_key": "Gamma_i",
                    "sample_idx_source": "deeponet_task:boundary_operator.sensor_indices",
                },
            },
        },
    }
    cfg_path = tmp_path / "train_deeponet_pde.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0

    with (run_dir / "checkpoints" / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_type"] == "deeponet_plasma_torch"
    assert (run_dir / "train" / "trainable_params_stage1.json").exists()
    assert (run_dir / "train" / "trainable_params_stage2.json").exists()
    assert (run_dir / "train" / "scalars" / "physics_terms.csv").exists()
    assert (run_dir / "train" / "scalars" / "optimization_diagnostics.csv").exists()
