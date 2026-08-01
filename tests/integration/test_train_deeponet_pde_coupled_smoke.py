from __future__ import annotations

import pytest
import json
from pathlib import Path

import yaml

from plasma_surrogate.cli.main import main
from tests._config_presets import (
    default_target_transforms_four_field_example,
    runtime_table_plus_structure,
)
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def test_train_deeponet_pde_coupled_smoke(tmp_path: Path):
    require_torch_runtime(enable_backend=False, refresh=False)
    run_dir = tmp_path / "deeponet_pde_train"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_plus_structure(feature_profile="geom_v1_mainline", adapter_mode="auto"),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 9},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {
                "target_transforms": default_target_transforms_four_field_example(),
            },
            "coord_features": {
                "enabled": True,
                "channels_from_profile": "geom_v1_mainline",
            },
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
                "terms": {
                    "poisson": {"weight": 0.02},
                    "boundary_operator": {"weight": 0.01},
                },
                "boundary_operator": {
                    "enabled": True,
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
    assert meta["model_type"] == "deeponet_plasma"
    assert (run_dir / "train" / "trainable_params_stage1.json").exists()
    assert (run_dir / "train" / "trainable_params_stage2.json").exists()
    assert (run_dir / "train" / "scalars" / "physics_terms.csv").exists()
    assert (run_dir / "train" / "scalars" / "optimization_diagnostics.csv").exists()
