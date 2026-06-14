from __future__ import annotations

import pytest
from pathlib import Path


import yaml

from plasma_surrogate.cli.main import main
from tests._runtime_requirements import require_torch_runtime
from tests._config_presets import (
    default_target_transforms_four_field_example,
    runtime_table_plus_structure,
)

pytestmark = pytest.mark.torch_runtime


@pytest.mark.parametrize(
    ("model_name", "extra_cfg"),
    [
        ("fno", {"fno_n_modes": 2}),
    ],
)
def test_cli_fno_hybrid_train_and_infer_smoke(tmp_path: Path, model_name: str, extra_cfg: dict[str, float]):
    require_torch_runtime()
    run_dir = tmp_path / f"{model_name}_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_plus_structure(feature_profile="geom_v1_mainline", adapter_mode="grid_pack"),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 3},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        "model": {"name": model_name, "phi_mode": "poisson_hybrid", "phi_hybrid_steps": 1, **extra_cfg},
        "train": {
            "epochs": 3,
            "lr": 0.02,
            "physics": {
                "enabled": True,
                "terms": {"poisson": {"weight": 0.02}, "boundary": {"weight": 0.01}},
            },
            "fno": {
                "selection": {"mode": "best_val_allvars_balance"},
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
            },
        },
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / f"{model_name}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
