from __future__ import annotations

import pytest
from pathlib import Path


import yaml

from plasma_surrogate.cli.main import main
from tests._config_presets import runtime_table_plus_structure
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


@pytest.mark.parametrize("local_skip_enabled", [False, True])
def test_infer_ffno_smoke(tmp_path: Path, local_skip_enabled: bool) -> None:
    require_torch_runtime()

    run_dir = tmp_path / "ffno_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_plus_structure(feature_profile="geom_v1_mainline", adapter_mode="auto"),
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
                "channels_from_profile": "geom_v1_mainline",
                "distance_transform_stats": {"enabled": True, "fit_scope": "train_split", "mask_scope": "plasma_plus_band"},
            },
        },
        "model": {
            "name": "ffno",
            "phi_mode": "direct",
            "backend": "torch",
            "n_modes": 3,
            "spectral_cfg": {
                "width": 16,
                "n_layers": 2,
                "dealias_ratio": 0.67,
                "taper_alpha": 4.0,
                "skip_filter": "match_spectral",
                "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
                "local_skip_cfg": {"enabled": bool(local_skip_enabled), "init_scale": 0.0},
            },
        },
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {"enabled": False},
            "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
            "ffno": {
                "target_family": "allvars",
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {
                    "backend": "torch",
                    "n_modes": 3,
                    "spectral_cfg": {
                        "width": 16,
                        "n_layers": 2,
                        "dealias_ratio": 0.67,
                        "taper_alpha": 4.0,
                        "skip_filter": "match_spectral",
                        "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
                        "local_skip_cfg": {"enabled": bool(local_skip_enabled), "init_scale": 0.0},
                    },
                },
            },
        },
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / "ffno.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
