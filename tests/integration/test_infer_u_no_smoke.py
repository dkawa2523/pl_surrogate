from __future__ import annotations

import pytest
from pathlib import Path


import yaml

from plasma_surrogate.cli.main import main
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.models.uno.simple_uno import UNOBaseline

pytestmark = pytest.mark.torch_runtime


def _u_no_dual_mode_cfg(*, run_dir: Path) -> dict:
    return {
        "run_dir": str(run_dir),
        "runtime": {
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "auto",
                "provider_mode": "fixed",
            },
        },
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 9},
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
            "name": "u_no",
            "phi_mode": "direct",
            "backend": "torch",
            "n_modes": 3,
            "uno_cfg": {"width": 16, "n_layers": 2, "dropout": 0.0},
        },
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {"enabled": False},
            "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
            "u_no": {
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
                    "uno_cfg": {"width": 16, "n_layers": 2, "dropout": 0.0},
                },
            },
        },
        "inference": {
            "single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}},
            "qoi": {"uniformity": {"target": "ne"}},
        },
    }


def test_infer_u_no_smoke(tmp_path: Path) -> None:
    require_torch_runtime()

    run_dir = tmp_path / "u_no_run"
    cfg = _u_no_dual_mode_cfg(run_dir=run_dir)
    cfg_path = tmp_path / "u_no.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()


def test_train_checkpoint_load_and_infer_u_no_smoke(tmp_path: Path) -> None:
    require_torch_runtime()

    run_dir = tmp_path / "u_no_run_ckpt"
    cfg = _u_no_dual_mode_cfg(run_dir=run_dir)
    cfg_path = tmp_path / "u_no_ckpt.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    ckpt_dir = run_dir / "checkpoints"
    loaded = load_checkpoint(ckpt_dir)
    assert isinstance(loaded, UNOBaseline)
    assert main(["infer", "--config", str(cfg_path)]) == 0
