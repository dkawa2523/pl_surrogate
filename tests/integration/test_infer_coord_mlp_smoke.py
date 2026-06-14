from __future__ import annotations

import pytest
from pathlib import Path


import yaml

from plasma_surrogate.cli.main import main
from tests._config_presets import runtime_table_plus_structure
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.checkpoint import load_checkpoint

pytestmark = pytest.mark.torch_runtime


@pytest.mark.parametrize(
    ("model_name", "per_model_cfg"),
    [
        (
            "coord_mlp_fourier",
            {
                "target_family": "allvars",
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "model_cfg": {
                    "cond_hidden": [16, 16],
                    "latent_dim": 12,
                    "decoder_hidden": [24, 24],
                    "decoder_activation": "gelu",
                    "embedding": {"type": "fourier", "n_frequencies": 4, "include_raw": True, "frequency_scale": 10.0},
                },
            },
        ),
        (
            "coord_mlp_siren",
            {
                "target_family": "allvars",
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "model_cfg": {
                    "cond_hidden": [16, 16],
                    "latent_dim": 12,
                    "decoder_hidden": [24, 24],
                    "embedding": {"type": "none"},
                    "siren": {
                        "enabled": True,
                        "fusion": "split_add",
                        "w0_initial": 10.0,
                        "w0_hidden": 1.0,
                        "fusion_cfg": {"cond_gain_init": 0.7, "point_gain_init": 1.3, "branch_norm": True},
                    },
                },
            },
        ),
    ],
)
def test_infer_coord_mlp_smoke(tmp_path: Path, model_name: str, per_model_cfg: dict[str, object]) -> None:
    require_torch_runtime()

    run_dir = tmp_path / "coord_mlp_run"
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
                "scaling": {"enabled": True, "mode": "zscore", "fit_scope": "train_split", "mask_scope": "plasma_plus_band"},
                "distance_transform_stats": {"enabled": True, "fit_scope": "train_split", "mask_scope": "plasma_plus_band"},
            },
        },
        "model": {"name": model_name, "phi_mode": "direct"},
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {"enabled": False},
            "unet_like": {"batch_size_cases": 4, "shuffle_cases": True},
            model_name: per_model_cfg,
        },
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / f"{model_name}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    loaded = load_checkpoint(run_dir / "checkpoints")
    assert loaded.input_feature_channels == ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    assert loaded.model_type == model_name
    if model_name == "coord_mlp_siren":
        assert loaded.model_cfg["embedding"] == {"type": "none"}
        assert "decoder_activation" not in loaded.model_cfg
    else:
        assert loaded.model_cfg["embedding"]["type"] == "fourier"
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
