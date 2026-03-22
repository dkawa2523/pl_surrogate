from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.cli.main import main
from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.models.deeponet.pod_deeponet_torch import POD_DEEPONET_IMPL_VERSION
from plasma_surrogate.models.mlp.io import load_mlp_checkpoint


def test_infer_pod_deeponet_smoke(tmp_path: Path) -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    run_dir = tmp_path / "pod_deeponet_run"
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
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / "infer_pod_deeponet.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    loaded = load_mlp_checkpoint(run_dir / "checkpoints")
    meta = loaded.to_meta()
    assert meta["model_type"] == "deeponet_pod"
    assert meta["basis_rank_by_var"]
    assert main(["infer", "--config", str(cfg_path)]) == 0

    with (run_dir / "checkpoints" / "meta.json").open("r", encoding="utf-8") as f:
        ckpt_meta = json.load(f)
    assert ckpt_meta["model_type"] == "deeponet_pod"
    assert ckpt_meta["impl_version"] == POD_DEEPONET_IMPL_VERSION
    assert ckpt_meta["basis_keys"] == ["ne", "ni", "Te", "phi"]
    assert "coeff_std_by_var" in ckpt_meta
    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
