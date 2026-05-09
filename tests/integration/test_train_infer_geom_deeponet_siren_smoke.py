from __future__ import annotations

import pytest
import json
from pathlib import Path

import numpy as np
import yaml

from plasma_surrogate.cli.workflows import run_infer, run_preprocess, run_train
from plasma_surrogate.core.input_modes import (
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
)
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def _write_cfg(cfg_path: Path, run_dir: Path) -> None:
    cfg = {
        "run_dir": str(run_dir),
        "runtime": {
            "input_mode": "table_plus_structure",
            "strict_input_mode": "error",
            "allow_mode_fallback": False,
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "struct_desc_v1",
                "latent_profile": "none",
                "adapter_mode": "hybrid_pack_descriptor",
                "provider_mode": "parametric_parts",
            },
        },
        "dataset": {"type": "synthetic", "n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 12},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "coord_features": {"enabled": True, "channels_from_profile": "geom_v1_mainline"},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "geom_deeponet_siren", "phi_mode": "direct"},
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "unet_like": {"batch_size_cases": 4},
            "geom_deeponet_siren": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "input_features": {
                    "mode": "geom_feature_pack",
                    "require_pack": "error",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {
                    "backend": "torch",
                    "geom_deeponet_siren_cfg": {
                        "latent_dim": 16,
                        "trunk_hidden": 24,
                        "trunk_layers": 2,
                        "branch_hidden": 32,
                        "branch_layers": 2,
                        "dropout": 0.0,
                        "trunk_w0": 20.0,
                    },
                },
            },
        },
        "inference": {
            "single": {"enabled": True, "geom": {"geom_id": "default"}},
            "batch": {"enabled": False},
            "optimize": {"enabled": False},
        },
    }
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _write_parametric_parts_artifacts(run_dir: Path) -> None:
    geom_dir = run_dir / "dataset" / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    h, w = 8, 8
    part_stack = np.zeros((2, h, w), dtype=np.float32)
    part_stack[0, 2:4, 2:4] = 1.0
    part_stack[1, 4:6, 5:7] = 1.0
    np.savez_compressed(
        geom_dir / "parts_pack.npz",
        mask_stack=part_stack,
        part_ids=np.asarray(["p0", "p1"], dtype=object),
    )
    manifest = {
        "part_ids": ["p0", "p1"],
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p1.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
        },
    }
    (geom_dir / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_train_infer_geom_deeponet_siren_smoke(tmp_path: Path) -> None:
    require_torch_runtime()
    cfg_path = tmp_path / "cfg_geom_deeponet_siren.yaml"
    run_dir = tmp_path / "run_geom_deeponet_siren"
    _write_cfg(cfg_path, run_dir)
    _write_parametric_parts_artifacts(run_dir)
    run_preprocess(cfg_path)
    run_train(cfg_path)
    infer_out = run_infer(cfg_path)
    summary = json.loads(Path(infer_out["summary"]).read_text(encoding="utf-8"))
    ckpt_meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    assert summary["input_mode_effective"] == "table_plus_structure"
    assert summary[GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY] == "struct_desc_v1"
    assert int(summary[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY]) > 0
    assert ckpt_meta[GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY] == "struct_desc_v1"
    assert int(ckpt_meta[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY]) > 0
