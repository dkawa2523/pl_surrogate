from __future__ import annotations

import pytest
import json
from pathlib import Path

import numpy as np
import yaml

from plasma_surrogate.cli.workflows import run_infer, run_preprocess, run_train
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def _write_cfg(
    cfg_path: Path,
    run_dir: Path,
    *,
    input_mode: str,
    feature_profile: str,
    descriptor_profile: str,
    latent_profile: str = "none",
    adapter_mode: str,
    provider_mode: str,
) -> None:
    cfg = {
        "run_dir": str(run_dir),
        "runtime": {
            "input_mode": input_mode,
            "strict_input_mode": "error",
            "allow_mode_fallback": False,
            "structure": {
                "feature_profile": feature_profile,
                "descriptor_profile": descriptor_profile,
                "latent_profile": latent_profile,
                "adapter_mode": adapter_mode,
                "provider_mode": provider_mode,
            },
        },
        "dataset": {"type": "synthetic", "n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 9},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "coord_features": {"enabled": False},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "deeponet_pod", "phi_mode": "direct"},
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "unet_like": {"batch_size_cases": 4},
            "deeponet_pod": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "model_cfg": {
                    "hidden_dim": 16,
                    "latent_dim": 12,
                    "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True},
                },
                "selection": {"mode": "best_val_allvars_balance"},
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


def test_deeponet_pod_table_only_smoke(tmp_path: Path) -> None:
    require_torch_runtime()
    cfg_path = tmp_path / "cfg_table_only.yaml"
    run_dir = tmp_path / "run_table_only"
    _write_cfg(
        cfg_path,
        run_dir,
        input_mode="table_only",
        feature_profile="none",
        descriptor_profile="none",
        latent_profile="none",
        adapter_mode="none",
        provider_mode="fixed",
    )
    run_preprocess(cfg_path)
    run_train(cfg_path)
    infer_out = run_infer(cfg_path)
    summary = json.loads(Path(infer_out["summary"]).read_text(encoding="utf-8"))
    ckpt_meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    assert summary["input_mode_effective"] == "table_only"
    assert summary["input_mode_fallback_applied"] is False
    assert summary["deeponet_pod_descriptor_profile_effective"] == "none"
    assert ckpt_meta["deeponet_pod_descriptor_dim_effective"] == 0
    assert ckpt_meta["deeponet_pod_descriptor_profile_effective"] == "none"
    assert ckpt_meta["deeponet_pod_latent_hook_effective"] is False


def test_deeponet_pod_table_plus_structure_descriptor_smoke(tmp_path: Path) -> None:
    require_torch_runtime()
    cfg_path = tmp_path / "cfg_table_plus_desc.yaml"
    run_dir = tmp_path / "run_table_plus_desc"
    _write_cfg(
        cfg_path,
        run_dir,
        input_mode="table_plus_structure",
        feature_profile="geom_v1_mainline",
        descriptor_profile="struct_desc_v1",
        latent_profile="none",
        adapter_mode="descriptor_branch",
        provider_mode="parametric_parts",
    )
    # Synthetic dataset writer creates mask/eps files under run_dir/dataset/geometry.
    # Inject parts-manifest artifacts expected by provider_mode=parametric_parts.
    _write_parametric_parts_artifacts(run_dir)
    run_preprocess(cfg_path)
    run_train(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["inference"]["batch"] = {
        "enabled": True,
        "cases": [
            {
                "case_id": "geom_default",
                "cond": {"c0": 0.2, "c1": 0.4, "c2": 0.6},
                "geom": {"geom_id": "default"},
            },
            {
                "case_id": "geom_shifted",
                "cond": {"c0": 0.3, "c1": 0.5, "c2": 0.7},
                "geom": {"geom_id": "default", "geom_param": {"part.p0.tx": 0.05}},
            },
        ],
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    infer_out = run_infer(cfg_path)
    summary = json.loads(Path(infer_out["summary"]).read_text(encoding="utf-8"))
    ckpt_meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    cases_summary = json.loads((run_dir / "inference" / "cases_summary.json").read_text(encoding="utf-8"))
    batch_cases = [row for row in cases_summary["cases"] if row["role"] == "batch"]
    assert summary["input_mode_effective"] == "table_plus_structure"
    assert summary["input_mode_fallback_applied"] is False
    assert summary["deeponet_pod_descriptor_profile_effective"] == "struct_desc_v1"
    assert ckpt_meta["deeponet_pod_descriptor_profile_effective"] == "struct_desc_v1"
    assert int(ckpt_meta["deeponet_pod_descriptor_dim_effective"]) > 0
    assert ckpt_meta["deeponet_pod_latent_profile_effective"] == "none"
    assert [row["case_id"] for row in batch_cases] == ["geom_default", "geom_shifted"]
    assert batch_cases[1]["geom"]["geom_param"] == {"part.p0.tx": 0.05}


def test_deeponet_pod_table_plus_structure_latent_hook_smoke(tmp_path: Path) -> None:
    require_torch_runtime()
    cfg_path = tmp_path / "cfg_table_plus_latent.yaml"
    run_dir = tmp_path / "run_table_plus_latent"
    _write_cfg(
        cfg_path,
        run_dir,
        input_mode="table_plus_structure",
        feature_profile="geom_v1_mainline",
        descriptor_profile="none",
        latent_profile="shape_ae_v1",
        adapter_mode="none",
        provider_mode="fixed",
    )
    geom_dir = run_dir / "dataset" / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        geom_dir / "latent_feature_pack.npz",
        vector=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        feature_names=np.asarray(["z0", "z1", "z2"], dtype=object),
    )
    run_preprocess(cfg_path)
    run_train(cfg_path)
    infer_out = run_infer(cfg_path)
    summary = json.loads(Path(infer_out["summary"]).read_text(encoding="utf-8"))
    ckpt_meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    assert ckpt_meta["deeponet_pod_latent_profile_effective"] == "shape_ae_v1"
    assert ckpt_meta["deeponet_pod_latent_hook_effective"] is True
    assert summary["deeponet_pod_latent_profile_effective"] == "shape_ae_v1"
    assert summary["deeponet_pod_latent_hook_effective"] is True
