from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.train.trainer import TrainOutput, Trainer


class _IdentityTransforms:
    def __init__(self) -> None:
        self.target_transforms: dict[str, dict[str, Any]] = {}
        self.y_scalers: dict[str, Any] = {}

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}



def _valid_cfg(target_vars: list[str]) -> dict[str, Any]:
    return {
        "epochs": 1,
        "lr": 1e-3,
        "target_family": "allvars",
        "target_vars": list(target_vars),
        "model_cfg": {
            "hidden_dim": 16,
            "latent_dim": 12,
            "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True},
        },
        "selection": {"mode": "best_val_allvars_balance"},
    }


def _ctx(
    tmp_path: Path,
    *,
    model_name: str = "deeponet_pod",
    input_mode: str,
    adapter_mode: str,
    descriptor_profile: str,
    latent_profile: str,
    descriptor_dim: int = 0,
) -> TrainDispatchContext:
    n, d, h, w = 9, 3, 4, 4
    rng = np.random.default_rng(7)
    cond = rng.normal(size=(n, d)).astype(np.float32)
    y_vars = ["density", "temperature"]
    y = rng.normal(size=(n, len(y_vars), h, w)).astype(np.float32)
    descriptor_pack = None
    if descriptor_dim > 0:
        descriptor_pack = {
            "vector": np.linspace(0.0, 1.0, descriptor_dim, dtype=np.float32),
            "feature_names": np.asarray([f"f{i}" for i in range(descriptor_dim)], dtype=object),
        }
    return TrainDispatchContext(
        run_cfg={
            "train": {
                model_name: _valid_cfg(y_vars),
                "unet_like": {"batch_size_cases": 4},
            }
        },
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name=model_name,
        model_dir=tmp_path / f"{model_name}_dispatch",
        global_seed=0,
        n_cases=n,
        h=h,
        w=w,
        y_vars=y_vars,
        cond_scaled=cond,
        y=y,
        y_scaled=y,
        tr=np.arange(0, 5, dtype=np.int64),
        va=np.arange(5, 7, dtype=np.int64),
        te=np.arange(7, 9, dtype=np.int64),
        transforms=_IdentityTransforms(),
        physics_cfg={"enabled": False},
        geom_ctx=None,
        deeponet_index={},
        deeponet_index_meta={},
        deeponet_poisson_index={},
        deeponet_poisson_meta={},
        deeponet_boundary_index={},
        deeponet_boundary_meta={},
        input_mode_effective=input_mode,
        structure_adapter_mode_effective=adapter_mode,
        structure_descriptor_profile_effective=descriptor_profile,
        structure_latent_profile_effective=latent_profile,
        structure_descriptor_pack=descriptor_pack,
    )


def test_table_only_deeponet_pod_runs_cond_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    require_torch_runtime()
    ctx = _ctx(
        tmp_path,
        input_mode="table_only",
        adapter_mode="none",
        descriptor_profile="none",
        latent_profile="none",
    )
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["cond_dim"] = int(np.asarray(cond_train).shape[1])
        return TrainOutput(history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model)

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    assert captured["cond_dim"] == int(ctx.cond_scaled.shape[1])
    assert int(out.extra_artifacts["deeponet_pod_descriptor_dim_effective"]) == 0
    assert str(out.extra_artifacts["deeponet_pod_descriptor_profile_effective"]) == "none"


def test_table_plus_descriptor_branch_expands_cond_dim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    require_torch_runtime()
    descriptor_dim = 5
    ctx = _ctx(
        tmp_path,
        input_mode="table_plus_structure",
        adapter_mode="descriptor_branch",
        descriptor_profile="struct_desc_v1",
        latent_profile="none",
        descriptor_dim=descriptor_dim,
    )
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["cond_dim"] = int(np.asarray(cond_train).shape[1])
        return TrainOutput(history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model)

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    assert captured["cond_dim"] == int(ctx.cond_scaled.shape[1] + descriptor_dim)
    assert int(out.extra_artifacts["deeponet_pod_descriptor_dim_effective"]) == descriptor_dim
    assert str(out.extra_artifacts["deeponet_pod_descriptor_profile_effective"]) == "struct_desc_v1"


def test_geom_deeponet_pod_uses_descriptor_pod_lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    require_torch_runtime()
    descriptor_dim = 5
    ctx = _ctx(
        tmp_path,
        model_name="geom_deeponet_pod",
        input_mode="table_plus_structure",
        adapter_mode="hybrid_pack_descriptor",
        descriptor_profile="struct_desc_v1",
        latent_profile="none",
        descriptor_dim=descriptor_dim,
    )
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["cond_dim"] = int(np.asarray(cond_train).shape[1])
        captured["model_type"] = str(getattr(model, "model_type", ""))
        return TrainOutput(history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model)

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    contract = out.extra_artifacts["deeponet_pod_contract_effective"]
    assert captured["cond_dim"] == int(ctx.cond_scaled.shape[1] + descriptor_dim)
    assert captured["model_type"] == "geom_deeponet_pod"
    assert contract["model_type_effective"] == "geom_deeponet_pod"
    assert int(contract["deeponet_pod_descriptor_dim_effective"]) == descriptor_dim


def test_table_plus_descriptor_profile_requires_descriptor_adapter(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        input_mode="table_plus_structure",
        adapter_mode="none",
        descriptor_profile="struct_desc_v1",
        latent_profile="none",
        descriptor_dim=4,
    )
    with pytest.raises(ValueError, match="descriptor profile requires adapter_mode_effective"):
        run_model_train_predict(ctx)
