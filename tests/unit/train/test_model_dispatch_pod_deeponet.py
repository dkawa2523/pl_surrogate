from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
from plasma_surrogate.train.trainer import TrainOutput, Trainer


class _IdentityTransforms:
    def __init__(self) -> None:
        self.target_transforms: dict[str, dict[str, Any]] = {}
        self.y_scalers: dict[str, Any] = {}

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}

    def inverse_fields(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)


def _enable_torch() -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"


def _ctx(tmp_path: Path, y_vars: list[str] | None = None) -> TrainDispatchContext:
    n, d, h, w = 9, 3, 4, 4
    rng = np.random.default_rng(32)
    cond = rng.normal(size=(n, d)).astype(np.float32)
    vars_eff = list(y_vars or ["density", "temperature"])
    y = rng.normal(size=(n, len(vars_eff), h, w)).astype(np.float32)
    return TrainDispatchContext(
        run_cfg={"train": {}},
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name="deeponet_pod",
        model_dir=tmp_path / "dispatch_pod_deeponet",
        global_seed=0,
        n_cases=n,
        h=h,
        w=w,
        y_vars=vars_eff,
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
        config_base_dir=tmp_path,
    )


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


def test_pod_deeponet_train_split_only_basis_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    ctx = _ctx(tmp_path)
    ctx.run_cfg = {"train": {"deeponet_pod": _valid_cfg(ctx.y_vars), "unet_like": {"batch_size_cases": 4}}}

    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["basis_rank_by_var"] = dict(getattr(model, "basis_rank_by_var", {}))
        captured["basis_mean_density"] = np.asarray(
            model.basis_bundle_numpy().mean_by_var[ctx.y_vars[0]],
            dtype=np.float32,
        ).copy()
        captured["train_targets"] = np.asarray(y_train, dtype=np.float32).copy()
        captured["selection_mode"] = str(dict(kwargs.get("selection_cfg", {})).get("mode", ""))
        captured["optimizer_schedule"] = str(dict(kwargs.get("unet_optimizer_cfg", {})).get("schedule", ""))
        return TrainOutput(history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model)

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    expected_mean = np.mean(ctx.y_scaled[ctx.tr][:, 0], axis=0, dtype=np.float32)
    assert np.allclose(captured["basis_mean_density"], expected_mean)
    assert np.allclose(captured["train_targets"], ctx.y_scaled[ctx.tr])
    assert out.extra_artifacts["deeponet_pod_contract_effective"]["basis_fit_scope_effective"] == "train_only"
    assert captured["selection_mode"] == "best_val_allvars_balance"
    assert captured["optimizer_schedule"] == "cosine"


def test_pod_deeponet_rejects_invalid_rank(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["model_cfg"]["basis"]["rank"] = 0
    ctx.run_cfg = {"train": {"deeponet_pod": cfg}}
    with pytest.raises(ValueError, match="basis.rank must be >= 1"):
        run_model_train_predict(ctx)


def test_pod_deeponet_rejects_invalid_fit_scope(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["model_cfg"]["basis"]["fit_scope"] = "all_split"
    ctx.run_cfg = {"train": {"deeponet_pod": cfg}}
    with pytest.raises(ValueError, match="basis.fit_scope must be train_only"):
        run_model_train_predict(ctx)


def test_pod_deeponet_injects_training_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg.pop("selection", None)
    cfg.pop("optimizer", None)
    ctx.run_cfg = {"train": {"deeponet_pod": cfg, "unet_like": {"batch_size_cases": 4}}}
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["selection_cfg"] = dict(kwargs.get("selection_cfg", {}))
        captured["unet_optimizer_cfg"] = dict(kwargs.get("unet_optimizer_cfg", {}))
        return TrainOutput(history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model)

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    assert captured["selection_cfg"]["mode"] == "best_val_allvars_balance"
    assert captured["unet_optimizer_cfg"]["type"] == "adamw"
    assert captured["unet_optimizer_cfg"]["schedule"] == "cosine"
    assert captured["unet_optimizer_cfg"]["warmup_epochs"] == 10
    assert out.extra_artifacts["deeponet_pod_contract_effective"]["optimizer_effective"]["schedule"] == "cosine"


def test_pod_deeponet_rejects_selection_last(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["selection"] = {"mode": "last"}
    ctx.run_cfg = {"train": {"deeponet_pod": cfg}}
    with pytest.raises(ValueError, match="selection.mode must be best_val_allvars_balance"):
        run_model_train_predict(ctx)
