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
    n, d, h, w = 8, 3, 4, 4
    rng = np.random.default_rng(42)
    cond = rng.normal(size=(n, d)).astype(np.float32)
    vars_eff = list(y_vars or ["ne", "ni", "Te", "phi"])
    y = np.abs(rng.normal(size=(n, len(vars_eff), h, w))).astype(np.float32)
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    coord_data = np.stack(
        [
            np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1)),
            np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w)),
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.25, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return TrainDispatchContext(
        run_cfg={"train": {}},
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name="ffno",
        model_dir=tmp_path / "dispatch_ffno",
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
        te=np.arange(7, 8, dtype=np.int64),
        transforms=_IdentityTransforms(),
        physics_cfg={"enabled": False, "boundary_operator": {"enabled": False}},
        geom_ctx=None,
        deeponet_index={},
        deeponet_index_meta={},
        deeponet_poisson_index={},
        deeponet_poisson_meta={},
        deeponet_boundary_index={},
        deeponet_boundary_meta={},
        coord_feature_pack={"data": coord_data, "channels": np.asarray(channels)},
        coord_distance_transform_stats={},
        config_base_dir=tmp_path,
    )


def _valid_cfg(target_vars: list[str]) -> dict[str, Any]:
    return {
        "epochs": 1,
        "lr": 1e-3,
        "target_family": "allvars",
        "target_vars": list(target_vars),
        "selection": {"mode": "best_val_allvars_balance"},
        "input_features": {
            "mode": "geom_feature_pack",
            "require_pack": "error",
            "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            "distance_transform": {"mode": "raw"},
        },
        "model_cfg": {
            "backend": "torch",
            "n_modes": 3,
            "spectral_cfg": {
                "width": 8,
                "n_layers": 2,
                "dealias_ratio": 0.67,
                "taper_alpha": 4.0,
                "skip_filter": "match_spectral",
                "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
            },
        },
    }


def test_ffno_mainline_accepts_valid_dynamic_allvars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_torch()
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    custom_vars = ["density", "temperature"]
    ctx = _ctx(tmp_path, y_vars=custom_vars)
    ctx.run_cfg = {"train": {"ffno": _valid_cfg(custom_vars)}}
    monkeypatch.setattr(
        Trainer,
        "run_unet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == set(custom_vars)
    contract = out.extra_artifacts.get("ffno_contract_effective", {})
    assert contract.get("target_family_effective") == "allvars"
    assert contract.get("selection_weights_effective") == {"density": 0.5, "temperature": 0.5}


def test_ffno_rejects_non_geom_feature_pack(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["input_features"]["mode"] = "legacy_xy"
    ctx.run_cfg = {"train": {"ffno": cfg}}
    with pytest.raises(ValueError, match="geom_feature_pack"):
        run_model_train_predict(ctx)


def test_ffno_rejects_wrong_feature_order(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["input_features"]["features"] = ["x", "y", "distance_signed", "distance_any", "mask_plasma"]
    ctx.run_cfg = {"train": {"ffno": cfg}}
    with pytest.raises(ValueError, match="input_features.features"):
        run_model_train_predict(ctx)


def test_ffno_rejects_non_mainline_selection_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["selection"] = {"mode": "last"}
    ctx.run_cfg = {"train": {"ffno": cfg}}
    with pytest.raises(ValueError, match="selection.mode must be best_val_allvars_balance"):
        run_model_train_predict(ctx)
