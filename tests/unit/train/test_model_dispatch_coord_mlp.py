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

    def inverse_fields(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)



def _ctx(tmp_path: Path, y_vars: list[str] | None = None) -> TrainDispatchContext:
    n, d, h, w = 8, 3, 4, 4
    rng = np.random.default_rng(24)
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
        model_name="coord_mlp_fourier",
        model_dir=tmp_path / "dispatch_coord_mlp",
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
        input_mode_effective="table_plus_structure",
        structure_adapter_mode_effective="auto",
        coord_feature_pack={"data": coord_data, "channels": np.asarray(channels)},
        coord_feature_scaler={
            "enabled": True,
            "mode": "zscore",
            "channels": {
                "x": {"type": "none"},
                "y": {"type": "none"},
                "mask_plasma": {"type": "none"},
                "distance_signed": {"type": "none"},
                "distance_any": {"type": "none"},
            },
        },
        coord_distance_transform_stats={},
        config_base_dir=tmp_path,
    )


def _valid_cfg(target_vars: list[str]) -> dict[str, Any]:
    return {
        "epochs": 1,
        "lr": 1e-3,
        "target_family": "allvars",
        "target_vars": list(target_vars),
        "input_features": {
            "mode": "geom_feature_pack",
            "require_pack": "error",
            "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            "distance_transform": {"mode": "raw"},
        },
        "model_cfg": {
            "cond_hidden": [16, 16],
            "latent_dim": 12,
            "decoder_hidden": [24, 24],
            "decoder_activation": "gelu",
            "embedding": {
                "type": "fourier",
                "n_frequencies": 4,
                "include_raw": True,
                "frequency_scale": 10.0,
            },
        },
    }


def _valid_siren_cfg(target_vars: list[str]) -> dict[str, Any]:
    cfg = _valid_cfg(target_vars)
    cfg["model_cfg"] = {
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
    }
    return cfg


def _valid_pod_residual_cfg(target_vars: list[str]) -> dict[str, Any]:
    cfg = _valid_cfg(target_vars)
    cfg["selection"] = {"mode": "best_val_allvars_balance"}
    cfg["model_cfg"] = {
        "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True},
        "cond_hidden": [12, 12],
        "latent_dim": 8,
        "residual_hidden": [16, 16],
        "residual_activation": "gelu",
        "point_encoder": {
            "xy_fourier_frequencies": 2,
            "xy_frequency_scale": 10.0,
            "include_xy_raw": True,
            "include_aux_raw": True,
        },
        "coeff_loss_weight": 0.1,
        "residual_scale_init": 0.05,
    }
    return cfg


@pytest.mark.parametrize(
    ("model_name", "cfg_factory"),
    [
        ("coord_mlp_fourier", _valid_cfg),
        ("coord_mlp_siren", _valid_siren_cfg),
        ("coord_mlp_pod_residual", _valid_pod_residual_cfg),
    ],
)
def test_coord_mlp_accepts_valid_dynamic_allvars(
    model_name: str,
    cfg_factory,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    require_torch_runtime()
    custom_vars = ["density", "temperature"]
    ctx = _ctx(tmp_path, y_vars=custom_vars)
    ctx.model_name = model_name
    ctx.run_cfg = {"train": {model_name: cfg_factory(custom_vars)}}
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
    contract = out.extra_artifacts.get("coord_mlp_contract_effective", {})
    assert contract.get("target_family_effective") == "allvars"
    assert contract.get("model_type_effective") == model_name
    expected_mode = "best_val_allvars_balance" if model_name != "coord_mlp_fourier" else "last"
    assert contract.get("selection_mode_effective") == expected_mode
    if model_name == "coord_mlp_pod_residual":
        assert contract.get("pod_residual", {}).get("basis_rank_by_var")


def test_coord_mlp_rejects_non_geom_feature_pack(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["input_features"]["mode"] = "legacy_xy"
    ctx.run_cfg = {"train": {"coord_mlp_fourier": cfg}}
    with pytest.raises(ValueError, match="geom_feature_pack"):
        run_model_train_predict(ctx)


def test_coord_mlp_siren_rejects_non_geom_feature_pack(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.model_name = "coord_mlp_siren"
    cfg = _valid_siren_cfg(ctx.y_vars)
    cfg["input_features"]["mode"] = "legacy_xy"
    ctx.run_cfg = {"train": {"coord_mlp_siren": cfg}}
    with pytest.raises(ValueError, match="geom_feature_pack"):
        run_model_train_predict(ctx)


def test_coord_mlp_rejects_duplicate_feature_list(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["input_features"]["features"] = ["x", "y", "x", "distance_signed", "distance_any"]
    ctx.run_cfg = {"train": {"coord_mlp_fourier": cfg}}
    with pytest.raises(ValueError, match="must not contain duplicates"):
        run_model_train_predict(ctx)


def test_coord_mlp_rejects_target_vars_mismatch(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cfg = _valid_cfg(ctx.y_vars)
    cfg["target_vars"] = ["ne", "Te", "phi", "ni"]
    ctx.run_cfg = {"train": {"coord_mlp_fourier": cfg}}
    with pytest.raises(ValueError, match="must match target_family=allvars"):
        run_model_train_predict(ctx)


def test_coord_mlp_rejects_missing_preprocess_pack(tmp_path: Path) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path)
    ctx.coord_feature_pack = None
    cfg = _valid_cfg(ctx.y_vars)
    cfg["input_features"]["require_pack"] = "warn"
    ctx.run_cfg = {"train": {"coord_mlp_fourier": cfg}}
    with pytest.raises(ValueError, match="preprocess coord_feature_pack was not used"):
        run_model_train_predict(ctx)


def test_coord_mlp_siren_rejects_decoder_activation_noop(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.model_name = "coord_mlp_siren"
    cfg = _valid_siren_cfg(ctx.y_vars)
    cfg["model_cfg"]["decoder_activation"] = "gelu"
    ctx.run_cfg = {"train": {"coord_mlp_siren": cfg}}
    with pytest.raises(ValueError, match="decoder_activation is not used for SIREN"):
        run_model_train_predict(ctx)


def test_coord_mlp_siren_injects_training_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path)
    ctx.model_name = "coord_mlp_siren"
    cfg = _valid_siren_cfg(ctx.y_vars)
    cfg.pop("epochs", None)
    cfg.pop("lr", None)
    cfg.pop("selection", None)
    cfg.pop("optimizer", None)
    ctx.run_cfg = {"train": {"coord_mlp_siren": cfg}}
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["epochs"] = kwargs.get("epochs")
        captured["lr"] = kwargs.get("lr")
        captured["unet_optimizer_cfg"] = dict(kwargs.get("unet_optimizer_cfg", {}))
        captured["selection_cfg"] = dict(kwargs.get("selection_cfg", {}))
        return TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    _ = run_model_train_predict(ctx)
    assert captured["epochs"] == 160
    assert captured["lr"] == pytest.approx(3e-4)
    assert captured["unet_optimizer_cfg"].get("schedule") == "cosine"
    assert captured["unet_optimizer_cfg"].get("warmup_epochs") == 20
    assert captured["selection_cfg"].get("mode") == "best_val_allvars_balance"


def test_coord_mlp_pod_residual_injects_training_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path)
    ctx.model_name = "coord_mlp_pod_residual"
    cfg = _valid_pod_residual_cfg(ctx.y_vars)
    cfg.pop("epochs", None)
    cfg.pop("lr", None)
    cfg.pop("selection", None)
    cfg.pop("optimizer", None)
    ctx.run_cfg = {"train": {"coord_mlp_pod_residual": cfg}}
    captured: dict[str, Any] = {}

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["epochs"] = kwargs.get("epochs")
        captured["lr"] = kwargs.get("lr")
        captured["unet_optimizer_cfg"] = dict(kwargs.get("unet_optimizer_cfg", {}))
        captured["selection_cfg"] = dict(kwargs.get("selection_cfg", {}))
        return TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)
    assert out.extra_artifacts["coord_mlp_contract_effective"]["pod_residual"]["basis_rank_by_var"]
    assert captured["epochs"] == 100
    assert captured["lr"] == pytest.approx(6e-4)
    assert captured["unet_optimizer_cfg"].get("type") == "adamw"
    assert captured["unet_optimizer_cfg"].get("schedule") == "cosine"
    assert captured["selection_cfg"].get("mode") == "best_val_allvars_balance"


def test_coord_mlp_rejects_disabled_coord_feature_scaling(tmp_path: Path) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path)
    ctx.coord_feature_scaler = {"enabled": False, "mode": "none", "channels": {}}
    cfg = _valid_cfg(ctx.y_vars)
    ctx.run_cfg = {"train": {"coord_mlp_fourier": cfg}}
    with pytest.raises(ValueError, match="requires preprocessing.coord_features.scaling.enabled=true"):
        run_model_train_predict(ctx)
