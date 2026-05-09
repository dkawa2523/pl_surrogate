from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.train.trainer import TrainOutput, Trainer
from plasma_surrogate.train.torch_trainer import TorchTrainOutput, TorchTrainer


class _IdentityTransforms:
    def __init__(self) -> None:
        self.target_transforms: dict[str, dict[str, Any]] = {}
        self.y_scalers: dict[str, Any] = {}

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}

    def inverse_fields(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)


def _ctx(tmp_path: Path, model_name: str, y_vars: list[str] | None = None) -> TrainDispatchContext:
    n, d, h, w = 8, 3, 4, 4
    rng = np.random.default_rng(42)
    cond = rng.normal(size=(n, d)).astype(np.float32)
    vars_eff = list(y_vars or ["ne", "ni", "Te", "phi"])
    y = np.abs(rng.normal(size=(n, len(vars_eff), h, w))).astype(np.float32)
    return TrainDispatchContext(
        run_cfg={"train": {}},
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name=model_name,
        model_dir=tmp_path / "dispatch_model",
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
        config_base_dir=tmp_path,
    )


def test_unet_rejects_non_allvars_family(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "unet")
    ctx.run_cfg = {
        "train": {
            "unet": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "field",
                "target_vars": ["Te", "phi"],
                "model_cfg": {"backend": "numpy"},
            }
        }
    }
    with pytest.raises(ValueError, match="must be allvars"):
        run_model_train_predict(ctx)


def test_unet_mainline_accepts_allvars_shared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path, "unet")
    ctx.run_cfg = {
        "train": {
            "unet": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "model_cfg": {"backend": "numpy", "output_heads": {"mode": "shared"}},
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
            }
        }
    }

    monkeypatch.setattr(
        Trainer,
        "run_unet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == {"ne", "ni", "Te", "phi"}
    assert out.extra_artifacts.get("unet_contract_effective", {}).get("target_family_effective") == "allvars"
    assert (
        out.extra_artifacts.get("unet_contract_effective", {})
        .get("unet_feature_contract_effective", {})
        .get("input_features_mode")
        == "geom_feature_pack"
    )


def test_unet_mainline_auto_uniform_selection_weights_for_dynamic_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_vars = ["density", "temperature"]
    ctx = _ctx(tmp_path, "unet", y_vars=custom_vars)
    ctx.run_cfg = {
        "train": {
            "unet": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": list(custom_vars),
                "model_cfg": {"backend": "numpy", "output_heads": {"mode": "shared"}},
                "selection": {
                    "mode": "best_val_allvars_balance",
                },
            }
        }
    }
    monkeypatch.setattr(
        Trainer,
        "run_unet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == set(custom_vars)
    contract = out.extra_artifacts.get("unet_contract_effective", {})
    assert contract.get("selection_weights_effective") == {"density": 0.5, "temperature": 0.5}


def test_fno_requires_geom_feature_pack(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "fno")
    ctx.run_cfg = {
        "train": {
            "fno": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
                "input_features": {"mode": "legacy_xy", "features": ["x", "y"]},
                "model_cfg": {"backend": "torch", "output_heads": {"mode": "shared"}},
            }
        }
    }
    with pytest.raises(ValueError, match="geom_feature_pack"):
        run_model_train_predict(ctx)


def test_deeponet_requires_cond_only_branch_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "deeponet_plasma")
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
                "operator_mode": "plain",
                "strict_mainline": True,
                "model_cfg": {"branch_mode": "moments", "trunk_input_mode": "geom_feature_pack"},
            }
        }
    }
    with pytest.raises(ValueError, match="branch_mode must be cond_only"):
        run_model_train_predict(ctx)


def test_deeponet_rejects_invalid_missing_geom_feature_policy(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "deeponet_plasma")
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
                "operator_mode": "plain",
                "strict_mainline": True,
                "model_cfg": {
                    "branch_mode": "cond_only",
                    "trunk_input_mode": "geom_feature_pack",
                    "missing_geom_feature_policy": "invalid_mode",
                },
            }
        }
    }
    with pytest.raises(ValueError, match="missing_geom_feature_policy"):
        run_model_train_predict(ctx)


def test_deeponet_rejects_invalid_output_path_dot_skip_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "deeponet_plasma")
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
                "operator_mode": "plain",
                "strict_mainline": True,
                "model_cfg": {
                    "branch_mode": "cond_only",
                    "trunk_input_mode": "geom_feature_pack",
                    "output_path": {"dot_skip_mode": "invalid"},
                },
            }
        }
    }
    with pytest.raises(ValueError, match="dot_skip_mode"):
        run_model_train_predict(ctx)


def test_deeponet_mainline_runs_with_plain_cond_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path, "deeponet_plasma")
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": ["ne", "ni", "Te", "phi"],
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
                },
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                    "distance_transform": {"mode": "raw"},
                },
                "operator_mode": "plain",
                "model_cfg": {"branch_mode": "cond_only", "trunk_input_mode": "geom_feature_pack"},
            }
        }
    }

    monkeypatch.setattr(
        TorchTrainer,
        "run_deeponet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TorchTrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}], model=model
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == {"ne", "ni", "Te", "phi"}
    contract = out.extra_artifacts.get("deeponet_contract_effective", {})
    assert contract.get("target_family_effective") == "allvars"
    assert contract.get("missing_geom_feature_policy_effective") == "warn_zero"
    assert contract.get("output_path_dot_skip_mode_effective") == "fixed"


def test_train_dispatch_result_contains_runtime_effective_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path, "global_mlp")
    ctx.input_mode_effective = "table_only"
    ctx.structure_adapter_mode_effective = "none"
    ctx.run_cfg = {
        "train": {
            "global_mlp": {
                "epochs": 1,
                "lr": 1e-3,
                "model_cfg": {"backend": "numpy"},
            }
        }
    }
    monkeypatch.setattr(
        Trainer,
        "run_global",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        ),
    )

    out = run_model_train_predict(ctx)
    assert out.extra_artifacts["input_mode_effective"] == "table_only"
    assert out.extra_artifacts["structure_adapter_mode_effective"] == "none"
