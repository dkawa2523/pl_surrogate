from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.train import model_dispatch
from plasma_surrogate.train.model_adapters import MODEL_ADAPTERS, ModelAdapter
from plasma_surrogate.train.model_dispatch import TrainDispatchContext, TrainDispatchResult, run_model_train_predict
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
    vars_eff = list(y_vars or ["density", "temperature", "potential", "flux"])
    y = np.abs(rng.normal(size=(n, len(vars_eff), h, w))).astype(np.float32)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coord_feature_pack = {
        "channels": ["x", "y", "distance_signed", "distance_any", "mask_plasma"],
        "data": np.stack(
            [
                xx,
                yy,
                np.zeros((h, w), dtype=np.float32),
                np.ones((h, w), dtype=np.float32),
                np.ones((h, w), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32),
    }
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
        coord_feature_pack=coord_feature_pack,
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
                "target_vars": ["temperature", "potential"],
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
    weights = {key: 1.0 / len(ctx.y_vars) for key in ctx.y_vars}
    ctx.run_cfg = {
        "train": {
            "unet": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": list(ctx.y_vars),
                "model_cfg": {"backend": "numpy", "output_heads": {"mode": "shared"}},
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": weights,
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
    assert set(out.metrics.keys()) == set(ctx.y_vars)
    assert out.extra_artifacts.get("model_contracts", {}).get("unet", {}).get("target_family_effective") == "allvars"
    assert (
        out.extra_artifacts.get("model_contracts", {})
        .get("unet", {})
        .get("feature", {})
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
    contract = out.extra_artifacts.get("model_contracts", {}).get("unet", {})
    assert contract.get("selection_weights_effective") == {"density": 0.5, "temperature": 0.5}


def test_deeponet_requires_cond_only_branch_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "deeponet_plasma")
    weights = {key: 1.0 / len(ctx.y_vars) for key in ctx.y_vars}
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": list(ctx.y_vars),
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": weights,
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


def test_train_dispatch_enters_adapter_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path, "global_mlp")
    captured: dict[str, Any] = {}

    def _fake_run(self: ModelAdapter, call_ctx: TrainDispatchContext) -> TrainDispatchResult:
        captured["adapter"] = self.name
        captured["ctx"] = call_ctx
        empty = np.zeros((0,), dtype=np.float32)
        return TrainDispatchResult(
            model=object(),
            history=[],
            pred_eval={},
            true_eval={},
            metrics={},
            r2_scores={},
            extra_artifacts={"marker": empty},
        )

    monkeypatch.setattr(ModelAdapter, "run", _fake_run)

    out = run_model_train_predict(ctx)

    assert captured["adapter"] == "global_mlp"
    assert captured["ctx"] is ctx
    assert "marker" in out.extra_artifacts


@pytest.mark.parametrize("adapter", MODEL_ADAPTERS)
def test_each_train_adapter_routes_to_lane_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter: ModelAdapter,
) -> None:
    ctx = _ctx(tmp_path, adapter.model_names[0])
    captured: dict[str, Any] = {}

    def _fake_runner(
        call_ctx: TrainDispatchContext,
        *,
        adapter: ModelAdapter,
    ) -> TrainDispatchResult:
        captured["ctx"] = call_ctx
        captured["adapter"] = adapter.name
        return TrainDispatchResult(
            model=object(),
            history=[],
            pred_eval={},
            true_eval={},
            metrics={},
            r2_scores={},
            extra_artifacts={"model_contracts": {adapter.name: {"lane": adapter.name}}},
        )

    monkeypatch.setattr(model_dispatch, adapter.runner_name, _fake_runner)

    out = model_dispatch.run_model_train_predict(ctx)

    assert captured == {"ctx": ctx, "adapter": adapter.name}
    assert out.extra_artifacts["model_contracts"][adapter.name]["lane"] == adapter.name


def test_deeponet_mainline_runs_with_plain_cond_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path, "deeponet_plasma")
    weights = {key: 1.0 / len(ctx.y_vars) for key in ctx.y_vars}
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": list(ctx.y_vars),
                "selection": {
                    "mode": "best_val_allvars_balance",
                    "weights": weights,
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
    assert set(out.metrics.keys()) == set(ctx.y_vars)
    contract = out.extra_artifacts.get("model_contracts", {}).get("deeponet", {})
    assert contract.get("target_family_effective") == "allvars"


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


def test_train_dispatch_resolves_loss_protocol_before_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    ctx = _ctx(tmp_path, "global_mlp", y_vars=["electron_density", "plasma_potential"])
    ctx.input_mode_effective = "table_only"
    ctx.structure_adapter_mode_effective = "none"
    ctx.physics_cfg = {
        "enabled": False,
        "boundary_operator": {"enabled": False},
        "target_role_schema": {
            "targets": [
                {
                    "id": "electron_density",
                    "role": "density_electron",
                    "positive": True,
                    "field_family": "density",
                },
                {
                    "id": "plasma_potential",
                    "role": "potential",
                    "positive": False,
                    "field_family": "electrostatic",
                },
            ]
        },
    }
    ctx.run_cfg = {
        "train": {
            "loss": {"protocol": "plasma_surrogate_v2"},
            "global_mlp": {
                "epochs": 1,
                "lr": 1e-3,
                "model_cfg": {"backend": "numpy"},
            }
        }
    }

    def _fake_run_global(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        captured["loss_cfg"] = kwargs["loss_cfg"]
        captured["supervised_mask"] = kwargs["supervised_mask"]
        return TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    monkeypatch.setattr(Trainer, "run_global", _fake_run_global)

    out = run_model_train_predict(ctx)

    loss_cfg = captured["loss_cfg"]
    assert loss_cfg["protocol_effective"] == "plasma_surrogate_v2"
    assert loss_cfg["supervised"] == {"type": "mse", "mask": "plasma_only"}
    assert captured["supervised_mask"] is None
    assert out.extra_artifacts["loss_protocol_effective"] == "plasma_surrogate_v2"
