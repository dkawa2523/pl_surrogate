from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.train import model_dispatch
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
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


def _attach_case_supervision_packs(
    ctx: TrainDispatchContext,
    *,
    include_distance: bool = True,
) -> None:
    case_ids = [f"case_{idx}" for idx in range(ctx.n_cases)]
    ctx.case_ids = case_ids
    ctx.static_spatial_feature_pack = {
        "channels": np.asarray(["x", "y"]),
        "data": np.zeros((2, ctx.h, ctx.w), dtype=np.float32),
    }
    channels = ["mask_plasma"]
    case_maps = [
        np.stack(
            [np.full((ctx.h, ctx.w), float(idx + 1), dtype=np.float32) for idx in range(ctx.n_cases)],
            axis=0,
        )
    ]
    if include_distance:
        channels.append("distance_any")
        case_maps.append(
            np.stack(
                [np.full((ctx.h, ctx.w), float(101 + idx), dtype=np.float32) for idx in range(ctx.n_cases)],
                axis=0,
            )
        )
    ctx.case_structure_feature_pack = {
        "channels": np.asarray(channels),
        "case_ids": np.asarray(case_ids),
        "data": np.stack(case_maps, axis=1).astype(np.float32),
    }


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
    with pytest.raises(ValueError, match="branch_mode must be one of"):
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


def test_deeponet_dispatch_passes_case_spatial_features_to_train_val_and_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path, "deeponet_plasma")
    h, w = ctx.h, ctx.w
    case_ids = [f"case_{i}" for i in range(ctx.n_cases)]
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    ctx.static_spatial_feature_pack = {
        "channels": np.asarray(["x", "y", "distance_signed", "distance_any"]),
        "data": np.stack(
            [xx, yy, np.zeros((h, w), dtype=np.float32), np.ones((h, w), dtype=np.float32)],
            axis=0,
        ),
    }
    case_masks = np.stack(
        [np.full((h, w), float(i), dtype=np.float32) for i in range(ctx.n_cases)],
        axis=0,
    )[:, None, ...]
    ctx.case_structure_feature_pack = {
        "channels": np.asarray(["mask_plasma"]),
        "case_ids": np.asarray(case_ids),
        "data": case_masks,
    }
    ctx.case_ids = case_ids
    weights = {key: 1.0 / len(ctx.y_vars) for key in ctx.y_vars}
    ctx.run_cfg = {
        "train": {
            "deeponet_plasma": {
                "epochs": 1,
                "lr": 1e-3,
                "target_family": "allvars",
                "target_vars": list(ctx.y_vars),
                "selection": {"mode": "best_val_allvars_balance", "weights": weights},
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
    captured: dict[str, Any] = {}

    def _fake_train(
        self: TorchTrainer,
        model: Any,
        cond_train: np.ndarray,
        y_train: np.ndarray,
        cond_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> TorchTrainOutput:
        del self, cond_train, y_train, cond_val, y_val
        train_source = kwargs["spatial_train"]
        val_source = kwargs["spatial_val"]
        captured["train"] = train_source.batch(np.arange(train_source.n_cases, dtype=np.int64))
        captured["val"] = val_source.batch(np.arange(val_source.n_cases, dtype=np.int64))
        return TorchTrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    def _fake_predict(
        self: DeepONetPlasmaOperatorTorch,
        cond_vec: np.ndarray,
        geom_ctx: Any | None = None,
        cache_key: str = "poisson_head_v1",
        spatial_features: Any | None = None,
    ) -> dict[str, np.ndarray]:
        del geom_ctx, cache_key
        captured["test"] = np.asarray(spatial_features, dtype=np.float32).copy()
        return {
            key: np.zeros((len(cond_vec), 1, h, w), dtype=np.float32)
            for key in self.output_keys
        }

    monkeypatch.setattr(TorchTrainer, "run_deeponet", _fake_train)
    monkeypatch.setattr(DeepONetPlasmaOperatorTorch, "predict_fields", _fake_predict)

    out = run_model_train_predict(ctx)

    mask_channel = 2
    np.testing.assert_allclose(captured["train"][:, 0, 0, mask_channel], ctx.tr.astype(np.float32))
    np.testing.assert_allclose(captured["val"][:, 0, 0, mask_channel], ctx.va.astype(np.float32))
    np.testing.assert_allclose(captured["test"][:, 0, 0, mask_channel], ctx.te.astype(np.float32))
    assert out.extra_artifacts["case_spatial_pack_used"] is True
    assert out.extra_artifacts["case_spatial_pack_storage"] == "compact_case_spatial_pack"


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


def test_global_dispatch_aligns_raw_case_supervision_to_train_and_val_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, "global_mlp", y_vars=["phi"])
    ctx.input_mode_effective = "table_only"
    ctx.structure_adapter_mode_effective = "none"
    _attach_case_supervision_packs(ctx)
    ctx.run_cfg = {
        "train": {
            "loss": {"supervised": {"type": "mse", "mask": "plasma_only"}},
            "global_mlp": {
                "epochs": 1,
                "model_cfg": {"hidden": [4]},
                "selection": {
                    "mode": "best_val_spatial_objective",
                    "spatial": {"boundary_weight": 0.25},
                },
            },
        }
    }
    captured: dict[str, np.ndarray] = {}

    def _fake_run_global(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        del self, cond_train, y_train, cond_val, y_val
        train_source = kwargs["supervision_train"]
        val_source = kwargs["supervision_val"]
        captured["train"] = train_source.raw_batch(
            np.arange(train_source.n_cases, dtype=np.int64),
            ["mask_plasma", "distance_any"],
        )
        captured["val"] = val_source.raw_batch(
            np.arange(val_source.n_cases, dtype=np.int64),
            ["mask_plasma", "distance_any"],
        )
        return TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    monkeypatch.setattr(Trainer, "run_global", _fake_run_global)
    out = run_model_train_predict(ctx)

    np.testing.assert_allclose(captured["train"][:, 0, 0, 0], ctx.tr.astype(np.float32) + 1.0)
    np.testing.assert_allclose(captured["train"][:, 0, 0, 1], ctx.tr.astype(np.float32) + 101.0)
    np.testing.assert_allclose(captured["val"][:, 0, 0, 0], ctx.va.astype(np.float32) + 1.0)
    np.testing.assert_allclose(captured["val"][:, 0, 0, 1], ctx.va.astype(np.float32) + 101.0)
    assert out.extra_artifacts["case_supervision_pack_used"] is True
    assert out.extra_artifacts["case_supervision_channels"] == ["mask_plasma", "distance_any"]


def test_pod_dispatch_passes_raw_case_supervision_without_model_spatial_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, "deeponet_pod", y_vars=["density", "phi"])
    ctx.input_mode_effective = "table_only"
    ctx.structure_adapter_mode_effective = "none"
    _attach_case_supervision_packs(ctx)
    ctx.run_cfg = {
        "train": {
            "loss": {"supervised": {"type": "mse", "mask": "plasma_only"}},
            "unet_like": {"batch_size_cases": 1, "shuffle_cases": False},
            "deeponet_pod": {
                "epochs": 1,
                "target_family": "allvars",
                "target_vars": list(ctx.y_vars),
                "model_cfg": {"basis": {"rank": 2, "fit_scope": "train_only", "per_var": True}},
                "selection": {
                    "mode": "best_val_spatial_objective",
                    "spatial": {"boundary_weight": 0.25},
                },
            },
        }
    }

    class _FakePodModel:
        def __init__(self) -> None:
            self.grid_shape = (ctx.h, ctx.w)
            self.output_keys = list(ctx.y_vars)
            self.out_channels = len(ctx.y_vars)

        def forward_features(self, cond: np.ndarray) -> dict[str, np.ndarray]:
            return {
                name: np.zeros((int(cond.shape[0]), 1, ctx.h, ctx.w), dtype=np.float32)
                for name in self.output_keys
            }

    fake_model = _FakePodModel()
    captured: dict[str, np.ndarray | Any] = {}

    def _fake_build_model_from_name(**kwargs):
        del kwargs
        return fake_model

    def _fake_run_unet(self, model, cond_train, y_train, cond_val, y_val, **kwargs):
        del self, cond_train, y_train, cond_val, y_val
        captured["model"] = model
        captured["spatial_train"] = kwargs.get("spatial_train")
        captured["spatial_val"] = kwargs.get("spatial_val")
        train_source = kwargs["supervision_train"]
        val_source = kwargs["supervision_val"]
        captured["train"] = train_source.raw_batch(
            np.arange(train_source.n_cases, dtype=np.int64),
            ["mask_plasma", "distance_any"],
        )
        captured["val"] = val_source.raw_batch(
            np.arange(val_source.n_cases, dtype=np.int64),
            ["mask_plasma", "distance_any"],
        )
        return TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        )

    monkeypatch.setattr(model_dispatch, "build_model_from_name", _fake_build_model_from_name)
    monkeypatch.setattr(Trainer, "run_unet", _fake_run_unet)
    out = run_model_train_predict(ctx)

    assert captured["model"] is fake_model
    assert captured["spatial_train"] is None
    assert captured["spatial_val"] is None
    train_raw = np.asarray(captured["train"], dtype=np.float32)
    val_raw = np.asarray(captured["val"], dtype=np.float32)
    np.testing.assert_allclose(train_raw[:, 0, 0, 0], ctx.tr.astype(np.float32) + 1.0)
    np.testing.assert_allclose(train_raw[:, 0, 0, 1], ctx.tr.astype(np.float32) + 101.0)
    np.testing.assert_allclose(val_raw[:, 0, 0, 0], ctx.va.astype(np.float32) + 1.0)
    np.testing.assert_allclose(val_raw[:, 0, 0, 1], ctx.va.astype(np.float32) + 101.0)
    assert out.extra_artifacts["case_supervision_channels"] == ["mask_plasma", "distance_any"]


def test_dispatch_fails_closed_when_case_geometry_misses_required_raw_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, "global_mlp", y_vars=["phi"])
    ctx.input_mode_effective = "table_only"
    ctx.structure_adapter_mode_effective = "none"
    _attach_case_supervision_packs(ctx, include_distance=False)
    ctx.run_cfg = {
        "train": {
            "loss": {"supervised": {"type": "mse", "mask": "plasma_only"}},
            "global_mlp": {
                "epochs": 1,
                "selection": {
                    "mode": "best_val_spatial_objective",
                    "spatial": {"boundary_weight": 0.25},
                },
            },
        }
    }
    monkeypatch.setattr(
        Trainer,
        "run_global",
        lambda *args, **kwargs: pytest.fail("trainer must not start with incomplete raw geometry"),
    )

    with pytest.raises(ValueError, match="missing raw supervision channels"):
        run_model_train_predict(ctx)
