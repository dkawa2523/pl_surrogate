from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.spatial_features import (
    CaseSpatialFeatureSource,
    apply_coord_feature_scaling,
    build_case_spatial_features,
    resolve_shared_distance_transform_cfg,
)
from plasma_surrogate.train import trainer as trainer_module
from plasma_surrogate.train.trainer import Trainer, train_one_epoch_unet
from plasma_surrogate.train.torch_trainer import TorchTrainer
from tests._runtime_requirements import require_torch_runtime


class _RecordingGridModel:
    output_keys = ["phi"]
    out_channels = 1
    with_rho_eff_head = False
    backend = "numpy"

    def __init__(self, grid_shape: tuple[int, int]):
        self.grid_shape = grid_shape
        self.spatial_batches: list[np.ndarray] = []

    def forward_raw(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del cond, training
        if spatial_features is None:
            raise AssertionError("spatial_features must be passed batch-by-batch")
        self.spatial_batches.append(np.asarray(spatial_features, dtype=np.float32).copy())
        bsz = int(spatial_features.shape[0])
        h, w = self.grid_shape
        return np.zeros((bsz, 1, h, w), dtype=np.float32)

    def backward_raw(self, grad_raw: np.ndarray, **kwargs: object) -> dict[str, float]:
        del grad_raw, kwargs
        return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}


class _ZeroMultiTargetGridModel:
    output_keys = ["electron_density", "ion_density", "electron_temperature"]
    out_channels = 3
    with_rho_eff_head = False
    backend = "numpy"

    def __init__(self, grid_shape: tuple[int, int]):
        self.grid_shape = grid_shape

    def forward_raw(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del training, spatial_features
        bsz = int(cond.shape[0])
        h, w = self.grid_shape
        return np.zeros((bsz, self.out_channels, h, w), dtype=np.float32)

    def backward_raw(self, grad_raw: np.ndarray, **kwargs: object) -> dict[str, float]:
        del grad_raw, kwargs
        return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}


class _ZeroRhoAuxGridModel(_ZeroMultiTargetGridModel):
    output_keys = ["electron_density", "electron_temperature", "plasma_potential"]
    out_channels = 3
    with_rho_eff_head = True

    def forward_raw(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del training, spatial_features
        bsz = int(cond.shape[0])
        h, w = self.grid_shape
        return np.zeros((bsz, self.out_channels + 1, h, w), dtype=np.float32)


class _ModelOwnedAuxGridModel(_RecordingGridModel):
    def forward_raw(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del training, spatial_features
        return np.zeros((int(cond.shape[0]), 1, *self.grid_shape), dtype=np.float32)

    def forward(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward_raw(cond, training=training, spatial_features=spatial_features)

    def backward_raw(self, grad_raw: np.ndarray, **kwargs: object) -> dict[str, float]:
        del grad_raw
        self.backward_supervised_masks = getattr(self, "backward_supervised_masks", [])
        mask = kwargs.get("supervised_mask")
        self.backward_supervised_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float32).copy()
        )
        return {
            "step_rel_hidden_mean": 0.0,
            "step_rel_output": 0.0,
            "coeff_loss": 1.5,
            "loss_aux_coeff": 1.5,
            "loss_aux_coeff_weighted": 0.3,
            "loss_aux_total": 0.3,
        }

    def evaluate_auxiliary_losses(
        self,
        cond: np.ndarray,
        target_raw: np.ndarray,
        *,
        spatial_features: np.ndarray | None = None,
        loss_cfg: dict | None = None,
        supervised_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        del cond, target_raw, spatial_features, loss_cfg
        self.validation_supervised_masks = getattr(self, "validation_supervised_masks", [])
        self.validation_supervised_masks.append(
            None if supervised_mask is None else np.asarray(supervised_mask, dtype=np.float32).copy()
        )
        return {
            "coeff_loss": 2.0,
            "loss_aux_coeff": 2.0,
            "loss_aux_coeff_weighted": 0.4,
            "loss_aux_total": 0.4,
        }

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return {"dummy": np.zeros((1,), dtype=np.float32)}

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        assert "dummy" in state


def _raw_supervision_source(
    masks: np.ndarray,
    distances: np.ndarray,
) -> CaseSpatialFeatureSource:
    masks_arr = np.asarray(masks, dtype=np.float32)
    distances_arr = np.asarray(distances, dtype=np.float32)
    if masks_arr.shape != distances_arr.shape or masks_arr.ndim != 3:
        raise ValueError("test supervision maps must be matching [N,H,W] arrays")
    n_cases, h, w = masks_arr.shape
    return CaseSpatialFeatureSource(
        channels=("mask_plasma", "distance_any"),
        h=int(h),
        w=int(w),
        n_cases=int(n_cases),
        source="test_raw_supervision",
        static_data=np.zeros((0, h, w), dtype=np.float32),
        static_channels=(),
        case_data=np.stack([masks_arr, distances_arr], axis=1),
        case_channels=("mask_plasma", "distance_any"),
    )


def test_train_one_epoch_unet_passes_case_spatial_batches() -> None:
    h, w, c = 4, 5, 3
    model = _RecordingGridModel((h, w))
    cond = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((2, 1, h, w), dtype=np.float32)
    spatial = np.zeros((2, h, w, c), dtype=np.float32)
    spatial[0, ..., 1] = 1.0
    spatial[1, ..., 1] = 2.0

    train_one_epoch_unet(
        model,
        cond,
        y,
        loss_cfg={"supervised": {"type": "mse", "mask": "all"}},
        spatial_features=spatial,
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert len(model.spatial_batches) == 2
    assert model.spatial_batches[0].shape == (1, h, w, c)
    assert float(np.mean(model.spatial_batches[0][..., 1])) == 1.0
    assert float(np.mean(model.spatial_batches[1][..., 1])) == 2.0


def test_train_one_epoch_unet_includes_model_owned_auxiliary_loss() -> None:
    model = _ModelOwnedAuxGridModel((2, 2))
    stats = train_one_epoch_unet(
        model,
        np.zeros((1, 2), dtype=np.float32),
        np.ones((1, 1, 2, 2), dtype=np.float32),
        loss_cfg={"supervised": {"type": "mse", "mask": "all"}},
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert stats["data"] == pytest.approx(0.5)
    assert stats["aux"] == pytest.approx(0.3)
    assert stats["total"] == pytest.approx(0.8)
    assert stats["loss_aux_coeff"] == pytest.approx(1.5)
    assert stats["loss_aux_coeff_weighted"] == pytest.approx(0.3)


def test_run_unet_uses_same_auxiliary_objective_for_validation(tmp_path) -> None:
    model = _ModelOwnedAuxGridModel((2, 2))
    out = Trainer(tmp_path / "aux_validation").run_unet(
        model,
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 1, 2, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.ones((1, 1, 2, 2), dtype=np.float32),
        epochs=1,
        loss_cfg={"supervised": {"type": "mse", "mask": "all"}},
        batch_size_cases=1,
        shuffle_cases=False,
        selection_cfg={"mode": "best_val_loss"},
    )

    row = out.history[0]
    assert row["train_loss"] == pytest.approx(0.8)
    assert row["train_aux_loss"] == pytest.approx(0.3)
    assert row["val_data_loss"] == pytest.approx(0.5)
    assert row["val_aux_loss"] == pytest.approx(0.4)
    assert row["val_loss"] == pytest.approx(0.9)
    assert row["val_loss_aux_coeff_weighted"] == pytest.approx(0.4)


def test_train_one_epoch_unet_uses_raw_case_mask_for_supervision() -> None:
    h, w = 2, 3
    masks = np.ones((2, 1, h, w), dtype=np.float32)
    masks[0, 0] = 0.0
    masks[0, 0, 0, 0] = 1.0
    source = CaseSpatialFeatureSource(
        channels=("mask_plasma",),
        h=h,
        w=w,
        n_cases=2,
        source="test",
        static_data=np.zeros((0, h, w), dtype=np.float32),
        static_channels=(),
        case_data=masks,
        case_channels=("mask_plasma",),
    )
    target = np.zeros((2, 1, h, w), dtype=np.float32)
    target[0, 0] = 1.0
    target[0, 0, 0, 0] = 0.0

    stats = train_one_epoch_unet(
        _RecordingGridModel((h, w)),
        np.zeros((2, 2), dtype=np.float32),
        target,
        loss_cfg={
            "supervised": {
                "type": "mse",
                "mask": "plasma_only",
                "normalization": "sample_mean",
            }
        },
        supervised_mask=np.ones((h, w), dtype=np.float32),
        spatial_features=source,
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert stats["data"] == pytest.approx(0.0)


def test_run_global_uses_independent_case_supervision_for_train_and_spatial_selection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, w = 2, 3
    train_source = _raw_supervision_source(
        np.ones((2, h, w), dtype=np.float32),
        np.stack([np.full((h, w), 1.0), np.full((h, w), 2.0)]).astype(np.float32),
    )
    val_source = _raw_supervision_source(
        np.ones((2, h, w), dtype=np.float32),
        np.stack([np.full((h, w), 11.0), np.full((h, w), 12.0)]).astype(np.float32),
    )
    compose_distances: list[np.ndarray] = []
    selection_distances: list[np.ndarray] = []
    original_compose = trainer_module.compose_supervised_numpy

    def _record_compose(*args, **kwargs):
        if kwargs.get("distance_any") is not None:
            compose_distances.append(np.asarray(kwargs["distance_any"], dtype=np.float32).copy())
        return original_compose(*args, **kwargs)

    def _record_selection(**kwargs):
        selection_distances.append(np.asarray(kwargs["distance_any"], dtype=np.float32).copy())
        return 1.0, {"spatial_phi": 1.0}

    monkeypatch.setattr(trainer_module, "compose_supervised_numpy", _record_compose)
    monkeypatch.setattr(trainer_module, "case_macro_spatial_objective", _record_selection)
    model = GlobalMLP(
        input_dim=2,
        grid_shape=(h, w),
        out_channels=1,
        output_keys=["phi"],
        hidden=[3],
        dropout=0.0,
        seed=3,
    )
    Trainer(tmp_path / "global_case_supervision").run_global(
        model,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, h * w), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, h * w), dtype=np.float32),
        epochs=1,
        loss_cfg={"supervised": {"type": "mse", "mask": "plasma_only"}},
        supervised_mask=np.ones((h, w), dtype=np.float32),
        supervised_distance=np.zeros((h, w), dtype=np.float32),
        batch_size_cases=1,
        shuffle_cases=False,
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "spatial": {"boundary_weight": 0.25},
        },
        supervision_train=train_source,
        supervision_val=val_source,
    )

    assert [float(batch[0, 0, 0]) for batch in compose_distances[:2]] == [1.0, 2.0]
    np.testing.assert_allclose(compose_distances[-1][:, 0, 0], [11.0, 12.0])
    assert len(selection_distances) == 1
    np.testing.assert_allclose(selection_distances[0][:, 0, 0], [11.0, 12.0])


def test_run_unet_pod_style_lane_keeps_model_inputs_separate_from_case_supervision(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, w = 2, 2
    train_masks = np.ones((2, h, w), dtype=np.float32)
    train_masks[1, 0, 0] = 0.0
    val_masks = np.ones((2, h, w), dtype=np.float32)
    val_masks[0, 1, 1] = 0.0
    train_source = _raw_supervision_source(
        train_masks,
        np.stack([np.full((h, w), 3.0), np.full((h, w), 4.0)]).astype(np.float32),
    )
    val_source = _raw_supervision_source(
        val_masks,
        np.stack([np.full((h, w), 13.0), np.full((h, w), 14.0)]).astype(np.float32),
    )
    compose_distances: list[np.ndarray] = []
    selection_distances: list[np.ndarray] = []
    original_compose = trainer_module.compose_supervised_numpy

    def _record_compose(*args, **kwargs):
        if kwargs.get("distance_any") is not None:
            compose_distances.append(np.asarray(kwargs["distance_any"], dtype=np.float32).copy())
        return original_compose(*args, **kwargs)

    def _record_selection(**kwargs):
        selection_distances.append(np.asarray(kwargs["distance_any"], dtype=np.float32).copy())
        return 1.0, {"spatial_phi": 1.0}

    monkeypatch.setattr(trainer_module, "compose_supervised_numpy", _record_compose)
    monkeypatch.setattr(trainer_module, "case_macro_spatial_objective", _record_selection)
    model = _ModelOwnedAuxGridModel((h, w))
    Trainer(tmp_path / "pod_style_case_supervision").run_unet(
        model,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 1, h, w), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 1, h, w), dtype=np.float32),
        epochs=1,
        loss_cfg={"supervised": {"type": "mse", "mask": "plasma_only"}},
        supervised_mask=np.ones((h, w), dtype=np.float32),
        supervised_distance=np.zeros((h, w), dtype=np.float32),
        spatial_train=None,
        spatial_val=None,
        supervision_train=train_source,
        supervision_val=val_source,
        batch_size_cases=1,
        shuffle_cases=False,
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "spatial": {"boundary_weight": 0.25},
        },
    )

    assert [float(batch[0, 0, 0]) for batch in compose_distances[:2]] == [3.0, 4.0]
    np.testing.assert_allclose(compose_distances[-1][:, 0, 0], [13.0, 14.0])
    assert len(selection_distances) == 1
    np.testing.assert_allclose(selection_distances[0][:, 0, 0], [13.0, 14.0])
    assert len(model.backward_supervised_masks) == 2
    np.testing.assert_allclose(model.backward_supervised_masks[0][0], train_masks[0])
    np.testing.assert_allclose(model.backward_supervised_masks[1][0], train_masks[1])
    assert len(model.validation_supervised_masks) == 2
    np.testing.assert_allclose(model.validation_supervised_masks[0][0], val_masks[0])
    np.testing.assert_allclose(model.validation_supervised_masks[1][0], val_masks[1])


@pytest.mark.torch_runtime
def test_torch_trainer_passes_case_spatial_batches_to_deeponet(tmp_path) -> None:
    require_torch_runtime()
    require_torch()
    h, w = 4, 4
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=2,
        grid_shape=(h, w),
        output_keys=["phi"],
        latent_dim=4,
        hidden_dim=8,
        branch_mode="cond_only",
        sensor_feature_names=channels,
        seed=7,
    )

    class _Geom:
        pass

    geom = _Geom()
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    geom.coord_grid = np.stack([xx, yy], axis=0).astype(np.float32)

    def _spatial(values: list[float]) -> np.ndarray:
        out = np.zeros((len(values), h, w, len(channels)), dtype=np.float32)
        out[..., 0] = xx
        out[..., 1] = yy
        for idx, value in enumerate(values):
            out[idx, ..., 2:] = float(value)
        return out

    spatial_train = _spatial([10.0, 11.0, 12.0, 13.0])
    spatial_val = _spatial([20.0, 21.0])
    calls: list[np.ndarray] = []
    original_predict = model.predict_fields_torch

    def _record_predict(self, cond_t, geom_ctx, spatial_features=None):
        del self
        calls.append(np.asarray(spatial_features, dtype=np.float32).copy())
        return original_predict(cond_t, geom_ctx=geom_ctx, spatial_features=spatial_features)

    model.predict_fields_torch = MethodType(_record_predict, model)  # type: ignore[method-assign]
    trainer = TorchTrainer(tmp_path / "deeponet_case_spatial")
    out = trainer.run_deeponet(
        model=model,
        cond_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4, 1, h, w), dtype=np.float32),
        cond_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2, 1, h, w), dtype=np.float32),
        geom_ctx=geom,
        y_vars=["phi"],
        stages=[{"name": "stage1", "epochs": 1, "lr": 1.0e-3}],
        physics_cfg={"enabled": False},
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "spatial": {"boundary_weight": 0.0},
        },
        batch_size_cases=2,
        shuffle_cases=False,
        spatial_train=spatial_train,
        spatial_val=spatial_val,
    )

    assert len(calls) == 3
    np.testing.assert_allclose(calls[0][:, 0, 0, 2], [10.0, 11.0])
    np.testing.assert_allclose(calls[1][:, 0, 0, 2], [12.0, 13.0])
    np.testing.assert_allclose(calls[2][:, 0, 0, 2], [20.0, 21.0])
    assert out.history[-1]["selection_mode_effective"] == "best_val_spatial_objective"
    assert out.history[-1]["selection_objective_version"] == "case_macro_spatial_rmse_v2"
    assert np.isfinite(out.history[-1]["selected_epoch_score"])


def test_train_one_epoch_unet_reports_group_weighting_breakdown() -> None:
    h, w = 3, 3
    model = _ZeroMultiTargetGridModel((h, w))
    cond = np.zeros((1, 2), dtype=np.float32)
    y = np.ones((1, 3, h, w), dtype=np.float32)
    stats = train_one_epoch_unet(
        model,
        cond,
        y,
        loss_cfg={
            "supervised": {"type": "mse"},
            "group_weighting": {"mode": "uniform_by_group"},
            "target_role_schema": {
                "targets": [
                    {"id": "electron_density", "field_family": "density"},
                    {"id": "ion_density", "field_family": "density"},
                    {"id": "electron_temperature", "field_family": "temperature"},
                ]
            },
        },
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert stats["data"] == 0.5
    assert stats["loss_supervised_group_density"] == 0.25
    assert stats["loss_supervised_group_temperature"] == 0.25


def test_train_one_epoch_unet_resolves_rho_aux_potential_by_role() -> None:
    h, w = 3, 3
    model = _ZeroRhoAuxGridModel((h, w))
    cond = np.zeros((1, 2), dtype=np.float32)
    y = np.zeros((1, 3, h, w), dtype=np.float32)
    y[:, 2, 1, 1] = 1.0
    stats = train_one_epoch_unet(
        model,
        cond,
        y,
        loss_cfg={
            "supervised": {"type": "mse"},
            "target_role_schema": {
                "targets": [
                    {"id": "electron_density", "field_family": "density", "role": "density_electron"},
                    {"id": "electron_temperature", "field_family": "temperature", "role": "temperature_electron"},
                    {"id": "plasma_potential", "field_family": "electrostatic", "role": "potential"},
                ]
            },
        },
        resolved_terms=[{"name": "rho", "enabled": True, "weight": 1.0}],
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert stats["rho"] > 0.0


def test_compact_case_spatial_source_matches_combined_pack() -> None:
    h, w = 4, 5
    channels = [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
        "distance_coil",
        "coil_proximity",
    ]
    static_data = np.zeros((5, h, w), dtype=np.float32)
    for i in range(5):
        static_data[i] = float(i + 1)
    case_data = np.zeros((2, 3, h, w), dtype=np.float32)
    case_data[0, 0] = 10.0
    case_data[0, 1] = 11.0
    case_data[0, 2] = 12.0
    case_data[1, 0] = 20.0
    case_data[1, 1] = 21.0
    case_data[1, 2] = 22.0

    combined = np.concatenate(
        [
            np.broadcast_to(static_data.reshape(1, 5, h, w), (2, 5, h, w)),
            case_data,
        ],
        axis=1,
    )
    source, source_name = build_case_spatial_features(
        channels=channels,
        h=h,
        w=w,
        static_pack={"data": static_data, "channels": np.asarray(channels[:5])},
        case_pack={
            "data": case_data,
            "channels": np.asarray(channels[5:]),
            "case_ids": np.asarray(["case_0", "case_1"]),
        },
        distance_transform_cfg={"mode": "raw"},
        coord_feature_scaler_artifact={"enabled": False, "channels": {}},
        expected_case_ids=["case_0", "case_1"],
    )

    assert source_name == "compact_case_spatial_pack"
    assert source is not None
    got = source.batch(np.asarray([1, 0], dtype=np.int64))
    expected = np.stack([combined[1].transpose(1, 2, 0), combined[0].transpose(1, 2, 0)], axis=0)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    "pack_case_ids",
    [np.asarray(["case_1", "case_0"]), None],
    ids=["reordered", "missing"],
)
def test_compact_case_spatial_source_rejects_case_id_contract_mismatch(
    pack_case_ids: np.ndarray | None,
) -> None:
    h, w = 2, 3
    static_pack = {
        "data": np.zeros((1, h, w), dtype=np.float32),
        "channels": np.asarray(["x"]),
    }
    case_pack = {
        "data": np.zeros((2, 1, h, w), dtype=np.float32),
        "channels": np.asarray(["mask_coil"]),
    }
    if pack_case_ids is not None:
        case_pack["case_ids"] = pack_case_ids

    with pytest.raises(ValueError, match="case_ids"):
        build_case_spatial_features(
            channels=["x", "mask_coil"],
            h=h,
            w=w,
            static_pack=static_pack,
            case_pack=case_pack,
            expected_case_ids=["case_0", "case_1"],
        )


def test_shared_distance_transform_cfg_rejects_active_model_conflict() -> None:
    with pytest.raises(ValueError, match="different input_features.distance_transform"):
        resolve_shared_distance_transform_cfg(
            train_cfg={
                "cno": {"input_features": {"distance_transform": {"mode": "bounded_auto"}}},
                "fno": {"input_features": {"distance_transform": {"mode": "raw"}}},
            },
            model_names=["cno", "fno"],
        )


def test_shared_distance_transform_cfg_ignores_non_spatial_models() -> None:
    resolved = resolve_shared_distance_transform_cfg(
        train_cfg={
            "global_mlp": {"epochs": 2},
            "cno": {"input_features": {"distance_transform": {"mode": "bounded_auto"}}},
        },
        model_names=["global_mlp", "cno"],
    )
    assert resolved["mode"] == "bounded_auto"


def test_coord_scaler_v2_rejects_runtime_distance_transform_mismatch() -> None:
    artifact = {
        "contract_version": 2,
        "enabled": True,
        "mode": "zscore",
        "input_space": "post_distance_transform",
        "distance_transform_effective": {
            "mode": "raw",
            "signed_tanh_tau": 8.0,
            "proximity_tau": 6.0,
            "replace_distance_any": True,
        },
        "channels": {
            "distance_signed": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        },
    }
    with pytest.raises(ValueError, match="different distance-transform space"):
        apply_coord_feature_scaling(
            np.zeros((2, 1), dtype=np.float32),
            channels=["distance_signed"],
            coord_feature_scaler_artifact=artifact,
            distance_transform_cfg={"mode": "bounded_auto"},
        )


def test_legacy_coord_scaler_rejects_nonraw_distance_transform() -> None:
    legacy = {
        "enabled": True,
        "mode": "zscore",
        "channels": {
            "distance_signed": {"type": "zscore", "mean": [44.0], "std": [30.0]},
        },
    }
    with pytest.raises(ValueError, match="legacy coord_feature_scaler"):
        apply_coord_feature_scaling(
            np.zeros((2, 1), dtype=np.float32),
            channels=["distance_signed"],
            coord_feature_scaler_artifact=legacy,
            distance_transform_cfg={"mode": "bounded_auto"},
        )


def test_legacy_coord_scaler_accepts_raw_distance_transform() -> None:
    legacy = {
        "enabled": True,
        "mode": "zscore",
        "channels": {
            "distance_signed": {"type": "zscore", "mean": [2.0], "std": [4.0]},
        },
    }
    scaled, mode, applied = apply_coord_feature_scaling(
        np.asarray([[2.0], [6.0]], dtype=np.float32),
        channels=["distance_signed"],
        coord_feature_scaler_artifact=legacy,
        distance_transform_cfg={"mode": "raw"},
    )
    assert mode == "zscore"
    assert applied is True
    np.testing.assert_allclose(scaled, np.asarray([[0.0], [1.0]], dtype=np.float32))


def test_missing_split_spatial_packs_returns_missing() -> None:
    h, w = 4, 5
    channels = ["x", "y"]

    source, status = build_case_spatial_features(
        channels=channels,
        h=h,
        w=w,
        distance_transform_cfg={"mode": "raw"},
        coord_feature_scaler_artifact={"enabled": False, "channels": {}},
    )
    assert source is None
    assert status == "missing"
