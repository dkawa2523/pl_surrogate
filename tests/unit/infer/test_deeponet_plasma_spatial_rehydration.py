from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.features import InferenceFeatureBuilder
from plasma_surrogate.infer.predictor import InferencePredictor
from plasma_surrogate.models.checkpoint import load_checkpoint, save_checkpoint
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from tests._runtime_requirements import require_torch_runtime


pytestmark = pytest.mark.torch_runtime


def _feature_builder(
    provider: FixedGeometryProvider,
    geom,
) -> tuple[InferenceFeatureBuilder, list[str]]:
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    coord = np.asarray(geom.coord_grid, dtype=np.float32)
    pack = {
        "channels": np.asarray(channels, dtype=object),
        "data": np.stack(
            [
                coord[0],
                coord[1],
                np.asarray(geom.mask_plasma, dtype=np.float32),
                np.asarray(geom.distance_signed, dtype=np.float32),
                np.asarray(geom.distance_any, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32),
    }
    input_features = {
        "mode": "geom_feature_pack",
        "features": channels,
        "distance_transform": {
            "mode": "bounded",
            "signed_tanh_tau": 2.0,
            "proximity_tau": 1.5,
            "replace_distance_any": True,
        },
    }
    builder = InferenceFeatureBuilder(
        geometry_provider=provider,
        coord_scaler={},
        coord_feature_scaler={},
        coord_feature_pack=pack,
        coord_distance_transform_stats={},
        coord_input_scaling_cfg={},
        coord_input_features_cfg=input_features,
        grid_input_features_cfg=input_features,
    )
    return builder, channels


def test_deeponet_plasma_checkpoint_inference_rehydrates_training_spatial_rows(
    tmp_path: Path,
    geometry_root: Path,
) -> None:
    require_torch_runtime()
    provider = FixedGeometryProvider(geometry_root)
    geom = provider.get({"geom_id": "default"})
    builder, channels = _feature_builder(provider, geom)
    expected_rows = builder.build_grid_feature_rows(geom, channels)

    model = DeepONetPlasmaOperatorTorch(
        cond_dim=3,
        grid_shape=tuple(geom.mask_plasma.shape),
        output_keys=["ne", "phi"],
        latent_dim=8,
        hidden_dim=16,
        branch_mode="cond_only",
        trunk_fourier_n_freq=2,
        sensor_feature_names=channels,
        seed=17,
    )
    model.set_static_spatial_features(expected_rows, channels=channels)
    cond = np.asarray([0.2, -0.1, 0.7], dtype=np.float32)
    expected = model.predict_fields(cond[None, :], spatial_features=expected_rows)

    ckpt = save_checkpoint(model, tmp_path / "checkpoint")
    loaded = load_checkpoint(ckpt)
    assert loaded._static_feature_rows is None

    actual = InferencePredictor(model=loaded, feature_builder=builder).predict_fields(cond, geom)

    # Runtime rows are supplied per prediction instead of being persisted in
    # checkpoint state or mutating the loaded model's shared static cache.
    assert loaded._static_feature_rows is None
    for key in ("ne", "phi"):
        np.testing.assert_allclose(actual[key], expected[key][0], atol=1.0e-6, rtol=1.0e-6)


def test_deeponet_plasma_exposes_common_spatial_channel_contract() -> None:
    require_torch_runtime()
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=2,
        grid_shape=(4, 4),
        output_keys=["phi"],
        sensor_feature_names=channels,
    )

    assert InferenceFeatureBuilder.resolve_coord_feature_channels(model) == channels
    assert model.to_meta()["input_feature_channels"] == channels
