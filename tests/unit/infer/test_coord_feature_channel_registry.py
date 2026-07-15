from __future__ import annotations

import pytest

from plasma_surrogate.infer.features import InferenceFeatureBuilder


class _DummyModel:
    input_feature_channels = ["x", "y", "distance_signed"]


class _BadDummyModel:
    input_feature_channels = ["x", "unknown_feature"]


class _LegacyDeepONetModel:
    sensor_feature_names = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]


def test_infer_resolve_coord_feature_channels_uses_registry_validator() -> None:
    channels = InferenceFeatureBuilder.resolve_coord_feature_channels(_DummyModel())
    assert channels == ["x", "y", "distance_signed"]


def test_infer_resolve_coord_feature_channels_rejects_unknown_channels() -> None:
    with pytest.raises(ValueError, match="Unsupported coord feature channels"):
        InferenceFeatureBuilder.resolve_coord_feature_channels(_BadDummyModel())


def test_infer_resolve_coord_feature_channels_supports_legacy_sensor_names() -> None:
    channels = InferenceFeatureBuilder.resolve_coord_feature_channels(_LegacyDeepONetModel())
    assert channels == ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
