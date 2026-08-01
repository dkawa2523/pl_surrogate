from __future__ import annotations

import pytest

from plasma_surrogate.features.structure_feature_registry import FEATURE_PROFILE_CHANNELS
from plasma_surrogate.train.deeponet_contracts import validate_deeponet_mainline_contract


TARGETS = ["ne", "ni", "Te", "phi"]
BASE_CHANNELS = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]


def _validate(
    channels: list[str],
    *,
    branch_mode: str = "cond_only",
    sensor_pool_mode: str = "moments",
) -> None:
    validate_deeponet_mainline_contract(
        deeponet_cfg={
            "input_features": {"distance_transform": {"mode": "raw"}},
            "operator_mode": "plain",
            "model_cfg": {
                "trunk_input_mode": "geom_feature_pack",
                "branch_mode": branch_mode,
                "sensor_pool_mode": sensor_pool_mode,
            },
        },
        selection_cfg={
            "mode": "best_val_allvars_balance",
            "weights": {name: 0.25 for name in TARGETS},
        },
        loss_cfg={},
        y_vars=TARGETS,
        target_family="allvars",
        target_vars=TARGETS,
        input_features_mode="geom_feature_pack",
        input_feature_channels=channels,
    )


@pytest.mark.parametrize(
    "channels",
    [BASE_CHANNELS, list(FEATURE_PROFILE_CHANNELS["part_lite_v1"])],
    ids=["base-five", "case-aware-part-lite"],
)
def test_deeponet_mainline_accepts_ordered_base_prefix_with_registered_extensions(
    channels: list[str],
) -> None:
    _validate(channels)


def test_deeponet_mainline_accepts_nonlocal_set_pool_branch() -> None:
    _validate(
        list(FEATURE_PROFILE_CHANNELS["part_lite_v1"]),
        branch_mode="set_mlp_pool",
        sensor_pool_mode="set_mlp_pool",
    )


def test_deeponet_mainline_rejects_branch_pool_semantic_mismatch() -> None:
    with pytest.raises(ValueError, match="sensor_pool_mode must be set_mlp_pool"):
        _validate(
            list(FEATURE_PROFILE_CHANNELS["part_lite_v1"]),
            branch_mode="set_mlp_pool",
            sensor_pool_mode="moments",
        )


@pytest.mark.parametrize(
    "channels",
    [
        ["x", "y", "distance_signed", "mask_plasma", "distance_any"],
        ["x", "y", "mask_plasma", "distance_signed", "normal_x"],
    ],
    ids=["reordered", "missing-required"],
)
def test_deeponet_mainline_rejects_invalid_base_prefix(channels: list[str]) -> None:
    with pytest.raises(ValueError, match="must start with the ordered mainline channels"):
        _validate(channels)


def test_deeponet_mainline_rejects_unregistered_extension_channel() -> None:
    with pytest.raises(ValueError, match="unsupported entries"):
        _validate([*BASE_CHANNELS, "not_a_structure_channel"])
