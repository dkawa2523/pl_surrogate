from __future__ import annotations

import pytest

from plasma_surrogate.train.grid_contracts import validate_unet_like_mainline_contract


STATIC_PART_LITE_CHANNELS = [
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "normal_x",
    "normal_y",
    "curvature_proxy",
    "boundary_band",
]


@pytest.mark.parametrize("model_name", ["u_no", "unet", "ffno"])
def test_mainline_grid_contract_accepts_static_part_lite_profile(model_name: str) -> None:
    validate_unet_like_mainline_contract(
        model_name=model_name,
        model_cfg={},
        selection_cfg={"mode": "best_val_allvars_balance"},
        loss_cfg={},
        y_vars=["ne", "ni", "Te", "phi"],
        target_family="allvars",
        target_vars=["ne", "ni", "Te", "phi"],
        require_shared_output_head=False,
        input_features_mode="geom_feature_pack",
        input_feature_channels=STATIC_PART_LITE_CHANNELS,
    )


@pytest.mark.parametrize("model_name", ["u_no", "unet", "ffno"])
def test_mainline_grid_contract_accepts_part_source_profile(model_name: str) -> None:
    validate_unet_like_mainline_contract(
        model_name=model_name,
        model_cfg={},
        selection_cfg={"mode": "best_val_allvars_balance"},
        loss_cfg={},
        y_vars=["ne", "ni", "Te", "phi"],
        target_family="allvars",
        target_vars=["ne", "ni", "Te", "phi"],
        require_shared_output_head=False,
        input_features_mode="geom_feature_pack",
        input_feature_channels=["x", "y", "distance_signed", "part_sdf_union", "part_source_sum"],
    )


def test_mainline_grid_contract_rejects_reordered_static_part_lite_profile() -> None:
    reordered = list(STATIC_PART_LITE_CHANNELS)
    reordered[-2:] = reversed(reordered[-2:])
    with pytest.raises(ValueError, match="part_lite_static_v1"):
        validate_unet_like_mainline_contract(
            model_name="u_no",
            model_cfg={},
            selection_cfg={"mode": "best_val_allvars_balance"},
            loss_cfg={},
            y_vars=["ne", "ni", "Te", "phi"],
            target_family="allvars",
            target_vars=["ne", "ni", "Te", "phi"],
            require_shared_output_head=False,
            input_features_mode="geom_feature_pack",
            input_feature_channels=reordered,
        )
