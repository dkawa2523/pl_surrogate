from __future__ import annotations

import pytest

from plasma_surrogate.features.structure_feature_registry import (
    ALLOWED_SPATIAL_CHANNELS,
    list_descriptor_profiles,
    list_feature_profiles,
    list_latent_profiles,
    resolve_spatial_channels_for_feature_profile,
    validate_coord_feature_channels,
)


def test_feature_profiles_include_reserved_names() -> None:
    names = list_feature_profiles()
    assert names == (
        "geom_v1_mainline",
        "boundary_plus_v1",
        "part_lite_v1",
        "part_semantic_v1",
        "icp_struct_spatial_v1",
        "icp_part_sdf_lite_v1",
    )


def test_geom_v1_mainline_resolves_exact_five_channels() -> None:
    assert resolve_spatial_channels_for_feature_profile("geom_v1_mainline") == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
    )


def test_icp_part_sdf_lite_resolves_exact_fourteen_channels() -> None:
    assert resolve_spatial_channels_for_feature_profile("icp_part_sdf_lite_v1") == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
        "distance_coil",
        "coil_proximity",
        "sdf_coil_01",
        "sdf_coil_02",
        "sdf_coil_03",
        "sdf_coil_04",
        "sdf_coil_05",
        "sdf_coil_06",
    )


def test_descriptor_and_latent_profile_lists() -> None:
    assert list_descriptor_profiles() == ("struct_desc_v1", "struct_desc_v2")
    assert list_latent_profiles() == ("shape_ae_v1", "part_latent_v1")


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature_profile"):
        resolve_spatial_channels_for_feature_profile("unknown_profile")


def test_validate_coord_feature_channels_accepts_and_rejects() -> None:
    validated = validate_coord_feature_channels(["x", "y", "distance_signed"])
    assert validated == ("x", "y", "distance_signed")
    with pytest.raises(ValueError, match="unsupported"):
        validate_coord_feature_channels(["x", "unknown_channel"])
    with pytest.raises(ValueError, match="duplicates"):
        validate_coord_feature_channels(["x", "x"])
    assert "mask_plasma" in set(ALLOWED_SPATIAL_CHANNELS)
