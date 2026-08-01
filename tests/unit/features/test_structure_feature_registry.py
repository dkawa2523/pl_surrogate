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
        "part_lite_static_v1",
        "smooth_structure_v1",
        "part_source_v1",
        "part_lite_v1",
        "part_semantic_v1",
        "icp_struct_spatial_v1",
        "icp_part_sdf_lite_v1",
    )


def test_smooth_structure_v1_excludes_derivative_and_per_part_channels() -> None:
    channels = resolve_spatial_channels_for_feature_profile("smooth_structure_v1")
    assert channels == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "boundary_band",
        "solid_proximity",
    )
    assert not {"normal_x", "normal_y", "curvature_proxy", "part_sdf_nearest"} & set(channels)


def test_part_source_v1_is_minimal_and_generation_ready() -> None:
    assert resolve_spatial_channels_for_feature_profile("part_source_v1") == (
        "x",
        "y",
        "distance_signed",
        "part_sdf_union",
        "part_source_sum",
    )


def test_geom_v1_mainline_resolves_exact_five_channels() -> None:
    assert resolve_spatial_channels_for_feature_profile("geom_v1_mainline") == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
    )


def test_part_lite_v1_resolves_lightweight_boundary_and_part_channels() -> None:
    assert resolve_spatial_channels_for_feature_profile("part_lite_v1") == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
        "boundary_band",
        "part_sdf_nearest",
        "part_sdf_second",
        "part_gap_proxy",
        "solid_proximity",
    )


def test_part_lite_static_v1_matches_part_lite_static_prefix() -> None:
    static_channels = resolve_spatial_channels_for_feature_profile("part_lite_static_v1")
    part_lite_channels = resolve_spatial_channels_for_feature_profile("part_lite_v1")
    assert static_channels == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
        "boundary_band",
    )
    assert part_lite_channels[: len(static_channels)] == static_channels


def test_part_semantic_v1_keeps_existing_boundary_plus_channels() -> None:
    assert resolve_spatial_channels_for_feature_profile("part_semantic_v1") == (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
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
    assert list_descriptor_profiles() == ("struct_desc_v1", "struct_desc_v2", "struct_desc_lite_v1")
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
    assert "boundary_band" in set(ALLOWED_SPATIAL_CHANNELS)
    assert "part_sdf_nearest" in set(ALLOWED_SPATIAL_CHANNELS)
