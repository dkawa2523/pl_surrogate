"""Structure feature-profile registry and coord-channel validation helpers."""

from __future__ import annotations

from typing import Any


ALLOWED_SPATIAL_CHANNELS: tuple[str, ...] = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "mask_coil",
    "valid_field_mask",
    "outside_mask",
    "distance_coil",
    "coil_proximity",
    "normal_x",
    "normal_y",
    "curvature_proxy",
    "boundary_band",
    "part_sdf_nearest",
    "part_sdf_second",
    "part_gap_proxy",
    "solid_proximity",
    "part_second_proximity",
    "part_competition",
    "part_sdf_union",
    "part_source_sum",
    "part_source_mean",
    "sdf_coil_01",
    "sdf_coil_02",
    "sdf_coil_03",
    "sdf_coil_04",
    "sdf_coil_05",
    "sdf_coil_06",
    "vacuum_aphi_unit",
    "vacuum_br_unit",
    "vacuum_bz_unit",
    "vacuum_bmag_unit",
)

GEOM_V1_MAINLINE_CHANNELS: tuple[str, ...] = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
)

FEATURE_PROFILE_CHANNELS: dict[str, tuple[str, ...]] = {
    # Mainline exact 5-channel contract (order-sensitive).
    "geom_v1_mainline": GEOM_V1_MAINLINE_CHANNELS,
    "boundary_plus_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
    ),
    # Static subset shared with part_lite_v1.  This profile is used by
    # dimension-conditioned ablations so the chamber/boundary information is
    # identical without admitting case-varying part maps.
    "part_lite_static_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
        "boundary_band",
    ),
    # Smooth, order-invariant structure contract for general grid surrogates.
    # Derivative channels and per-part nearest/second summaries are deliberately
    # excluded because their medial-axis discontinuities can be copied into the
    # predicted physical fields.
    "smooth_structure_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "boundary_band",
        "solid_proximity",
    ),
    # Geometry-generation-ready contract.  The union SDF carries shape while
    # the additive source field preserves the number and proximity of parts
    # without assigning semantic meaning to their storage order.
    "part_source_v1": (
        "x",
        "y",
        "distance_signed",
        "part_sdf_union",
        "part_source_sum",
    ),
    # Order-invariant ICP coil-source contract.  The occupancy field carries
    # topology without per-coil slot labels or union-SDF medial axes.  Loaded
    # electromagnetic fields are supervised as causal intermediate targets.
    "icp_coil_source_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
    ),
    # Fair shape-representation comparison profile.  It differs from
    # geom_v1_mainline only by the order-invariant union coil SDF and carries
    # neither coil-slot identity nor additional electromagnetic information.
    "icp_coil_union_sdf_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "part_sdf_union",
    ),
    # Optimization-ready ICP representation.  The union SDF carries geometry
    # and the additive source field preserves multiplicity that a nearest-part
    # distance alone can hide after encoder downsampling.
    "icp_coil_sdf_source_v2": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "part_sdf_union",
        "part_source_sum",
    ),
    # Count-neutral successor to v2. Averaging active equal-strength source
    # maps removes the automatic amplitude increase when coil count changes,
    # while union SDF continues to encode geometry and topology.
    "icp_coil_sdf_source_mean_v3": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "part_sdf_union",
        "part_source_mean",
    ),
    # Structure-only long-range electromagnetic encoding.  These fields are
    # deterministic unit-current vacuum solutions, so they remain available
    # for unseen coil generation and do not leak plasma simulation outputs.
    "icp_vacuum_field_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "vacuum_aphi_unit",
        "vacuum_br_unit",
        "vacuum_bz_unit",
        "vacuum_bmag_unit",
    ),
    # Conference-continuity electromagnetic carrier. This preserves the
    # adopted union-SDF/source representation and appends only deterministic
    # unit-current vacuum fields that remain available for unseen layouts.
    "icp_coil_sdf_source_mean_vacuum_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "part_sdf_union",
        "part_source_mean",
        "vacuum_aphi_unit",
        "vacuum_br_unit",
        "vacuum_bz_unit",
        "vacuum_bmag_unit",
    ),
    # Order-invariant structural refinement of the electromagnetic carrier.
    # Multi-coil spacing, size and height remain visible without coil slots.
    "icp_coil_sdf_source_mean_vacuum_structure_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "part_sdf_union",
        "part_source_mean",
        "vacuum_aphi_unit",
        "vacuum_br_unit",
        "vacuum_bz_unit",
        "vacuum_bmag_unit",
        "part_second_proximity",
        "part_competition",
        "solid_proximity",
    ),
    # part_lite_v1 uses order-invariant summaries; semantic/ICP profiles keep their existing shapes.
    "part_lite_v1": (
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
    ),
    "part_semantic_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
    ),
    "icp_struct_spatial_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
        "distance_coil",
        "coil_proximity",
    ),
    "icp_part_sdf_lite_v1": (
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
    ),
}

DESCRIPTOR_PROFILES: tuple[str, ...] = ("struct_desc_v1", "struct_desc_v2", "struct_desc_lite_v1")
LATENT_PROFILES: tuple[str, ...] = ("shape_ae_v1", "part_latent_v1")


def _normalize_profile_name(value: Any, *, kind: str) -> str:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        raise ValueError(f"{kind} profile name must be a non-empty string")
    return text


def normalize_feature_profile_name(value: Any) -> str:
    return _normalize_profile_name(value, kind="feature")


def normalize_descriptor_profile_name(value: Any) -> str:
    return _normalize_profile_name(value, kind="descriptor")


def normalize_latent_profile_name(value: Any) -> str:
    return _normalize_profile_name(value, kind="latent")


def validate_feature_profile_name(value: Any) -> str:
    profile_name = normalize_feature_profile_name(value)
    if profile_name not in FEATURE_PROFILE_CHANNELS:
        raise ValueError(
            "runtime.structure.feature_profile must be one of: "
            f"{list(FEATURE_PROFILE_CHANNELS.keys())}; got={profile_name!r}"
        )
    return profile_name


def validate_descriptor_profile_name(value: Any) -> str:
    profile_name = normalize_descriptor_profile_name(value)
    if profile_name not in set(DESCRIPTOR_PROFILES):
        raise ValueError(
            "runtime.structure.descriptor_profile must be one of: "
            f"{list(DESCRIPTOR_PROFILES)}; got={profile_name!r}"
        )
    return profile_name


def validate_latent_profile_name(value: Any) -> str:
    profile_name = normalize_latent_profile_name(value)
    if profile_name not in set(LATENT_PROFILES):
        raise ValueError(
            "runtime.structure.latent_profile must be one of: "
            f"{list(LATENT_PROFILES)}; got={profile_name!r}"
        )
    return profile_name


def resolve_spatial_channels_for_feature_profile(profile_name: Any) -> tuple[str, ...]:
    profile = validate_feature_profile_name(profile_name)
    return tuple(FEATURE_PROFILE_CHANNELS[profile])


def list_feature_profiles() -> tuple[str, ...]:
    return tuple(FEATURE_PROFILE_CHANNELS.keys())


def list_descriptor_profiles() -> tuple[str, ...]:
    return tuple(DESCRIPTOR_PROFILES)


def list_latent_profiles() -> tuple[str, ...]:
    return tuple(LATENT_PROFILES)


def validate_coord_feature_channels(channels: Any) -> tuple[str, ...]:
    if not isinstance(channels, (list, tuple)):
        raise ValueError("coord feature channels must be a non-empty list")
    if len(channels) == 0:
        raise ValueError("coord feature channels must be a non-empty list")
    values = [str(v).strip() for v in channels]
    if any(v == "" for v in values):
        raise ValueError("coord feature channels must not contain empty entries")
    unknown = [v for v in values if v not in set(ALLOWED_SPATIAL_CHANNELS)]
    if unknown:
        raise ValueError(
            "coord feature channels contains unsupported entries: "
            f"{unknown}; allowed={list(ALLOWED_SPATIAL_CHANNELS)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("coord feature channels must not contain duplicates")
    return tuple(values)


__all__ = [
    "ALLOWED_SPATIAL_CHANNELS",
    "DESCRIPTOR_PROFILES",
    "FEATURE_PROFILE_CHANNELS",
    "GEOM_V1_MAINLINE_CHANNELS",
    "LATENT_PROFILES",
    "list_descriptor_profiles",
    "list_feature_profiles",
    "list_latent_profiles",
    "normalize_descriptor_profile_name",
    "normalize_feature_profile_name",
    "normalize_latent_profile_name",
    "resolve_spatial_channels_for_feature_profile",
    "validate_coord_feature_channels",
    "validate_descriptor_profile_name",
    "validate_feature_profile_name",
    "validate_latent_profile_name",
]
