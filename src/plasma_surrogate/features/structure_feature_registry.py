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
    "sdf_coil_01",
    "sdf_coil_02",
    "sdf_coil_03",
    "sdf_coil_04",
    "sdf_coil_05",
    "sdf_coil_06",
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
    # Reserved profile names are already resolvable in Phase 1b; part lanes are Phase 2+.
    "part_lite_v1": (
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
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

DESCRIPTOR_PROFILES: tuple[str, ...] = ("struct_desc_v1", "struct_desc_v2")
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
