"""Head modules for plasma field generation."""

from plasma_surrogate.models.heads.role_grouped import (
    GROUPED_OUTPUT_HEAD_MODES,
    OUTPUT_HEAD_MODE_CUSTOM_GROUPS,
    OUTPUT_HEAD_MODE_ROLE_GROUPED,
    OUTPUT_HEAD_MODE_SHARED,
    OUTPUT_GROUP_HEAD_DEFAULT,
    OUTPUT_GROUP_HEAD_POISSON_HYBRID,
    OUTPUT_GROUP_HEAD_SPATIAL_REFINE,
    OUTPUT_GROUP_HEAD_TYPES,
    ROLE_GROUPED_OUTPUT_HEAD_MODELS,
    build_role_grouped_conv2d_head,
    configure_output_head_metadata,
    custom_groups_from_output_head_config,
    is_grouped_output_head_mode,
    output_head_strict_from_config,
    resolve_output_head_groups,
    target_groups_from_metadata,
    target_groups_to_metadata,
    validate_output_head_group_options,
)


def __getattr__(name: str):
    if name == "PlasmaHead":
        from plasma_surrogate.models.heads.plasma_head import PlasmaHead

        return PlasmaHead
    raise AttributeError(name)

__all__ = [
    "OUTPUT_HEAD_MODE_ROLE_GROUPED",
    "OUTPUT_HEAD_MODE_CUSTOM_GROUPS",
    "OUTPUT_HEAD_MODE_SHARED",
    "OUTPUT_GROUP_HEAD_DEFAULT",
    "OUTPUT_GROUP_HEAD_POISSON_HYBRID",
    "OUTPUT_GROUP_HEAD_SPATIAL_REFINE",
    "OUTPUT_GROUP_HEAD_TYPES",
    "PlasmaHead",
    "GROUPED_OUTPUT_HEAD_MODES",
    "ROLE_GROUPED_OUTPUT_HEAD_MODELS",
    "build_role_grouped_conv2d_head",
    "configure_output_head_metadata",
    "custom_groups_from_output_head_config",
    "is_grouped_output_head_mode",
    "output_head_strict_from_config",
    "resolve_output_head_groups",
    "target_groups_from_metadata",
    "target_groups_to_metadata",
    "validate_output_head_group_options",
]
