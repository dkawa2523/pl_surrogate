"""Product loss contract constants shared by training entrypoints."""

from __future__ import annotations

from typing import Any


REMOVED_SUPERVISED_KEYS = (
    "boundary_profile_weighting",
    "boundary_type_weighting",
    "boundary_weight",
    "chamber_aux",
    "density_positivity_penalty",
    "density_relative_weighting",
    "global_target_region",
    "positive_penalty",
    "region_balance",
    "region_weighting",
    "relative_weighting",
    "robust_weighting",
    "sdf_weighting",
    "spatial_consistency",
    "target_region_by_var",
)


def removed_supervised_keys(supervised_cfg: dict[str, Any] | None) -> list[str]:
    sup = dict(supervised_cfg or {})
    return [key for key in REMOVED_SUPERVISED_KEYS if key in sup]


def reject_removed_supervised_keys(
    supervised_cfg: dict[str, Any] | None,
    *,
    message_prefix: str,
) -> None:
    found = removed_supervised_keys(supervised_cfg)
    if found:
        raise ValueError(f"{message_prefix} got={found}")


__all__ = [
    "REMOVED_SUPERVISED_KEYS",
    "reject_removed_supervised_keys",
    "removed_supervised_keys",
]
