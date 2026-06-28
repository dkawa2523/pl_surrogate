from __future__ import annotations

from typing import Any


def runtime_table_only() -> dict[str, Any]:
    return {
        "input_mode": "table_only",
        "structure": {
            "feature_profile": "none",
            "descriptor_profile": "none",
            "latent_profile": "none",
            "adapter_mode": "none",
            "provider_mode": "fixed",
        },
    }


def runtime_table_plus_structure(
    *,
    feature_profile: str = "geom_v1_mainline",
    adapter_mode: str = "grid_pack",
    descriptor_profile: str = "none",
    latent_profile: str = "none",
    provider_mode: str = "fixed",
) -> dict[str, Any]:
    return {
        "input_mode": "table_plus_structure",
        "structure": {
            "feature_profile": str(feature_profile),
            "descriptor_profile": str(descriptor_profile),
            "latent_profile": str(latent_profile),
            "adapter_mode": str(adapter_mode),
            "provider_mode": str(provider_mode),
        },
    }


def csv_npz_targets_three_field_example() -> list[dict[str, str]]:
    return [
        {"id": "ne", "source_key": "ne", "value_transform": "identity"},
        {"id": "Te", "source_key": "Te", "value_transform": "identity"},
        {"id": "phi", "source_key": "phi", "value_transform": "identity"},
    ]


def csv_npz_targets_four_field_example() -> list[dict[str, str]]:
    return [
        {"id": "ne", "source_key": "ne", "value_transform": "identity"},
        {"id": "ni", "source_key": "ni", "value_transform": "identity"},
        {"id": "Te", "source_key": "Te", "value_transform": "identity"},
        {"id": "phi", "source_key": "phi", "value_transform": "identity"},
    ]


def default_target_transforms_four_field_example() -> dict[str, dict[str, Any]]:
    return {
        "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
    }
