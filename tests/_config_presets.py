from __future__ import annotations

from typing import Any


def runtime_table_only() -> dict[str, Any]:
    return runtime_table_only_with_controls()


def runtime_table_only_with_controls(
    *,
    strict_input_mode: str = "error",
    allow_mode_fallback: bool = False,
) -> dict[str, Any]:
    return {
        "input_mode": "table_only",
        "strict_input_mode": str(strict_input_mode),
        "allow_mode_fallback": bool(allow_mode_fallback),
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
    strict_input_mode: str = "error",
    allow_mode_fallback: bool = False,
) -> dict[str, Any]:
    return {
        "input_mode": "table_plus_structure",
        "strict_input_mode": str(strict_input_mode),
        "allow_mode_fallback": bool(allow_mode_fallback),
        "structure": {
            "feature_profile": str(feature_profile),
            "descriptor_profile": str(descriptor_profile),
            "latent_profile": str(latent_profile),
            "adapter_mode": str(adapter_mode),
            "provider_mode": str(provider_mode),
        },
    }


def csv_npz_targets_with_ne_te_phi() -> list[dict[str, str]]:
    return [
        {"id": "ne", "source_key": "log_ne", "value_transform": "pow10"},
        {"id": "Te", "source_key": "Te", "value_transform": "identity"},
        {"id": "phi", "source_key": "phi", "value_transform": "identity"},
    ]


def csv_npz_targets_with_ne_ni_te_phi() -> list[dict[str, str]]:
    return [
        {"id": "ne", "source_key": "ne", "value_transform": "identity"},
        {"id": "ni", "source_key": "ni", "value_transform": "identity"},
        {"id": "Te", "source_key": "Te", "value_transform": "identity"},
        {"id": "phi", "source_key": "phi", "value_transform": "identity"},
    ]


def default_target_transforms_ne_ni_te_phi() -> dict[str, dict[str, Any]]:
    return {
        "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
    }
