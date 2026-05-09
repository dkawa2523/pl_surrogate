from __future__ import annotations

import pytest

from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
    build_input_mode_effective_metadata,
    merge_effective_runtime_metadata,
    normalize_input_mode_cfg,
    resolve_benchmark_runtime_controls,
    resolve_input_mode_metadata_contract,
    resolve_runtime_controls,
    validate_input_mode_cfg,
)


def test_normalize_input_mode_cfg_legacy_default() -> None:
    out = normalize_input_mode_cfg({})
    assert out["runtime"]["input_mode"] == TABLE_PLUS_STRUCTURE
    assert out["runtime"]["structure"]["feature_profile"] == "none"
    assert out["runtime"]["structure"]["descriptor_profile"] == "none"
    assert out["runtime"]["structure"]["latent_profile"] == "none"
    assert out["runtime"]["structure"]["adapter_mode"] == "auto"
    assert out["runtime"]["structure"]["provider_mode"] == "fixed"


def test_validate_input_mode_cfg_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="runtime.input_mode must be one of"):
        validate_input_mode_cfg({"runtime": {"input_mode": "invalid_mode"}})


def test_validate_input_mode_cfg_rejects_structure_profiles_in_table_only() -> None:
    cfg = {
        "runtime": {
            "input_mode": TABLE_ONLY,
            "structure": {
                "feature_profile": "part_lite_v1",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        }
    }
    with pytest.raises(ValueError, match="table_only"):
        validate_input_mode_cfg(cfg)


def test_validate_input_mode_cfg_rejects_missing_feature_profile_in_table_plus_structure() -> None:
    cfg = {
        "runtime": {
            "input_mode": TABLE_PLUS_STRUCTURE,
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        }
    }
    with pytest.raises(ValueError, match="feature_profile"):
        validate_input_mode_cfg(cfg)


def test_validate_input_mode_cfg_accepts_valid_feature_profile_in_table_plus_structure() -> None:
    cfg = {
        "runtime": {
            "input_mode": TABLE_PLUS_STRUCTURE,
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        }
    }
    validate_input_mode_cfg(cfg)


def test_build_input_mode_effective_metadata_returns_canonical_keys() -> None:
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_ONLY,
                "structure": {
                    "feature_profile": "none",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    assert meta["input_mode_effective"] == TABLE_ONLY
    assert meta["structure_feature_profile_effective"] == "none"
    assert meta["structure_descriptor_profile_effective"] == "none"
    assert meta["structure_latent_profile_effective"] == "none"
    assert meta["structure_adapter_mode_effective"] == "none"
    assert meta["geometry_provider_mode_effective"] == "fixed"
    assert meta["has_structure_inputs_effective"] is False


def test_merge_effective_runtime_metadata_includes_dispatch_descriptor_latent_keys() -> None:
    runtime_meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "structure": {
                    "feature_profile": "geom_v1_mainline",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    merged = merge_effective_runtime_metadata(
        runtime_meta=runtime_meta,
        dispatch_meta={
            DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: 0,
            DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: "none",
            DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: "none",
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
        },
    )
    assert merged[DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY] == 0
    assert merged[DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY] == "none"
    assert merged[DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY] == "none"
    assert merged[DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY] is False


def test_merge_effective_runtime_metadata_rejects_conflicting_values() -> None:
    runtime_meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_ONLY,
                "structure": {
                    "feature_profile": "none",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_effective_runtime_metadata(
            runtime_meta=runtime_meta,
            dispatch_meta={"input_mode_effective": "table_plus_structure"},
        )


def test_resolve_runtime_controls_returns_normalized_values() -> None:
    strict_mode, allow_fallback = resolve_runtime_controls(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "strict_input_mode": "warn",
                "allow_mode_fallback": True,
                "structure": {"feature_profile": "geom_v1_mainline"},
            }
        }
    )
    assert strict_mode == "warn"
    assert allow_fallback is True


def test_resolve_benchmark_runtime_controls_rejects_non_error_mode() -> None:
    with pytest.raises(ValueError, match="requires runtime.strict_input_mode='error'"):
        resolve_benchmark_runtime_controls(
            {
                "runtime": {
                    "input_mode": TABLE_PLUS_STRUCTURE,
                    "strict_input_mode": "warn",
                    "allow_mode_fallback": False,
                    "structure": {"feature_profile": "geom_v1_mainline"},
                }
            }
        )


def test_resolve_benchmark_runtime_controls_rejects_fallback_true() -> None:
    with pytest.raises(ValueError, match="forbids runtime.allow_mode_fallback=true"):
        resolve_benchmark_runtime_controls(
            {
                "runtime": {
                    "input_mode": TABLE_PLUS_STRUCTURE,
                    "strict_input_mode": "error",
                    "allow_mode_fallback": True,
                    "structure": {"feature_profile": "geom_v1_mainline"},
                }
            }
        )


def test_resolve_input_mode_metadata_contract_applies_fallback() -> None:
    resolved, warnings_out, fallback_applied = resolve_input_mode_metadata_contract(
        request_meta={
            "input_mode_effective": "table_only",
            "structure_feature_profile_effective": "none",
            "structure_descriptor_profile_effective": "none",
            "structure_latent_profile_effective": "none",
            "structure_adapter_mode_effective": "none",
            "geometry_provider_mode_effective": "fixed",
            "has_structure_inputs_effective": False,
        },
        checkpoint_meta={
            "input_mode_effective": "table_only",
            "structure_feature_profile_effective": "none",
            "structure_descriptor_profile_effective": "none",
            "structure_latent_profile_effective": "none",
            "structure_adapter_mode_effective": "auto",
            "geometry_provider_mode_effective": "fixed",
            "has_structure_inputs_effective": False,
        },
        strict_input_mode="error",
        allow_mode_fallback=True,
    )
    assert fallback_applied is True
    assert resolved["structure_adapter_mode_effective"] == "auto"
    assert len(warnings_out) >= 1
