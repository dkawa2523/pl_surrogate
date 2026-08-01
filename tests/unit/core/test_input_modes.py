from __future__ import annotations

import pytest

from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    FEATURE_SCHEMA_HASH_KEY,
    RUNTIME_SCHEMA_HASH_PENDING,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
    TARGET_SCHEMA_HASH_KEY,
    attach_runtime_schema_hashes,
    build_input_mode_effective_metadata,
    build_runtime_schema_hashes,
    input_mode_metadata_keys,
    merge_effective_runtime_metadata,
    normalize_input_mode_cfg,
    validate_input_mode_cfg,
    validate_runtime_metadata_contract,
)


def test_normalize_input_mode_cfg_product_default() -> None:
    out = normalize_input_mode_cfg({})
    assert out["runtime"]["input_mode"] == TABLE_PLUS_STRUCTURE
    assert out["runtime"]["structure"]["feature_profile"] == "none"
    assert out["runtime"]["structure"]["descriptor_profile"] == "none"
    assert out["runtime"]["structure"]["latent_profile"] == "none"
    assert out["runtime"]["structure"]["adapter_mode"] == "auto"
    assert out["runtime"]["structure"]["provider_mode"] == "fixed"


def test_validate_input_mode_cfg_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="runtime.input_mode"):
        validate_input_mode_cfg({"runtime": {"input_mode": "invalid_mode"}})


def test_validate_input_mode_cfg_rejects_structure_profiles_in_table_only() -> None:
    cfg = {
        "runtime": {
            "input_mode": TABLE_ONLY,
            "structure": {"feature_profile": "part_lite_v1"},
        }
    }
    with pytest.raises(ValueError, match="table_only"):
        validate_input_mode_cfg(cfg)


def test_validate_input_mode_cfg_rejects_missing_feature_profile_in_table_plus_structure() -> None:
    cfg = {"runtime": {"input_mode": TABLE_PLUS_STRUCTURE, "structure": {"feature_profile": "none"}}}
    with pytest.raises(ValueError, match="feature_profile"):
        validate_input_mode_cfg(cfg)


def test_validate_input_mode_cfg_accepts_valid_feature_profile_in_table_plus_structure() -> None:
    validate_input_mode_cfg(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "structure": {"feature_profile": "geom_v1_mainline"},
            }
        }
    )


def test_build_input_mode_effective_metadata_returns_required_keys() -> None:
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_ONLY,
                "structure": {
                    "feature_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    assert set(input_mode_metadata_keys()).issubset(meta)
    assert meta["input_mode_effective"] == TABLE_ONLY
    assert meta["structure_feature_profile_effective"] == "none"
    assert meta["structure_adapter_mode_effective"] == "none"
    assert meta["geometry_provider_mode_effective"] == "fixed"
    assert meta[TARGET_SCHEMA_HASH_KEY] == RUNTIME_SCHEMA_HASH_PENDING
    assert meta[FEATURE_SCHEMA_HASH_KEY] == RUNTIME_SCHEMA_HASH_PENDING
    assert "descriptor_profile" not in meta
    assert "latent_profile" not in meta


def test_build_input_mode_effective_metadata_keeps_optional_descriptor_latent_profiles() -> None:
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "structure": {
                    "feature_profile": "part_lite_v1",
                    "descriptor_profile": "struct_desc_v1",
                    "latent_profile": "shape_ae_v1",
                    "adapter_mode": "hybrid_pack_descriptor",
                    "provider_mode": "parametric_parts",
                },
            }
        }
    )
    assert meta["descriptor_profile"] == "struct_desc_v1"
    assert meta["latent_profile"] == "shape_ae_v1"


def test_attach_runtime_schema_hashes_overwrites_pending_hashes() -> None:
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "structure": {"feature_profile": "geom_v1_mainline", "adapter_mode": "grid_pack"},
            }
        }
    )
    hashes = build_runtime_schema_hashes(
        {
            "output_layout": {"vars": ["electron_density"]},
            "target_role_schema": {"targets": [{"id": "electron_density", "role": "density_electron"}]},
            "channel_map": {"channels": []},
            "coord_feature_pack_meta": {"channels": ["x", "y"]},
        }
    )
    out = attach_runtime_schema_hashes(meta, schema_hashes=hashes)
    assert out[TARGET_SCHEMA_HASH_KEY] == hashes[TARGET_SCHEMA_HASH_KEY]
    assert out[FEATURE_SCHEMA_HASH_KEY] == hashes[FEATURE_SCHEMA_HASH_KEY]


def test_validate_runtime_metadata_contract_fails_on_mismatch() -> None:
    request_meta = build_input_mode_effective_metadata(
        {"runtime": {"input_mode": TABLE_ONLY, "structure": {"feature_profile": "none", "adapter_mode": "none"}}}
    )
    checkpoint_meta = dict(request_meta)
    checkpoint_meta["structure_adapter_mode_effective"] = "auto"
    with pytest.raises(ValueError, match="expected='auto'.*got='none'"):
        validate_runtime_metadata_contract(
            request_meta=request_meta,
            checkpoint_meta=checkpoint_meta,
            context="test",
        )


def test_validate_runtime_metadata_contract_fails_on_missing_required_key() -> None:
    request_meta = build_input_mode_effective_metadata(
        {"runtime": {"input_mode": TABLE_ONLY, "structure": {"feature_profile": "none", "adapter_mode": "none"}}}
    )
    checkpoint_meta = dict(request_meta)
    checkpoint_meta.pop(TARGET_SCHEMA_HASH_KEY)
    with pytest.raises(ValueError, match=TARGET_SCHEMA_HASH_KEY):
        validate_runtime_metadata_contract(
            request_meta=request_meta,
            checkpoint_meta=checkpoint_meta,
            context="test",
        )


def test_merge_effective_runtime_metadata_includes_model_internal_descriptor_latent_keys() -> None:
    runtime_meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": TABLE_PLUS_STRUCTURE,
                "structure": {
                    "feature_profile": "geom_v1_mainline",
                    "adapter_mode": "grid_pack",
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
        {"runtime": {"input_mode": TABLE_ONLY, "structure": {"feature_profile": "none", "adapter_mode": "none"}}}
    )
    with pytest.raises(ValueError, match="runtime metadata mismatch"):
        merge_effective_runtime_metadata(
            runtime_meta=runtime_meta,
            dispatch_meta={"input_mode_effective": "table_plus_structure"},
        )
