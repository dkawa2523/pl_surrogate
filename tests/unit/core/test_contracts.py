from __future__ import annotations

import pytest

from plasma_surrogate.core.contracts import (
    EvaluationProtocol,
    RuntimeContract,
    build_product_manifest,
    validate_geom_deeponet_siren_descriptor_contract,
    validate_pod_descriptor_latent_contract,
    validate_runtime_metadata_pair,
)


def test_runtime_contract_round_trips_effective_metadata() -> None:
    meta = {
        "input_mode_effective": "table_plus_structure",
        "structure_feature_profile_effective": "geom_v1_mainline",
        "structure_adapter_mode_effective": "grid_pack",
        "geometry_provider_mode_effective": "fixed",
        "target_schema_hash": "target123",
        "feature_schema_hash": "feature456",
        "descriptor_profile": "struct_desc_v1",
    }

    contract = RuntimeContract.from_metadata(meta)

    assert contract.input_mode == "table_plus_structure"
    assert contract.descriptor_profile == "struct_desc_v1"
    assert contract.to_metadata()["target_schema_hash"] == "target123"


def test_runtime_contract_rejects_pending_hashes_for_execution() -> None:
    meta = {
        "input_mode_effective": "table_only",
        "structure_feature_profile_effective": "none",
        "structure_adapter_mode_effective": "none",
        "geometry_provider_mode_effective": "fixed",
        "target_schema_hash": "pending",
        "feature_schema_hash": "feature456",
    }

    with pytest.raises(ValueError, match="schema hashes must be concrete"):
        RuntimeContract.from_metadata(meta)

    assert RuntimeContract.from_metadata(meta, allow_pending_schema_hashes=True).target_schema_hash == "pending"


def test_validate_runtime_metadata_pair_uses_runtime_contract_values() -> None:
    meta = {
        "input_mode_effective": "table_only",
        "structure_feature_profile_effective": "none",
        "structure_adapter_mode_effective": "none",
        "geometry_provider_mode_effective": "fixed",
        "target_schema_hash": "pending",
        "feature_schema_hash": "pending",
    }

    assert validate_runtime_metadata_pair(request_meta=dict(meta), checkpoint_meta=dict(meta)) == meta


def test_validate_pod_descriptor_latent_contract_rejects_table_only_descriptor() -> None:
    with pytest.raises(ValueError, match="table_only"):
        validate_pod_descriptor_latent_contract(
            input_mode="table_only",
            adapter_mode="none",
            descriptor_profile="struct_desc_v1",
            latent_profile="none",
        )


def test_validate_geom_deeponet_siren_descriptor_contract_returns_checkpoint_dim() -> None:
    meta = validate_geom_deeponet_siren_descriptor_contract(
        input_mode="table_plus_structure",
        adapter_mode="hybrid_pack_descriptor",
        descriptor_profile="struct_desc_v1",
        provider_mode="parametric_parts",
        checkpoint_meta={
            "geom_deeponet_siren_descriptor_dim_effective": 8,
            "geom_deeponet_siren_descriptor_profile_effective": "struct_desc_v1",
        },
        require_checkpoint_metadata=True,
    )

    assert meta["geom_deeponet_siren_descriptor_dim_effective"] == 8


def test_evaluation_protocol_serializes_target_and_region_policy() -> None:
    protocol = EvaluationProtocol(
        mode="dual_axis",
        primary_split="interp",
        primary_metric="surrogate_quality_score",
        objective_mode="min",
        target_vars=("ne", "ni", "Te", "phi"),
        region_bands={"mode": "fixed_px", "boundary_in_px": 2.0},
        interp_weight=0.5,
        extrap_weight=0.5,
    )

    payload = protocol.as_dict()

    assert payload["mode"] == "dual_axis"
    assert payload["target_vars"] == ["ne", "ni", "Te", "phi"]
    assert payload["region_bands"]["boundary_in_px"] == 2.0


def test_product_manifest_groups_contract_domains_without_flat_compatibility() -> None:
    resolved = {
        "input_mode_effective": "table_only",
        "structure_feature_profile_effective": "none",
        "structure_adapter_mode_effective": "none",
        "geometry_provider_mode_effective": "fixed",
        "target_schema_hash": "target123",
        "feature_schema_hash": "feature456",
        "model_contracts": {"unet": {"target_family_effective": "allvars"}},
        "effective_steps_per_model": {"unet": 1},
        "diagnostics_effective": {"spatial": {"enabled": True}},
        "artifact_hashes": {"lock_hash": "abc"},
        "split": {"seed": 3, "ratios": [0.7, 0.2, 0.1]},
    }

    manifest = build_product_manifest(split={"train": [0]}, resolved=resolved).as_dict()

    assert manifest["input_mode"]["input_mode_effective"] == "table_only"
    assert manifest["model"]["contracts"]["unet"] == {"target_family_effective": "allvars"}
    assert "unet_contract_effective" not in manifest["model"]
    assert manifest["training"]["effective_steps_per_model"] == {"unet": 1}
    assert manifest["physics"]["diagnostics_effective"] == {"spatial": {"enabled": True}}
    assert manifest["artifacts"]["split"] == {"train": [0]}
    assert manifest["artifacts"]["split_config"] == {"seed": 3, "ratios": [0.7, 0.2, 0.1]}
    assert manifest["artifacts"]["artifact_hashes"]["lock_hash"] == "abc"
    assert "compatibility" not in manifest
