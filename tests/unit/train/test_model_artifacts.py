from __future__ import annotations

import pytest

from plasma_surrogate.train.model_artifacts import merge_checkpoint_dispatch_metadata


def test_checkpoint_dispatch_metadata_keeps_objective_provenance_only() -> None:
    runtime = {
        "input_mode_effective": "table_only",
        "structure_adapter_mode_effective": "none",
    }
    dispatch = {
        **runtime,
        "loss_protocol_effective": "plasma_surrogate_v3",
        "loss_protocol_version": 3,
        "loss_protocol_definition_hash": "loss-hash",
        "loss_protocol_effective_config": '{"protocol":"plasma_surrogate_v3"}',
        "case_supervision_pack_used": True,
        "case_supervision_pack_storage": "case_structure_feature_pack.npz",
        "case_supervision_channels": ["mask_plasma", "distance_any", "part_sdf_nearest"],
        "effective_steps": 123,
        "model_contracts": {"large": {"payload": "manifest-only"}},
    }

    merged = merge_checkpoint_dispatch_metadata(runtime_meta=runtime, dispatch_meta=dispatch)

    assert merged["input_mode_effective"] == "table_only"
    assert merged["loss_protocol_effective"] == "plasma_surrogate_v3"
    assert merged["loss_protocol_version"] == 3
    assert merged["loss_protocol_definition_hash"] == "loss-hash"
    assert merged["case_supervision_pack_used"] is True
    assert merged["case_supervision_channels"] == [
        "mask_plasma",
        "distance_any",
        "part_sdf_nearest",
    ]
    assert "effective_steps" not in merged
    assert "model_contracts" not in merged


def test_checkpoint_dispatch_metadata_rejects_runtime_conflicts() -> None:
    with pytest.raises(ValueError, match="runtime metadata mismatch"):
        merge_checkpoint_dispatch_metadata(
            runtime_meta={"input_mode_effective": "table_only"},
            dispatch_meta={"input_mode_effective": "table_plus_structure"},
        )
