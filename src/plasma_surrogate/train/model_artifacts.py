"""Small helpers for train-time model artifact metadata."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.input_modes import (
    EFFECTIVE_RUNTIME_METADATA_KEYS,
    merge_effective_runtime_metadata,
)


CHECKPOINT_TRAINING_PROVENANCE_KEYS = (
    "loss_protocol_effective",
    "loss_protocol_version",
    "loss_protocol_definition_hash",
    "loss_protocol_effective_config",
    "case_supervision_pack_used",
    "case_supervision_pack_storage",
    "case_supervision_channels",
)


def merge_checkpoint_dispatch_metadata(
    *,
    runtime_meta: dict[str, Any] | None,
    dispatch_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep runtime and objective provenance needed to trust a checkpoint.

    Training adapters expose additional diagnostics through ``dispatch_meta``.
    Checkpoints intentionally retain only the stable runtime contract plus the
    loss/case-supervision provenance that proves which objective produced the
    weights.  Other potentially large or model-specific diagnostics stay in
    the run manifest.
    """

    return merge_effective_runtime_metadata(
        runtime_meta=runtime_meta,
        dispatch_meta=dispatch_meta,
        keys=(*EFFECTIVE_RUNTIME_METADATA_KEYS, *CHECKPOINT_TRAINING_PROVENANCE_KEYS),
    )


def record_model_contract(extra_artifacts: dict[str, Any], model_name: str, contract: dict[str, Any]) -> None:
    model_contracts = dict(extra_artifacts.get("model_contracts", {}) or {})
    model_contracts[str(model_name)] = dict(contract)
    extra_artifacts["model_contracts"] = model_contracts


__all__ = [
    "CHECKPOINT_TRAINING_PROVENANCE_KEYS",
    "merge_checkpoint_dispatch_metadata",
    "record_model_contract",
]
