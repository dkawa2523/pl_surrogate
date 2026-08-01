from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plasma_surrogate.benchmark.contract_summary import BenchmarkContractEmitter


@dataclass
class _ContractSamples:
    unet: list[dict[str, Any]] = field(default_factory=list)
    fno: list[dict[str, Any]] = field(default_factory=list)
    ffno: list[dict[str, Any]] = field(default_factory=list)
    coord_mlp: list[dict[str, Any]] = field(default_factory=list)
    deeponet: list[dict[str, Any]] = field(default_factory=list)
    deeponet_pod: list[dict[str, Any]] = field(default_factory=list)


def _emitter() -> BenchmarkContractEmitter:
    return BenchmarkContractEmitter(unet_contract_optional_scopes={"ffno_isolated"})


def test_common_scope_does_not_emit_unet_contract_without_active_unet() -> None:
    resolved: dict[str, Any] = {}
    samples = _ContractSamples(
        deeponet_pod=[
            {
                "model_type_effective": "deeponet_pod",
                "target_family_effective": "allvars",
                "target_vars_effective": ["ne", "phi"],
            }
        ]
    )

    _emitter().emit(
        resolved=resolved,
        train_cfg={},
        y_vars=["ne", "phi"],
        active_unet_model=None,
        eval_protocol_scope="common",
        contract_samples=samples,
    )

    assert set(resolved["model_contracts"]) == {"deeponet_pod"}


def test_common_scope_emits_fallback_contract_for_active_unet() -> None:
    resolved: dict[str, Any] = {}

    _emitter().emit(
        resolved=resolved,
        train_cfg={"unet": {"target_family": "allvars"}},
        y_vars=["ne", "phi"],
        active_unet_model="unet",
        eval_protocol_scope="common",
        contract_samples=_ContractSamples(),
    )

    assert set(resolved["model_contracts"]) == {"unet"}
    assert resolved["model_contracts"]["unet"]["target_vars_effective"] == ["ne", "phi"]
