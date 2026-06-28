"""Small helpers for train-time model artifact metadata."""

from __future__ import annotations

from typing import Any


def record_model_contract(extra_artifacts: dict[str, Any], model_name: str, contract: dict[str, Any]) -> None:
    model_contracts = dict(extra_artifacts.get("model_contracts", {}) or {})
    model_contracts[str(model_name)] = dict(contract)
    extra_artifacts["model_contracts"] = model_contracts


__all__ = ["record_model_contract"]
