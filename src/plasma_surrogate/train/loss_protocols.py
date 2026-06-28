"""Product loss protocol defaults for training entrypoints."""

from __future__ import annotations

import copy
from typing import Any

from plasma_surrogate.train.loss_contract import reject_removed_supervised_keys


LOSS_PROTOCOL_PLASMA_SURROGATE_V2 = "plasma_surrogate_v2"


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _validate_protocol_keys(loss_cfg: dict[str, Any]) -> None:
    reject_removed_supervised_keys(
        _dict_or_empty(loss_cfg.get("supervised")),
        message_prefix=(
            "train.loss.protocol=plasma_surrogate_v2 supports only supervised.type and supervised.mask; "
            "move research loss extensions out of the product protocol."
        ),
    )


def _v2_defaults(target_role_schema: dict[str, Any] | None) -> dict[str, Any]:
    _ = target_role_schema
    supervised: dict[str, Any] = {
        "type": "mse",
        "mask": "plasma_only",
    }
    return {
        "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V2,
        "protocol_effective": LOSS_PROTOCOL_PLASMA_SURROGATE_V2,
        "supervised": supervised,
    }


def resolve_loss_protocol(
    loss_cfg: dict[str, Any] | None,
    target_role_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expand a product loss protocol into the concrete composer config."""

    raw = copy.deepcopy(dict(loss_cfg or {}))
    protocol = str(raw.get("protocol", "")).strip().lower()
    if not protocol:
        out = copy.deepcopy(raw)
        out.setdefault("protocol_effective", "none")
        return out
    if protocol != LOSS_PROTOCOL_PLASMA_SURROGATE_V2:
        raise ValueError(f"train.loss.protocol must be {LOSS_PROTOCOL_PLASMA_SURROGATE_V2!r}; got={protocol!r}")
    _validate_protocol_keys(raw)
    merged = _deep_merge(_v2_defaults(target_role_schema), raw)
    merged["protocol"] = LOSS_PROTOCOL_PLASMA_SURROGATE_V2
    merged["protocol_effective"] = LOSS_PROTOCOL_PLASMA_SURROGATE_V2
    return merged


__all__ = ["LOSS_PROTOCOL_PLASMA_SURROGATE_V2", "resolve_loss_protocol"]
