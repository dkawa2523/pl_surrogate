"""Product loss protocol defaults for training entrypoints."""

from __future__ import annotations

import copy
import math
from typing import Any

from plasma_surrogate.train.loss_contract import reject_removed_supervised_keys


LOSS_PROTOCOL_PLASMA_SURROGATE_V2 = "plasma_surrogate_v2"
SUPERVISED_TYPE_MSE = "mse"
SUPERVISED_TYPE_HUBER = "huber"
SUPERVISED_TYPES = (
    SUPERVISED_TYPE_MSE,
    SUPERVISED_TYPE_HUBER,
)
GROUP_WEIGHTING_NONE = "none"
GROUP_WEIGHTING_UNIFORM_BY_TARGET = "uniform_by_target"
GROUP_WEIGHTING_UNIFORM_BY_GROUP = "uniform_by_group"
GROUP_WEIGHTING_MODES = (
    GROUP_WEIGHTING_NONE,
    GROUP_WEIGHTING_UNIFORM_BY_TARGET,
    GROUP_WEIGHTING_UNIFORM_BY_GROUP,
)
_V2_TOP_LEVEL_KEYS = {"protocol", "supervised", "group_weighting"}
_V2_SUPERVISED_KEYS = {"type", "huber_delta", "mask"}
_V2_GROUP_WEIGHTING_KEYS = {"mode"}


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
    unknown_top = sorted(set(str(key) for key in loss_cfg) - _V2_TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"train.loss.protocol=plasma_surrogate_v2 does not support keys: {unknown_top}")

    supervised = _dict_or_empty(loss_cfg.get("supervised"))
    reject_removed_supervised_keys(
        supervised,
        message_prefix=(
            "train.loss.protocol=plasma_surrogate_v2 supports only supervised.type, "
            "supervised.huber_delta, supervised.mask, and train.loss.group_weighting; "
            "move research loss extensions out of the product protocol."
        ),
    )
    if "base" in supervised:
        raise ValueError("train.loss.supervised.base is removed; use supervised.type")
    if "delta" in supervised:
        raise ValueError("train.loss.supervised.delta is removed; use supervised.huber_delta")

    unknown_supervised = sorted(set(str(key) for key in supervised) - _V2_SUPERVISED_KEYS)
    if unknown_supervised:
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v2 supports only supervised.type, "
            f"supervised.huber_delta, and supervised.mask; got={unknown_supervised}"
        )

    supervised_type = str(supervised.get("type", SUPERVISED_TYPE_MSE)).strip().lower()
    if supervised_type not in SUPERVISED_TYPES:
        raise ValueError(
            "train.loss.supervised.type must be one of: "
            f"{', '.join(SUPERVISED_TYPES)}"
        )
    if supervised_type == SUPERVISED_TYPE_HUBER or "huber_delta" in supervised:
        raw_delta = supervised.get("huber_delta", 1.0)
        delta = float(raw_delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("train.loss.supervised.huber_delta must be finite and > 0")

    group_weighting = _dict_or_empty(loss_cfg.get("group_weighting"))
    unknown_group_weighting = sorted(set(str(key) for key in group_weighting) - _V2_GROUP_WEIGHTING_KEYS)
    if unknown_group_weighting:
        raise ValueError(
            "train.loss.group_weighting supports only group_weighting.mode; "
            f"got={unknown_group_weighting}"
        )
    mode = str(group_weighting.get("mode", GROUP_WEIGHTING_NONE)).strip().lower()
    if mode not in GROUP_WEIGHTING_MODES:
        raise ValueError(
            "train.loss.group_weighting.mode must be one of: "
            f"{', '.join(GROUP_WEIGHTING_MODES)}"
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
        "group_weighting": {"mode": GROUP_WEIGHTING_NONE},
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


__all__ = [
    "GROUP_WEIGHTING_MODES",
    "GROUP_WEIGHTING_NONE",
    "GROUP_WEIGHTING_UNIFORM_BY_GROUP",
    "GROUP_WEIGHTING_UNIFORM_BY_TARGET",
    "LOSS_PROTOCOL_PLASMA_SURROGATE_V2",
    "SUPERVISED_TYPES",
    "SUPERVISED_TYPE_HUBER",
    "SUPERVISED_TYPE_MSE",
    "resolve_loss_protocol",
]
