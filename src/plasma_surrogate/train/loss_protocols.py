"""Product loss protocol defaults for training entrypoints."""

from __future__ import annotations

import copy
from typing import Any


LOSS_PROTOCOL_PLASMA_SURROGATE_V2 = "plasma_surrogate_v2"
_REMOVED_PROTOCOL_KEYS = (
    "region_weighting",
    "density_positivity_penalty",
    "density_relative_weighting",
)


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


def _schema_targets(target_role_schema: dict[str, Any] | None) -> list[dict[str, Any]]:
    targets = _dict_or_empty(target_role_schema).get("targets", [])
    if not isinstance(targets, list):
        return []
    return [dict(t) for t in targets if isinstance(t, dict) and str(t.get("id", "")).strip()]


def _positive_targets(target_role_schema: dict[str, Any] | None) -> list[str]:
    schema = _dict_or_empty(target_role_schema)
    raw = schema.get("positive_targets", [])
    if isinstance(raw, list) and raw:
        return [str(v) for v in raw if str(v).strip()]
    return [str(t["id"]) for t in _schema_targets(schema) if t.get("positive") is True]


def _default_region_for_target(entry: dict[str, Any]) -> str | None:
    explicit = str(entry.get("default_region", "")).strip()
    if explicit:
        return explicit
    role = str(entry.get("role", "")).strip().lower()
    family = str(entry.get("field_family", "")).strip().lower()
    if family in {"density", "temperature"} or role.startswith(("density", "temperature")):
        return "plasma_only"
    if family in {"electrostatic", "potential"} or "potential" in role:
        return "all_domain"
    return None


def _target_region_by_var(target_role_schema: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in _schema_targets(target_role_schema):
        target_id = str(entry["id"])
        region = _default_region_for_target(entry)
        if region:
            out[target_id] = region
    return out


def _validate_protocol_keys(loss_cfg: dict[str, Any]) -> None:
    sup = _dict_or_empty(loss_cfg.get("supervised"))
    found = [key for key in _REMOVED_PROTOCOL_KEYS if key in sup]
    if found:
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v2 does not accept deprecated supervised keys: "
            f"{found}"
        )


def _v2_defaults(target_role_schema: dict[str, Any] | None) -> dict[str, Any]:
    positive = _positive_targets(target_role_schema)
    target_regions = _target_region_by_var(target_role_schema)
    supervised: dict[str, Any] = {
        "type": "huber",
        "mask": "plasma_only",
        "region_balance": {
            "enabled": True,
            "mode": "replace",
            "weight_boundary_in": 0.6,
            "weight_plasma_mid": 0.0,
            "weight_deep_plasma": 0.4,
        },
        "spatial_consistency": {
            "enabled": True,
            "apply_region": "target_region",
            "multiscale": {"enabled": True},
        },
        "positive_penalty": {
            "enabled": bool(positive),
            "vars": positive,
            "floor": 0.0,
            "lambda": 0.01,
        },
        "relative_weighting": {"enabled": False},
    }
    if target_regions:
        supervised["target_region_by_var"] = target_regions
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
