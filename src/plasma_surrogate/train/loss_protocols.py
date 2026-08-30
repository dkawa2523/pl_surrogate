"""Product loss protocol defaults for training entrypoints."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from plasma_surrogate.core.boundary_distance import normalize_boundary_distance_channels
from plasma_surrogate.train.loss_contract import reject_removed_supervised_keys


LOSS_PROTOCOL_PLASMA_SURROGATE_V2 = "plasma_surrogate_v2"
LOSS_PROTOCOL_PLASMA_SURROGATE_V3 = "plasma_surrogate_v3"
SUPERVISED_TYPE_MSE = "mse"
SUPERVISED_TYPE_HUBER = "huber"
SUPERVISED_TYPES = (
    SUPERVISED_TYPE_MSE,
    SUPERVISED_TYPE_HUBER,
)
GROUP_WEIGHTING_NONE = "none"
GROUP_WEIGHTING_UNIFORM_BY_TARGET = "uniform_by_target"
GROUP_WEIGHTING_UNIFORM_BY_GROUP = "uniform_by_group"
GROUP_WEIGHTING_WEIGHTED_BY_GROUP = "weighted_by_group"
GROUP_WEIGHTING_MODES = (
    GROUP_WEIGHTING_NONE,
    GROUP_WEIGHTING_UNIFORM_BY_TARGET,
    GROUP_WEIGHTING_UNIFORM_BY_GROUP,
    GROUP_WEIGHTING_WEIGHTED_BY_GROUP,
)
_V2_TOP_LEVEL_KEYS = {"protocol", "protocol_effective", "supervised", "group_weighting"}
_V2_SUPERVISED_KEYS = {"type", "huber_delta", "mask"}
_V2_GROUP_WEIGHTING_KEYS = {"mode", "weights"}
_V3_TOP_LEVEL_KEYS = _V2_TOP_LEVEL_KEYS | {"target_role_schema"}
_V3_SUPERVISED_KEYS = {
    "type",
    "huber_delta",
    "mask",
    "normalization",
    "sample_mean_weight_denominator",
    "spatial",
    "physical_weighting",
    "derived_qoi",
}
_V3_SPATIAL_KEYS = {
    "boundary_band_px",
    "boundary_distance_channels",
    "boundary_weight",
    "gradient_spacing",
    "gradient_normalization",
    "gradient_epsilon",
    "gradient_weight",
    "multiscale_scales",
    "multiscale_weight",
    "targets",
}
_V3_PHYSICAL_WEIGHTING_KEYS = {
    "axisymmetric_volume",
    "density_source",
    "density_weighted_targets",
}
_SUPERVISED_MASK_MODES = {"none", "plasma_only"}


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


def _validate_group_weighting(loss_cfg: dict[str, Any]) -> None:
    group_weighting = _dict_or_empty(loss_cfg.get("group_weighting"))
    unknown_group_weighting = sorted(set(str(key) for key in group_weighting) - _V2_GROUP_WEIGHTING_KEYS)
    if unknown_group_weighting:
        raise ValueError(
            "train.loss.group_weighting supports only group_weighting.mode and group_weighting.weights; "
            f"got={unknown_group_weighting}"
        )
    mode = str(group_weighting.get("mode", GROUP_WEIGHTING_NONE)).strip().lower()
    if mode not in GROUP_WEIGHTING_MODES:
        raise ValueError(
            "train.loss.group_weighting.mode must be one of: "
            f"{', '.join(GROUP_WEIGHTING_MODES)}"
        )
    weights = _dict_or_empty(group_weighting.get("weights"))
    if weights and mode != GROUP_WEIGHTING_WEIGHTED_BY_GROUP:
        raise ValueError("train.loss.group_weighting.weights is supported only when mode=weighted_by_group")
    if mode == GROUP_WEIGHTING_WEIGHTED_BY_GROUP:
        if not weights:
            raise ValueError("train.loss.group_weighting.mode=weighted_by_group requires weights")
        for name, raw_weight in weights.items():
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"train.loss.group_weighting.weights[{name}] must be finite and > 0")


def _supervised_mask_mode(supervised: dict[str, Any], *, default: str = "plasma_only") -> str:
    mode = str(supervised.get("mask", default)).strip().lower()
    if mode not in _SUPERVISED_MASK_MODES:
        raise ValueError(
            "train.loss.supervised.mask must be one of: none, plasma_only; "
            f"got={supervised.get('mask')!r}"
        )
    return mode


def _validate_protocol_v2_keys(loss_cfg: dict[str, Any]) -> None:
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

    _supervised_mask_mode(supervised)
    _validate_group_weighting(loss_cfg)


def _finite_non_negative(raw: Any, *, key_name: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key_name} must be finite and >= 0")
    return value


def _validate_protocol_v3_keys(loss_cfg: dict[str, Any]) -> None:
    unknown_top = sorted(set(str(key) for key in loss_cfg) - _V3_TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"train.loss.protocol=plasma_surrogate_v3 does not support keys: {unknown_top}")
    if "target_role_schema" in loss_cfg and not isinstance(loss_cfg["target_role_schema"], dict):
        raise ValueError("train.loss.target_role_schema must be a mapping")

    supervised = _dict_or_empty(loss_cfg.get("supervised"))
    reject_removed_supervised_keys(
        supervised,
        message_prefix=(
            "train.loss.protocol=plasma_surrogate_v3 uses the canonical supervised.spatial contract;"
        ),
    )
    if "base" in supervised:
        raise ValueError("train.loss.supervised.base is removed; use supervised.type")
    if "delta" in supervised:
        raise ValueError("train.loss.supervised.delta is removed; use supervised.huber_delta")
    unknown_supervised = sorted(set(str(key) for key in supervised) - _V3_SUPERVISED_KEYS)
    if unknown_supervised:
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v3 contains unsupported supervised keys: "
            f"{unknown_supervised}"
        )

    supervised_type = str(supervised.get("type", SUPERVISED_TYPE_HUBER)).strip().lower()
    if supervised_type not in SUPERVISED_TYPES:
        raise ValueError(
            "train.loss.supervised.type must be one of: "
            f"{', '.join(SUPERVISED_TYPES)}"
        )
    raw_delta = supervised.get("huber_delta", 1.0)
    delta = float(raw_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("train.loss.supervised.huber_delta must be finite and > 0")

    normalization = str(supervised.get("normalization", "sample_mean")).strip().lower()
    if normalization not in {"pixel_mean", "sample_mean", "none"}:
        raise ValueError("train.loss.supervised.normalization must be one of: pixel_mean, sample_mean, none")
    denominator = str(supervised.get("sample_mean_weight_denominator", "weighted")).strip().lower()
    if denominator not in {"weighted", "count"}:
        raise ValueError(
            "train.loss.supervised.sample_mean_weight_denominator must be one of: weighted, count"
        )

    spatial = _dict_or_empty(supervised.get("spatial"))
    unknown_spatial = sorted(set(str(key) for key in spatial) - _V3_SPATIAL_KEYS)
    if unknown_spatial:
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v3 contains unsupported supervised.spatial keys: "
            f"{unknown_spatial}"
        )
    physical = _dict_or_empty(supervised.get("physical_weighting"))
    unknown_physical = sorted(set(str(key) for key in physical) - _V3_PHYSICAL_WEIGHTING_KEYS)
    if unknown_physical:
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v3 contains unsupported "
            f"supervised.physical_weighting keys: {unknown_physical}"
        )
    density_targets = physical.get("density_weighted_targets", [])
    if not isinstance(density_targets, (list, tuple)) or any(not str(name).strip() for name in density_targets):
        raise ValueError(
            "train.loss.supervised.physical_weighting.density_weighted_targets must be a list of names"
        )
    if density_targets and not str(physical.get("density_source", "ne")).strip():
        raise ValueError("train.loss.supervised.physical_weighting.density_source must be non-empty")
    _finite_non_negative(
        spatial.get("gradient_weight", 0.10),
        key_name="train.loss.supervised.spatial.gradient_weight",
    )
    _finite_non_negative(
        spatial.get("multiscale_weight", 0.05),
        key_name="train.loss.supervised.spatial.multiscale_weight",
    )
    boundary_weight = _finite_non_negative(
        spatial.get("boundary_weight", 0.25),
        key_name="train.loss.supervised.spatial.boundary_weight",
    )
    mask_mode = _supervised_mask_mode(supervised)
    if boundary_weight > 0.0 and mask_mode != "plasma_only":
        raise ValueError(
            "train.loss.protocol=plasma_surrogate_v3 with supervised.spatial.boundary_weight>0 "
            "requires supervised.mask=plasma_only"
        )
    boundary_band = float(spatial.get("boundary_band_px", 2.0))
    if not math.isfinite(boundary_band) or boundary_band <= 0.0:
        raise ValueError("train.loss.supervised.spatial.boundary_band_px must be finite and > 0")
    normalize_boundary_distance_channels(
        spatial.get("boundary_distance_channels"),
        key="train.loss.supervised.spatial.boundary_distance_channels",
    )

    raw_scales = spatial.get("multiscale_scales", [2, 4])
    if not isinstance(raw_scales, (list, tuple)) or not raw_scales:
        raise ValueError("train.loss.supervised.spatial.multiscale_scales must be a non-empty list")
    scales: list[int] = []
    for raw_scale in raw_scales:
        scale = int(raw_scale)
        if isinstance(raw_scale, bool) or float(raw_scale) != float(scale) or scale < 2:
            raise ValueError(
                "train.loss.supervised.spatial.multiscale_scales must contain unique integers >= 2"
            )
        scales.append(scale)
    if len(set(scales)) != len(scales):
        raise ValueError("train.loss.supervised.spatial.multiscale_scales must contain unique integers >= 2")

    raw_spacing = spatial.get("gradient_spacing", [1.0, 1.0])
    if not isinstance(raw_spacing, (list, tuple)) or len(raw_spacing) != 2:
        raise ValueError("train.loss.supervised.spatial.gradient_spacing must be [dy, dx]")
    spacing = [float(value) for value in raw_spacing]
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("train.loss.supervised.spatial.gradient_spacing values must be finite and > 0")
    gradient_normalization = str(spatial.get("gradient_normalization", "none")).strip().lower()
    if gradient_normalization not in {"none", "target_rms"}:
        raise ValueError(
            "train.loss.supervised.spatial.gradient_normalization must be one of: none, target_rms"
        )
    gradient_epsilon = float(spatial.get("gradient_epsilon", 0.05))
    if not math.isfinite(gradient_epsilon) or gradient_epsilon <= 0.0:
        raise ValueError("train.loss.supervised.spatial.gradient_epsilon must be finite and > 0")

    _validate_group_weighting(loss_cfg)


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


def _v3_defaults(target_role_schema: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
        "protocol_effective": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
        "supervised": {
            "type": SUPERVISED_TYPE_HUBER,
            "huber_delta": 1.0,
            "mask": "plasma_only",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "weighted",
            "spatial": {
                "gradient_weight": 0.10,
                "gradient_spacing": [1.0, 1.0],
                "multiscale_weight": 0.05,
                "multiscale_scales": [2, 4],
                "boundary_weight": 0.25,
                "boundary_band_px": 2.0,
                "boundary_distance_channels": ["distance_any"],
            },
        },
        "group_weighting": {"mode": GROUP_WEIGHTING_UNIFORM_BY_GROUP},
        "target_role_schema": copy.deepcopy(dict(target_role_schema or {})),
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
    supported = {LOSS_PROTOCOL_PLASMA_SURROGATE_V2, LOSS_PROTOCOL_PLASMA_SURROGATE_V3}
    if protocol not in supported:
        raise ValueError(
            "train.loss.protocol must be one of: "
            f"{', '.join(sorted(supported))}; got={protocol!r}"
        )
    if protocol == LOSS_PROTOCOL_PLASMA_SURROGATE_V2:
        _validate_protocol_v2_keys(raw)
        merged = _deep_merge(_v2_defaults(target_role_schema), raw)
    else:
        _validate_protocol_v3_keys(raw)
        merged = _deep_merge(_v3_defaults(target_role_schema), raw)
        merged_spatial = dict(dict(merged["supervised"]).get("spatial", {}) or {})
        merged_spatial["boundary_distance_channels"] = list(
            normalize_boundary_distance_channels(
                merged_spatial.get("boundary_distance_channels"),
                key="train.loss.supervised.spatial.boundary_distance_channels",
            )
        )
        merged["supervised"]["spatial"] = merged_spatial
        if target_role_schema is not None:
            # The schema supplied by the dataset contract is authoritative over
            # a stale schema carried by an already-resolved loss dictionary.
            merged["target_role_schema"] = copy.deepcopy(dict(target_role_schema))
    merged["protocol"] = protocol
    merged["protocol_effective"] = protocol
    return merged


def loss_protocol_metadata(loss_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the immutable definition carried by run and checkpoint artifacts."""

    effective = copy.deepcopy(dict(loss_cfg or {}))
    protocol = str(
        effective.get("protocol_effective", effective.get("protocol", "none"))
    ).strip().lower() or "none"
    version = {
        LOSS_PROTOCOL_PLASMA_SURROGATE_V2: 2,
        LOSS_PROTOCOL_PLASMA_SURROGATE_V3: 3,
    }.get(protocol, 0)
    effective_json = json.dumps(
        effective,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    definition = json.dumps(
        {
            "protocol": protocol,
            "version": int(version),
            "effective_config": effective,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return {
        "loss_protocol_effective": protocol,
        "loss_protocol_version": int(version),
        "loss_protocol_definition_hash": hashlib.sha256(
            definition.encode("utf-8")
        ).hexdigest(),
        "loss_protocol_effective_config": effective_json,
    }


__all__ = [
    "GROUP_WEIGHTING_MODES",
    "GROUP_WEIGHTING_NONE",
    "GROUP_WEIGHTING_UNIFORM_BY_GROUP",
    "GROUP_WEIGHTING_UNIFORM_BY_TARGET",
    "GROUP_WEIGHTING_WEIGHTED_BY_GROUP",
    "LOSS_PROTOCOL_PLASMA_SURROGATE_V2",
    "LOSS_PROTOCOL_PLASMA_SURROGATE_V3",
    "SUPERVISED_TYPES",
    "SUPERVISED_TYPE_HUBER",
    "SUPERVISED_TYPE_MSE",
    "loss_protocol_metadata",
    "resolve_loss_protocol",
]
