"""Surrogate quality-score components."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups
from plasma_surrogate.preprocessing.scalers import transform_target_with_artifact

from plasma_surrogate.eval.spatial_metrics import _field_as_nhw, _mask_as_nhw


QUALITY_SCORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "nrmse": 0.45,
    "boundary": 0.20,
    "continuity": 0.15,
    "physics": 0.15,
    "sign": 0.05,
}

QUALITY_PROTOCOL_LEGACY_COMPOSITE_V1 = "legacy_composite_v1"
QUALITY_PROTOCOL_SPATIAL_HUBER_V1 = "spatial_huber_case_balanced_v1"
QUALITY_PROTOCOL_SPATIAL_HUBER_V2 = "spatial_huber_case_balanced_v2"

_SPATIAL_HUBER_DEFAULTS: dict[str, Any] = {
    "mode": "spatial_huber",
    "delta": 1.0,
    "boundary_alpha": 2.0,
    "boundary_band_px": 2.0,
    "avgpool_lambda": 0.05,
    "gradient_lambda": 0.10,
    "multiscale_scales": [2, 4],
    "p90_weight": 0.25,
    "worst_weight": 0.10,
    "target_aggregation": "uniform_by_group",
}
_SPATIAL_HUBER_KEYS = set(_SPATIAL_HUBER_DEFAULTS)
_SPATIAL_HUBER_SCALE_ALIASES = ("pool_scale", "pool_kernel")
_SPATIAL_HUBER_TARGET_AGGREGATIONS = {"uniform_by_group", "uniform_by_target"}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _quality_scale(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"benchmark.eval.quality_score.{field} must contain integers >= 1")
    try:
        value_f = float(value)
        value_i = int(value_f)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"benchmark.eval.quality_score.{field} must contain integers >= 1"
        ) from exc
    if not math.isfinite(value_f) or value_f != float(value_i) or value_i < 1:
        raise ValueError(f"benchmark.eval.quality_score.{field} must contain integers >= 1")
    return value_i


def resolve_quality_score_protocol(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve one versioned, hashable quality-score definition.

    The returned ``effective_config`` contains every formula-affecting default.
    Historical ``pool_scale``/``pool_kernel`` inputs are accepted only as
    explicit aliases for a one-element ``multiscale_scales`` list.
    """

    raw = dict(cfg or {})
    mode = str(raw.get("mode", "")).strip().lower()
    if mode in {"", "legacy_composite"}:
        unknown = sorted(set(raw) - {"mode", "weights"})
        if unknown:
            raise ValueError(f"benchmark.eval.quality_score has unsupported keys: {unknown}")
        raw_weights = dict(raw.get("weights", {}) or {})
        unknown_weights = sorted(set(raw_weights) - set(QUALITY_SCORE_DEFAULT_WEIGHTS))
        if unknown_weights:
            raise ValueError(
                "benchmark.eval.quality_score.weights has unsupported keys: "
                f"{unknown_weights}"
            )
        weights = dict(QUALITY_SCORE_DEFAULT_WEIGHTS)
        weights.update({str(key): float(value) for key, value in raw_weights.items()})
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("benchmark.eval.quality_score.weights must be finite and >= 0")
        protocol = QUALITY_PROTOCOL_LEGACY_COMPOSITE_V1
        version = 1
        effective: dict[str, Any] = {"mode": "legacy_composite", "weights": weights}
    elif mode == "spatial_huber":
        alias_keys = [key for key in _SPATIAL_HUBER_SCALE_ALIASES if key in raw]
        if len(alias_keys) > 1:
            raise ValueError(
                "benchmark.eval.quality_score accepts only one legacy scale alias; "
                f"received {alias_keys}"
            )
        if alias_keys and "multiscale_scales" in raw:
            raise ValueError(
                "benchmark.eval.quality_score cannot combine multiscale_scales with "
                f"legacy aliases {alias_keys}"
            )
        unknown = sorted(set(raw) - _SPATIAL_HUBER_KEYS - set(_SPATIAL_HUBER_SCALE_ALIASES))
        if unknown:
            raise ValueError(f"benchmark.eval.quality_score spatial_huber has unsupported keys: {unknown}")
        effective = dict(_SPATIAL_HUBER_DEFAULTS)
        for key in _SPATIAL_HUBER_KEYS - {"mode", "multiscale_scales", "target_aggregation"}:
            if key in raw:
                effective[key] = float(raw[key])
        target_aggregation = str(raw.get("target_aggregation", effective["target_aggregation"])).strip().lower()
        if target_aggregation not in _SPATIAL_HUBER_TARGET_AGGREGATIONS:
            raise ValueError(
                "benchmark.eval.quality_score.target_aggregation must be one of: "
                "uniform_by_group, uniform_by_target"
            )
        effective["target_aggregation"] = target_aggregation
        if alias_keys:
            alias = alias_keys[0]
            effective["multiscale_scales"] = [_quality_scale(raw[alias], field=alias)]
        elif "multiscale_scales" in raw:
            scales_raw = raw["multiscale_scales"]
            if not isinstance(scales_raw, (list, tuple)) or not scales_raw:
                raise ValueError(
                    "benchmark.eval.quality_score.multiscale_scales must be a non-empty list"
                )
            effective["multiscale_scales"] = [
                _quality_scale(value, field="multiscale_scales") for value in scales_raw
            ]
        scales = list(effective["multiscale_scales"])
        if any(value < 1 for value in scales) or len(set(scales)) != len(scales):
            raise ValueError(
                "benchmark.eval.quality_score.multiscale_scales must contain unique integers >= 1"
            )
        positive_keys = ("delta", "boundary_band_px")
        nonnegative_keys = (
            "boundary_alpha",
            "avgpool_lambda",
            "gradient_lambda",
            "p90_weight",
            "worst_weight",
        )
        if any(
            not math.isfinite(float(effective[key])) or float(effective[key]) <= 0.0
            for key in positive_keys
        ):
            raise ValueError(
                "benchmark.eval.quality_score delta and boundary_band_px must be finite and > 0"
            )
        if any(
            not math.isfinite(float(effective[key])) or float(effective[key]) < 0.0
            for key in nonnegative_keys
        ):
            raise ValueError(
                "benchmark.eval.quality_score weights must be finite and >= 0"
            )
        protocol = QUALITY_PROTOCOL_SPATIAL_HUBER_V2
        version = 2
    else:
        raise ValueError(
            "benchmark.eval.quality_score.mode must be one of: legacy_composite, spatial_huber"
        )

    definition = {
        "protocol": protocol,
        "version": version,
        "effective_config": effective,
    }
    definition_json = _canonical_json(definition)
    return {
        **definition,
        "effective_config_json": _canonical_json(effective),
        "definition_hash": hashlib.sha256(definition_json.encode("utf-8")).hexdigest(),
    }


def quality_score_protocol_metadata(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_score_protocol": str(protocol["protocol"]),
        "quality_score_protocol_version": int(protocol["version"]),
        "quality_score_definition_hash": str(protocol["definition_hash"]),
        "quality_score_effective_config": str(protocol["effective_config_json"]),
    }


def finite_mean(values: list[float], *, default: float = float("nan")) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def zero_if_nonfinite(value: float | None) -> float:
    if value is None:
        return 0.0
    value_f = float(value)
    if not np.isfinite(value_f):
        return 0.0
    return value_f


def inf_if_nonfinite(value: float | None) -> float:
    if value is None:
        return float("inf")
    value_f = float(value)
    if not np.isfinite(value_f):
        return float("inf")
    return value_f


def abs_log_ratio(value: float | None) -> float:
    value_f = inf_if_nonfinite(value)
    if not np.isfinite(value_f):
        return float("inf")
    if value_f <= 0.0:
        return 0.0
    return float(abs(np.log(max(value_f, 1.0e-12))))


def _quality_score_weights(cfg: dict[str, Any] | None) -> dict[str, float]:
    raw_weights = dict(dict(cfg or {}).get("weights", {}) or {})
    weights = dict(QUALITY_SCORE_DEFAULT_WEIGHTS)
    for key in weights:
        if key in raw_weights:
            weights[key] = float(raw_weights[key])
    return weights


def build_surrogate_quality_components(
    *,
    mean_nrmse_plasma_by_target: float,
    boundary_to_deep_rmse_ratio_mean: float,
    continuity_grad_ratio: float,
    continuity_lap_ratio: float,
    poisson_residual_penalty: float,
    boundary_residual_penalty: float,
    positive_target_negative_ratio_penalty: float,
    quality_score_cfg: dict[str, Any] | None,
) -> dict[str, float]:
    weights = _quality_score_weights(quality_score_cfg)
    nrmse_component = inf_if_nonfinite(mean_nrmse_plasma_by_target)
    boundary_ratio = zero_if_nonfinite(boundary_to_deep_rmse_ratio_mean)
    boundary_component = float(max(0.0, boundary_ratio - 1.0)) if boundary_ratio > 0.0 else 0.0
    continuity_component = float(
        0.5 * (abs_log_ratio(continuity_grad_ratio) + abs_log_ratio(continuity_lap_ratio))
    )
    physics_component = float(
        0.5
        * (
            np.log1p(abs(inf_if_nonfinite(poisson_residual_penalty)))
            + np.log1p(abs(inf_if_nonfinite(boundary_residual_penalty)))
        )
    )
    sign_component = inf_if_nonfinite(positive_target_negative_ratio_penalty)
    total = float(
        weights["nrmse"] * nrmse_component
        + weights["boundary"] * boundary_component
        + weights["continuity"] * continuity_component
        + weights["physics"] * physics_component
        + weights["sign"] * sign_component
    )
    return {
        "surrogate_quality_score": total,
        "score_nrmse_component": nrmse_component,
        "score_boundary_component": boundary_component,
        "score_continuity_component": continuity_component,
        "score_physics_component": physics_component,
        "score_sign_component": sign_component,
    }


def _scaler_affine_for_var(target_scalers: dict[str, Any] | None, name: str) -> tuple[float, float] | None:
    if not target_scalers or name not in target_scalers:
        return None
    scaler = dict(target_scalers.get(name, {}) or {})
    if str(scaler.get("type", "none")).strip().lower() != "zscore":
        return None
    mean_raw = scaler.get("mean", 0.0)
    std_raw = scaler.get("std", 1.0)
    mean = float(mean_raw[0] if isinstance(mean_raw, list) and mean_raw else mean_raw)
    std = float(std_raw[0] if isinstance(std_raw, list) and std_raw else std_raw)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1.0e-12:
        return None
    return mean, std


def _huber_map(err: np.ndarray, *, delta: float) -> np.ndarray:
    d = float(max(delta, 1.0e-8))
    abs_err = np.abs(np.asarray(err, dtype=np.float32))
    return np.where(abs_err <= d, 0.5 * abs_err * abs_err, d * (abs_err - 0.5 * d)).astype(np.float32)


def _avg_pool2d_nhw(arr: np.ndarray, scale: int) -> np.ndarray:
    s = int(max(int(scale), 1))
    a = np.asarray(arr, dtype=np.float32)
    if s <= 1:
        return a
    h = int(a.shape[-2])
    w = int(a.shape[-1])
    out_h = (h + s - 1) // s
    out_w = (w + s - 1) // s
    padded = np.pad(
        a,
        ((0, 0), (0, out_h * s - h), (0, out_w * s - w)),
        mode="constant",
        constant_values=0.0,
    )
    return padded.reshape(padded.shape[0], out_h, s, out_w, s).mean(axis=(2, 4)).astype(np.float32)


def _masked_avg_pool2d_nhw(arr: np.ndarray, mask: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    weighted = _avg_pool2d_nhw(np.asarray(arr, dtype=np.float32) * np.asarray(mask, dtype=np.float32), scale)
    fraction = _avg_pool2d_nhw(np.asarray(mask, dtype=np.float32), scale)
    pooled = weighted / np.maximum(fraction, 1.0e-12)
    return pooled.astype(np.float32), fraction.astype(np.float32)


def _case_percentiles(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(finite)), float(np.percentile(finite, 90.0)), float(np.max(finite))


def _resolve_quality_target_groups(
    *,
    target_vars: list[str],
    target_role_schema: dict[str, Any] | None,
) -> dict[str, TargetGroup]:
    schema = dict(target_role_schema or {})
    raw_targets = schema.get("targets", [])
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(
            "benchmark.eval.quality_score.target_aggregation=uniform_by_group "
            "requires target_role_schema.targets"
        )
    selected = set(str(name) for name in target_vars)
    filtered_schema = dict(schema)
    filtered_schema["targets"] = [
        dict(raw)
        for raw in raw_targets
        if isinstance(raw, dict) and str(raw.get("id", "")).strip() in selected
    ]
    groups = resolve_target_groups(
        output_vars=[str(name) for name in target_vars],
        target_role_schema=filtered_schema,
        strict=True,
    )
    if not groups:
        raise ValueError(
            "benchmark.eval.quality_score.target_aggregation=uniform_by_group "
            "resolved no target groups"
        )
    missing_families = [
        str(target)
        for group in groups.values()
        if group.field_family is None
        for target in group.targets
    ]
    if missing_families:
        raise ValueError(
            "benchmark.eval.quality_score.target_aggregation=uniform_by_group "
            "requires field_family metadata for every scored target; "
            f"missing={missing_families}"
        )
    return groups


def _aggregate_quality_target_values(
    values_by_var: dict[str, float],
    *,
    target_vars: list[str],
    target_aggregation: str,
    groups: dict[str, TargetGroup],
) -> tuple[float, dict[str, float]]:
    missing = [name for name in target_vars if name not in values_by_var]
    if missing:
        raise ValueError(
            "spatial_huber quality could not evaluate every configured target; "
            f"missing={missing}"
        )
    if target_aggregation == "uniform_by_target":
        return finite_mean(
            [values_by_var[name] for name in target_vars],
            default=float("nan"),
        ), {}

    by_group: dict[str, float] = {}
    for group_name, group in groups.items():
        by_group[str(group_name)] = finite_mean(
            [values_by_var[str(name)] for name in group.targets],
            default=float("nan"),
        )
    return finite_mean(list(by_group.values()), default=float("nan")), by_group


def build_spatial_huber_quality_components(
    *,
    true_eval: dict[str, np.ndarray] | None,
    pred_eval: dict[str, np.ndarray],
    mask_plasma: np.ndarray | None,
    distance_any: np.ndarray | None,
    target_vars: list[str],
    target_scalers: dict[str, Any] | None,
    target_transforms: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    if true_eval is None:
        return {}
    protocol = resolve_quality_score_protocol(cfg)
    if str(protocol["protocol"]) != QUALITY_PROTOCOL_SPATIAL_HUBER_V2:
        return {}
    score_cfg = dict(protocol["effective_config"])
    delta = float(score_cfg.get("delta", 1.0))
    boundary_alpha = float(score_cfg.get("boundary_alpha", 2.0))
    boundary_band_px = float(score_cfg.get("boundary_band_px", 2.0))
    avgpool_lambda = float(score_cfg.get("avgpool_lambda", 0.05))
    gradient_lambda = float(score_cfg.get("gradient_lambda", 0.10))
    raw_scales = score_cfg.get("multiscale_scales", [2, 4])
    pool_scales = tuple(int(max(int(scale), 1)) for scale in raw_scales)
    p90_weight = float(score_cfg.get("p90_weight", 0.25))
    worst_weight = float(score_cfg.get("worst_weight", 0.10))
    target_aggregation = str(score_cfg["target_aggregation"])
    target_names = [str(name) for name in target_vars]
    target_groups = (
        _resolve_quality_target_groups(
            target_vars=target_names,
            target_role_schema=target_role_schema,
        )
        if target_aggregation == "uniform_by_group"
        else {}
    )
    distance_nhw: np.ndarray | None = None
    if boundary_alpha > 0.0:
        if distance_any is None:
            raise ValueError(
                "benchmark.eval.quality_score spatial_huber with boundary_alpha>0 "
                "requires distance_any"
            )
        distance_nhw = _field_as_nhw(distance_any, key="distance_any")
    base_by_var: dict[str, float] = {}
    pool_by_var: dict[str, float] = {}
    gradient_by_var: dict[str, float] = {}
    total_by_var: dict[str, float] = {}
    summaries_by_var: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name in target_names:
        if name not in true_eval or name not in pred_eval:
            raise ValueError(
                "benchmark.eval.quality_score is missing a configured target in true_eval or pred_eval: "
                f"target={name}"
            )
        true_arr = _field_as_nhw(true_eval[name], key=f"true_eval[{name}]")
        pred_arr = _field_as_nhw(pred_eval[name], key=f"pred_eval[{name}]")
        if true_arr.shape != pred_arr.shape:
            raise ValueError(
                "benchmark.eval.quality_score prediction/target shape mismatch: "
                f"target={name}, true={true_arr.shape}, pred={pred_arr.shape}"
            )
        mask = _mask_as_nhw(
            mask_plasma,
            n_cases=int(true_arr.shape[0]),
            shape=tuple(true_arr.shape[-2:]),
        )
        true_for_transform = np.asarray(true_arr, dtype=np.float32).copy()
        pred_for_transform = np.asarray(pred_arr, dtype=np.float32).copy()
        for case_idx in range(int(true_arr.shape[0])):
            active = np.asarray(mask[case_idx], dtype=bool)
            if not np.any(active):
                raise ValueError(
                    "benchmark.eval.quality_score resolved an empty active mask: "
                    f"target={name}, case={case_idx}"
                )
            invalid_active = active & (
                ~np.isfinite(true_arr[case_idx]) | ~np.isfinite(pred_arr[case_idx])
            )
            if np.any(invalid_active):
                raise ValueError(
                    "benchmark.eval.quality_score contains non-finite truth/prediction "
                    f"inside the active mask: target={name}, case={case_idx}, "
                    f"count={int(np.sum(invalid_active))}"
                )
            # Values outside the supervised plasma region are not part of the
            # score.  Fill them with one valid in-domain value before applying
            # log transforms so masked NaN/Inf never leak into arithmetic.
            fill_value = float(true_arr[case_idx][active][0])
            true_for_transform[case_idx][~active] = fill_value
            pred_for_transform[case_idx][~active] = fill_value
        dist: np.ndarray | None = None
        if distance_nhw is not None:
            dist = distance_nhw
            if int(dist.shape[0]) == 1 and int(true_arr.shape[0]) > 1:
                dist = np.repeat(dist, int(true_arr.shape[0]), axis=0)
            if dist.shape != true_arr.shape:
                raise ValueError(
                    "benchmark.eval.quality_score distance_any must align with every scored "
                    f"target [N,H,W]; target={name}, expected={true_arr.shape}, got={dist.shape}"
                )
            invalid_distance = np.asarray(mask, dtype=bool) & ~np.isfinite(dist)
            if np.any(invalid_distance):
                raise ValueError(
                    "benchmark.eval.quality_score distance_any is non-finite inside "
                    f"the active mask: target={name}, count={int(np.sum(invalid_distance))}"
                )
        transform_spec = dict((target_transforms or {}).get(name, {}) or {})
        scaler_spec = dict((target_scalers or {}).get(name, {}) or {})
        if transform_spec:
            true_norm = transform_target_with_artifact(
                true_for_transform,
                var=name,
                scaler_artifact=scaler_spec,
                transform_artifact=transform_spec,
            )
            pred_norm = transform_target_with_artifact(
                pred_for_transform,
                var=name,
                scaler_artifact=scaler_spec,
                transform_artifact=transform_spec,
            )
        else:
            affine = _scaler_affine_for_var(target_scalers, name)
            if affine is None:
                active_for_std = _mask_as_nhw(
                    mask_plasma,
                    n_cases=int(true_arr.shape[0]),
                    shape=tuple(true_arr.shape[-2:]),
                )
                vals = true_arr[np.asarray(active_for_std, dtype=bool)]
                std = float(np.std(vals)) if vals.size > 1 else float("nan")
                mean = float(np.mean(vals)) if vals.size > 0 else 0.0
                if not np.isfinite(std) or std <= 1.0e-12:
                    continue
            else:
                mean, std = affine
            true_norm = ((true_for_transform - float(mean)) / float(std)).astype(np.float32)
            pred_norm = ((pred_for_transform - float(mean)) / float(std)).astype(np.float32)
        weight = mask.astype(np.float32)
        if boundary_alpha > 0.0:
            assert dist is not None
            weight = weight * (
                1.0
                + float(boundary_alpha)
                * (np.abs(dist) <= float(boundary_band_px)).astype(np.float32)
            )
        case_base: list[float] = []
        case_pool: list[float] = []
        case_gradient: list[float] = []
        case_total: list[float] = []
        case_physical_l2: list[float] = []
        case_physical_gradient: list[float] = []
        case_mean_error: list[float] = []
        case_bound_fraction: list[float] = []
        for case_idx in range(int(true_arr.shape[0])):
            active = np.asarray(mask[case_idx], dtype=bool)
            if boundary_alpha > 0.0:
                assert dist is not None
                boundary_active = active & (np.abs(dist[case_idx]) <= float(boundary_band_px))
                if not np.any(boundary_active):
                    raise ValueError(
                        "benchmark.eval.quality_score resolved an empty boundary band: "
                        f"target={name}, case={case_idx}"
                    )
            case_weight = np.asarray(weight[case_idx], dtype=np.float32)
            denom = float(max(np.sum(case_weight), 1.0))
            base = float(
                np.sum(
                    _huber_map(pred_norm[case_idx] - true_norm[case_idx], delta=delta)
                    * case_weight
                )
                / denom
            )
            pools: list[float] = []
            for scale in pool_scales:
                true_pool, pool_fraction = _masked_avg_pool2d_nhw(
                    true_norm[case_idx : case_idx + 1],
                    mask[case_idx : case_idx + 1],
                    scale,
                )
                pred_pool, _ = _masked_avg_pool2d_nhw(
                    pred_norm[case_idx : case_idx + 1],
                    mask[case_idx : case_idx + 1],
                    scale,
                )
                pool_denom = float(np.sum(pool_fraction))
                pools.append(
                    float(
                        np.sum(_huber_map(pred_pool - true_pool, delta=delta) * pool_fraction)
                        / max(pool_denom, 1.0e-12)
                    )
                    if pool_denom > 0.0
                    else 0.0
                )
            pool = float(np.mean(pools)) if pools else 0.0
            grad_terms: list[np.ndarray] = []
            physical_grad_num = 0.0
            physical_grad_den = 0.0
            for axis in (0, 1):
                pred_diff = np.diff(pred_norm[case_idx], axis=axis)
                true_diff = np.diff(true_norm[case_idx], axis=axis)
                pair_mask = (active[1:, :] & active[:-1, :]) if axis == 0 else (active[:, 1:] & active[:, :-1])
                if np.any(pair_mask):
                    grad_terms.append(_huber_map(pred_diff[pair_mask] - true_diff[pair_mask], delta=delta))
                    pred_phys_diff = np.diff(pred_arr[case_idx], axis=axis)[pair_mask]
                    true_phys_diff = np.diff(true_arr[case_idx], axis=axis)[pair_mask]
                    physical_grad_num += float(np.sum((pred_phys_diff - true_phys_diff) ** 2))
                    physical_grad_den += float(np.sum(true_phys_diff**2))
            gradient = float(np.mean(np.concatenate(grad_terms))) if grad_terms else 0.0
            total_case = float(base + avgpool_lambda * pool + gradient_lambda * gradient)
            true_values = true_arr[case_idx][active].astype(np.float64)
            pred_values = pred_arr[case_idx][active].astype(np.float64)
            l2_den = max(float(np.linalg.norm(true_values)), 1.0e-12)
            physical_l2 = float(np.linalg.norm(pred_values - true_values) / l2_den)
            physical_gradient = float(np.sqrt(physical_grad_num) / max(np.sqrt(physical_grad_den), 1.0e-12))
            true_mean = float(np.mean(true_values)) if true_values.size else 0.0
            pred_mean = float(np.mean(pred_values)) if pred_values.size else 0.0
            mean_error = abs(pred_mean - true_mean) / max(abs(true_mean), 1.0e-12)
            clip = dict(transform_spec.get("clip", {}) or {})
            if str(clip.get("mode", "none")) == "physical_bounds" and pred_values.size:
                lo, hi = float(clip["min"]), float(clip["max"])
                tol = max(abs(hi - lo) * 1.0e-6, 1.0e-12)
                bound_fraction = float(np.mean((pred_values <= lo + tol) | (pred_values >= hi - tol)))
            else:
                bound_fraction = 0.0
            case_base.append(base)
            case_pool.append(pool)
            case_gradient.append(gradient)
            case_total.append(total_case)
            case_physical_l2.append(physical_l2)
            case_physical_gradient.append(physical_gradient)
            case_mean_error.append(mean_error)
            case_bound_fraction.append(bound_fraction)
        base_stats = _case_percentiles(case_base)
        pool_stats = _case_percentiles(case_pool)
        gradient_stats = _case_percentiles(case_gradient)
        total_stats = _case_percentiles(case_total)
        base_by_var[name] = base_stats[0]
        pool_by_var[name] = pool_stats[0]
        gradient_by_var[name] = gradient_stats[0]
        total_by_var[name] = float(total_stats[0] + p90_weight * total_stats[1] + worst_weight * total_stats[2])
        summaries_by_var[name] = {
            "total": total_stats,
            "physical_rel_l2": _case_percentiles(case_physical_l2),
            "physical_gradient_rel_l2": _case_percentiles(case_physical_gradient),
            "physical_mean_rel_error": _case_percentiles(case_mean_error),
            "physical_bound_fraction": _case_percentiles(case_bound_fraction),
        }
    base_mean, base_by_group = _aggregate_quality_target_values(
        base_by_var,
        target_vars=target_names,
        target_aggregation=target_aggregation,
        groups=target_groups,
    )
    pool_mean, pool_by_group = _aggregate_quality_target_values(
        pool_by_var,
        target_vars=target_names,
        target_aggregation=target_aggregation,
        groups=target_groups,
    )
    gradient_mean, gradient_by_group = _aggregate_quality_target_values(
        gradient_by_var,
        target_vars=target_names,
        target_aggregation=target_aggregation,
        groups=target_groups,
    )
    total, total_by_group = _aggregate_quality_target_values(
        total_by_var,
        target_vars=target_names,
        target_aggregation=target_aggregation,
        groups=target_groups,
    )
    out = {
        "surrogate_quality_score": total,
        "score_spatial_huber_component": base_mean,
        "score_avgpool_huber_component": pool_mean,
        "score_gradient_huber_component": gradient_mean,
        "score_nrmse_component": base_mean,
        "score_boundary_component": 0.0,
        "score_continuity_component": pool_mean,
        "score_physics_component": 0.0,
        "score_sign_component": 0.0,
    }
    for name in sorted(total_by_var):
        out[f"score_spatial_huber_{name}"] = float(base_by_var[name])
        out[f"score_avgpool_huber_{name}"] = float(pool_by_var[name])
        out[f"score_gradient_huber_{name}"] = float(gradient_by_var[name])
        out[f"score_total_{name}"] = float(total_by_var[name])
        for metric_name, stats in summaries_by_var[name].items():
            out[f"score_{metric_name}_median_{name}"] = float(stats[0])
            out[f"score_{metric_name}_p90_{name}"] = float(stats[1])
            out[f"score_{metric_name}_worst_{name}"] = float(stats[2])
    for group_name in target_groups:
        out[f"score_spatial_huber_group_{group_name}"] = float(base_by_group[group_name])
        out[f"score_avgpool_huber_group_{group_name}"] = float(pool_by_group[group_name])
        out[f"score_gradient_huber_group_{group_name}"] = float(gradient_by_group[group_name])
        out[f"score_total_group_{group_name}"] = float(total_by_group[group_name])
    return out


def target_std_for_score(values: np.ndarray, mask: np.ndarray | None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    active = np.isfinite(arr)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if arr.ndim == 4 and m.ndim == 3 and m.shape[-2:] == arr.shape[-2:]:
            m = m[:, None, :, :]
        while m.ndim < arr.ndim:
            m = m[None, ...]
        try:
            m = np.broadcast_to(m, arr.shape)
        except ValueError:
            return float("nan")
        active &= m
    vals = arr[active]
    if vals.size < 2:
        return float("nan")
    std = float(np.std(vals))
    if not np.isfinite(std) or std <= 1.0e-12:
        return float("nan")
    return std


__all__ = [
    "QUALITY_PROTOCOL_LEGACY_COMPOSITE_V1",
    "QUALITY_PROTOCOL_SPATIAL_HUBER_V1",
    "QUALITY_PROTOCOL_SPATIAL_HUBER_V2",
    "abs_log_ratio",
    "build_spatial_huber_quality_components",
    "build_surrogate_quality_components",
    "finite_mean",
    "inf_if_nonfinite",
    "quality_score_protocol_metadata",
    "resolve_quality_score_protocol",
    "target_std_for_score",
    "zero_if_nonfinite",
]
