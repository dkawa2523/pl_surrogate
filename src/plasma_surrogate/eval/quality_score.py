"""Surrogate quality-score components."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.eval.spatial_metrics import _field_as_nhw, _mask_as_nhw


QUALITY_SCORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "nrmse": 0.45,
    "boundary": 0.20,
    "continuity": 0.15,
    "physics": 0.15,
    "sign": 0.05,
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
    h_eff = (h // s) * s
    w_eff = (w // s) * s
    if h_eff <= 0 or w_eff <= 0:
        return a
    cropped = a[..., :h_eff, :w_eff]
    return cropped.reshape(cropped.shape[0], h_eff // s, s, w_eff // s, s).mean(axis=(2, 4)).astype(np.float32)


def build_spatial_huber_quality_components(
    *,
    true_eval: dict[str, np.ndarray] | None,
    pred_eval: dict[str, np.ndarray],
    mask_plasma: np.ndarray | None,
    distance_any: np.ndarray | None,
    target_vars: list[str],
    target_scalers: dict[str, Any] | None,
    cfg: dict[str, Any] | None,
) -> dict[str, float]:
    if true_eval is None:
        return {}
    score_cfg = dict(cfg or {})
    mode = str(score_cfg.get("mode", "")).strip().lower()
    if mode != "spatial_huber":
        return {}
    delta = float(score_cfg.get("delta", 1.0))
    boundary_alpha = float(score_cfg.get("boundary_alpha", 2.0))
    boundary_band_px = float(score_cfg.get("boundary_band_px", 2.0))
    avgpool_lambda = float(score_cfg.get("avgpool_lambda", 0.05))
    pool_scale = int(max(int(score_cfg.get("pool_scale", score_cfg.get("pool_kernel", 3))), 1))
    base_by_var: dict[str, float] = {}
    pool_by_var: dict[str, float] = {}
    total_by_var: dict[str, float] = {}
    for name in target_vars:
        if name not in true_eval or name not in pred_eval:
            continue
        true_arr = _field_as_nhw(true_eval[name], key=f"true_eval[{name}]")
        pred_arr = _field_as_nhw(pred_eval[name], key=f"pred_eval[{name}]")
        if true_arr.shape != pred_arr.shape:
            continue
        affine = _scaler_affine_for_var(target_scalers, name)
        if affine is None:
            active_for_std = _mask_as_nhw(mask_plasma, n_cases=int(true_arr.shape[0]), shape=tuple(true_arr.shape[-2:]))
            vals = true_arr[np.asarray(active_for_std, dtype=bool)]
            std = float(np.std(vals)) if vals.size > 1 else float("nan")
            mean = float(np.mean(vals)) if vals.size > 0 else 0.0
            if not np.isfinite(std) or std <= 1.0e-12:
                continue
        else:
            mean, std = affine
        true_norm = ((true_arr - float(mean)) / float(std)).astype(np.float32)
        pred_norm = ((pred_arr - float(mean)) / float(std)).astype(np.float32)
        mask = _mask_as_nhw(mask_plasma, n_cases=int(true_arr.shape[0]), shape=tuple(true_arr.shape[-2:]))
        weight = mask.astype(np.float32)
        if distance_any is not None:
            dist = _field_as_nhw(distance_any, key="distance_any")
            if int(dist.shape[0]) == 1 and int(true_arr.shape[0]) > 1:
                dist = np.repeat(dist, int(true_arr.shape[0]), axis=0)
            if dist.shape == true_arr.shape:
                weight = weight * (1.0 + float(boundary_alpha) * (np.abs(dist) <= float(boundary_band_px)).astype(np.float32))
        huber = _huber_map(pred_norm - true_norm, delta=delta)
        denom = float(max(np.sum(mask), 1.0))
        base = float(np.sum(huber * weight) / denom)
        true_pool = _avg_pool2d_nhw(true_norm, pool_scale)
        pred_pool = _avg_pool2d_nhw(pred_norm, pool_scale)
        mask_pool = (_avg_pool2d_nhw(mask.astype(np.float32), pool_scale) > 0.5).astype(np.float32)
        pool_denom = float(max(np.sum(mask_pool), 1.0))
        pool = float(np.sum(_huber_map(pred_pool - true_pool, delta=delta) * mask_pool) / pool_denom)
        base_by_var[name] = base
        pool_by_var[name] = pool
        total_by_var[name] = float(base + avgpool_lambda * pool)
    base_mean = finite_mean(list(base_by_var.values()), default=float("nan"))
    pool_mean = finite_mean(list(pool_by_var.values()), default=float("nan"))
    total = finite_mean(list(total_by_var.values()), default=float("nan"))
    out = {
        "surrogate_quality_score": total,
        "score_spatial_huber_component": base_mean,
        "score_avgpool_huber_component": pool_mean,
        "score_nrmse_component": base_mean,
        "score_boundary_component": 0.0,
        "score_continuity_component": pool_mean,
        "score_physics_component": 0.0,
        "score_sign_component": 0.0,
    }
    for name in sorted(total_by_var):
        out[f"score_spatial_huber_{name}"] = float(base_by_var[name])
        out[f"score_avgpool_huber_{name}"] = float(pool_by_var[name])
        out[f"score_total_{name}"] = float(total_by_var[name])
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
    "abs_log_ratio",
    "build_spatial_huber_quality_components",
    "build_surrogate_quality_components",
    "finite_mean",
    "inf_if_nonfinite",
    "target_std_for_score",
    "zero_if_nonfinite",
]
