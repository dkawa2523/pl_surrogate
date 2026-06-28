"""Core leaderboard metrics and quality score builders."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from plasma_surrogate.core.spatial_regions import (
    build_boundary_type_masks,
    build_region_masks,
)
from plasma_surrogate.eval.metrics import (
    finite_pair_stats,
    r2_masked,
    rmse_masked,
)
from plasma_surrogate.eval.sanity_checks import build_metric_validity_flags
from plasma_surrogate.eval.spatial_metrics import _field_as_nhw, _mask_as_nhw


def _laplacian2d_bhw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(a, dtype=np.float32)
    out[:, 1:-1, 1:-1] = (
        a[:, 1:-1, 2:]
        + a[:, 1:-1, :-2]
        + a[:, 2:, 1:-1]
        + a[:, :-2, 1:-1]
        - 4.0 * a[:, 1:-1, 1:-1]
    )
    return out


def _continuity_ratios(
    *,
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(true, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if p.ndim != 3 or t.ndim != 3:
        return float("nan"), float("nan")
    if m.ndim == 2:
        m = np.repeat(m[None, ...], int(p.shape[0]), axis=0)
    elif m.ndim == 3 and int(m.shape[0]) == 1 and int(p.shape[0]) > 1:
        m = np.repeat(m, int(p.shape[0]), axis=0)
    if m.shape != p.shape:
        return float("nan"), float("nan")
    pg_y, pg_x = np.gradient(p, axis=(1, 2), edge_order=1)
    tg_y, tg_x = np.gradient(t, axis=(1, 2), edge_order=1)
    pg = np.sqrt(pg_x * pg_x + pg_y * pg_y).astype(np.float32)
    tg = np.sqrt(tg_x * tg_x + tg_y * tg_y).astype(np.float32)
    active = m > 0.5
    if not np.any(active):
        return float("nan"), float("nan")
    pg_mean = float(np.mean(pg[active]))
    tg_mean = float(np.mean(tg[active]))
    grad_ratio = float(pg_mean / max(tg_mean, 1.0e-12))
    p_lap = np.abs(_laplacian2d_bhw(p))
    t_lap = np.abs(_laplacian2d_bhw(t))
    pl_mean = float(np.mean(p_lap[active]))
    tl_mean = float(np.mean(t_lap[active]))
    lap_ratio = float(pl_mean / max(tl_mean, 1.0e-12))
    return grad_ratio, lap_ratio


def _safe_ratio(num: float | None, den: float | None) -> float:
    if num is None or den is None:
        return float("nan")
    num_f = float(num)
    den_f = float(den)
    if not np.isfinite(num_f) or not np.isfinite(den_f) or abs(den_f) <= 1.0e-12:
        return float("nan")
    return float(num_f / den_f)


def _safe_diff(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("nan")
    left_f = float(left)
    right_f = float(right)
    if not np.isfinite(left_f) or not np.isfinite(right_f):
        return float("nan")
    return float(left_f - right_f)


def _finite_mean(values: list[float], *, default: float = float("nan")) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return value_f if np.isfinite(value_f) else None
    return value


def _json_dump_compact(payload: dict[str, Any]) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


_QUALITY_SCORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "nrmse": 0.45,
    "boundary": 0.20,
    "continuity": 0.15,
    "physics": 0.15,
    "sign": 0.05,
}


def _quality_score_weights(cfg: dict[str, Any] | None) -> dict[str, float]:
    raw_weights = dict(dict(cfg or {}).get("weights", {}) or {})
    weights = dict(_QUALITY_SCORE_DEFAULT_WEIGHTS)
    for key in weights:
        if key in raw_weights:
            weights[key] = float(raw_weights[key])
    return weights


def _zero_if_nonfinite(value: float | None) -> float:
    if value is None:
        return 0.0
    value_f = float(value)
    if not np.isfinite(value_f):
        return 0.0
    return value_f


def _inf_if_nonfinite(value: float | None) -> float:
    if value is None:
        return float("inf")
    value_f = float(value)
    if not np.isfinite(value_f):
        return float("inf")
    return value_f


def _abs_log_ratio(value: float | None) -> float:
    value_f = _inf_if_nonfinite(value)
    if not np.isfinite(value_f):
        return float("inf")
    if value_f <= 0.0:
        return 0.0
    return float(abs(np.log(max(value_f, 1.0e-12))))


def _positive_targets_from_schema(target_role_schema: dict[str, Any] | None) -> set[str]:
    schema = dict(target_role_schema or {})
    positive = schema.get("positive_targets", [])
    if isinstance(positive, list):
        return {str(v) for v in positive}
    targets = schema.get("targets", [])
    if not isinstance(targets, list):
        return set()
    out: set[str] = set()
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        target_id = str(entry.get("id", "")).strip()
        if target_id and entry.get("positive") is True:
            out.add(target_id)
    return out


def _build_surrogate_quality_components(
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
    nrmse_component = _inf_if_nonfinite(mean_nrmse_plasma_by_target)
    boundary_ratio = _zero_if_nonfinite(boundary_to_deep_rmse_ratio_mean)
    boundary_component = float(max(0.0, boundary_ratio - 1.0)) if boundary_ratio > 0.0 else 0.0
    continuity_component = float(
        0.5 * (_abs_log_ratio(continuity_grad_ratio) + _abs_log_ratio(continuity_lap_ratio))
    )
    physics_component = float(
        0.5
        * (
            np.log1p(abs(_inf_if_nonfinite(poisson_residual_penalty)))
            + np.log1p(abs(_inf_if_nonfinite(boundary_residual_penalty)))
        )
    )
    sign_component = _inf_if_nonfinite(positive_target_negative_ratio_penalty)
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


def _spatial_huber_quality_components(
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
    base_mean = _finite_mean(list(base_by_var.values()), default=float("nan"))
    pool_mean = _finite_mean(list(pool_by_var.values()), default=float("nan"))
    total = _finite_mean(list(total_by_var.values()), default=float("nan"))
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


def _target_std_for_score(values: np.ndarray, mask: np.ndarray | None) -> float:
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


def build_region_metrics(
    case_key: str,
    mask_plasma: np.ndarray,
    mask_bulk: np.ndarray,
    mask_boundary: np.ndarray,
    density: np.ndarray | None = None,
    density_key: str = "density",
) -> dict[str, float | str | bool]:
    density_arr = np.asarray(density, dtype=np.float32)
    return {
        "case_key": case_key,
        "density_key": str(density_key),
        "plasma_mean_density": float(np.mean(density_arr[mask_plasma])),
        "bulk_mean_density": float(np.mean(density_arr[mask_bulk])) if np.any(mask_bulk) else 0.0,
        "boundary_mean_density": float(np.mean(density_arr[mask_boundary])) if np.any(mask_boundary) else 0.0,
    }


def build_benchmark_eval_row(
    model_id: str,
    metrics: dict[str, float],
    r2_scores: dict[str, float] | None,
    pred_eval: dict[str, np.ndarray],
    single_diagnostics: dict[str, Any],
    true_eval: dict[str, np.ndarray] | None = None,
    mask_plasma: np.ndarray | None = None,
    distance_any: np.ndarray | None = None,
    distance_signed: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    target_vars_for_score: list[str] | None = None,
    region_band_cfg: dict[str, Any] | None = None,
    quality_score_cfg: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    target_scalers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r2_scores = dict(r2_scores or {})
    paired_eval_keys = list(set(pred_eval.keys()) & set((true_eval or {}).keys()))
    target_vars_effective = [str(v) for v in list(target_vars_for_score or [])]
    core_eval_keys = list(paired_eval_keys)

    def _metric_val(src: dict[str, float], name: str | None, *, default: float = float("nan")) -> float:
        if name is None:
            return float(default)
        if name not in src:
            return float(default)
        return float(src[name])

    metrics_plasma: dict[str, float] = {}
    r2_plasma: dict[str, float] = {}
    finite_stats_plasma: dict[str, dict[str, float]] = {}
    if true_eval is not None and mask_plasma is not None:
        for name in core_eval_keys:
            if name in true_eval and name in pred_eval:
                finite_stats_plasma[name] = finite_pair_stats(true_eval[name], pred_eval[name], mask_plasma)
                metrics_plasma[name] = float(rmse_masked(true_eval[name], pred_eval[name], mask_plasma))
                r2_plasma[name] = float(r2_masked(true_eval[name], pred_eval[name], mask_plasma))

    boundary_rmse: dict[str, float] = {}
    boundary_r2: dict[str, float] = {}
    boundary_rmse_by_type: dict[str, dict[str, float]] = {"interface": {}, "bc_dir": {}, "wafer": {}}
    boundary_r2_by_type: dict[str, dict[str, float]] = {"interface": {}, "bc_dir": {}, "wafer": {}}
    deep_rmse: dict[str, float] = {}
    deep_r2: dict[str, float] = {}
    chamber_near_rmse: dict[str, float] = {}
    neg_ratio_plasma: dict[str, float] = {}
    continuity_grad_ratios: list[float] = []
    continuity_lap_ratios: list[float] = []
    region_masks: dict[str, np.ndarray] = {}
    bands_cfg = dict(region_band_cfg or {})
    band_mode = str(bands_cfg.get("mode", "fixed_px")).strip().lower()
    boundary_in_px = float(bands_cfg.get("boundary_in_px", 2.0))
    deep_plasma_px = float(bands_cfg.get("deep_plasma_px", 10.0))
    boundary_q = float(bands_cfg.get("boundary_q", 0.15))
    deep_q = float(bands_cfg.get("deep_q", 0.70))
    if true_eval is not None and (distance_signed is not None or (mask_plasma is not None and distance_any is not None)):
        region_masks = build_region_masks(
            mask_plasma=mask_plasma,
            distance_any=distance_any,
            distance_signed=distance_signed,
            mode=band_mode,
            boundary_in_px=boundary_in_px,
            mid_plasma_px=deep_plasma_px,
            deep_plasma_px=deep_plasma_px,
            boundary_q=boundary_q,
            deep_q=deep_q,
        )
    if true_eval is not None and region_masks:
        boundary_mask = np.asarray(region_masks["boundary_in"], dtype=bool)
        if np.any(boundary_mask):
            for name in core_eval_keys:
                if name in true_eval and name in pred_eval:
                    boundary_rmse[name] = float(rmse_masked(true_eval[name], pred_eval[name], boundary_mask))
                    boundary_r2[name] = float(r2_masked(true_eval[name], pred_eval[name], boundary_mask))
        type_masks = build_boundary_type_masks(
            mask_plasma=mask_plasma if mask_plasma is not None else region_masks["all_plasma"].astype(np.float32),
            distance_any=distance_any if distance_any is not None else np.abs(np.asarray(distance_signed, dtype=np.float32)),
            band_px=boundary_in_px,
            bc_dir_mask=bc_dir_mask,
            wafer_mask=wafer_mask,
        )
        for type_name in ["interface", "bc_dir", "wafer"]:
            mask_t = np.asarray(type_masks[type_name], dtype=bool)
            if not np.any(mask_t):
                continue
            for name in core_eval_keys:
                if name in true_eval and name in pred_eval:
                    boundary_rmse_by_type[type_name][name] = float(rmse_masked(true_eval[name], pred_eval[name], mask_t))
                    boundary_r2_by_type[type_name][name] = float(r2_masked(true_eval[name], pred_eval[name], mask_t))
    if true_eval is not None and region_masks:
        deep_mask = np.asarray(region_masks["plasma_deep"], dtype=bool)
        chamber_near_mask = np.asarray(region_masks["chamber_near"], dtype=bool)
        if np.any(deep_mask):
            for name in core_eval_keys:
                if name in true_eval and name in pred_eval:
                    deep_rmse[name] = float(rmse_masked(true_eval[name], pred_eval[name], deep_mask))
                    deep_r2[name] = float(r2_masked(true_eval[name], pred_eval[name], deep_mask))
        if np.any(chamber_near_mask):
            for name in core_eval_keys:
                if name in true_eval and name in pred_eval:
                    chamber_near_rmse[name] = float(rmse_masked(true_eval[name], pred_eval[name], chamber_near_mask))

    quality_nrmse_values: list[float] = []
    metric_src_for_quality = metrics_plasma if metrics_plasma else metrics
    score_keys_for_quality = [k for k in target_vars_effective if k in core_eval_keys] or core_eval_keys
    for key in score_keys_for_quality:
        if true_eval is None or key not in true_eval or key not in metric_src_for_quality:
            continue
        rmse_val = float(metric_src_for_quality[key])
        scale = _target_std_for_score(
            true_eval[key],
            mask_plasma if (metrics_plasma and mask_plasma is not None) else None,
        )
        if np.isfinite(rmse_val) and np.isfinite(scale):
            quality_nrmse_values.append(float(rmse_val / scale))

    if mask_plasma is not None:
        plasma_mask_bool = np.asarray(mask_plasma, dtype=np.float32) > 0.5
        for name in core_eval_keys:
            if name not in pred_eval:
                continue
            pred_arr = np.asarray(pred_eval[name], dtype=np.float32)
            if pred_arr.ndim == 4 and int(pred_arr.shape[1]) == 1:
                pred_arr = pred_arr[:, 0]
            elif pred_arr.ndim == 2:
                pred_arr = pred_arr[None, ...]
            if pred_arr.ndim != 3:
                continue
            mask_eff = plasma_mask_bool
            if mask_eff.ndim == 2:
                mask_eff = np.repeat(mask_eff[None, ...], int(pred_arr.shape[0]), axis=0)
            elif mask_eff.ndim == 3 and int(mask_eff.shape[0]) == 1 and int(pred_arr.shape[0]) > 1:
                mask_eff = np.repeat(mask_eff, int(pred_arr.shape[0]), axis=0)
            if mask_eff.shape != pred_arr.shape:
                continue
            active = mask_eff > 0.5
            if np.any(active):
                neg_ratio_plasma[name] = float(np.mean((pred_arr[active] < 0.0).astype(np.float32)))

    if true_eval is not None and mask_plasma is not None:
        plasma_mask_bool = np.asarray(mask_plasma, dtype=np.float32) > 0.5
        for name in core_eval_keys:
            if name not in true_eval or name not in pred_eval:
                continue
            pred_arr = np.asarray(pred_eval[name], dtype=np.float32)
            true_arr = np.asarray(true_eval[name], dtype=np.float32)
            if pred_arr.ndim == 4 and int(pred_arr.shape[1]) == 1:
                pred_arr = pred_arr[:, 0]
            if true_arr.ndim == 4 and int(true_arr.shape[1]) == 1:
                true_arr = true_arr[:, 0]
            if pred_arr.ndim != 3 or true_arr.ndim != 3:
                continue
            g_ratio, l_ratio = _continuity_ratios(pred=pred_arr, true=true_arr, mask=plasma_mask_bool)
            if np.isfinite(g_ratio):
                continuity_grad_ratios.append(float(g_ratio))
            if np.isfinite(l_ratio):
                continuity_lap_ratios.append(float(l_ratio))
    continuity_grad_ratio_all = float(np.mean(np.asarray(continuity_grad_ratios, dtype=np.float64))) if continuity_grad_ratios else float("nan")
    continuity_lap_ratio_all = float(np.mean(np.asarray(continuity_lap_ratios, dtype=np.float64))) if continuity_lap_ratios else float("nan")
    boundary_deep_rmse_ratio: dict[str, float] = {}
    boundary_deep_r2_gap: dict[str, float] = {}
    for name in core_eval_keys:
        boundary_deep_rmse_ratio[name] = _safe_ratio(boundary_rmse.get(name), deep_rmse.get(name))
        boundary_deep_r2_gap[name] = _safe_diff(boundary_r2.get(name), deep_r2.get(name))
    boundary_to_deep_rmse_ratio_mean = _finite_mean(list(boundary_deep_rmse_ratio.values()))
    mean_nrmse_plasma_by_target = _finite_mean(quality_nrmse_values)
    abs_log_continuity_grad_ratio = _abs_log_ratio(continuity_grad_ratio_all)
    abs_log_continuity_lap_ratio = _abs_log_ratio(continuity_lap_ratio_all)
    poisson_residual_raw = single_diagnostics.get("poisson_residual_norm")
    poisson_residual_penalty = 0.0 if poisson_residual_raw is None else _inf_if_nonfinite(poisson_residual_raw)
    boundary_residual_raw = single_diagnostics.get("boundary_operator_proxy_loss")
    boundary_residual_penalty = (
        0.0 if boundary_residual_raw is None else _inf_if_nonfinite(boundary_residual_raw)
    )
    positive_targets = _positive_targets_from_schema(target_role_schema)
    positive_target_negative_ratio_penalty = _finite_mean(
        [float(neg_ratio_plasma[name]) for name in sorted(positive_targets) if name in neg_ratio_plasma],
        default=0.0,
    )
    quality_components = _build_surrogate_quality_components(
        mean_nrmse_plasma_by_target=mean_nrmse_plasma_by_target,
        boundary_to_deep_rmse_ratio_mean=boundary_to_deep_rmse_ratio_mean,
        continuity_grad_ratio=continuity_grad_ratio_all,
        continuity_lap_ratio=continuity_lap_ratio_all,
        poisson_residual_penalty=poisson_residual_penalty,
        boundary_residual_penalty=boundary_residual_penalty,
        positive_target_negative_ratio_penalty=positive_target_negative_ratio_penalty,
        quality_score_cfg=quality_score_cfg,
    )
    spatial_huber_components = _spatial_huber_quality_components(
        true_eval=true_eval,
        pred_eval=pred_eval,
        mask_plasma=mask_plasma,
        distance_any=distance_any,
        target_vars=score_keys_for_quality,
        target_scalers=target_scalers,
        cfg=quality_score_cfg,
    )
    if spatial_huber_components:
        quality_components.update(spatial_huber_components)
    validity_flags = build_metric_validity_flags(
        target_vars=score_keys_for_quality,
        metrics_plasma=metrics_plasma,
        r2_plasma=r2_plasma,
        finite_stats_plasma=finite_stats_plasma,
        mask_plasma=mask_plasma,
        quality_components=quality_components,
    )
    if not bool(validity_flags["target_metrics_valid"]) or not np.isfinite(
        float(quality_components.get("surrogate_quality_score", float("nan")))
    ):
        quality_components["surrogate_quality_score"] = float("inf")
        validity_flags["quality_score_valid"] = False

    diagnostics: dict[str, Any] = {
        "continuity_grad_ratio_all_plasma": float(continuity_grad_ratio_all),
        "continuity_lap_ratio_all_plasma": float(continuity_lap_ratio_all),
        "sdf_boundary_to_deep_rmse_ratio_mean": boundary_to_deep_rmse_ratio_mean,
        "mean_nrmse_plasma_by_target": mean_nrmse_plasma_by_target,
        "boundary_to_deep_rmse_ratio_mean": boundary_to_deep_rmse_ratio_mean,
        "abs_log_continuity_grad_ratio": abs_log_continuity_grad_ratio,
        "abs_log_continuity_lap_ratio": abs_log_continuity_lap_ratio,
        "poisson_residual_penalty": poisson_residual_penalty,
        "boundary_residual_penalty": boundary_residual_penalty,
        "positive_target_negative_ratio_penalty": positive_target_negative_ratio_penalty,
        "sdf_boundary_minus_deep_r2_mean": _finite_mean(list(boundary_deep_r2_gap.values())),
        "quality_components": _json_dump_compact(quality_components),
        "validity_flags": _json_dump_compact(validity_flags),
        "target_metrics_invalid_vars": "|".join(str(v) for v in validity_flags["invalid_target_vars"]),
        **quality_components,
    }
    row: dict[str, Any] = {
        "model_id": model_id,
        "surrogate_quality_score": float(quality_components.get("surrogate_quality_score", float("nan"))),
        "target_metrics_valid": bool(validity_flags["target_metrics_valid"]),
        "_diagnostics": diagnostics,
    }
    for name in core_eval_keys:
        row[f"test_rmse_{name}"] = _metric_val(metrics, name)
        row[f"test_r2_{name}"] = _metric_val(r2_scores, name)
        row[f"test_rmse_{name}_plasma"] = _metric_val(metrics_plasma, name)
        row[f"test_r2_{name}_plasma"] = _metric_val(r2_plasma, name)
        if name in finite_stats_plasma:
            stats = finite_stats_plasma[name]
            diagnostics[f"test_active_count_{name}_plasma"] = float(stats["n_active"])
            diagnostics[f"test_nonfinite_count_{name}_plasma"] = float(stats["n_nonfinite"])
            diagnostics[f"test_finite_ratio_{name}_plasma"] = float(stats["finite_ratio"])
        diagnostics[f"test_rmse_{name}_boundary_in"] = _metric_val(boundary_rmse, name)
        diagnostics[f"test_r2_{name}_boundary_in"] = _metric_val(boundary_r2, name)
        diagnostics[f"test_rmse_{name}_plasma_deep"] = _metric_val(deep_rmse, name)
        diagnostics[f"test_r2_{name}_plasma_deep"] = _metric_val(deep_r2, name)
        diagnostics[f"test_rmse_{name}_boundary_to_deep_ratio"] = _metric_val(boundary_deep_rmse_ratio, name)
        diagnostics[f"test_r2_{name}_boundary_minus_deep"] = _metric_val(boundary_deep_r2_gap, name)
        diagnostics[f"test_rmse_{name}_chamber_near"] = _metric_val(chamber_near_rmse, name)
        diagnostics[f"test_neg_ratio_{name}_plasma"] = _metric_val(neg_ratio_plasma, name)
        for boundary_type in ("interface", "bc_dir", "wafer"):
            diagnostics[f"test_rmse_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_rmse_by_type.get(boundary_type, {}),
                name,
            )
            diagnostics[f"test_r2_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_r2_by_type.get(boundary_type, {}),
                name,
            )
    return row


__all__ = ["build_benchmark_eval_row", "build_region_metrics"]
