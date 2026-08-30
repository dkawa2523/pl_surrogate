"""Core leaderboard metrics and quality score builders."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from plasma_surrogate.core.spatial_regions import (
    build_boundary_type_masks,
    build_region_masks,
)
from plasma_surrogate.eval.group_metrics import build_target_group_metric_columns
from plasma_surrogate.eval.metrics import (
    finite_pair_stats,
    r2_masked,
    rmse_masked,
)
from plasma_surrogate.eval.positive_diagnostics import (
    positive_diagnostic_columns,
    positive_targets_from_schema,
)
from plasma_surrogate.eval.quality_score import (
    abs_log_ratio,
    build_spatial_huber_quality_components,
    build_surrogate_quality_components,
    finite_mean,
    inf_if_nonfinite,
    quality_score_protocol_metadata,
    resolve_quality_score_protocol,
    target_std_for_score,
)
from plasma_surrogate.eval.sanity_checks import build_metric_validity_flags


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
    # Density gradients can be O(1e20). Squaring them in float32 overflows
    # even though the gradient magnitude itself is representable. Evaluation
    # must not turn a finite physical prediction into an infinite continuity
    # diagnostic, so only this norm reduction is promoted to float64.
    pg = np.hypot(pg_x.astype(np.float64), pg_y.astype(np.float64))
    tg = np.hypot(tg_x.astype(np.float64), tg_y.astype(np.float64))
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
    region_mask_plasma: np.ndarray | None = None,
    distance_any: np.ndarray | None = None,
    distance_signed: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    target_vars_for_score: list[str] | None = None,
    region_band_cfg: dict[str, Any] | None = None,
    quality_score_cfg: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    target_scalers: dict[str, Any] | None = None,
    target_transforms: dict[str, Any] | None = None,
    output_vars: list[str] | None = None,
    extended_diagnostics_enabled: bool = False,
) -> dict[str, Any]:
    r2_scores = dict(r2_scores or {})
    quality_protocol = resolve_quality_score_protocol(quality_score_cfg)
    quality_score_cfg_effective = dict(quality_protocol["effective_config"])
    quality_protocol_meta = quality_score_protocol_metadata(quality_protocol)
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
    outside_rmse: dict[str, float] = {}
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
    region_plasma_mask = region_mask_plasma if region_mask_plasma is not None else mask_plasma
    if true_eval is not None and (
        distance_signed is not None or (region_plasma_mask is not None and distance_any is not None)
    ):
        region_masks = build_region_masks(
            mask_plasma=region_plasma_mask,
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
            mask_plasma=(
                region_plasma_mask
                if region_plasma_mask is not None
                else region_masks["all_plasma"].astype(np.float32)
            ),
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
        outside_mask = np.logical_or(
            np.asarray(region_masks["chamber_near"], dtype=bool),
            np.asarray(region_masks["chamber_far"], dtype=bool),
        )
        if np.any(outside_mask):
            for name in core_eval_keys:
                if name in true_eval and name in pred_eval:
                    outside_rmse[name] = float(rmse_masked(true_eval[name], pred_eval[name], outside_mask))

    quality_nrmse_values: list[float] = []
    metric_src_for_quality = metrics_plasma if metrics_plasma else metrics
    score_keys_for_quality = [k for k in target_vars_effective if k in core_eval_keys] or core_eval_keys
    for key in score_keys_for_quality:
        if true_eval is None or key not in true_eval or key not in metric_src_for_quality:
            continue
        rmse_val = float(metric_src_for_quality[key])
        scale = target_std_for_score(
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
    boundary_to_deep_rmse_ratio_mean = finite_mean(list(boundary_deep_rmse_ratio.values()))
    mean_nrmse_plasma_by_target = finite_mean(quality_nrmse_values)
    abs_log_continuity_grad_ratio = abs_log_ratio(continuity_grad_ratio_all)
    abs_log_continuity_lap_ratio = abs_log_ratio(continuity_lap_ratio_all)
    poisson_residual_raw = single_diagnostics.get("poisson_residual_norm")
    poisson_residual_penalty = 0.0 if poisson_residual_raw is None else inf_if_nonfinite(poisson_residual_raw)
    boundary_residual_raw = single_diagnostics.get("boundary_operator_proxy_loss")
    boundary_residual_penalty = (
        0.0 if boundary_residual_raw is None else inf_if_nonfinite(boundary_residual_raw)
    )
    positive_targets = positive_targets_from_schema(target_role_schema)
    positive_target_negative_ratio_penalty = finite_mean(
        [float(neg_ratio_plasma[name]) for name in sorted(positive_targets) if name in neg_ratio_plasma],
        default=0.0,
    )
    quality_components = build_surrogate_quality_components(
        mean_nrmse_plasma_by_target=mean_nrmse_plasma_by_target,
        boundary_to_deep_rmse_ratio_mean=boundary_to_deep_rmse_ratio_mean,
        continuity_grad_ratio=continuity_grad_ratio_all,
        continuity_lap_ratio=continuity_lap_ratio_all,
        poisson_residual_penalty=poisson_residual_penalty,
        boundary_residual_penalty=boundary_residual_penalty,
        positive_target_negative_ratio_penalty=positive_target_negative_ratio_penalty,
        quality_score_cfg=quality_score_cfg_effective,
    )
    spatial_huber_components = build_spatial_huber_quality_components(
        true_eval=true_eval,
        pred_eval=pred_eval,
        mask_plasma=mask_plasma,
        distance_any=distance_any,
        target_vars=score_keys_for_quality,
        target_scalers=target_scalers,
        target_transforms=target_transforms,
        target_role_schema=target_role_schema,
        cfg=quality_score_cfg_effective,
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
        "sdf_boundary_minus_deep_r2_mean": finite_mean(list(boundary_deep_r2_gap.values())),
        "quality_components": _json_dump_compact(quality_components),
        "validity_flags": _json_dump_compact(validity_flags),
        "target_metrics_invalid_vars": "|".join(str(v) for v in validity_flags["invalid_target_vars"]),
        **quality_protocol_meta,
        **quality_components,
    }
    row: dict[str, Any] = {
        "model_id": model_id,
        "surrogate_quality_score": float(quality_components.get("surrogate_quality_score", float("nan"))),
        "target_metrics_valid": bool(validity_flags["target_metrics_valid"]),
        **quality_protocol_meta,
        "_diagnostics": diagnostics,
    }
    positive_columns, positive_violation_rate, positive_negative_min = positive_diagnostic_columns(
        pred_eval=pred_eval,
        target_role_schema=target_role_schema,
    )
    for name in core_eval_keys:
        row[f"test_rmse_{name}"] = _metric_val(metrics, name)
        row[f"test_r2_{name}"] = _metric_val(r2_scores, name)
        row[f"test_rmse_{name}_plasma"] = _metric_val(metrics_plasma, name)
        row[f"test_r2_{name}_plasma"] = _metric_val(r2_plasma, name)
        if extended_diagnostics_enabled and name in boundary_rmse:
            row[f"test_rmse_{name}_boundary_band"] = float(boundary_rmse[name])
        if extended_diagnostics_enabled and name in deep_rmse:
            row[f"test_rmse_{name}_deep_plasma"] = float(deep_rmse[name])
        if extended_diagnostics_enabled and name in outside_rmse:
            row[f"test_rmse_{name}_outside"] = float(outside_rmse[name])
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
        diagnostics[f"test_rmse_{name}_outside"] = _metric_val(outside_rmse, name)
        diagnostics[f"test_neg_ratio_{name}_plasma"] = _metric_val(neg_ratio_plasma, name)
        if name in positive_violation_rate:
            diagnostics[f"positive_violation_rate_{name}"] = float(positive_violation_rate[name])
        if name in positive_negative_min:
            diagnostics[f"negative_min_{name}"] = float(positive_negative_min[name])
        for boundary_type in ("interface", "bc_dir", "wafer"):
            diagnostics[f"test_rmse_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_rmse_by_type.get(boundary_type, {}),
                name,
            )
            diagnostics[f"test_r2_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_r2_by_type.get(boundary_type, {}),
                name,
            )
    row.update(
        build_target_group_metric_columns(
            output_vars=output_vars,
            target_role_schema=target_role_schema,
            metrics=metrics,
            r2_scores=r2_scores,
            metrics_plasma=metrics_plasma,
            r2_plasma=r2_plasma,
            metrics_boundary_band=boundary_rmse if extended_diagnostics_enabled else None,
            metrics_deep_plasma=deep_rmse if extended_diagnostics_enabled else None,
            metrics_outside=outside_rmse if extended_diagnostics_enabled else None,
            positive_violation_rate=positive_violation_rate if extended_diagnostics_enabled else None,
        )
    )
    if extended_diagnostics_enabled:
        row.update(positive_columns)
    return row


__all__ = ["build_benchmark_eval_row", "build_region_metrics", "build_target_group_metric_columns"]
