"""Shared metrics-row builders for viz and benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.spatial_regions import (
    build_boundary_type_masks,
    build_region_masks,
    normalize_target_region_by_var,
    target_region_for_var,
)
from plasma_surrogate.eval.metrics import (
    finite_pair_stats,
    poisson_residual_norm,
    r2_by_var,
    r2_masked,
    rmse_by_var,
    rmse_masked,
)
from plasma_surrogate.train.losses import poisson_residual_loss


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


def _abs_log_ratio(value: float | None) -> float:
    value_f = _zero_if_nonfinite(value)
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
    nrmse_component = _zero_if_nonfinite(mean_nrmse_plasma_by_target)
    boundary_ratio = _zero_if_nonfinite(boundary_to_deep_rmse_ratio_mean)
    boundary_component = float(max(0.0, boundary_ratio - 1.0)) if boundary_ratio > 0.0 else 0.0
    continuity_component = float(
        0.5 * (_abs_log_ratio(continuity_grad_ratio) + _abs_log_ratio(continuity_lap_ratio))
    )
    physics_component = float(
        0.5
        * (
            np.log1p(abs(_zero_if_nonfinite(poisson_residual_penalty)))
            + np.log1p(abs(_zero_if_nonfinite(boundary_residual_penalty)))
        )
    )
    sign_component = _zero_if_nonfinite(positive_target_negative_ratio_penalty)
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_single_case_physics_metrics(case_dir: Path) -> dict[str, float | str]:
    diag = _load_json(case_dir / "diagnostics.json")
    qoi = _load_json(case_dir / "qoi.json")
    map_l2_poisson = float(diag.get("poisson_residual_map_l2", 0.0))
    map_l2_boundary = float(diag.get("boundary_operator_residual_map_l2", 0.0))
    maps_path = case_dir / "diagnostics_maps.npz"
    if maps_path.exists():
        maps = np.load(maps_path)
        if "poisson_residual_map" in maps:
            arr = np.asarray(maps["poisson_residual_map"], dtype=np.float32)
            map_l2_poisson = float(np.sqrt(np.mean(arr**2)))
        if "boundary_operator_residual_map" in maps:
            arr = np.asarray(maps["boundary_operator_residual_map"], dtype=np.float32)
            map_l2_boundary = float(np.sqrt(np.mean(arr**2)))
    return {
        "case_key": case_dir.name,
        "poisson_residual_norm": float(diag.get("poisson_residual_norm", 0.0)),
        "bc_phi_mae": float(diag.get("bc_phi_mae", 0.0)),
        "uniformity": float(qoi.get("uniformity", 0.0)),
        "poisson_residual_map_l2": map_l2_poisson,
        "boundary_residual_map_l2": map_l2_boundary,
    }


def build_region_metrics(
    case_key: str,
    mask_plasma: np.ndarray,
    mask_bulk: np.ndarray,
    mask_boundary: np.ndarray,
    density: np.ndarray | None = None,
    density_key: str = "density",
) -> dict[str, float | str]:
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
    single_qoi: dict[str, Any],
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
) -> dict[str, float | str]:
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
    poisson_residual_penalty = _zero_if_nonfinite(
        single_diagnostics.get(
            "poisson_residual_map_l2",
            single_diagnostics.get("poisson_residual_norm", 0.0),
        )
    )
    boundary_residual_penalty = _zero_if_nonfinite(
        single_diagnostics.get(
            "boundary_operator_residual_map_l2",
            single_diagnostics.get("boundary_operator_proxy_loss", 0.0),
        )
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

    row: dict[str, float | str] = {
        "model_id": model_id,
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
        **quality_components,
    }
    for name in core_eval_keys:
        row[f"test_rmse_{name}"] = _metric_val(metrics, name)
        row[f"test_r2_{name}"] = _metric_val(r2_scores, name)
        row[f"test_rmse_{name}_plasma"] = _metric_val(metrics_plasma, name)
        row[f"test_r2_{name}_plasma"] = _metric_val(r2_plasma, name)
        if name in finite_stats_plasma:
            stats = finite_stats_plasma[name]
            row[f"test_active_count_{name}_plasma"] = float(stats["n_active"])
            row[f"test_nonfinite_count_{name}_plasma"] = float(stats["n_nonfinite"])
            row[f"test_finite_ratio_{name}_plasma"] = float(stats["finite_ratio"])
        row[f"test_rmse_{name}_boundary_in"] = _metric_val(boundary_rmse, name)
        row[f"test_r2_{name}_boundary_in"] = _metric_val(boundary_r2, name)
        row[f"test_rmse_{name}_plasma_deep"] = _metric_val(deep_rmse, name)
        row[f"test_r2_{name}_plasma_deep"] = _metric_val(deep_r2, name)
        row[f"test_rmse_{name}_boundary_to_deep_ratio"] = _metric_val(boundary_deep_rmse_ratio, name)
        row[f"test_r2_{name}_boundary_minus_deep"] = _metric_val(boundary_deep_r2_gap, name)
        row[f"test_rmse_{name}_chamber_near"] = _metric_val(chamber_near_rmse, name)
        row[f"test_neg_ratio_{name}_plasma"] = _metric_val(neg_ratio_plasma, name)
        for boundary_type in ("interface", "bc_dir", "wafer"):
            row[f"test_rmse_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_rmse_by_type.get(boundary_type, {}),
                name,
            )
            row[f"test_r2_{name}_boundary_in_{boundary_type}"] = _metric_val(
                boundary_r2_by_type.get(boundary_type, {}),
                name,
            )
    return row


def build_spatial_error_summary_rows(
    *,
    pred_eval: dict[str, np.ndarray],
    true_eval: dict[str, np.ndarray],
    mask_plasma: np.ndarray | None,
    distance_signed: np.ndarray | None,
    distance_any: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    boundary_type_breakdown: bool = False,
    vars_for_summary: list[str] | None = None,
    region_band_cfg: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    target_vars = [str(v) for v in (vars_for_summary or ["Te", "phi"])]
    rows: list[dict[str, float | str]] = []
    if distance_signed is None and (mask_plasma is None or distance_any is None):
        return rows
    bands_cfg = dict(region_band_cfg or {})
    band_mode = str(bands_cfg.get("mode", "fixed_px")).strip().lower()
    boundary_in_px = float(bands_cfg.get("boundary_in_px", 2.0))
    deep_plasma_px = float(bands_cfg.get("deep_plasma_px", 10.0))
    boundary_q = float(bands_cfg.get("boundary_q", 0.15))
    deep_q = float(bands_cfg.get("deep_q", 0.70))
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
    type_masks = (
        build_boundary_type_masks(
            mask_plasma=mask_plasma if mask_plasma is not None else region_masks["all_plasma"].astype(np.float32),
            distance_any=distance_any if distance_any is not None else np.abs(np.asarray(distance_signed, dtype=np.float32)),
            band_px=boundary_in_px,
            bc_dir_mask=bc_dir_mask,
            wafer_mask=wafer_mask,
        )
        if boundary_type_breakdown
        else {}
    )
    region_names = ["boundary_in", "plasma_mid", "plasma_deep", "chamber_near", "chamber_far", "all_plasma"]
    for name in target_vars:
        if name not in true_eval or name not in pred_eval:
            continue
        for region_name in region_names:
            region_mask = np.asarray(region_masks[region_name], dtype=bool)
            boundary_types = ["na"]
            if boundary_type_breakdown and region_name == "boundary_in":
                boundary_types = ["interface", "bc_dir", "wafer"]
            for boundary_type in boundary_types:
                region_mask_eff = region_mask if boundary_type == "na" else type_masks.get(boundary_type, region_mask)
                n_points = int(np.sum(np.asarray(region_mask_eff, dtype=np.float32) > 0.5))
                if n_points <= 0:
                    rows.append(
                        {
                            "var": str(name),
                            "region": str(region_name),
                            "boundary_type": str(boundary_type),
                            "n_points": 0.0,
                            "rmse": float("nan"),
                            "r2": float("nan"),
                        }
                    )
                    continue
                rows.append(
                    {
                        "var": str(name),
                        "region": str(region_name),
                        "boundary_type": str(boundary_type),
                        "n_points": float(n_points),
                        "rmse": float(rmse_masked(true_eval[name], pred_eval[name], region_mask_eff)),
                        "r2": float(r2_masked(true_eval[name], pred_eval[name], region_mask_eff)),
                    }
                )
    return rows


def build_spatial_error_by_case_rows(
    *,
    pred_eval: dict[str, np.ndarray],
    true_eval: dict[str, np.ndarray],
    mask_plasma: np.ndarray | None,
    distance_signed: np.ndarray | None,
    distance_any: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    boundary_type_breakdown: bool = False,
    vars_for_summary: list[str] | None = None,
    case_ids: list[str] | None = None,
    region_band_cfg: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    target_vars = [str(v) for v in (vars_for_summary or ["Te", "phi"])]
    rows: list[dict[str, float | str]] = []
    if distance_signed is None and (mask_plasma is None or distance_any is None):
        return rows
    n_cases = int(next(iter(true_eval.values())).shape[0]) if true_eval else 0
    if n_cases <= 0:
        return rows
    bands_cfg = dict(region_band_cfg or {})
    band_mode = str(bands_cfg.get("mode", "fixed_px")).strip().lower()
    boundary_in_px = float(bands_cfg.get("boundary_in_px", 2.0))
    deep_plasma_px = float(bands_cfg.get("deep_plasma_px", 10.0))
    boundary_q = float(bands_cfg.get("boundary_q", 0.15))
    deep_q = float(bands_cfg.get("deep_q", 0.70))
    d_signed = np.asarray(distance_signed, dtype=np.float32)
    if d_signed.ndim == 2:
        d_signed = np.repeat(d_signed[None, ...], n_cases, axis=0)
    elif d_signed.ndim == 3 and int(d_signed.shape[0]) == 1 and n_cases > 1:
        d_signed = np.repeat(d_signed, n_cases, axis=0)
    if d_signed.ndim != 3 or int(d_signed.shape[0]) != n_cases:
        raise ValueError(
            f"distance_signed shape must be [H,W] or [N,H,W] for by-case audit; got {tuple(d_signed.shape)}"
        )
    if mask_plasma is None:
        plasma = d_signed >= 0.0
    else:
        plasma = np.asarray(mask_plasma, dtype=np.float32)
        if plasma.ndim == 2:
            plasma = np.repeat(plasma[None, ...], n_cases, axis=0)
        elif plasma.ndim == 3 and int(plasma.shape[0]) == 1 and n_cases > 1:
            plasma = np.repeat(plasma, n_cases, axis=0)
        if plasma.ndim != 3 or int(plasma.shape[0]) != n_cases:
            raise ValueError(
                f"mask_plasma shape must be [H,W] or [N,H,W] for by-case audit; got {tuple(plasma.shape)}"
            )
        plasma = plasma > 0.5

    case_keys = [str(v) for v in list(case_ids or [])]
    if len(case_keys) != n_cases:
        case_keys = [str(i) for i in range(n_cases)]
    d_any = None
    if distance_any is not None:
        d_any = np.asarray(distance_any, dtype=np.float32)
        if d_any.ndim == 2:
            d_any = np.repeat(d_any[None, ...], n_cases, axis=0)
        elif d_any.ndim == 3 and int(d_any.shape[0]) == 1 and n_cases > 1:
            d_any = np.repeat(d_any, n_cases, axis=0)
    region_names = ["boundary_in", "plasma_mid", "plasma_deep", "chamber_near", "chamber_far", "all_plasma"]
    for case_idx in range(n_cases):
        d_case = d_signed[case_idx]
        plasma_case = plasma[case_idx]
        region_masks = build_region_masks(
            mask_plasma=plasma_case.astype(np.float32),
            distance_any=(d_any[case_idx] if d_any is not None else None),
            distance_signed=d_case,
            mode=band_mode,
            boundary_in_px=boundary_in_px,
            mid_plasma_px=deep_plasma_px,
            deep_plasma_px=deep_plasma_px,
            boundary_q=boundary_q,
            deep_q=deep_q,
        )
        type_masks = (
            build_boundary_type_masks(
                mask_plasma=plasma_case.astype(np.float32),
                distance_any=(d_any[case_idx] if d_any is not None else np.abs(d_case)),
                band_px=boundary_in_px,
                bc_dir_mask=(np.asarray(bc_dir_mask, dtype=np.float32) if bc_dir_mask is not None else None),
                wafer_mask=(np.asarray(wafer_mask, dtype=np.float32) if wafer_mask is not None else None),
            )
            if boundary_type_breakdown
            else {}
        )
        for name in target_vars:
            if name not in true_eval or name not in pred_eval:
                continue
            y_true_case = np.asarray(true_eval[name][case_idx], dtype=np.float32)
            y_pred_case = np.asarray(pred_eval[name][case_idx], dtype=np.float32)
            for region_name in region_names:
                region_mask = np.asarray(region_masks[region_name], dtype=bool)
                boundary_types = ["na"]
                if boundary_type_breakdown and region_name == "boundary_in":
                    boundary_types = ["interface", "bc_dir", "wafer"]
                for boundary_type in boundary_types:
                    region_mask_eff = region_mask if boundary_type == "na" else type_masks.get(boundary_type, region_mask)
                    n_points = int(np.sum(np.asarray(region_mask_eff, dtype=np.float32) > 0.5))
                    if n_points <= 0:
                        rows.append(
                            {
                                "case_id": str(case_keys[case_idx]),
                                "case_index": float(case_idx),
                                "var": str(name),
                                "region": str(region_name),
                                "boundary_type": str(boundary_type),
                                "n_points": 0.0,
                                "rmse": float("nan"),
                                "r2": float("nan"),
                            }
                        )
                        continue
                    rows.append(
                        {
                            "case_id": str(case_keys[case_idx]),
                            "case_index": float(case_idx),
                            "var": str(name),
                            "region": str(region_name),
                            "boundary_type": str(boundary_type),
                            "n_points": float(n_points),
                            "rmse": float(rmse_masked(y_true_case, y_pred_case, region_mask_eff)),
                            "r2": float(r2_masked(y_true_case, y_pred_case, region_mask_eff)),
                        }
                    )
    return rows


def _field_as_nhw(values: np.ndarray, *, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if int(arr.shape[1]) < 1:
            raise ValueError(f"{key} must include at least one channel")
        return arr[:, 0, :, :]
    raise ValueError(f"{key} must be [H,W], [N,H,W], or [N,C,H,W], got={tuple(arr.shape)}")


def _mask_as_nhw(mask: np.ndarray | None, *, n_cases: int, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones((n_cases, *shape), dtype=bool)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, ...], n_cases, axis=0)
    elif arr.ndim == 3 and int(arr.shape[0]) == 1 and n_cases > 1:
        arr = np.repeat(arr, n_cases, axis=0)
    if arr.ndim != 3 or int(arr.shape[0]) != n_cases or tuple(arr.shape[1:]) != shape:
        raise ValueError(f"mask shape must be [H,W] or [N,H,W], got={tuple(arr.shape)}")
    return arr > 0.5


def _safe_rel_abs(pred: float, true: float, *, eps: float) -> float:
    p = float(pred)
    t = float(true)
    if not np.isfinite(p) or not np.isfinite(t):
        return float("nan")
    return float(abs(p - t) / max(abs(t), float(eps)))


def _masked_flat(true_field: np.ndarray, pred_field: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(mask, dtype=bool) & np.isfinite(true_field) & np.isfinite(pred_field)
    return np.asarray(true_field[active], dtype=np.float64), np.asarray(pred_field[active], dtype=np.float64)


def _peak_location(field: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    active = np.asarray(mask, dtype=bool) & np.isfinite(field)
    if not np.any(active):
        return None
    masked = np.where(active, np.asarray(field, dtype=np.float64), -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(y), float(x)


def _center_of_mass(field: np.ndarray, mask: np.ndarray, *, eps: float) -> tuple[float, float] | None:
    active = np.asarray(mask, dtype=bool) & np.isfinite(field)
    if not np.any(active):
        return None
    vals = np.where(active, np.maximum(np.asarray(field, dtype=np.float64), 0.0), 0.0)
    denom = float(np.sum(vals))
    if denom <= eps:
        vals = np.where(active, np.abs(np.asarray(field, dtype=np.float64)), 0.0)
        denom = float(np.sum(vals))
    if denom <= eps:
        return None
    yy, xx = np.indices(vals.shape, dtype=np.float64)
    return float(np.sum(yy * vals) / denom), float(np.sum(xx * vals) / denom)


def _profile_rmse(true_field: np.ndarray, pred_field: np.ndarray, mask: np.ndarray, *, reduce_axis: int) -> float:
    active = np.asarray(mask, dtype=bool) & np.isfinite(true_field) & np.isfinite(pred_field)
    counts = np.sum(active, axis=reduce_axis).astype(np.float64)
    valid = counts > 0.0
    if not np.any(valid):
        return float("nan")
    true_sum = np.sum(np.where(active, true_field, 0.0), axis=reduce_axis).astype(np.float64)
    pred_sum = np.sum(np.where(active, pred_field, 0.0), axis=reduce_axis).astype(np.float64)
    true_prof = true_sum[valid] / counts[valid]
    pred_prof = pred_sum[valid] / counts[valid]
    return float(np.sqrt(np.mean((pred_prof - true_prof) ** 2)))


def _grad_rmse(true_field: np.ndarray, pred_field: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool) & np.isfinite(true_field) & np.isfinite(pred_field)
    if not np.any(active):
        return float("nan")
    true_y, true_x = np.gradient(np.asarray(true_field, dtype=np.float64), edge_order=1)
    pred_y, pred_x = np.gradient(np.asarray(pred_field, dtype=np.float64), edge_order=1)
    err2 = (pred_y - true_y) ** 2 + (pred_x - true_x) ** 2
    return float(np.sqrt(np.mean(err2[active])))


def _top_fraction_rmse(true_field: np.ndarray, pred_field: np.ndarray, mask: np.ndarray, *, fraction: float) -> float:
    true_vals, _ = _masked_flat(true_field, pred_field, mask)
    if true_vals.size == 0:
        return float("nan")
    frac = float(np.clip(fraction, 1.0e-6, 1.0))
    threshold = float(np.percentile(true_vals, 100.0 * (1.0 - frac)))
    active = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(true_field)
        & np.isfinite(pred_field)
        & (np.asarray(true_field, dtype=np.float64) >= threshold)
    )
    if not np.any(active):
        return float("nan")
    err = np.asarray(pred_field, dtype=np.float64)[active] - np.asarray(true_field, dtype=np.float64)[active]
    return float(np.sqrt(np.mean(err * err)))


def _shape_corr(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    if true_vals.size < 2 or pred_vals.size < 2:
        return float("nan")
    t_std = float(np.std(true_vals))
    p_std = float(np.std(pred_vals))
    if t_std <= 1.0e-12 or p_std <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(true_vals, pred_vals)[0, 1])


def build_spatial_distribution_by_case_rows(
    *,
    pred_eval: dict[str, np.ndarray],
    true_eval: dict[str, np.ndarray],
    mask_plasma: np.ndarray | None,
    vars_for_summary: list[str] | None = None,
    case_ids: list[str] | None = None,
    target_region_by_var: dict[str, str] | None = None,
    top_fraction: float = 0.10,
    eps: float = 1.0e-12,
) -> list[dict[str, float | str]]:
    """Build case-level distribution-shape metrics beyond pointwise R2/RMSE."""

    target_vars = [str(v) for v in (vars_for_summary or sorted(set(pred_eval.keys()) & set(true_eval.keys())))]
    region_by_var = normalize_target_region_by_var(target_region_by_var, target_vars=target_vars)
    rows: list[dict[str, float | str]] = []
    for name in target_vars:
        if name not in true_eval or name not in pred_eval:
            continue
        true_arr = _field_as_nhw(true_eval[name], key=f"true_eval[{name}]")
        pred_arr = _field_as_nhw(pred_eval[name], key=f"pred_eval[{name}]")
        if true_arr.shape != pred_arr.shape:
            raise ValueError(f"distribution metric shape mismatch for {name}: {true_arr.shape} vs {pred_arr.shape}")
        n_cases = int(true_arr.shape[0])
        h, w = int(true_arr.shape[1]), int(true_arr.shape[2])
        target_region = target_region_for_var(region_by_var, name)
        masks = (
            np.ones((n_cases, h, w), dtype=bool)
            if target_region == "all_domain"
            else _mask_as_nhw(mask_plasma, n_cases=n_cases, shape=(h, w))
        )
        case_keys = [str(v) for v in list(case_ids or [])]
        if len(case_keys) != n_cases:
            case_keys = [str(i) for i in range(n_cases)]
        for case_idx in range(n_cases):
            t = np.asarray(true_arr[case_idx], dtype=np.float64)
            p = np.asarray(pred_arr[case_idx], dtype=np.float64)
            m = masks[case_idx]
            t_vals, p_vals = _masked_flat(t, p, m)
            n_points = int(t_vals.size)
            if n_points <= 0:
                rows.append(
                    {
                        "case_id": str(case_keys[case_idx]),
                        "case_index": float(case_idx),
                        "var": str(name),
                        "target_region": str(target_region),
                        "n_points": 0.0,
                        "integral_true": float("nan"),
                        "integral_pred": float("nan"),
                        "integral_rel_error": float("nan"),
                        "p95_true": float("nan"),
                        "p95_pred": float("nan"),
                        "p95_rel_error": float("nan"),
                        "p99_true": float("nan"),
                        "p99_pred": float("nan"),
                        "p99_rel_error": float("nan"),
                        "peak_location_error_px": float("nan"),
                        "center_of_mass_error_px": float("nan"),
                        "profile_rmse_r": float("nan"),
                        "profile_rmse_z": float("nan"),
                        "grad_rmse": float("nan"),
                        "top10_rmse": float("nan"),
                        "shape_corr": float("nan"),
                        "distribution_error_score": float("nan"),
                    }
                )
                continue
            integral_true = float(np.sum(t_vals))
            integral_pred = float(np.sum(p_vals))
            p95_true = float(np.percentile(t_vals, 95.0))
            p95_pred = float(np.percentile(p_vals, 95.0))
            p99_true = float(np.percentile(t_vals, 99.0))
            p99_pred = float(np.percentile(p_vals, 99.0))
            peak_t = _peak_location(t, m)
            peak_p = _peak_location(p, m)
            peak_err = (
                float(np.hypot(float(peak_p[0]) - float(peak_t[0]), float(peak_p[1]) - float(peak_t[1])))
                if peak_t is not None and peak_p is not None
                else float("nan")
            )
            com_t = _center_of_mass(t, m, eps=eps)
            com_p = _center_of_mass(p, m, eps=eps)
            com_err = (
                float(np.hypot(float(com_p[0]) - float(com_t[0]), float(com_p[1]) - float(com_t[1])))
                if com_t is not None and com_p is not None
                else float("nan")
            )
            profile_r = _profile_rmse(t, p, m, reduce_axis=0)
            profile_z = _profile_rmse(t, p, m, reduce_axis=1)
            grad = _grad_rmse(t, p, m)
            top_rmse = _top_fraction_rmse(t, p, m, fraction=top_fraction)
            corr = _shape_corr(t_vals, p_vals)
            true_std = float(np.std(t_vals))
            top_nrmse = float(top_rmse / max(true_std, eps)) if np.isfinite(top_rmse) else float("nan")
            score_parts = [
                _safe_rel_abs(integral_pred, integral_true, eps=eps),
                _safe_rel_abs(p99_pred, p99_true, eps=eps),
                top_nrmse,
                float(peak_err / max(h, w)) if np.isfinite(peak_err) else float("nan"),
                float(com_err / max(h, w)) if np.isfinite(com_err) else float("nan"),
                float(max(0.0, 1.0 - corr)) if np.isfinite(corr) else float("nan"),
            ]
            finite_parts = [v for v in score_parts if np.isfinite(float(v))]
            rows.append(
                {
                    "case_id": str(case_keys[case_idx]),
                    "case_index": float(case_idx),
                    "var": str(name),
                    "target_region": str(target_region),
                    "n_points": float(n_points),
                    "integral_true": integral_true,
                    "integral_pred": integral_pred,
                    "integral_rel_error": _safe_rel_abs(integral_pred, integral_true, eps=eps),
                    "p95_true": p95_true,
                    "p95_pred": p95_pred,
                    "p95_rel_error": _safe_rel_abs(p95_pred, p95_true, eps=eps),
                    "p99_true": p99_true,
                    "p99_pred": p99_pred,
                    "p99_rel_error": _safe_rel_abs(p99_pred, p99_true, eps=eps),
                    "peak_location_error_px": peak_err,
                    "center_of_mass_error_px": com_err,
                    "profile_rmse_r": profile_r,
                    "profile_rmse_z": profile_z,
                    "grad_rmse": grad,
                    "top10_rmse": top_rmse,
                    "shape_corr": corr,
                    "distribution_error_score": float(np.mean(finite_parts)) if finite_parts else float("nan"),
                }
            )
    return rows


def build_spatial_distribution_summary_rows(
    by_case_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    metric_names = [
        "integral_rel_error",
        "p95_rel_error",
        "p99_rel_error",
        "peak_location_error_px",
        "center_of_mass_error_px",
        "profile_rmse_r",
        "profile_rmse_z",
        "grad_rmse",
        "top10_rmse",
        "shape_corr",
        "distribution_error_score",
    ]
    rows: list[dict[str, float | str]] = []
    vars_seen = sorted({str(r.get("var", "")) for r in by_case_rows if str(r.get("var", ""))})
    for name in vars_seen:
        var_rows = [r for r in by_case_rows if str(r.get("var", "")) == name]
        regions_seen = sorted({str(r.get("target_region", "")) for r in var_rows if str(r.get("target_region", ""))})
        row: dict[str, float | str] = {
            "var": name,
            "target_region": regions_seen[0] if len(regions_seen) == 1 else "|".join(regions_seen),
            "n_cases": float(len(var_rows)),
        }
        for metric_name in metric_names:
            values = np.asarray(
                [float(r.get(metric_name, float("nan"))) for r in var_rows if np.isfinite(float(r.get(metric_name, float("nan"))))],
                dtype=np.float64,
            )
            if values.size == 0:
                row[f"{metric_name}_mean"] = float("nan")
                row[f"{metric_name}_median"] = float("nan")
                row[f"{metric_name}_p90"] = float("nan")
                row[f"{metric_name}_max"] = float("nan")
                continue
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_median"] = float(np.median(values))
            row[f"{metric_name}_p90"] = float(np.percentile(values, 90.0))
            row[f"{metric_name}_max"] = float(np.max(values))
        rows.append(row)
    return rows


def build_eval_metrics_payload(
    *,
    true_eval: dict[str, np.ndarray],
    pred_eval: dict[str, np.ndarray],
    eps: np.ndarray | None = None,
    mask_plasma: np.ndarray | None = None,
) -> dict[str, Any]:
    keys = [k for k in true_eval.keys() if k in pred_eval]
    rmse = rmse_by_var({k: true_eval[k] for k in keys}, {k: pred_eval[k] for k in keys})
    r2 = r2_by_var({k: true_eval[k] for k in keys}, {k: pred_eval[k] for k in keys})
    rmse_plasma: dict[str, float] = {}
    r2_plasma: dict[str, float] = {}
    if mask_plasma is not None:
        for k in keys:
            rmse_plasma[k] = float(rmse_masked(true_eval[k], pred_eval[k], mask_plasma))
            r2_plasma[k] = float(r2_masked(true_eval[k], pred_eval[k], mask_plasma))
    if "phi" in pred_eval:
        rmse["phi_poisson_residual"] = float(poisson_residual_loss(pred_eval["phi"][:, 0]))
        rmse["phi_poisson_residual_norm"] = float(poisson_residual_norm(pred_eval["phi"][:, 0], eps=eps))
    return {"rmse": rmse, "r2": r2, "rmse_plasma": rmse_plasma, "r2_plasma": r2_plasma}


def build_viz_tables_payload(
    *,
    diag_rows: list[dict[str, float | str]],
    region_rows: list[dict[str, float | str]],
) -> dict[str, Any]:
    diag_header = [
        "case_key",
        "poisson_residual_norm",
        "bc_phi_mae",
        "uniformity",
        "poisson_residual_map_l2",
        "boundary_residual_map_l2",
    ]
    region_header = [
        "case_key",
        "density_key",
        "plasma_mean_density",
        "bulk_mean_density",
        "boundary_mean_density",
    ]
    return {
        "diag_header": diag_header,
        "diag_rows": [[row.get(h, 0.0) for h in diag_header] for row in diag_rows],
        "region_header": region_header,
        "region_rows": [[row.get(h, 0.0) for h in region_header] for row in region_rows],
    }
