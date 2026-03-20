"""Shared metrics-row builders for viz and benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.density_contract import resolve_density_key
from plasma_surrogate.core.spatial_regions import build_boundary_type_masks, build_region_masks
from plasma_surrogate.eval.metrics import poisson_residual_norm, r2_by_var, r2_masked, rmse_by_var, rmse_masked
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
    log_ne: np.ndarray | None = None,
    density: np.ndarray | None = None,
    density_key: str = "ne",
) -> dict[str, float | str]:
    density_arr = np.asarray(density if density is not None else log_ne, dtype=np.float32)
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
    opt_best_uniformity: float,
    true_eval: dict[str, np.ndarray] | None = None,
    mask_plasma: np.ndarray | None = None,
    distance_any: np.ndarray | None = None,
    distance_signed: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    aggregate_cfg: dict[str, Any] | None = None,
    target_vars_for_score: list[str] | None = None,
    region_band_cfg: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    r2_scores = dict(r2_scores or {})
    metric_keys = list({*metrics.keys(), *r2_scores.keys()})
    paired_eval_keys = list(set(pred_eval.keys()) & set((true_eval or {}).keys()))
    target_vars_effective = [str(v) for v in list(target_vars_for_score or [])]
    density_ne_metric_key = resolve_density_key(metric_keys, canonical="ne", prefer_linear=True)
    density_ni_metric_key = resolve_density_key(metric_keys, canonical="ni", prefer_linear=True)
    density_ne_eval_key = resolve_density_key(paired_eval_keys, canonical="ne", prefer_linear=True)
    density_ni_eval_key = resolve_density_key(paired_eval_keys, canonical="ni", prefer_linear=True)
    core_eval_keys = list(paired_eval_keys)

    def _metric_val(src: dict[str, float], name: str | None, *, default: float = float("nan")) -> float:
        if name is None:
            return float(default)
        if name not in src:
            return float(default)
        return float(src[name])

    metrics_plasma: dict[str, float] = {}
    r2_plasma: dict[str, float] = {}
    if true_eval is not None and mask_plasma is not None:
        for name in core_eval_keys:
            if name in true_eval and name in pred_eval:
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

    agg = dict(aggregate_cfg or {})
    use_score = bool(agg.get("enabled", False))
    use_plasma = bool(agg.get("use_plasma_metrics", True))
    rmse_weight = float(agg.get("rmse_weight", 0.5))
    r2_weight = float(agg.get("r2_weight", 0.4))
    boundary_penalty_weight = float(agg.get("boundary_penalty_weight", 0.1))
    score_rmse = 0.0
    score_r2 = 0.0
    score_boundary_penalty = 0.0
    score_total = 0.0
    if use_score:
        metric_src = metrics_plasma if (use_plasma and metrics_plasma) else metrics
        r2_src = r2_plasma if (use_plasma and r2_plasma) else r2_scores
        vars_present = [k for k in core_eval_keys if k in metric_src and k in r2_src]
        if vars_present:
            score_rmse = float(np.mean([float(metric_src[k]) for k in vars_present]))
            score_r2 = float(np.mean([float(r2_src[k]) for k in vars_present]))
        score_boundary_penalty = float(
            np.log1p(abs(float(single_diagnostics.get("boundary_operator_proxy_loss", 0.0))))
            + np.log1p(abs(float(single_qoi.get("boundary_gamma_uniformity", 0.0))))
        )
        score_total = (rmse_weight * score_rmse) - (r2_weight * score_r2) + (boundary_penalty_weight * score_boundary_penalty)

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

    row: dict[str, float | str] = {
        "model_id": model_id,
        "continuity_grad_ratio_all_plasma": float(continuity_grad_ratio_all),
        "continuity_lap_ratio_all_plasma": float(continuity_lap_ratio_all),
        "test_poisson_phi": float(poisson_residual_loss(pred_eval["phi"][:, 0])) if "phi" in pred_eval else 0.0,
        "qoi_uniformity": float(single_qoi["uniformity"]),
        "qoi_boundary_gamma_uniformity": float(single_qoi.get("boundary_gamma_uniformity", 0.0)),
        "single_poisson_residual": float(single_diagnostics["poisson_residual_norm"]),
        "single_poisson_residual_map_l2": float(single_diagnostics.get("poisson_residual_map_l2", 0.0)),
        "single_boundary_operator_proxy_loss": float(single_diagnostics.get("boundary_operator_proxy_loss", 0.0)),
        "single_boundary_residual_map_l2": float(single_diagnostics.get("boundary_operator_residual_map_l2", 0.0)),
        "opt_best_uniformity": float(opt_best_uniformity),
        "score_rmse_plasma_mean": float(score_rmse),
        "score_r2_plasma_mean": float(score_r2),
        "score_boundary_penalty": float(score_boundary_penalty),
        "score_total": float(score_total),
    }
    for name in core_eval_keys:
        row[f"test_rmse_{name}"] = _metric_val(metrics, name)
        row[f"test_r2_{name}"] = _metric_val(r2_scores, name)
        row[f"test_rmse_{name}_plasma"] = _metric_val(metrics_plasma, name)
        row[f"test_r2_{name}_plasma"] = _metric_val(r2_plasma, name)
        row[f"test_rmse_{name}_boundary_in"] = _metric_val(boundary_rmse, name)
        row[f"test_r2_{name}_boundary_in"] = _metric_val(boundary_r2, name)
        row[f"test_rmse_{name}_plasma_deep"] = _metric_val(deep_rmse, name)
        row[f"test_r2_{name}_plasma_deep"] = _metric_val(deep_r2, name)
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
    if density_ne_metric_key is not None:
        row["test_rmse_ne"] = _metric_val(metrics, density_ne_metric_key)
        row["test_r2_ne"] = _metric_val(r2_scores, density_ne_metric_key)
        row["test_rmse_ne_plasma"] = _metric_val(metrics_plasma, density_ne_eval_key)
        row["test_r2_ne_plasma"] = _metric_val(r2_plasma, density_ne_eval_key)
    if density_ni_metric_key is not None:
        row["test_rmse_ni"] = _metric_val(metrics, density_ni_metric_key)
        row["test_r2_ni"] = _metric_val(r2_scores, density_ni_metric_key)
        row["test_rmse_ni_plasma"] = _metric_val(metrics_plasma, density_ni_eval_key)
        row["test_r2_ni_plasma"] = _metric_val(r2_plasma, density_ni_eval_key)
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
