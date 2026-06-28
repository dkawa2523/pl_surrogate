"""Spatial error and distribution metric builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.spatial_regions import (
    build_boundary_type_masks,
    build_region_masks,
    normalize_region_by_var,
    target_region_for_var,
)
from plasma_surrogate.eval.metrics import r2_masked, rmse_masked


def _resolve_summary_targets(
    *,
    vars_for_summary: list[str] | None,
    true_eval: dict[str, np.ndarray],
    pred_eval: dict[str, np.ndarray],
) -> list[str]:
    if vars_for_summary is not None:
        return [str(v) for v in vars_for_summary]
    return [str(key) for key in true_eval.keys() if str(key) in pred_eval][:2]


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
    target_vars = _resolve_summary_targets(
        vars_for_summary=vars_for_summary,
        true_eval=true_eval,
        pred_eval=pred_eval,
    )
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
    target_vars = _resolve_summary_targets(
        vars_for_summary=vars_for_summary,
        true_eval=true_eval,
        pred_eval=pred_eval,
    )
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
    region_by_var: dict[str, str] | None = None,
    top_fraction: float = 0.10,
    eps: float = 1.0e-12,
) -> list[dict[str, float | str]]:
    """Build case-level distribution-shape metrics beyond pointwise R2/RMSE."""

    target_vars = [str(v) for v in (vars_for_summary or sorted(set(pred_eval.keys()) & set(true_eval.keys())))]
    region_by_var = normalize_region_by_var(
        region_by_var,
        target_vars=target_vars,
        key_name="eval.diagnostics.distribution.region_by_var",
    )
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


__all__ = [
    "build_spatial_distribution_by_case_rows",
    "build_spatial_distribution_summary_rows",
    "build_spatial_error_by_case_rows",
    "build_spatial_error_summary_rows",
]
