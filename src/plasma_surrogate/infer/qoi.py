"""QoI computation for single-case inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.eval.metrics import boundary_band_mask, boundary_gamma_proxy, uniformity
from plasma_surrogate.eval.positive_diagnostics import positive_targets_from_schema


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _config_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    return [str(v).strip() for v in list(raw) if str(v).strip()]


def axisymmetric_weighted_cv(values: np.ndarray, radial_weights: np.ndarray) -> tuple[float, float]:
    """Return area-weighted mean and CV for an axisymmetric radial sample."""

    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(radial_weights, dtype=np.float64).reshape(-1)
    if vals.size == 0 or vals.shape != weights.shape:
        return float("nan"), float("nan")
    if np.any(~np.isfinite(vals)) or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        return float("nan"), float("nan")
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan"), float("nan")
    mean = float(np.sum(weights * vals) / denom)
    variance = float(np.sum(weights * (vals - mean) ** 2) / denom)
    return mean, float(np.sqrt(max(variance, 0.0)) / (abs(mean) + 1.0e-12))


def pick_uniformity_target(
    fields_phys: dict[str, np.ndarray],
    *,
    preferred_keys: list[str] | None = None,
    target_role_schema: dict[str, Any] | None = None,
) -> tuple[str | None, np.ndarray | None]:
    excluded = {"rho_eff", "e_theta", "e_mag", "phi_raw", "phi_refined"}
    configured = [str(v).strip() for v in (preferred_keys or []) if str(v).strip()]
    for key in configured:
        if key in fields_phys and key not in excluded:
            return key, np.asarray(fields_phys[key], dtype=np.float32)[0]
    if configured:
        raise ValueError(
            "qoi.uniformity.target did not match available prediction fields; "
            f"configured={configured}; available={sorted(k for k in fields_phys if k not in excluded)}"
        )
    positive_targets = [
        key
        for key in positive_targets_from_schema(target_role_schema)
        if key in fields_phys and key not in excluded
    ]
    unique = sorted(set(positive_targets))
    if len(unique) == 1:
        key = unique[0]
        return key, np.asarray(fields_phys[key], dtype=np.float32)[0]
    if len(unique) > 1:
        raise ValueError(
            "qoi.uniformity.target is ambiguous because multiple positive targets are available; "
            f"positive_targets={unique}. Set inference.qoi.uniformity.target explicitly."
        )
    raise ValueError(
        "qoi.uniformity.target is required unless target_role_schema.positive_targets resolves "
        "to exactly one available target."
    )


def uniformity_values_for_region(
    qoi_target: np.ndarray,
    *,
    mask_plasma: np.ndarray,
    wafer_mask: np.ndarray | None,
    region: str,
    mid_height_band_px: int = 0,
    row_index: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    target = np.asarray(qoi_target, dtype=np.float32)
    plasma = np.asarray(mask_plasma, dtype=np.float32) > 0.5
    region_norm = str(region or "auto").strip().lower()
    if region_norm in {"fixed_row", "row_range", "z_row", "z_line"}:
        h, w = target.shape
        if row_index is None:
            raise ValueError("qoi.uniformity.row_index is required for fixed_row uniformity")
        row = int(row_index)
        if row < 0 or row >= h:
            raise ValueError(f"qoi.uniformity.row_index must be within [0, {h - 1}]")
        c0 = 0 if col_start is None else int(col_start)
        c1 = (w - 1) if col_end is None else int(col_end)
        if c0 < 0 or c1 < c0 or c0 >= w:
            raise ValueError(f"qoi.uniformity column range must overlap [0, {w - 1}]")
        c1 = min(c1, w - 1)
        vals = target[row, c0 : c1 + 1]
        return vals, {
            "uniformity_region": "fixed_row",
            "uniformity_row_index": float(row),
            "uniformity_col_start": float(c0),
            "uniformity_col_end": float(c1),
            "uniformity_sample_count": float(vals.size),
        }
    if region_norm in {
        "plasma_mid_height",
        "mid_height",
        "plasma_midline",
        "plasma_mean_height",
        "mean_height",
    }:
        active_rows = np.where(np.any(plasma, axis=1))[0]
        if active_rows.size <= 0:
            region_label = (
                "plasma_mean_height" if region_norm in {"plasma_mean_height", "mean_height"} else "plasma_mid_height"
            )
            return np.asarray([], dtype=np.float32), {
                "uniformity_region": region_label,
                "uniformity_mid_height_row": -1.0,
                "uniformity_mean_height_row": -1.0,
                "uniformity_sample_count": 0.0,
            }
        if region_norm in {"plasma_mean_height", "mean_height"}:
            yy = np.nonzero(plasma)[0].astype(np.float64)
            target_mid = float(np.mean(yy))
            region_label = "plasma_mean_height"
        else:
            target_mid = 0.5 * (float(active_rows[0]) + float(active_rows[-1]))
            region_label = "plasma_mid_height"
        mid_row = int(active_rows[int(np.argmin(np.abs(active_rows.astype(np.float64) - target_mid)))])
        band = max(0, int(mid_height_band_px))
        y0 = max(0, mid_row - band)
        y1 = min(int(plasma.shape[0]), mid_row + band + 1)
        selector = plasma[y0:y1]
        vals = target[y0:y1][selector]
        return vals, {
            "uniformity_region": region_label,
            "uniformity_mid_height_row": float(mid_row),
            "uniformity_mean_height_row": float(target_mid),
            "uniformity_sample_count": float(vals.size),
        }
    if region_norm in {"wafer_near", "wafer_near_band"}:
        if wafer_mask is None:
            return np.asarray([], dtype=np.float32), {
                "uniformity_region": "wafer_near",
                "uniformity_sample_count": 0.0,
            }
        wafer = np.asarray(wafer_mask, dtype=np.float32) > 0.5
        layers = max(1, int(mid_height_band_px))
        overlap = wafer & plasma
        if np.any(overlap):
            # ICP wafer_mask marks the volume below the wafer surface. Select
            # only its top in-plasma layers, independently in each r column.
            selector = np.zeros_like(overlap, dtype=bool)
            for col in range(overlap.shape[1]):
                active = np.flatnonzero(overlap[:, col])
                if active.size == 0:
                    continue
                top = int(active[-1])
                selector[max(0, top - layers + 1) : top + 1, col] = overlap[
                    max(0, top - layers + 1) : top + 1, col
                ]
        else:
            expanded = wafer
            for _ in range(layers):
                padded = np.pad(expanded, 1, mode="constant", constant_values=False)
                expanded = (
                    expanded
                    | padded[:-2, 1:-1]
                    | padded[2:, 1:-1]
                    | padded[1:-1, :-2]
                    | padded[1:-1, 2:]
                )
            selector = expanded & plasma
        vals = target[selector]
        return vals, {
            "uniformity_region": "wafer_near",
            "uniformity_wafer_near_layers": float(layers),
            "uniformity_sample_count": float(vals.size),
        }
    if region_norm == "plasma":
        vals = target[plasma]
        return vals, {"uniformity_region": "plasma", "uniformity_sample_count": float(vals.size)}
    if wafer_mask is not None and region_norm in {"auto", "wafer"}:
        wafer = np.asarray(wafer_mask, dtype=np.float32) > 0.5
        vals = target[wafer]
        if vals.size > 0 or region_norm == "wafer":
            return vals, {"uniformity_region": "wafer", "uniformity_sample_count": float(vals.size)}
    vals = target[plasma]
    return vals, {"uniformity_region": "plasma", "uniformity_sample_count": float(vals.size)}


def compute_inference_qoi(
    *,
    fields_phys: dict[str, np.ndarray],
    geom_ctx: GeometryContext,
    potential_field: np.ndarray | None,
    ood_cfg: dict[str, Any],
    target_role_schema: dict[str, Any],
    resolve_boundary_operator_inputs: Any,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray] | None, dict[str, Any]]:
    wafer = geom_ctx.regions.get("wafer_mask")
    qoi_cfg = _dict_or_empty(ood_cfg.get("qoi"))
    uniformity_cfg = _dict_or_empty(qoi_cfg.get("uniformity"))
    preferred_uniformity_keys: list[str] = []
    preferred_uniformity_keys.extend(_config_string_list(uniformity_cfg.get("target")))
    preferred_uniformity_keys.extend(_config_string_list(uniformity_cfg.get("preferred_targets")))
    preferred_uniformity_keys.extend(_config_string_list(ood_cfg.get("uniformity_target")))
    qoi_target_key, qoi_target = pick_uniformity_target(
        fields_phys,
        preferred_keys=preferred_uniformity_keys,
        target_role_schema=target_role_schema,
    )

    region_raw = uniformity_cfg.get("region")
    if region_raw is None:
        region_raw = ood_cfg.get("uniformity_region")
    if region_raw is None:
        region_raw = "auto"
    band_px_raw = uniformity_cfg.get("mid_height_band_px")
    if band_px_raw is None:
        band_px_raw = ood_cfg.get("mid_height_band_px")
    if band_px_raw is None:
        band_px_raw = 0
    row_index_raw = uniformity_cfg.get("row_index", uniformity_cfg.get("z_index"))
    col_start_raw = uniformity_cfg.get("col_start", uniformity_cfg.get("r_start"))
    col_end_raw = uniformity_cfg.get("col_end", uniformity_cfg.get("r_end"))
    vals, uniformity_meta = uniformity_values_for_region(
        qoi_target,
        mask_plasma=geom_ctx.mask_plasma,
        wafer_mask=wafer,
        region=str(region_raw),
        mid_height_band_px=int(band_px_raw or 0),
        row_index=None if row_index_raw is None else int(row_index_raw),
        col_start=None if col_start_raw is None else int(col_start_raw),
        col_end=None if col_end_raw is None else int(col_end_raw),
    )
    relative_uniformity = uniformity(vals)
    vals64 = np.asarray(vals, dtype=np.float64).reshape(-1)
    if vals64.size > 0 and np.all(np.isfinite(vals64)):
        mean_density = float(np.mean(vals64))
        min_density = float(np.min(vals64))
        max_density = float(np.max(vals64))
        p95_density = float(np.percentile(vals64, 95.0))
    else:
        mean_density = float("nan")
        min_density = float("nan")
        max_density = float("nan")
        p95_density = float("nan")

    score_mode_raw = uniformity_cfg.get("score_mode")
    if score_mode_raw is None:
        score_mode_raw = ood_cfg.get("uniformity_score_mode")
    if score_mode_raw is None:
        score_mode_raw = "relative"
    score_mode = str(score_mode_raw).strip().lower()
    if score_mode not in {"relative", "cv"}:
        raise ValueError("ood.uniformity_score_mode must be one of: relative, cv")
    qoi = {
        "uniformity": float(relative_uniformity),
        "uniformity_relative": float(relative_uniformity),
        "uniformity_mean_density": mean_density,
        "uniformity_min_density": min_density,
        "uniformity_max_density": max_density,
        "uniformity_p95_density": p95_density,
        "uniformity_score_mode": score_mode,
        "uniformity_target": str(qoi_target_key),
    }
    qoi.update(uniformity_meta)

    bohm_qoi_cfg = _dict_or_empty(qoi_cfg.get("bohm_flux"))
    density_key = str(bohm_qoi_cfg.get("density_target", "ni")).strip()
    temperature_key = str(bohm_qoi_cfg.get("temperature_target", "Te")).strip()
    missing_bohm = [key for key in (density_key, temperature_key) if key not in fields_phys]
    if missing_bohm and bohm_qoi_cfg:
        raise ValueError(
            "qoi.bohm_flux targets did not match prediction fields; "
            f"missing={missing_bohm}, available={sorted(fields_phys)}"
        )
    if not missing_bohm:
        density_field = np.asarray(fields_phys[density_key], dtype=np.float32)
        temperature_field = np.asarray(fields_phys[temperature_key], dtype=np.float32)
        if density_field.ndim == 3:
            density_field = density_field[0]
        if temperature_field.ndim == 3:
            temperature_field = temperature_field[0]
        bohm_flux = np.maximum(density_field, 0.0) * np.sqrt(np.maximum(temperature_field, 0.0) + 1.0e-6)
        bohm_vals, _ = uniformity_values_for_region(
            bohm_flux,
            mask_plasma=geom_ctx.mask_plasma,
            wafer_mask=wafer,
            region=str(region_raw),
            mid_height_band_px=int(band_px_raw or 0),
            row_index=None if row_index_raw is None else int(row_index_raw),
            col_start=None if col_start_raw is None else int(col_start_raw),
            col_end=None if col_end_raw is None else int(col_end_raw),
        )
        qoi["bohm_flux_mean"] = float(np.mean(bohm_vals)) if bohm_vals.size > 0 else 0.0
        qoi["bohm_flux_uniformity"] = uniformity(bohm_vals)
        coord = np.asarray(geom_ctx.coord_grid, dtype=np.float32)
        if coord.shape != (2, *bohm_flux.shape):
            raise ValueError(
                "Bohm flux axisymmetric weighting requires coord_grid [2,H,W], "
                f"got={coord.shape}"
            )
        radial_vals, _ = uniformity_values_for_region(
            coord[0],
            mask_plasma=geom_ctx.mask_plasma,
            wafer_mask=wafer,
            region=str(region_raw),
            mid_height_band_px=int(band_px_raw or 0),
            row_index=None if row_index_raw is None else int(row_index_raw),
            col_start=None if col_start_raw is None else int(col_start_raw),
            col_end=None if col_end_raw is None else int(col_end_raw),
        )
        area_mean, area_cv = axisymmetric_weighted_cv(bohm_vals, radial_vals)
        qoi["bohm_flux_area_mean"] = float(area_mean)
        qoi["bohm_flux_area_cv"] = float(area_cv)
        qoi["bohm_flux_density_target"] = density_key
        qoi["bohm_flux_temperature_target"] = temperature_key

    bo_cfg = _dict_or_empty(ood_cfg.get("boundary_operator"))
    band_mask = boundary_band_mask(
        mask_plasma=geom_ctx.mask_plasma,
        distance_any=geom_ctx.distance_any,
        delta_edge=float(bo_cfg.get("delta_edge", 1.5)),
        wafer_mask=wafer,
        wafer_only=bool(bo_cfg.get("wafer_only", False)),
    )
    try:
        op_inputs = resolve_boundary_operator_inputs(fields_phys)
    except ValueError:
        if potential_field is not None:
            raise
        op_inputs = None
    if op_inputs is not None:
        gamma_vals = boundary_gamma_proxy(
            density=op_inputs["density"][0],
            te=op_inputs["temperature"][0],
            phi=op_inputs["potential"][0],
            mask_band=band_mask,
        )
        if gamma_vals.size > 0:
            qoi["boundary_gamma_mean"] = float(np.mean(gamma_vals))
            qoi["boundary_gamma_uniformity"] = uniformity(gamma_vals)
        else:
            qoi["boundary_gamma_mean"] = 0.0
            qoi["boundary_gamma_uniformity"] = 0.0
    return qoi, band_mask, op_inputs, bo_cfg


__all__ = [
    "axisymmetric_weighted_cv",
    "compute_inference_qoi",
    "pick_uniformity_target",
    "uniformity_values_for_region",
]
