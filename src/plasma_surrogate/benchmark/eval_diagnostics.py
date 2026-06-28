"""Benchmark eval diagnostics writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.spatial_regions import normalize_region_by_var, target_region_for_var
from plasma_surrogate.eval.spatial_metrics import (
    build_spatial_distribution_by_case_rows,
    build_spatial_distribution_summary_rows,
    build_spatial_error_by_case_rows,
    build_spatial_error_summary_rows,
)


def resolve_eval_diagnostics_cfg(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(eval_cfg or {})
    raw = dict(cfg.get("diagnostics", {}) or {})
    spatial = dict(cfg.get("spatial_error_audit", {}) or {})
    spatial.update(dict(raw.get("spatial", {}) or {}))
    distribution = dict(cfg.get("spatial_distribution_audit", {}) or {})
    distribution.update(dict(raw.get("distribution", {}) or {}))
    physics = dict(raw.get("physics", {}) or {})
    enabled_default = bool(
        spatial.get("enabled", False)
        or distribution.get("enabled", False)
        or physics.get("enabled", False)
    )
    enabled = bool(raw.get("enabled", enabled_default))
    if not enabled:
        spatial["enabled"] = False
        distribution["enabled"] = False
        physics["enabled"] = False
    return {
        "enabled": enabled,
        "spatial": spatial,
        "distribution": distribution,
        "physics": physics,
    }


def write_eval_diagnostics(
    *,
    eval_cfg: dict[str, Any],
    model_dir: Path,
    row: dict[str, Any],
    context: Any,
    geom_ctx: Any,
    viz: Any,
    pred_eval: dict[str, Any],
    true_eval: dict[str, Any],
    metric_mask: np.ndarray | None,
    te_idx: np.ndarray,
    target_vars_for_score: list[str],
    region_band_cfg: dict[str, Any],
) -> None:
    diagnostics_cfg = resolve_eval_diagnostics_cfg(eval_cfg)
    pred_spatial = dict(pred_eval)
    true_spatial = dict(true_eval)
    spatial_common_keys = list(set(pred_spatial.keys()) & set(true_spatial.keys()))
    spatial_vars = [v for v in target_vars_for_score if v in set(spatial_common_keys)]
    if not spatial_vars:
        spatial_vars = list(spatial_common_keys)
    eval_case_ids = [
        str(context.dataset.cases[int(i)].get("case_id", int(i)))
        for i in np.asarray(te_idx, dtype=np.int64).tolist()
    ]
    wafer_mask = (
        np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
        if (geom_ctx is not None and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None)
        else None
    )

    spatial_cfg = dict(diagnostics_cfg.get("spatial", {}) or {})
    if bool(spatial_cfg.get("enabled", False)):
        _write_spatial_error_tables(
            model_dir=model_dir,
            pred_spatial=pred_spatial,
            true_spatial=true_spatial,
            geom_ctx=geom_ctx,
            metric_mask=metric_mask,
            wafer_mask=wafer_mask,
            spatial_vars=spatial_vars,
            case_ids=eval_case_ids,
            region_band_cfg=region_band_cfg,
            boundary_type_breakdown=bool(spatial_cfg.get("boundary_type_breakdown", False)),
        )

    distribution_cfg = dict(diagnostics_cfg.get("distribution", {}) or {})
    if not bool(distribution_cfg.get("enabled", False)):
        return

    distribution_region_by_var = normalize_region_by_var(
        distribution_cfg.get("region_by_var", {}),
        target_vars=spatial_vars,
        key_name="eval.diagnostics.distribution.region_by_var",
    )
    distribution_vars = _resolve_distribution_vars(
        raw_vars=distribution_cfg.get("vars", "all"),
        spatial_vars=spatial_vars,
        available=set(true_spatial.keys()) & set(pred_spatial.keys()),
    )
    distribution_case_rows = build_spatial_distribution_by_case_rows(
        pred_eval=pred_spatial,
        true_eval=true_spatial,
        mask_plasma=metric_mask,
        vars_for_summary=distribution_vars,
        case_ids=eval_case_ids,
        region_by_var=distribution_region_by_var,
        top_fraction=float(distribution_cfg.get("top_fraction", 0.10)),
    )
    distribution_summary_rows = build_spatial_distribution_summary_rows(distribution_case_rows)
    _write_distribution_tables(
        model_dir=model_dir,
        row=row,
        case_rows=distribution_case_rows,
        summary_rows=distribution_summary_rows,
    )
    _plot_worst_distribution_cases(
        distribution_cfg=distribution_cfg,
        distribution_case_rows=distribution_case_rows,
        distribution_region_by_var=distribution_region_by_var,
        true_spatial=true_spatial,
        pred_spatial=pred_spatial,
        metric_mask=metric_mask,
        viz=viz,
    )


def _write_spatial_error_tables(
    *,
    model_dir: Path,
    pred_spatial: dict[str, Any],
    true_spatial: dict[str, Any],
    geom_ctx: Any,
    metric_mask: np.ndarray | None,
    wafer_mask: np.ndarray | None,
    spatial_vars: list[str],
    case_ids: list[str],
    region_band_cfg: dict[str, Any],
    boundary_type_breakdown: bool,
) -> None:
    store = ArtifactStore(model_dir / "eval")
    common_kwargs = {
        "pred_eval": pred_spatial,
        "true_eval": true_spatial,
        "mask_plasma": metric_mask,
        "distance_signed": (geom_ctx.distance_signed if geom_ctx is not None else None),
        "distance_any": (geom_ctx.distance_any if geom_ctx is not None else None),
        "bc_dir_mask": (geom_ctx.bc_dir_mask if geom_ctx is not None else None),
        "wafer_mask": wafer_mask,
        "boundary_type_breakdown": boundary_type_breakdown,
        "vars_for_summary": [v for v in spatial_vars if v in set(true_spatial.keys())],
        "region_band_cfg": region_band_cfg,
    }
    spatial_rows = build_spatial_error_summary_rows(**common_kwargs)
    if spatial_rows:
        store.save_csv(
            "spatial_error_summary.csv",
            ["var", "region", "boundary_type", "n_points", "rmse", "r2"],
            [
                [
                    r.get("var", ""),
                    r.get("region", ""),
                    r.get("boundary_type", "na"),
                    r.get("n_points", 0.0),
                    r.get("rmse", 0.0),
                    r.get("r2", 0.0),
                ]
                for r in spatial_rows
            ],
        )
    spatial_case_rows = build_spatial_error_by_case_rows(**common_kwargs, case_ids=case_ids)
    if spatial_case_rows:
        store.save_csv(
            "spatial_error_by_case.csv",
            ["case_id", "case_index", "var", "region", "boundary_type", "n_points", "rmse", "r2"],
            [
                [
                    r.get("case_id", ""),
                    r.get("case_index", 0.0),
                    r.get("var", ""),
                    r.get("region", ""),
                    r.get("boundary_type", "na"),
                    r.get("n_points", 0.0),
                    r.get("rmse", 0.0),
                    r.get("r2", 0.0),
                ]
                for r in spatial_case_rows
            ],
        )


def _resolve_distribution_vars(
    *,
    raw_vars: Any,
    spatial_vars: list[str],
    available: set[str],
) -> list[str]:
    if isinstance(raw_vars, list):
        return [str(v) for v in raw_vars if str(v) in available]
    if str(raw_vars).strip().lower() == "all":
        return [v for v in spatial_vars if v in available]
    requested = str(raw_vars).strip()
    return [requested] if requested in available else []


def _write_distribution_tables(
    *,
    model_dir: Path,
    row: dict[str, Any],
    case_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    store = ArtifactStore(model_dir / "eval")
    if case_rows:
        header = list(case_rows[0].keys())
        store.save_csv(
            "spatial_distribution_by_case.csv",
            header,
            [[r.get(key, "") for key in header] for r in case_rows],
        )
    if not summary_rows:
        return
    header = list(summary_rows[0].keys())
    store.save_csv(
        "spatial_distribution_summary.csv",
        header,
        [[r.get(key, "") for key in header] for r in summary_rows],
    )
    for summary_row in summary_rows:
        var_name = str(summary_row.get("var", ""))
        if not var_name:
            continue
        diagnostics = row.setdefault("_diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            row["_diagnostics"] = diagnostics
        for metric_name in [
            "integral_rel_error_mean",
            "p99_rel_error_mean",
            "center_of_mass_error_px_mean",
            "peak_location_error_px_mean",
            "distribution_error_score_mean",
        ]:
            diagnostics[f"dist_{var_name}_{metric_name}"] = float(summary_row.get(metric_name, float("nan")))


def _plot_worst_distribution_cases(
    *,
    distribution_cfg: dict[str, Any],
    distribution_case_rows: list[dict[str, Any]],
    distribution_region_by_var: dict[str, str],
    true_spatial: dict[str, Any],
    pred_spatial: dict[str, Any],
    metric_mask: np.ndarray | None,
    viz: Any,
) -> None:
    plot_worst_cases = int(distribution_cfg.get("plot_worst_cases", 0))
    if plot_worst_cases <= 0 or not distribution_case_rows:
        return
    worst_rows = sorted(
        [
            r
            for r in distribution_case_rows
            if np.isfinite(float(r.get("distribution_error_score", float("nan"))))
        ],
        key=lambda r: float(r.get("distribution_error_score", float("nan"))),
        reverse=True,
    )[:plot_worst_cases]
    for worst in worst_rows:
        var_name = str(worst.get("var", ""))
        case_idx = int(float(worst.get("case_index", 0.0)))
        if var_name not in true_spatial or var_name not in pred_spatial:
            continue
        true_arr = np.asarray(true_spatial[var_name], dtype=np.float32)
        pred_arr = np.asarray(pred_spatial[var_name], dtype=np.float32)
        if true_arr.ndim == 4:
            true_field = true_arr[case_idx, 0]
            pred_field = pred_arr[case_idx, 0]
        elif true_arr.ndim == 3:
            true_field = true_arr[case_idx]
            pred_field = pred_arr[case_idx]
        else:
            continue
        target_region = target_region_for_var(distribution_region_by_var, var_name)
        plot_mask = _plot_mask_for_distribution(
            target_region=target_region,
            metric_mask=metric_mask,
            case_idx=case_idx,
            n_cases=int(true_arr.shape[0]),
        )
        case_token = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(worst.get("case_id", case_idx))
        )[:80]
        viz.plot_field_triplet(
            true_field,
            pred_field,
            rel_path=f"plots/spatial_distribution_worst_{var_name}_{case_token}.png",
            mask=plot_mask,
        )


def _plot_mask_for_distribution(
    *,
    target_region: str,
    metric_mask: np.ndarray | None,
    case_idx: int,
    n_cases: int,
) -> np.ndarray | None:
    if target_region != "plasma_only" or metric_mask is None:
        return None
    mask_arr = np.asarray(metric_mask, dtype=bool)
    if mask_arr.ndim == 2:
        return mask_arr
    if mask_arr.ndim == 3 and int(mask_arr.shape[0]) == n_cases:
        return mask_arr[case_idx]
    if mask_arr.ndim == 3 and int(mask_arr.shape[0]) == 1:
        return mask_arr[0]
    return None


__all__ = ["resolve_eval_diagnostics_cfg", "write_eval_diagnostics"]
