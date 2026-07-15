from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from icp_stage4_protocol_reliability import assess_protocol_reliability, read_validation_selection


CORE_MODELS = (
    "global_mlp",
    "deeponet_pod",
    "u_no",
    "unet",
    "unetpp",
    "unetpp_attn",
    "fno",
    "ffno",
    "cno",
    "cno_operator_unet",
)
SIZE_ORDER = ("smoke", "full")
TARGETS = ("ne", "ni", "Te", "phi")
SUMMARY_COLUMNS = (
    "dataset_size",
    "model_id",
    "status",
    "primary_metric",
    "primary_metric_value",
    "primary_metric_reliable",
    "eval_protocol_reliable",
    "eval_protocol_issue",
    "primary_metric_protocol_reliable",
    "eval_protocol_mode_effective",
    "primary_split_effective",
    "min_primary_test_cases",
    "min_primary_test_groups",
    "primary_test_groups",
    "validation_selection_reliable",
    "validation_selection_issue",
    "validation_selection_score",
    "validation_selected_epoch",
    "validation_metrics",
    "interp_overlap_fallback_applied",
    "interp_mode_effective",
    "interp_test_cases",
    "extrap_test_cases",
    "structure_holdout_test_cases",
    "scaler_fit_split",
    "density_transform_ne",
    "density_transform_ni",
    "structure_input_kind_effective",
    "has_case_varying_structure_inputs_effective",
    "test_r2_plasma_mean_dual",
    "test_r2_plasma_mean_interp",
    "test_r2_plasma_mean_extrap",
    "score_total_dual",
    "score_nrmse_plasma_mean_dual",
    "test_rmse_Te_plasma",
    "test_r2_Te_plasma",
    "test_rmse_phi_plasma",
    "test_r2_phi_plasma",
    "test_r2_ne_plasma",
    "test_r2_ni_plasma",
    "dist_ne_integral_rel_error_mean",
    "dist_ne_p99_rel_error_mean",
    "dist_ne_center_of_mass_error_px_mean",
    "dist_ne_peak_location_error_px_mean",
    "dist_ne_distribution_error_score_mean",
    "dist_ni_integral_rel_error_mean",
    "dist_ni_p99_rel_error_mean",
    "dist_ni_center_of_mass_error_px_mean",
    "dist_ni_peak_location_error_px_mean",
    "dist_ni_distribution_error_score_mean",
    "test_rmse_ne_plasma_is_finite",
    "test_rmse_ni_plasma_is_finite",
    "density_metrics_valid",
    "score_total_dual_is_finite",
    "inference_uniformity",
    "inference_boundary_gamma_uniformity",
    "poisson_residual_norm",
    "boundary_operator_proxy_loss",
    "leaderboard",
)
FLOAT64_AUX_COLUMNS = (
    "dataset_size",
    "model_id",
    "eval_split",
    "status",
    "n_cases",
    "var",
    "target_value_transform",
    "plasma_float64_rmse",
    "plasma_float64_r2",
    "full_float64_rmse",
    "full_float64_r2",
    "checkpoint",
    "error",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ICP Stage4 Core4 benchmark outputs.")
    parser.add_argument("--sizes", nargs="+", choices=SIZE_ORDER, default=list(SIZE_ORDER))
    parser.add_argument("--models", nargs="+", choices=CORE_MODELS, default=list(CORE_MODELS))
    parser.add_argument("--config-root", default="configs/experimental/icp_stage4/generated")
    parser.add_argument("--run-root", default="runs/icp_stage4_core4")
    parser.add_argument("--out-dir", default="runs/icp_stage4_core4/summary")
    parser.add_argument("--skip-density-aux", action="store_true")
    return parser.parse_args()


def _prepare_imports() -> None:
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")


def _read_first_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {str(k): str(v) for k, v in dict(row).items()}
    return None


def _as_float(raw: Any) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _is_finite_text(raw: Any) -> str:
    value = _as_float(raw)
    return "true" if math.isfinite(value) else "false"


def _pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _split_test_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    test_ids = payload.get("test", [])
    if not isinstance(test_ids, list):
        return None
    return int(len(test_ids))


def _protocol_summary_from_output(
    output_dir: Path,
    *,
    leaderboard_row: dict[str, Any] | None = None,
) -> dict[str, str]:
    return assess_protocol_reliability(output_dir, leaderboard_row=leaderboard_row)


def _reliability_intersection(raw: Any, assessed: Any) -> str:
    """Never allow a legacy/raw true value to override a failed assessment."""

    raw_text = str(raw or "").strip().lower()
    assessed_text = str(assessed or "").strip().lower()
    if "false" in {raw_text, assessed_text}:
        return "false"
    if raw_text == "true" and assessed_text == "true":
        return "true"
    return assessed_text if assessed_text in {"true", "false"} else "false"


def _summary_row(*, size: str, model: str, leaderboard: Path) -> dict[str, str]:
    raw = _read_first_csv_row(leaderboard)
    if raw is None:
        return {
            "dataset_size": size,
            "model_id": model,
            "status": "missing",
            "leaderboard": str(leaderboard),
        }
    density_valid = all(
        math.isfinite(_as_float(raw.get(key)))
        for key in (
            "test_rmse_ne_plasma",
            "test_rmse_ni_plasma",
            "test_r2_ne_plasma",
            "test_r2_ni_plasma",
        )
    )
    primary_finite = math.isfinite(_as_float(raw.get("primary_metric_value")))
    protocol_summary = _protocol_summary_from_output(
        leaderboard.parent,
        leaderboard_row=raw,
    )
    validation_summary = read_validation_selection(
        leaderboard.parent,
        model_id=str(raw.get("model_id") or model),
        primary_split=str(protocol_summary.get("primary_split_effective", "")),
    )
    primary_metric_protocol_reliable = _reliability_intersection(
        raw.get("primary_metric_protocol_reliable"),
        protocol_summary.get("primary_metric_protocol_reliable"),
    )
    primary_reliable_raw = str(raw.get("primary_metric_reliable", "")).strip().lower()
    primary_reliable = (
        primary_reliable_raw
        if primary_reliable_raw in {"true", "false"}
        else ("true" if (primary_finite and density_valid) else "false")
    )
    if primary_metric_protocol_reliable != "true":
        primary_reliable = "false"
    eval_protocol_reliable = _reliability_intersection(
        raw.get("eval_protocol_reliable"),
        protocol_summary.get("eval_protocol_reliable"),
    )
    issue_values = [
        str(raw.get("eval_protocol_issue", "")).strip(),
        str(protocol_summary.get("eval_protocol_issue", "")).strip(),
    ]
    eval_protocol_issue = "|".join(dict.fromkeys(value for value in issue_values if value))
    score_total_finite = math.isfinite(_as_float(raw.get("score_total_dual", raw.get("score_total", ""))))
    row = {key: "" for key in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset_size": size,
            "model_id": model,
            "status": "ok",
            "primary_metric": raw.get("primary_metric", ""),
            "primary_metric_value": raw.get("primary_metric_value", ""),
            "primary_metric_reliable": primary_reliable,
            "eval_protocol_reliable": eval_protocol_reliable,
            "eval_protocol_issue": eval_protocol_issue,
            "primary_metric_protocol_reliable": primary_metric_protocol_reliable,
            "eval_protocol_mode_effective": protocol_summary.get("eval_protocol_mode_effective", ""),
            "primary_split_effective": protocol_summary.get("primary_split_effective", ""),
            "min_primary_test_cases": protocol_summary.get("min_primary_test_cases", ""),
            "min_primary_test_groups": protocol_summary.get("min_primary_test_groups", ""),
            "primary_test_groups": protocol_summary.get("primary_test_groups", ""),
            "validation_selection_reliable": validation_summary.get("validation_selection_reliable", ""),
            "validation_selection_issue": validation_summary.get("validation_selection_issue", ""),
            "validation_selection_score": validation_summary.get("validation_selection_score", ""),
            "validation_selected_epoch": validation_summary.get("validation_selected_epoch", ""),
            "validation_metrics": validation_summary.get("validation_metrics", ""),
            "interp_overlap_fallback_applied": raw.get(
                "interp_overlap_fallback_applied",
                protocol_summary.get("interp_overlap_fallback_applied", ""),
            ),
            "interp_mode_effective": raw.get("interp_mode_effective", protocol_summary.get("interp_mode_effective", "")),
            "interp_test_cases": raw.get("interp_test_cases", protocol_summary.get("interp_test_cases", "")),
            "extrap_test_cases": raw.get("extrap_test_cases", protocol_summary.get("extrap_test_cases", "")),
            "structure_holdout_test_cases": raw.get(
                "structure_holdout_test_cases",
                protocol_summary.get("structure_holdout_test_cases", ""),
            ),
            "scaler_fit_split": raw.get("scaler_fit_split", ""),
            "density_transform_ne": raw.get("density_transform_ne", ""),
            "density_transform_ni": raw.get("density_transform_ni", ""),
            "structure_input_kind_effective": raw.get("structure_input_kind_effective", ""),
            "has_case_varying_structure_inputs_effective": raw.get(
                "has_case_varying_structure_inputs_effective",
                "",
            ),
            "test_r2_plasma_mean_dual": raw.get("test_r2_plasma_mean_dual", ""),
            "test_r2_plasma_mean_interp": raw.get("test_r2_plasma_mean_interp", ""),
            "test_r2_plasma_mean_extrap": raw.get("test_r2_plasma_mean_extrap", ""),
            "score_total_dual": raw.get("score_total_dual", raw.get("score_total", "")),
            "score_nrmse_plasma_mean_dual": raw.get(
                "score_nrmse_plasma_mean_dual",
                raw.get("score_nrmse_plasma_mean", ""),
            ),
            "test_rmse_Te_plasma": raw.get("test_rmse_Te_plasma", ""),
            "test_r2_Te_plasma": raw.get("test_r2_Te_plasma", ""),
            "test_rmse_phi_plasma": raw.get("test_rmse_phi_plasma", ""),
            "test_r2_phi_plasma": raw.get("test_r2_phi_plasma", ""),
            "test_r2_ne_plasma": raw.get("test_r2_ne_plasma", ""),
            "test_r2_ni_plasma": raw.get("test_r2_ni_plasma", ""),
            "dist_ne_integral_rel_error_mean": raw.get("dist_ne_integral_rel_error_mean", ""),
            "dist_ne_p99_rel_error_mean": raw.get("dist_ne_p99_rel_error_mean", ""),
            "dist_ne_center_of_mass_error_px_mean": raw.get("dist_ne_center_of_mass_error_px_mean", ""),
            "dist_ne_peak_location_error_px_mean": raw.get("dist_ne_peak_location_error_px_mean", ""),
            "dist_ne_distribution_error_score_mean": raw.get("dist_ne_distribution_error_score_mean", ""),
            "dist_ni_integral_rel_error_mean": raw.get("dist_ni_integral_rel_error_mean", ""),
            "dist_ni_p99_rel_error_mean": raw.get("dist_ni_p99_rel_error_mean", ""),
            "dist_ni_center_of_mass_error_px_mean": raw.get("dist_ni_center_of_mass_error_px_mean", ""),
            "dist_ni_peak_location_error_px_mean": raw.get("dist_ni_peak_location_error_px_mean", ""),
            "dist_ni_distribution_error_score_mean": raw.get("dist_ni_distribution_error_score_mean", ""),
            "test_rmse_ne_plasma_is_finite": _is_finite_text(raw.get("test_rmse_ne_plasma")),
            "test_rmse_ni_plasma_is_finite": _is_finite_text(raw.get("test_rmse_ni_plasma")),
            "density_metrics_valid": "true" if density_valid else "false",
            "score_total_dual_is_finite": "true" if score_total_finite else "false",
            "inference_uniformity": _pick(raw, "inference_uniformity", "single_qoi_uniformity", "qoi_uniformity"),
            "inference_boundary_gamma_uniformity": _pick(
                raw,
                "inference_boundary_gamma_uniformity",
                "single_qoi_boundary_gamma_uniformity",
                "qoi_boundary_gamma_uniformity",
            ),
            "poisson_residual_norm": _pick(raw, "poisson_residual_norm", "single_poisson_residual_norm"),
            "boundary_operator_proxy_loss": _pick(
                raw,
                "boundary_operator_proxy_loss",
                "single_boundary_operator_proxy_loss",
            ),
            "leaderboard": str(leaderboard),
        }
    )
    return row


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_mask(dataset: Any) -> np.ndarray:
    root = Path(dataset.geometry_root)
    candidates = (root / "geometry" / "mask_plasma.npy", root / "mask_plasma.npy")
    for path in candidates:
        if path.exists():
            arr = np.asarray(np.load(path), dtype=bool)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
    raise FileNotFoundError(f"mask_plasma.npy not found under {root}")


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(t) & np.isfinite(p)
    if not np.any(finite):
        return float("nan")
    t = t[finite]
    p = p[finite]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(t) & np.isfinite(p)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean((t[finite] - p[finite]) ** 2)))


def _set_static_spatial_features(model_obj: Any, bench: dict[str, Any], model: str, bundle: Any, h: int, w: int) -> None:
    if not hasattr(model_obj, "set_static_spatial_features"):
        return
    train_cfg = dict(bench.get("train", {})).get(model, {})
    input_cfg = dict(train_cfg.get("input_features", {}))
    if str(input_cfg.get("mode", "")).strip().lower() != "geom_feature_pack":
        return

    from plasma_surrogate.train.spatial_features import (
        apply_coord_feature_scaling,
        apply_distance_transform,
        build_coord_feature_rows,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    channels = resolve_coord_feature_channels(input_cfg.get("features"))
    rows, _ = build_coord_feature_rows(
        channels=channels,
        pack=bundle.schemas.get("coord_feature_pack"),
        geom_ctx=None,
        h=h,
        w=w,
    )
    distance_cfg = resolve_distance_transform_cfg(dict(input_cfg.get("distance_transform") or {}))
    distance_cfg, _ = resolve_distance_transform_effective(
        distance_cfg,
        stats=dict(bundle.transforms.get("distance_transform_stats", {})),
    )
    rows, _ = apply_distance_transform(rows.astype(np.float32), channels=channels, cfg=distance_cfg)
    rows, _, _ = apply_coord_feature_scaling(
        rows.astype(np.float32),
        channels=channels,
        coord_feature_scaler_artifact=dict(bundle.transforms.get("coord_feature_scaler", {})),
    )
    model_obj.set_static_spatial_features(rows.reshape(h, w, len(channels)).astype(np.float32))


def _predict_scaled(model_obj: Any, cond_scaled: np.ndarray, y_order: list[str]) -> dict[str, np.ndarray]:
    pred = model_obj.predict_fields(cond_scaled)
    return {name: np.asarray(pred[name], dtype=np.float32) for name in y_order if name in pred}


def _density_aux_for_run(*, size: str, model: str, config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = dict(cfg.get("benchmark", {}))
    if not output_dir.exists():
        return [
            {
                "dataset_size": size,
                "model_id": model,
                "eval_split": "",
                "status": "missing_output_dir",
                "checkpoint": "",
                "error": str(output_dir),
            }
        ]

    dataset = load_dataset({"dataset": bench["dataset"]}, output_dir)
    bundle = RunBundleLoader.load(output_dir)
    transforms = bundle.transform_bundle()
    y_order = [str(v) for v in bundle.schemas.get("output_layout", {}).get("vars", TARGETS)]
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    y = np.stack(
        [np.stack([case["y"][name] for name in y_order], axis=0).astype(np.float32) for case in dataset.cases],
        axis=0,
    )
    case_id_to_idx = {str(case["case_id"]): i for i, case in enumerate(dataset.cases)}
    mask = _metric_mask(dataset)

    eval_mode = str(bench.get("eval_protocol", {}).get("mode", "single")).strip().lower()
    splits = ["interp", "extrap"] if eval_mode == "dual_axis" else ["single"]
    rows: list[dict[str, Any]] = []
    for split in splits:
        if split == "single":
            split_path = output_dir / "preprocessing" / "split" / "split_random_v1.json"
            ckpt = output_dir / "models" / model / "checkpoints"
        else:
            split_path = output_dir / "preprocessing" / "split" / f"split_{split}_v1.json"
            ckpt = output_dir / "models" / model / "eval_protocol" / split / "checkpoints"
        if not ckpt.exists():
            rows.append(
                {
                    "dataset_size": size,
                    "model_id": model,
                    "eval_split": split,
                    "status": "missing_checkpoint",
                    "checkpoint": str(ckpt),
                    "error": "",
                }
            )
            continue
        try:
            split_def = _load_json(split_path)
            test_ids = [str(v) for v in split_def.get("test", [])]
            idx = np.asarray([case_id_to_idx[cid] for cid in test_ids], dtype=np.int64)
            model_obj = load_checkpoint(ckpt)
            _set_static_spatial_features(model_obj, bench, model, bundle, int(dataset.shape[0]), int(dataset.shape[1]))
            pred_scaled = _predict_scaled(model_obj, cond_scaled[idx], y_order)
            pred_phys = transforms.inverse_field_dict(pred_scaled)
            for var in ("ne", "ni"):
                if var not in pred_scaled or var not in y_order:
                    continue
                var_idx = y_order.index(var)
                true_phys = np.asarray(y[idx, var_idx : var_idx + 1], dtype=np.float64)
                pred_p = np.asarray(pred_phys[var], dtype=np.float64)
                transform_mode = str(
                    dict(getattr(transforms, "target_transforms", {}) or {})
                    .get(var, {})
                    .get("value_transform", "identity")
                )
                plasma_rmse = _rmse(true_phys[..., mask], pred_p[..., mask])
                plasma_r2 = _r2(true_phys[..., mask], pred_p[..., mask])
                full_rmse = _rmse(true_phys, pred_p)
                full_r2 = _r2(true_phys, pred_p)
                status = "ok" if all(math.isfinite(v) for v in (plasma_rmse, plasma_r2, full_rmse, full_r2)) else "nonfinite_prediction"
                rows.append(
                    {
                        "dataset_size": size,
                        "model_id": model,
                        "eval_split": split,
                        "status": status,
                        "n_cases": int(idx.shape[0]),
                        "var": var,
                        "target_value_transform": transform_mode,
                        "plasma_float64_rmse": plasma_rmse,
                        "plasma_float64_r2": plasma_r2,
                        "full_float64_rmse": full_rmse,
                        "full_float64_r2": full_r2,
                        "checkpoint": str(ckpt),
                        "error": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "dataset_size": size,
                    "model_id": model,
                    "eval_split": split,
                    "status": "failed",
                    "checkpoint": str(ckpt),
                    "error": repr(exc),
                }
            )
    return rows


def main() -> int:
    args = _parse_args()
    _prepare_imports()
    config_root = Path(args.config_root)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)

    summary_rows: list[dict[str, str]] = []
    density_rows: list[dict[str, Any]] = []
    for size in args.sizes:
        for model in args.models:
            output_dir = run_root / size / model
            leaderboard = output_dir / "leaderboard.csv"
            summary_rows.append(_summary_row(size=size, model=model, leaderboard=leaderboard))
            if not args.skip_density_aux:
                cfg = config_root / size / f"benchmark_icp_stage4_core4_{size}_{model}.yaml"
                if cfg.exists():
                    density_rows.extend(
                        _density_aux_for_run(size=size, model=model, config_path=cfg, output_dir=output_dir)
                    )

    _write_csv(out_dir / "comparison_summary.csv", SUMMARY_COLUMNS, summary_rows)
    if not args.skip_density_aux:
        _write_csv(out_dir / "density_aux_metrics.csv", FLOAT64_AUX_COLUMNS, density_rows)
    print((out_dir / "comparison_summary.csv").as_posix())
    if not args.skip_density_aux:
        print((out_dir / "density_aux_metrics.csv").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
