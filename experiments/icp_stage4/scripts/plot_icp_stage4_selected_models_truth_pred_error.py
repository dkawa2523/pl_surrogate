#!/usr/bin/env python3
"""Compare ICP Stage4 truth/prediction/error fields for selected trained models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

import matplotlib.pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[3]
for path in (ROOT_DIR / "src", ROOT_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_icp_stage4_spatial_truth_pred_error import (  # noqa: E402
    _case_conditions,
    _case_lookup,
    _finite_percentile,
    _metric_mask,
    _plot_case,
    _plot_extent,
    _predict_case_batch,
    _spatial_features_for_indices,
)


TARGETS = ("ne", "ni", "Te", "phi")
UNITS = {"ne": "m^-3", "ni": "m^-3", "Te": "eV", "phi": "V"}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    run_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uno-run-dir",
        default="runs/icp_stage4_major_fixes_part_lite_v2_e80_uno3/uno_tuning/u_no__u_no_m10_w48_l4_lr3e4__seed411",
    )
    parser.add_argument(
        "--unet-run-dir",
        default="runs/icp_stage4_major_fixes_part_lite_v2_e80_unet_compact/full/unet",
    )
    parser.add_argument(
        "--ffno-run-dir",
        default="runs/icp_stage4_major_fixes_part_lite_v2_e80_ffno_m12_w48_l4_lr4e4/full/ffno",
    )
    parser.add_argument("--protocol", default="structure_holdout")
    parser.add_argument("--out-dir", default="runs/icp_stage4_selected_models_spatial_truth_pred_error")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(p - t))) / max(float(np.sqrt(np.mean(np.square(t)))), 1.0e-30))


def _field_2d(value: np.ndarray) -> np.ndarray:
    field = np.asarray(value, dtype=np.float32)
    while field.ndim > 2 and 1 in field.shape:
        field = np.squeeze(field)
    if field.ndim != 2:
        raise ValueError(f"expected 2D field, got {field.shape}")
    return field


def _gradient_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    valid_r = m[:, 1:] & m[:, :-1]
    valid_z = m[1:, :] & m[:-1, :]
    pg_r, tg_r = np.diff(p, axis=1)[valid_r], np.diff(t, axis=1)[valid_r]
    pg_z, tg_z = np.diff(p, axis=0)[valid_z], np.diff(t, axis=0)[valid_z]
    grad_err = float(np.sqrt(np.sum(np.square(pg_r - tg_r)) + np.sum(np.square(pg_z - tg_z))))
    grad_ref = max(float(np.sqrt(np.sum(np.square(tg_r)) + np.sum(np.square(tg_z)))), 1.0e-30)
    true_grad = np.concatenate([np.abs(tg_r), np.abs(tg_z)])
    pred_grad = np.concatenate([np.abs(pg_r), np.abs(pg_z)])
    p99_ratio = float(np.percentile(pred_grad, 99.0) / max(float(np.percentile(true_grad, 99.0)), 1.0e-30))

    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    lap_p = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]
    lap_t = t[:-2, 1:-1] + t[2:, 1:-1] + t[1:-1, :-2] + t[1:-1, 2:] - 4.0 * t[1:-1, 1:-1]
    lap_rel = _relative_l2(lap_p[inner], lap_t[inner])
    return grad_err / grad_ref, lap_rel, p99_ratio


def _case_metrics(*, model: str, case_id: str, target: str, truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    t = np.asarray(truth, dtype=np.float64)[selected]
    p = np.asarray(pred, dtype=np.float64)[selected]
    rmse = float(np.sqrt(np.mean(np.square(p - t))))
    mae = float(np.mean(np.abs(p - t)))
    corr = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 0 and np.std(p) > 0 else float("nan")
    grad_rel, lap_rel, p99_grad_ratio = _gradient_metrics(pred, truth, mask)
    return {
        "model": model,
        "case_id": case_id,
        "target": target,
        "unit": UNITS[target],
        "rmse": rmse,
        "mae": mae,
        "relative_l2": _relative_l2(p, t),
        "correlation": corr,
        "mean_bias_over_truth_rms": float(np.mean(p - t) / max(float(np.sqrt(np.mean(np.square(t)))), 1.0e-30)),
        "gradient_relative_l2": grad_rel,
        "laplacian_relative_l2": lap_rel,
        "p99_gradient_ratio": p99_grad_ratio,
    }


def _load_predictions(spec: ModelSpec, protocol: str) -> tuple[list[str], dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], np.ndarray, tuple[float, float, float, float] | None, dict[str, str]]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint

    cfg = _load_yaml(spec.run_dir / "resolved_config.yaml")
    dataset = load_dataset({"dataset": cfg["dataset"]}, spec.run_dir)
    checkpoint = spec.run_dir / "models" / spec.model_id / "eval_protocol" / protocol / "checkpoints"
    model_obj = load_checkpoint(checkpoint)
    bundle = RunBundleLoader.load(spec.run_dir, model=model_obj)
    transforms = bundle.transform_bundle(protocol, require_protocol=True)
    y_order = [str(value) for value in bundle.schemas.get("output_layout", {}).get("vars", TARGETS)]
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    lookup = _case_lookup(dataset)
    split = _load_json(spec.run_dir / "preprocessing" / "split" / f"split_{protocol}_v1.json")
    case_ids = [str(value) for value in split["test"]]
    indices = np.asarray([lookup[case_id] for case_id in case_ids], dtype=np.int64)
    h, w = int(dataset.shape[0]), int(dataset.shape[1])
    spatial = _spatial_features_for_indices(bench=cfg, model_id=spec.model_id, bundle=bundle, h=h, w=w, indices=indices)
    truth: dict[str, dict[str, np.ndarray]] = {}
    pred: dict[str, dict[str, np.ndarray]] = {}
    conditions: dict[str, str] = {}
    for local_index, case_id in enumerate(case_ids):
        dataset_index = int(indices[local_index])
        conditions[case_id] = _case_conditions(dataset, dataset_index)
        truth[case_id] = {target: _field_2d(dataset.cases[dataset_index]["y"][target]) for target in TARGETS}
        spatial_one = None if spatial is None else spatial[local_index : local_index + 1]
        scaled = _predict_case_batch(
            model_obj=model_obj,
            cond_scaled=cond_scaled[dataset_index : dataset_index + 1],
            spatial_features=spatial_one,
            targets=list(TARGETS),
        )
        physical = transforms.inverse_field_dict(scaled)
        pred[case_id] = {target: _field_2d(physical[target]) for target in TARGETS}
    return case_ids, truth, pred, _metric_mask(dataset), _plot_extent(dataset), conditions


def _summarize(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted({str(row["model"]) for row in metric_rows}):
        for target in TARGETS:
            group = [row for row in metric_rows if row["model"] == model and row["target"] == target]
            ordered = sorted(group, key=lambda row: float(row["relative_l2"]))
            representative = ordered[len(ordered) // 2]
            worst = max(group, key=lambda row: float(row["laplacian_relative_l2"]))
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "unit": UNITS[target],
                    "median_relative_l2": float(np.median([float(row["relative_l2"]) for row in group])),
                    "max_relative_l2": max(float(row["relative_l2"]) for row in group),
                    "median_correlation": float(np.median([float(row["correlation"]) for row in group])),
                    "median_gradient_relative_l2": float(np.median([float(row["gradient_relative_l2"]) for row in group])),
                    "median_laplacian_relative_l2": float(np.median([float(row["laplacian_relative_l2"]) for row in group])),
                    "median_p99_gradient_ratio": float(np.median([float(row["p99_gradient_ratio"]) for row in group])),
                    "representative_case": representative["case_id"],
                    "line_worst_case": worst["case_id"],
                }
            )
    return rows


def _plot_summary(summary: list[dict[str, Any]], out_dir: Path, dpi: int) -> None:
    models = list(dict.fromkeys(str(row["model"]) for row in summary))
    x = np.arange(len(TARGETS), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    for index, model in enumerate(models):
        rows = [next(row for row in summary if row["model"] == model and row["target"] == target) for target in TARGETS]
        offset = (index - 0.5 * (len(models) - 1)) * width
        axes[0].bar(x + offset, [float(row["median_relative_l2"]) for row in rows], width=width, label=model)
        axes[1].bar(x + offset, [float(row["median_gradient_relative_l2"]) for row in rows], width=width, label=model)
        axes[2].bar(x + offset, [float(row["median_laplacian_relative_l2"]) for row in rows], width=width, label=model)
    axes[0].set_title("median field relative L2")
    axes[1].set_title("median gradient relative L2")
    axes[2].set_title("median Laplacian relative L2")
    for ax in axes:
        ax.set_xticks(x, TARGETS)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out_dir / "model_target_error_summary.png", dpi=dpi)
    plt.close(fig)


def _plot_input_feature_maps(run_dir: Path, case_id: str, out_path: Path, dpi: int) -> None:
    static = np.load(run_dir / "preprocessing" / "features" / "static_spatial_feature_pack.npz", allow_pickle=True)
    case_pack = np.load(run_dir / "preprocessing" / "features" / "case_structure_feature_pack.npz", allow_pickle=True)
    static_names = [str(value) for value in static["channels"]]
    case_names = [str(value) for value in case_pack["channels"]]
    case_ids = [str(value) for value in case_pack["case_ids"]]
    case_index = case_ids.index(case_id)
    fields = {name: np.asarray(static["data"][static_names.index(name)]) for name in static_names}
    fields.update({name: np.asarray(case_pack["data"][case_index, case_names.index(name)]) for name in case_names})
    keys = (
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
        "boundary_band",
        "part_sdf_nearest",
        "part_sdf_second",
        "part_gap_proxy",
        "solid_proximity",
    )
    extent = [float(fields["x"].min()), float(fields["x"].max()), float(fields["y"].min()), float(fields["y"].max())]
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 6.5), constrained_layout=True)
    for ax, key in zip(axes.reshape(-1), keys, strict=True):
        cmap = "coolwarm" if key in {"distance_signed", "normal_x", "normal_y"} else "viridis"
        image = ax.imshow(fields[key], origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_title(key)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle(f"Raw geometry input maps ({case_id})", fontweight="bold")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_cross_model(
    *,
    case_id: str,
    target: str,
    models: list[str],
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    mask: np.ndarray,
    extent: tuple[float, float, float, float] | None,
    out_path: Path,
    dpi: int,
) -> None:
    truth_masked = np.where(mask, truth, np.nan)
    pred_masked = {model: np.where(mask, predictions[model], np.nan) for model in models}
    vmin, vmax = _finite_percentile([truth_masked, *pred_masked.values()], (1.0, 99.0))
    error_abs = max(float(np.nanpercentile(np.abs(value - truth_masked), 99.0)) for value in pred_masked.values())
    fig, axes = plt.subplots(len(models), 3, figsize=(12.0, 3.0 * len(models)), constrained_layout=True)
    for row_index, model in enumerate(models):
        panels = (
            ("truth", truth_masked, "viridis", vmin, vmax),
            ("prediction", pred_masked[model], "viridis", vmin, vmax),
            ("prediction - truth", pred_masked[model] - truth_masked, "coolwarm", -error_abs, error_abs),
        )
        for col_index, (title, field, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_index, col_index]
            image = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{model}: {title}", fontsize=9)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{target} [{UNITS[target]}] | common line-error case {case_id}", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ModelSpec("UNO m10", "u_no", Path(args.uno_run_dir)),
        ModelSpec("U-Net compact", "unet", Path(args.unet_run_dir)),
        ModelSpec("FFNO m12", "ffno", Path(args.ffno_run_dir)),
    ]
    all_truth: dict[str, dict[str, np.ndarray]] | None = None
    all_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    metric_rows: list[dict[str, Any]] = []
    case_ids_ref: list[str] | None = None
    mask_ref: np.ndarray | None = None
    extent_ref: tuple[float, float, float, float] | None = None
    conditions_ref: dict[str, str] = {}
    for spec in specs:
        print(f"loading {spec.label}", flush=True)
        case_ids, truth, pred, mask, extent, conditions = _load_predictions(spec, str(args.protocol))
        if case_ids_ref is not None and case_ids != case_ids_ref:
            raise ValueError(f"test case mismatch for {spec.label}")
        case_ids_ref = case_ids
        all_truth = truth if all_truth is None else all_truth
        all_predictions[spec.label] = pred
        mask_ref = mask if mask_ref is None else mask_ref
        extent_ref = extent if extent_ref is None else extent_ref
        conditions_ref = conditions if not conditions_ref else conditions_ref
        for case_id in case_ids:
            for target in TARGETS:
                metric_rows.append(
                    _case_metrics(model=spec.label, case_id=case_id, target=target, truth=truth[case_id][target], pred=pred[case_id][target], mask=mask)
                )
    assert case_ids_ref is not None and all_truth is not None and mask_ref is not None
    summary = _summarize(metric_rows)
    _write_csv(out_dir / "per_case_target_metrics.csv", metric_rows)
    _write_csv(out_dir / "model_target_summary.csv", summary)
    _plot_summary(summary, out_dir, int(args.dpi))
    feature_case = "case_g032_op03" if "case_g032_op03" in case_ids_ref else case_ids_ref[0]
    _plot_input_feature_maps(specs[0].run_dir, feature_case, out_dir / "input_feature_maps.png", int(args.dpi))

    for spec in specs:
        model_rows = [row for row in summary if row["model"] == spec.label]
        for row in model_rows:
            target = str(row["target"])
            for kind, case_key in (("representative", "representative_case"), ("line_worst", "line_worst_case")):
                case_id = str(row[case_key])
                _plot_case(
                    truth={target: all_truth[case_id][target]},
                    pred={target: all_predictions[spec.label][case_id][target]},
                    targets=[target],
                    mask=mask_ref,
                    extent=extent_ref,
                    title=f"{spec.label} | {target} [{UNITS[target]}] | {kind} | {case_id} | {conditions_ref[case_id]}",
                    out_path=out_dir / spec.label.replace(" ", "_") / target / f"{kind}_{case_id}.png",
                    dpi=int(args.dpi),
                )

    models = [spec.label for spec in specs]
    common_cases: dict[str, str] = {}
    for target in TARGETS:
        by_case = {
            case_id: float(np.mean([float(row["laplacian_relative_l2"]) for row in metric_rows if row["target"] == target and row["case_id"] == case_id]))
            for case_id in case_ids_ref
        }
        case_id = max(by_case, key=by_case.get)
        common_cases[target] = case_id
        _plot_cross_model(
            case_id=case_id,
            target=target,
            models=models,
            truth=all_truth[case_id][target],
            predictions={model: all_predictions[model][case_id][target] for model in models},
            mask=mask_ref,
            extent=extent_ref,
            out_path=out_dir / "cross_model" / f"{target}_common_line_worst_{case_id}.png",
            dpi=int(args.dpi),
        )

    lines = [
        "# ICP Stage4 selected-model spatial truth/prediction/error review",
        "",
        f"Protocol: `{args.protocol}`; common test cases: `{len(case_ids_ref)}`; non-plasma pixels are masked.",
        "Truth and prediction panels use the same per-figure color range. Error is prediction minus truth.",
        "",
        "## Review findings",
        "",
        "- Internal diagonal/horizontal lines are absent from truth and occur in all model predictions.",
        "- Their locations coincide with ridges/discontinuities in the distance, normal, and curvature input maps.",
        "- FFNO also imprints strong vertical stripes that coincide with coil-derived part SDF/gap maps.",
        "- U-Net compact has the lowest median field error; UNO is second; FFNO has the largest density and high-frequency errors.",
        "- Near-floor case `case_g012_op01` is predicted as an active nonzero plasma by all models and must be handled separately.",
        "- These models should not be used for shape optimization until the geometry channels are ablated/regularized and retrained.",
        "",
        "[Geometry input maps showing the same line patterns](input_feature_maps.png)",
        "",
        "## Aggregate metrics",
        "",
        "| Model | Target | Median relative L2 | Median correlation | Gradient relative L2 | Laplacian relative L2 | Representative | Line-worst |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary:
        model_dir = str(row["model"]).replace(" ", "_")
        target = str(row["target"])
        representative = f"{model_dir}/{target}/representative_{row['representative_case']}.png"
        worst = f"{model_dir}/{target}/line_worst_{row['line_worst_case']}.png"
        lines.append(
            f"| {row['model']} | {target} | {float(row['median_relative_l2']):.4f} | {float(row['median_correlation']):.4f} | "
            f"{float(row['median_gradient_relative_l2']):.4f} | {float(row['median_laplacian_relative_l2']):.4f} | "
            f"[plot]({representative}) | [plot]({worst}) |"
        )
    lines += ["", "## Cross-model common line-error cases", ""]
    for target, case_id in common_cases.items():
        lines.append(f"- [{target}: {case_id}](cross_model/{target}_common_line_worst_{case_id}.png)")
    lines += [
        "",
        "## Files",
        "",
        "- [Metric comparison](model_target_error_summary.png)",
        "- [Per-case metrics](per_case_target_metrics.csv)",
        "- [Model/target summary](model_target_summary.csv)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
