from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

PLASMA_VARS = ["ne", "ni", "Te", "phi"]
DEFAULT_PLASMA_RUN = Path(
    "runs/icp_stage4_multifield_plasma_ffno_modes24_e80/full/"
    "ffno_plasma_modes24_grad005_gpu80_20260528_211257"
)
DEFAULT_OUT_DIR = Path("reports/icp_stage4_ffno_publication_figures_20260529_plasma_only")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _robust_limits(arrays: list[np.ndarray], q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    parts: list[np.ndarray] = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            parts.append(flat)
    if not parts:
        return 0.0, 1.0
    values = np.concatenate(parts)
    vmin = float(np.nanpercentile(values, q_low))
    vmax = float(np.nanpercentile(values, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _symmetric_limit(arrays: list[np.ndarray], q: float = 99.0) -> float:
    parts: list[np.ndarray] = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            parts.append(flat)
    if not parts:
        return 1.0
    values = np.concatenate(parts)
    lim = float(np.nanpercentile(np.abs(values), q))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(np.nanmax(np.abs(values)))
    return lim if np.isfinite(lim) and lim > 0.0 else 1.0


def _field_display(values: np.ndarray, var: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=np.float32)
    if var in {"ne", "ni"}:
        return arr / 1.0e18, f"{var} [10^18 m^-3]"
    if var == "Te":
        return arr, "Te [eV]"
    if var == "phi":
        return arr, "phi [V]"
    return arr, var


def _field_cmap(var: str) -> str:
    if var == "phi":
        return "jet"
    return "viridis"


def _target_region(var: str) -> str:
    return "plasma_only"


def _load_mask(dataset: Any) -> np.ndarray | None:
    root = Path(dataset.geometry_root)
    for path in (root / "geometry" / "mask_plasma.npy", root / "mask_plasma.npy"):
        if path.exists():
            mask = np.asarray(np.load(path), dtype=bool)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            return mask
    return None


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_spatial_axis(
    ax: plt.Axes,
    *,
    original_shape: tuple[int, int],
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    h, w = original_shape
    plot_h = len(range(0, h, 2))
    plot_w = len(range(0, w, 2))
    xticks = [0, max((plot_w - 1) // 2, 0), max(plot_w - 1, 0)]
    yticks = [0, max((plot_h - 1) // 2, 0), max(plot_h - 1, 0)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if show_xlabel:
        ax.set_xticklabels([str(min(int(v * 2), w - 1)) for v in xticks])
        ax.set_xlabel("r pixel")
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_yticklabels([str(min(int(v * 2), h - 1)) for v in yticks])
        ax.set_ylabel("z pixel")
    else:
        ax.set_yticklabels([])


def _case_quality(run_dir: Path, vars_list: list[str]) -> list[dict[str, Any]]:
    rows = _read_csv(run_dir / "models/ffno/eval_protocol/extrap/eval/spatial_error_by_case.csv")
    by_case: dict[str, list[float]] = {}
    for row in rows:
        var = str(row.get("var", ""))
        if var not in vars_list:
            continue
        region = str(row.get("region", ""))
        if region != "all_plasma":
            continue
        value = _float(row, "r2")
        if np.isfinite(value):
            by_case.setdefault(str(row["case_id"]), []).append(value)
    out = [{"case_id": case_id, "mean_r2": float(np.mean(values))} for case_id, values in by_case.items() if values]
    return sorted(out, key=lambda item: float(item["mean_r2"]))


def _selected_cases(run_dir: Path, vars_list: list[str]) -> list[str]:
    ranked = _case_quality(run_dir, vars_list)
    if not ranked:
        raise RuntimeError(f"no case quality rows for {run_dir}")
    return [str(ranked[-1]["case_id"]), str(ranked[len(ranked) // 2]["case_id"]), str(ranked[0]["case_id"])]


def _load_prediction_bundle(run_dir: Path, case_ids: list[str]) -> dict[str, Any]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint
    from plasma_surrogate.train.grid_training import _predict_features_batched
    from plasma_surrogate.train.spatial_features import (
        build_case_spatial_features,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    cfg = yaml.safe_load((run_dir / "run_config_effective.yaml").read_text(encoding="utf-8")) or {}
    bench = dict(cfg.get("benchmark", {}) or {})
    train_cfg = dict(dict(bench.get("train", {}) or {}).get("ffno", {}) or {})
    y_order = [str(v) for v in train_cfg.get("target_vars", PLASMA_VARS)]
    dataset = load_dataset({"dataset": bench["dataset"]}, run_dir)
    bundle = RunBundleLoader.load(run_dir)
    transforms = bundle.transform_bundle()
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    case_to_idx = {str(case["case_id"]): i for i, case in enumerate(dataset.cases)}
    idx = np.asarray([case_to_idx[str(case_id)] for case_id in case_ids], dtype=np.int64)

    model = load_checkpoint(run_dir / "models/ffno/eval_protocol/extrap/checkpoints")
    input_cfg = dict(train_cfg.get("input_features", {}) or {})
    channels = resolve_coord_feature_channels(input_cfg.get("features"))
    distance_cfg = resolve_distance_transform_cfg(dict(input_cfg.get("distance_transform") or {}))
    distance_cfg, _ = resolve_distance_transform_effective(
        distance_cfg,
        stats=dict(bundle.transforms.get("distance_transform_stats", {}) or {}),
    )
    spatial_source, source = build_case_spatial_features(
        channels=channels,
        pack=bundle.schemas.get("case_spatial_feature_pack"),
        static_pack=bundle.schemas.get("static_spatial_feature_pack"),
        case_pack=bundle.schemas.get("case_structure_feature_pack"),
        h=int(dataset.shape[0]),
        w=int(dataset.shape[1]),
        distance_transform_cfg=distance_cfg,
        coord_feature_scaler_artifact=dict(bundle.transforms.get("coord_feature_scaler", {}) or {}),
    )
    if spatial_source is None:
        raise RuntimeError(f"case spatial feature pack unavailable for {run_dir}: {source}")
    pred_scaled = _predict_features_batched(
        model,
        cond_scaled[idx],
        spatial_features=spatial_source.subset(idx),
        batch_size_cases=1,
    )
    pred_phys = transforms.inverse_field_dict({name: np.asarray(pred_scaled[name], dtype=np.float32) for name in y_order})
    pred = {name: np.asarray(pred_phys[name], dtype=np.float32)[:, 0] for name in y_order}
    true = {
        name: np.stack([np.asarray(dataset.cases[int(i)]["y"][name], dtype=np.float32) for i in idx], axis=0)
        for name in y_order
    }
    return {"case_ids": case_ids, "true": true, "pred": pred, "mask": _load_mask(dataset)}


def _imshow_field(
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    mask: np.ndarray | None,
    apply_mask: bool,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    plot_arr: np.ndarray | np.ma.MaskedArray = np.asarray(arr, dtype=np.float32)[::2, ::2]
    if apply_mask and mask is not None:
        mask_ds = np.asarray(mask, dtype=bool)[::2, ::2]
        plot_arr = np.ma.array(plot_arr, mask=~mask_ds)
    im = ax.imshow(plot_arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if mask is not None:
        ax.contour(mask[::2, ::2], levels=[0.5], colors="black", linewidths=0.42, alpha=0.55, origin="lower")
    return im


def plot_learning_curve(plasma_run: Path, out_dir: Path) -> str:
    rows = _read_csv(plasma_run / "models/ffno/eval_protocol/extrap/train/scalars/metrics_partial.csv")
    epoch = np.asarray([_float(row, "epoch") for row in rows], dtype=np.float64)
    train = np.asarray([_float(row, "train_loss") for row in rows], dtype=np.float64)
    val = np.asarray([_float(row, "val_loss") for row in rows], dtype=np.float64)
    score = np.asarray([_float(row, "val_balance_score") for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(epoch, train, label="train", lw=2.0)
    axes[0].plot(epoch, val, label="validation", lw=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])
    axes[1].plot(epoch, score, color="#2E8B57", lw=2.0)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation balance score")
    axes[1].set_ylim(0.6, 1.01)
    _style_axis(axes[1])
    fig.suptitle("FFNO plasma-field training curve", fontsize=15, weight="bold")
    return _save(fig, out_dir, "fig01_learning_curve_plasma_ffno_log.png")


def plot_target_r2(plasma_run: Path, out_dir: Path) -> str:
    plasma = _read_csv(plasma_run / "leaderboard.csv")[0]
    labels = PLASMA_VARS
    plasma_values = [_float(plasma, f"test_r2_{var}_plasma") for var in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = ax.bar(x, plasma_values, width=0.62, color=["#3B6FB6", "#3B6FB6", "#D97732", "#7B5EA7"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.94, 1.0)
    ax.set_xlabel("target variable")
    ax.set_ylabel("plasma-region test R2")
    ax.set_title("Target-wise accuracy of the 4-field FFNO", weight="bold")
    for bar, value in zip(bars, plasma_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _style_axis(ax)
    return _save(fig, out_dir, "fig02_target_r2_plasma_only.png")


def plot_distribution_metrics(plasma_run: Path, out_dir: Path) -> str:
    rows = _read_csv(plasma_run / "models/ffno/eval_protocol/extrap/eval/spatial_distribution_summary.csv")
    by_var = {}
    for row in rows:
        var = str(row.get("var"))
        region = str(row.get("target_region"))
        if var in PLASMA_VARS and region == "plasma_only":
            by_var[var] = row
    labels = PLASMA_VARS
    dist = np.asarray([_float(by_var[var], "distribution_error_score_mean") for var in labels], dtype=np.float64)
    shape_loss = np.asarray([1.0 - _float(by_var[var], "shape_corr_mean") for var in labels], dtype=np.float64)
    peak = np.asarray([_float(by_var[var], "peak_location_error_px_mean") for var in labels], dtype=np.float64)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    axes[0].bar(x - 0.18, dist, width=0.36, label="distribution error", color="#C44E52")
    axes[0].bar(x + 0.18, shape_loss, width=0.36, label="1 - shape correlation", color="#666666")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("target variable")
    axes[0].set_ylabel("mean score")
    axes[0].set_title("Shape agreement in plasma region")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _style_axis(axes[0])
    axes[1].bar(x, peak, color="#7B5EA7")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("target variable")
    axes[1].set_ylabel("peak-location error [pixels]")
    axes[1].set_title("Peak location error")
    _style_axis(axes[1])
    fig.suptitle("Distribution-aware metrics for plasma-only targets", weight="bold")
    return _save(fig, out_dir, "fig03_distribution_metrics_plasma_only.png")


def plot_case_quality_distribution(plasma_run: Path, out_dir: Path) -> str:
    rows = _case_quality(plasma_run, PLASMA_VARS)
    values = np.asarray([float(row["mean_r2"]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), constrained_layout=True)
    axes[0].hist(values, bins=18, color="#3B6FB6", edgecolor="white")
    axes[0].axvline(np.nanmedian(values), color="#111827", lw=1.6, ls="--", label=f"median={np.nanmedian(values):.3f}")
    axes[0].set_xlabel("case-mean plasma R2")
    axes[0].set_ylabel("held-out cases")
    axes[0].set_title("Held-out case quality distribution")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    x = np.arange(len(values))
    axes[1].plot(x, values, color="#2E8B57", lw=2.0)
    axes[1].scatter([0, len(values) // 2, len(values) - 1], [values[0], values[len(values) // 2], values[-1]], color="#C44E52", zorder=3)
    axes[1].set_xlabel("cases sorted by mean R2")
    axes[1].set_ylabel("case-mean plasma R2")
    axes[1].set_title("Best / median / worst case selection")
    _style_axis(axes[1])
    fig.suptitle("Evaluation case quality used for spatial figures", fontsize=14, weight="bold")
    return _save(fig, out_dir, "fig04_case_quality_distribution.png")


def plot_triplet_grid(
    bundle: dict[str, Any],
    vars_list: list[str],
    *,
    case_index: int,
    title: str,
    out_dir: Path,
    filename: str,
) -> str:
    mask = bundle.get("mask")
    case_id = str(bundle["case_ids"][case_index])
    fig, axes = plt.subplots(len(vars_list), 3, figsize=(10.6, 2.55 * len(vars_list)), constrained_layout=True)
    if len(vars_list) == 1:
        axes = np.asarray([axes])
    for row_idx, var in enumerate(vars_list):
        true, label = _field_display(bundle["true"][var][case_index], var)
        pred, _ = _field_display(bundle["pred"][var][case_index], var)
        err = pred - true
        cmap = _field_cmap(var)
        apply_mask = True
        if var == "phi":
            lim = _symmetric_limit([true, pred], q=99.0)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = _robust_limits([true, pred])
        elim = _symmetric_limit([err], q=99.0)
        im0 = _imshow_field(axes[row_idx, 0], true, mask=mask, apply_mask=apply_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow_field(axes[row_idx, 1], pred, mask=mask, apply_mask=apply_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        im2 = _imshow_field(
            axes[row_idx, 2],
            err,
            mask=mask,
            apply_mask=apply_mask,
            cmap="coolwarm",
            vmin=-elim,
            vmax=elim,
        )
        for col_idx in range(3):
            _format_spatial_axis(
                axes[row_idx, col_idx],
                original_shape=true.shape,
                show_xlabel=(row_idx == len(vars_list) - 1),
                show_ylabel=(col_idx == 0),
            )
        axes[row_idx, 0].set_ylabel(f"{label}\nz pixel", fontsize=9)
        fig.colorbar(im0, ax=axes[row_idx, :2], shrink=0.74, pad=0.01)
        fig.colorbar(im2, ax=axes[row_idx, 2], shrink=0.74, pad=0.01)
    for ax, col in zip(axes[0], ("ground truth", "prediction", "prediction - truth")):
        ax.set_title(col, fontsize=11, weight="bold")
    fig.suptitle(f"{title}: {case_id}", fontsize=14, weight="bold")
    return _save(fig, out_dir, filename)


def plot_best_median_worst_single_var(
    bundle: dict[str, Any],
    var: str,
    *,
    quality_rows: list[dict[str, Any]],
    title: str,
    out_dir: Path,
    filename: str,
) -> str:
    quality_map = {str(row["case_id"]): float(row["mean_r2"]) for row in quality_rows}
    labels = ["best", "median", "worst"]
    mask = bundle.get("mask")
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 8.0), constrained_layout=True)
    for idx, label_name in enumerate(labels):
        case_id = str(bundle["case_ids"][idx])
        true, unit = _field_display(bundle["true"][var][idx], var)
        pred, _ = _field_display(bundle["pred"][var][idx], var)
        err = pred - true
        cmap = _field_cmap(var)
        apply_mask = True
        if var == "phi":
            lim = _symmetric_limit([true, pred], q=99.0)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = _robust_limits([true, pred])
        elim = _symmetric_limit([err], q=99.0)
        im0 = _imshow_field(axes[idx, 0], true, mask=mask, apply_mask=apply_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow_field(axes[idx, 1], pred, mask=mask, apply_mask=apply_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        im2 = _imshow_field(
            axes[idx, 2],
            err,
            mask=mask,
            apply_mask=apply_mask,
            cmap="coolwarm",
            vmin=-elim,
            vmax=elim,
        )
        for col_idx in range(3):
            _format_spatial_axis(
                axes[idx, col_idx],
                original_shape=true.shape,
                show_xlabel=(idx == len(labels) - 1),
                show_ylabel=(col_idx == 0),
            )
        axes[idx, 0].set_ylabel(
            f"{label_name}\n{case_id}\nR2={quality_map.get(case_id, np.nan):.3f}\n{unit}\nz pixel",
            fontsize=9,
        )
        fig.colorbar(im0, ax=axes[idx, :2], shrink=0.68, pad=0.01)
        fig.colorbar(im2, ax=axes[idx, 2], shrink=0.68, pad=0.01)
    for ax, col in zip(axes[0], ("ground truth", "prediction", "prediction - truth")):
        ax.set_title(col, fontsize=11, weight="bold")
    fig.suptitle(title, fontsize=14, weight="bold")
    return _save(fig, out_dir, filename)


def plot_phi_shared_scale(
    bundle: dict[str, Any],
    *,
    quality_rows: list[dict[str, Any]],
    out_dir: Path,
) -> str:
    var = "phi"
    quality_map = {str(row["case_id"]): float(row["mean_r2"]) for row in quality_rows}
    labels = ["best", "median", "worst"]
    mask = bundle.get("mask")
    true_stack = [_field_display(bundle["true"][var][i], var)[0] for i in range(3)]
    pred_stack = [_field_display(bundle["pred"][var][i], var)[0] for i in range(3)]
    err_stack = [pred - true for true, pred in zip(true_stack, pred_stack)]
    lim = _symmetric_limit(true_stack + pred_stack, q=99.0)
    elim = _symmetric_limit(err_stack, q=99.0)
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 8.0), constrained_layout=True)
    field_im = None
    err_im = None
    for idx, label_name in enumerate(labels):
        case_id = str(bundle["case_ids"][idx])
        field_im = _imshow_field(
            axes[idx, 0],
            true_stack[idx],
            mask=mask,
            apply_mask=True,
            cmap="jet",
            vmin=-lim,
            vmax=lim,
        )
        _imshow_field(
            axes[idx, 1],
            pred_stack[idx],
            mask=mask,
            apply_mask=True,
            cmap="jet",
            vmin=-lim,
            vmax=lim,
        )
        err_im = _imshow_field(
            axes[idx, 2],
            err_stack[idx],
            mask=mask,
            apply_mask=True,
            cmap="coolwarm",
            vmin=-elim,
            vmax=elim,
        )
        for col_idx in range(3):
            _format_spatial_axis(
                axes[idx, col_idx],
                original_shape=true_stack[idx].shape,
                show_xlabel=(idx == len(labels) - 1),
                show_ylabel=(col_idx == 0),
            )
        axes[idx, 0].set_ylabel(
            f"{label_name}\n{case_id}\nR2={quality_map.get(case_id, np.nan):.3f}\nphi [V]\nz pixel",
            fontsize=9,
        )
    for ax, col in zip(axes[0], ("ground truth", "prediction", "prediction - truth")):
        ax.set_title(col, fontsize=11, weight="bold")
    if field_im is not None:
        fig.colorbar(field_im, ax=axes[:, :2], shrink=0.82, pad=0.01, label="phi [V]")
    if err_im is not None:
        fig.colorbar(err_im, ax=axes[:, 2], shrink=0.82, pad=0.01, label="error [V]")
    fig.suptitle("Electric potential phi, shared jet color scale across cases", fontsize=14, weight="bold")
    return _save(fig, out_dir, "fig08_phi_shared_jet_scale_best_median_worst.png")


def plot_line_profiles(bundle: dict[str, Any], vars_list: list[str], out_dir: Path, filename: str, title: str) -> str:
    case_ids = [str(v) for v in bundle["case_ids"]]
    h, w = np.asarray(bundle["true"][vars_list[0]][0]).shape
    row = h // 2
    x = np.arange(w)
    fig, axes = plt.subplots(len(vars_list), len(case_ids), figsize=(4.9 * len(case_ids), 2.5 * len(vars_list)), constrained_layout=True)
    if len(vars_list) == 1:
        axes = np.asarray([axes])
    for vi, var in enumerate(vars_list):
        for ci, case_id in enumerate(case_ids):
            ax = axes[vi, ci]
            true, unit = _field_display(bundle["true"][var][ci, row, :], var)
            pred, _ = _field_display(bundle["pred"][var][ci, row, :], var)
            ax.plot(x, true, color="#111827", lw=1.6, label="truth")
            ax.plot(x, pred, color="#3B6FB6", lw=1.6, ls="--", label="prediction")
            ax.set_title(case_id if vi == 0 else "")
            ax.set_ylabel(unit)
            ax.set_xlabel("r pixel")
            _style_axis(ax)
            if vi == 0 and ci == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title} (mid-height row={row})", fontsize=14, weight="bold")
    return _save(fig, out_dir, filename)


def write_summary(
    out_dir: Path,
    plasma_run: Path,
    plots: list[str],
) -> str:
    plasma = _read_csv(plasma_run / "leaderboard.csv")[0]
    payload = {
        "plasma_run": str(plasma_run),
        "plasma_ffno": {
            "primary_metric": plasma.get("primary_metric"),
            "primary_metric_value": _float(plasma, "primary_metric_value"),
            "r2": {var: _float(plasma, f"test_r2_{var}_plasma") for var in PLASMA_VARS},
            "continuity_grad_ratio_all_plasma": _float(plasma, "continuity_grad_ratio_all_plasma"),
            "continuity_lap_ratio_all_plasma": _float(plasma, "continuity_lap_ratio_all_plasma"),
        },
        "figure_notes": {
            "phi_colormap": "jet for ground-truth and prediction panels; error remains coolwarm",
            "domain_masking": "ne/ni/Te/phi are displayed in the plasma mask with a plasma boundary contour",
            "excluded_targets": "Br/Bz/Jelr/Jelz are intentionally excluded from this figure set.",
        },
        "plots": plots,
    }
    json_path = out_dir / "publication_figure_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# ICP_stage4 FFNO Publication Figure Summary",
        "",
        "## Runs",
        f"- Plasma 4-field FFNO: `{plasma_run}`",
        "",
        "## Key Metrics",
        f"- Plasma 4-field primary: `{payload['plasma_ffno']['primary_metric']}` = "
        f"{payload['plasma_ffno']['primary_metric_value']:.4f}",
        f"- Plasma FFNO continuity grad/lap ratios: "
        f"{payload['plasma_ffno']['continuity_grad_ratio_all_plasma']:.3f} / "
        f"{payload['plasma_ffno']['continuity_lap_ratio_all_plasma']:.3f}",
        "",
        "## Interpretation",
        "- `phi` is replotted with a jet field colormap for visual readability; residual panels use a diverging colormap.",
        "- Magnetic and current fields are excluded from this report to keep the evaluation focused on the current 4-field FFNO.",
        "- The figure set separates scalar R2, distribution-aware metrics, spatial true/pred/error maps, and mid-height profiles.",
        "",
        "## Figures",
    ]
    md.extend([f"- `{path}`" for path in plots])
    md_path = out_dir / "publication_figure_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return str(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ICP_stage4 FFNO publication/conference figures.")
    parser.add_argument("--plasma-run", type=Path, default=DEFAULT_PLASMA_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    plots: list[str] = []
    plots.append(plot_learning_curve(args.plasma_run, out_dir))
    plots.append(plot_target_r2(args.plasma_run, out_dir))
    plots.append(plot_distribution_metrics(args.plasma_run, out_dir))
    plots.append(plot_case_quality_distribution(args.plasma_run, out_dir))

    plasma_cases = _selected_cases(args.plasma_run, PLASMA_VARS)
    plasma_quality = _case_quality(args.plasma_run, PLASMA_VARS)
    plasma_bundle = _load_prediction_bundle(args.plasma_run, plasma_cases)
    plots.append(
        plot_triplet_grid(
            plasma_bundle,
            PLASMA_VARS,
            case_index=2,
            title="4-field FFNO plasma maps on the hardest held-out case",
            out_dir=out_dir,
            filename="fig05_plasma_fields_worst_case_triplet_phi_jet.png",
        )
    )
    for var in PLASMA_VARS:
        plots.append(
            plot_best_median_worst_single_var(
                plasma_bundle,
                var,
                quality_rows=plasma_quality,
                title=f"{var} spatial distribution, best / median / worst plasma cases",
                out_dir=out_dir,
                filename=f"fig06_{var}_best_median_worst.png",
            )
        )
    plots.append(
        plot_best_median_worst_single_var(
            plasma_bundle,
            "phi",
            quality_rows=plasma_quality,
            title="Electric potential phi with jet colormap, best / median / worst cases",
            out_dir=out_dir,
            filename="fig07_phi_best_median_worst_jet_row_scales.png",
        )
    )
    plots.append(plot_phi_shared_scale(plasma_bundle, quality_rows=plasma_quality, out_dir=out_dir))
    plots.append(
        plot_line_profiles(
            plasma_bundle,
            ["ne", "Te", "phi"],
            out_dir,
            "fig09_plasma_midheight_profiles.png",
            "Plasma-field radial profiles",
        )
    )

    summary_path = write_summary(out_dir, args.plasma_run, plots)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary_path, "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
