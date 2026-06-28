#!/usr/bin/env python3
"""Plot inspection figures for the GEC-CCP trustworthy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "unet_operator_v2",
    "fno",
    "ffno",
    "coord_mlp_fourier",
    "coord_mlp_siren",
    "coord_mlp_pod_residual",
    "u_no",
    "cno",
    "cno_operator_unet",
    "geom_deeponet_pod",
    "geom_deeponet_siren",
    "deeponet_plasma",
)

CORE_MODELS = (
    "global_mlp",
    "deeponet_pod",
    "geom_deeponet_pod",
    "coord_mlp_pod_residual",
    "fno",
    "ffno",
    "unet_operator_v2",
    "cno_operator_unet",
)

TARGETS = ("ne", "ni", "Te", "phi")
SIZES = (27, 54, 78)

COLORS = {
    "core": "#2B6CB0",
    "appendix": "#8A8F98",
    "table_only": "#4C78A8",
    "table_plus_structure": "#F58518",
    "nrmse": "#4C78A8",
    "spatial_huber": "#4C78A8",
    "avgpool_huber": "#54A24B",
    "boundary": "#F58518",
    "continuity": "#54A24B",
    "physics": "#B279A2",
    "sign": "#E45756",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GEC-CCP trustworthy benchmark inspection plots.")
    parser.add_argument("--run-root", default="runs/gec_ccp_trustworthy_v1")
    parser.add_argument("--summary-csv", default="reports/gec_ccp_trustworthy_v1/summary_trustworthy_27_54_78.csv")
    parser.add_argument("--out-dir", default="reports/gec_ccp_trustworthy_v1/plots")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _rows_for_size(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(float(row.get("dataset_size", -1))) == int(size)]


def _by_key(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(float(row["dataset_size"])), str(row["model_id"])): row for row in rows}


def _load_leaderboards(run_root: Path, models: tuple[str, ...], sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for model in models:
            path = run_root / f"n{size}" / model / "leaderboard.csv"
            if not path.exists():
                continue
            loaded = _read_csv(path)
            if not loaded:
                continue
            row = dict(loaded[0])
            row["dataset_size"] = str(size)
            row["model_id"] = str(row.get("model_id") or model)
            row["leaderboard_csv"] = str(path)
            rows.append(row)
    return rows


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def _setup_axis(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)


def _annotate_bars(ax: plt.Axes, values: list[float], *, fmt: str = "{:.3f}") -> None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return
    pad = 0.012 * max(finite)
    for idx, value in enumerate(values):
        if math.isfinite(value):
            ax.text(value + pad, idx, fmt.format(value), va="center", fontsize=8)


def plot_quality_rank(summary_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = sorted(_rows_for_size(summary_rows, 78), key=lambda row: _float(row.get("surrogate_quality_score")))
    labels = [str(row["model_id"]) for row in rows]
    values = [_float(row.get("surrogate_quality_score")) for row in rows]
    colors = [COLORS["core"] if label in CORE_MODELS else COLORS["appendix"] for label in labels]

    fig, ax = plt.subplots(figsize=(9.6, 7.2), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    _setup_axis(ax, "n78 reliability ranking", "surrogate_quality_score (lower is better)")
    _annotate_bars(ax, values)
    ax.margins(x=0.08)
    return _save(fig, out_dir / "quality_rank_n78_all_models.png")


def plot_quality_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> str:
    row_by_key = _by_key(summary_rows)
    matrix = np.array(
        [[_float(row_by_key.get((size, model), {}).get("surrogate_quality_score")) for size in SIZES] for model in MODEL_ORDER],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.8, 8.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(SIZES)), [f"n{size}" for size in SIZES])
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Reliability score by model and dataset size", loc="left", fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 1.2 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("surrogate_quality_score")
    return _save(fig, out_dir / "quality_heatmap_all_models.png")


def plot_core_quality_trends(summary_rows: list[dict[str, Any]], out_dir: Path) -> str:
    row_by_key = _by_key(summary_rows)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    palette = plt.get_cmap("tab10")
    for idx, model in enumerate(CORE_MODELS):
        values = [_float(row_by_key.get((size, model), {}).get("surrogate_quality_score")) for size in SIZES]
        ax.plot(SIZES, values, marker=markers[idx % len(markers)], linewidth=2.1, color=palette(idx), label=model)
    _setup_axis(ax, "Core-model reliability trend", "dataset size", "surrogate_quality_score")
    ax.set_xticks(SIZES)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    return _save(fig, out_dir / "quality_trend_core_models.png")


def plot_score_components(summary_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = sorted(
        [row for row in _rows_for_size(summary_rows, 78) if str(row["model_id"]) in CORE_MODELS],
        key=lambda row: _float(row.get("surrogate_quality_score")),
    )
    labels = [str(row["model_id"]) for row in rows]
    if any(math.isfinite(_float(row.get("score_spatial_huber_component"))) for row in rows):
        components = [
            ("score_spatial_huber_component", "spatial huber", COLORS["spatial_huber"]),
            ("score_avgpool_huber_component", "avgpool huber", COLORS["avgpool_huber"]),
        ]
    else:
        components = [
            ("score_nrmse_component", "nrmse", COLORS["nrmse"]),
            ("score_boundary_component", "boundary", COLORS["boundary"]),
            ("score_continuity_component", "continuity", COLORS["continuity"]),
            ("score_physics_component", "physics", COLORS["physics"]),
            ("score_sign_component", "sign", COLORS["sign"]),
        ]
    values = np.array([[_float(row.get(key)) for key, _, _ in components] for row in rows], dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(10.4, 5.8), constrained_layout=True)
    y = np.arange(len(labels))
    left = np.zeros(len(labels), dtype=float)
    for idx, (_, label, color) in enumerate(components):
        ax.barh(y, values[:, idx], left=left, color=color, label=label)
        left += values[:, idx]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    _setup_axis(ax, "n78 score components for core models", "component value")
    ax.legend(ncol=5, fontsize=8, frameon=False)
    return _save(fig, out_dir / "score_components_n78_core_models.png")


def plot_target_r2_heatmap(leaderboard_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = [row for row in leaderboard_rows if int(float(row["dataset_size"])) == 78]
    rows = sorted(rows, key=lambda row: _float(row.get("surrogate_quality_score")))
    labels = [str(row["model_id"]) for row in rows]
    matrix = np.array([[_float(row.get(f"test_r2_{target}_plasma")) for target in TARGETS] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 8.0), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)), TARGETS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("n78 target-wise plasma R2", loc="left", fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("R2")
    return _save(fig, out_dir / "target_r2_plasma_heatmap_n78.png")


def plot_target_rmse_heatmap(leaderboard_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = [row for row in leaderboard_rows if int(float(row["dataset_size"])) == 78]
    rows = sorted(rows, key=lambda row: _float(row.get("surrogate_quality_score")))
    labels = [str(row["model_id"]) for row in rows]
    raw = np.array([[_float(row.get(f"test_rmse_{target}_plasma")) for target in TARGETS] for row in rows], dtype=float)
    col_max = np.nanmax(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = raw / col_max[None, :]
    fig, ax = plt.subplots(figsize=(7.6, 8.0), constrained_layout=True)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)), TARGETS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("n78 target-wise plasma RMSE (physical units)", loc="left", fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = raw[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.1e}", ha="center", va="center", fontsize=6, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE / target-column max")
    return _save(fig, out_dir / "target_rmse_plasma_heatmap_n78.png")


def plot_negative_ratio_heatmap(leaderboard_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = [row for row in leaderboard_rows if int(float(row["dataset_size"])) == 78]
    rows = sorted(rows, key=lambda row: _float(row.get("positive_target_negative_ratio_penalty")), reverse=True)
    labels = [str(row["model_id"]) for row in rows]
    matrix = np.array([[_float(row.get(f"test_neg_ratio_{target}_plasma")) for target in TARGETS] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 8.0), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)), TARGETS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("n78 negative-value ratio by target", loc="left", fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("negative ratio")
    return _save(fig, out_dir / "negative_ratio_heatmap_n78.png")


def plot_quality_boundary_scatter(summary_rows: list[dict[str, Any]], out_dir: Path) -> str:
    rows = _rows_for_size(summary_rows, 78)
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    for mode in ("table_only", "table_plus_structure"):
        subset = [row for row in rows if str(row.get("input_mode_effective")) == mode]
        x = [_float(row.get("sdf_boundary_to_deep_rmse_ratio_mean")) for row in subset]
        y = [_float(row.get("surrogate_quality_score")) for row in subset]
        sizes = [60.0 + 1800.0 * max(_float(row.get("positive_target_negative_ratio_penalty")), 0.0) for row in subset]
        ax.scatter(x, y, s=sizes, alpha=0.72, label=mode, color=COLORS.get(mode, "#333333"), edgecolor="white", linewidth=0.7)
        for row, xx, yy in zip(subset, x, y):
            if math.isfinite(xx) and math.isfinite(yy):
                ax.text(xx, yy, str(row["model_id"]), fontsize=7, ha="left", va="bottom")
    _setup_axis(
        ax,
        "n78 quality vs boundary/deep RMSE ratio",
        "sdf_boundary_to_deep_rmse_ratio_mean",
        "surrogate_quality_score",
    )
    ax.legend(frameon=False)
    return _save(fig, out_dir / "quality_vs_boundary_ratio_n78.png")


def plot_runtime_heatmap(run_root: Path, out_dir: Path) -> str:
    status_path = run_root / "run_status.csv"
    rows = _read_csv(status_path)
    row_by_key = {(int(float(row["dataset_size"])), str(row["model_id"])): row for row in rows}
    matrix = np.array(
        [[_float(row_by_key.get((size, model), {}).get("seconds")) / 60.0 for size in SIZES] for model in MODEL_ORDER],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.8, 8.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(SIZES)), [f"n{size}" for size in SIZES])
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Runtime by model and dataset size", loc="left", fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("minutes")
    return _save(fig, out_dir / "runtime_heatmap_minutes.png")


def _write_manifest(out_dir: Path, plots: list[str]) -> None:
    descriptions = {
        "quality_rank_n78_all_models.png": "n78 reliability ranking across all models.",
        "quality_heatmap_all_models.png": "Reliability score by model and dataset size.",
        "quality_trend_core_models.png": "Dataset-size trend for the core paper models.",
        "score_components_n78_core_models.png": "Score component comparison for n78 core models.",
        "target_r2_plasma_heatmap_n78.png": "Target-wise plasma R2 for n78.",
        "target_rmse_plasma_heatmap_n78.png": "Target-wise plasma RMSE for n78 in physical units.",
        "negative_ratio_heatmap_n78.png": "Negative-value ratios by target for n78.",
        "quality_vs_boundary_ratio_n78.png": "Scatter plot of quality score versus boundary/deep RMSE ratio.",
        "runtime_heatmap_minutes.png": "GPU benchmark runtime by model and dataset size.",
    }
    payload = {
        "plots": plots,
        "descriptions": descriptions,
        "notes": [
            "surrogate_quality_score is lower-better.",
            "R2 and RMSE plots use leaderboard.csv target metrics for n78.",
            "Correct dual/interp/extrap columns require leaderboards generated with the dual-axis aggregation fix.",
        ],
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# GEC-CCP Trustworthy Plot Index",
        "",
        "- Lower `surrogate_quality_score` is better.",
        "- `target_r2_plasma_heatmap_n78.png` and `target_rmse_plasma_heatmap_n78.png` use per-target metrics from each model leaderboard.",
        "- Correct dual/interp/extrap columns require leaderboards generated with the dual-axis aggregation fix.",
        "",
    ]
    for path in plots:
        name = Path(path).name
        lines.extend([f"## {name}", "", descriptions.get(name, ""), "", f"![{name}]({name})", ""])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root)
    summary_rows = _read_csv(Path(args.summary_csv))
    leaderboard_rows = _load_leaderboards(run_root, MODEL_ORDER, SIZES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        plot_quality_rank(summary_rows, out_dir),
        plot_quality_heatmap(summary_rows, out_dir),
        plot_core_quality_trends(summary_rows, out_dir),
        plot_score_components(summary_rows, out_dir),
        plot_target_r2_heatmap(leaderboard_rows, out_dir),
        plot_target_rmse_heatmap(leaderboard_rows, out_dir),
        plot_negative_ratio_heatmap(leaderboard_rows, out_dir),
        plot_quality_boundary_scatter(summary_rows, out_dir),
        plot_runtime_heatmap(run_root, out_dir),
    ]
    _write_manifest(out_dir, plots)
    for path in plots:
        print(path)
    print((out_dir / "plot_manifest.json").as_posix())
    print((out_dir / "README.md").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
