from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path("reports/icp_stage4_conference_problem_setting")
OUT = ROOT / "slide_figures"
DATASET_SUMMARY = Path("data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1/conversion_summary.json")
DATASET_ROOT = Path("data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1")
ABLATION_CSV = ROOT / "multifield_improvement_ablation_summary.csv"
MODEL_CSV = ROOT / "multifield_part_sdf_lite_v1_e80_model_comparison.csv"
RUN_ROOT = Path("runs/icp_stage4_multifield_8field_part_sdf_lite_v1_e80/full")
CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_multifield_8field_part_sdf_lite_v1_e80/full")
sys.path.insert(0, str(Path("src").resolve()))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

TARGETS = ["ne", "ni", "Te", "phi", "Br", "Bz", "Jelr", "Jelz"]
TARGET_LABELS = ["$n_e$", "$n_i$", "$T_e$", "$\\phi$", "$B_r$", "$B_z$", "$J_{el,r}$", "$J_{el,z}$"]
COLORS = {
    "blue": "#3B6FB6",
    "orange": "#D97732",
    "green": "#2E8B57",
    "red": "#C44E52",
    "purple": "#7B5EA7",
    "gray": "#666666",
    "light_gray": "#D9DEE7",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors="#374151")


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(OUT / f"{name}{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _display_field(values: np.ndarray, var_name: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=np.float32)
    if var_name in {"ne", "ni"}:
        return arr / 1.0e18, f"{var_name} [$10^{{18}}$ m$^{{-3}}$]"
    if var_name in {"Br", "Bz"}:
        return arr * 1.0e3, f"{var_name} [mT]"
    if var_name in {"Jelr", "Jelz"}:
        return arr, f"{var_name} [A m$^{{-2}}$]"
    if var_name == "Te":
        return arr, "$T_e$ [eV]"
    if var_name == "phi":
        return arr, "$\\phi$ [V]"
    return arr, var_name


def _is_signed_field(var_name: str) -> bool:
    return var_name in {"phi", "Br", "Bz", "Jelr", "Jelz"}


def _robust_limits(arrays: Sequence[np.ndarray], *, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    finite_parts = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays if np.size(a)]
    finite = np.concatenate(finite_parts) if finite_parts else np.array([0.0, 1.0], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, q_low))
    vmax = float(np.nanpercentile(finite, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _symmetric_limit(values: np.ndarray, *, q: float = 99.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    lim = float(np.nanpercentile(np.abs(arr), q))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    return lim if np.isfinite(lim) and lim > 0.0 else 1.0


def _annotate_bars(ax: plt.Axes, bars: Iterable, *, fmt: str = "{:.2f}", dy: float = 0.02) -> None:
    for bar in bars:
        h = float(bar.get_height())
        if not np.isfinite(h):
            continue
        x = float(bar.get_x() + bar.get_width() / 2.0)
        va = "bottom" if h >= 0 else "top"
        offset = dy if h >= 0 else -dy
        ax.text(x, h + offset, fmt.format(h), ha="center", va=va, fontsize=9, color="#111827")


def plot_dataset_design_space() -> None:
    summary = json.loads(DATASET_SUMMARY.read_text(encoding="utf-8"))
    counts = {int(k): int(v) for k, v in summary["nncoil_counts"].items()}
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar([str(x) for x in xs], ys, color=COLORS["blue"], width=0.62)
    _annotate_bars(ax, bars, fmt="{:.0f}", dy=2.0)
    _style_axes(ax)
    ax.set_title("ICP_stage4 dataset design space", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("Number of active coils ($n_{coil}$)", fontsize=12)
    ax.set_ylabel("Cases", fontsize=12)
    ax.set_ylim(0, max(ys) * 1.28)
    ax.text(
        0.02,
        0.92,
        "360 cases = 60 coil structures x 6 process settings\n"
        "Scalar process inputs: pp, pp0\n"
        "Spatial targets: ne, ni, Te, phi, Br, Bz, Jelr, Jelz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )
    _save(fig, "fig01_dataset_design_space")


def plot_feature_ablation() -> None:
    rows = _read_csv(ABLATION_CSV)
    model_rows = _read_csv(MODEL_CSV)
    labels = [
        "part_lite\n8 fields\nE20",
        "part_lite\nCore4 only\nE20",
        "part_lite\nweighted\nE20",
        "slot SDF\n8 fields\nE20",
        "slot SDF\nFFNO\nE80",
        "slot SDF\nUNet\nE80",
    ]
    values = [
        _float(next(r for r in rows if r["run_id"] == "bad_baseline_8field_part_lite_v2"), "primary_metric_value"),
        _float(next(r for r in rows if r["run_id"] == "core4_only_part_lite_e20"), "primary_metric_value"),
        _float(next(r for r in rows if r["run_id"] == "eightfield_part_lite_density_weighted_e20"), "primary_metric_value"),
        _float(next(r for r in rows if r["run_id"] == "eightfield_part_sdf_lite_e20"), "primary_metric_value"),
        _float(next(r for r in model_rows if r["model"] == "ffno"), "primary_metric_value"),
        _float(next(r for r in model_rows if r["model"] == "unet"), "primary_metric_value"),
    ]
    colors = [COLORS["red"], COLORS["red"], COLORS["red"], COLORS["green"], COLORS["blue"], COLORS["orange"]]

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.68)
    _annotate_bars(ax, bars)
    _style_axes(ax)
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(-0.12, 1.08)
    ax.set_ylabel("Mean plasma $R^2$ on held-out structures", fontsize=12)
    ax.set_title("Structure representation dominates multi-field surrogate accuracy", fontsize=17, weight="bold", pad=14)
    ax.text(
        0.02,
        0.95,
        "Order-invariant part summaries lose radial coil-slot information.\n"
        "Fixed per-coil SDF slots recover the ICP_stage4 density and field maps.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )
    _save(fig, "fig02_feature_ablation_mean_r2")


def plot_e80_per_target_r2() -> None:
    rows = _read_csv(MODEL_CSV)
    ffno = next(r for r in rows if r["model"] == "ffno")
    unet = next(r for r in rows if r["model"] == "unet")
    ffno_vals = [_float(ffno, f"r2_{t}_plasma") for t in TARGETS]
    unet_vals = [_float(unet, f"r2_{t}_plasma") for t in TARGETS]

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(TARGETS))
    w = 0.36
    b1 = ax.bar(x - w / 2, ffno_vals, width=w, color=COLORS["blue"], label="FFNO E80")
    b2 = ax.bar(x + w / 2, unet_vals, width=w, color=COLORS["orange"], label="UNet E80")
    _style_axes(ax)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylabel("Plasma-region $R^2$", fontsize=12)
    ax.set_title("Eight-field prediction accuracy by target", fontsize=17, weight="bold", pad=14)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _annotate_bars(ax, b1, dy=0.012)
    _annotate_bars(ax, b2, dy=0.012)
    _save(fig, "fig03_e80_per_target_r2")


def _read_metrics(model: str) -> list[dict[str, str]]:
    path = RUN_ROOT / model / "models" / model / "eval_protocol" / "extrap" / "train" / "scalars" / "metrics.csv"
    return _read_csv(path)


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=True)
    for model, color, label in (("ffno", COLORS["blue"], "FFNO"), ("unet", COLORS["orange"], "UNet")):
        rows = _read_metrics(model)
        epochs = np.array([_float(r, "epoch") for r in rows], dtype=float)
        train_loss = np.array([_float(r, "train_loss") for r in rows], dtype=float)
        val_score = np.array([_float(r, "val_balance_score") for r in rows], dtype=float)
        val_loss = np.array([_float(r, "val_loss") for r in rows], dtype=float)
        axes[0].plot(epochs, train_loss, color=color, linewidth=2.2, linestyle="-", label=f"{label} train")
        axes[0].plot(epochs, val_loss, color=color, linewidth=2.2, linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, val_score, color=color, linewidth=2.2, label=label)
        if np.isfinite(val_score).any():
            idx = int(np.nanargmax(val_score))
            axes[1].scatter([epochs[idx]], [val_score[idx]], color=color, edgecolor="white", linewidth=1.2, zorder=3)
            axes[1].text(epochs[idx], val_score[idx] + 0.025, f"{label} best {val_score[idx]:.3f}", fontsize=9, color=color)
    axes[0].set_title("Train / validation loss", fontsize=14, weight="bold")
    axes[0].set_ylabel("Loss (log scale)", fontsize=12)
    axes[0].set_yscale("log")
    axes[1].set_title("Validation balance score", fontsize=14, weight="bold")
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_ylim(0.25, 1.02)
    for ax in axes:
        _style_axes(ax)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Training dynamics for 8-field slot-SDF surrogates", fontsize=17, weight="bold", y=1.02)
    _save(fig, "fig04_training_curves_e80")


def plot_e20_to_e80_improvement() -> None:
    e20 = next(r for r in _read_csv(ABLATION_CSV) if r["run_id"] == "eightfield_part_sdf_lite_e20")
    e80 = next(r for r in _read_csv(MODEL_CSV) if r["model"] == "ffno")
    e20_vals = [_float(e20, f"r2_{t}_plasma") for t in TARGETS]
    e80_vals = [_float(e80, f"r2_{t}_plasma") for t in TARGETS]

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(TARGETS))
    w = 0.36
    ax.bar(x - w / 2, e20_vals, width=w, color="#9CA3AF", label="FFNO E20")
    ax.bar(x + w / 2, e80_vals, width=w, color=COLORS["blue"], label="FFNO E80")
    _style_axes(ax)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylabel("Plasma-region $R^2$", fontsize=12)
    ax.set_title("Longer training resolves density-field accuracy", fontsize=17, weight="bold", pad=14)
    ax.legend(frameon=False, loc="lower right")
    for i, (a, b) in enumerate(zip(e20_vals, e80_vals)):
        ax.annotate(
            "",
            xy=(i + w / 2, b),
            xytext=(i - w / 2, a),
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.0, "alpha": 0.75},
        )
    _save(fig, "fig05_e20_to_e80_improvement_ffno")


def plot_region_r2_heatmap() -> None:
    rows = _read_csv(RUN_ROOT / "ffno" / "models" / "ffno" / "eval_protocol" / "extrap" / "eval" / "spatial_error_summary.csv")
    regions = ["boundary_in", "plasma_mid", "plasma_deep", "all_plasma"]
    matrix = np.full((len(TARGETS), len(regions)), np.nan)
    for i, target in enumerate(TARGETS):
        for j, region in enumerate(regions):
            match = [r for r in rows if r.get("var") == target and r.get("region") == region]
            if match:
                matrix[i, j] = _float(match[0], "r2")

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.85, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(["Boundary", "Mid plasma", "Deep plasma", "All plasma"], fontsize=11)
    ax.set_yticks(np.arange(len(TARGETS)))
    ax.set_yticklabels(TARGET_LABELS, fontsize=12)
    ax.set_title("FFNO E80 regional $R^2$ keeps boundary accuracy high", fontsize=17, weight="bold", pad=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$R^2$", fontsize=12)
    _save(fig, "fig06_ffno_e80_regional_r2_heatmap")


def plot_density_distribution_errors() -> None:
    rows = _read_csv(RUN_ROOT / "ffno" / "models" / "ffno" / "eval_protocol" / "extrap" / "eval" / "spatial_distribution_summary.csv")
    metrics = ["integral_rel_error_mean", "p99_rel_error_mean", "distribution_error_score_mean"]
    labels = ["Integral rel. error", "P99 rel. error", "Distribution score"]
    vals = {target: [] for target in ("ne", "ni")}
    for target in vals:
        row = next(r for r in rows if r["var"] == target)
        vals[target] = [_float(row, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, vals["ne"], width=w, color=COLORS["blue"], label="$n_e$")
    ax.bar(x + w / 2, vals["ni"], width=w, color=COLORS["green"], label="$n_i$")
    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean error over held-out cases", fontsize=12)
    ax.set_title("Density distribution errors remain small after E80 training", fontsize=17, weight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig07_density_distribution_errors_ffno")


def _test_case_quality() -> list[dict[str, Any]]:
    rows = _read_csv(RUN_ROOT / "ffno" / "models" / "ffno" / "eval_protocol" / "extrap" / "eval" / "spatial_error_by_case.csv")
    by_case: dict[str, list[float]] = {}
    for row in rows:
        if row.get("region") != "all_plasma":
            continue
        value = _float(row, "r2")
        if np.isfinite(value):
            by_case.setdefault(str(row["case_id"]), []).append(value)
    out = [
        {"case_id": case_id, "mean_r2": float(np.mean(values))}
        for case_id, values in by_case.items()
        if values
    ]
    return sorted(out, key=lambda r: float(r["mean_r2"]))


def _selected_visual_cases() -> list[str]:
    ranked = _test_case_quality()
    if not ranked:
        return ["case_g008_op01"]
    return [
        str(ranked[-1]["case_id"]),
        str(ranked[len(ranked) // 2]["case_id"]),
        str(ranked[0]["case_id"]),
    ]


def _load_prediction_bundle(model_name: str, case_ids: Sequence[str]) -> dict[str, Any]:
    if str(model_name).strip().lower() == "ffno" and not _torch_runtime_available():
        return _load_ffno_prediction_bundle_numpy(case_ids)

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

    cfg_path = CONFIG_ROOT / f"benchmark_icp_stage4_core4_full_{model_name}.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = dict(cfg.get("benchmark", {}) or {})
    run_dir = RUN_ROOT / model_name
    dataset = load_dataset({"dataset": bench["dataset"]}, run_dir)
    bundle = RunBundleLoader.load(run_dir)
    transforms = bundle.transform_bundle()
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    case_to_idx = {str(case["case_id"]): i for i, case in enumerate(dataset.cases)}
    idx = np.asarray([case_to_idx[str(case_id)] for case_id in case_ids], dtype=np.int64)

    model = load_checkpoint(run_dir / "models" / model_name / "eval_protocol" / "extrap" / "checkpoints")
    train_cfg = dict(dict(bench.get("train", {}) or {}).get(model_name, {}) or {})
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
        raise RuntimeError(f"case spatial feature pack is unavailable for {model_name}: {source}")
    spatial_selected = spatial_source.subset(idx)
    pred_scaled = _predict_features_batched(
        model,
        cond_scaled[idx],
        spatial_features=spatial_selected,
        batch_size_cases=1,
    )
    y_order = [str(v) for v in bundle.schemas.get("output_layout", {}).get("vars", TARGETS)]
    pred_phys = transforms.inverse_field_dict({name: np.asarray(pred_scaled[name], dtype=np.float32) for name in y_order})
    pred = {name: np.asarray(pred_phys[name], dtype=np.float32)[:, 0] for name in y_order}
    true = {
        name: np.stack([np.asarray(dataset.cases[int(i)]["y"][name], dtype=np.float32) for i in idx], axis=0)
        for name in y_order
    }
    return {
        "case_ids": [str(v) for v in case_ids],
        "true": true,
        "pred": pred,
        "mask": _load_plasma_mask(Path(dataset.geometry_root)),
    }


def _gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))).astype(np.float32)


def _conv1x1(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.einsum("bihw,oi->bohw", x, w[:, :, 0, 0], optimize=True)
    out += b[None, :, None, None]
    return out.astype(np.float32)


def _depthwise_conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    bsz, channels, h, ww = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.zeros((bsz, channels, h, ww), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += padded[:, :, dy : dy + h, dx : dx + ww] * w[:, 0, dy, dx][None, :, None, None]
    out += b[None, :, None, None]
    return out.astype(np.float32)


def _spectral_filter(size: int, modes: int, *, dealias_ratio: float = 0.85, taper_alpha: float = 1.5) -> np.ndarray:
    freq = np.fft.rfftfreq(int(size), d=1.0).astype(np.float32) / 0.5
    max_freq = float(np.max(freq[: max(int(modes), 1)]))
    cutoff = max(float(dealias_ratio) * max_freq, 1.0e-6)
    ratio = np.clip(freq / cutoff, 0.0, None)
    mask = (freq <= cutoff).astype(np.float32) * np.exp(-float(taper_alpha) * ratio**4).astype(np.float32)
    return mask


def _spectral_axis(x: np.ndarray, weight_pair: np.ndarray, *, axis: str, modes: int) -> np.ndarray:
    weight = (weight_pair[..., 0] + 1j * weight_pair[..., 1]).astype(np.complex64)
    if axis == "h":
        size = int(x.shape[-2])
        x_ft = np.fft.rfft(x, axis=-2, norm="ortho").astype(np.complex64)
        m = min(int(modes), int(x_ft.shape[-2]))
        out_ft = np.zeros((x.shape[0], weight.shape[1], x_ft.shape[-2], x.shape[-1]), dtype=np.complex64)
        out_ft[:, :, :m, :] = np.einsum("bimw,iom->bomw", x_ft[:, :, :m, :], weight[:, :, :m], optimize=True)
        out_ft *= _spectral_filter(size, m)[None, None, :, None]
        return np.fft.irfft(out_ft, n=size, axis=-2, norm="ortho").astype(np.float32)
    size = int(x.shape[-1])
    x_ft = np.fft.rfft(x, axis=-1, norm="ortho").astype(np.complex64)
    m = min(int(modes), int(x_ft.shape[-1]))
    out_ft = np.zeros((x.shape[0], weight.shape[1], x.shape[-2], x_ft.shape[-1]), dtype=np.complex64)
    out_ft[:, :, :, :m] = np.einsum("bihm,iom->bohm", x_ft[:, :, :, :m], weight[:, :, :m], optimize=True)
    out_ft *= _spectral_filter(size, m)[None, None, None, :]
    return np.fft.irfft(out_ft, n=size, axis=-1, norm="ortho").astype(np.float32)


def _manual_ffno_forward_scaled(feature_map: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    x = np.moveaxis(np.asarray(feature_map, dtype=np.float32), -1, 1)
    h = _conv1x1(x, weights["torch::in_proj.weight"], weights["torch::in_proj.bias"])
    for layer in range(4):
        prefix = f"torch::blocks.{layer}"
        spec_h = _spectral_axis(h, weights[f"{prefix}.spec.weight_h"], axis="h", modes=16)
        spec_w = _spectral_axis(h, weights[f"{prefix}.spec.weight_w"], axis="w", modes=16)
        skip = _conv1x1(h, weights[f"{prefix}.skip.weight"], weights[f"{prefix}.skip.bias"])
        local = _depthwise_conv3x3(skip, weights[f"{prefix}.local_skip.weight"], weights[f"{prefix}.local_skip.bias"])
        alpha = float(np.asarray(weights[f"{prefix}.local_skip_alpha"]))
        h = _gelu(spec_h + spec_w + skip + alpha * local)
    h = _gelu(_conv1x1(h, weights["torch::post.0.weight"], weights["torch::post.0.bias"]))
    return _conv1x1(h, weights["torch::post.3.weight"], weights["torch::post.3.bias"])


def _load_ffno_prediction_bundle_numpy(case_ids: Sequence[str]) -> dict[str, Any]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.train.spatial_features import (
        build_case_spatial_features,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    model_name = "ffno"
    cfg_path = CONFIG_ROOT / "benchmark_icp_stage4_core4_full_ffno.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = dict(cfg.get("benchmark", {}) or {})
    run_dir = RUN_ROOT / model_name
    dataset = load_dataset({"dataset": bench["dataset"]}, run_dir)
    bundle = RunBundleLoader.load(run_dir)
    transforms = bundle.transform_bundle()
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    case_to_idx = {str(case["case_id"]): i for i, case in enumerate(dataset.cases)}
    idx = np.asarray([case_to_idx[str(case_id)] for case_id in case_ids], dtype=np.int64)
    train_cfg = dict(dict(bench.get("train", {}) or {}).get(model_name, {}) or {})
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
        raise RuntimeError(f"case spatial feature pack is unavailable for ffno: {source}")
    spatial = spatial_source.subset(idx).batch(np.arange(len(idx), dtype=np.int64))
    h, w = int(dataset.shape[0]), int(dataset.shape[1])
    cond_map = np.repeat(cond_scaled[idx, None, None, :], h, axis=1)
    cond_map = np.repeat(cond_map, w, axis=2)
    feature_map = np.concatenate([cond_map, spatial], axis=-1).astype(np.float32)
    ckpt = run_dir / "models" / model_name / "eval_protocol" / "extrap" / "checkpoints" / "weights.npz"
    with np.load(ckpt) as raw:
        weights = {k: np.asarray(raw[k], dtype=np.float32) for k in raw.files}
    raw_pred = _manual_ffno_forward_scaled(feature_map, weights)
    y_order = [str(v) for v in bundle.schemas.get("output_layout", {}).get("vars", TARGETS)]
    pred_scaled = {name: raw_pred[:, i : i + 1] for i, name in enumerate(y_order)}
    pred_phys = transforms.inverse_field_dict(pred_scaled)
    pred = {name: np.asarray(pred_phys[name], dtype=np.float32)[:, 0] for name in y_order}
    true = {
        name: np.stack([np.asarray(dataset.cases[int(i)]["y"][name], dtype=np.float32) for i in idx], axis=0)
        for name in y_order
    }
    return {
        "case_ids": [str(v) for v in case_ids],
        "true": true,
        "pred": pred,
        "mask": _load_plasma_mask(Path(dataset.geometry_root)),
    }


def _load_plasma_mask(geometry_root: Path) -> np.ndarray | None:
    candidates = (
        geometry_root / "geometry" / "mask_plasma.npy",
        geometry_root / "mask_plasma.npy",
    )
    for path in candidates:
        if path.exists():
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
    return None


def _map_for_plot(values: np.ndarray, var_name: str) -> tuple[np.ndarray, str]:
    return _display_field(values, var_name)


def _draw_mask_contour(ax: plt.Axes, mask: np.ndarray | None) -> None:
    if mask is None:
        return
    ax.contour(mask[::2, ::2], levels=[0.5], colors="black", linewidths=0.45, alpha=0.55, origin="lower")


def _imshow(ax: plt.Axes, arr: np.ndarray, *, cmap: str, vmin: float, vmax: float, mask: np.ndarray | None) -> Any:
    im = ax.imshow(np.asarray(arr)[::2, ::2], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    _draw_mask_contour(ax, mask)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _case_metric_label(case_id: str) -> str:
    ranked = {str(row["case_id"]): float(row["mean_r2"]) for row in _test_case_quality()}
    value = ranked.get(str(case_id))
    return f"{case_id}\nmean R2={value:.3f}" if value is not None else str(case_id)


def _plot_triplet_rows(
    *,
    data: dict[str, Any],
    case_index: int,
    vars_list: Sequence[str],
    name: str,
    title: str,
) -> None:
    mask = data.get("mask")
    case_id = str(data["case_ids"][case_index])
    fig, axes = plt.subplots(len(vars_list), 3, figsize=(10.5, 2.55 * len(vars_list)), constrained_layout=True)
    if len(vars_list) == 1:
        axes = np.asarray([axes])
    for i, var_name in enumerate(vars_list):
        true_raw = np.asarray(data["true"][var_name][case_index], dtype=np.float32)
        pred_raw = np.asarray(data["pred"][var_name][case_index], dtype=np.float32)
        true_plot, label = _map_for_plot(true_raw, var_name)
        pred_plot, _ = _map_for_plot(pred_raw, var_name)
        err = pred_plot - true_plot
        vmin, vmax = _robust_limits([true_plot, pred_plot], q_low=1.0, q_high=99.0)
        elim = _symmetric_limit(err, q=99.0)
        cmap = "coolwarm" if _is_signed_field(var_name) else "viridis"
        if _is_signed_field(var_name):
            lim = _symmetric_limit(np.stack([true_plot, pred_plot]), q=99.0)
            vmin, vmax = -lim, lim
        im0 = _imshow(axes[i, 0], true_plot, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask)
        _imshow(axes[i, 1], pred_plot, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask)
        im2 = _imshow(axes[i, 2], err, cmap="coolwarm", vmin=-elim, vmax=elim, mask=mask)
        axes[i, 0].set_ylabel(label, fontsize=11)
        fig.colorbar(im0, ax=axes[i, :2], shrink=0.72, pad=0.01)
        fig.colorbar(im2, ax=axes[i, 2], shrink=0.72, pad=0.01)
    for ax, col in zip(axes[0], ("Ground truth", "FFNO prediction", "Prediction - truth")):
        ax.set_title(col, fontsize=12, weight="bold")
    fig.suptitle(f"{title}: {_case_metric_label(case_id)}", fontsize=16, weight="bold")
    _save(fig, name)


def plot_ffno_spatial_triplets() -> None:
    case_id = _selected_visual_cases()[1]
    data = _load_prediction_bundle("ffno", [case_id])
    _plot_triplet_rows(
        data=data,
        case_index=0,
        vars_list=["ne", "ni", "Te", "phi"],
        name="fig08_ffno_core_field_triplets",
        title="Held-out test spatial maps, plasma variables",
    )
    _plot_triplet_rows(
        data=data,
        case_index=0,
        vars_list=["Br", "Bz", "Jelr", "Jelz"],
        name="fig09_ffno_electromagnetic_field_triplets",
        title="Held-out test spatial maps, electromagnetic variables",
    )


def _plot_density_quality_grid(data: dict[str, Any], *, var_name: str, name: str, title: str) -> None:
    mask = data.get("mask")
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.0), constrained_layout=True)
    for row_idx, quality in enumerate(("best", "median", "worst")):
        case_id = str(data["case_ids"][row_idx])
        true_plot, label = _display_field(data["true"][var_name][row_idx], var_name)
        pred_plot, _ = _display_field(data["pred"][var_name][row_idx], var_name)
        err = pred_plot - true_plot
        vmin, vmax = _robust_limits([true_plot, pred_plot], q_low=1.0, q_high=99.0)
        elim = _symmetric_limit(err, q=99.0)
        im0 = _imshow(axes[row_idx, 0], true_plot, cmap="viridis", vmin=vmin, vmax=vmax, mask=mask)
        _imshow(axes[row_idx, 1], pred_plot, cmap="viridis", vmin=vmin, vmax=vmax, mask=mask)
        im2 = _imshow(axes[row_idx, 2], err, cmap="coolwarm", vmin=-elim, vmax=elim, mask=mask)
        axes[row_idx, 0].set_ylabel(f"{quality}\n{_case_metric_label(case_id)}\n{label}", fontsize=10)
        fig.colorbar(im0, ax=axes[row_idx, :2], shrink=0.62, pad=0.01)
        fig.colorbar(im2, ax=axes[row_idx, 2], shrink=0.62, pad=0.01)
    for ax, col in zip(axes[0], ("Ground truth", "FFNO prediction", "Prediction - truth")):
        ax.set_title(col, fontsize=12, weight="bold")
    fig.suptitle(title, fontsize=16, weight="bold")
    _save(fig, name)


def plot_density_best_median_worst() -> None:
    case_ids = _selected_visual_cases()
    data = _load_prediction_bundle("ffno", case_ids)
    _plot_density_quality_grid(
        data,
        var_name="ne",
        name="fig10_ffno_ne_best_median_worst_maps",
        title="Electron density visual fit across best / median / worst held-out cases",
    )
    _plot_density_quality_grid(
        data,
        var_name="ni",
        name="fig11_ffno_ni_best_median_worst_maps",
        title="Ion density visual fit across best / median / worst held-out cases",
    )


def plot_line_profiles_best_median_worst() -> None:
    case_ids = _selected_visual_cases()
    data = _load_prediction_bundle("ffno", case_ids)
    h, w = np.asarray(data["true"]["ne"][0]).shape
    row = int(round(0.5 * (h - 1)))
    x = np.arange(w)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.6), constrained_layout=True)
    for col, case_id in enumerate(case_ids):
        for ax, var_name, ylabel in (
            (axes[0, col], "ne", "$n_e$"),
            (axes[1, col], "Te", "$T_e$"),
        ):
            true, ylabel_eff = _display_field(data["true"][var_name][col, row, :], var_name)
            pred, _ = _display_field(data["pred"][var_name][col, row, :], var_name)
            ax.plot(x, true, color="#111827", linewidth=1.8, label="truth")
            ax.plot(x, pred, color=COLORS["blue"], linewidth=1.8, linestyle="--", label="FFNO")
            _style_axes(ax)
            ax.set_title(_case_metric_label(str(case_id)), fontsize=10)
            ax.set_xlabel("r pixel", fontsize=10)
            ax.set_ylabel(ylabel_eff if var_name == "ne" else ylabel, fontsize=10)
            if col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Mid-height radial line profiles on held-out structures (row={row})", fontsize=16, weight="bold")
    _save(fig, "fig12_ffno_midheight_line_profiles")


def plot_ffno_unet_visual_comparison() -> None:
    case_id = _selected_visual_cases()[2]
    ffno = _load_prediction_bundle("ffno", [case_id])
    unet = _load_prediction_bundle("unet", [case_id])
    mask = ffno.get("mask")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.7), constrained_layout=True)
    for row_idx, var_name in enumerate(("ne", "ni")):
        truth, label = _display_field(ffno["true"][var_name][0], var_name)
        ffno_pred, _ = _display_field(ffno["pred"][var_name][0], var_name)
        unet_pred, _ = _display_field(unet["pred"][var_name][0], var_name)
        vmin, vmax = _robust_limits([truth, ffno_pred, unet_pred], q_low=1.0, q_high=99.0)
        im = _imshow(axes[row_idx, 0], truth, cmap="viridis", vmin=vmin, vmax=vmax, mask=mask)
        _imshow(axes[row_idx, 1], ffno_pred, cmap="viridis", vmin=vmin, vmax=vmax, mask=mask)
        _imshow(axes[row_idx, 2], unet_pred, cmap="viridis", vmin=vmin, vmax=vmax, mask=mask)
        axes[row_idx, 0].set_ylabel(label, fontsize=11)
        fig.colorbar(im, ax=axes[row_idx, :], shrink=0.72, pad=0.01)
    for ax, title in zip(axes[0], ("Ground truth", "FFNO E80", "UNet E80")):
        ax.set_title(title, fontsize=12, weight="bold")
    fig.suptitle(f"Model-family visual cross-check on a difficult held-out case: {case_id}", fontsize=16, weight="bold")
    _save(fig, "fig15_ffno_unet_density_visual_compare")


def _torch_runtime_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def plot_density_spatial_error_components() -> None:
    rows = _read_csv(ROOT.parent / "icp_stage4_part_sdf_lite_v1_e80_review" / "spatial_distribution_summary_combined.csv")
    labels = ["Integral error", "P99 error", "Peak loc. error", "Shape corr.", "Distribution score"]
    metrics = [
        "integral_rel_error_mean",
        "p99_rel_error_mean",
        "peak_location_error_px_mean",
        "shape_corr_mean",
        "distribution_error_score_mean",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    for ax, var_name in zip(axes, ("ne", "ni")):
        ffno = next(r for r in rows if r.get("model_id") == "ffno" and r.get("var") == var_name)
        unet = next(r for r in rows if r.get("model_id") == "unet" and r.get("var") == var_name)
        x = np.arange(len(metrics))
        width = 0.36
        ffno_vals = [_float(ffno, m) for m in metrics]
        unet_vals = [_float(unet, m) for m in metrics]
        ax.bar(x - width / 2, ffno_vals, width=width, color=COLORS["blue"], label="FFNO")
        ax.bar(x + width / 2, unet_vals, width=width, color=COLORS["orange"], label="UNet")
        _style_axes(ax)
        ax.set_title(f"{var_name} spatial-distribution diagnostics", fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(frameon=False)
    fig.suptitle("Spatial diagnostics beyond scalar R2", fontsize=16, weight="bold")
    _save(fig, "fig13_density_spatial_error_components")


def plot_ground_truth_eight_field_example() -> None:
    case_id = "case_g027_op03"
    row = next(r for r in _read_csv(DATASET_ROOT / "index.csv") if r.get("case_id") == case_id)
    fields_path = DATASET_ROOT / str(row["fields_npz"])
    with np.load(fields_path) as data:
        fields = {name: np.asarray(data[name], dtype=np.float32) for name in TARGETS}
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.2), constrained_layout=True)
    for ax, target, label in zip(axes.reshape(-1), TARGETS, TARGET_LABELS):
        arr, plot_label = _map_for_plot(fields[target], target)
        if target in {"Br", "Bz", "Jelr", "Jelz", "phi"}:
            lim = _symmetric_limit(arr, q=99.0)
            im = ax.imshow(arr[::2, ::2], origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
        else:
            vmin, vmax = _robust_limits([arr], q_low=1.0, q_high=99.0)
            im = ax.imshow(arr[::2, ::2], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label if target not in {"ne", "ni"} else plot_label, fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.68, pad=0.01)
    fig.suptitle(f"Ground-truth eight-field spatial targets for held-out case {case_id}", fontsize=16, weight="bold")
    _save(fig, "fig14_ground_truth_eight_field_test_case")


def plot_spatial_evaluation_figures() -> None:
    try:
        plot_ffno_spatial_triplets()
        plot_density_best_median_worst()
        plot_line_profiles_best_median_worst()
    except Exception as exc:
        print(f"FFNO spatial prediction plots were skipped: {exc!r}")
    if _torch_runtime_available():
        try:
            plot_ffno_unet_visual_comparison()
        except Exception as exc:
            print(f"UNet visual comparison was skipped: {exc!r}")
    else:
        print("UNet visual comparison skipped because torch is unavailable")
    plot_density_spatial_error_components()
    plot_ground_truth_eight_field_example()


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    plot_dataset_design_space()
    plot_feature_ablation()
    plot_e80_per_target_r2()
    plot_training_curves()
    plot_e20_to_e80_improvement()
    plot_region_r2_heatmap()
    plot_density_distribution_errors()
    plot_spatial_evaluation_figures()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
