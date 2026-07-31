"""Create GEC-ICP conference figures for dimension-vector and union-SDF inputs.

The presentation follows the adopted GEC-CCP model-detail figures:

* logarithmic training/validation histories with the selected epoch, and
* electron-density Geometry/Truth/Prediction/Signed-error panels.

The two ICP runs are a controlled pair (same data split, seed, U-Net, loss,
batch size, and epoch budget).  Only the coil-structure representation differs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "icp_conference_materials" / "representation_model_details"
DEFAULT_ADOPTED_FIGURES_CSV = DEFAULT_OUT_DIR.parent / "adopted_figures.csv"
MODEL_ID = "unet"
MODEL_DISPLAY_NAME = "U-Net"
PROTOCOL = "structure_holdout"
TARGET = "ne"
TARGET_LABEL = r"$n_e$"
TARGET_UNITS = r"m$^{-3}$"
DEFAULT_ERROR_LIMIT_PERCENT = 30.0

OUTSIDE_COLOR = np.asarray([0.90, 0.90, 0.90, 1.0], dtype=np.float32)
PLASMA_COLOR = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
COIL_COLORS = (
    np.asarray([0.35, 0.35, 0.35, 1.0], dtype=np.float32),
    np.asarray([0.48, 0.42, 0.34, 1.0], dtype=np.float32),
    np.asarray([0.56, 0.48, 0.35, 1.0], dtype=np.float32),
    np.asarray([0.38, 0.46, 0.55, 1.0], dtype=np.float32),
)
PLASMA_EDGE = "#00C8E8"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class RepresentationRun:
    key: str
    display_name: str
    run_root: Path
    color: str
    structure_input: str

    @property
    def metrics_path(self) -> Path:
        return (
            self.run_root
            / "models"
            / MODEL_ID
            / "eval_protocol"
            / PROTOCOL
            / "train"
            / "scalars"
            / "metrics.csv"
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_root / "models" / MODEL_ID / "eval_protocol" / PROTOCOL / "checkpoints"

    @property
    def distribution_metrics_path(self) -> Path:
        return (
            self.run_root
            / "models"
            / MODEL_ID
            / "eval_protocol"
            / PROTOCOL
            / "eval"
            / "spatial_distribution_by_case.csv"
        )


DEFAULT_RUNS = (
    RepresentationRun(
        key="dimension",
        display_name="Dimension vector",
        run_root=REPO_ROOT / "runs" / "icp_stage4_regular_layout_representation_v1_dimension",
        color="#0072B2",
        structure_input="llcoil, rrc, nncoil, rrce, zzc (plus pp, pp0)",
    ),
    RepresentationRun(
        key="sdf",
        display_name="Union SDF",
        run_root=REPO_ROOT / "runs" / "icp_stage4_regular_layout_representation_v1_union_sdf",
        color="#D55E00",
        structure_input="case-varying part_sdf_union (plus pp, pp0)",
    ),
)


@dataclass(frozen=True)
class CaseGeometry:
    mask: np.ndarray
    part_masks: tuple[np.ndarray, ...]
    r: np.ndarray
    z: np.ndarray

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (*_axis_edges(self.r), *_axis_edges(self.z))


@dataclass(frozen=True)
class CasePrediction:
    case_id: str
    truth: np.ndarray
    prediction: np.ndarray
    geometry: CaseGeometry
    conditions: dict[str, float]
    spatial_feature_source: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream) or {})


def _selected_epoch(run: RepresentationRun) -> int:
    rows = _read_csv(run.run_root / "leaderboard.csv")
    match = [row for row in rows if str(row.get("model_id", "")).strip() == MODEL_ID]
    if len(match) != 1:
        raise ValueError(f"expected one {MODEL_ID} leaderboard row under {run.run_root}, found {len(match)}")
    return int(float(match[0]["validation_selected_epoch"]))


def _history(run: RepresentationRun) -> dict[str, np.ndarray]:
    rows = _read_csv(run.metrics_path)
    required = ("epoch", "train_loss", "val_loss")
    if not rows or any(column not in rows[0] for column in required):
        raise ValueError(f"{run.metrics_path} must contain {required}")
    values = {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in required
    }
    valid = np.isfinite(values["epoch"])
    for column in ("train_loss", "val_loss"):
        valid &= np.isfinite(values[column]) & (values[column] > 0.0)
    if not np.any(valid):
        raise ValueError(f"no positive finite training history under {run.metrics_path}")
    return {column: values[column][valid] for column in required}


def _draw_history(ax: plt.Axes, run: RepresentationRun, history: dict[str, np.ndarray]) -> None:
    epoch = history["epoch"]
    train = history["train_loss"]
    val = history["val_loss"]
    selected = _selected_epoch(run)
    selected_idx = int(np.argmin(np.abs(epoch - selected)))
    ax.plot(epoch, train, color=run.color, linewidth=1.35, alpha=0.72, label="Training loss")
    ax.plot(epoch, val, color=run.color, linewidth=1.8, linestyle="--", label="Validation loss")
    ax.axvline(selected, color="#333333", linewidth=1.0, linestyle=":", label=f"Selected epoch ({selected})")
    ax.scatter(
        [epoch[selected_idx]],
        [val[selected_idx]],
        s=28,
        color=run.color,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective value")
    ax.set_title(run.display_name)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="#eeeeee", linewidth=0.4)
    ax.legend(frameon=False, loc="best")


def _save_figure(fig: plt.Figure, stem: Path, *, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
        outputs.append(path)
    return outputs


def _plot_learning_curves(
    runs: tuple[RepresentationRun, ...],
    histories: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    *,
    dpi: int,
    figure_title: str = "GEC-ICP learning curves (controlled U-Net comparison)",
) -> list[Path]:
    outputs: list[Path] = []
    for run in runs:
        fig, ax = plt.subplots(figsize=(4.8, 3.3), constrained_layout=True)
        _draw_history(ax, run, histories[run.key])
        outputs.extend(_save_figure(fig, out_dir / f"{run.key}_learning_curve", dpi=dpi))
        plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(8.3, 3.25), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, run in zip(axes_array, runs, strict=True):
        _draw_history(ax, run, histories[run.key])
    fig.suptitle(figure_title, fontsize=11)
    outputs.extend(_save_figure(fig, out_dir / "dimension_sdf_learning_curves", dpi=dpi))
    plt.close(fig)
    return outputs


def _case_physical_rel_l2(run: RepresentationRun) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in _read_csv(run.distribution_metrics_path):
        if str(row.get("var", "")) != TARGET:
            continue
        value = float(row["physical_rel_l2"])
        if np.isfinite(value):
            values[str(row["case_id"])] = value
    if not values:
        raise ValueError(f"no {TARGET} physical_rel_l2 rows in {run.distribution_metrics_path}")
    return values


def _select_common_case(runs: tuple[RepresentationRun, ...]) -> tuple[str, dict[str, Any]]:
    by_run = {run.key: _case_physical_rel_l2(run) for run in runs}
    common = set.intersection(*(set(values) for values in by_run.values()))
    if not common:
        raise ValueError("representation runs have no common evaluated electron-density cases")
    ranked: list[tuple[float, str]] = []
    for case_id in common:
        score = float(np.mean([by_run[run.key][case_id] for run in runs]))
        ranked.append((score, case_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    median_index = len(ranked) // 2
    score, case_id = ranked[median_index]
    return case_id, {
        "method": "median of mean plasma physical relative L2 across both representations",
        "common_case_count": len(ranked),
        "rank_zero_based": median_index,
        "mean_physical_rel_l2": score,
        "per_representation_physical_rel_l2": {
            run.key: by_run[run.key][case_id]
            for run in runs
        },
    }


def _squeeze_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D field after squeezing, got {arr.shape}")
    return arr


def _case_spatial_features(
    *,
    cfg: dict[str, Any],
    bundle: Any,
    spatial_artifacts: dict[str, dict[str, Any]],
    dataset: Any,
    case_index: int,
) -> tuple[np.ndarray, str]:
    from plasma_surrogate.preprocessing.spatial_features import (
        apply_coord_feature_scaling,
        apply_distance_transform,
        build_case_spatial_features,
        build_coord_feature_rows,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    input_cfg = dict(dict(cfg.get("train", {})).get(MODEL_ID, {}).get("input_features", {}) or {})
    channels = resolve_coord_feature_channels(input_cfg.get("features"))
    distance_cfg = resolve_distance_transform_cfg(input_cfg.get("distance_transform"))
    distance_cfg, _ = resolve_distance_transform_effective(
        distance_cfg,
        stats=spatial_artifacts["distance_transform_stats"],
    )
    h, w = int(dataset.shape[0]), int(dataset.shape[1])
    expected_case_ids = [str(case["case_id"]) for case in dataset.cases]
    source, source_name = build_case_spatial_features(
        channels=channels,
        static_pack=bundle.schemas.get("static_spatial_feature_pack"),
        case_pack=bundle.schemas.get("case_structure_feature_pack"),
        h=h,
        w=w,
        distance_transform_cfg=distance_cfg,
        coord_feature_scaler_artifact=spatial_artifacts["coord_feature_scaler"],
        expected_case_ids=expected_case_ids,
    )
    if source is not None:
        return source.batch(np.asarray([case_index], dtype=np.int64)), source_name

    rows, source_name = build_coord_feature_rows(
        channels=channels,
        pack=bundle.schemas.get("coord_feature_pack"),
        geom_ctx=None,
        h=h,
        w=w,
    )
    rows, _ = apply_distance_transform(rows, channels=channels, cfg=distance_cfg)
    rows, _, _ = apply_coord_feature_scaling(
        rows,
        channels=channels,
        coord_feature_scaler_artifact=spatial_artifacts["coord_feature_scaler"],
        distance_transform_cfg=distance_cfg,
    )
    return rows.reshape(1, h, w, len(channels)).astype(np.float32), source_name


def _load_case_geometry(case: dict[str, Any], *, fallback_root: Path) -> CaseGeometry:
    raw_structure = str(case.get("structure_npz", "")).strip()
    structure_path = Path(raw_structure) if raw_structure else Path()
    if raw_structure and not structure_path.is_absolute():
        structure_path = fallback_root / structure_path
    if raw_structure and structure_path.exists():
        with np.load(structure_path, allow_pickle=True) as pack:
            mask = np.asarray(pack["mask_plasma"], dtype=np.float32) > 0.5
            r = np.asarray(pack["r_coords"], dtype=np.float64)
            z = np.asarray(pack["z_coords"], dtype=np.float64)
            raw_parts = np.asarray(pack["part_mask_stack"], dtype=np.float32)
            part_masks = tuple(np.asarray(part > 0.5, dtype=bool) for part in raw_parts if np.any(part > 0.5))
        return CaseGeometry(mask=mask, part_masks=part_masks, r=r, z=z)

    geometry_root = fallback_root / "geometry" if (fallback_root / "geometry").exists() else fallback_root
    mask = np.asarray(np.load(geometry_root / "mask_plasma.npy"), dtype=np.float32) > 0.5
    r = np.asarray(np.load(geometry_root / "r_coords.npy"), dtype=np.float64)
    z = np.asarray(np.load(geometry_root / "z_coords.npy"), dtype=np.float64)
    return CaseGeometry(mask=mask, part_masks=(), r=r, z=z)


def _predict_case(run: RepresentationRun, case_id: str) -> CasePrediction:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.input_modes import load_checkpoint_metadata_with_input_mode
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint

    cfg = _read_yaml(run.run_root / "resolved_config.yaml")
    dataset = load_dataset({"dataset": cfg["dataset"]}, run.run_root)
    lookup = {str(case["case_id"]): index for index, case in enumerate(dataset.cases)}
    if case_id not in lookup:
        raise KeyError(f"{case_id} is absent from {run.run_root}")
    split = _read_json(run.run_root / "preprocessing" / "split" / f"split_{PROTOCOL}_v1.json")
    if case_id not in {str(value) for value in split.get("test", [])}:
        raise ValueError(f"{case_id} is not a {PROTOCOL} test case for {run.key}")

    case_index = lookup[case_id]
    case = dict(dataset.cases[case_index])
    model = load_checkpoint(run.checkpoint_dir)
    checkpoint_meta, _ = load_checkpoint_metadata_with_input_mode(run.checkpoint_dir / "meta.json")
    bundle = RunBundleLoader.load(run.run_root, model=model)
    transforms = bundle.transform_bundle_for_checkpoint(checkpoint_meta)
    spatial_artifacts = bundle.spatial_transform_artifacts_for_checkpoint(checkpoint_meta)

    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    spatial, source_name = _case_spatial_features(
        cfg=cfg,
        bundle=bundle,
        spatial_artifacts=spatial_artifacts,
        dataset=dataset,
        case_index=case_index,
    )
    raw_prediction = model.forward_features(
        cond_scaled[case_index : case_index + 1],
        spatial_features=spatial,
    )
    physical_prediction = transforms.inverse_field_dict(
        {name: np.asarray(values, dtype=np.float32) for name, values in raw_prediction.items()}
    )
    prediction = _squeeze_2d(physical_prediction[TARGET])
    truth = _squeeze_2d(case["y"][TARGET])
    geometry = _load_case_geometry(case, fallback_root=Path(dataset.geometry_root))
    if truth.shape != geometry.mask.shape or prediction.shape != truth.shape:
        raise ValueError(
            f"field/geometry shape mismatch for {run.key}: truth={truth.shape}, "
            f"prediction={prediction.shape}, mask={geometry.mask.shape}"
        )
    conditions = {str(key): float(value) for key, value in dict(case.get("cond", {})).items()}
    return CasePrediction(
        case_id=case_id,
        truth=truth,
        prediction=prediction,
        geometry=geometry,
        conditions=conditions,
        spatial_feature_source=source_name,
    )


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _axis_edges(axis: np.ndarray) -> tuple[float, float]:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return float(values[0] - 0.5), float(values[0] + 0.5)
    left_step = float(values[1] - values[0])
    right_step = float(values[-1] - values[-2])
    return float(values[0] - 0.5 * left_step), float(values[-1] + 0.5 * right_step)


def _relative_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    t = np.asarray(truth, dtype=np.float64)[active]
    p = np.asarray(prediction, dtype=np.float64)[active]
    finite = np.isfinite(t) & np.isfinite(p)
    numerator = float(np.sqrt(np.mean(np.square(p[finite] - t[finite]))))
    denominator = float(np.sqrt(np.mean(np.square(t[finite]))))
    return numerator / denominator if denominator > 0.0 else float("nan")


def _r2(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    t = np.asarray(truth, dtype=np.float64)[active]
    p = np.asarray(prediction, dtype=np.float64)[active]
    finite = np.isfinite(t) & np.isfinite(p)
    t = t[finite]
    p = p[finite]
    denominator = float(np.sum(np.square(t - np.mean(t))))
    return 1.0 - float(np.sum(np.square(p - t))) / denominator if denominator > 0.0 else float("nan")


def _signed_percent_error(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    t = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    finite_truth = np.abs(t[active][np.isfinite(t[active])])
    peak = float(np.max(finite_truth)) if finite_truth.size else float("nan")
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full_like(t, np.nan, dtype=np.float64)
    return 100.0 * (p - t) / peak


def _shared_field_limits(predictions: Iterable[CasePrediction]) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for item in predictions:
        for field in (item.truth, item.prediction):
            values = np.asarray(field, dtype=np.float64)[item.geometry.mask]
            values = values[np.isfinite(values)]
            if values.size:
                finite_parts.append(values)
    if not finite_parts:
        return 0.0, 1.0
    finite = np.concatenate(finite_parts)
    low, high = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(float(low), float(high)):
        low, high = float(np.min(finite)), float(np.max(finite))
    if math.isclose(float(low), float(high)):
        high = float(low) + 1.0
    return float(low), float(high)


def _contour_args(mask: np.ndarray, geometry: CaseGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return geometry.r, geometry.z, np.asarray(mask, dtype=np.float32)


def _draw_background(ax: plt.Axes, geometry: CaseGeometry) -> None:
    background = np.empty((*geometry.mask.shape, 4), dtype=np.float32)
    background[:] = OUTSIDE_COLOR
    background[geometry.mask] = PLASMA_COLOR
    covered = np.zeros_like(geometry.mask, dtype=bool)
    for index, part_mask in enumerate(geometry.part_masks):
        current = np.asarray(part_mask, dtype=bool) & ~covered
        background[current] = COIL_COLORS[index % len(COIL_COLORS)]
        covered |= np.asarray(part_mask, dtype=bool)
    ax.imshow(background, origin="lower", extent=geometry.extent, interpolation="nearest", aspect="equal")


def _draw_contours(ax: plt.Axes, geometry: CaseGeometry) -> None:
    ax.contour(*_contour_args(geometry.mask, geometry), levels=[0.5], colors=PLASMA_EDGE, linewidths=0.55)
    for index, part_mask in enumerate(geometry.part_masks):
        ax.contour(
            *_contour_args(part_mask, geometry),
            levels=[0.5],
            colors="#222222" if index % 2 == 0 else "#7A4F00",
            linewidths=0.65,
        )


def _format_spatial_axis(ax: plt.Axes, *, show_y_label: bool) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.7, labelleft=show_y_label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(r"$r$ [cm]")
    ax.set_ylabel(r"$z$ [cm]" if show_y_label else "")


def _draw_field(
    ax: plt.Axes,
    values: np.ndarray,
    geometry: CaseGeometry,
    *,
    cmap: Any,
    vmin: float,
    vmax: float,
    show_y_label: bool,
) -> Any:
    _draw_background(ax, geometry)
    masked = np.ma.array(np.asarray(values, dtype=np.float64), mask=~geometry.mask)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=geometry.extent,
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _draw_contours(ax, geometry)
    _format_spatial_axis(ax, show_y_label=show_y_label)
    return image


def _format_density_colorbar(colorbar: Any) -> None:
    colorbar.set_label(TARGET_UNITS)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    colorbar.formatter = formatter
    colorbar.update_ticks()


def _legend_handles(geometry: CaseGeometry) -> list[Any]:
    return [
        mpatches.Patch(facecolor=PLASMA_COLOR, edgecolor=PLASMA_EDGE, label="plasma target"),
        mpatches.Patch(facecolor=OUTSIDE_COLOR, edgecolor="#999999", label="masked non-target"),
        mpatches.Patch(
            facecolor=COIL_COLORS[0],
            edgecolor="#222222",
            label=f"case coils ({len(geometry.part_masks)} active)",
        ),
    ]


def _plot_individual_spatial(
    *,
    run: RepresentationRun,
    data: CasePrediction,
    field_limits: tuple[float, float],
    error_limit_percent: float,
    out_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, float]]:
    field_cmap = plt.get_cmap("jet").copy()
    field_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    error_cmap = plt.get_cmap("coolwarm").copy()
    error_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    signed_error = _signed_percent_error(data.truth, data.prediction, data.geometry.mask)
    relative_rmse = _relative_rmse(data.truth, data.prediction, data.geometry.mask)
    r2 = _r2(data.truth, data.prediction, data.geometry.mask)

    fig, axes = plt.subplots(1, 4, figsize=(9.3, 2.9), constrained_layout=True, sharex=True, sharey=True)
    for ax, title in zip(axes, ("Geometry / Mask", "Truth", "Prediction", "Signed error"), strict=True):
        ax.set_title(title, pad=4)
    _draw_background(axes[0], data.geometry)
    _draw_contours(axes[0], data.geometry)
    _format_spatial_axis(axes[0], show_y_label=True)
    _draw_field(
        axes[1],
        data.truth,
        data.geometry,
        cmap=field_cmap,
        vmin=field_limits[0],
        vmax=field_limits[1],
        show_y_label=False,
    )
    prediction_image = _draw_field(
        axes[2],
        data.prediction,
        data.geometry,
        cmap=field_cmap,
        vmin=field_limits[0],
        vmax=field_limits[1],
        show_y_label=False,
    )
    error_image = _draw_field(
        axes[3],
        signed_error,
        data.geometry,
        cmap=error_cmap,
        vmin=-error_limit_percent,
        vmax=error_limit_percent,
        show_y_label=False,
    )
    density_colorbar = fig.colorbar(prediction_image, ax=axes[1:3], fraction=0.045, pad=0.025, shrink=0.78)
    _format_density_colorbar(density_colorbar)
    error_colorbar = fig.colorbar(error_image, ax=axes[3], fraction=0.045, pad=0.025, shrink=0.78)
    error_colorbar.set_label(r"$(\hat{n}_e-n_e)/\max(|n_e|)$ [%]")
    error_colorbar.set_ticks(
        [-error_limit_percent, -0.5 * error_limit_percent, 0.0, 0.5 * error_limit_percent, error_limit_percent]
    )
    fig.legend(
        handles=_legend_handles(data.geometry),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"{MODEL_DISPLAY_NAME} | structure holdout | {run.display_name} | {TARGET_LABEL} | rel. RMSE={relative_rmse:.4f}",
        fontsize=9,
        y=1.02,
    )
    outputs = _save_figure(fig, out_dir / f"{run.key}_ne_truth_prediction_error", dpi=dpi)
    plt.close(fig)

    active_error = signed_error[data.geometry.mask]
    finite_error = active_error[np.isfinite(active_error)]
    metrics = {
        "relative_rmse": relative_rmse,
        "r2": r2,
        "signed_error_percent_mean": float(np.mean(finite_error)),
        "signed_error_percent_abs_mean": float(np.mean(np.abs(finite_error))),
        "signed_error_percent_abs_p99": float(np.percentile(np.abs(finite_error), 99.0)),
        "signed_error_percent_abs_max": float(np.max(np.abs(finite_error))),
    }
    return outputs, metrics


def _plot_comparison(
    *,
    runs: tuple[RepresentationRun, ...],
    predictions: dict[str, CasePrediction],
    field_limits: tuple[float, float],
    error_limit_percent: float,
    metrics: dict[str, dict[str, float]],
    out_dir: Path,
    dpi: int,
    figure_title_prefix: str = "GEC-ICP electron density",
) -> list[Path]:
    field_cmap = plt.get_cmap("jet").copy()
    field_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    error_cmap = plt.get_cmap("coolwarm").copy()
    error_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    fig, axes = plt.subplots(len(runs), 3, figsize=(7.9, 5.0), constrained_layout=True, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(runs), 3)
    prediction_image = error_image = None
    for row_index, run in enumerate(runs):
        data = predictions[run.key]
        signed_error = _signed_percent_error(data.truth, data.prediction, data.geometry.mask)
        _draw_field(
            axes[row_index, 0],
            data.truth,
            data.geometry,
            cmap=field_cmap,
            vmin=field_limits[0],
            vmax=field_limits[1],
            show_y_label=True,
        )
        prediction_image = _draw_field(
            axes[row_index, 1],
            data.prediction,
            data.geometry,
            cmap=field_cmap,
            vmin=field_limits[0],
            vmax=field_limits[1],
            show_y_label=False,
        )
        error_image = _draw_field(
            axes[row_index, 2],
            signed_error,
            data.geometry,
            cmap=error_cmap,
            vmin=-error_limit_percent,
            vmax=error_limit_percent,
            show_y_label=False,
        )
        axes[row_index, 0].set_ylabel(
            f"{run.display_name}\nrel. RMSE={metrics[run.key]['relative_rmse']:.4f}\n$z$ [cm]"
        )
    for ax, title in zip(axes[0], ("Ground truth", "Prediction", "Signed error"), strict=True):
        ax.set_title(title, fontsize=10, pad=4)
    assert prediction_image is not None and error_image is not None
    density_colorbar = fig.colorbar(prediction_image, ax=axes[:, :2], fraction=0.035, pad=0.02, shrink=0.84)
    _format_density_colorbar(density_colorbar)
    error_colorbar = fig.colorbar(error_image, ax=axes[:, 2], fraction=0.045, pad=0.02, shrink=0.84)
    error_colorbar.set_label(r"$(\hat{n}_e-n_e)/\max(|n_e|)$ [%]")
    error_colorbar.set_ticks(
        [-error_limit_percent, -0.5 * error_limit_percent, 0.0, 0.5 * error_limit_percent, error_limit_percent]
    )
    case_id = predictions[runs[0].key].case_id
    fig.suptitle(
        f"{figure_title_prefix} | common structure-holdout {case_id}",
        fontsize=11,
        y=1.02,
    )
    outputs = _save_figure(fig, out_dir / "dimension_vs_sdf_ne_comparison", dpi=dpi)
    plt.close(fig)
    return outputs


def _write_learning_curves(
    path: Path,
    runs: tuple[RepresentationRun, ...],
    histories: dict[str, dict[str, np.ndarray]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("representation", "display_name", "epoch", "train_loss", "val_loss", "selected_epoch"),
        )
        writer.writeheader()
        for run in runs:
            selected = _selected_epoch(run)
            history = histories[run.key]
            for epoch, train_loss, val_loss in zip(
                history["epoch"], history["train_loss"], history["val_loss"], strict=True
            ):
                writer.writerow(
                    {
                        "representation": run.key,
                        "display_name": run.display_name,
                        "epoch": f"{epoch:.12g}",
                        "train_loss": f"{train_loss:.12g}",
                        "val_loss": f"{val_loss:.12g}",
                        "selected_epoch": selected,
                    }
                )


def _leaderboard_row(run: RepresentationRun) -> dict[str, str]:
    return next(row for row in _read_csv(run.run_root / "leaderboard.csv") if row["model_id"] == MODEL_ID)


def _write_spatial_metrics(
    path: Path,
    *,
    runs: tuple[RepresentationRun, ...],
    case_id: str,
    selection: dict[str, Any],
    metrics: dict[str, dict[str, float]],
) -> None:
    fields = (
        "representation",
        "display_name",
        "case_id",
        "relative_rmse",
        "r2",
        "audit_physical_rel_l2",
        "signed_error_percent_mean",
        "signed_error_percent_abs_mean",
        "signed_error_percent_abs_p99",
        "signed_error_percent_abs_max",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "representation": run.key,
                    "display_name": run.display_name,
                    "case_id": case_id,
                    "audit_physical_rel_l2": selection["per_representation_physical_rel_l2"][run.key],
                    **metrics[run.key],
                }
            )


def _adopted_figure_rows(*, base: str = "representation_model_details") -> list[dict[str, str | int]]:
    prefix = f"{base.rstrip('/')}/" if base and base != "." else ""
    model_slug = "uno" if MODEL_ID == "u_no" else MODEL_ID.replace("_", "-")
    return [
        {
            "order": 1,
            "status": "adopted_for_icp_conference",
            "figure_id": f"icp_{model_slug}_dimension_sdf_learning_curves",
            "claim": (
                f"Controlled {MODEL_DISPLAY_NAME} training histories for "
                "Dimension-vector and Union-SDF structure inputs"
            ),
            "canonical_png": f"{prefix}dimension_sdf_learning_curves.png",
            "canonical_pdf": f"{prefix}dimension_sdf_learning_curves.pdf",
            "canonical_svg": f"{prefix}dimension_sdf_learning_curves.svg",
            "data": f"{prefix}learning_curves.csv",
            "metadata": f"{prefix}metadata.json",
        },
        {
            "order": 2,
            "status": "adopted_for_icp_conference",
            "figure_id": f"icp_{model_slug}_dimension_vs_sdf_ne_comparison",
            "claim": (
                f"{MODEL_DISPLAY_NAME} electron-density truth prediction and signed-error "
                "comparison on one common held-out structure"
            ),
            "canonical_png": f"{prefix}dimension_vs_sdf_ne_comparison.png",
            "canonical_pdf": f"{prefix}dimension_vs_sdf_ne_comparison.pdf",
            "canonical_svg": f"{prefix}dimension_vs_sdf_ne_comparison.svg",
            "data": f"{prefix}spatial_metrics.csv",
            "metadata": f"{prefix}metadata.json",
        },
    ]


def _write_adopted_figures(path: Path, *, base: str) -> list[dict[str, str | int]]:
    rows = _adopted_figure_rows(base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_index(
    path: Path,
    *,
    runs: tuple[RepresentationRun, ...],
    case_id: str,
    selection: dict[str, Any],
    metrics: dict[str, dict[str, float]],
) -> None:
    rows = []
    for run in runs:
        leaderboard = _leaderboard_row(run)
        rows.append(
            "| {name} | {epoch} | {val:.6f} | {test_r2:.6f} | {case_rrmse:.4f} | {p99:.2f}% |".format(
                name=run.display_name,
                epoch=_selected_epoch(run),
                val=float(leaderboard["validation_selection_value"]),
                test_r2=float(leaderboard["test_r2_ne_plasma"]),
                case_rrmse=metrics[run.key]["relative_rmse"],
                p99=metrics[run.key]["signed_error_percent_abs_p99"],
            )
        )
    lines = [
        "# ICP学会本番採用：寸法入力 / SDF入力の学習履歴と電子密度空間分布",
        "",
        "GEC-CCP学会図と同じ構成で、寸法ベクトルとunion SDFを入力したGEC-ICP U-Netを比較します。",
        "両runはデータ、structure-holdout split、seed、モデル、損失、batch size、70 epochを共通化しており、構造表現だけが異なります。",
        "",
        "## 本番採用図（使用順）",
        "",
        "| 順序 | スライドで示す内容 | 本番PNG | 投稿PDF | 編集SVG |",
        "|---:|---|---|---|---|",
        "| 1 | 同一条件での寸法 / SDF学習収束 | [PNG](dimension_sdf_learning_curves.png) | [PDF](dimension_sdf_learning_curves.pdf) | [SVG](dimension_sdf_learning_curves.svg) |",
        "| 2 | 共通holdout構造に対する電子密度の真値・予測値・符号付き誤差 | [PNG](dimension_vs_sdf_ne_comparison.png) | [PDF](dimension_vs_sdf_ne_comparison.pdf) | [SVG](dimension_vs_sdf_ne_comparison.svg) |",
        "",
        "固定採用一覧は [`../adopted_figures.csv`](../adopted_figures.csv) です。後から参照する場合は、",
        "このCSVの `figure_id` と `canonical_*` を基準にしてください。",
        "",
        "## 採用図1：学習履歴",
        "",
        "[![寸法 / SDF 学習履歴](dimension_sdf_learning_curves.png)](dimension_sdf_learning_curves.png)",
        "",
        "- 寸法: [PNG](dimension_learning_curve.png) / [PDF](dimension_learning_curve.pdf) / [SVG](dimension_learning_curve.svg)",
        "- SDF: [PNG](sdf_learning_curve.png) / [PDF](sdf_learning_curve.pdf) / [SVG](sdf_learning_curve.svg)",
        "- 横並び: [PNG](dimension_sdf_learning_curves.png) / [PDF](dimension_sdf_learning_curves.pdf) / [SVG](dimension_sdf_learning_curves.svg)",
        "",
        "## 採用図2：電子密度の真値 / 予測値 / 符号付き誤差",
        "",
        f"共通の未学習コイル構造 `{case_id}` を表示しています。選択規則は「{selection['method']}」です（共通test {selection['common_case_count']}件）。",
        "",
        "[![寸法とSDFの電子密度比較](dimension_vs_sdf_ne_comparison.png)](dimension_vs_sdf_ne_comparison.png)",
        "",
        "- 寸法: [PNG](dimension_ne_truth_prediction_error.png) / [PDF](dimension_ne_truth_prediction_error.pdf) / [SVG](dimension_ne_truth_prediction_error.svg)",
        "- SDF: [PNG](sdf_ne_truth_prediction_error.png) / [PDF](sdf_ne_truth_prediction_error.pdf) / [SVG](sdf_ne_truth_prediction_error.svg)",
        "- 直接比較: [PNG](dimension_vs_sdf_ne_comparison.png) / [PDF](dimension_vs_sdf_ne_comparison.pdf) / [SVG](dimension_vs_sdf_ne_comparison.svg)",
        "",
        "TruthとPredictionは全図で同じ2–98 percentile色域です。Signed errorはplasma内で",
        r"`100 × (prediction − truth) / max(|truth|)` とし、GEC-CCP図と同じ ±30% 色域を用いています。",
        "",
        "| 構造表現 | 採用epoch | validation loss | test plasma $n_e$ $R^2$ | 表示case rel.RMSE | $|error|$ p99 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 発表時の解釈",
        "",
        "- 表示caseは両表現の平均relative RMSEが中央値となる共通testケースで、best caseではありません。",
        "- この統制実験では寸法入力のtest plasma電子密度 $R^2$ がunion SDFより高い結果です。",
        "- これはこのデータ・モデル・union-SDF定義での比較であり、SDF一般の優劣を示す主張には用いません。",
        "- 構造最適化結果とは分けて、サロゲートの学習収束と空間再現性の根拠として使用します。",
        "",
        "## 補助図",
        "",
        "個別の寸法 / SDF学習曲線とGeometry / Truth / Prediction / Signed error図は、",
        "質疑・補足スライド用として上記リンクから参照できます。",
        "",
        "## 数値と再生成",
        "",
        "- [学習履歴CSV](learning_curves.csv)",
        "- [空間誤差CSV](spatial_metrics.csv)",
        "- [生成metadata](metadata.json)",
        "- [ICP本番採用一覧](../adopted_figures.csv)",
        "",
        "```powershell",
        "$env:PLASMA_SURROGATE_ENABLE_TORCH='1'",
        ".venv-torch\\Scripts\\python.exe experiments\\conference\\scripts\\plot_gec_icp_representation_model_details.py",
        "```",
        "",
        "生成元: [plot_gec_icp_representation_model_details.py](../../../experiments/conference/scripts/plot_gec_icp_representation_model_details.py)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-display-name", default=MODEL_DISPLAY_NAME)
    parser.add_argument("--dimension-run", type=Path, default=DEFAULT_RUNS[0].run_root)
    parser.add_argument("--sdf-run", type=Path, default=DEFAULT_RUNS[1].run_root)
    parser.add_argument("--adopted-figures-csv", type=Path, default=DEFAULT_ADOPTED_FIGURES_CSV)
    parser.add_argument(
        "--case-id",
        default="",
        help="Common structure-holdout test case. Empty selects the median pooled electron-density case.",
    )
    parser.add_argument("--error-limit-percent", type=float, default=DEFAULT_ERROR_LIMIT_PERCENT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    global MODEL_ID, MODEL_DISPLAY_NAME
    args = _parse_args()
    MODEL_ID = str(args.model_id).strip()
    MODEL_DISPLAY_NAME = str(args.model_display_name).strip()
    if not MODEL_ID or not MODEL_DISPLAY_NAME:
        raise ValueError("--model-id and --model-display-name must be non-empty")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = (
        RepresentationRun(
            key="dimension",
            display_name="Dimension vector",
            run_root=Path(args.dimension_run).resolve(),
            color="#0072B2",
            structure_input="llcoil, rrc, nncoil, rrce, zzc (plus pp, pp0)",
        ),
        RepresentationRun(
            key="sdf",
            display_name="Union SDF",
            run_root=Path(args.sdf_run).resolve(),
            color="#D55E00",
            structure_input="case-varying part_sdf_union (plus pp, pp0)",
        ),
    )
    histories = {run.key: _history(run) for run in runs}
    figure_outputs = _plot_learning_curves(
        runs,
        histories,
        out_dir,
        dpi=int(args.dpi),
        figure_title=f"GEC-ICP learning curves (controlled {MODEL_DISPLAY_NAME} comparison)",
    )
    _write_learning_curves(out_dir / "learning_curves.csv", runs, histories)

    auto_case_id, auto_selection = _select_common_case(runs)
    case_id = str(args.case_id).strip() or auto_case_id
    if case_id == auto_case_id:
        selection = auto_selection
    else:
        audit = {run.key: _case_physical_rel_l2(run) for run in runs}
        missing = [run.key for run in runs if case_id not in audit[run.key]]
        if missing:
            raise ValueError(f"case {case_id!r} is not common to representations: {missing}")
        selection = {
            "method": "explicit --case-id",
            "common_case_count": len(set.intersection(*(set(values) for values in audit.values()))),
            "rank_zero_based": None,
            "mean_physical_rel_l2": float(np.mean([audit[run.key][case_id] for run in runs])),
            "per_representation_physical_rel_l2": {run.key: audit[run.key][case_id] for run in runs},
        }

    predictions: dict[str, CasePrediction] = {}
    for run in runs:
        print(f"inferring {run.display_name}: {case_id}", flush=True)
        predictions[run.key] = _predict_case(run, case_id)
        _release_torch_cache()
    reference_truth = predictions[runs[0].key].truth
    for run in runs[1:]:
        if not np.allclose(reference_truth, predictions[run.key].truth, rtol=0.0, atol=0.0, equal_nan=True):
            raise ValueError(f"truth fields differ between controlled runs for {case_id}: {run.key}")

    field_limits = _shared_field_limits(predictions.values())
    error_limit_percent = float(args.error_limit_percent)
    if not np.isfinite(error_limit_percent) or error_limit_percent <= 0.0:
        raise ValueError("--error-limit-percent must be positive and finite")
    spatial_metrics: dict[str, dict[str, float]] = {}
    for run in runs:
        outputs, current_metrics = _plot_individual_spatial(
            run=run,
            data=predictions[run.key],
            field_limits=field_limits,
            error_limit_percent=error_limit_percent,
            out_dir=out_dir,
            dpi=int(args.dpi),
        )
        figure_outputs.extend(outputs)
        spatial_metrics[run.key] = current_metrics
        audit_value = float(selection["per_representation_physical_rel_l2"][run.key])
        if not np.isclose(current_metrics["relative_rmse"], audit_value, rtol=1.0e-3, atol=1.0e-7):
            raise ValueError(
                f"re-inferred relative RMSE does not match evaluation audit for {run.key}: "
                f"inference={current_metrics['relative_rmse']}, audit={audit_value}"
            )
    figure_outputs.extend(
        _plot_comparison(
            runs=runs,
            predictions=predictions,
            field_limits=field_limits,
            error_limit_percent=error_limit_percent,
            metrics=spatial_metrics,
            out_dir=out_dir,
            dpi=int(args.dpi),
            figure_title_prefix=f"GEC-ICP electron density ({MODEL_DISPLAY_NAME})",
        )
    )

    _write_spatial_metrics(
        out_dir / "spatial_metrics.csv",
        runs=runs,
        case_id=case_id,
        selection=selection,
        metrics=spatial_metrics,
    )
    _write_index(
        out_dir / "index.md",
        runs=runs,
        case_id=case_id,
        selection=selection,
        metrics=spatial_metrics,
    )
    adopted_figures_csv = Path(args.adopted_figures_csv).resolve()
    catalog_base = Path(os.path.relpath(out_dir, adopted_figures_csv.parent)).as_posix()
    adoption_rows = _write_adopted_figures(adopted_figures_csv, base=catalog_base)
    split_payload = _read_json(runs[0].run_root / "preprocessing" / "split" / f"split_{PROTOCOL}_v1.json")
    metadata = {
        "purpose": "GEC-ICP conference comparison matching the adopted GEC-CCP learning/spatial layout",
        "generator_script": "experiments/conference/scripts/plot_gec_icp_representation_model_details.py",
        "conference_adoption": {
            "status": "adopted_for_icp_conference",
            "catalog": adopted_figures_csv.relative_to(REPO_ROOT).as_posix(),
            "primary_figure_ids": [str(row["figure_id"]) for row in adoption_rows],
            "presentation_order": [int(row["order"]) for row in adoption_rows],
        },
        "model_id": MODEL_ID,
        "protocol": PROTOCOL,
        "controlled_comparison": (
            f"same dataset, split, seed, {MODEL_DISPLAY_NAME}, loss, batch size, and epoch budget"
        ),
        "dataset_case_count": sum(len(split_payload.get(name, [])) for name in ("train", "val", "test")),
        "split_counts": {name: len(split_payload.get(name, [])) for name in ("train", "val", "test")},
        "case_id": case_id,
        "case_selection": selection,
        "case_conditions": predictions[runs[0].key].conditions,
        "density_field": {"id": TARGET, "units": "m^-3"},
        "field_color_scale": {
            "shared_between_truth_prediction_and_representations": True,
            "percentiles": [2.0, 98.0],
            "limits": list(field_limits),
            "colormap": "jet",
        },
        "signed_error": {
            "definition": "100 * (prediction - truth) / max(abs(truth[plasma]))",
            "limits_percent": [-error_limit_percent, error_limit_percent],
            "colormap": "coolwarm",
        },
        "representations": [
            {
                "key": run.key,
                "display_name": run.display_name,
                "structure_input": run.structure_input,
                "run_root": run.run_root.relative_to(REPO_ROOT).as_posix(),
                "history_csv": run.metrics_path.relative_to(REPO_ROOT).as_posix(),
                "checkpoint": run.checkpoint_dir.relative_to(REPO_ROOT).as_posix(),
                "selected_epoch": _selected_epoch(run),
                "spatial_feature_source": predictions[run.key].spatial_feature_source,
                "metrics": spatial_metrics[run.key],
            }
            for run in runs
        ],
        "outputs": [path.relative_to(REPO_ROOT).as_posix() for path in figure_outputs],
        "derived_data": [
            (out_dir / "learning_curves.csv").relative_to(REPO_ROOT).as_posix(),
            (out_dir / "spatial_metrics.csv").relative_to(REPO_ROOT).as_posix(),
            adopted_figures_csv.relative_to(REPO_ROOT).as_posix(),
        ],
        "reference_scripts": [
            "experiments/conference/scripts/plot_gec_ccp_selected_model_details.py",
            "scripts/plot_gec_ccp_all_best_publication_fields.py",
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
