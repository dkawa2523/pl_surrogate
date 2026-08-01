#!/usr/bin/env python3
"""Plot n54 electron-density truth, prediction, and error for three GEC-CCP models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plasma_surrogate.core.dataset_io import load_dataset
from plot_gec_ccp_all_best_publication_fields import (
    _draw_background,
    _draw_contours,
    _image_kwargs,
    _load_masks,
    _make_engine,
    _masked,
    _output_vars,
    _percent_error,
    _predict_case,
    _read_best_runs,
    _read_case_ids,
    _read_yaml,
    _rel_rmse,
)


CASE_ID = "case_td003_pp0_3_gamma_004__steady"
FIELD = "ne"
MODELS = (
    ("ffno", "FFNO"),
    ("deeponet_pod", "POD-DeepONet"),
    ("global_densemlp", "DenseMLP"),
)
DEFAULT_MANIFEST = Path(
    "runs/gec_ccp_training_size_ablation_v1/summary_all_models/selection_manifest_n54.csv"
)
DEFAULT_OUT = Path(
    "reports/gec_conference_materials/training_size_ablation/"
    "ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison"
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=CASE_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_predictions(
    manifest: Path,
    case_id: str,
) -> tuple[dict[str, Any], Any, np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    selected_runs = _read_best_runs(manifest)
    first_run = Path(selected_runs[MODELS[0][0]]["run_root"])
    dataset = load_dataset(_read_yaml(first_run / "resolved_config.yaml"), run_dir=Path("."))
    cases = {str(case["case_id"]): case for case in dataset.cases}
    if case_id not in cases:
        raise KeyError(f"case {case_id!r} is not present in the dataset")

    case = cases[case_id]
    masks = _load_masks(Path(dataset.geometry_root), case=case)
    truth = np.asarray(case["y"][FIELD], dtype=np.float64)
    predictions: dict[str, np.ndarray] = {}
    relative_rmse: dict[str, float] = {}
    run_roots: dict[str, str] = {}

    for model, _display_name in MODELS:
        spec = selected_runs[model]
        run_root = Path(spec["run_root"])
        split_file = run_root / "preprocessing" / "split" / "split_interp_marginal_v1.json"
        if case_id not in _read_case_ids(split_file):
            raise ValueError(f"case {case_id!r} is not in the fixed interpolation test split: {run_root}")
        cfg = _read_yaml(run_root / "resolved_config.yaml")
        output_vars = _output_vars(run_root)
        if FIELD not in output_vars:
            raise ValueError(f"{FIELD!r} is not an output of {run_root}")
        engine = _make_engine(run_root, "interp", model, cfg, Path(dataset.geometry_root))
        pred = np.asarray(_predict_case(engine, case, output_vars)[FIELD], dtype=np.float64)
        if pred.shape != truth.shape:
            raise ValueError(f"shape mismatch for {model}: {pred.shape} != {truth.shape}")
        predictions[model] = pred
        relative_rmse[model] = _rel_rmse(truth[masks.target], pred[masks.target])
        run_roots[model] = run_root.as_posix()

    return case, masks, truth, predictions, relative_rmse, run_roots


def _format_axis(ax: Any, *, show_y: bool, show_x: bool) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(r"$r$ [mm]" if show_x else "")
    ax.set_ylabel(r"$z$ [mm]" if show_y else "")
    ax.tick_params(labelleft=show_y, labelbottom=show_x)


def _plot(
    *,
    case_conditions: dict[str, float],
    masks: Any,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    relative_rmse: dict[str, float],
    out: Path,
) -> tuple[float, float]:
    density_scale = 1.0e15
    density_fields = [truth, *predictions.values()]
    density_max = max(
        float(np.nanmax(np.asarray(field, dtype=np.float64)[masks.target]))
        for field in density_fields
    )
    density_vmax = density_max / density_scale
    if not np.isfinite(density_vmax) or density_vmax <= 0.0:
        raise ValueError("electron-density fields do not contain a positive finite value")

    error_limit = 10.0
    spatial_errors = {
        model: _percent_error(truth, pred, masks.target)
        for model, pred in predictions.items()
    }

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.8, 8.26),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.13},
    )
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.08, top=0.86)
    column_titles = ("Ground truth", "Prediction", "Signed error")
    density_image = None
    error_image = None

    for row, (model, display_name) in enumerate(MODELS):
        row_fields = (truth, predictions[model], spatial_errors[model])
        for col, field in enumerate(row_fields):
            ax = axes[row, col]
            _draw_background(ax, masks)
            if col < 2:
                density_image = ax.imshow(
                    _masked(field / density_scale, masks),
                    **_image_kwargs(masks),
                    cmap="turbo",
                    vmin=0.0,
                    vmax=density_vmax,
                    interpolation="bilinear",
                )
            else:
                error_image = ax.imshow(
                    _masked(field, masks),
                    **_image_kwargs(masks),
                    cmap="coolwarm",
                    vmin=-error_limit,
                    vmax=error_limit,
                    interpolation="nearest",
                )
                ax.text(
                    0.97,
                    0.04,
                    f"rel. RMSE {100.0 * relative_rmse[model]:.2f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10.5,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#555555", "pad": 2.5},
                )
            _draw_contours(ax, masks)
            _format_axis(ax, show_y=(col == 0), show_x=(row == len(MODELS) - 1))
            if row == 0:
                ax.set_title(column_titles[col], pad=8, fontweight="semibold")

        row_box = axes[row, 0].get_position()
        fig.text(
            0.025,
            0.5 * (row_box.y0 + row_box.y1),
            display_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
        )

    if density_image is None or error_image is None:
        raise RuntimeError("no panels were plotted")

    density_colorbar_ax = fig.add_axes([0.905, 0.55, 0.021, 0.27])
    density_cbar = fig.colorbar(density_image, cax=density_colorbar_ax)
    density_cbar.set_label(r"$n_e$ [$10^{15}$ m$^{-3}$]", labelpad=10)
    density_cbar.ax.tick_params(labelsize=11)

    error_colorbar_ax = fig.add_axes([0.905, 0.15, 0.021, 0.27])
    error_cbar = fig.colorbar(error_image, cax=error_colorbar_ax, extend="max")
    error_cbar.set_label("Signed error [% of truth peak]", labelpad=10)
    error_cbar.set_ticks([-10, -5, 0, 5, 10])
    error_cbar.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Electron-density prediction at 54 training conditions",
        x=0.505,
        y=0.965,
        fontsize=21,
        fontweight="semibold",
    )
    fig.text(
        0.505,
        0.91,
        (
            "Common medium test case: "
            f"PP0={case_conditions['PP0']:g}, "
            f"Td={case_conditions['Td']:g}, "
            rf"$\gamma$={case_conditions['gamma']:g}, "
            f"PA={case_conditions['PA']:g}"
        ),
        ha="center",
        va="center",
        fontsize=12,
        color="#333333",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix))
    plt.close(fig)
    return density_max, error_limit


def main() -> None:
    args = _parse_args()
    case, masks, truth, predictions, relative_rmse, run_roots = _load_predictions(
        args.manifest,
        args.case_id,
    )
    case_conditions = {str(k): float(v) for k, v in dict(case["cond"]).items()}
    density_max, error_limit = _plot(
        case_conditions=case_conditions,
        masks=masks,
        truth=truth,
        predictions=predictions,
        relative_rmse=relative_rmse,
        out=args.out,
    )
    metadata = {
        "case_id": args.case_id,
        "case_conditions": case_conditions,
        "split": "interp",
        "case_label": "medium",
        "training_condition_count": 54,
        "field": FIELD,
        "units": "m^-3",
        "common_density_color_scale_m3": [0.0, density_max],
        "error_definition": "100 * (prediction - truth) / max(abs(truth)) within plasma target",
        "common_error_color_scale_percent": [-error_limit, error_limit],
        "panel_layout": ["ground_truth", "prediction", "signed_error"],
        "figure_height_to_width_ratio": 0.7,
        "model_order": [model for model, _display_name in MODELS],
        "relative_rmse": {model: float(relative_rmse[model]) for model, _ in MODELS},
        "run_roots": run_roots,
        "selection_manifest": args.manifest.as_posix(),
        "outputs": [args.out.with_suffix(suffix).as_posix() for suffix in (".png", ".pdf", ".svg")],
        "generation_command": (
            ".\\.venv-torch\\Scripts\\python.exe "
            "scripts/plot_gec_ccp_n54_model_density_comparison.py"
        ),
    }
    args.out.with_name(args.out.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out.with_suffix(".png"))
    print(json.dumps(metadata["relative_rmse"], indent=2))


if __name__ == "__main__":
    main()
