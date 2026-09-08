#!/usr/bin/env python
"""Create conference spatial truth/prediction/error figures for Formal models.

The script reads the frozen seed-1237, 200-epoch representative prediction
pack.  It does not train either UNO model or rerun COMSOL.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "reports/icp_conference_materials/icp_uno_epoch500_four_model_v47"
    / "formal_200epoch/unknown_structure_eval"
)
OUT = (
    ROOT
    / "reports/icp_conference_materials/icp_uno_e1_dimension_conference_v49"
    / "formal_dimension_sdf_spatial_v52"
)
PACK_PATH = SOURCE / "representative_predictions.npz"
METRICS_PATH = SOURCE / "case_metrics_unknown75.csv"
ACCURACY_PATH = SOURCE.parent / "accuracy_evaluation/accuracy_evaluation.json"

MODELS = ("formal_dimension", "formal_sdf")
MODEL_LABELS = {
    "formal_dimension": "Formal Dimension",
    "formal_sdf": "Formal SDF",
}
CASES = (
    "v43_te_n4_variant_1__center",
    "v43_te_n4_variant_2__center",
    "v43_te_n4_variant_3__center",
)
CASE_LABELS = {
    CASES[0]: "A  Unequal spacing",
    CASES[1]: "B  Height + size",
    CASES[2]: "C  Spacing + height + size",
}
INK = "#172033"
MID = "#667085"
OUTLINE = "#344054"
WHITE = "#FFFFFF"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Yu Gothic", "Meiryo", "Arial", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.85,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _key(model: str, case_id: str, field: str) -> str:
    return f"{model}__{case_id}__{field}"


def _field(pack: Any, model: str, case_id: str, field: str) -> np.ndarray:
    return np.asarray(pack[_key(model, case_id, field)], dtype=np.float64)


def _read_metrics() -> dict[tuple[str, str], float]:
    with METRICS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["model"], row["case_id"]): float(row["ni_rel_l2_pct"])
        for row in rows
        if row["model"] in MODELS and row["case_id"] in CASES
    }


def _masked(pack: Any, model: str, case_id: str, field: str) -> np.ndarray:
    values = _field(pack, model, case_id, field)
    mask = _field(pack, model, case_id, "mask").astype(bool)
    return np.where(mask, values, np.nan)


def _relative_l2(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    numerator = float(np.linalg.norm(prediction[valid] - truth[valid]))
    denominator = float(np.linalg.norm(truth[valid]))
    return 100.0 * numerator / max(denominator, np.finfo(np.float64).tiny)


def _draw_field(
    axis: plt.Axes,
    pack: Any,
    model: str,
    case_id: str,
    field: np.ndarray,
    *,
    cmap: str,
    norm: Normalize,
) -> None:
    r = _field(pack, model, case_id, "r_coords")
    z = _field(pack, model, case_id, "z_coords")
    mask = _field(pack, model, case_id, "mask").astype(bool)
    displayed = np.where(mask, field, np.nan)
    axis.pcolormesh(r, z, displayed, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    axis.contour(r, z, mask.astype(float), levels=[0.5], colors=OUTLINE, linewidths=0.65)
    axis.set_xlim(float(r.min()), float(r.max()))
    axis.set_ylim(float(z.min()), float(z.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=8.5)


def _common_scales(pack: Any) -> tuple[float, float]:
    physical: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for model in MODELS:
        for case_id in CASES:
            truth = _masked(pack, model, case_id, "ni_truth") / 1.0e17
            prediction = _masked(pack, model, case_id, "ni_prediction") / 1.0e17
            physical.extend((truth[np.isfinite(truth)], prediction[np.isfinite(prediction)]))
            error = prediction - truth
            errors.append(np.abs(error[np.isfinite(error)]))
    physical_limit = float(np.quantile(np.concatenate(physical), 0.995))
    error_limit = float(np.quantile(np.concatenate(errors), 0.995))
    if not physical_limit > 0.0 or not error_limit > 0.0:
        raise ValueError("invalid shared display limits")
    return physical_limit, error_limit


def _add_colorbars(
    fig: plt.Figure,
    physical_norm: Normalize,
    error_norm: TwoSlopeNorm,
    *,
    physical_box: tuple[float, float, float, float],
    error_box: tuple[float, float, float, float],
) -> None:
    physical_axis = fig.add_axes(physical_box)
    error_axis = fig.add_axes(error_box)
    physical = fig.colorbar(
        ScalarMappable(norm=physical_norm, cmap="viridis"),
        cax=physical_axis,
        orientation="horizontal",
    )
    physical.set_label(r"Ion density $n_i$  [$10^{17}$ m$^{-3}$]", fontsize=9.5)
    error = fig.colorbar(
        ScalarMappable(norm=error_norm, cmap="RdBu_r"),
        cax=error_axis,
        orientation="horizontal",
    )
    error.set_label(r"Signed error: prediction - COMSOL  [$10^{17}$ m$^{-3}$]", fontsize=9.5)
    physical.ax.tick_params(labelsize=8)
    error.ax.tick_params(labelsize=8)


def _save(fig: plt.Figure, stem: str) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, kwargs in (("png", {"dpi": 220}), ("svg", {}), ("pdf", {})):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return outputs


def _model_figure(
    pack: Any,
    model: str,
    metrics: dict[tuple[str, str], float],
    physical_norm: Normalize,
    error_norm: TwoSlopeNorm,
) -> list[str]:
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 9.2))
    fig.subplots_adjust(left=0.12, right=0.975, top=0.84, bottom=0.17, wspace=0.16, hspace=0.28)
    fig.text(
        0.04,
        0.975,
        f"{MODEL_LABELS[model]}: COMSOL truth, prediction, and spatial error",
        ha="left",
        va="top",
        fontsize=18.5,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.937,
        "Frozen seed-1237 model, 200 training epochs. Common scales; display clipped at the joint 99.5th percentile.",
        ha="left",
        va="top",
        fontsize=10,
        color=MID,
    )
    for column, title in enumerate(("COMSOL truth", "Prediction", "Signed error")):
        axes[0, column].set_title(title, fontsize=13, fontweight="bold", pad=9)
    for row, case_id in enumerate(CASES):
        truth = _field(pack, model, case_id, "ni_truth") / 1.0e17
        prediction = _field(pack, model, case_id, "ni_prediction") / 1.0e17
        error = prediction - truth
        for column, (values, cmap, norm) in enumerate(
            ((truth, "viridis", physical_norm), (prediction, "viridis", physical_norm), (error, "RdBu_r", error_norm))
        ):
            _draw_field(axes[row, column], pack, model, case_id, values, cmap=cmap, norm=norm)
            if row < len(CASES) - 1:
                axes[row, column].tick_params(labelbottom=False)
            else:
                axes[row, column].set_xlabel("radius r [cm]")
            if column > 0:
                axes[row, column].tick_params(labelleft=False)
            else:
                axes[row, column].set_ylabel("plasma height z [cm]")
        axes[row, 0].text(
            0.03,
            0.94,
            CASE_LABELS[case_id],
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
        )
        axes[row, 2].text(
            0.97,
            0.94,
            f"plasma relative L2 = {metrics[(model, case_id)]:.1f}%",
            transform=axes[row, 2].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
        )
    _add_colorbars(
        fig,
        physical_norm,
        error_norm,
        physical_box=(0.18, 0.070, 0.28, 0.020),
        error_box=(0.57, 0.070, 0.28, 0.020),
    )
    return _save(fig, f"20_{model}_truth_prediction_error")


def _comparison_figure(
    pack: Any,
    metrics: dict[tuple[str, str], float],
    physical_norm: Normalize,
    error_norm: TwoSlopeNorm,
) -> list[str]:
    fig, axes = plt.subplots(3, 5, figsize=(18.4, 8.8))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.17, wspace=0.12, hspace=0.30)
    fig.text(
        0.03,
        0.975,
        "Unknown coil structures: where do Formal Dimension and Formal SDF differ from COMSOL?",
        ha="left",
        va="top",
        fontsize=18.5,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.935,
        "Physical ion density and signed error on common scales. Frozen seed-1237, 200-epoch models; display clipped at joint 99.5th percentiles.",
        ha="left",
        va="top",
        fontsize=10,
        color=MID,
    )
    headers = (
        "COMSOL truth",
        "Dimension prediction",
        "Dimension error",
        "SDF prediction",
        "SDF error",
    )
    for column, title in enumerate(headers):
        axes[0, column].set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    for row, case_id in enumerate(CASES):
        truth = _field(pack, MODELS[0], case_id, "ni_truth") / 1.0e17
        dimension = _field(pack, "formal_dimension", case_id, "ni_prediction") / 1.0e17
        sdf = _field(pack, "formal_sdf", case_id, "ni_prediction") / 1.0e17
        panels = (
            (MODELS[0], truth, "viridis", physical_norm),
            ("formal_dimension", dimension, "viridis", physical_norm),
            ("formal_dimension", dimension - truth, "RdBu_r", error_norm),
            ("formal_sdf", sdf, "viridis", physical_norm),
            ("formal_sdf", sdf - truth, "RdBu_r", error_norm),
        )
        for column, (source_model, values, cmap, norm) in enumerate(panels):
            _draw_field(axes[row, column], pack, source_model, case_id, values, cmap=cmap, norm=norm)
            if row < len(CASES) - 1:
                axes[row, column].tick_params(labelbottom=False)
            else:
                axes[row, column].set_xlabel("radius r [cm]")
            if column > 0:
                axes[row, column].tick_params(labelleft=False)
            else:
                axes[row, column].set_ylabel("plasma height z [cm]")
        axes[row, 0].text(
            0.03,
            0.94,
            CASE_LABELS[case_id],
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.86, "pad": 1.6},
        )
        for column, model in ((2, "formal_dimension"), (4, "formal_sdf")):
            axes[row, column].text(
                0.97,
                0.94,
                f"L2 = {metrics[(model, case_id)]:.1f}%",
                transform=axes[row, column].transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.84, "pad": 1.8},
            )
    _add_colorbars(
        fig,
        physical_norm,
        error_norm,
        physical_box=(0.17, 0.065, 0.28, 0.020),
        error_box=(0.57, 0.065, 0.28, 0.020),
    )
    return _save(fig, "22_formal_dimension_sdf_truth_prediction_error")


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = _read_metrics()
    accuracy = json.loads(ACCURACY_PATH.read_text(encoding="utf-8"))
    with np.load(PACK_PATH, allow_pickle=False) as pack:
        physical_limit, error_limit = _common_scales(pack)
        physical_norm = Normalize(vmin=0.0, vmax=physical_limit, clip=True)
        error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
        metric_rows: list[dict[str, Any]] = []
        truth_match: dict[str, bool] = {}
        grid_match: dict[str, bool] = {}
        for case_id in CASES:
            dim_truth = _field(pack, "formal_dimension", case_id, "ni_truth")
            sdf_truth = _field(pack, "formal_sdf", case_id, "ni_truth")
            dim_mask = _field(pack, "formal_dimension", case_id, "mask").astype(bool)
            sdf_mask = _field(pack, "formal_sdf", case_id, "mask").astype(bool)
            truth_match[case_id] = bool(np.allclose(dim_truth, sdf_truth, rtol=0.0, atol=0.0, equal_nan=True))
            grid_match[case_id] = bool(
                np.array_equal(dim_mask, sdf_mask)
                and np.array_equal(
                    _field(pack, "formal_dimension", case_id, "r_coords"),
                    _field(pack, "formal_sdf", case_id, "r_coords"),
                )
                and np.array_equal(
                    _field(pack, "formal_dimension", case_id, "z_coords"),
                    _field(pack, "formal_sdf", case_id, "z_coords"),
                )
            )
            for model in MODELS:
                truth = _field(pack, model, case_id, "ni_truth")
                prediction = _field(pack, model, case_id, "ni_prediction")
                mask = _field(pack, model, case_id, "mask").astype(bool)
                recomputed = _relative_l2(truth, prediction, mask)
                saved = metrics[(model, case_id)]
                metric_rows.append(
                    {
                        "model": model,
                        "case_id": case_id,
                        "saved_ni_relative_l2_pct": saved,
                        "recomputed_ni_relative_l2_pct": recomputed,
                        "absolute_delta_percentage_points": abs(recomputed - saved),
                    }
                )
        outputs = {
            "formal_dimension": _model_figure(pack, "formal_dimension", metrics, physical_norm, error_norm),
            "formal_sdf": _model_figure(pack, "formal_sdf", metrics, physical_norm, error_norm),
            "direct_comparison": _comparison_figure(pack, metrics, physical_norm, error_norm),
        }
    metrics_out = OUT / "spatial_case_metrics_v52.csv"
    with metrics_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    max_metric_delta = max(float(row["absolute_delta_percentage_points"]) for row in metric_rows)
    summary = {
        "schema": "icp-formal-dimension-sdf-spatial-truth-prediction-error-v52",
        "models": {
            name: {
                "training_epochs": 200,
                "selected_epoch": int(accuracy["leaderboard"][name]["selected_epoch"]),
            }
            for name in MODELS
        },
        "source_prediction_pack": PACK_PATH.relative_to(ROOT).as_posix(),
        "source_case_metrics": METRICS_PATH.relative_to(ROOT).as_posix(),
        "cases": list(CASES),
        "grid_shape": [440, 600],
        "field": "physical ion density ni",
        "physical_units": "1e17 m^-3",
        "error_definition": "prediction - COMSOL",
        "physical_display_limit": physical_limit,
        "signed_error_display_limit": error_limit,
        "display_quantile": 0.995,
        "truth_exact_match_between_model_records": truth_match,
        "mask_and_grid_exact_match_between_model_records": grid_match,
        "max_recomputed_metric_delta_percentage_points": max_metric_delta,
        "representative_case_scope": "three displayed examples; population claims use all 75 unknown-structure cases",
        "outputs": outputs,
        "metrics_csv": metrics_out.relative_to(ROOT).as_posix(),
        "validation_passed": bool(
            all(truth_match.values())
            and all(grid_match.values())
            and max_metric_delta < 1.0e-4
            and physical_limit > 0.0
            and error_limit > 0.0
        ),
    }
    summary_path = OUT / "spatial_figure_validation_v52.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
