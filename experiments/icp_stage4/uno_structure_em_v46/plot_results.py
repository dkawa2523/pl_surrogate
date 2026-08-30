#!/usr/bin/env python
"""Create direct comparison figures for the isolated v46 UNO candidates."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V45_DIR = HERE.parent / "conference_continuity_v45"
if str(V45_DIR) not in sys.path:
    sys.path.insert(0, str(V45_DIR))
import plot_unknown_structure_results as old_plot  # noqa: E402


REPORT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46"
DATASET = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
CASES = old_plot.CASES
CASE_TITLES = old_plot.CASE_TITLES
FAMILIES = old_plot.FAMILIES
FAMILY_LABELS = old_plot.FAMILY_LABELS
CANDIDATES = (
    "A_multires",
    "AB_separate_fusion",
    "ABC_adaptive_mix",
    "ABD_shared_coils",
    "ABE_em_shape_amplitude",
    "ABF_structure_delta",
    "ABG_sdf_em_aux",
    "ABH_target_decoders",
    "ALL_combined",
)
MODELS = ("formal_dimension", "formal_sdf", *CANDIDATES)
LABELS = {
    "formal_dimension": "Formal Dimension",
    "formal_sdf": "Formal SDF",
    "A_multires": "A  multires",
    "AB_separate_fusion": "AB  separate fusion",
    "ABC_adaptive_mix": "ABC  adaptive mix",
    "ABD_shared_coils": "ABD  shared coils",
    "ABE_em_shape_amplitude": "ABE  EM shape/amplitude",
    "ABF_structure_delta": "ABF  structural delta",
    "ABG_sdf_em_aux": "ABG  SDF→EM auxiliary",
    "ABH_target_decoders": "ABH  target decoders",
    "ALL_combined": "ALL  combined",
}
SHORT = {
    **LABELS,
    "formal_dimension": "Dimension",
    "formal_sdf": "Formal SDF",
    "AB_separate_fusion": "AB fusion",
    "ABC_adaptive_mix": "ABC mix",
    "ABD_shared_coils": "ABD coils",
    "ABE_em_shape_amplitude": "ABE EM",
    "ABF_structure_delta": "ABF delta",
    "ABG_sdf_em_aux": "ABG EM aux",
    "ABH_target_decoders": "ABH heads",
    "ALL_combined": "ALL",
}
INK = "#172033"
MID = "#667085"
GRID = "#D8DDE6"
BLUE = "#0072B2"
GOLD = "#E69F00"
GREEN = "#009E73"
GREY = "#A5ABB6"
WHITE = "#FFFFFF"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, stem: str) -> list[str]:
    outputs: list[str] = []
    for suffix, kwargs in (("png", {"dpi": 190}), ("svg", {}), ("pdf", {})):
        path = REPORT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _key(model: str, case_id: str, name: str) -> str:
    return f"{model}__{case_id}__{name}"


def _row_colors(order: list[str], best: str) -> list[str]:
    return [
        GREY if model == "formal_dimension" else BLUE if model == "formal_sdf" else GREEN if model == best else GOLD
        for model in order
    ]


def _best_and_top(summary: list[dict[str, str]]) -> tuple[str, list[str]]:
    candidates = [row for row in summary if row["model"] in CANDIDATES]
    ordered = sorted(candidates, key=lambda row: float(row["unknown_ni_rel_l2_median_pct"]))
    return ordered[0]["model"], [row["model"] for row in ordered[:3]]


def _metric_lookup(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    return {(row["model"], row["case_id"]): float(row[metric]) for row in rows}


def figure_00(pack: Any, case_rows: list[dict[str, str]], best: str) -> list[str]:
    errors = _metric_lookup(case_rows, "ni_rel_l2_pct")
    row_models: tuple[str | None, ...] = (None, "truth", "formal_dimension", "formal_sdf", best)
    row_labels = ("Coil geometry", "COMSOL", "Formal Dimension", "Formal SDF", f"Best candidate\n{SHORT[best]}")
    fig, axes = plt.subplots(len(row_models), len(CASES), figsize=(15.6, 15.6))
    old_plot._header(
        fig,
        "Unknown coil combinations: direct field comparison",
        "Ion density on one physical color scale (1e17 m^-3). Annotated plasma relative L2: lower is better.",
    )
    truth_fields = [np.asarray(pack[_key("formal_dimension", case, "ni_truth")]) for case in CASES]
    vmax = max(float(np.nanquantile(np.concatenate([field.ravel() for field in truth_fields]), 0.995) / 1e17), 0.1)
    mappable = None
    for column, (case_id, title) in enumerate(zip(CASES, CASE_TITLES)):
        axes[0, column].set_title(title, fontsize=13.5, fontweight="bold", pad=9)
        old_plot._geometry_panel(axes[0, column], case_id)
        for row_index, model in enumerate(row_models[1:], start=1):
            source = "formal_dimension" if model == "truth" else str(model)
            field_name = "ni_truth" if model == "truth" else "ni_prediction"
            field = np.asarray(pack[_key(source, case_id, field_name)], dtype=np.float64) / 1e17
            mask = np.asarray(pack[_key(source, case_id, "mask")], dtype=bool)
            r = np.asarray(pack[_key(source, case_id, "r_coords")], dtype=np.float64)
            z = np.asarray(pack[_key(source, case_id, "z_coords")], dtype=np.float64)
            mappable = axes[row_index, column].pcolormesh(
                r, z, np.where(mask, np.maximum(field, 0.0), np.nan),
                shading="auto", cmap="viridis", vmin=0.0, vmax=vmax, rasterized=True,
            )
            axes[row_index, column].set_aspect("equal", adjustable="box")
            if model != "truth":
                axes[row_index, column].text(
                    0.98, 0.96, f"L2 = {errors[(str(model), case_id)]:.1f}%",
                    transform=axes[row_index, column].transAxes, ha="right", va="top", fontsize=9,
                    bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.84, "pad": 2},
                )
            if row_index < len(row_models) - 1:
                axes[row_index, column].set_xticklabels([])
            else:
                axes[row_index, column].set_xlabel("radius r [cm]")
            if column > 0:
                axes[row_index, column].set_yticklabels([])
            else:
                axes[row_index, column].set_ylabel("height z [cm]")
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].text(
            -0.31, 0.5, label, transform=axes[row_index, 0].transAxes,
            ha="right", va="center", fontsize=11, fontweight="bold" if row_index < 2 else "normal",
        )
    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.014, 0.77])
        fig.colorbar(mappable, cax=cbar_ax).set_label("ion density ni [1e17 m^-3]")
    fig.subplots_adjust(left=0.18, right=0.90, top=0.94, bottom=0.05, hspace=0.20, wspace=0.08)
    return _save(fig, "00_direct_field_comparison")


def figure_01(pack: Any, case_rows: list[dict[str, str]], top: list[str]) -> list[str]:
    errors = _metric_lookup(case_rows, "bohm_profile_rel_l2_pct")
    shown = ["formal_dimension", "formal_sdf", *top]
    line_colors = {"formal_dimension": GREY, "formal_sdf": BLUE, **{model: GOLD for model in top}}
    if top:
        line_colors[top[0]] = GREEN
    styles = ["-", "-", "-", "--", ":"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3))
    old_plot._header(
        fig,
        "Wafer Bohm-flux profiles for the same unknown structures",
        "x: wafer radius [cm]. y: Bohm ion flux from ni and Te [m^-2 s^-1]. COMSOL is the black reference.",
    )
    for ax, case_id, title in zip(axes, CASES, CASE_TITLES):
        truth = np.asarray(pack[_key("formal_dimension", case_id, "bohm_profile_truth")], dtype=float)
        radius = np.asarray(pack[_key("formal_dimension", case_id, "radial_centers")], dtype=float)
        ax.plot(radius, truth, color=INK, linewidth=3.0, label="COMSOL")
        for model, style in zip(shown, styles):
            prediction = np.asarray(pack[_key(model, case_id, "bohm_profile_prediction")], dtype=float)
            ax.plot(
                radius, prediction, color=line_colors[model], linestyle=style, linewidth=1.9,
                label=f"{SHORT[model]} ({errors[(model, case_id)]:.0f}%)",
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("wafer radius r [cm]")
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Bohm flux Γi [m^-2 s^-1]")
    axes[-1].legend(loc="best", fontsize=8.1)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.15, wspace=0.22)
    return _save(fig, "01_bohm_profiles")


def figure_02(model_summary: list[dict[str, str]], best: str) -> list[str]:
    values = {row["model"]: row for row in model_summary}
    order = sorted(MODELS, key=lambda model: float(values[model]["unknown_ni_rel_l2_median_pct"]), reverse=True)
    colors = _row_colors(order, best)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8))
    old_plot._header(
        fig,
        "Overall accuracy on 75 frozen unknown structures",
        "Median physical relative L2 across 25 spacing, 25 height+size, and 25 coupled-change cases. Lower is better.",
    )
    for ax, metric, title in (
        (axes[0], "unknown_ni_rel_l2_median_pct", "2-D ion-density field"),
        (axes[1], "unknown_bohm_rel_l2_median_pct", "Wafer Bohm-flux profile"),
    ):
        numbers = [float(values[model][metric]) for model in order]
        y = np.arange(len(order))
        ax.barh(y, numbers, color=colors, edgecolor=INK, linewidth=0.55)
        ax.set_yticks(y, [LABELS[model] for model in order] if ax is axes[0] else ["" for _ in order])
        ax.set_xlim(left=0.0)
        ax.set_xlabel("median relative L2 [%]")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        for yi, value in zip(y, numbers):
            ax.text(value, yi, f"  {value:.1f}%", va="center", fontsize=9)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.10, wspace=0.18)
    return _save(fig, "02_overall_accuracy_ranking")


def _summary_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    return {
        (row["source"], row["model"], row["layout_kind"], row["metric"]): float(row["median"])
        for row in rows
    }


def figure_03(summary: list[dict[str, str]], model_summary: list[dict[str, str]]) -> list[str]:
    idx = _summary_index(summary)
    overall = {row["model"]: float(row["unknown_ni_rel_l2_median_pct"]) for row in model_summary}
    order = sorted(MODELS, key=lambda model: overall[model])
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 9.0))
    old_plot._header(
        fig,
        "Accuracy by unknown-structure family",
        "Each cell is the median error for 25 COMSOL cases [%]. This exposes gains hidden by the overall median.",
    )
    for ax, metric, title in (
        (axes[0], "ni_rel_l2_pct", "2-D ion-density error [%]"),
        (axes[1], "bohm_profile_rel_l2_pct", "Bohm-profile error [%]"),
    ):
        matrix = np.asarray([[idx[("case", model, family, metric)] for family in FAMILIES] for model in order])
        image = ax.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=float(np.quantile(matrix, 0.95)))
        ax.set_xticks(np.arange(len(FAMILIES)), [FAMILY_LABELS[family] for family in FAMILIES], rotation=18, ha="right")
        ax.set_yticks(np.arange(len(order)), [LABELS[model] for model in order] if ax is axes[0] else ["" for _ in order])
        ax.set_title(title, fontweight="bold")
        threshold = 0.62 * float(np.nanmax(matrix))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=8.5,
                        color=WHITE if matrix[row, column] > threshold else INK)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).set_label("median relative L2 [%]")
    fig.subplots_adjust(left=0.23, right=0.97, top=0.86, bottom=0.13, wspace=0.25)
    return _save(fig, "03_family_accuracy_heatmap")


def figure_04(model_summary: list[dict[str, str]], best: str) -> list[str]:
    values = {row["model"]: row for row in model_summary}
    order = sorted(MODELS, key=lambda model: float(values[model]["structural_delta_ni_rel_l2_median_pct"]), reverse=True)
    colors = _row_colors(order, best)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8))
    old_plot._header(
        fig,
        "Can each model reproduce the change caused by coil structure?",
        "Prediction and COMSOL are each differenced from the matched regular anchor; bars show median relative error of that change. Lower is better.",
    )
    for ax, metric, title in (
        (axes[0], "structural_delta_ni_rel_l2_median_pct", "Change in 2-D ion density"),
        (axes[1], "structural_delta_bohm_rel_l2_median_pct", "Change in Bohm profile"),
    ):
        numbers = [float(values[model][metric]) for model in order]
        y = np.arange(len(order))
        ax.barh(y, numbers, color=colors, edgecolor=INK, linewidth=0.55)
        ax.set_yticks(y, [LABELS[model] for model in order] if ax is axes[0] else ["" for _ in order])
        ax.set_xlim(left=0.0)
        ax.set_xlabel("structural-change relative L2 [%]")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        for yi, value in zip(y, numbers):
            ax.text(value, yi, f"  {value:.0f}%", va="center", fontsize=8.8)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.84, bottom=0.10, wspace=0.18)
    return _save(fig, "04_structural_response_error")


def figure_05(model_summary: list[dict[str, str]], best: str) -> list[str]:
    selected = [row for row in model_summary if row["model"] in CANDIDATES]
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    old_plot._header(
        fig,
        "Candidate accuracy versus fixed-budget training cost",
        "All experimental candidates use the same 15-epoch split/seed budget. Lower-left is preferable; this is an efficiency view, not a promotion rule.",
    )
    offsets = {
        "A_multires": (8, -19),
        "AB_separate_fusion": (7, 8),
        "ABC_adaptive_mix": (8, 9),
        "ABD_shared_coils": (-56, -8),
        "ABE_em_shape_amplitude": (7, 8),
        "ABF_structure_delta": (8, -17),
        "ABG_sdf_em_aux": (8, 11),
        "ABH_target_decoders": (-2, -23),
        "ALL_combined": (7, 8),
    }
    for row in selected:
        model = row["model"]
        x = float(row["training_elapsed_min"]) / 60.0
        y = float(row["unknown_ni_rel_l2_median_pct"])
        color = GREEN if model == best else GOLD
        ax.scatter(x, y, s=85, color=color, edgecolor=INK, linewidth=0.8, zorder=3)
        ax.annotate(
            SHORT[model], (x, y), xytext=offsets[model], textcoords="offset points",
            fontsize=8.8, ha="right" if offsets[model][0] < 0 else "left",
        )
    ax.set_xlabel("training elapsed time [h]")
    ax.set_ylabel("unknown ni median relative L2 [%]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color=GRID, linewidth=0.65)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.12)
    return _save(fig, "05_accuracy_cost_pareto")


def figure_06(paired_rows: list[dict[str, str]]) -> list[str]:
    candidates = list(CANDIDATES)
    label_order = list(reversed(candidates))
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.8), sharey=True)
    old_plot._header(
        fig,
        "Paired change versus Formal SDF on the same 75 structures",
        "Point: median candidate-minus-Formal-SDF error. Line: 95% case-bootstrap interval. Negative is better; intervals crossing zero are inconclusive.",
    )
    for ax, metric, title in (
        (axes[0], "ni_rel_l2_pct", "2-D ion-density field"),
        (axes[1], "bohm_profile_rel_l2_pct", "Wafer Bohm-flux profile"),
    ):
        index = {row["model"]: row for row in paired_rows if row["metric"] == metric}
        y = np.arange(len(label_order))
        centers = np.asarray([float(index[model]["median_error_difference_pp_vs_formal_sdf"]) for model in label_order])
        lows = np.asarray([float(index[model]["bootstrap_ci_low_pp"]) for model in label_order])
        highs = np.asarray([float(index[model]["bootstrap_ci_high_pp"]) for model in label_order])
        colors = [
            GREEN if high < 0.0 else "#C44E52" if low > 0.0 else GREY
            for low, high in zip(lows, highs)
        ]
        ax.axvline(0.0, color=INK, linewidth=1.4)
        for yi, center, low, high, color in zip(y, centers, lows, highs, colors):
            ax.errorbar(
                center, yi, xerr=np.asarray([[center - low], [high - center]]), fmt="o",
                color=color, ecolor=color, markersize=6.5, capsize=3.5, linewidth=1.8,
            )
        ax.set_yticks(y)
        ax.set_xlabel("candidate − Formal SDF error [percentage points]")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.65)
    axes[0].set_yticks(np.arange(len(label_order)), [LABELS[model] for model in label_order])
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(
        0.5, 0.035,
        "green: decisive improvement   gray: inconclusive   red: decisive regression",
        ha="center", color=MID, fontsize=10,
    )
    fig.subplots_adjust(left=0.23, right=0.97, top=0.84, bottom=0.14, wspace=0.16)
    return _save(fig, "06_paired_effect_vs_formal_sdf")


def main() -> int:
    old_plot._style()
    old_plot.REPORT = REPORT
    REPORT.mkdir(parents=True, exist_ok=True)
    case_rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    summary = _read_csv(REPORT / "summary_metrics.csv")
    model_summary = _read_csv(REPORT / "model_summary.csv")
    paired_rows = _read_csv(REPORT / "paired_effects_vs_formal_sdf.csv")
    best, top = _best_and_top(model_summary)
    with np.load(REPORT / "representative_predictions.npz", allow_pickle=False) as pack:
        figures = {
            "00": figure_00(pack, case_rows, best),
            "01": figure_01(pack, case_rows, top),
            "02": figure_02(model_summary, best),
            "03": figure_03(summary, model_summary),
            "04": figure_04(model_summary, best),
            "05": figure_05(model_summary, best),
            "06": figure_06(paired_rows),
        }
    chart_map = {
        "00": {"question": "Do representative predicted fields resemble COMSOL?", "claim": "Direct visual check only; common physical scale.", "files": figures["00"]},
        "01": {"question": "Are wafer Bohm-flux profiles reproduced?", "claim": "Profile shape and level on three fixed cases.", "files": figures["01"]},
        "02": {"question": "Which model is most accurate overall?", "claim": "75-case median ni and Bohm errors.", "files": figures["02"]},
        "03": {"question": "Is the result robust across structure families?", "claim": "25 cases per family; no family is hidden by pooling.", "files": figures["03"]},
        "04": {"question": "Does the model reproduce structure-induced changes?", "claim": "Direct error of prediction-minus-anchor versus COMSOL-minus-anchor.", "files": figures["04"]},
        "05": {"question": "What accuracy is obtained per fixed-budget training cost?", "claim": "Efficiency diagnostic among experimental candidates only.", "files": figures["05"]},
        "06": {"question": "Are candidate improvements consistent on the same cases?", "claim": "Paired median error change and 95% case-bootstrap interval versus Formal SDF.", "files": figures["06"]},
        "best_candidate_by_unknown_ni": best,
        "top_three_candidates": top,
    }
    (REPORT / "figure_manifest.json").write_text(json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(chart_map, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
