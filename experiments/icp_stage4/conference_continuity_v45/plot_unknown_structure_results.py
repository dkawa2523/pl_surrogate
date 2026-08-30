#!/usr/bin/env python
"""Create direct, conference-ready figures from the audited v45 evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/conference_continuity_v45"
DATASET = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
V43_RAW = ROOT / "data/outputs_icp_v43_structure_600"
MODELS = (
    "conference_dimension",
    "explicit_dimension",
    "conference_sdf",
    "sdf_vacuum",
    "sdf_vacuum_structure",
)
MODEL_LABELS = {
    "conference_dimension": "Formal Dimension\n(7 scalars)",
    "explicit_dimension": "Explicit Dimension\n(all coils)",
    "conference_sdf": "Formal SDF",
    "sdf_vacuum": "SDF + vacuum B",
    "sdf_vacuum_structure": "SDF + B + structure",
}
MODEL_SHORT = {
    "conference_dimension": "Formal Dim.",
    "explicit_dimension": "Explicit Dim.",
    "conference_sdf": "SDF",
    "sdf_vacuum": "SDF + B",
    "sdf_vacuum_structure": "SDF + B + struct.",
}
MODEL_COLORS = {
    "conference_dimension": "#9CA3AF",
    "explicit_dimension": "#D55E00",
    "conference_sdf": "#0072B2",
    "sdf_vacuum": "#009E73",
    "sdf_vacuum_structure": "#7E57C2",
}
FAMILIES = (
    "unknown_gap_topology",
    "unseen_rank_height_size",
    "coupled_transform",
)
FAMILY_LABELS = {
    "unknown_gap_topology": "Unknown spacing",
    "unseen_rank_height_size": "Unknown height + size",
    "coupled_transform": "Coupled change",
}
CASES = (
    "v43_te_n4_variant_1__center",
    "v43_te_n4_variant_2__center",
    "v43_te_n4_variant_3__center",
)
CASE_TITLES = (
    "A  Unequal spacing",
    "B  Height + size combination",
    "C  Coupled spacing/height/size",
)
INK = "#172033"
MID = "#667085"
GRID = "#D8DDE6"
WHITE = "#FFFFFF"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Yu Gothic", "Meiryo", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, stem: str) -> list[str]:
    names: list[str] = []
    for suffix, kwargs in (("png", {"dpi": 190}), ("svg", {}), ("pdf", {})):
        path = REPORT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        names.append(path.name)
    plt.close(fig)
    return names


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.985, title, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(0.035, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=MID)


def _key(model: str, case_id: str, name: str) -> str:
    return f"{model}__{case_id}__{name}"


def _layout(case_id: str) -> list[dict[str, str]]:
    return _read_csv(V43_RAW / "structure/coil_layout" / f"{case_id}__coil_layout.csv")


def _geometry_panel(ax: plt.Axes, case_id: str) -> None:
    active = [row for row in _layout(case_id) if int(row["active"]) == 1]
    for row in active:
        rectangle = patches.Rectangle(
            (float(row["r_min"]), float(row["z_min"])),
            float(row["width"]),
            float(row["height"]),
            facecolor="#4C78A8",
            edgecolor=INK,
            linewidth=0.8,
        )
        ax.add_patch(rectangle)
        ax.text(
            float(row["r_center"]),
            float(row["z_center"]),
            str(row["coil_index"]),
            ha="center",
            va="center",
            color=WHITE,
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(14.0, 21.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_xlabel("radius r [cm]")
    ax.set_ylabel("height z [cm]")


def _metric_lookup(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    return {(row["model"], row["case_id"]): float(row[metric]) for row in rows}


def figure_00(pack: Any, case_rows: list[dict[str, str]]) -> list[str]:
    rel_l2 = _metric_lookup(case_rows, "ni_rel_l2_pct")
    row_models: tuple[str | None, ...] = (None, "truth", *MODELS)
    row_labels = ("Coil geometry", "COMSOL", *(MODEL_LABELS[model] for model in MODELS))
    fig, axes = plt.subplots(
        len(row_models), len(CASES), figsize=(15.8, 22.0), constrained_layout=False
    )
    _header(
        fig,
        "Unknown coil combinations: geometry, COMSOL, and model predictions",
        "Ion density is shown on one common physical color scale (1e17 m^-3).  Lower annotated relative L2 is better.",
    )
    truth_fields = [
        np.asarray(pack[_key(MODELS[0], case_id, "ni_truth")], dtype=np.float64) for case_id in CASES
    ]
    vmax = float(np.nanquantile(np.concatenate([field.reshape(-1) for field in truth_fields]), 0.995) / 1.0e17)
    vmax = max(vmax, 0.1)
    mappable = None
    for column, (case_id, title) in enumerate(zip(CASES, CASE_TITLES)):
        axes[0, column].set_title(title, fontsize=14, fontweight="bold", pad=10)
        _geometry_panel(axes[0, column], case_id)
        for row_index, model in enumerate(row_models[1:], start=1):
            source_model = MODELS[0] if model == "truth" else str(model)
            field_name = "ni_truth" if model == "truth" else "ni_prediction"
            field = np.asarray(pack[_key(source_model, case_id, field_name)], dtype=np.float64) / 1.0e17
            mask = np.asarray(pack[_key(source_model, case_id, "mask")], dtype=bool)
            r = np.asarray(pack[_key(source_model, case_id, "r_coords")], dtype=np.float64)
            z = np.asarray(pack[_key(source_model, case_id, "z_coords")], dtype=np.float64)
            displayed = np.where(mask, np.maximum(field, 0.0), np.nan)
            mappable = axes[row_index, column].pcolormesh(
                r, z, displayed, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax, rasterized=True
            )
            axes[row_index, column].set_xlim(float(np.nanmin(r)), float(np.nanmax(r)))
            axes[row_index, column].set_ylim(float(np.nanmin(z)), float(np.nanmax(z)))
            axes[row_index, column].set_aspect("equal", adjustable="box")
            if model != "truth":
                axes[row_index, column].text(
                    0.98,
                    0.96,
                    f"rel. L2 = {rel_l2[(str(model), case_id)]:.1f}%",
                    transform=axes[row_index, column].transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
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
            -0.34,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold" if row_index in (0, 1) else "normal",
        )
    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.075, 0.015, 0.78])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("ion density ni [1e17 m^-3]")
    fig.subplots_adjust(left=0.18, right=0.90, top=0.945, bottom=0.045, hspace=0.22, wspace=0.08)
    return _save(fig, "00_unknown_combination_field_comparison")


def figure_01(pack: Any, case_rows: list[dict[str, str]]) -> list[str]:
    errors = _metric_lookup(case_rows, "bohm_profile_rel_l2_pct")
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3))
    _header(
        fig,
        "Wafer Bohm-flux profiles for the same three unknown structures",
        "x: wafer radius [cm].  y: Bohm ion flux from predicted ni and Te [m^-2 s^-1].  COMSOL is the black reference.",
    )
    for ax, case_id, title in zip(axes, CASES, CASE_TITLES):
        truth = np.asarray(pack[_key(MODELS[0], case_id, "bohm_profile_truth")], dtype=np.float64)
        radius = np.asarray(pack[_key(MODELS[0], case_id, "radial_centers")], dtype=np.float64)
        ax.plot(radius, truth, color=INK, linewidth=3.0, label="COMSOL", zorder=10)
        for model in MODELS:
            predicted = np.asarray(pack[_key(model, case_id, "bohm_profile_prediction")], dtype=np.float64)
            ax.plot(
                radius,
                predicted,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                label=f"{MODEL_SHORT[model]} ({errors[(model, case_id)]:.0f}%)",
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("wafer radius r [cm]")
        ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Bohm flux Gamma_i [m^-2 s^-1]")
    axes[-1].legend(loc="upper right", fontsize=8.4, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.15, wspace=0.22)
    return _save(fig, "01_unknown_combination_bohm_profiles")


def _summary_lookup(summary: list[dict[str, str]], source: str, metric: str) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    for row in summary:
        if row["source"] != source or row["metric"] != metric:
            continue
        result[(row["model"], row["layout_kind"])] = {
            key: float(row[key]) for key in ("median", "bootstrap_median_ci_low", "bootstrap_median_ci_high")
        }
    return result


def figure_02(summary: list[dict[str, str]]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.9))
    _header(
        fig,
        "Accuracy on 75 frozen unknown-combination COMSOL cases",
        "Points are family medians; vertical bars are case-bootstrap 95% confidence intervals.  Lower is better.",
    )
    for ax, metric, title, ylabel in (
        (axes[0], "ni_rel_l2_pct", "2-D ion-density field", "plasma relative L2 [%]"),
        (axes[1], "bohm_profile_rel_l2_pct", "Wafer Bohm-flux profile", "profile relative L2 [%]"),
    ):
        lookup = _summary_lookup(summary, "case", metric)
        x = np.arange(len(FAMILIES), dtype=float)
        offsets = np.linspace(-0.28, 0.28, len(MODELS))
        for offset, model in zip(offsets, MODELS):
            values = np.asarray([lookup[(model, family)]["median"] for family in FAMILIES])
            lower = np.asarray([lookup[(model, family)]["bootstrap_median_ci_low"] for family in FAMILIES])
            upper = np.asarray([lookup[(model, family)]["bootstrap_median_ci_high"] for family in FAMILIES])
            ax.errorbar(
                x + offset,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                fmt="o",
                markersize=7,
                capsize=3,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=MODEL_SHORT[model],
            )
        ax.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILIES])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, linewidth=0.65)
        ax.set_ylim(bottom=0.0)
    axes[1].legend(loc="upper left", fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.18, wspace=0.22)
    return _save(fig, "02_unknown_family_accuracy")


def figure_03(summary: list[dict[str, str]], case_rows: list[dict[str, str]]) -> list[str]:
    sdf_models = MODELS[2:]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    _header(
        fig,
        "Controlled additions to the conference SDF model",
        "Same UNO, split, seed and epoch budget.  Bars are medians over the same 75 cases; labels show paired win rate versus the previous rung.",
    )
    for ax, metric, title in (
        (axes[0], "ni_rel_l2_pct", "2-D ion-density error"),
        (axes[1], "bohm_profile_rel_l2_pct", "Bohm-profile error"),
    ):
        lookup = _summary_lookup(summary, "case", metric)
        medians = [lookup[(model, "all_unknown")]["median"] for model in sdf_models]
        bars = ax.bar(
            np.arange(3), medians, color=[MODEL_COLORS[model] for model in sdf_models], width=0.65
        )
        by_key = {(row["model"], row["case_id"]): float(row[metric]) for row in case_rows}
        labels = ["baseline"]
        for previous, current in zip(sdf_models[:-1], sdf_models[1:]):
            common = sorted(
                {case_id for model, case_id in by_key if model == previous}
                & {case_id for model, case_id in by_key if model == current}
            )
            wins = 100.0 * np.mean([by_key[(current, case_id)] < by_key[(previous, case_id)] for case_id in common])
            labels.append(f"wins {wins:.0f}%")
        for bar, value, label in zip(bars, medians, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%\n{label}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(np.arange(3), [MODEL_SHORT[model] for model in sdf_models])
        ax.set_ylabel("relative L2 [%] (lower is better)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", color=GRID, linewidth=0.65)
        ax.set_ylim(0.0, max(medians) * 1.30)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.81, bottom=0.18, wspace=0.24)
    return _save(fig, "03_incremental_sdf_improvement")


def figure_04(response_rows: list[dict[str, str]]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5))
    _header(
        fig,
        "Does the model reproduce the change caused by geometry?",
        "Each point summarizes 25 paired unknown-minus-regular responses.  Ideal is (cosine, gain) = (1, 1): correct direction and magnitude.",
    )
    for ax, family in zip(axes, FAMILIES):
        for model in MODELS:
            selected = [row for row in response_rows if row["model"] == model and row["layout_kind"] == family]
            cosine = np.asarray([float(row["ni_delta_cosine"]) for row in selected])
            gain = np.asarray([float(row["ni_delta_gain"]) for row in selected])
            ax.errorbar(
                float(np.nanmedian(cosine)),
                float(np.nanmedian(gain)),
                xerr=np.asarray([[np.nanmedian(cosine) - np.nanquantile(cosine, 0.25)], [np.nanquantile(cosine, 0.75) - np.nanmedian(cosine)]]),
                yerr=np.asarray([[np.nanmedian(gain) - np.nanquantile(gain, 0.25)], [np.nanquantile(gain, 0.75) - np.nanmedian(gain)]]),
                fmt="o",
                markersize=8,
                capsize=3,
                color=MODEL_COLORS[model],
                label=MODEL_SHORT[model],
            )
        ax.scatter([1.0], [1.0], marker="*", s=180, color=INK, label="ideal", zorder=10)
        ax.axvline(1.0, color=GRID, linewidth=0.9)
        ax.axhline(1.0, color=GRID, linewidth=0.9)
        ax.set_title(FAMILY_LABELS[family], fontweight="bold")
        ax.set_xlabel("response cosine (1 = correct direction)")
        ax.grid(True, color=GRID, linewidth=0.55, alpha=0.6)
    axes[0].set_ylabel("response gain (1 = correct magnitude)")
    axes[-1].legend(loc="best", fontsize=8.4, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.18, wspace=0.22)
    return _save(fig, "04_structural_response_fidelity")


def figure_05(summary: list[dict[str, str]]) -> list[str]:
    collision = _read_csv(DATASET / "formal_dimension_collision_cases.csv")
    collision_case_count = len({row["case_id"] for row in collision})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    _header(
        fig,
        "Fairness audit: formal Dimension versus full-information Dimension",
        "The seven-scalar conference input aliases distinct coil layouts.  Explicit Dimension receives every coil rectangle, so SDF is not compared only against a handicapped baseline.",
    )
    names = ("Formal 7-scalar\nDimension", "Explicit all-coil\nDimension", "SDF geometry\nfield")
    counts = (collision_case_count, 0, 0)
    bars = axes[0].bar(np.arange(3), counts, color=[MODEL_COLORS[MODELS[0]], MODEL_COLORS[MODELS[1]], MODEL_COLORS[MODELS[2]]])
    for bar, value in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", va="bottom", fontweight="bold")
    axes[0].set_xticks(np.arange(3), names)
    axes[0].set_ylabel("cases sharing a representation with another layout")
    axes[0].set_title("Representation collisions in 957 cases", fontweight="bold")
    axes[0].grid(True, axis="y", color=GRID, linewidth=0.65)

    lookup_field = _summary_lookup(summary, "case", "ni_rel_l2_pct")
    lookup_bohm = _summary_lookup(summary, "case", "bohm_profile_rel_l2_pct")
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36
    field = [lookup_field[(model, "all_unknown")]["median"] for model in MODELS]
    bohm = [lookup_bohm[(model, "all_unknown")]["median"] for model in MODELS]
    axes[1].bar(x - width / 2, field, width, color="#4C78A8", label="ion-density field")
    axes[1].bar(x + width / 2, bohm, width, color="#F2A541", label="Bohm profile")
    axes[1].set_xticks(x, [MODEL_SHORT[model] for model in MODELS], rotation=18, ha="right")
    axes[1].set_ylabel("median relative L2 [%]")
    axes[1].set_title("Same 75 unknown cases", fontweight="bold")
    axes[1].grid(True, axis="y", color=GRID, linewidth=0.65)
    axes[1].legend(fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.23, wspace=0.24)
    return _save(fig, "05_dimension_fairness_audit")


def main() -> int:
    _style()
    REPORT.mkdir(parents=True, exist_ok=True)
    pack = np.load(REPORT / "representative_predictions.npz", allow_pickle=False)
    case_rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    response_rows = _read_csv(REPORT / "structural_response_metrics_unknown75.csv")
    summary = _read_csv(REPORT / "summary_metrics.csv")
    outputs = {
        "00": figure_00(pack, case_rows),
        "01": figure_01(pack, case_rows),
        "02": figure_02(summary),
        "03": figure_03(summary, case_rows),
        "04": figure_04(response_rows),
        "05": figure_05(summary),
    }
    metadata: dict[str, Any] = {
        "outputs": outputs,
        "chart_map": "experiments/icp_stage4/conference_continuity_v45/CHART_MAP.md",
        "source_files": [
            "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/index.csv",
            "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/design_metadata_v43.csv",
            "reports/icp_conference_materials/conference_continuity_v45/case_metrics_unknown75.csv",
            "reports/icp_conference_materials/conference_continuity_v45/structural_response_metrics_unknown75.csv",
            "reports/icp_conference_materials/conference_continuity_v45/summary_metrics.csv",
        ],
        "figure_00_color_scale": "shared physical ion density in 1e17 m^-3, vmax=99.5th percentile of three COMSOL fields",
        "uncertainty": "2000-case-bootstrap 95% CI of median",
    }
    (REPORT / "figure_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
