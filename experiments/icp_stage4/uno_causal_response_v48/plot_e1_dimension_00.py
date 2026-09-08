#!/usr/bin/env python
"""Rebuild the simple-geometry 00 figure for Dimension versus E1 only."""

from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "reports/icp_conference_materials/icp_uno_causal_response_v48/simple_geometry_seed1237"
OUT = ROOT / "reports/icp_conference_materials/icp_uno_e1_dimension_conference_v49"
CASES = ("spacing_right", "size_gradient", "height_gradient_mild")
TITLES = {
    "spacing_left": "Unequal spacing: middle -2.5 cm",
    "spacing_right": "Unequal spacing: middle +2.5 cm",
    "size_gradient": "Size only: 0.60 / 1.00 / 1.40 cm",
    "height_gradient_mild": "Height only: -0.25 / 0 / +0.25 cm",
}
COLORS = {"COMSOL": "#171A1F", "Formal Dimension": "#2F6BFF", "E1 causal EM": "#D55E00"}
INK, MID, GRID = "#182230", "#667085", "#E4E7EC"


def _read(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_geometry(axis: plt.Axes, case: str, qa: list[dict[str, str]]) -> None:
    anchor = [row for row in qa if row["case"] == "anchor"]
    changed = [row for row in qa if row["case"] == case]
    for row in anchor:
        width, height = float(row["layout_width_cm"]), float(row["layout_height_cm"])
        axis.add_patch(
            Rectangle(
                (float(row["layout_r_center_cm"]) - width / 2, float(row["layout_z_center_cm"]) - height / 2),
                width,
                height,
                fill=False,
                edgecolor="#667085",
                linewidth=1.35,
                linestyle="--",
            )
        )
    for row in changed:
        width, height = float(row["layout_width_cm"]), float(row["layout_height_cm"])
        axis.add_patch(
            Rectangle(
                (float(row["layout_r_center_cm"]) - width / 2, float(row["layout_z_center_cm"]) - height / 2),
                width,
                height,
                facecolor="#E69F00",
                edgecolor="#8A4F00",
                linewidth=1.25,
            )
        )
    axis.set(xlim=(6.5, 19.0), ylim=(16.55, 19.20), xlabel="r (cm)")
    axis.tick_params(labelsize=7, length=2)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    profile_rows = _read("profile_bins.csv")
    metrics = _read("profile_accuracy.csv")
    qa = _read("input_geometry_qa.csv")
    by_case = {case: [row for row in profile_rows if row["case"] == case] for case in CASES}
    values = np.concatenate(
        [
            np.asarray([float(row[column]) for row in by_case[case]], dtype=float) / 1.0e20
            for case in CASES
            for column in ("comsol_bohm_flux_m2_s", "formal_dimension_bohm_flux_m2_s", "e1_causal_em_bohm_flux_m2_s")
        ]
    )
    pad = 0.06 * float(np.ptp(values))
    metric = {
        (row["case"], row["model"]): float(row["bohm_profile_relative_l2_pct"])
        for row in metrics
    }
    ymax_bar = 1.17 * max(metric[(case, model)] for case in CASES for model in ("Formal Dimension", "E1 causal EM"))

    fig = plt.figure(figsize=(16.8, 8.7))
    # Keep the original conference canvas while expanding the three retained
    # cases into equal-width columns.
    grid = fig.add_gridspec(3, len(CASES), height_ratios=(0.70, 2.35, 1.0), hspace=0.40, wspace=0.24)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.09)
    fig.suptitle("Simple coil interventions: Dimension UNO versus E1 causal EM", fontsize=19, weight="bold", y=0.985, color=INK)
    fig.text(
        0.5,
        0.948,
        "COMSOL reference; frozen seed-1237 models trained for 200 epochs on the same 957 cases. Lower relative-L2 error is better.",
        ha="center",
        fontsize=10.2,
        color=MID,
    )

    legend_handles = None
    for column_index, case in enumerate(CASES):
        geo = fig.add_subplot(grid[0, column_index])
        _draw_geometry(geo, case, qa)
        geo.set_title(TITLES[case], fontsize=11.5, weight="bold", pad=6)
        if column_index == 0:
            geo.set_ylabel("z (cm)")
        else:
            geo.tick_params(labelleft=False)

        axis = fig.add_subplot(grid[1, column_index])
        rows = by_case[case]
        radius = np.asarray([float(row["radius_cm"]) for row in rows])
        specs = (
            ("COMSOL", "comsol_bohm_flux_m2_s", "-", "o", 2.7),
            ("Formal Dimension", "formal_dimension_bohm_flux_m2_s", "--", "^", 2.2),
            ("E1 causal EM", "e1_causal_em_bohm_flux_m2_s", "-", "s", 2.7),
        )
        lines = []
        for label, value_column, style, marker, width in specs:
            line = axis.plot(
                radius,
                np.asarray([float(row[value_column]) for row in rows]) / 1.0e20,
                color=COLORS[label],
                linestyle=style,
                marker=marker,
                markersize=3.5,
                markevery=2,
                markerfacecolor="white" if label != "E1 causal EM" else COLORS[label],
                linewidth=width,
                label=label,
            )[0]
            lines.append(line)
        if legend_handles is None:
            legend_handles = lines
        axis.set(xlabel="Wafer radius r (cm)", ylim=(float(values.min() - pad), float(values.max() + pad)))
        if column_index == 0:
            axis.set_ylabel(r"Bohm flux $\Gamma_B$ ($10^{20}$ m$^{-2}$ s$^{-1}$)")
        else:
            axis.tick_params(labelleft=False)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

        bars = fig.add_subplot(grid[2, column_index])
        labels = ("Dimension", "E1")
        heights = (metric[(case, "Formal Dimension")], metric[(case, "E1 causal EM")])
        rectangles = bars.bar(labels, heights, color=(COLORS["Formal Dimension"], COLORS["E1 causal EM"]), edgecolor=INK, linewidth=0.8, width=0.62)
        for rectangle, value in zip(rectangles, heights, strict=True):
            bars.text(rectangle.get_x() + rectangle.get_width() / 2, value + 0.45, f"{value:.1f}%", ha="center", va="bottom", fontsize=9.5)
        bars.set_ylim(0.0, ymax_bar)
        if column_index == 0:
            bars.set_ylabel("Profile error (%)")
        else:
            bars.tick_params(labelleft=False)
        bars.grid(axis="y", color=GRID, linewidth=0.8)
        bars.spines[["top", "right"]].set_visible(False)

    assert legend_handles is not None
    fig.legend(legend_handles, [line.get_label() for line in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3, frameon=False, fontsize=11)
    fig.text(
        0.5,
        0.025,
        "Dashed rectangles show the original layout. Dimension remains unchanged because these per-coil interventions map to the same seven scalar inputs.",
        ha="center",
        fontsize=9.5,
        color=MID,
    )
    for suffix in ("png", "svg"):
        path = OUT / f"00_e1_vs_dimension_simple_geometry_bohm.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ET.parse(OUT / "00_e1_vs_dimension_simple_geometry_bohm.svg")
    print(OUT / "00_e1_vs_dimension_simple_geometry_bohm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
