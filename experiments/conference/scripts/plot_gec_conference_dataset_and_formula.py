"""Create slide-ready dataset-composition and calculation-sheet figures.

These figures complement (and do not replace) the deliberately minimal
overview figures from ``plot_gec_conference_simple.py``.  All counts,
condition ranges, split memberships, scaler values, and response statistics
are read from the effective training artifacts used by the current CCP and
ICP studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.conference.scripts import plot_gec_conference_simple as simple  # noqa: E402
from experiments.conference.scripts import plot_gec_loss_design as loss_detail  # noqa: E402


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials")
FORMATS = ("png", "pdf", "svg")

CCP_ROOT = REPO_ROOT / "data" / "outputs_merged_td_csv_periodic_ext0520_v2"
CCP_INDEX = CCP_ROOT / "index_78.csv"
CCP_RUN = REPO_ROOT / "runs" / "gec_ccp_nn_operator_comparison_v1" / "seed_412" / "n78" / "fno"
CCP_SPLIT = CCP_RUN / "preprocessing" / "split" / "split_interp_v1.json"
CCP_SCALERS = CCP_RUN / "preprocessing" / "scalers" / "by_split" / "interp" / "y_scalers.json"
CCP_FIT_POLICY = CCP_RUN / "preprocessing" / "scalers" / "by_split" / "interp" / "fit_policy.json"

ICP_ROOT = REPO_ROOT / "data" / "outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2"
ICP_INDEX = ICP_ROOT / "index.csv"
ICP_RUN = REPO_ROOT / "runs" / "icp_stage4_coil_structure_v1" / "unet"
ICP_SPLIT = ICP_RUN / "preprocessing" / "split" / "split_structure_holdout_v1.json"
ICP_SPLIT_META = ICP_RUN / "preprocessing" / "split" / "split_structure_holdout_meta_v1.json"
ICP_CONFIG = ICP_RUN / "resolved_config.yaml"

COLORS = simple.COLORS
SPLIT_COLORS = {
    "train": COLORS["blue"],
    "val": COLORS["orange"],
    "test": COLORS["green"],
}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _save_slide(
    fig: plt.Figure,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Save an exact 16:9 canvas while preserving editable vector text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with plt.rc_context(simple.STYLE):
        for suffix in dict.fromkeys(formats):
            path = out_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, facecolor="white")
            outputs.append(path)
    return outputs


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _data_path(root: Path, raw: str) -> Path:
    return root / Path(raw.replace("\\", "/"))


def _split_lookup(split: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for case_id in split[name]:
            if case_id in lookup:
                raise ValueError(f"duplicate split membership: {case_id}")
            lookup[case_id] = name
    return lookup


def _plasma_ne_p95(root: Path, rows: Sequence[dict[str, str]]) -> np.ndarray:
    masks: dict[str, np.ndarray] = {}
    result: list[float] = []
    for row in rows:
        mask_key = row.get("base_case_id") or row["structure_npz"]
        if mask_key not in masks:
            with np.load(_data_path(root, row["structure_npz"]), allow_pickle=True) as structure:
                masks[mask_key] = np.asarray(structure["mask_plasma"], dtype=bool)
        with np.load(_data_path(root, row["fields_npz"]), allow_pickle=False) as fields:
            ne = np.asarray(fields["ne"], dtype=np.float64)
        values = ne[masks[mask_key]]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            raise ValueError(f"no finite positive plasma ne values for {row['case_id']}")
        result.append(float(np.quantile(values, 0.95)))
    return np.asarray(result, dtype=np.float64)


def _plasma_fit_pixel_count(
    root: Path,
    rows: Sequence[dict[str, str]],
    train_ids: set[str],
) -> int:
    cache: dict[str, int] = {}
    total = 0
    for row in rows:
        if row["case_id"] not in train_ids:
            continue
        key = row.get("base_case_id") or row["structure_npz"]
        if key not in cache:
            with np.load(_data_path(root, row["structure_npz"]), allow_pickle=True) as structure:
                cache[key] = int(np.count_nonzero(np.asarray(structure["mask_plasma"], dtype=bool)))
        total += cache[key]
    return total


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str,
    face: str = "#FFFFFF",
    linewidth: float = 1.6,
    radius: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def _draw_split_bar(
    ax: plt.Axes,
    counts: dict[str, int],
    *,
    total: int,
    y: float = 0.5,
    height: float = 0.34,
    show_labels: bool = True,
) -> None:
    left = 0
    for name in ("train", "val", "test"):
        count = int(counts[name])
        ax.barh(y, count, left=left, height=height, color=SPLIT_COLORS[name], edgecolor="white", linewidth=1.5)
        ax.text(
            left + count / 2.0,
            y,
            f"{count}",
            ha="center",
            va="center",
            color="white",
            fontsize=16.0,
            weight="bold",
        )
        if show_labels:
            label_x = left + count / 2.0
            if count / total < 0.16 and name == "val":
                label_x -= 0.015 * total
            if count / total < 0.16 and name == "test":
                label_x += 0.015 * total
            ax.text(
                label_x,
                min(0.96, y + height / 2.0 + 0.16),
                SPLIT_LABELS[name],
                ha="center",
                va="bottom",
                color=SPLIT_COLORS[name],
                fontsize=12.0,
                weight="bold",
            )
        left += count
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _condition_counts(rows: Sequence[dict[str, str]], key: str) -> dict[float, int]:
    counts: dict[float, int] = {}
    for row in rows:
        value = float(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _condition_split_counts(
    rows: Sequence[dict[str, str]],
    key: str,
    split_lookup: dict[str, str],
) -> dict[float, dict[str, int]]:
    counts: dict[float, dict[str, int]] = {}
    for row in rows:
        value = float(row[key])
        entry = counts.setdefault(value, {"train": 0, "val": 0, "test": 0})
        entry[split_lookup[row["case_id"]]] += 1
    return counts


def create_ccp_dataset_composition() -> tuple[plt.Figure, dict[str, Any]]:
    rows = _read_rows(CCP_INDEX)
    split = _read_json(CCP_SPLIT)
    lookup = _split_lookup(split)
    if len(rows) != 78 or set(lookup) != {row["case_id"] for row in rows}:
        raise ValueError("CCP current-run dataset/split contract mismatch")
    p95 = _plasma_ne_p95(CCP_ROOT, rows)

    counts = {
        "Td": _condition_counts(rows, "Td"),
        "PP0": _condition_counts(rows, "PP0"),
        "PA": _condition_counts(rows, "PA"),
        "gamma": _condition_counts(rows, "gamma"),
    }
    counts_by_split = {
        key: _condition_split_counts(rows, key, lookup)
        for key in ("Td", "PP0", "PA", "gamma")
    }
    split_counts = {name: len(split[name]) for name in ("train", "val", "test")}
    geometry_for_td = {0.03: "base4", 0.16: "base3", 0.30: "base2"}

    with plt.rc_context(simple.STYLE):
        fig = plt.figure(figsize=(16.0, 9.0))
        grid = fig.add_gridspec(
            2,
            2,
            left=0.06,
            right=0.97,
            bottom=0.10,
            top=0.83,
            width_ratios=(1.0, 1.06),
            height_ratios=(0.40, 0.60),
            wspace=0.20,
            hspace=0.34,
        )
        ax_design = fig.add_subplot(grid[0, 0])
        ax_split = fig.add_subplot(grid[0, 1])
        ax_conditions = fig.add_subplot(grid[1, 0])
        ax_response = fig.add_subplot(grid[1, 1])
        fig.suptitle("GEC-CCP dataset at a glance", fontsize=26.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, "Current training population  ·  78 simulations", ha="center", fontsize=17.0, color=COLORS["muted"])

        ax_design.set_axis_off()
        ax_design.set_title("Dataset design", loc="left", fontsize=19.5, weight="bold", pad=10.0)
        _rounded_box(ax_design, 0.00, 0.05, 0.98, 0.77, edge=COLORS["blue"], face=COLORS["panel"], linewidth=1.8, radius=0.025)
        ax_design.text(0.49, 0.58, "78 / 81", transform=ax_design.transAxes, ha="center", va="center", fontsize=35.0, weight="bold", color=COLORS["blue"])
        ax_design.text(0.49, 0.40, "condition tuples available  (96%)", transform=ax_design.transAxes, ha="center", fontsize=16.0, weight="bold")
        ax_design.text(0.49, 0.24, r"3 Geometry/$T_d$ choices  ×  3 $P_{P0}$  ×  3 $P_A$  ×  3 $\gamma$", transform=ax_design.transAxes, ha="center", fontsize=14.0, color=COLORS["muted"])
        ax_design.text(0.49, 0.11, r"Absent: base3 / $T_d=.16$, $P_{P0}=5$, $P_A=.05$  (3 $\gamma$ values)", transform=ax_design.transAxes, ha="center", fontsize=11.8, color=COLORS["muted"])

        ax_split.set_title("Train / Validation / Test", fontsize=19.5, weight="bold", loc="left", pad=10.0)
        _draw_split_bar(ax_split, split_counts, total=78, y=0.55, height=0.38)
        ax_split.text(0.50, 0.13, "54 / 11 / 13 cases  ·  no case overlap", transform=ax_split.transAxes, ha="center", fontsize=14.0, weight="bold", color=COLORS["ink"])
        ax_split.text(0.50, 0.02, "Every condition level appears in every partition", transform=ax_split.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])

        ax_conditions.set_axis_off()
        ax_conditions.set_title("Condition coverage", loc="left", fontsize=19.5, weight="bold", pad=10.0)
        row_specs = [
            ("Geometry / $T_d$", [(0.03, "base4 / .03"), (0.16, "base3 / .16"), (0.30, "base2 / .30")]),
            ("$P_{P0}$", [(1.0, "1"), (3.0, "3"), (5.0, "5")]),
            ("$P_A$", [(0.05, ".05"), (0.10, ".10"), (0.20, ".20")]),
            (r"$\gamma$", [(0.04, ".04"), (0.07, ".07"), (0.10, ".10")]),
        ]
        key_by_row = ("Td", "PP0", "PA", "gamma")
        y_positions = (0.76, 0.53, 0.30, 0.07)
        x_positions = (0.43, 0.66, 0.89)
        for (label, levels), key, y in zip(row_specs, key_by_row, y_positions, strict=True):
            ax_conditions.text(0.00, y + 0.06, label, transform=ax_conditions.transAxes, fontsize=15.0, weight="bold", va="center")
            for x, (value, display) in zip(x_positions, levels, strict=True):
                _rounded_box(ax_conditions, x - 0.09, y - 0.02, 0.18, 0.16, edge=COLORS["line"], face="#FFFFFF", linewidth=1.15, radius=0.018)
                ax_conditions.text(x, y + 0.077, display, transform=ax_conditions.transAxes, ha="center", va="center", fontsize=13.0, weight="bold")
                ax_conditions.text(x, y + 0.018, f"n = {counts[key][value]}", transform=ax_conditions.transAxes, ha="center", va="center", fontsize=12.0, color=COLORS["blue"], weight="bold")

        group_order = ["base4", "base3", "base2"]
        group_labels = ["base4", "base3", "base2"]
        response_groups = [
            p95[np.asarray([row["base_name"] == group for row in rows], dtype=bool)]
            for group in group_order
        ]
        box = ax_response.boxplot(
            response_groups,
            positions=[3, 2, 1],
            orientation="horizontal",
            widths=0.44,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=COLORS["blue"], edgecolor=COLORS["blue"], alpha=0.20, linewidth=1.8)
        for artist in (*box["whiskers"], *box["caps"]):
            artist.set(color=COLORS["blue"], linewidth=1.8)
        for artist in box["medians"]:
            artist.set(color=COLORS["ink"], linewidth=2.4)
        ax_response.set_xscale("log")
        ax_response.set_xlim(1.0e15, 1.3e16)
        ax_response.set_xticks([1.0e15, 3.0e15, 1.0e16])
        ax_response.xaxis.set_major_formatter(LogFormatterMathtext())
        ax_response.set_yticks([3, 2, 1], group_labels)
        ax_response.set_ylim(0.45, 3.55)
        ax_response.set_xlabel(r"casewise plasma $n_e$ P95  (m$^{-3}$)")
        ax_response.set_title("Response coverage by geometry", fontsize=19.5, weight="bold", loc="left", pad=10.0)
        ax_response.grid(axis="x", color=COLORS["line"], linewidth=0.8, alpha=0.75)
        ax_response.text(0.99, 0.96, "box = middle 50%  ·  whiskers = full range", transform=ax_response.transAxes, ha="right", va="top", fontsize=11.8, color=COLORS["muted"])
        ax_response.text(0.02, 0.05, f"Overall span: {p95.min():.2e} – {p95.max():.2e}  (×{p95.max()/p95.min():.1f})", transform=ax_response.transAxes, fontsize=13.0, color=COLORS["ink"], weight="bold")

    missing = [
        {"Td": 0.16, "base_name": "base3", "PP0": 5.0, "PA": 0.05, "gamma": gamma}
        for gamma in (0.04, 0.07, 0.10)
    ]
    metadata = {
        "figure": "GEC-CCP current training dataset composition and diversity",
        "presentation_revision": "conference_at_a_glance_v2",
        "visual_summary": {
            "condition_encoding": "total count per condition level; split-level tile counts intentionally omitted",
            "response_encoding": "geometry-group boxplot; box is Q25-Q75, median is black, whiskers are min-max",
        },
        "scope": "current gec_ccp_nn_operator_comparison_v1 n78 primary interp split",
        "n_cases": len(rows),
        "condition_counts": {
            "geometry_Td": {f"{geometry_for_td[value]} / {value:.2f}": counts["Td"][value] for value in sorted(counts["Td"])},
            "PP0": {str(value): count for value, count in sorted(counts["PP0"].items())},
            "PA": {str(value): count for value, count in sorted(counts["PA"].items())},
            "gamma": {str(value): count for value, count in sorted(counts["gamma"].items())},
        },
        "condition_split_counts": {
            key: {
                str(value): entry
                for value, entry in sorted(value_counts.items())
            }
            for key, value_counts in counts_by_split.items()
        },
        "factorial_coverage": {"present": 78, "candidate": 81, "fraction": 78 / 81, "missing": missing},
        "split": {"name": "interp", "seed": 7, "counts": split_counts, "ratios_requested": [0.70, 0.15, 0.15]},
        "ne_casewise_plasma_p95_m3": {
            "min": float(np.min(p95)),
            "median": float(np.median(p95)),
            "max": float(np.max(p95)),
            "max_over_min": float(np.max(p95) / np.min(p95)),
        },
        "notes": [
            "Td and geometry are coupled; the design is not a four-axis independent factorial.",
            "The split is condition-based, not stratified by electron-density response.",
            "Repeated models and random seeds reuse the same 78 physical simulations.",
        ],
        "sources": {"index": _rel(CCP_INDEX), "split": _rel(CCP_SPLIT)},
    }
    return fig, metadata


def create_icp_dataset_composition() -> tuple[plt.Figure, dict[str, Any]]:
    rows = _read_rows(ICP_INDEX)
    split = _read_json(ICP_SPLIT)
    split_meta = _read_json(ICP_SPLIT_META)
    lookup = _split_lookup(split)
    if len(rows) != 360 or set(lookup) != {row["case_id"] for row in rows}:
        raise ValueError("ICP current-run dataset/split contract mismatch")
    if not bool(split_meta.get("is_real_structure_holdout")):
        raise ValueError("ICP split is not a verified structure holdout")
    p95 = _plasma_ne_p95(ICP_ROOT, rows)

    geometry_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        geometry_rows.setdefault(row["base_case_id"], row)
    geometry = list(geometry_rows.values())
    split_counts = {name: len(split[name]) for name in ("train", "val", "test")}
    split_group_counts = {
        name: len({case_id.rsplit("_op", 1)[0] for case_id in split[name]})
        for name in ("train", "val", "test")
    }

    parameter_specs = [
        ("$l_c$", "llcoil", geometry, 0.5, 1.5, False, "0.51 – 1.49 cm"),
        ("$r_c$", "rrc", geometry, 2.0, 10.0, False, "2.02 – 9.92 cm"),
        ("$N_c$", "nncoil", geometry, 2.0, 6.0, False, "2 – 6  ·  12 shapes each"),
        ("$r_{ce}$", "rrce", geometry, 20.0, 30.0, False, "20.06 – 29.85 cm"),
        ("$z_c$", "zzc", geometry, 0.0, 5.0, False, "0.07 – 4.92 cm"),
        ("$p_p$", "pp", rows, 500.0, 3000.0, False, "506 – 3000  ·  360 values"),
        ("$p_{p0}$", "pp0", rows, 0.002, 0.1, True, "0.0030 – 0.0998  ·  360 values"),
    ]

    with plt.rc_context(simple.STYLE):
        fig = plt.figure(figsize=(16.0, 9.0))
        grid = fig.add_gridspec(
            2,
            2,
            left=0.06,
            right=0.97,
            bottom=0.13,
            top=0.83,
            width_ratios=(1.0, 1.06),
            height_ratios=(0.40, 0.60),
            wspace=0.20,
            hspace=0.34,
        )
        ax_design = fig.add_subplot(grid[0, 0])
        ax_split = fig.add_subplot(grid[0, 1])
        ax_inputs = fig.add_subplot(grid[1, 0])
        ax_response = fig.add_subplot(grid[1, 1])
        fig.suptitle("GEC-ICP dataset at a glance", fontsize=26.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, "Current structure-held-out training population", ha="center", fontsize=17.0, color=COLORS["muted"])

        ax_design.set_axis_off()
        ax_design.text(0.0, 1.06, "Dataset design", transform=ax_design.transAxes, fontsize=19.5, weight="bold", va="bottom", clip_on=False)
        _rounded_box(ax_design, 0.00, 0.05, 0.98, 0.77, edge=COLORS["blue"], face=COLORS["panel"], linewidth=1.8, radius=0.025)
        ax_design.text(0.49, 0.56, "60  ×  6  =  360", transform=ax_design.transAxes, ha="center", va="center", fontsize=32.0, weight="bold", color=COLORS["blue"])
        ax_design.text(0.49, 0.35, "geometries     operating points     simulations", transform=ax_design.transAxes, ha="center", fontsize=14.0, weight="bold")
        ax_design.text(0.49, 0.17, "Each geometry contributes the same 6 operating cases", transform=ax_design.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])

        ax_split.set_title("Structure-held-out split", fontsize=19.5, weight="bold", loc="left", pad=10.0)
        case_axis = ax_split.inset_axes([0.0, 0.48, 1.0, 0.42])
        group_axis = ax_split.inset_axes([0.0, 0.03, 1.0, 0.36])
        _draw_split_bar(case_axis, split_counts, total=360, y=0.43, height=0.40, show_labels=True)
        _draw_split_bar(group_axis, split_group_counts, total=60, y=0.50, height=0.46, show_labels=False)
        case_axis.text(-0.01, 0.98, "simulations", transform=case_axis.transAxes, fontsize=12.5, weight="bold", va="top")
        group_axis.text(-0.01, 0.98, "geometries", transform=group_axis.transAxes, fontsize=12.5, weight="bold", va="top")
        ax_split.text(0.50, -0.03, "No geometry appears in more than one partition", transform=ax_split.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])
        ax_split.set_axis_off()

        ax_inputs.set_axis_off()
        ax_inputs.set_title("Input coverage", loc="left", fontsize=19.5, weight="bold", pad=10.0)
        _rounded_box(ax_inputs, 0.00, 0.39, 0.98, 0.49, edge=COLORS["line"], face="#FFFFFF", linewidth=1.3, radius=0.022)
        ax_inputs.text(0.04, 0.80, "60 geometry designs", transform=ax_inputs.transAxes, fontsize=15.5, weight="bold", color=COLORS["blue"])
        ax_inputs.text(0.04, 0.68, r"$l_c$   0.51–1.49 cm     ·     $r_c$   2.02–9.92 cm", transform=ax_inputs.transAxes, fontsize=14.2)
        ax_inputs.text(0.04, 0.56, r"$N_c$   2–6  (12 shapes each)     ·     $r_{ce}$   20.06–29.85 cm", transform=ax_inputs.transAxes, fontsize=13.5)
        ax_inputs.text(0.04, 0.44, r"$z_c$   0.07–4.92 cm", transform=ax_inputs.transAxes, fontsize=14.2)
        _rounded_box(ax_inputs, 0.00, 0.03, 0.98, 0.27, edge=COLORS["line"], face=COLORS["panel"], linewidth=1.3, radius=0.022)
        ax_inputs.text(0.04, 0.23, "360 unique operating-input pairs", transform=ax_inputs.transAxes, fontsize=15.0, weight="bold", color=COLORS["blue"])
        ax_inputs.text(0.04, 0.11, r"$p_p$   506–3000     ·     $p_{p0}$   0.0030–0.0998     (dataset-defined units)", transform=ax_inputs.transAxes, fontsize=14.0)

        response_summaries: dict[str, dict[str, float]] = {}
        for y, name in zip((3, 2, 1), ("train", "val", "test"), strict=True):
            mask = np.asarray([lookup[row["case_id"]] == name for row in rows], dtype=bool)
            values = p95[mask]
            q05, median, q95 = np.quantile(values, [0.05, 0.50, 0.95])
            response_summaries[name] = {"q05": float(q05), "median": float(median), "q95": float(q95)}
            ax_response.plot([q05, q95], [y, y], color=SPLIT_COLORS[name], linewidth=10.0, alpha=0.30, solid_capstyle="round")
            ax_response.plot([q05, q95], [y, y], color=SPLIT_COLORS[name], linewidth=2.0)
            ax_response.scatter([median], [y], s=105, color=SPLIT_COLORS[name], edgecolors="white", linewidths=1.1, zorder=3)
        overall_q05, overall_q95 = np.quantile(p95, [0.05, 0.95])
        ax_response.set_xscale("log")
        ax_response.set_xlim(1.0e15, 1.0e19)
        ax_response.set_xticks([1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19])
        ax_response.xaxis.set_major_formatter(LogFormatterMathtext())
        ax_response.set_yticks([3, 2, 1], ["Train", "Validation", "Test"])
        ax_response.set_ylim(0.25, 3.55)
        ax_response.set_xlabel(r"casewise plasma $n_e$ P95  (m$^{-3}$)")
        ax_response.set_title("Density coverage by split", fontsize=19.5, weight="bold", loc="left", pad=10.0)
        ax_response.grid(axis="x", which="major", color=COLORS["line"], linewidth=0.8, alpha=0.72)
        ax_response.text(0.99, 0.96, "line = P5–P95  ·  dot = median", transform=ax_response.transAxes, ha="right", va="top", fontsize=11.8, color=COLORS["muted"])
        ax_response.text(0.02, 0.15, f"Overall central 90%: {overall_q05:.2e} – {overall_q95:.2e}  (×{overall_q95/overall_q05:.0f})", transform=ax_response.transAxes, fontsize=12.6, color=COLORS["ink"], weight="bold")
        ax_response.text(0.02, 0.065, r"1 finite near-floor Test case: $2.79\times10^8\ \mathrm{m^{-3}}$  (separate from main scale)", transform=ax_response.transAxes, fontsize=11.5, color=COLORS["green"])

    parameter_summary: dict[str, Any] = {}
    parameter_split_summary: dict[str, Any] = {}
    for _, key, source_rows, _, _, _, _ in parameter_specs:
        values = np.asarray([float(row[key]) for row in source_rows], dtype=float)
        parameter_summary[key] = {
            "count": int(values.size),
            "unique": int(np.unique(values).size),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        split_entry: dict[str, Any] = {}
        for name in ("train", "val", "test"):
            split_values = np.asarray(
                [float(row[key]) for row in source_rows if lookup[row["case_id"]] == name],
                dtype=float,
            )
            entry: dict[str, Any] = {
                "count": int(split_values.size),
                "unique": int(np.unique(split_values).size),
                "min": float(np.min(split_values)),
                "max": float(np.max(split_values)),
            }
            unique_values = np.unique(split_values)
            if unique_values.size <= 10:
                entry["values"] = unique_values.tolist()
            split_entry[name] = entry
        parameter_split_summary[key] = split_entry
    metadata = {
        "figure": "GEC-ICP current training dataset composition and diversity",
        "presentation_revision": "conference_at_a_glance_v2",
        "visual_summary": {
            "input_encoding": "range summary; individual parameter scatter intentionally omitted",
            "response_encoding": "split-wise P5-P95 line and median dot; finite near-floor case called out separately",
            "response_by_split": response_summaries,
        },
        "scope": "trained icp_stage4_coil_structure_v1 structure-holdout contract",
        "n_cases": len(rows),
        "n_geometries": len(geometry),
        "operating_points_per_geometry": 6,
        "parameters": parameter_summary,
        "parameters_by_split": parameter_split_summary,
        "split": {
            "name": "structure_holdout",
            "seed": 7,
            "case_counts": split_counts,
            "geometry_counts": split_group_counts,
            "group_key": "base_case_id",
            "group_overlap": 0,
        },
        "ne_casewise_plasma_p95_m3": {
            "min": float(np.min(p95)),
            "q05": float(np.quantile(p95, 0.05)),
            "median": float(np.median(p95)),
            "q95": float(np.quantile(p95, 0.95)),
            "max": float(np.max(p95)),
            "minimum_case_id": rows[int(np.argmin(p95))]["case_id"],
            "display_main_range_m3": [1.0e15, 1.0e19],
            "minimum_is_callout_outside_main_scale": True,
        },
        "notes": [
            "case_g002 is one train geometry and is not the dataset-wide quality scope.",
            "Validation and test are held-out shapes but are not balanced by coil count.",
            "The minimum response is a finite near-floor solution and is shown, not discarded.",
            "The trained targets are ne, ni, Te, and phi (four fields).",
        ],
        "sources": {"index": _rel(ICP_INDEX), "split": _rel(ICP_SPLIT), "split_meta": _rel(ICP_SPLIT_META), "resolved_config": _rel(ICP_CONFIG)},
    }
    return fig, metadata


def _flow_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": COLORS["blue"], "mutation_scale": 18},
    )


def create_ccp_zscore_calculation() -> tuple[plt.Figure, dict[str, Any]]:
    rows = _read_rows(CCP_INDEX)
    split = _read_json(CCP_SPLIT)
    scalers = _read_json(CCP_SCALERS)
    mu = float(scalers["ne"]["mean"][0])
    sigma = float(scalers["ne"]["std"][0])
    example = 3.0e15
    example_z = (example - mu) / sigma
    fit_pixels = _plasma_fit_pixel_count(CCP_ROOT, rows, set(split["train"]))

    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=(16.0, 9.0))
        ax.set_axis_off()
        fig.suptitle("Linear Z-score: calculation sheet", fontsize=27.0, weight="bold", y=0.955)
        fig.text(0.5, 0.885, r"GEC-CCP electron density $n_e$  ·  current interp-train statistics", ha="center", fontsize=17.0, color=COLORS["muted"])

        boxes = [(0.04, COLORS["blue"], "1  Fit on Train"), (0.365, COLORS["orange"], "2  Standardize"), (0.69, COLORS["green"], "3  Return to physics")]
        for x, color, title in boxes:
            _rounded_box(ax, x, 0.22, 0.27, 0.55, edge=color, face="#FFFFFF", linewidth=2.0, radius=0.025)
            ax.text(x + 0.135, 0.715, title, transform=ax.transAxes, ha="center", fontsize=19.0, weight="bold", color=color)

        ax.text(0.175, 0.595, r"$\mu=\dfrac{\sum m_i n_{e,i}}{\sum m_i}$", transform=ax.transAxes, ha="center", fontsize=20.5)
        ax.text(0.175, 0.485, r"$\sigma=\sqrt{\dfrac{\sum m_i(n_{e,i}-\mu)^2}{\sum m_i}}$", transform=ax.transAxes, ha="center", fontsize=17.5)
        ax.text(0.175, 0.370, rf"$\mu={mu/1e15:.4f}\times10^{{15}}\ \mathrm{{m^{{-3}}}}$", transform=ax.transAxes, ha="center", fontsize=15.8, weight="bold")
        ax.text(0.175, 0.315, rf"$\sigma={sigma/1e15:.4f}\times10^{{15}}\ \mathrm{{m^{{-3}}}}$", transform=ax.transAxes, ha="center", fontsize=15.3)
        ax.text(0.175, 0.270, r"$m_i=1$ in plasma, $0$ outside", transform=ax.transAxes, ha="center", fontsize=11.5, color=COLORS["muted"])
        ax.text(0.175, 0.235, f"54 cases · {fit_pixels/1e6:.2f} M plasma cells", transform=ax.transAxes, ha="center", fontsize=11.5, color=COLORS["muted"])

        ax.text(0.50, 0.58, r"$z_e=\dfrac{n_e-\mu}{\sigma}$", transform=ax.transAxes, ha="center", fontsize=30.0, weight="bold")
        ax.text(0.50, 0.455, rf"$n_e=3.00\times10^{{15}}$", transform=ax.transAxes, ha="center", fontsize=17.0)
        ax.text(0.50, 0.385, rf"$z_e=\dfrac{{3.00-1.2915}}{{1.8713}}={example_z:.3f}$", transform=ax.transAxes, ha="center", fontsize=16.3, color=COLORS["orange"], weight="bold")
        ax.text(0.50, 0.325, r"numbers shown in $10^{15}\ \mathrm{m^{-3}}$", transform=ax.transAxes, ha="center", fontsize=11.7, color=COLORS["muted"])
        ax.text(0.50, 0.265, "same ordering · new center and scale", transform=ax.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])

        ax.text(0.825, 0.58, r"$\widehat n_e=\mu+\sigma\widehat z_e$", transform=ax.transAxes, ha="center", fontsize=27.0, weight="bold")
        ax.text(0.825, 0.455, r"model prediction $\widehat{z}_e$", transform=ax.transAxes, ha="center", fontsize=17.5)
        ax.text(0.825, 0.375, r"$\Downarrow$", transform=ax.transAxes, ha="center", fontsize=25.0, color=COLORS["green"])
        ax.text(0.825, 0.315, r"physical $\widehat{n}_e$  (m$^{-3}$)", transform=ax.transAxes, ha="center", fontsize=18.0, weight="bold")

        _flow_arrow(ax, 0.315, 0.355, 0.50)
        _flow_arrow(ax, 0.64, 0.68, 0.50)
        _rounded_box(ax, 0.15, 0.105, 0.70, 0.075, edge=COLORS["line"], face=COLORS["panel"], linewidth=1.0, radius=0.016)
        ax.text(0.5, 0.142, "identity physical values  ·  train only  ·  plasma only  ·  no clipping  ·  population std (ddof = 0)", transform=ax.transAxes, ha="center", va="center", fontsize=14.0, color=COLORS["muted"])
        ax.text(0.5, 0.055, "Z-score standardization does not make a skewed distribution Gaussian.", transform=ax.transAxes, ha="center", fontsize=14.0, color=COLORS["ink"], weight="bold")

    metadata = {
        "figure": "GEC-CCP electron-density linear Z-score calculation sheet",
        "fit_split": "interp train only",
        "fit_case_count": len(split["train"]),
        "fit_region": "plasma_only",
        "fit_pixel_count": fit_pixels,
        "value_transform": "identity",
        "clip": "none",
        "std_ddof": 0,
        "mean_m3": mu,
        "std_m3": sigma,
        "forward_formula": "z=(ne-mu)/sigma",
        "inverse_formula": "ne=mu+sigma*z",
        "example": {"ne_m3": example, "z": example_z},
        "sources": {"scalers": _rel(CCP_SCALERS), "fit_policy": _rel(CCP_FIT_POLICY), "split": _rel(CCP_SPLIT)},
    }
    return fig, metadata


def _loss_step_box(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str, title: str) -> None:
    _rounded_box(ax, x, y, w, h, edge=color, face="#FFFFFF", linewidth=1.8, radius=0.022)
    ax.text(x + 0.025, y + h - 0.055, title, transform=ax.transAxes, fontsize=18.0, weight="bold", color=color, va="center")


def create_ccp_loss_calculation() -> tuple[plt.Figure, dict[str, Any]]:
    contract = loss_detail.build_ccp_metadata(output_stem="gec_ccp_loss_calculation")
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=(16.0, 9.0))
        ax.set_axis_off()
        fig.suptitle("GEC-CCP loss: calculation sheet", fontsize=27.0, weight="bold", y=0.955)
        fig.text(0.5, 0.89, "All residuals are computed after train-only plasma Z-score", ha="center", fontsize=16.5, color=COLORS["muted"])

        _loss_step_box(ax, 0.04, 0.45, 0.29, 0.35, COLORS["blue"], "1  Residual → Huber")
        ax.text(0.185, 0.665, r"$e_t=\widehat{\widetilde y}_t-\widetilde y_t$", transform=ax.transAxes, ha="center", fontsize=23.0)
        ax.text(0.185, 0.575, r"$H_1(e)=\frac{1}{2}e^2 \quad (|e|\leq1)$", transform=ax.transAxes, ha="center", fontsize=17.5)
        ax.text(0.185, 0.505, r"$H_1(e)=|e|-\frac{1}{2} \quad (|e|>1)$", transform=ax.transAxes, ha="center", fontsize=17.5)

        _loss_step_box(ax, 0.37, 0.45, 0.59, 0.35, COLORS["orange"], "2  Per-target spatial loss")
        ax.text(0.665, 0.675, r"$\ell_t=L_{point}+0.25L_{boundary}$", transform=ax.transAxes, ha="center", fontsize=22.0, weight="bold")
        ax.text(0.665, 0.615, r"$\quad+0.10L_{gradient}+0.05L_{multi}$", transform=ax.transAxes, ha="center", fontsize=22.0, weight="bold")
        components = [
            (0.405, COLORS["blue"], "point", r"$\langle H_1(e)\rangle_{\Omega_p}$"),
            (0.545, COLORS["orange"], "boundary", r"$\langle H_1(e)\rangle_{D\leq2\,px}$"),
            (0.685, COLORS["green"], "gradient", r"$\mathrm{mean}_{r,z}\,H_1(\Delta e)$"),
            (0.825, COLORS["purple"], "multi", r"$\mathrm{mean}_{s=2,4}\,H_1(A_s e)$"),
        ]
        for x, color, heading, detail in components:
            _rounded_box(ax, x, 0.49, 0.115, 0.105, edge=color, face=COLORS["panel"], linewidth=1.1, radius=0.012)
            ax.text(x + 0.0575, 0.555, heading, transform=ax.transAxes, ha="center", fontsize=13.0, weight="bold", color=color)
            ax.text(x + 0.0575, 0.512, detail, transform=ax.transAxes, ha="center", fontsize=11.2, color=COLORS["muted"])

        _flow_arrow(ax, 0.335, 0.36, 0.625)
        _loss_step_box(ax, 0.18, 0.11, 0.64, 0.25, COLORS["green"], "3  Equal weight by physical target family")
        ax.text(0.50, 0.215, r"$L_{CCP}=\dfrac{1}{3}\left[\dfrac{\ell_{n_e}+\ell_{n_i}}{2}+\ell_{T_e}+\ell_{\phi}\right]$", transform=ax.transAxes, ha="center", fontsize=25.5, weight="bold")
        ax.text(0.50, 0.145, r"effective weights: $n_e=1/6,\ n_i=1/6,\ T_e=1/3,\ \phi=1/3$", transform=ax.transAxes, ha="center", fontsize=13.2, color=COLORS["muted"])
        ax.text(0.5, 0.06, "casewise masked mean  ·  plasma only  ·  boundary band uses nearest chamber/part distance  ·  Physics term disabled (weight 0)", transform=ax.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])

    metadata = dict(contract)
    metadata.update(
        {
            "figure": "GEC-CCP loss calculation sheet",
            "presentation_revision": "three_step_calculation_sheet_v1",
            "effective_target_weights": {"ne": 1 / 6, "ni": 1 / 6, "Te": 1 / 3, "phi": 1 / 3},
            "explicit_huber_piecewise": True,
        }
    )
    return fig, metadata


def create_icp_loss_calculation() -> tuple[plt.Figure, dict[str, Any]]:
    contract = loss_detail.build_icp_metadata(output_stem="gec_icp_loss_calculation")
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=(16.0, 9.0))
        ax.set_axis_off()
        fig.suptitle("GEC-ICP trained loss: calculation sheet", fontsize=27.0, weight="bold", y=0.955)
        fig.text(0.5, 0.89, "Executed part_source_v1 baseline · transformed target space", ha="center", fontsize=16.5, color=COLORS["muted"])

        _loss_step_box(ax, 0.04, 0.45, 0.29, 0.35, COLORS["blue"], "1  Residual → Huber")
        ax.text(0.185, 0.665, r"$e_t=\widehat{\widetilde y}_t-\widetilde y_t$", transform=ax.transAxes, ha="center", fontsize=23.0)
        ax.text(0.185, 0.575, r"$H_1(e)=\frac{1}{2}e^2 \quad (|e|\leq1)$", transform=ax.transAxes, ha="center", fontsize=17.5)
        ax.text(0.185, 0.505, r"$H_1(e)=|e|-\frac{1}{2} \quad (|e|>1)$", transform=ax.transAxes, ha="center", fontsize=17.5)

        _loss_step_box(ax, 0.37, 0.45, 0.59, 0.35, COLORS["orange"], "2  Per-target spatial loss")
        ax.text(0.665, 0.665, r"$\ell_t=L_{point}+0.02L_{gradient}^{norm}+0.05L_{multi}$", transform=ax.transAxes, ha="center", fontsize=25.0, weight="bold")
        ax.text(0.665, 0.575, r"$s_{b,t,q}=\max\!\left(\mathrm{RMS}_{\Omega_p}(\Delta_q\widetilde y_{b,t}),\ 0.05\right)$", transform=ax.transAxes, ha="center", fontsize=17.0)
        ax.text(0.665, 0.515, r"$L_{point}=\langle H_1(e)\rangle_{\Omega_p}$   ·   $L_{gradient}^{norm}=\mathrm{mean}_{q=r,z}\langle H_1(\Delta_q e/s_{b,t,q})\rangle$", transform=ax.transAxes, ha="center", fontsize=11.9, color=COLORS["ink"])
        ax.text(0.665, 0.468, r"$L_{multi}=\mathrm{mean}_{s=2,4}\langle H_1(A_s e)\rangle$   ·   $s_{b,t,q}$ is per case / target / direction", transform=ax.transAxes, ha="center", fontsize=11.9, color=COLORS["muted"])
        _flow_arrow(ax, 0.335, 0.36, 0.625)

        _loss_step_box(ax, 0.18, 0.14, 0.64, 0.21, COLORS["green"], "3  Target weighting")
        ax.text(0.50, 0.235, r"$L_{ICP}=1.0\ell_{n_e}+1.0\ell_{n_i}+1.2\ell_{T_e}+0.5\ell_{\phi}$", transform=ax.transAxes, ha="center", fontsize=27.0, weight="bold")
        ax.text(0.50, 0.17, "weights are not divided by 3.7", transform=ax.transAxes, ha="center", fontsize=14.0, color=COLORS["muted"])
        ax.text(0.5, 0.083, r"$n_e,n_i$: log$_{10}$ + Z-score   ·   $T_e$: log1p + robust   ·   $\phi$: signed-log1p + robust", transform=ax.transAxes, ha="center", fontsize=13.4, color=COLORS["ink"], weight="bold")
        ax.text(0.5, 0.045, "structure-holdout Train only  ·  casewise plasma mean  ·  boundary term disabled  ·  Physics term disabled", transform=ax.transAxes, ha="center", fontsize=12.5, color=COLORS["muted"])

    metadata = dict(contract)
    metadata.update(
        {
            "figure": "GEC-ICP trained baseline loss calculation sheet",
            "presentation_revision": "three_step_calculation_sheet_v1",
            "explicit_huber_piecewise": True,
            "target_weight_sum_normalization": False,
            "comparison_contract_note": "This is the executed baseline, not the untrained SDF-versus-dimension configuration.",
        }
    )
    return fig, metadata


Builder = Callable[[], tuple[plt.Figure, dict[str, Any]]]
BUILDERS: dict[str, tuple[str, Builder]] = {
    "ccp_zscore_calc": ("ccp_zscore_calculation", create_ccp_zscore_calculation),
    "ccp_loss_calc": ("gec_ccp_loss_calculation", create_ccp_loss_calculation),
    "icp_loss_calc": ("gec_icp_loss_calculation", create_icp_loss_calculation),
}


def generate(
    *,
    out_dir: Path,
    figures: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    selected = list(BUILDERS) if "all" in figures else list(dict.fromkeys(figures))
    outputs: list[Path] = []
    for name in selected:
        stem, builder = BUILDERS[name]
        figure, metadata = builder()
        try:
            figure_outputs = _save_slide(figure, out_dir=out_dir, stem=stem, formats=formats, dpi=dpi)
        finally:
            plt.close(figure)
        metadata["output_files"] = [path.name for path in figure_outputs]
        outputs.extend(figure_outputs)
        outputs.append(simple._write_metadata(out_dir, stem, metadata))
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=FORMATS)
    parser.add_argument("--figures", nargs="+", choices=("all", *BUILDERS), default=("all",))
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in generate(out_dir=args.out_dir, figures=args.figures, formats=args.formats, dpi=args.dpi):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
