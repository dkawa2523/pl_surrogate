"""Create simplified, slide-ready GEC-CCP and GEC-ICP conference figures.

Each figure communicates one main point with large type and at most three
visual panels.  The detailed audit/generation scripts remain the numerical
source of truth; this module is the deliberately minimal presentation layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, PercentFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.conference.scripts import plot_gec_data_and_features as detail  # noqa: E402
from experiments.conference.scripts import plot_gec_loss_design as loss_detail  # noqa: E402
from experiments.gec_ccp.scripts.plot_gec_ccp_geometry_overview import (  # noqa: E402
    DEFAULT_MANIFEST as CCP_GEOMETRY_MANIFEST,
    DEFAULT_SOURCE_ROOT as CCP_GEOMETRY_SOURCE_ROOT,
    DEFAULT_STRUCTURE_INDEX as CCP_GEOMETRY_STRUCTURE_INDEX,
    ROLE_ENTITY_IDS as CCP_ROLE_ENTITY_IDS,
    load_geometry as load_ccp_geometry,
)


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials")
FORMATS = ("png", "pdf", "svg")

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#182230",
    "muted": "#667085",
    "line": "#D0D5DD",
    "panel": "#F8FAFC",
    "mask": "#EEF1F4",
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14.0,
    "axes.titlesize": 20.0,
    "axes.labelsize": 17.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
    "axes.edgecolor": "#98A2B3",
    "axes.linewidth": 1.0,
    "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"],
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_metadata(out_dir: Path, stem: str, payload: dict[str, Any]) -> Path:
    path = out_dir / f"{stem}_metadata.json"
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _save(
    fig: plt.Figure,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with plt.rc_context(STYLE):
        for suffix in dict.fromkeys(formats):
            path = out_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
            outputs.append(path)
    return outputs


def _masked_cmap(name: str) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(COLORS["mask"])
    return cmap


def _rounded_log_limits(values: np.ndarray) -> tuple[float, float]:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("electron-density display has no positive finite values")
    low_reference = float(np.quantile(positive, 0.01))
    high_reference = float(np.max(positive))
    vmin = 10.0 ** math.floor(math.log10(low_reference))
    vmax = 10.0 ** math.ceil(math.log10(high_reference))
    if not vmin < vmax:
        vmax = 10.0 * vmin
    return vmin, vmax


def _log_ticks(vmin: float, vmax: float) -> np.ndarray:
    lo = int(round(math.log10(vmin)))
    hi = int(round(math.log10(vmax)))
    exponents = np.arange(lo, hi + 1, dtype=np.float64)
    if exponents.size > 4:
        exponents = np.unique(np.concatenate((exponents[::2], exponents[-1:])))
    return np.power(10.0, exponents)


def _overlay_ccp_simple(ax: plt.Axes, geometry: Any) -> None:
    styles = {
        "grounded_electrode_and_walls": (COLORS["ink"], 1.6, "solid"),
        "driven_electrode": (COLORS["red"], 3.0, "solid"),
        "dielectric_contact": (COLORS["green"], 3.0, "solid"),
        "axis_of_symmetry": (COLORS["ink"], 1.2, (0, (4.0, 3.0))),
    }
    for role, entity_ids in CCP_ROLE_ENTITY_IDS.items():
        color, linewidth, linestyle = styles[role]
        for entity_id in entity_ids:
            segment = geometry.boundary(entity_id)
            ax.plot(
                [100.0 * segment.r0, 100.0 * segment.r1],
                [100.0 * segment.z0, 100.0 * segment.z1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                solid_capstyle="butt",
                dash_capstyle="butt",
                zorder=5,
            )


def _overlay_icp_simple(ax: plt.Axes, coils: Sequence[dict[str, float | int]]) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, facecolor="none", edgecolor=COLORS["ink"], linewidth=1.6, zorder=5))
    ax.add_patch(Rectangle((0.0, 0.0), 15.0, 2.0, facecolor="none", edgecolor=COLORS["ink"], linewidth=1.4, zorder=5))
    ax.add_patch(Rectangle((15.0, 0.0), 5.0, 2.0, facecolor="none", edgecolor=COLORS["green"], linewidth=2.0, zorder=5))
    ax.add_patch(Rectangle((0.0, 12.0), 30.0, 2.0, facecolor="none", edgecolor=COLORS["green"], linewidth=2.0, zorder=5))
    for coil in coils:
        ax.add_patch(
            Rectangle(
                (float(coil["r_min"]), float(coil["z_min"])),
                float(coil["r_max"]) - float(coil["r_min"]),
                float(coil["z_max"]) - float(coil["z_min"]),
                facecolor="none",
                edgecolor=COLORS["orange"],
                linewidth=2.0,
                zorder=6,
            )
        )
    ax.plot([0.0, 0.0], [0.0, 22.0], color=COLORS["ink"], linewidth=1.2, linestyle=(0, (4.0, 3.0)), zorder=7)


def _physical_field_metadata(
    *,
    system: str,
    case_id: str,
    source_field: Path,
    source_structure: Path,
    mask: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    return {
        "figure": f"Simplified {system} representative electron-density distribution",
        "presentation_revision": "single_field_large_type_v2",
        "case_id": case_id,
        "shown_field": "ne",
        "physical_unit": "m^-3",
        "source_fields_npz": source_field,
        "source_structure_npz": source_structure,
        "statistics_inside_plasma": detail._finite_masked_stats(values, mask),
        "display": {
            "normalization": "logarithmic color scale on physical ne values",
            "vmin": vmin,
            "vmax": vmax,
            "values_below_vmin": "saturated and marked by the lower colorbar extension",
            "vmax_policy": "rounded upward to a full decade; no high-end saturation",
            "colormap": "viridis",
            "outside_plasma": "light gray",
            "labels": "large-font slide layout; no case footer or legend",
        },
    }


def create_ccp_field_simple() -> tuple[plt.Figure, dict[str, Any]]:
    fields, structure, row = detail._load_ccp_case()
    geometry = load_ccp_geometry(
        base_name=row["base_name"],
        source_root=CCP_GEOMETRY_SOURCE_ROOT,
        structure_index_path=CCP_GEOMETRY_STRUCTURE_INDEX,
        manifest_path=CCP_GEOMETRY_MANIFEST,
    )
    mask = structure["mask_plasma"]
    values = fields["ne"]
    vmin, vmax = _rounded_log_limits(values[mask])
    r_cm = 100.0 * structure["r_coords"]
    z_cm = 100.0 * structure["z_coords"]
    shown = np.ma.array(values, mask=(~mask) | (~np.isfinite(values)) | (values <= 0.0))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 7.2))
        image = ax.imshow(
            shown,
            origin="lower",
            extent=[float(r_cm[0]), float(r_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            interpolation="nearest",
            cmap=_masked_cmap("viridis"),
            norm=LogNorm(vmin=vmin, vmax=vmax),
            aspect="equal",
            rasterized=True,
        )
        _overlay_ccp_simple(ax, geometry)
        ax.set_xlim(-0.05, 100.0 * geometry.dimensions.outer_radius + 0.05)
        ax.set_ylim(100.0 * geometry.dimensions.chamber_bottom - 0.05, 100.0 * geometry.dimensions.chamber_top + 0.05)
        ax.set_xticks(np.arange(0.0, 10.1, 2.0))
        ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
        ax.set_xlabel(r"$r$ (cm)")
        ax.set_ylabel(r"$z$ (cm)")
        ax.set_title("GEC-CCP   Electron density", loc="left", pad=10.0, weight="bold")
        ax.set_facecolor(COLORS["mask"])
        ax.set_aspect("equal", adjustable="box")
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.052,
            pad=0.035,
            extend="min",
            ticks=_log_ticks(vmin, vmax),
            format=LogFormatterMathtext(),
        )
        colorbar.set_label(r"$n_e$ (m$^{-3}$)", fontsize=17.0, labelpad=10.0)
        colorbar.ax.tick_params(labelsize=15.0, length=3.5)
        colorbar.outline.set_linewidth(0.8)
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.92)

    metadata = _physical_field_metadata(
        system="GEC-CCP",
        case_id=detail.CCP_CASE_ID,
        source_field=detail.CCP_DATA_ROOT / row["fields_npz"],
        source_structure=detail.CCP_DATA_ROOT / row["structure_npz"],
        mask=mask,
        values=values,
        vmin=vmin,
        vmax=vmax,
    )
    metadata["geometry_alternative"] = row["base_name"]
    metadata["source_geometry_mphtxt"] = geometry.mphtxt_path
    return fig, metadata


def _ccp_train_ne() -> tuple[np.ndarray, float, float, list[str]]:
    split = json.loads(detail.CCP_SPLIT_PATH.read_text(encoding="utf-8"))
    train_ids = [str(case_id) for case_id in split["train"]]
    rows = {row["case_id"]: row for row in detail._read_csv(detail.CCP_INDEX_PATH)}
    chunks: list[np.ndarray] = []
    for case_id in train_ids:
        row = rows[case_id]
        with np.load(detail.CCP_DATA_ROOT / row["structure_npz"], allow_pickle=False) as structure:
            mask = np.asarray(structure["mask_plasma"]) > 0.5
        with np.load(detail.CCP_DATA_ROOT / row["fields_npz"], allow_pickle=False) as fields:
            selected = np.asarray(fields["ne"], dtype=np.float64)[mask]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"non-finite ne values in {case_id}")
        chunks.append(selected)
    scalers = json.loads(detail.CCP_SCALER_PATH.read_text(encoding="utf-8"))
    mu = float(scalers["ne"]["mean"][0])
    sigma = float(scalers["ne"]["std"][0])
    values = np.concatenate(chunks)
    if not np.isclose(np.mean(values), mu, rtol=5.0e-7):
        raise ValueError("saved ne scaler mean does not match train plasma values")
    if not np.isclose(np.std(values, ddof=0), sigma, rtol=5.0e-7):
        raise ValueError("saved ne scaler std does not match train plasma values")
    return values, mu, sigma, train_ids


def create_ccp_zscore_simple() -> tuple[plt.Figure, dict[str, Any]]:
    values, mu, sigma, train_ids = _ccp_train_ne()
    raw = values / 1.0e15
    standardized = (values - mu) / sigma
    raw_lo, raw_hi = (float(value) for value in np.quantile(raw, [0.001, 0.995]))
    edges_raw = np.linspace(raw_lo, raw_hi, 50)
    edges_z = (edges_raw * 1.0e15 - mu) / sigma
    weights = np.full(raw.shape, 100.0 / raw.size, dtype=np.float64)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
        axes[0].hist(raw, bins=edges_raw, weights=weights, color="#98A2B3", edgecolor="white", linewidth=0.35)
        axes[0].axvspan((mu - sigma) / 1.0e15, (mu + sigma) / 1.0e15, color="#D0D5DD", alpha=0.45, linewidth=0.0)
        axes[0].axvline(mu / 1.0e15, color=COLORS["ink"], linewidth=2.0)
        axes[0].set_xlim(raw_lo, raw_hi)
        axes[0].set_title("Before", weight="bold", pad=8.0)
        axes[0].set_xlabel(r"$n_e$ ($10^{15}$ m$^{-3}$)")
        axes[0].set_ylabel("Plasma samples (%)")

        axes[1].hist(standardized, bins=edges_z, weights=weights, color=COLORS["blue"], edgecolor="white", linewidth=0.35)
        axes[1].axvspan(-1.0, 1.0, color="#DCEAF7", alpha=0.75, linewidth=0.0)
        axes[1].axvline(0.0, color=COLORS["ink"], linewidth=2.0)
        axes[1].set_xlim(float(edges_z[0]), float(edges_z[-1]))
        axes[1].set_title("After", weight="bold", pad=8.0)
        axes[1].set_xlabel(r"standardized electron density  $z_e$")

        for ax in axes:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
            ax.grid(axis="y", color=COLORS["line"], linewidth=0.7, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=4.0, width=0.9)
        fig.suptitle("Linear Z-score for electron density", fontsize=22.0, weight="bold", y=0.98)
        fig.text(
            0.5,
            0.865,
            r"$z_e=\dfrac{n_e-1.29\times10^{15}}{1.87\times10^{15}}$",
            ha="center",
            va="center",
            fontsize=20.0,
        )
        fig.text(0.5, 0.48, r"$\longrightarrow$", ha="center", va="center", fontsize=30.0, color=COLORS["blue"])
        fig.text(
            0.5,
            0.025,
            "Same samples; only center and scale change",
            ha="center",
            va="bottom",
            fontsize=13.5,
            color=COLORS["muted"],
        )
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.74, wspace=0.22)

    metadata = {
        "figure": "Simplified GEC-CCP linear Z-score effect for electron density",
        "presentation_revision": "single_target_two_panel_v2",
        "shown_target": "ne",
        "sample_scope": "train-only plasma cells",
        "train_case_count": len(train_ids),
        "sample_count": int(values.size),
        "value_transform": "identity",
        "clip": "none",
        "mean": mu,
        "std_ddof0": sigma,
        "z_mean": float(np.mean(standardized)),
        "z_std_ddof0": float(np.std(standardized, ddof=0)),
        "interpretation": "The linear Z-score changes location and scale, not distribution shape.",
        "sources": {"split": detail.CCP_SPLIT_PATH, "scaler": detail.CCP_SCALER_PATH},
        "display_quantiles": [0.001, 0.995],
    }
    return fig, metadata


def _metric_card(
    ax: plt.Axes,
    *,
    value: str,
    label: str,
    accent: str,
) -> None:
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.90,
        boxstyle="round,pad=0.014,rounding_size=0.035",
        facecolor="white",
        edgecolor=COLORS["line"],
        linewidth=1.4,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((0.02, 0.04), 0.022, 0.90, transform=ax.transAxes, facecolor=accent, edgecolor="none"))
    ax.text(0.10, 0.59, value, transform=ax.transAxes, ha="left", va="center", fontsize=34.0, weight="bold")
    ax.text(0.10, 0.27, label, transform=ax.transAxes, ha="left", va="center", fontsize=15.5, color=COLORS["muted"])


def _ccp_base4_quality() -> dict[str, Any]:
    rows = [row for row in detail._read_csv(detail.CCP_INDEX_PATH) if row["base_name"] == "base4"]
    if len(rows) != 27:
        raise ValueError(f"expected 27 base4 rows, found {len(rows)}")
    combinations = {(float(row["PP0"]), float(row["gamma"]), float(row["PA"])) for row in rows}
    field_packs = 0
    finite = 0
    total = 0
    with np.load(detail.CCP_DATA_ROOT / rows[0]["structure_npz"], allow_pickle=False) as structure:
        mask = np.asarray(structure["mask_plasma"]) > 0.5
    for row in rows:
        path = detail.CCP_DATA_ROOT / row["fields_npz"]
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as fields:
            valid = True
            for target in detail.TARGETS:
                values = np.asarray(fields[target])[mask]
                finite += int(np.isfinite(values).sum())
                total += int(values.size)
                valid &= values.shape == (int(mask.sum()),)
        field_packs += int(valid)
    return {
        "base_name": "base4",
        "Td": 0.03,
        "case_count": len(rows),
        "unique_operating_combinations": len(combinations),
        "factor_levels": {key: len({float(row[key]) for row in rows}) for key in ("PP0", "gamma", "PA")},
        "field_packs_valid": field_packs,
        "finite_rate": float(finite / total),
        "plasma_cells_per_case": int(mask.sum()),
        "source_index": detail.CCP_INDEX_PATH,
    }


def create_ccp_quality_simple() -> tuple[plt.Figure, dict[str, Any]]:
    metrics = _ccp_base4_quality()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        _metric_card(axes[0], value="27", label="operating cases", accent=COLORS["blue"])
        _metric_card(axes[1], value="27 / 27", label="valid field packs", accent=COLORS["green"])
        _metric_card(axes[2], value="100%", label="finite plasma values", accent=COLORS["orange"])
        fig.suptitle("GEC-CCP representative subset", fontsize=23.0, weight="bold", y=0.98)
        fig.text(0.5, 0.835, r"coverage & integrity   ·   base4 / $T_d=0.03$   ·   complete $3\times3\times3$ grid", ha="center", fontsize=15.0, color=COLORS["muted"])
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.72, wspace=0.10)
    return fig, {
        "figure": "Simplified GEC-CCP dataset quality for the representative base4 geometry",
        "presentation_revision": "representative_base_three_cards_v2",
        "scope_note": "Broader 78-case audit is intentionally omitted from the main slide figure.",
        **metrics,
    }


def create_icp_field_simple() -> tuple[plt.Figure, dict[str, Any]]:
    fields, structure, row = detail._load_icp_case()
    coils = detail._read_icp_coils(detail.ICP_CASE_ID)
    mask = structure["mask_plasma"]
    values = fields["ne"]
    vmin, vmax = _rounded_log_limits(values[mask])
    r = structure["r_coords"]
    z = structure["z_coords"]
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    shown = np.ma.array(values, mask=(~mask) | (~np.isfinite(values)) | (values <= 0.0))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.8, 7.2))
        image = ax.imshow(
            shown,
            origin="lower",
            extent=[float(r[0] - dr / 2.0), float(r[-1] + dr / 2.0), float(z[0] - dz / 2.0), float(z[-1] + dz / 2.0)],
            interpolation="nearest",
            cmap=_masked_cmap("viridis"),
            norm=LogNorm(vmin=vmin, vmax=vmax),
            aspect="equal",
            rasterized=True,
        )
        _overlay_icp_simple(ax, coils)
        ax.set_xlim(0.0, 30.0)
        ax.set_ylim(0.0, 22.0)
        ax.set_xticks(np.arange(0.0, 30.1, 5.0))
        ax.set_yticks(np.arange(0.0, 20.1, 5.0))
        ax.set_xlabel(r"$r$ (cm)")
        ax.set_ylabel(r"$z$ (cm)")
        ax.set_title("GEC-ICP   Electron density", loc="left", pad=10.0, weight="bold")
        ax.set_facecolor(COLORS["mask"])
        ax.set_aspect("equal", adjustable="box")
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.045,
            pad=0.035,
            extend="min",
            ticks=_log_ticks(vmin, vmax),
            format=LogFormatterMathtext(),
        )
        colorbar.set_label(r"$n_e$ (m$^{-3}$)", fontsize=17.0, labelpad=10.0)
        colorbar.ax.tick_params(labelsize=15.0, length=3.5)
        colorbar.outline.set_linewidth(0.8)
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.11, top=0.91)

    metadata = _physical_field_metadata(
        system="GEC-ICP",
        case_id=detail.ICP_CASE_ID,
        source_field=detail.ICP_DATA_ROOT / row["fields_npz"],
        source_structure=detail.ICP_DATA_ROOT / row["structure_npz"],
        mask=mask,
        values=values,
        vmin=vmin,
        vmax=vmax,
    )
    metadata["active_coil_count"] = len(coils)
    metadata["source_coil_layout"] = detail.ICP_LAYOUT_ROOT / f"{detail.ICP_CASE_ID}__coil_layout.csv"
    return fig, metadata


def _simple_dimension_panel(
    ax: plt.Axes,
    *,
    row: dict[str, str],
    coils: Sequence[dict[str, float | int]],
) -> None:
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, facecolor="white", edgecolor=COLORS["ink"], linewidth=1.4))
    ax.add_patch(Rectangle((0.0, 12.0), 30.0, 2.0, facecolor="#A7DCCB", edgecolor=COLORS["green"], linewidth=1.4))
    for coil in coils:
        ax.add_patch(
            Rectangle(
                (float(coil["r_min"]), float(coil["z_min"])),
                float(coil["r_max"]) - float(coil["r_min"]),
                float(coil["z_max"]) - float(coil["z_min"]),
                facecolor=COLORS["orange"],
                edgecolor="#8C5C00",
                linewidth=1.2,
                zorder=4,
            )
        )
    r_c = float(row["rrc"])
    r_ce = float(row["rrce"])
    z_c = float(row["zzc"])
    l_c = float(row["llcoil"])
    n_c = int(float(row["nncoil"]))
    first = coils[0]
    arrow = {"arrowstyle": "<->", "linewidth": 1.4, "color": COLORS["ink"], "shrinkA": 0, "shrinkB": 0}
    ax.annotate("", xy=(r_c, 18.2), xytext=(0.0, 18.2), arrowprops=arrow)
    ax.text(r_c / 2.0, 18.65, r"$r_c$", ha="center", fontsize=14.0)
    ax.annotate("", xy=(r_ce, 20.1), xytext=(0.0, 20.1), arrowprops=arrow)
    ax.text(r_ce / 2.0, 20.55, r"$r_{ce}$", ha="center", fontsize=14.0)
    for x_value, y0, y1, label in (
        (1.0, 14.0, 14.0 + z_c, r"$z_c$"),
        (float(first["r_max"]) + 0.65, float(first["z_min"]), float(first["z_max"]), r"$l_c$"),
    ):
        ax.plot([x_value, x_value], [y0, y1], color=COLORS["ink"], linewidth=1.4)
        ax.plot([x_value - 0.22, x_value + 0.22], [y0, y0], color=COLORS["ink"], linewidth=1.4)
        ax.plot([x_value - 0.22, x_value + 0.22], [y1, y1], color=COLORS["ink"], linewidth=1.4)
        ax.text(x_value + 0.8, 0.5 * (y0 + y1), label, va="center", fontsize=13.5)
    ax.text(18.0, 17.1, rf"$N_c={n_c}$", ha="center", fontsize=15.0, weight="bold")
    ax.text(15.0, 7.2, r"$\mathbf{g}=[l_c,\;r_c,\;N_c,\;r_{ce},\;z_c]$", ha="center", fontsize=18.0, weight="bold")
    ax.text(15.0, 5.5, "5 global geometry values", ha="center", fontsize=13.0, color=COLORS["muted"])
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(0.0, 22.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 15.0, 30.0])
    ax.set_yticks([0.0, 11.0, 22.0])
    ax.set_xlabel(r"$r$ (cm)")
    ax.set_ylabel(r"$z$ (cm)")
    ax.set_title("Dimension vector", weight="bold", pad=8.0)


def create_icp_sdf_simple() -> tuple[plt.Figure, dict[str, Any]]:
    _fields, structure, row = detail._load_icp_case()
    coils = detail._read_icp_coils(detail.ICP_CASE_ID)
    r = structure["r_coords"]
    z = structure["z_coords"]
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    if not np.isclose(dr, dz, atol=1.0e-8):
        raise ValueError("SDF display requires isotropic r-z spacing")
    sdf_slots = [structure[f"sdf_coil_{index:02d}"] * dr for index in range(1, 7)]
    representative_index = 3
    representative_sdf = sdf_slots[representative_index - 1]
    union_sdf = np.minimum.reduce(sdf_slots)
    union_mask = np.maximum.reduce(structure["part_mask_stack"] > 0.5)
    contradictions = int(
        np.count_nonzero((union_sdf < 0.0) & (~union_mask))
        + np.count_nonzero((union_sdf > 0.0) & union_mask)
    )
    if contradictions:
        raise ValueError(f"union SDF sign contradicts coil mask at {contradictions} cells")
    extent = [float(r[0] - dr / 2.0), float(r[-1] + dr / 2.0), float(z[0] - dz / 2.0), float(z[-1] + dz / 2.0)]
    cmap = LinearSegmentedColormap.from_list("simple_sdf", ["#2B59C3", "#FFFFFF", "#D55E00"], N=256)
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=3.0)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.8))
        _simple_dimension_panel(axes[0], row=row, coils=coils)
        images = []
        for ax, values, title in (
            (axes[1], representative_sdf, "One coil SDF · zoom"),
            (axes[2], union_sdf, "Union SDF"),
        ):
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                aspect="equal",
                rasterized=True,
            )
            images.append(image)
            ax.contour(r, z, values, levels=[0.0], colors=[COLORS["ink"]], linewidths=1.0)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelleft=False)
            ax.set_title(title, weight="bold", pad=8.0)
        representative_coil = coils[representative_index - 1]
        coil_r_center = 0.5 * (float(representative_coil["r_min"]) + float(representative_coil["r_max"]))
        coil_z_center = 0.5 * (float(representative_coil["z_min"]) + float(representative_coil["z_max"]))
        zoom_half_width = 3.25
        axes[1].set_xlim(coil_r_center - zoom_half_width, coil_r_center + zoom_half_width)
        axes[1].set_ylim(coil_z_center - zoom_half_width, coil_z_center + zoom_half_width)
        tick_r_center = float(round(coil_r_center))
        tick_z_center = float(round(coil_z_center))
        axes[1].set_xticks([tick_r_center - 2.0, tick_r_center, tick_r_center + 2.0])
        axes[1].set_yticks([tick_z_center - 2.0, tick_z_center, tick_z_center + 2.0])
        axes[2].set_xlim(0.0, 30.0)
        axes[2].set_ylim(0.0, 22.0)
        axes[2].set_xticks([0.0, 15.0, 30.0])
        axes[2].set_yticks([0.0, 11.0, 22.0])
        fig.suptitle("GEC-ICP geometry encoding", fontsize=22.0, weight="bold", y=0.99)
        colorbar_axis = fig.add_axes([0.43, 0.075, 0.49, 0.035])
        colorbar = fig.colorbar(
            images[-1],
            cax=colorbar_axis,
            orientation="horizontal",
            extend="max",
            ticks=[-0.25, 0.0, 1.5, 3.0],
        )
        colorbar.set_label("Signed distance (cm)   ·   negative inside / zero at boundary / positive outside", fontsize=13.5)
        colorbar.ax.tick_params(labelsize=14.0, length=3.0)
        colorbar.outline.set_linewidth(0.7)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.86, wspace=0.16)

    return fig, {
        "figure": "Simplified GEC-ICP dimension-vector and SDF geometry encoding",
        "presentation_revision": "three_panel_representative_sdf_v2",
        "case_id": detail.ICP_CASE_ID,
        "dimension_vector": ["llcoil", "rrc", "nncoil", "rrce", "zzc"],
        "dimension_values": {key: float(row[key]) for key in ("llcoil", "rrc", "nncoil", "rrce", "zzc")},
        "shown_sdf_slot": f"sdf_coil_{representative_index:02d}",
        "shown_summary": "part_sdf_union = min_j sdf_coil_j",
        "sdf_source_unit": "grid pixels",
        "display_unit": "cm",
        "grid_spacing_cm": dr,
        "display_limits_cm": [-0.25, 3.0],
        "strict_sign_contradiction_cells": contradictions,
        "omitted_from_main_figure": ["five other slot maps", "part_source_sum"],
        "source_structure_npz": detail.ICP_DATA_ROOT / row["structure_npz"],
        "source_coil_layout": detail.ICP_LAYOUT_ROOT / f"{detail.ICP_CASE_ID}__coil_layout.csv",
    }


def _icp_g002_quality() -> dict[str, Any]:
    rows = [row for row in detail._read_csv(detail.ICP_INDEX_PATH) if row["base_case_id"] == "case_g002"]
    if len(rows) != 6:
        raise ValueError(f"expected 6 case_g002 operating rows, found {len(rows)}")
    field_packs = 0
    structure_packs = 0
    finite = 0
    total = 0
    field_keys = ("ne", "ni", "Te", "phi", "Br", "Bz", "Jelr", "Jelz")
    for row in rows:
        field_path = detail.ICP_DATA_ROOT / row["fields_npz"]
        structure_path = detail.ICP_DATA_ROOT / row["structure_npz"]
        if field_path.is_file():
            with np.load(field_path, allow_pickle=False) as fields:
                valid = set(field_keys).issubset(fields.files)
                if valid:
                    for key in field_keys:
                        values = np.asarray(fields[key])
                        finite += int(np.isfinite(values).sum())
                        total += int(values.size)
            field_packs += int(valid)
        if structure_path.is_file():
            with np.load(structure_path, allow_pickle=False) as structure:
                valid_structure = {"mask_plasma", "mask_coil", "part_mask_stack"}.issubset(structure.files)
            structure_packs += int(valid_structure)
    return {
        "base_case_id": "case_g002",
        "active_coils": 6,
        "operating_cases": len(rows),
        "field_packs_valid": field_packs,
        "structure_packs_valid": structure_packs,
        "finite_rate_eight_fields": float(finite / total),
        "field_keys": list(field_keys),
        "source_index": detail.ICP_INDEX_PATH,
    }


def create_icp_quality_simple() -> tuple[plt.Figure, dict[str, Any]]:
    metrics = _icp_g002_quality()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        _metric_card(axes[0], value="6", label="operating points", accent=COLORS["blue"])
        _metric_card(axes[1], value="6 / 6", label="field + structure packs", accent=COLORS["green"])
        _metric_card(axes[2], value="100%", label="finite values · 8 fields", accent=COLORS["orange"])
        fig.suptitle("GEC-ICP representative subset", fontsize=23.0, weight="bold", y=0.98)
        fig.text(0.5, 0.835, r"coverage & integrity   ·   case_g002 / one 6-coil geometry", ha="center", fontsize=15.0, color=COLORS["muted"])
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.72, wspace=0.10)
    return fig, {
        "figure": "Simplified GEC-ICP dataset quality for representative geometry group case_g002",
        "presentation_revision": "representative_group_three_cards_v2",
        "scope_note": "Broader 360-case coverage and split audit is intentionally omitted from the main slide figure.",
        **metrics,
    }


def _loss_card(
    ax: plt.Axes,
    *,
    x: float,
    width: float,
    color: str,
    heading: str,
    purpose: str,
) -> None:
    patch = FancyBboxPatch(
        (x, 0.31),
        width,
        0.23,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#F8FAFC",
        edgecolor=color,
        linewidth=1.7,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2.0, 0.445, heading, transform=ax.transAxes, ha="center", va="center", fontsize=16.0, weight="bold", color=color)
    ax.text(x + width / 2.0, 0.355, purpose, transform=ax.transAxes, ha="center", va="center", fontsize=12.8, color=COLORS["muted"])


def create_ccp_loss_simple() -> tuple[plt.Figure, dict[str, Any]]:
    metadata = loss_detail.build_ccp_metadata()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.0, 5.8))
        ax.set_axis_off()
        ax.text(0.04, 0.93, "GEC-CCP loss design", transform=ax.transAxes, fontsize=23.0, weight="bold", ha="left")
        ax.text(0.5, 0.73, r"$\ell_t=\mathrm{Huber}+0.25\,L_{\mathrm{boundary}}+0.10\,L_{\nabla}+0.05\,L_{\mathrm{multi}}$", transform=ax.transAxes, ha="center", fontsize=26.0)
        cards = (
            (0.035, COLORS["blue"], "Huber", "robust point fit"),
            (0.280, COLORS["orange"], "Boundary × 0.25", "protect sheath edges"),
            (0.525, COLORS["green"], "Gradient × 0.10", "preserve spatial shape"),
            (0.770, COLORS["purple"], "Multi-scale × 0.05", "retain broad structure"),
        )
        for x, color, heading, purpose in cards:
            _loss_card(ax, x=x, width=0.195, color=color, heading=heading, purpose=purpose)
        ax.text(0.5, 0.17, r"$L_{\mathrm{CCP}}=\dfrac{1}{3}\left[\dfrac{\ell_{n_e}+\ell_{n_i}}{2}+\ell_{T_e}+\ell_{\phi}\right]$", transform=ax.transAxes, ha="center", fontsize=24.0)
        ax.text(0.5, 0.06, "equal weight for density, temperature, and electric potential", transform=ax.transAxes, ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)
    metadata.update(
        {
            "presentation_revision": "two_equations_four_ideas_v2",
            "main_slide_message": "robust point fit plus boundary, gradient, and multiscale structure",
            "omitted_from_main_figure": ["Huber piecewise definition", "target standardization equation", "implementation details"],
        }
    )
    return fig, metadata


def create_icp_loss_simple() -> tuple[plt.Figure, dict[str, Any]]:
    metadata = loss_detail.build_icp_metadata()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.0, 5.8))
        ax.set_axis_off()
        ax.text(0.04, 0.93, "GEC-ICP loss design", transform=ax.transAxes, fontsize=23.0, weight="bold", ha="left")
        ax.text(0.5, 0.73, r"$\ell_t=\mathrm{Huber}+0.02\,L_{\nabla}+0.05\,L_{\mathrm{multi}}$", transform=ax.transAxes, ha="center", fontsize=29.0)
        cards = (
            (0.09, COLORS["blue"], "Huber", "robust point fit"),
            (0.365, COLORS["green"], "Gradient × 0.02", "preserve local shape"),
            (0.640, COLORS["purple"], "Multi-scale × 0.05", "retain broad structure"),
        )
        for x, color, heading, purpose in cards:
            _loss_card(ax, x=x, width=0.235, color=color, heading=heading, purpose=purpose)
        ax.text(0.5, 0.17, r"$L_{\mathrm{ICP}}=1.0\,\ell_{n_e}+1.0\,\ell_{n_i}+1.2\,\ell_{T_e}+0.5\,\ell_{\phi}$", transform=ax.transAxes, ha="center", fontsize=24.0)
        ax.text(0.5, 0.06, "plasma region only · transformed/scaled target space", transform=ax.transAxes, ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)
    metadata.update(
        {
            "presentation_revision": "two_equations_three_ideas_v2",
            "main_slide_message": "robust point fit with light gradient and multiscale shape preservation",
            "omitted_from_main_figure": ["target transform details", "Huber piecewise definition", "untrained alternative contract"],
        }
    )
    return fig, metadata


BUILDERS: dict[str, tuple[str, Callable[[], tuple[plt.Figure, dict[str, Any]]]]] = {
    "ccp_fields": ("ccp_representative_physical_fields", create_ccp_field_simple),
    "ccp_zscore": ("ccp_linear_zscore_effect", create_ccp_zscore_simple),
    "ccp_loss": ("gec_ccp_loss_design", create_ccp_loss_simple),
    "icp_fields": ("icp_representative_physical_fields", create_icp_field_simple),
    "icp_sdf": ("icp_geometry_features_sdf", create_icp_sdf_simple),
    "icp_loss": ("gec_icp_loss_design", create_icp_loss_simple),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=FORMATS)
    parser.add_argument("--figures", nargs="+", choices=("all", *BUILDERS), default=("all",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi < 72:
        raise ValueError("--dpi must be >= 72")
    selected = list(BUILDERS) if "all" in args.figures else list(dict.fromkeys(args.figures))
    generated: list[Path] = []
    for key in selected:
        stem, builder = BUILDERS[key]
        fig, metadata = builder()
        try:
            outputs = _save(fig, out_dir=args.out_dir, stem=stem, formats=args.formats, dpi=args.dpi)
        finally:
            plt.close(fig)
        metadata["output_files"] = [path.name for path in outputs]
        metadata_path = _write_metadata(args.out_dir, stem, metadata)
        generated.extend((*outputs, metadata_path))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
