"""Generate standalone conference assets from the current GEC-CCP/ICP runs.

Each output makes one visual claim and can be placed independently on a slide.
The script intentionally avoids cropping panels out of composite figures: every
asset is redrawn from the effective dataset and split artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, PercentFormatter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.conference.scripts import plot_gec_conference_dataset_and_formula as calc  # noqa: E402
from experiments.conference.scripts import plot_gec_conference_simple as simple  # noqa: E402
from external_tools.structure_converter.convert_mphtxt_to_parametric_parts import (  # noqa: E402
    _read_mphtxt_edge_entities,
)


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials/independent_assets")
FORMATS = ("png", "pdf", "svg")
FIGSIZE = (8.0, 6.0)
COLORS = simple.COLORS
SPLIT_COLORS = calc.SPLIT_COLORS


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _save(
    figure: plt.Figure,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with plt.rc_context(simple.STYLE):
        for suffix in dict.fromkeys(formats):
            path = out_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=dpi, facecolor="white")
            outputs.append(path)
    return outputs


def _metadata_base(*, claim: str, sources: dict[str, Path | str]) -> dict[str, Any]:
    return {
        "presentation_revision": "standalone_single_claim_v1",
        "claim": claim,
        "sources": {name: _rel(Path(path)) if isinstance(path, Path) else path for name, path in sources.items()},
    }


@lru_cache(maxsize=1)
def _ccp_data() -> dict[str, Any]:
    rows = calc._read_rows(calc.CCP_INDEX)
    split = calc._read_json(calc.CCP_SPLIT)
    lookup = calc._split_lookup(split)
    p95 = calc._plasma_ne_p95(calc.CCP_ROOT, rows)
    counts = {key: calc._condition_counts(rows, key) for key in ("Td", "PP0", "PA", "gamma")}
    return {"rows": rows, "split": split, "lookup": lookup, "p95": p95, "counts": counts}


@lru_cache(maxsize=1)
def _icp_data() -> dict[str, Any]:
    rows = calc._read_rows(calc.ICP_INDEX)
    split = calc._read_json(calc.ICP_SPLIT)
    lookup = calc._split_lookup(split)
    p95 = calc._plasma_ne_p95(calc.ICP_ROOT, rows)
    geometry_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        geometry_by_id.setdefault(row["base_case_id"], row)
    geometry = list(geometry_by_id.values())
    return {"rows": rows, "split": split, "lookup": lookup, "p95": p95, "geometry": geometry}


@lru_cache(maxsize=1)
def _zscore_data() -> tuple[np.ndarray, float, float, list[str]]:
    return simple._ccp_train_ne()


@lru_cache(maxsize=1)
def _sdf_data() -> dict[str, Any]:
    _fields, structure, row = simple.detail._load_icp_case()
    coils = simple.detail._read_icp_coils(simple.detail.ICP_CASE_ID)
    r = np.asarray(structure["r_coords"], dtype=float)
    z = np.asarray(structure["z_coords"], dtype=float)
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    if not np.isclose(dr, dz, atol=1.0e-8):
        raise ValueError("SDF display requires isotropic r-z spacing")
    slots = [np.asarray(structure[f"sdf_coil_{index:02d}"], dtype=float) * dr for index in range(1, 7)]
    union = np.minimum.reduce(slots)
    union_mask = np.maximum.reduce(np.asarray(structure["part_mask_stack"]) > 0.5)
    contradictions = int(np.count_nonzero((union < 0.0) & (~union_mask)) + np.count_nonzero((union > 0.0) & union_mask))
    if contradictions:
        raise ValueError(f"union SDF sign contradicts coil mask at {contradictions} cells")
    extent = [float(r[0] - dr / 2), float(r[-1] + dr / 2), float(z[0] - dz / 2), float(z[-1] + dz / 2)]
    return {
        "structure": structure,
        "row": row,
        "coils": coils,
        "r": r,
        "z": z,
        "dr": dr,
        "extent": extent,
        "representative_index": 3,
        "representative": slots[2],
        "union": union,
        "contradictions": contradictions,
    }


@lru_cache(maxsize=1)
def _ccp_case_data() -> dict[str, Any]:
    fields, structure, row = simple.detail._load_ccp_case()
    geometry = simple.load_ccp_geometry(
        base_name=row["base_name"],
        source_root=simple.CCP_GEOMETRY_SOURCE_ROOT,
        structure_index_path=simple.CCP_GEOMETRY_STRUCTURE_INDEX,
        manifest_path=simple.CCP_GEOMETRY_MANIFEST,
    )
    return {"fields": fields, "structure": structure, "row": row, "geometry": geometry}


@lru_cache(maxsize=1)
def _icp_case_data() -> dict[str, Any]:
    fields, structure, row = simple.detail._load_icp_case()
    coils = simple.detail._read_icp_coils(simple.detail.ICP_CASE_ID)
    return {"fields": fields, "structure": structure, "row": row, "coils": coils}


@lru_cache(maxsize=1)
def _ccp_quad_mesh() -> tuple[np.ndarray, np.ndarray, Path]:
    geometry = _ccp_case_data()["geometry"]
    path = Path(geometry.mphtxt_path)
    vertices, _edges, _entity_ids = _read_mphtxt_edge_entities(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        quad_type = next(
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*\d+\s+quad\s*#\s*type name\s*$", line, re.IGNORECASE)
        )
        element_marker = next(
            index
            for index in range(quad_type + 1, len(lines))
            if lines[index].strip().lower() == "# elements"
        )
    except StopIteration as exc:
        raise ValueError(f"failed to locate quad elements in {path}") from exc
    count = int(lines[element_marker - 1].split()[0])
    quads: list[tuple[int, int, int, int]] = []
    for line in lines[element_marker + 1 :]:
        tokens = line.split()
        if len(tokens) != 4:
            if quads:
                break
            continue
        try:
            quads.append(tuple(int(token) for token in tokens))
        except ValueError:
            if quads:
                break
        if len(quads) == count:
            break
    connectivity = np.asarray(quads, dtype=np.int64)
    if connectivity.shape != (count, 4):
        raise ValueError(f"quad element count mismatch in {path}: {connectivity.shape} != {(count, 4)}")
    if np.any(connectivity < 0) or np.any(connectivity >= vertices.shape[0]):
        raise ValueError(f"quad connectivity references an invalid vertex in {path}")
    return vertices, connectivity, path


def _style_spatial_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$r$ (cm)")
    ax.set_ylabel(r"$z$ (cm)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, width=0.8)


def _icp_component_patches(ax: plt.Axes, coils: Sequence[dict[str, float | int]]) -> None:
    ax.add_patch(Rectangle((0.0, 14.0), 30.0, 8.0, facecolor="#F2F4F7", edgecolor="none", zorder=0))
    plasma_vertices = np.asarray([(0.0, 2.0), (20.0, 2.0), (20.0, 0.0), (30.0, 0.0), (30.0, 12.0), (0.0, 12.0)])
    ax.add_patch(Polygon(plasma_vertices, closed=True, facecolor="#DCEAF7", edgecolor="#7DB5D8", linewidth=1.2, zorder=1))
    ax.add_patch(Rectangle((0.0, 0.0), 15.0, 2.0, facecolor="#98A2B3", edgecolor="#475467", linewidth=1.2, zorder=3))
    ax.add_patch(Rectangle((15.0, 0.0), 5.0, 2.0, facecolor="#A7DCCB", edgecolor=COLORS["green"], linewidth=1.2, zorder=3))
    ax.add_patch(Rectangle((0.0, 12.0), 30.0, 2.0, facecolor="#A7DCCB", edgecolor=COLORS["green"], linewidth=1.2, zorder=3))
    for coil in coils:
        ax.add_patch(
            Rectangle(
                (float(coil["r_min"]), float(coil["z_min"])),
                float(coil["r_max"]) - float(coil["r_min"]),
                float(coil["z_max"]) - float(coil["z_min"]),
                facecolor="#F2A900",
                edgecolor="#8C5C00",
                linewidth=1.0,
                zorder=4,
            )
        )
    ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, facecolor="none", edgecolor=COLORS["ink"], linewidth=1.4, zorder=5))
    ax.plot([0.0, 0.0], [0.0, 22.0], color=COLORS["ink"], linewidth=1.0, linestyle=(0, (4.0, 3.0)), zorder=6)


FIELD_SPECS: dict[str, dict[str, str]] = {
    "ne": {
        "title": "electron density",
        "slug": "electron_density",
        "symbol": r"$n_e$",
        "unit": r"m$^{-3}$",
        "metadata_unit": "m^-3",
        "scale": "logarithmic",
    },
    "ni": {
        "title": "ion density",
        "slug": "ion_density",
        "symbol": r"$n_i$",
        "unit": r"m$^{-3}$",
        "metadata_unit": "m^-3",
        "scale": "logarithmic",
    },
    "Te": {
        "title": "electron temperature",
        "slug": "electron_temperature",
        "symbol": r"$T_e$",
        "unit": "eV",
        "metadata_unit": "eV",
        "scale": "linear",
    },
    "phi": {
        "title": "electric potential",
        "slug": "electric_potential",
        "symbol": r"$\phi$",
        "unit": "V",
        "metadata_unit": "V",
        "scale": "linear",
    },
}


def _field_display(system: str, field_name: str = "ne") -> dict[str, Any]:
    if field_name not in FIELD_SPECS:
        raise ValueError(f"unknown field: {field_name}")
    spec = FIELD_SPECS[field_name]
    if system == "ccp":
        data = _ccp_case_data()
        values = np.asarray(data["fields"][field_name], dtype=float)
        structure = data["structure"]
        mask = np.asarray(structure["mask_plasma"], dtype=bool)
        r = 100.0 * np.asarray(structure["r_coords"], dtype=float)
        z = 100.0 * np.asarray(structure["z_coords"], dtype=float)
        case_id = simple.detail.CCP_CASE_ID
        source_field = simple.detail.CCP_DATA_ROOT / data["row"]["fields_npz"]
        source_structure = simple.detail.CCP_DATA_ROOT / data["row"]["structure_npz"]
    elif system == "icp":
        data = _icp_case_data()
        values = np.asarray(data["fields"][field_name], dtype=float)
        structure = data["structure"]
        mask = np.asarray(structure["mask_plasma"], dtype=bool)
        r = np.asarray(structure["r_coords"], dtype=float)
        z = np.asarray(structure["z_coords"], dtype=float)
        case_id = simple.detail.ICP_CASE_ID
        source_field = simple.detail.ICP_DATA_ROOT / data["row"]["fields_npz"]
        source_structure = simple.detail.ICP_DATA_ROOT / data["row"]["structure_npz"]
    else:
        raise ValueError(f"unknown system: {system}")
    valid_values = values[mask & np.isfinite(values)]
    if spec["scale"] == "logarithmic":
        positive_values = valid_values[valid_values > 0.0]
        vmin, vmax = simple._rounded_log_limits(positive_values)
        norm: Normalize = LogNorm(vmin=vmin, vmax=vmax)
        cmap_name = "viridis"
        shown = np.ma.array(values, mask=(~mask) | (~np.isfinite(values)) | (values <= 0.0))
        colorbar_ticks = simple._log_ticks(vmin, vmax)
        extend = "min"
        color_encoding = f"logarithmic {field_name} [{spec['metadata_unit']}]"
    elif field_name == "Te":
        vmin = float(np.min(valid_values))
        vmax = float(np.quantile(valid_values, 0.995)) if system == "ccp" else float(np.max(valid_values))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_name = "plasma"
        shown = np.ma.array(values, mask=(~mask) | (~np.isfinite(values)))
        colorbar_ticks = None
        extend = "max" if np.max(valid_values) > vmax else "neither"
        color_encoding = f"linear {field_name} [{spec['metadata_unit']}]"
    else:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        if vmin < 0.0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap_name = "coolwarm"
            color_encoding = f"linear {field_name} [{spec['metadata_unit']}], centered at zero"
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap_name = "cividis"
            color_encoding = f"linear {field_name} [{spec['metadata_unit']}]"
        shown = np.ma.array(values, mask=(~mask) | (~np.isfinite(values)))
        colorbar_ticks = None
        extend = "neither"
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    extent = [float(r[0] - dr / 2.0), float(r[-1] + dr / 2.0), float(z[0] - dz / 2.0), float(z[-1] + dz / 2.0)]
    return {
        "data": data,
        "values": values,
        "shown": shown,
        "mask": mask,
        "r": r,
        "z": z,
        "extent": extent,
        "vmin": vmin,
        "vmax": vmax,
        "norm": norm,
        "cmap_name": cmap_name,
        "colorbar_ticks": colorbar_ticks,
        "extend": extend,
        "color_encoding": color_encoding,
        "spec": spec,
        "field_name": field_name,
        "case_id": case_id,
        "source_field": source_field,
        "source_structure": source_structure,
    }


def create_ccp_geometry_asset() -> tuple[plt.Figure, dict[str, Any]]:
    data = _ccp_case_data()
    geometry = data["geometry"]
    domain_cm = 100.0 * np.asarray(geometry.domain_vertices_m, dtype=float)
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.add_patch(Polygon(domain_cm, closed=True, facecolor="#DCEAF7", edgecolor="#7DB5D8", linewidth=1.1, zorder=0))
        simple._overlay_ccp_simple(ax, geometry)
        _style_spatial_axis(ax, xlim=(-0.1, 10.3), ylim=(-4.0, 6.6))
        ax.set_xticks(np.arange(0.0, 10.1, 2.0))
        ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
        handles = [
            Patch(facecolor="#DCEAF7", edgecolor="#7DB5D8", label="Computational domain"),
            Line2D([0], [0], color=COLORS["red"], linewidth=3.0, label="Driven electrode"),
            Line2D([0], [0], color=COLORS["green"], linewidth=3.0, label="Dielectric contact"),
            Line2D([0], [0], color=COLORS["ink"], linewidth=1.6, label="Grounded walls"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10.5, columnspacing=1.4, handlelength=2.0)
        fig.suptitle("GEC-CCP representative geometry", fontsize=21.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, "base4  ·  axisymmetric r–z section", ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.25, top=0.82)
    return fig, {
        **_metadata_base(claim="The base4 axisymmetric section identifies the CCP computational boundaries", sources={"mphtxt": geometry.mphtxt_path}),
        "system": "GEC-CCP",
        "case_id": simple.detail.CCP_CASE_ID,
        "geometry_alternative": geometry.base_name,
        "legend_location": "outside plot below",
    }


def create_ccp_mesh_asset() -> tuple[plt.Figure, dict[str, Any]]:
    data = _ccp_case_data()
    geometry = data["geometry"]
    vertices, quads, path = _ccp_quad_mesh()
    polygons_cm = 100.0 * vertices[quads]
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        collection = PolyCollection(polygons_cm, facecolors="#F8FAFC", edgecolors="#667085", linewidths=0.23, zorder=1)
        ax.add_collection(collection)
        simple._overlay_ccp_simple(ax, geometry)
        _style_spatial_axis(ax, xlim=(-0.1, 10.3), ylim=(-4.0, 6.6))
        ax.set_xticks(np.arange(0.0, 10.1, 2.0))
        ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
        fig.suptitle("GEC-CCP computational mesh", fontsize=21.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, f"COMSOL export  ·  {quads.shape[0]:,} quadrilateral elements", ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.82)
    return fig, {
        **_metadata_base(claim="The representative CCP case uses the exported COMSOL quadrilateral mesh", sources={"mphtxt": path}),
        "system": "GEC-CCP",
        "case_id": simple.detail.CCP_CASE_ID,
        "mesh_kind": "COMSOL quadrilateral mesh",
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_elements": int(quads.shape[0]),
        "displayed_elements": int(quads.shape[0]),
    }


def create_icp_geometry_asset() -> tuple[plt.Figure, dict[str, Any]]:
    data = _icp_case_data()
    coils = data["coils"]
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _icp_component_patches(ax, coils)
        _style_spatial_axis(ax, xlim=(-0.2, 30.3), ylim=(-0.2, 22.3))
        ax.set_xticks(np.arange(0.0, 30.1, 5.0))
        ax.set_yticks(np.arange(0.0, 22.1, 5.0))
        handles = [
            Patch(facecolor="#DCEAF7", edgecolor="#7DB5D8", label="Plasma"),
            Patch(facecolor="#A7DCCB", edgecolor=COLORS["green"], label="Dielectric"),
            Patch(facecolor="#98A2B3", edgecolor="#475467", label="Substrate"),
            Patch(facecolor="#F2A900", edgecolor="#8C5C00", label="Coils"),
            Patch(facecolor="#F2F4F7", edgecolor="#D0D5DD", label="EM domain"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=9.6, columnspacing=1.0, handlelength=1.5)
        fig.suptitle("GEC-ICP representative geometry", fontsize=21.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, "case_g002  ·  six-coil axisymmetric r–z section", ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.25, top=0.82)
    return fig, {
        **_metadata_base(claim="The representative ICP section distinguishes the plasma, dielectrics, substrate, and coils", sources={"structure": simple.detail.ICP_DATA_ROOT / data["row"]["structure_npz"], "coil_layout": simple.detail.ICP_LAYOUT_ROOT / f"{simple.detail.ICP_CASE_ID}__coil_layout.csv"}),
        "system": "GEC-ICP",
        "case_id": simple.detail.ICP_CASE_ID,
        "active_coils": len(coils),
        "legend_location": "outside plot below",
    }


def create_icp_mesh_asset() -> tuple[plt.Figure, dict[str, Any]]:
    data = _icp_case_data()
    structure = data["structure"]
    r = np.asarray(structure["r_coords"], dtype=float)
    z = np.asarray(structure["z_coords"], dtype=float)
    display_stride = 10
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, facecolor="#F8FAFC", edgecolor="none", zorder=0))
        ax.vlines(r[::display_stride], float(z[0]), float(z[-1]), color="#98A2B3", linewidth=0.35, alpha=0.75, zorder=1)
        ax.hlines(z[::display_stride], float(r[0]), float(r[-1]), color="#98A2B3", linewidth=0.35, alpha=0.75, zorder=1)
        simple._overlay_icp_simple(ax, data["coils"])
        _style_spatial_axis(ax, xlim=(-0.2, 30.3), ylim=(-0.2, 22.3))
        ax.set_xticks(np.arange(0.0, 30.1, 5.0))
        ax.set_yticks(np.arange(0.0, 22.1, 5.0))
        fig.suptitle("GEC-ICP structured field grid", fontsize=21.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, f"native grid: {r.size} × {z.size}  ·  every {display_stride}th line shown", ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.14, top=0.82)
    return fig, {
        **_metadata_base(claim="The representative ICP learning field is stored on a regular structured grid", sources={"structure": simple.detail.ICP_DATA_ROOT / data["row"]["structure_npz"]}),
        "system": "GEC-ICP",
        "case_id": simple.detail.ICP_CASE_ID,
        "mesh_kind": "structured surrogate field grid",
        "grid_shape_zr": [int(z.size), int(r.size)],
        "display_line_stride": display_stride,
        "native_spacing_cm": [float(np.median(np.diff(z))), float(np.median(np.diff(r)))],
    }


def _physical_field_2d(system: str, field_name: str) -> tuple[plt.Figure, dict[str, Any]]:
    field = _field_display(system, field_name)
    spec = field["spec"]
    system_label = "GEC-CCP" if system == "ccp" else "GEC-ICP"
    with plt.rc_context(simple.STYLE):
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes([0.12 if system == "ccp" else 0.10, 0.15, 0.66 if system == "ccp" else 0.69, 0.66])
        image = ax.imshow(
            field["shown"],
            origin="lower",
            extent=field["extent"],
            interpolation="nearest",
            cmap=simple._masked_cmap(field["cmap_name"]),
            norm=field["norm"],
            aspect="equal",
            rasterized=True,
        )
        if system == "ccp":
            simple._overlay_ccp_simple(ax, field["data"]["geometry"])
            _style_spatial_axis(ax, xlim=(-0.05, 10.21), ylim=(-3.86, 6.46))
            ax.set_xticks(np.arange(0.0, 10.1, 2.0))
            ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
        else:
            simple._overlay_icp_simple(ax, field["data"]["coils"])
            _style_spatial_axis(ax, xlim=(0.0, 30.0), ylim=(0.0, 22.0))
            ax.set_xticks(np.arange(0.0, 30.1, 5.0))
            ax.set_yticks(np.arange(0.0, 22.1, 5.0))
        ax.set_facecolor(COLORS["mask"])
        colorbar_axis = fig.add_axes([0.82, 0.20, 0.030, 0.54])
        colorbar = fig.colorbar(image, cax=colorbar_axis, extend=field["extend"], ticks=field["colorbar_ticks"])
        if spec["scale"] == "logarithmic":
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        colorbar.set_label(f"{spec['symbol']} ({spec['unit']})", fontsize=13.0, labelpad=8.0)
        colorbar.ax.tick_params(labelsize=11.5, length=3.0)
        fig.suptitle(f"{system_label} {spec['title']} map", fontsize=21.0, weight="bold", y=0.965)
        fig.text(0.5, 0.895, f"representative case  ·  {spec['scale']} color scale", ha="center", fontsize=13.5, color=COLORS["muted"])
    metadata = {
        **_metadata_base(claim=f"The representative {system_label} case shows the spatial {spec['title']} distribution", sources={"field": field["source_field"], "structure": field["source_structure"]}),
        "system": system_label,
        "case_id": field["case_id"],
        "view": "2D axisymmetric map",
        "shown_field": field_name,
        "unit": spec["metadata_unit"],
        "normalization": f"{spec['scale']} color scale",
        "color_encoding": field["color_encoding"],
        "colormap": field["cmap_name"],
        "display_limits": [field["vmin"], field["vmax"]],
    }
    if spec["metadata_unit"] == "m^-3":
        metadata["display_limits_m3"] = [field["vmin"], field["vmax"]]
    return fig, metadata


def create_ccp_electron_density_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("ccp", "ne")


def create_icp_electron_density_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("icp", "ne")


def create_ccp_ion_density_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("ccp", "ni")


def create_icp_ion_density_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("icp", "ni")


def create_ccp_electron_temperature_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("ccp", "Te")


def create_icp_electron_temperature_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("icp", "Te")


def create_ccp_electric_potential_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("ccp", "phi")


def create_icp_electric_potential_2d() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_2d("icp", "phi")


def _physical_field_perspective(system: str, field_name: str) -> tuple[plt.Figure, dict[str, Any]]:
    field = _field_display(system, field_name)
    spec = field["spec"]
    system_label = "GEC-CCP" if system == "ccp" else "GEC-ICP"
    stride = 3 if system == "ccp" else 6
    r = field["r"][::stride]
    z = field["z"][::stride]
    mask = field["mask"][::stride, ::stride]
    values = field["values"][::stride, ::stride]
    valid = mask & np.isfinite(values)
    if spec["scale"] == "logarithmic":
        valid &= values > 0.0
    rr, zz = np.meshgrid(r, z)
    norm = field["norm"]
    cmap = plt.get_cmap(field["cmap_name"])
    facecolors = np.empty((*values.shape, 4), dtype=float)
    facecolors[...] = to_rgba(COLORS["mask"])
    facecolors[valid] = cmap(norm(values[valid]))
    plane_depth = 0.75 if system == "ccp" else 1.6
    front_y = -0.02 * plane_depth
    r0, r1 = float(field["extent"][0]), float(field["extent"][1])
    z0, z1 = float(field["extent"][2]), float(field["extent"][3])

    def plot_line(ax: plt.Axes, x_values: Sequence[float], z_values: Sequence[float], **kwargs: Any) -> None:
        ax.plot(x_values, [front_y] * len(x_values), z_values, **kwargs)

    def plot_rectangle(ax: plt.Axes, x0: float, z0_rect: float, width: float, height: float, **kwargs: Any) -> None:
        plot_line(
            ax,
            [x0, x0 + width, x0 + width, x0, x0],
            [z0_rect, z0_rect, z0_rect + height, z0_rect + height, z0_rect],
            **kwargs,
        )

    with plt.rc_context(simple.STYLE):
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes([0.02, 0.08, 0.76, 0.76], projection="3d")
        ax.plot_surface(
            rr,
            np.zeros_like(rr),
            zz,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            rasterized=True,
        )
        side_faces = [
            [(r0, 0.0, z0), (r1, 0.0, z0), (r1, plane_depth, z0), (r0, plane_depth, z0)],
            [(r0, 0.0, z1), (r1, 0.0, z1), (r1, plane_depth, z1), (r0, plane_depth, z1)],
            [(r0, 0.0, z0), (r0, 0.0, z1), (r0, plane_depth, z1), (r0, plane_depth, z0)],
            [(r1, 0.0, z0), (r1, 0.0, z1), (r1, plane_depth, z1), (r1, plane_depth, z0)],
        ]
        plate = Poly3DCollection(side_faces, facecolors="#CBD5E1", edgecolors=COLORS["ink"], linewidths=0.8, alpha=1.0)
        ax.add_collection3d(plate)
        plot_rectangle(ax, r0, z0, r1 - r0, z1 - z0, color=COLORS["ink"], linewidth=1.2, zorder=8)

        if system == "ccp":
            geometry = field["data"]["geometry"]
            role_styles = {
                "grounded_electrode_and_walls": (COLORS["ink"], 1.4, "solid"),
                "driven_electrode": (COLORS["red"], 2.5, "solid"),
                "dielectric_contact": (COLORS["green"], 2.5, "solid"),
                "axis_of_symmetry": (COLORS["ink"], 1.0, (0, (4.0, 3.0))),
            }
            for role, entity_ids in simple.CCP_ROLE_ENTITY_IDS.items():
                color, linewidth, linestyle = role_styles[role]
                for entity_id in entity_ids:
                    segment = geometry.boundary(entity_id)
                    plot_line(
                        ax,
                        [100.0 * segment.r0, 100.0 * segment.r1],
                        [100.0 * segment.z0, 100.0 * segment.z1],
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        zorder=9,
                    )
        else:
            plot_rectangle(ax, 0.0, 0.0, 30.0, 22.0, color=COLORS["ink"], linewidth=1.3, zorder=9)
            plot_rectangle(ax, 0.0, 0.0, 15.0, 2.0, color=COLORS["ink"], linewidth=1.2, zorder=9)
            plot_rectangle(ax, 15.0, 0.0, 5.0, 2.0, color=COLORS["green"], linewidth=1.8, zorder=9)
            plot_rectangle(ax, 0.0, 12.0, 30.0, 2.0, color=COLORS["green"], linewidth=1.8, zorder=9)
            for coil in field["data"]["coils"]:
                plot_rectangle(
                    ax,
                    float(coil["r_min"]),
                    float(coil["z_min"]),
                    float(coil["r_max"]) - float(coil["r_min"]),
                    float(coil["z_max"]) - float(coil["z_min"]),
                    color=COLORS["orange"],
                    linewidth=1.8,
                    zorder=9,
                )

        ax.view_init(elev=9.0, azim=-58.0)
        ax.set_xlim(r0, r1)
        ax.set_ylim(-0.25 * plane_depth, 1.45 * plane_depth)
        ax.set_zlim(z0, z1)
        ax.set_box_aspect(
            (r1 - r0, 0.23 * (r1 - r0), z1 - z0),
            zoom=1.18,
        )
        ax.set_axis_off()
        colorbar_axis = fig.add_axes([0.82, 0.20, 0.028, 0.52])
        scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar_map, cax=colorbar_axis, extend=field["extend"], ticks=field["colorbar_ticks"])
        if spec["scale"] == "logarithmic":
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        colorbar.set_label(f"{spec['symbol']} ({spec['unit']})", fontsize=12.5, labelpad=7.0)
        colorbar.ax.tick_params(labelsize=11.0, length=3.0)
        fig.text(0.39, 0.070, r"$r$ (cm)", ha="center", fontsize=13.5, color=COLORS["ink"], zorder=20)
        fig.text(0.075, 0.46, r"$z$ (cm)", ha="center", va="center", rotation=90, fontsize=13.5, color=COLORS["ink"], zorder=20)
        fig.suptitle(f"{system_label} {spec['title']} · tilted 2D map", fontsize=20.5, weight="bold", y=0.965)
        fig.text(0.5, 0.895, f"flat map tilted in depth  ·  color = {spec['symbol']}", ha="center", fontsize=13.5, color=COLORS["muted"])
    metadata = {
        **_metadata_base(claim=f"The representative {system_label} 2D {spec['title']} map is tilted as a flat plate for perspective presentation", sources={"field": field["source_field"], "structure": field["source_structure"]}),
        "system": system_label,
        "case_id": field["case_id"],
        "view": "flat 2D map tilted in depth",
        "shown_field": field_name,
        "unit": spec["metadata_unit"],
        "height_encoding": "none; the plate is flat",
        "color_encoding": field["color_encoding"],
        "colormap": field["cmap_name"],
        "display_stride": stride,
        "display_limits": [field["vmin"], field["vmax"]],
        "display_only_plate_depth": plane_depth,
        "camera": {"elevation_deg": 9.0, "azimuth_deg": -58.0},
    }
    if spec["metadata_unit"] == "m^-3":
        metadata["display_limits_m3"] = [field["vmin"], field["vmax"]]
    return fig, metadata


def create_ccp_electron_density_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("ccp", "ne")


def create_icp_electron_density_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("icp", "ne")


def create_ccp_ion_density_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("ccp", "ni")


def create_icp_ion_density_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("icp", "ni")


def create_ccp_electron_temperature_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("ccp", "Te")


def create_icp_electron_temperature_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("icp", "Te")


def create_ccp_electric_potential_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("ccp", "phi")


def create_icp_electric_potential_perspective() -> tuple[plt.Figure, dict[str, Any]]:
    return _physical_field_perspective("icp", "phi")


def create_ccp_response_coverage() -> tuple[plt.Figure, dict[str, Any]]:
    data = _ccp_data()
    rows, p95 = data["rows"], data["p95"]
    groups = ["base4", "base3", "base2"]
    values = [p95[np.asarray([row["base_name"] == group for row in rows], dtype=bool)] for group in groups]
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        box = ax.boxplot(values, positions=[3, 2, 1], orientation="horizontal", widths=0.42, whis=(0, 100), showfliers=False, patch_artist=True)
        for patch in box["boxes"]:
            patch.set(facecolor=COLORS["blue"], edgecolor=COLORS["blue"], alpha=0.20, linewidth=1.8)
        for artist in (*box["whiskers"], *box["caps"]):
            artist.set(color=COLORS["blue"], linewidth=1.8)
        for artist in box["medians"]:
            artist.set(color=COLORS["ink"], linewidth=2.5)
        ax.set_xscale("log")
        ax.set_xlim(1.0e15, 1.3e16)
        ax.set_xticks([1.0e15, 3.0e15, 1.0e16])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_yticks([3, 2, 1], groups)
        ax.set_xlabel(r"casewise plasma $n_e$ P95  (m$^{-3}$)")
        ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
        ax.text(0.02, 0.06, f"Overall span  ×{p95.max()/p95.min():.1f}", transform=ax.transAxes, fontsize=13, weight="bold")
        fig.suptitle("GEC-CCP electron-density coverage", fontsize=22, weight="bold", y=0.97)
        fig.text(0.5, 0.895, "box: middle 50%  ·  whiskers: full range", ha="center", fontsize=14, color=COLORS["muted"])
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.84)
    return fig, {
        **_metadata_base(claim="CCP casewise electron-density P95 spans 8.7x across all geometries", sources={"index": calc.CCP_INDEX}),
        "min": float(p95.min()),
        "median": float(np.median(p95)),
        "max": float(p95.max()),
        "max_over_min": float(p95.max() / p95.min()),
    }


def create_icp_geometry_coverage() -> tuple[plt.Figure, dict[str, Any]]:
    geometry = _icp_data()["geometry"]
    specs = [
        (r"$l_c$", "llcoil", "cm"),
        (r"$r_c$", "rrc", "cm"),
        (r"$N_c$", "nncoil", ""),
        (r"$r_{ce}$", "rrce", "cm"),
        (r"$z_c$", "zzc", "cm"),
    ]
    summary: dict[str, Any] = {}
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_xlim(-0.20, 1.20)
        ax.set_ylim(0.45, 5.55)
        for y, (label, key, unit) in zip((5, 4, 3, 2, 1), specs, strict=True):
            values = np.asarray([float(row[key]) for row in geometry], dtype=float)
            lo, median, hi = float(values.min()), float(np.median(values)), float(values.max())
            normalized = (values - lo) / (hi - lo)
            jitter = 0.045 * np.sin(np.arange(values.size) * 2.31)
            ax.plot([0, 1], [y, y], color=COLORS["line"], linewidth=5, solid_capstyle="round")
            ax.scatter(normalized, y + jitter, s=22, color=COLORS["blue"], alpha=0.55, edgecolors="none")
            ax.scatter([(median - lo) / (hi - lo)], [y], s=70, color=COLORS["ink"], edgecolors="white", linewidths=1.0, zorder=3)
            suffix = f" {unit}" if unit else ""
            ax.text(-0.08, y, label, ha="right", va="center", fontsize=18, weight="bold")
            ax.text(0, y - 0.18, f"{lo:.2f}{suffix}" if key != "nncoil" else f"{lo:.0f}", ha="left", va="top", fontsize=11.5, color=COLORS["muted"])
            ax.text(1, y - 0.18, f"{hi:.2f}{suffix}" if key != "nncoil" else f"{hi:.0f}", ha="right", va="top", fontsize=11.5, color=COLORS["muted"])
            summary[key] = {"min": lo, "median": median, "max": hi, "unique": int(np.unique(values).size)}
        ax.set_axis_off()
        fig.suptitle("GEC-ICP geometry coverage", fontsize=21, weight="bold", y=0.965)
        fig.text(0.5, 0.875, "60 designs  ·  black marker = median  ·  each row uses its own scale", ha="center", fontsize=14, color=COLORS["muted"])
        fig.subplots_adjust(left=0.14, right=0.94, bottom=0.08, top=0.79)
    return fig, {
        **_metadata_base(claim="The ICP dataset spans all five geometry parameters across 60 designs", sources={"index": calc.ICP_INDEX}),
        "parameters": summary,
    }


def create_icp_operating_coverage() -> tuple[plt.Figure, dict[str, Any]]:
    data = _icp_data()
    rows, p95 = data["rows"], data["p95"]
    pp = np.asarray([float(row["pp"]) for row in rows])
    pp0 = np.asarray([float(row["pp0"]) for row in rows])
    log_pp0 = np.log10(pp0)
    log_response = np.log10(p95)
    near_floor_index = int(np.argmin(p95))
    surface_mask = np.ones(p95.size, dtype=bool)
    surface_mask[near_floor_index] = False
    q05, q95 = np.quantile(log_response[surface_mask], [0.05, 0.95])
    levels = np.linspace(float(q05), float(q95), 15)
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        surface = ax.tricontourf(
            log_pp0[surface_mask],
            pp[surface_mask],
            log_response[surface_mask],
            levels=levels,
            cmap="viridis",
            extend="both",
        )
        ax.tricontour(
            log_pp0[surface_mask],
            pp[surface_mask],
            log_response[surface_mask],
            levels=levels[::3],
            colors=COLORS["ink"],
            linewidths=0.45,
            alpha=0.28,
        )
        ax.scatter(
            log_pp0,
            pp,
            s=25,
            facecolors="white",
            edgecolors=COLORS["ink"],
            linewidths=0.55,
            alpha=0.78,
            zorder=3,
        )
        ax.set_xlim(float(log_pp0.min()) - 0.02, float(log_pp0.max()) + 0.02)
        ax.set_xticks([-2.5, -2.0, -1.5, -1.0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: rf"$10^{{{value:g}}}$"))
        ax.set_xlabel(r"operating input $p_{p0}$  (dataset units)")
        ax.set_ylabel(r"operating input $p_p$  (dataset units)")
        fig.suptitle("GEC-ICP operating inputs and response", fontsize=20.5, weight="bold", y=0.965)
        fig.text(0.5, 0.875, r"color: interpolated plasma $n_e$ P95  ·  white circles: 360 cases", ha="center", fontsize=13.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.14, right=0.80, bottom=0.14, top=0.79)
        colorbar_axis = fig.add_axes([0.83, 0.17, 0.028, 0.58])
        colorbar = fig.colorbar(surface, cax=colorbar_axis, ticks=[16.0, 17.0, 18.0])
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: rf"$10^{{{value:.0f}}}$"))
        colorbar.set_label(r"plasma $n_e$ P95  (m$^{-3}$)", fontsize=11.5, labelpad=7)
    return fig, {
        **_metadata_base(claim="The ICP operating-input samples cover the interpolated electron-density response surface", sources={"index": calc.ICP_INDEX, "split": calc.ICP_SPLIT}),
        "unique_pairs": int(np.unique(np.column_stack((pp, pp0)), axis=0).shape[0]),
        "pp_range": [float(pp.min()), float(pp.max())],
        "pp0_range": [float(pp0.min()), float(pp0.max())],
        "response": "casewise plasma ne P95",
        "response_unit": "m^-3",
        "surface_interpolation": "piecewise linear triangulation in (log10(pp0), pp, log10(ne_P95))",
        "surface_extrapolation": "none; only the sampled convex hull is filled",
        "surface_display_log10_quantiles": [float(q05), float(q95)],
        "surface_case_count": int(surface_mask.sum()),
        "scatter_case_count": int(p95.size),
        "excluded_from_surface_case_id": rows[near_floor_index]["case_id"],
        "excluded_from_surface_response_m3": float(p95[near_floor_index]),
    }


def create_icp_response_coverage() -> tuple[plt.Figure, dict[str, Any]]:
    data = _icp_data()
    rows, lookup, p95 = data["rows"], data["lookup"], data["p95"]
    summary: dict[str, Any] = {}
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for y, name in zip((3, 2, 1), ("train", "val", "test"), strict=True):
            mask = np.asarray([lookup[row["case_id"]] == name for row in rows], dtype=bool)
            q05, median, q95 = np.quantile(p95[mask], [0.05, 0.50, 0.95])
            ax.plot([q05, q95], [y, y], color=SPLIT_COLORS[name], linewidth=12, alpha=0.28, solid_capstyle="round")
            ax.plot([q05, q95], [y, y], color=SPLIT_COLORS[name], linewidth=2.2)
            ax.scatter([median], [y], s=115, color=SPLIT_COLORS[name], edgecolors="white", linewidths=1.1, zorder=3)
            summary[name] = {"q05": float(q05), "median": float(median), "q95": float(q95)}
        q05, q95 = np.quantile(p95, [0.05, 0.95])
        ax.set_xscale("log")
        ax.set_xlim(1.0e15, 1.0e19)
        ax.set_xticks([1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_yticks([3, 2, 1], ["Train", "Validation", "Test"])
        ax.set_ylim(0.45, 3.55)
        ax.set_xlabel(r"casewise plasma $n_e$ P95  (m$^{-3}$)")
        ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
        ax.text(0.02, 0.07, f"Overall central 90%  ×{q95/q05:.0f}", transform=ax.transAxes, fontsize=12.7, weight="bold")
        fig.suptitle("GEC-ICP electron-density coverage", fontsize=20.5, weight="bold", y=0.965)
        fig.text(0.5, 0.875, "line: P5–P95  ·  dot: median", ha="center", fontsize=14.0, color=COLORS["muted"])
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.79)
    return fig, {
        **_metadata_base(claim="Train, Validation, and Test cover comparable ICP response ranges", sources={"index": calc.ICP_INDEX, "split": calc.ICP_SPLIT}),
        "by_split": summary,
        "overall_q05": float(q05),
        "overall_q95": float(q95),
        "minimum": float(p95.min()),
    }


def _zscore_histogram(*, standardized: bool) -> tuple[plt.Figure, dict[str, Any]]:
    values, mu, sigma, train_ids = _zscore_data()
    raw = values / 1.0e15
    z = (values - mu) / sigma
    lo, hi = (float(value) for value in np.quantile(raw, [0.001, 0.995]))
    raw_edges = np.linspace(lo, hi, 50)
    z_edges = (raw_edges * 1.0e15 - mu) / sigma
    shown = z if standardized else raw
    edges = z_edges if standardized else raw_edges
    weights = np.full(shown.shape, 100.0 / shown.size)
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.hist(shown, bins=edges, weights=weights, color=COLORS["blue"] if standardized else "#98A2B3", edgecolor="white", linewidth=0.4)
        if standardized:
            ax.axvspan(-1, 1, color="#DCEAF7", alpha=0.75, linewidth=0)
            ax.axvline(0, color=COLORS["ink"], linewidth=2.2)
            ax.set_xlim(float(edges[0]), float(edges[-1]))
            ax.set_xticks(np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1.0, 1.0))
            ax.set_xlabel(r"standardized electron density  $z_e$")
            title = "After linear Z-score"
            subtitle = r"GEC-CCP  ·  $z_e=(n_e-1.29\times10^{15})/(1.87\times10^{15})$"
        else:
            ax.axvspan((mu - sigma) / 1e15, (mu + sigma) / 1e15, color="#D0D5DD", alpha=0.55, linewidth=0)
            ax.axvline(mu / 1e15, color=COLORS["ink"], linewidth=2.2)
            ax.set_xlim(float(edges[0]), float(edges[-1]))
            ax.set_xticks(np.arange(0.0, np.floor(edges[-1]) + 1.0, 2.0))
            ax.set_xlabel(r"electron density $n_e$  ($10^{15}$ m$^{-3}$)")
            title = "Before Z-score"
            subtitle = r"GEC-CCP  ·  Train-only plasma samples  ·  line = mean  ·  band = $\mu\pm\sigma$"
        ax.set_ylabel("Plasma samples (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color=COLORS["line"], linewidth=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.suptitle(title, fontsize=21.5, weight="bold", y=0.965)
        fig.text(0.5, 0.875, subtitle, ha="center", fontsize=14.0, color=COLORS["muted"])
        if standardized:
            ax.text(0.98, 0.94, "Same samples and distribution shape", transform=ax.transAxes, ha="right", va="top", fontsize=12.5, color=COLORS["muted"])
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.79)
    stem_claim = "standardized electron density has zero mean and unit population standard deviation" if standardized else "raw electron density uses the physical linear scale"
    return fig, {
        **_metadata_base(claim=stem_claim, sources={"split": simple.detail.CCP_SPLIT_PATH, "scaler": simple.detail.CCP_SCALER_PATH}),
        "stage": "after" if standardized else "before",
        "train_cases": len(train_ids),
        "samples": int(values.size),
        "mean_m3": mu,
        "std_m3": sigma,
        "z_mean": float(np.mean(z)),
        "z_std_ddof0": float(np.std(z, ddof=0)),
    }


def create_ccp_zscore_before() -> tuple[plt.Figure, dict[str, Any]]:
    return _zscore_histogram(standardized=False)


def create_ccp_zscore_after() -> tuple[plt.Figure, dict[str, Any]]:
    return _zscore_histogram(standardized=True)


def create_icp_dimension_vector() -> tuple[plt.Figure, dict[str, Any]]:
    data = _sdf_data()
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        simple._simple_dimension_panel(ax, row=data["row"], coils=data["coils"])
        ax.set_title("GEC-ICP dimension-vector encoding", fontsize=20, weight="bold", pad=10)
        fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.90)
    return fig, {
        **_metadata_base(claim="The ICP geometry is encoded by five global dimensions", sources={"structure": simple.detail.ICP_DATA_ROOT / data["row"]["structure_npz"]}),
        "case_id": simple.detail.ICP_CASE_ID,
        "dimension_vector": ["llcoil", "rrc", "nncoil", "rrce", "zzc"],
    }


def _sdf_figure(*, union: bool) -> tuple[plt.Figure, dict[str, Any]]:
    data = _sdf_data()
    values = data["union"] if union else data["representative"]
    cmap = LinearSegmentedColormap.from_list("standalone_sdf", ["#2B59C3", "#FFFFFF", "#D55E00"], N=256)
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=3.0)
    with plt.rc_context(simple.STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        image = ax.imshow(values, origin="lower", extent=data["extent"], interpolation="nearest", cmap=cmap, norm=norm, aspect="equal", rasterized=True)
        ax.contour(data["r"], data["z"], values, levels=[0], colors=[COLORS["ink"]], linewidths=1.1)
        if union:
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 22)
            title = "GEC-ICP union SDF"
            claim = "The union SDF gives one order-invariant spatial representation of all coils"
        else:
            coil = data["coils"][data["representative_index"] - 1]
            r_center = 0.5 * (float(coil["r_min"]) + float(coil["r_max"]))
            z_center = 0.5 * (float(coil["z_min"]) + float(coil["z_max"]))
            ax.set_xlim(r_center - 3.25, r_center + 3.25)
            ax.set_ylim(z_center - 3.25, z_center + 3.25)
            title = "GEC-ICP single-coil SDF"
            claim = "A signed-distance field represents a coil boundary and distance locally"
        ax.set_xlabel(r"$r$ (cm)")
        ax.set_ylabel(r"$z$ (cm)")
        ax.set_title(title, fontsize=21, weight="bold", pad=10)
        fig.subplots_adjust(left=0.11, right=0.95, bottom=0.31, top=0.90)
        fig.text(0.5, 0.165, "negative inside  ·  zero at boundary  ·  positive outside", ha="center", fontsize=11.5, color=COLORS["muted"])
        colorbar_axis = fig.add_axes([0.19, 0.095, 0.62, 0.030])
        colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal", extend="max", ticks=[-0.25, 0, 1.5, 3])
        colorbar.set_label("signed distance (cm)", fontsize=12.0, labelpad=4.0)
        colorbar.ax.tick_params(labelsize=10.5, pad=2.0)
    return fig, {
        **_metadata_base(claim=claim, sources={"structure": simple.detail.ICP_DATA_ROOT / data["row"]["structure_npz"]}),
        "case_id": simple.detail.ICP_CASE_ID,
        "kind": "union" if union else "single_coil",
        "display_unit": "cm",
        "display_limits_cm": [-0.25, 3.0],
        "strict_sign_contradiction_cells": data["contradictions"],
    }


def create_icp_single_coil_sdf() -> tuple[plt.Figure, dict[str, Any]]:
    return _sdf_figure(union=False)


def create_icp_union_sdf() -> tuple[plt.Figure, dict[str, Any]]:
    return _sdf_figure(union=True)


FIGURES: dict[str, tuple[str, Callable[[], tuple[plt.Figure, dict[str, Any]]]]] = {
    "ccp_geometry_diagram": ("ccp_geometry_conference", create_ccp_geometry_asset),
    "ccp_mesh": ("ccp_mesh_conference", create_ccp_mesh_asset),
    "icp_geometry_diagram": ("icp_geometry_conference", create_icp_geometry_asset),
    "icp_mesh": ("icp_mesh_conference", create_icp_mesh_asset),
    "ccp_ne_2d": ("ccp_electron_density_2d", create_ccp_electron_density_2d),
    "ccp_ne_3d": ("ccp_electron_density_perspective", create_ccp_electron_density_perspective),
    "ccp_ni_2d": ("ccp_ion_density_2d", create_ccp_ion_density_2d),
    "ccp_ni_3d": ("ccp_ion_density_perspective", create_ccp_ion_density_perspective),
    "ccp_te_2d": ("ccp_electron_temperature_2d", create_ccp_electron_temperature_2d),
    "ccp_te_3d": ("ccp_electron_temperature_perspective", create_ccp_electron_temperature_perspective),
    "ccp_phi_2d": ("ccp_electric_potential_2d", create_ccp_electric_potential_2d),
    "ccp_phi_3d": ("ccp_electric_potential_perspective", create_ccp_electric_potential_perspective),
    "icp_ne_2d": ("icp_electron_density_2d", create_icp_electron_density_2d),
    "icp_ne_3d": ("icp_electron_density_perspective", create_icp_electron_density_perspective),
    "icp_ni_2d": ("icp_ion_density_2d", create_icp_ion_density_2d),
    "icp_ni_3d": ("icp_ion_density_perspective", create_icp_ion_density_perspective),
    "icp_te_2d": ("icp_electron_temperature_2d", create_icp_electron_temperature_2d),
    "icp_te_3d": ("icp_electron_temperature_perspective", create_icp_electron_temperature_perspective),
    "icp_phi_2d": ("icp_electric_potential_2d", create_icp_electric_potential_2d),
    "icp_phi_3d": ("icp_electric_potential_perspective", create_icp_electric_potential_perspective),
    "ccp_response": ("ccp_dataset_response_coverage", create_ccp_response_coverage),
    "icp_geometry": ("icp_dataset_geometry_coverage", create_icp_geometry_coverage),
    "icp_operating": ("icp_dataset_operating_coverage", create_icp_operating_coverage),
    "icp_response": ("icp_dataset_response_coverage", create_icp_response_coverage),
    "ccp_zscore_before": ("ccp_zscore_before", create_ccp_zscore_before),
    "ccp_zscore_after": ("ccp_zscore_after", create_ccp_zscore_after),
    "icp_dimension": ("icp_dimension_vector", create_icp_dimension_vector),
    "icp_single_sdf": ("icp_single_coil_sdf", create_icp_single_coil_sdf),
    "icp_union_sdf": ("icp_union_sdf", create_icp_union_sdf),
}


def generate(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    figures: Sequence[str] = ("all",),
    formats: Sequence[str] = FORMATS,
    dpi: int = 400,
) -> list[Path]:
    selected = list(FIGURES) if "all" in figures else list(dict.fromkeys(figures))
    outputs: list[Path] = []
    for key in selected:
        stem, builder = FIGURES[key]
        figure, metadata = builder()
        try:
            paths = _save(figure, out_dir=out_dir, stem=stem, formats=formats, dpi=dpi)
        finally:
            plt.close(figure)
        metadata["output_stem"] = stem
        metadata["outputs"] = [_rel(path) for path in paths]
        metadata_path = out_dir / f"{stem}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        outputs.extend(paths)
        outputs.append(metadata_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--figures", nargs="+", choices=["all", *FIGURES], default=["all"])
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in generate(out_dir=args.out_dir, figures=args.figures, formats=args.formats, dpi=args.dpi):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
