"""Create conference-ready color and outline views of the GEC-CCP geometry.

The drawing is reconstructed from the original COMSOL MPHTXT boundary
connectivity.  Electrodes and the dielectric contact are boundary conditions,
not fabricated solid domains.  The representative ``base4`` alternative is
used because its 3 mm dielectric break matches the standard COMSOL example.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_tools.structure_converter.convert_mphtxt_to_parametric_parts import (  # noqa: E402
    _read_mphtxt_edge_entities,
)


DEFAULT_BASE_NAME = "base4"
DEFAULT_SOURCE_ROOT = Path("data/outputs_merged_td_all_success_pa_ext0520")
DEFAULT_STRUCTURE_INDEX = DEFAULT_SOURCE_ROOT / "structure_file_index.csv"
DEFAULT_MANIFEST = Path(
    "data/outputs_merged_td_csv_periodic_ext0520_v2/geometry/parts_manifest.json"
)
DEFAULT_OUT_DIR = Path("reports/gec_ccp_geometry")

MODEL_DEFINITION_URL = (
    "https://doc.comsol.com/6.4/doc/"
    "com.comsol.help.models.plasma.argon_gec_ccp/argon_gec_ccp.html"
)
MODEL_PAGE_URL = "https://www.comsol.com/model/gec-ccp-reactor-argon-chemistry-55011"
MODEL_DIELECTRIC_CONTACT_URL = (
    "https://doc.comsol.com/6.4/doc/"
    "com.comsol.help.plasma/plasma_ug_plasma.09.16.html"
)

FIGURE_VARIANTS = ("color", "outline")
ROLE_ENTITY_IDS = {
    "axis_of_symmetry": (0,),
    "driven_electrode": (1,),
    "dielectric_contact": (3, 5),
    "grounded_electrode_and_walls": (2, 4, 6, 7, 8, 9, 10, 11),
}

COLORS = {
    "plasma": "#56B4E9",
    "driven": "#D55E00",
    "dielectric": "#009E73",
    "grounded": "#374151",
    "axis": "#111827",
    "spine": "#9CA3AF",
    "tick": "#374151",
}
OUTLINE_COLOR = "#111111"
OUTLINE_SPINE = "#9A9A9A"
OUTLINE_TICK = "#3F3F3F"


def _cm(value_m: float) -> float:
    return round(100.0 * float(value_m), 10)


@dataclass(frozen=True)
class BoundarySegment:
    """One straight COMSOL geometric boundary entity, in meters."""

    entity_id: int
    r0: float
    r1: float
    z0: float
    z1: float

    @property
    def orientation(self) -> str:
        if np.isclose(self.z0, self.z1, atol=1.0e-12):
            return "horizontal"
        if np.isclose(self.r0, self.r1, atol=1.0e-12):
            return "vertical"
        return "oblique"

    @property
    def length(self) -> float:
        return float(np.hypot(self.r1 - self.r0, self.z1 - self.z0))


@dataclass(frozen=True)
class GeometryDimensions:
    """Dimensions derived from the verified boundary topology, in meters."""

    discharge_gap: float
    inner_radius: float
    outer_radius: float
    chamber_bottom: float
    chamber_top: float
    chamber_height: float
    driven_electrode_radius: float
    grounded_electrode_radius: float
    powered_side_dielectric_break: float


@dataclass(frozen=True)
class GecCcpGeometry:
    base_name: str
    td_value: float
    boundaries: tuple[BoundarySegment, ...]
    dimensions: GeometryDimensions
    domain_vertices_m: tuple[tuple[float, float], ...]
    mphtxt_path: Path
    structure_index_path: Path
    manifest_path: Path

    def boundary(self, entity_id: int) -> BoundarySegment:
        for segment in self.boundaries:
            if segment.entity_id == entity_id:
                return segment
        raise KeyError(entity_id)


def _extract_boundary_segments(
    vertices: np.ndarray,
    edges: np.ndarray,
    entity_ids: np.ndarray,
) -> tuple[BoundarySegment, ...]:
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or entity_ids.shape != (edges.shape[0],):
        raise ValueError("invalid edge connectivity arrays")

    segments: list[BoundarySegment] = []
    for entity_id in sorted(set(entity_ids.tolist())):
        selected = edges[entity_ids == entity_id]
        edge_points = vertices[selected]
        delta = np.abs(edge_points[:, 1] - edge_points[:, 0])
        if np.any((delta[:, 0] > 1.0e-11) & (delta[:, 1] > 1.0e-11)):
            raise ValueError(f"boundary entity {entity_id} contains an oblique mesh edge")
        points = edge_points.reshape(-1, 2)
        r_min = float(np.min(points[:, 0]))
        r_max = float(np.max(points[:, 0]))
        z_min = float(np.min(points[:, 1]))
        z_max = float(np.max(points[:, 1]))
        horizontal = np.isclose(z_min, z_max, atol=1.0e-11)
        vertical = np.isclose(r_min, r_max, atol=1.0e-11)
        if horizontal == vertical:
            raise ValueError(f"boundary entity {entity_id} is not one nonzero straight segment")
        edge_length = float(np.sum(np.linalg.norm(edge_points[:, 1] - edge_points[:, 0], axis=1)))
        extent = (r_max - r_min) if horizontal else (z_max - z_min)
        if not np.isclose(edge_length, extent, rtol=1.0e-8, atol=1.0e-11):
            raise ValueError(f"boundary entity {entity_id} is disconnected or overlapping")
        if horizontal:
            segments.append(BoundarySegment(entity_id, r_min, r_max, z_min, z_min))
        else:
            segments.append(BoundarySegment(entity_id, r_min, r_min, z_min, z_max))
    return tuple(segments)


def _assert_segment(
    segment: BoundarySegment,
    expected: tuple[float, float, float, float],
) -> None:
    actual = np.asarray((segment.r0, segment.r1, segment.z0, segment.z1))
    if not np.allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"boundary {segment.entity_id} topology mismatch: "
            f"actual={tuple(actual)}, expected={expected}"
        )


def _validate_boundary_topology(
    boundaries: Sequence[BoundarySegment],
    *,
    base_name: str,
) -> tuple[GeometryDimensions, tuple[tuple[float, float], ...]]:
    by_id = {segment.entity_id: segment for segment in boundaries}
    expected_ids = set(range(12))
    if set(by_id) != expected_ids or len(boundaries) != len(expected_ids):
        raise ValueError(f"expected boundary entity ids 0..11, found {sorted(by_id)}")

    role_ids = [entity_id for ids in ROLE_ENTITY_IDS.values() for entity_id in ids]
    if len(role_ids) != len(set(role_ids)) or set(role_ids) != expected_ids:
        raise RuntimeError("boundary role map must be disjoint and cover all entities")

    gap_bottom = by_id[0].z0
    gap_top = by_id[0].z1
    driven_radius = by_id[1].r1
    grounded_radius = by_id[2].r1
    inner_radius = by_id[3].r1
    outer_radius = by_id[6].r1
    chamber_bottom = by_id[5].z0
    chamber_top = by_id[7].z1

    expected = {
        0: (0.0, 0.0, gap_bottom, gap_top),
        1: (0.0, driven_radius, gap_bottom, gap_bottom),
        2: (0.0, grounded_radius, gap_top, gap_top),
        3: (driven_radius, inner_radius, gap_bottom, gap_bottom),
        4: (grounded_radius, inner_radius, gap_top, gap_top),
        5: (inner_radius, inner_radius, chamber_bottom, gap_bottom),
        6: (inner_radius, outer_radius, chamber_bottom, chamber_bottom),
        7: (inner_radius, inner_radius, gap_top, chamber_top),
        8: (inner_radius, outer_radius, chamber_top, chamber_top),
        9: (outer_radius, outer_radius, chamber_bottom, gap_bottom),
        10: (outer_radius, outer_radius, gap_bottom, gap_top),
        11: (outer_radius, outer_radius, gap_top, chamber_top),
    }
    for entity_id, coordinates in expected.items():
        _assert_segment(by_id[entity_id], coordinates)

    dimensions = GeometryDimensions(
        discharge_gap=gap_top - gap_bottom,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        chamber_bottom=chamber_bottom,
        chamber_top=chamber_top,
        chamber_height=chamber_top - chamber_bottom,
        driven_electrode_radius=driven_radius,
        grounded_electrode_radius=grounded_radius,
        powered_side_dielectric_break=inner_radius - driven_radius,
    )
    expected_envelope = {
        "discharge_gap": 0.0254,
        "inner_radius": 0.0538,
        "outer_radius": 0.1016,
        "chamber_bottom": -0.0381,
        "chamber_top": 0.0635,
        "chamber_height": 0.1016,
        "grounded_electrode_radius": 0.0508,
    }
    for name, value in expected_envelope.items():
        if not np.isclose(getattr(dimensions, name), value, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"unexpected GEC-CCP {name}: {getattr(dimensions, name)} != {value}")
    if base_name == "base4":
        if not np.isclose(dimensions.driven_electrode_radius, 0.0508, atol=1.0e-9):
            raise ValueError("base4 driven electrode radius must be 50.8 mm")
        if not np.isclose(dimensions.powered_side_dielectric_break, 0.003, atol=1.0e-9):
            raise ValueError("base4 dielectric break must be 3 mm")

    domain_vertices = (
        (0.0, gap_bottom),
        (inner_radius, gap_bottom),
        (inner_radius, chamber_bottom),
        (outer_radius, chamber_bottom),
        (outer_radius, chamber_top),
        (inner_radius, chamber_top),
        (inner_radius, gap_top),
        (0.0, gap_top),
    )
    return dimensions, domain_vertices


def _read_td_value(path: Path, base_name: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"structure index not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("base_name", "").strip() == base_name]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one structure-index row for {base_name}, found {len(rows)}")
    return float(rows[0]["td_value"])


def _validate_manifest(path: Path, base_name: str, td_value: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"parts manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("geometry_fidelity") != "mphtxt_edg_connectivity" or payload.get("approximate") is not False:
        raise ValueError(f"manifest does not describe exact MPHTXT connectivity: {path}")
    if base_name not in payload.get("part_ids", []):
        raise ValueError(f"{base_name} is absent from {path}")
    conditions = payload.get("alternative_conditions", {}).get(base_name, {})
    if "Td" not in conditions or not np.isclose(float(conditions["Td"]), td_value, atol=1.0e-12):
        raise ValueError(f"Td mapping for {base_name} differs between manifest and structure index")
    if payload.get("geometric_entity_ids") != list(range(12)):
        raise ValueError(f"unexpected boundary entity ids in {path}")


def load_geometry(
    *,
    base_name: str,
    source_root: Path,
    structure_index_path: Path,
    manifest_path: Path,
) -> GecCcpGeometry:
    td_value = _read_td_value(structure_index_path, base_name)
    _validate_manifest(manifest_path, base_name, td_value)
    mphtxt_path = source_root / "structure" / base_name / f"argon_gec_ccp_{base_name}.mphtxt"
    if not mphtxt_path.exists():
        raise FileNotFoundError(f"MPHTXT geometry not found: {mphtxt_path}")
    vertices, edges, entity_ids = _read_mphtxt_edge_entities(mphtxt_path)
    boundaries = _extract_boundary_segments(vertices, edges, entity_ids)
    dimensions, domain_vertices = _validate_boundary_topology(boundaries, base_name=base_name)
    return GecCcpGeometry(
        base_name=base_name,
        td_value=td_value,
        boundaries=boundaries,
        dimensions=dimensions,
        domain_vertices_m=domain_vertices,
        mphtxt_path=mphtxt_path,
        structure_index_path=structure_index_path,
        manifest_path=manifest_path,
    )


def _plot_role(
    ax: plt.Axes,
    geometry: GecCcpGeometry,
    entity_ids: Sequence[int],
    *,
    color: str,
    linewidth: float,
    linestyle: str | tuple[int, tuple[float, ...]] = "solid",
    zorder: int,
) -> None:
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
            zorder=zorder,
        )


def create_geometry_figure(
    geometry: GecCcpGeometry,
    *,
    variant: str = "color",
) -> plt.Figure:
    if variant not in FIGURE_VARIANTS:
        raise ValueError(f"variant must be one of {FIGURE_VARIANTS}, got {variant!r}")
    is_color = variant == "color"
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.9, 6.3))
        domain_cm = 100.0 * np.asarray(geometry.domain_vertices_m, dtype=np.float64)
        ax.add_patch(
            Polygon(
                domain_cm,
                closed=True,
                facecolor=COLORS["plasma"] if is_color else "none",
                edgecolor="none",
                alpha=0.34 if is_color else 1.0,
                zorder=0,
            )
        )

        if is_color:
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["grounded_electrode_and_walls"],
                color=COLORS["grounded"],
                linewidth=2.35,
                zorder=3,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["driven_electrode"],
                color=COLORS["driven"],
                linewidth=4.0,
                zorder=5,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["dielectric_contact"],
                color=COLORS["dielectric"],
                linewidth=4.0,
                zorder=6,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["axis_of_symmetry"],
                color=COLORS["axis"],
                linewidth=1.25,
                linestyle=(0, (4.0, 3.0)),
                zorder=7,
            )
            handles = [
                Patch(facecolor=COLORS["plasma"], edgecolor="none", alpha=0.34, label="Plasma domain"),
                Line2D([0], [0], color=COLORS["driven"], linewidth=4.0, label="Driven electrode"),
                Line2D(
                    [0],
                    [0],
                    color=COLORS["dielectric"],
                    linewidth=4.0,
                    label="Dielectric-contact boundary",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLORS["grounded"],
                    linewidth=2.35,
                    label="Grounded electrode / walls",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLORS["axis"],
                    linewidth=1.25,
                    linestyle=(0, (4.0, 3.0)),
                    label="Axis of symmetry",
                ),
            ]
        else:
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["grounded_electrode_and_walls"],
                color=OUTLINE_COLOR,
                linewidth=1.25,
                zorder=3,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["driven_electrode"],
                color=OUTLINE_COLOR,
                linewidth=2.6,
                zorder=5,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["dielectric_contact"],
                color=OUTLINE_COLOR,
                linewidth=2.0,
                linestyle=(0, (4.0, 2.0, 1.0, 2.0)),
                zorder=6,
            )
            _plot_role(
                ax,
                geometry,
                ROLE_ENTITY_IDS["axis_of_symmetry"],
                color=OUTLINE_COLOR,
                linewidth=1.1,
                linestyle=(0, (4.0, 3.0)),
                zorder=7,
            )
            handles = [
                Line2D([0], [0], color=OUTLINE_COLOR, linewidth=2.6, label="Driven electrode"),
                Line2D(
                    [0],
                    [0],
                    color=OUTLINE_COLOR,
                    linewidth=2.0,
                    linestyle=(0, (4.0, 2.0, 1.0, 2.0)),
                    label="Dielectric-contact boundary",
                ),
                Line2D(
                    [0],
                    [0],
                    color=OUTLINE_COLOR,
                    linewidth=1.25,
                    label="Grounded electrode / walls",
                ),
                Line2D(
                    [0],
                    [0],
                    color=OUTLINE_COLOR,
                    linewidth=1.1,
                    linestyle=(0, (4.0, 3.0)),
                    label="Axis of symmetry",
                ),
            ]

        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.025, 0.5),
            frameon=False,
            borderaxespad=0.0,
            handlelength=2.4,
            labelspacing=0.9,
        )
        dimensions_cm = {key: _cm(value) for key, value in asdict(geometry.dimensions).items()}
        ax.set_xlim(-0.35, dimensions_cm["outer_radius"] + 0.35)
        ax.set_ylim(dimensions_cm["chamber_bottom"] - 0.35, dimensions_cm["chamber_top"] + 0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$r$ (cm)")
        ax.set_ylabel(r"$z$ (cm)")
        ax.set_xticks(np.arange(0.0, 10.1, 2.0))
        ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color(COLORS["spine"] if is_color else OUTLINE_SPINE)
            ax.spines[name].set_linewidth(0.8)
        ax.tick_params(
            colors=COLORS["tick"] if is_color else OUTLINE_TICK,
            width=0.8,
            length=3.5,
        )
        fig.subplots_adjust(left=0.10, right=0.72, bottom=0.13, top=0.97)
    return fig


def save_geometry_figure(
    *,
    fig: plt.Figure,
    out_dir: Path,
    prefix: str,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    export_style = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(export_style):
        for suffix in formats:
            path = out_dir / f"{prefix}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            outputs.append(path)
    return tuple(outputs)


def _write_metadata(
    *,
    out_dir: Path,
    prefix: str,
    geometry: GecCcpGeometry,
    variant: str,
    outputs: Sequence[Path],
) -> Path:
    dimensions_cm = {key: _cm(value) for key, value in asdict(geometry.dimensions).items()}
    payload = {
        "figure": "Conference-ready representative GEC-CCP computational geometry",
        "figure_variant": variant,
        "representative_alternative": geometry.base_name,
        "representative_rationale": (
            "base4 matches the standard COMSOL dThick=3 mm dielectric break"
        ),
        "coordinate_system": "2D axisymmetric r-z section",
        "length_unit": "cm",
        "dataset_condition_record": {
            "Td": geometry.td_value,
            "interpretation_note": "Td is retained as a dataset condition and is not treated as dThick.",
        },
        "geometry_dimensions_cm": dimensions_cm,
        "boundary_roles_zero_based_mphtxt_entity_id": {
            role: list(entity_ids) for role, entity_ids in ROLE_ENTITY_IDS.items()
        },
        "boundary_segments_cm": [
            {
                **asdict(segment),
                "r0": _cm(segment.r0),
                "r1": _cm(segment.r1),
                "z0": _cm(segment.z0),
                "z1": _cm(segment.z1),
            }
            for segment in geometry.boundaries
        ],
        "dielectric_representation": (
            "The source model provides a Dielectric Contact boundary condition, not a solid/material "
            "dielectric domain; the figure therefore shows only the verified contact boundaries."
        ),
        "material_claims": "No dielectric or electrode material is inferred from the available geometry data.",
        "omitted_nonphysical_geometry": "Mesh-control edges are not presentation parts and are omitted.",
        "sources": {
            "mphtxt_geometry": str(geometry.mphtxt_path),
            "structure_index_csv": str(geometry.structure_index_path),
            "parts_manifest_json": str(geometry.manifest_path),
            "comsol_model_definition": MODEL_DEFINITION_URL,
            "comsol_model_page": MODEL_PAGE_URL,
            "comsol_dielectric_contact_definition": MODEL_DIELECTRIC_CONTACT_URL,
        },
        "output_files": [path.name for path in outputs],
    }
    path = out_dir / f"{prefix}_metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_index(
    *,
    out_dir: Path,
    geometry: GecCcpGeometry,
    artifacts: dict[str, tuple[str, Sequence[Path], Path]],
) -> Path:
    sections: list[str] = []
    labels = {"color": "Color", "outline": "Outline only"}
    for variant in FIGURE_VARIANTS:
        if variant not in artifacts:
            continue
        prefix, outputs, metadata_path = artifacts[variant]
        formats = ", ".join(f"[{path.suffix[1:].upper()}]({path.name})" for path in outputs)
        sections.append(
            f"## {labels[variant]}\n\n"
            f"![GEC-CCP geometry ({variant})]({prefix}.png)\n\n"
            f"- Formats: {formats}\n"
            f"- Provenance: [{metadata_path.name}]({metadata_path.name})"
        )
    text = (
        "# GEC-CCP representative geometry\n\n"
        f"- Representative alternative: `{geometry.base_name}`\n"
        "- Selection basis: matches the standard COMSOL 3 mm dielectric break\n"
        "- Coordinate system: axisymmetric r-z section, centimeters\n\n"
        + "\n\n".join(sections)
        + "\n\nThe dielectric is represented faithfully as a boundary contact, not as an "
        "invented solid region. Mesh-control edges are omitted.\n"
    )
    path = out_dir / "index.md"
    path.write_text(text, encoding="utf-8")
    return path


def _variant_prefix(prefix: str, variant: str) -> str:
    if variant == "color":
        return prefix
    suffix = "_overview"
    if prefix.endswith(suffix):
        return f"{prefix[:-len(suffix)]}_outline"
    return f"{prefix}_outline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name", default=DEFAULT_BASE_NAME)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--structure-index", type=Path, default=DEFAULT_STRUCTURE_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="gec_ccp_geometry_overview")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=FIGURE_VARIANTS,
        default=FIGURE_VARIANTS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    geometry = load_geometry(
        base_name=args.base_name,
        source_root=args.source_root,
        structure_index_path=args.structure_index,
        manifest_path=args.manifest,
    )
    artifacts: dict[str, tuple[str, Sequence[Path], Path]] = {}
    generated_paths: list[Path] = []
    for variant in dict.fromkeys(args.variants):
        prefix = _variant_prefix(args.prefix, variant)
        fig = create_geometry_figure(geometry, variant=variant)
        try:
            outputs = save_geometry_figure(
                fig=fig,
                out_dir=args.out_dir,
                prefix=prefix,
                formats=args.formats,
                dpi=args.dpi,
            )
        finally:
            plt.close(fig)
        metadata_path = _write_metadata(
            out_dir=args.out_dir,
            prefix=prefix,
            geometry=geometry,
            variant=variant,
            outputs=outputs,
        )
        artifacts[variant] = (prefix, outputs, metadata_path)
        generated_paths.extend((*outputs, metadata_path))
    index_path = _write_index(out_dir=args.out_dir, geometry=geometry, artifacts=artifacts)
    for path in (*generated_paths, index_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
