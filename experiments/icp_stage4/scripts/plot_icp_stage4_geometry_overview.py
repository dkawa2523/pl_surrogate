"""Create a conference-ready overview of the GEC-ICP computational geometry.

The fixed geometry follows COMSOL's ``icp_coil_optimization`` application
definition.  Case-specific coil rectangles are read from the enriched Stage 4
dataset and checked against the exported coil mask before plotting.  The
nonphysical integration segment at z=7 cm is intentionally omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


DEFAULT_CASE_ID = "case_g002_op01"
DEFAULT_STRUCTURE_ROOT = Path(
    "data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2/structure_features"
)
DEFAULT_LAYOUT_ROOT = Path("data/outputs_icp_stage4_enriched_360/structure/coil_layout")
DEFAULT_LABEL_ROOT = Path("data/outputs_icp_stage4_enriched_360/labels")
DEFAULT_SUMMARY_ROOT = Path("data/outputs_icp_stage4_enriched_360/structure/geometry_summary")
DEFAULT_OUT_DIR = Path("reports/gec_icp_geometry")

MODEL_DEFINITION_URL = (
    "https://doc.comsol.com/6.4/doc/"
    "com.comsol.help.models.plasma.icp_coil_optimization/icp_coil_optimization.html"
)
MODEL_PARAMETER_URL = "https://www.comsol.com/model/download/1570841/icp_coil_optimization_parameters.txt"

COLORS = {
    "plasma": "#56B4E9",
    "plasma_edge": "#0072B2",
    "coil": "#E69F00",
    "coil_edge": "#8C5C00",
    "window": "#009E73",
    "window_edge": "#00664B",
    "block": "#8CCBB5",
    "substrate": "#7B8794",
    "substrate_edge": "#3F4852",
    "em_domain": "#F2F4F7",
    "domain_edge": "#374151",
    "axis": "#111827",
    "spine": "#9CA3AF",
    "tick": "#374151",
}

OUTLINE = {
    "component": "#111111",
    "axis": "#111111",
    "spine": "#9A9A9A",
    "tick": "#3F3F3F",
}

FIGURE_VARIANTS = ("color", "outline")


@dataclass(frozen=True)
class GeometrySpec:
    """Fixed COMSOL geometry dimensions, in centimeters."""

    chamber_width: float = 30.0
    chamber_height: float = 22.0
    substrate_width: float = 15.0
    substrate_height: float = 2.0
    dielectric_block_width: float = 5.0
    dielectric_block_height: float = 2.0
    plasma_gap: float = 10.0
    dielectric_window_width: float = 30.0
    dielectric_window_height: float = 2.0

    @property
    def plasma_top(self) -> float:
        return self.substrate_height + self.plasma_gap

    @property
    def window_top(self) -> float:
        return self.plasma_top + self.dielectric_window_height

    @property
    def dielectric_block_right(self) -> float:
        return self.substrate_width + self.dielectric_block_width


@dataclass(frozen=True)
class CoilRectangle:
    index: int
    order: int
    r_min: float
    r_max: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.r_max - self.r_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min


@dataclass(frozen=True)
class CaseGeometry:
    case_id: str
    r_coords: np.ndarray
    z_coords: np.ndarray
    plasma_mask: np.ndarray
    coil_mask: np.ndarray
    valid_field_mask: np.ndarray
    outside_mask: np.ndarray
    coils: tuple[CoilRectangle, ...]
    structure_path: Path
    layout_path: Path
    labels_path: Path
    summary_path: Path


def _read_active_coils(path: Path, case_id: str) -> tuple[CoilRectangle, ...]:
    if not path.exists():
        raise FileNotFoundError(f"coil layout not found: {path}")
    coils: list[CoilRectangle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_case = str(row.get("case_id", "")).strip()
            if row_case and row_case != case_id:
                raise ValueError(f"coil layout case mismatch: expected={case_id}, found={row_case}")
            if int(float(row.get("active", "0") or 0)) <= 0:
                continue
            coil = CoilRectangle(
                index=int(float(row["coil_index"])),
                order=int(float(row["order"])),
                r_min=float(row["r_min"]),
                r_max=float(row["r_max"]),
                z_min=float(row["z_min"]),
                z_max=float(row["z_max"]),
            )
            if coil.width <= 0.0 or coil.height <= 0.0:
                raise ValueError(f"invalid coil rectangle in {path}: {coil}")
            coils.append(coil)
    if not coils:
        raise ValueError(f"no active coils found in {path}")
    return tuple(sorted(coils, key=lambda coil: (coil.order, coil.index)))


def _read_selection_roles(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"selection labels not found: {path}")
    roles: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = str(row.get("role", "")).strip().lower()
            if role:
                roles.add(role)
    required = {"axis", "coil", "dielectric", "plasma", "wall"}
    missing = required - roles
    if missing:
        raise ValueError(f"selection roles missing {sorted(missing)} in {path}")
    return roles


def _expected_masks(
    r_coords: np.ndarray,
    z_coords: np.ndarray,
    spec: GeometrySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_grid, z_grid = np.meshgrid(r_coords, z_coords)
    in_domain = (
        (r_grid >= 0.0)
        & (r_grid <= spec.chamber_width)
        & (z_grid >= 0.0)
        & (z_grid <= spec.chamber_height)
    )
    substrate = (
        (r_grid < spec.substrate_width)
        & (z_grid < spec.substrate_height)
        & in_domain
    )
    dielectric_block = (
        (r_grid >= spec.substrate_width)
        & (r_grid < spec.dielectric_block_right)
        & (z_grid < spec.dielectric_block_height)
        & in_domain
    )
    outside = substrate | dielectric_block
    plasma = (z_grid < spec.plasma_top) & in_domain & ~outside
    valid = in_domain & ~outside
    return plasma, valid, outside, substrate


def _assert_same_mask(name: str, actual: np.ndarray, expected: np.ndarray, source: Path) -> None:
    actual_bool = np.asarray(actual) > 0.5
    expected_bool = np.asarray(expected, dtype=bool)
    if actual_bool.shape != expected_bool.shape:
        raise ValueError(
            f"{name} shape mismatch in {source}: actual={actual_bool.shape}, expected={expected_bool.shape}"
        )
    mismatch = int(np.count_nonzero(actual_bool ^ expected_bool))
    if mismatch:
        raise ValueError(f"{name} differs from the verified COMSOL geometry at {mismatch} cells: {source}")


def _validate_summary(path: Path, spec: GeometrySpec) -> None:
    if not path.exists():
        raise FileNotFoundError(f"geometry summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    bbox = payload.get("geometry_bbox", {})
    expected = {
        "r_min": 0.0,
        "r_max": spec.chamber_width,
        "z_min": 0.0,
        "z_max": spec.chamber_height,
    }
    for key, value in expected.items():
        if key not in bbox or not np.isclose(float(bbox[key]), value, atol=1.0e-8):
            raise ValueError(f"geometry bbox mismatch for {key} in {path}: {bbox.get(key)!r} != {value}")


def _validate_case_geometry(case: CaseGeometry, spec: GeometrySpec) -> None:
    if case.r_coords.ndim != 1 or case.z_coords.ndim != 1:
        raise ValueError("r_coords and z_coords must be one-dimensional")
    if case.r_coords.size < 2 or case.z_coords.size < 2:
        raise ValueError("geometry coordinates must each contain at least two points")
    if np.any(np.diff(case.r_coords) <= 0.0) or np.any(np.diff(case.z_coords) <= 0.0):
        raise ValueError("geometry coordinates must be strictly increasing")

    shape = (case.z_coords.size, case.r_coords.size)
    for name, mask in (
        ("mask_plasma", case.plasma_mask),
        ("mask_coil", case.coil_mask),
        ("valid_field_mask", case.valid_field_mask),
        ("outside_mask", case.outside_mask),
    ):
        if mask.shape != shape:
            raise ValueError(f"{name} shape={mask.shape} does not match coordinate shape={shape}")

    expected_plasma, expected_valid, expected_outside, _ = _expected_masks(
        case.r_coords,
        case.z_coords,
        spec,
    )
    _assert_same_mask("mask_plasma", case.plasma_mask, expected_plasma, case.structure_path)
    _assert_same_mask("valid_field_mask", case.valid_field_mask, expected_valid, case.structure_path)
    _assert_same_mask("outside_mask", case.outside_mask, expected_outside, case.structure_path)

    r_grid, z_grid = np.meshgrid(case.r_coords, case.z_coords)
    expected_coils = np.zeros(shape, dtype=bool)
    for coil in case.coils:
        if (
            coil.r_min < 0.0
            or coil.r_max > spec.chamber_width
            or coil.z_min < spec.window_top
            or coil.z_max > spec.chamber_height
        ):
            raise ValueError(f"coil lies outside the verified upper domain: {coil}")
        expected_coils |= (
            (r_grid >= coil.r_min)
            & (r_grid <= coil.r_max)
            & (z_grid >= coil.z_min)
            & (z_grid <= coil.z_max)
        )
    _assert_same_mask("mask_coil", case.coil_mask, expected_coils, case.structure_path)


def load_case_geometry(
    *,
    case_id: str,
    structure_root: Path,
    layout_root: Path,
    label_root: Path,
    summary_root: Path,
    spec: GeometrySpec,
) -> CaseGeometry:
    structure_path = structure_root / f"{case_id}.npz"
    layout_path = layout_root / f"{case_id}__coil_layout.csv"
    labels_path = label_root / f"{case_id}__selection_entities.csv"
    summary_path = summary_root / f"{case_id}__geometry_summary.json"
    if not structure_path.exists():
        raise FileNotFoundError(f"case structure features not found: {structure_path}")
    with np.load(structure_path, allow_pickle=False) as data:
        required = {
            "r_coords",
            "z_coords",
            "mask_plasma",
            "mask_coil",
            "valid_field_mask",
            "outside_mask",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"case structure features missing {sorted(missing)}: {structure_path}")
        case = CaseGeometry(
            case_id=case_id,
            r_coords=np.asarray(data["r_coords"], dtype=np.float64),
            z_coords=np.asarray(data["z_coords"], dtype=np.float64),
            plasma_mask=np.asarray(data["mask_plasma"]),
            coil_mask=np.asarray(data["mask_coil"]),
            valid_field_mask=np.asarray(data["valid_field_mask"]),
            outside_mask=np.asarray(data["outside_mask"]),
            coils=_read_active_coils(layout_path, case_id),
            structure_path=structure_path,
            layout_path=layout_path,
            labels_path=labels_path,
            summary_path=summary_path,
        )
    _read_selection_roles(labels_path)
    _validate_summary(summary_path, spec)
    _validate_case_geometry(case, spec)
    return case


def _plasma_patch(spec: GeometrySpec, *, variant: str) -> PathPatch:
    vertices = [
        (0.0, spec.substrate_height),
        (spec.dielectric_block_right, spec.substrate_height),
        (spec.dielectric_block_right, 0.0),
        (spec.chamber_width, 0.0),
        (spec.chamber_width, spec.plasma_top),
        (0.0, spec.plasma_top),
        (0.0, spec.substrate_height),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    if variant == "color":
        facecolor = COLORS["plasma"]
        edgecolor = COLORS["plasma_edge"]
        linewidth = 1.5
        alpha = 0.62
    elif variant == "outline":
        facecolor = "none"
        edgecolor = OUTLINE["component"]
        linewidth = 1.35
        alpha = 1.0
    else:
        raise ValueError(f"unknown figure variant: {variant}")
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="miter",
        zorder=2,
    )


def create_geometry_figure(
    case: CaseGeometry,
    spec: GeometrySpec,
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
        "legend.fontsize": 10.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9.2, 5.8))

        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                spec.chamber_width,
                spec.chamber_height,
                facecolor=COLORS["em_domain"] if is_color else "none",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(_plasma_patch(spec, variant=variant))
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                spec.substrate_width,
                spec.substrate_height,
                facecolor=COLORS["substrate"] if is_color else "none",
                edgecolor=COLORS["substrate_edge"] if is_color else OUTLINE["component"],
                linewidth=1.4 if is_color else 1.35,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (spec.substrate_width, 0.0),
                spec.dielectric_block_width,
                spec.dielectric_block_height,
                facecolor=COLORS["block"] if is_color else "none",
                edgecolor=COLORS["window_edge"] if is_color else OUTLINE["component"],
                linewidth=1.4 if is_color else 1.35,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.0, spec.plasma_top),
                spec.dielectric_window_width,
                spec.dielectric_window_height,
                facecolor=COLORS["window"] if is_color else "none",
                edgecolor=COLORS["window_edge"] if is_color else OUTLINE["component"],
                linewidth=1.5 if is_color else 1.35,
                alpha=0.82 if is_color else 1.0,
                zorder=3,
            )
        )
        for coil in case.coils:
            ax.add_patch(
                Rectangle(
                    (coil.r_min, coil.z_min),
                    coil.width,
                    coil.height,
                    facecolor=COLORS["coil"] if is_color else "none",
                    edgecolor=COLORS["coil_edge"] if is_color else OUTLINE["component"],
                    linewidth=1.15 if is_color else 1.35,
                    zorder=5,
                )
            )

        outer_vertices = [
            (0.0, 0.0),
            (spec.chamber_width, 0.0),
            (spec.chamber_width, spec.chamber_height),
            (0.0, spec.chamber_height),
        ]
        ax.add_patch(
            PathPatch(
                MplPath(
                    outer_vertices,
                    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
                ),
                facecolor="none",
                edgecolor=COLORS["domain_edge"] if is_color else OUTLINE["component"],
                linewidth=1.5 if is_color else 1.35,
                joinstyle="miter",
                zorder=6,
            )
        )
        ax.plot(
            [0.0, 0.0],
            [0.0, spec.chamber_height],
            color="white",
            linewidth=2.8,
            linestyle="solid",
            zorder=7,
        )
        ax.plot(
            [0.0, 0.0],
            [0.0, spec.chamber_height],
            color=COLORS["axis"] if is_color else OUTLINE["axis"],
            linewidth=1.25,
            linestyle=(0, (4.0, 3.0)) if is_color else (0, (6.0, 2.2, 1.2, 2.2)),
            zorder=8,
        )

        if is_color:
            handles = [
                Patch(facecolor=COLORS["coil"], edgecolor=COLORS["coil_edge"], label="Induction coils"),
                Patch(facecolor=COLORS["em_domain"], edgecolor="#C7CDD5", label="Upper EM domain"),
                Patch(facecolor=COLORS["window"], edgecolor=COLORS["window_edge"], label="Dielectric window"),
                Patch(facecolor=COLORS["plasma"], edgecolor=COLORS["plasma_edge"], label="Plasma"),
                Patch(facecolor=COLORS["block"], edgecolor=COLORS["window_edge"], label="Dielectric block"),
                Patch(
                    facecolor=COLORS["substrate"],
                    edgecolor=COLORS["substrate_edge"],
                    label="Substrate",
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
        if is_color:
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.025, 0.5),
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.8,
                handleheight=1.0,
                labelspacing=0.9,
            )

        ax.set_xlim(-0.8, spec.chamber_width + 0.8)
        ax.set_ylim(-0.8, spec.chamber_height + 0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$r$ (cm)")
        ax.set_ylabel(r"$z$ (cm)")
        ax.set_xticks(np.arange(0.0, spec.chamber_width + 0.1, 5.0))
        ax.set_yticks(np.arange(0.0, spec.chamber_height + 0.1, 5.0))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color(COLORS["spine"] if is_color else OUTLINE["spine"])
            ax.spines[name].set_linewidth(0.8)
        ax.tick_params(colors=COLORS["tick"] if is_color else OUTLINE["tick"], width=0.8, length=3.5)
        fig.subplots_adjust(
            left=0.10,
            right=0.76 if is_color else 0.97,
            bottom=0.14,
            top=0.97,
        )
    return fig


def _write_metadata(
    *,
    out_dir: Path,
    prefix: str,
    case: CaseGeometry,
    spec: GeometrySpec,
    outputs: Sequence[Path],
    variant: str = "color",
) -> Path:
    payload = {
        "figure": "Conference-ready GEC-ICP computational geometry overview",
        "figure_variant": variant,
        "case_id": case.case_id,
        "coordinate_system": "2D axisymmetric r-z section",
        "length_unit": "cm",
        "active_coil_count": len(case.coils),
        "fixed_geometry_cm": asdict(spec),
        "derived_geometry_cm": {
            "plasma_top": spec.plasma_top,
            "dielectric_window_top": spec.window_top,
            "dielectric_block_right": spec.dielectric_block_right,
        },
        "coils": [asdict(coil) for coil in case.coils],
        "sources": {
            "case_structure_npz": str(case.structure_path),
            "coil_layout_csv": str(case.layout_path),
            "selection_labels_csv": str(case.labels_path),
            "geometry_summary_json": str(case.summary_path),
            "comsol_model_definition": MODEL_DEFINITION_URL,
            "comsol_parameter_file": MODEL_PARAMETER_URL,
        },
        "omitted_nonphysical_geometry": "Optimization integration segment at z=7 cm, 0<=r<=18 cm",
        "output_files": [path.name for path in outputs],
    }
    path = out_dir / f"{prefix}_metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_index(
    *,
    out_dir: Path,
    case: CaseGeometry,
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
            f"![GEC-ICP geometry ({variant})]({prefix}.png)\n\n"
            f"- Formats: {formats}\n"
            f"- Provenance: [{metadata_path.name}]({metadata_path.name})"
        )
    text = (
        "# GEC-ICP geometry overview\n\n"
        f"- Representative case: `{case.case_id}` ({len(case.coils)} active coils)\n"
        "- Coordinate system: axisymmetric r-z section, centimeters\n\n"
        + "\n\n".join(sections)
        + "\n\nThe optimization-only integration segment is intentionally omitted from the presentation figures.\n"
    )
    path = out_dir / "index.md"
    path.write_text(text, encoding="utf-8")
    return path


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--structure-root", type=Path, default=DEFAULT_STRUCTURE_ROOT)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="gec_icp_geometry_overview")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=FIGURE_VARIANTS,
        default=FIGURE_VARIANTS,
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    return parser


def _variant_prefix(prefix: str, variant: str) -> str:
    if variant == "color":
        return prefix
    suffix = "_overview"
    if prefix.endswith(suffix):
        return f"{prefix[:-len(suffix)]}_outline"
    return f"{prefix}_outline"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    spec = GeometrySpec()
    case = load_case_geometry(
        case_id=args.case_id,
        structure_root=args.structure_root,
        layout_root=args.layout_root,
        label_root=args.label_root,
        summary_root=args.summary_root,
        spec=spec,
    )
    artifacts: dict[str, tuple[str, Sequence[Path], Path]] = {}
    generated_paths: list[Path] = []
    for variant in dict.fromkeys(args.variants):
        prefix = _variant_prefix(args.prefix, variant)
        fig = create_geometry_figure(case, spec, variant=variant)
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
            case=case,
            spec=spec,
            outputs=outputs,
            variant=variant,
        )
        artifacts[variant] = (prefix, outputs, metadata_path)
        generated_paths.extend((*outputs, metadata_path))
    _write_index(
        out_dir=args.out_dir,
        case=case,
        artifacts=artifacts,
    )
    for path in (*generated_paths, args.out_dir / "index.md"):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
