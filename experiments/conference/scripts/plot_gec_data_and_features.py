"""Build data-grounded conference figures for the GEC-CCP and GEC-ICP cases.

The script intentionally separates physical source values from display-only
transforms.  Every figure is exported as a high-resolution PNG, a PDF, and an
SVG with vector text/annotations and rasterized scalar fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gec_ccp.scripts.plot_gec_ccp_geometry_overview import (  # noqa: E402
    DEFAULT_MANIFEST as CCP_GEOMETRY_MANIFEST,
    DEFAULT_SOURCE_ROOT as CCP_GEOMETRY_SOURCE_ROOT,
    DEFAULT_STRUCTURE_INDEX as CCP_GEOMETRY_STRUCTURE_INDEX,
    ROLE_ENTITY_IDS as CCP_ROLE_ENTITY_IDS,
    GecCcpGeometry,
    load_geometry as load_ccp_geometry,
)


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials")

CCP_DATA_ROOT = Path("data/outputs_merged_td_csv_periodic_ext0520_v2")
CCP_INDEX_PATH = CCP_DATA_ROOT / "index_78.csv"
CCP_CASE_ID = "case_td003_pp0_3_gamma_007__steady"
CCP_SPLIT_PATH = Path(
    "runs/gec_ccp_nn_operator_comparison_v1/seed_411/n78/ffno/"
    "preprocessing/split/split_interp_v1.json"
)
CCP_SCALER_PATH = Path(
    "runs/gec_ccp_nn_operator_comparison_v1/seed_411/n78/ffno/"
    "preprocessing/scalers/by_split/interp/y_scalers.json"
)

ICP_DATA_ROOT = Path(
    "data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1"
)
ICP_INDEX_PATH = ICP_DATA_ROOT / "index.csv"
ICP_LAYOUT_ROOT = Path("data/outputs_icp_stage4_enriched_360/structure/coil_layout")
ICP_CASE_ID = "case_g002_op01"
ICP_TRAINED_CONFIG = Path("runs/icp_stage4_coil_structure_v1/unet/resolved_config.yaml")
ICP_DIMENSION_CONFIG = Path(
    "configs/experimental/icp_stage4/generated_dimension_parameter_v1/"
    "benchmark_icp_stage4_dimension_parameter_v1_unet.yaml"
)

TARGETS = ("ne", "ni", "Te", "phi")
FORMATS = ("png", "pdf", "svg")

COLORS = {
    "ne": "#0072B2",
    "ni": "#E69F00",
    "Te": "#009E73",
    "phi": "#CC79A7",
    "text": "#1F2937",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "panel": "#F8FAFC",
    "mask": "#F1F3F5",
    "coil": "#E69F00",
    "coil_edge": "#8C5C00",
    "window": "#009E73",
    "driven": "#D55E00",
    "ground": "#374151",
    "axis": "#111827",
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "axes.edgecolor": "#98A2B3",
    "axes.linewidth": 0.8,
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_for_case(path: Path, case_id: str) -> dict[str, str]:
    rows = [row for row in _read_csv(path) if str(row.get("case_id", "")).strip() == case_id]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {case_id!r} in {path}, found {len(rows)}")
    return rows[0]


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


def _write_metadata(out_dir: Path, prefix: str, payload: dict[str, Any]) -> Path:
    path = out_dir / f"{prefix}_metadata.json"
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _save_figure(
    fig: plt.Figure,
    *,
    out_dir: Path,
    prefix: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with plt.rc_context(STYLE):
        for suffix in dict.fromkeys(formats):
            path = out_dir / f"{prefix}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            paths.append(path)
    return paths


def _masked_cmap(name: str) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(COLORS["mask"])
    return cmap


def _finite_masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("no finite values inside the requested mask")
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "q01": float(np.quantile(selected, 0.01)),
        "median": float(np.median(selected)),
        "q99": float(np.quantile(selected, 0.99)),
        "max": float(np.max(selected)),
    }


def _style_scalar_axis(ax: plt.Axes, *, xlabel: bool, ylabel: bool) -> None:
    ax.set_facecolor(COLORS["mask"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(r"$r$ (cm)")
    else:
        ax.tick_params(labelbottom=False)
    if ylabel:
        ax.set_ylabel(r"$z$ (cm)")
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=3.0, width=0.75)


def _overlay_ccp_boundaries(ax: plt.Axes, geometry: GecCcpGeometry) -> None:
    role_styles: dict[str, tuple[str, float, Any]] = {
        "grounded_electrode_and_walls": (COLORS["ground"], 0.9, "solid"),
        "driven_electrode": (COLORS["driven"], 1.8, "solid"),
        "dielectric_contact": (COLORS["window"], 1.8, "solid"),
        "axis_of_symmetry": (COLORS["axis"], 0.8, (0, (3.0, 2.2))),
    }
    for role, entity_ids in CCP_ROLE_ENTITY_IDS.items():
        color, linewidth, linestyle = role_styles[role]
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


def _load_ccp_case() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    row = _row_for_case(CCP_INDEX_PATH, CCP_CASE_ID)
    field_path = CCP_DATA_ROOT / row["fields_npz"]
    structure_path = CCP_DATA_ROOT / row["structure_npz"]
    with np.load(field_path, allow_pickle=False) as data:
        fields = {target: np.asarray(data[target], dtype=np.float64) for target in TARGETS}
    with np.load(structure_path, allow_pickle=False) as data:
        structure = {
            "mask_plasma": np.asarray(data["mask_plasma"]) > 0.5,
            "r_coords": np.asarray(data["r_coords"], dtype=np.float64),
            "z_coords": np.asarray(data["z_coords"], dtype=np.float64),
        }
    expected_shape = (structure["z_coords"].size, structure["r_coords"].size)
    for name, values in fields.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape} != {expected_shape}")
    return fields, structure, row


def create_ccp_physical_fields_figure() -> tuple[plt.Figure, dict[str, Any]]:
    fields, structure, row = _load_ccp_case()
    geometry = load_ccp_geometry(
        base_name=row["base_name"],
        source_root=CCP_GEOMETRY_SOURCE_ROOT,
        structure_index_path=CCP_GEOMETRY_STRUCTURE_INDEX,
        manifest_path=CCP_GEOMETRY_MANIFEST,
    )
    mask = structure["mask_plasma"]
    r_cm = 100.0 * structure["r_coords"]
    z_cm = 100.0 * structure["z_coords"]
    extent = [float(r_cm[0]), float(r_cm[-1]), float(z_cm[0]), float(z_cm[-1])]

    positive_ne = fields["ne"][mask & (fields["ne"] > 0.0)]
    positive_ni = fields["ni"][mask & (fields["ni"] > 0.0)]
    if positive_ne.size == 0 or positive_ni.size == 0:
        raise ValueError("CCP density fields contain no positive plasma values")
    density_limits = (
        float(min(np.log10(positive_ne).min(), np.log10(positive_ni).min())),
        float(max(np.log10(positive_ne).max(), np.log10(positive_ni).max())),
    )
    te_limits = tuple(float(value) for value in np.quantile(fields["Te"][mask], [0.0, 0.995]))
    phi_values = fields["phi"][mask]
    phi_limits = (float(np.min(phi_values)), float(np.max(phi_values)))

    panels = [
        ("ne", np.log10(np.where(fields["ne"] > 0.0, fields["ne"], np.nan)), _masked_cmap("viridis"), Normalize(*density_limits), r"$\log_{10}(n_e\;[\mathrm{m}^{-3}])$"),
        ("ni", np.log10(np.where(fields["ni"] > 0.0, fields["ni"], np.nan)), _masked_cmap("viridis"), Normalize(*density_limits), r"$\log_{10}(n_i\;[\mathrm{m}^{-3}])$"),
        ("Te", fields["Te"], _masked_cmap("plasma"), Normalize(*te_limits), r"$T_e$ (eV)"),
        ("phi", fields["phi"], _masked_cmap("coolwarm"), TwoSlopeNorm(vmin=phi_limits[0], vcenter=0.0, vmax=phi_limits[1]), r"$\phi$ (V)"),
    ]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), constrained_layout=False)
        for panel_index, (ax, (name, display, cmap, norm, title)) in enumerate(zip(axes.flat, panels)):
            shown = np.ma.array(display, mask=(~mask) | (~np.isfinite(display)))
            image = ax.imshow(
                shown,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                aspect="equal",
                rasterized=True,
            )
            _overlay_ccp_boundaries(ax, geometry)
            ax.set_xlim(-0.05, 100.0 * geometry.dimensions.outer_radius + 0.05)
            ax.set_ylim(
                100.0 * geometry.dimensions.chamber_bottom - 0.05,
                100.0 * geometry.dimensions.chamber_top + 0.05,
            )
            ax.set_xticks(np.arange(0.0, 10.1, 2.0))
            ax.set_yticks(np.arange(-4.0, 6.1, 2.0))
            _style_scalar_axis(ax, xlabel=panel_index >= 2, ylabel=panel_index % 2 == 0)
            ax.set_title(title, loc="left", pad=5.0)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            colorbar.ax.tick_params(labelsize=8.0, length=2.5)
            colorbar.outline.set_linewidth(0.6)
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.105, top=0.975, wspace=0.24, hspace=0.16)
        fig.text(
            0.51,
            0.025,
            (
                f"{CCP_CASE_ID}   |   PP0={float(row['PP0']):g}, "
                f"Td={float(row['Td']):.2f}, $\\gamma$={float(row['gamma']):.2f}, "
                f"PA={float(row['PA']):.2f}   |   period-averaged truth"
            ),
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=COLORS["muted"],
        )

    metadata = {
        "figure": "GEC-CCP representative physical-field spatial distributions",
        "case_id": CCP_CASE_ID,
        "geometry_alternative": row["base_name"],
        "conditions": {key: float(row[key]) for key in ("PP0", "Td", "gamma", "PA")},
        "coordinate_system": "2D axisymmetric r-z",
        "coordinate_unit": "cm",
        "source_fields_npz": str(CCP_DATA_ROOT / row["fields_npz"]),
        "source_structure_npz": str(CCP_DATA_ROOT / row["structure_npz"]),
        "source_geometry_mphtxt": str(geometry.mphtxt_path),
        "mask": "mask_plasma",
        "physical_field_statistics_inside_plasma": {
            name: _finite_masked_stats(fields[name], mask) for name in TARGETS
        },
        "display": {
            "ne_ni": "display-only log10; source values remain linear m^-3",
            "Te": f"linear, range=[{te_limits[0]}, {te_limits[1]}] (max to q99.5)",
            "phi": "linear diverging scale centered at 0 V",
            "outside_plasma": "light gray",
            "geometry_overlay": "verified COMSOL MPHTXT boundaries",
            "rendering": "scalar fields rasterized; boundaries, axes, and text remain vector",
        },
    }
    return fig, metadata


def _load_ccp_training_values() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split = json.loads(CCP_SPLIT_PATH.read_text(encoding="utf-8"))
    train_ids = [str(case_id) for case_id in split["train"]]
    rows = {str(row["case_id"]): row for row in _read_csv(CCP_INDEX_PATH)}
    missing = sorted(set(train_ids) - set(rows))
    if missing:
        raise ValueError(f"train split contains case IDs absent from index: {missing[:5]}")

    chunks: dict[str, list[np.ndarray]] = {target: [] for target in TARGETS}
    plasma_cell_count = 0
    for case_id in train_ids:
        row = rows[case_id]
        field_path = CCP_DATA_ROOT / row["fields_npz"]
        structure_path = CCP_DATA_ROOT / row["structure_npz"]
        with np.load(structure_path, allow_pickle=False) as structure:
            mask = np.asarray(structure["mask_plasma"]) > 0.5
        plasma_cell_count += int(mask.sum())
        with np.load(field_path, allow_pickle=False) as fields:
            for target in TARGETS:
                values = np.asarray(fields[target], dtype=np.float64)
                selected = values[mask]
                if not np.all(np.isfinite(selected)):
                    raise ValueError(f"non-finite {target} values in train case {case_id}")
                chunks[target].append(selected)
    values_by_target = {target: np.concatenate(chunks[target]) for target in TARGETS}
    if any(values.size != plasma_cell_count for values in values_by_target.values()):
        raise ValueError("inconsistent target sample count in CCP train plasma")
    return values_by_target, {
        "train_case_ids": train_ids,
        "train_case_count": len(train_ids),
        "plasma_sample_count_per_target": plasma_cell_count,
    }


def _format_scaler_value(target: str, value: float) -> str:
    if target in {"ne", "ni"}:
        return f"{value / 1.0e15:.3f}"
    return f"{value:.3f}"


def create_ccp_zscore_figure() -> tuple[plt.Figure, dict[str, Any]]:
    values_by_target, audit = _load_ccp_training_values()
    scalers = json.loads(CCP_SCALER_PATH.read_text(encoding="utf-8"))
    scaled: dict[str, np.ndarray] = {}
    verification: dict[str, Any] = {}
    for target in TARGETS:
        mu = float(scalers[target]["mean"][0])
        sigma = float(scalers[target]["std"][0])
        values = values_by_target[target]
        computed_mu = float(np.mean(values))
        computed_sigma = float(np.std(values, ddof=0))
        if not np.isclose(computed_mu, mu, rtol=5.0e-7, atol=1.0e-10):
            raise ValueError(f"{target} scaler mean does not match the train plasma values")
        if not np.isclose(computed_sigma, sigma, rtol=5.0e-7, atol=1.0e-10):
            raise ValueError(f"{target} scaler std does not match the train plasma values")
        scaled[target] = (values - mu) / sigma
        verification[target] = {
            "scaler_mean": mu,
            "scaler_std_ddof0": sigma,
            "recomputed_mean": computed_mu,
            "recomputed_std_ddof0": computed_sigma,
            "z_mean": float(np.mean(scaled[target])),
            "z_std_ddof0": float(np.std(scaled[target], ddof=0)),
        }

    raw_units = {
        "ne": (1.0e15, r"$n_e$ ($10^{15}$ m$^{-3}$)"),
        "ni": (1.0e15, r"$n_i$ ($10^{15}$ m$^{-3}$)"),
        "Te": (1.0, r"$T_e$ (eV)"),
        "phi": (1.0, r"$\phi$ (V)"),
    }
    target_titles = {"ne": r"$n_e$", "ni": r"$n_i$", "Te": r"$T_e$", "phi": r"$\phi$"}
    z_bounds = (
        float(min(np.quantile(scaled[target], 0.001) for target in TARGETS)),
        float(max(np.quantile(scaled[target], 0.999) for target in TARGETS)),
    )
    raw_display_ranges: dict[str, list[float]] = {}
    hist_bins = 72

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.5), sharey="row")
        for col, target in enumerate(TARGETS):
            color = COLORS[target]
            scale, xlabel = raw_units[target]
            raw = values_by_target[target] / scale
            raw_lo, raw_hi = (float(value) for value in np.quantile(raw, [0.001, 0.999]))
            raw_display_ranges[target] = [raw_lo, raw_hi]
            weight = np.full(raw.shape, 100.0 / raw.size, dtype=np.float64)
            axes[0, col].hist(
                raw,
                bins=hist_bins,
                range=(raw_lo, raw_hi),
                weights=weight,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[0, col].set_xlim(raw_lo, raw_hi)
            axes[0, col].set_xlabel(xlabel)
            mu = float(scalers[target]["mean"][0])
            sigma = float(scalers[target]["std"][0])
            unit_note = (
                r"$\times 10^{15}\,\mathrm{m}^{-3}$"
                if target in {"ne", "ni"}
                else ("eV" if target == "Te" else "V")
            )
            axes[0, col].set_title(
                target_titles[target]
                + "\n"
                + rf"$\mu={_format_scaler_value(target, mu)},\;\sigma={_format_scaler_value(target, sigma)}$ {unit_note}",
                pad=6.0,
            )

            z = scaled[target]
            z_weight = np.full(z.shape, 100.0 / z.size, dtype=np.float64)
            axes[1, col].axvspan(-1.0, 1.0, color="#DCEAF7", alpha=0.6, linewidth=0.0)
            axes[1, col].hist(
                z,
                bins=hist_bins,
                range=z_bounds,
                weights=z_weight,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[1, col].axvline(0.0, color=COLORS["text"], linewidth=1.0, zorder=3)
            axes[1, col].set_xlim(*z_bounds)
            axes[1, col].set_xlabel(r"$z=(y-\mu_{\rm train})/\sigma_{\rm train}$")

        for row in range(2):
            axes[row, 0].set_ylabel("Train plasma cells (%)")
            axes[row, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
            for ax in axes[row]:
                ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(length=3.0, width=0.75)

        fig.text(0.008, 0.70, "Linear physical value", rotation=90, ha="center", va="center", fontsize=11.0, weight="bold")
        fig.text(0.008, 0.27, "Linear z-score", rotation=90, ha="center", va="center", fontsize=11.0, weight="bold")
        fig.text(
            0.5,
            0.975,
            r"Identity transform  $\longrightarrow$  train-only, plasma-only z-score",
            ha="center",
            va="top",
            fontsize=12.5,
            weight="bold",
        )
        fig.text(
            0.5,
            0.016,
            "No log transform  |  no clipping  |  mean = 0 and standard deviation = 1; distribution shape is unchanged",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=COLORS["muted"],
        )
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.84, wspace=0.20, hspace=0.42)

    metadata = {
        "figure": "Effect of linear target z-score preprocessing for GEC-CCP",
        "preprocessing_contract": {
            "value_transform": "identity for all four targets",
            "scaler": "zscore",
            "fit_split": "interp train split, seed 7",
            "fit_scope": "plasma_only",
            "clip": "none",
            "formula": "z = (y - mean_train_plasma) / std_train_plasma (ddof=0)",
            "interpretation": "z-score centers and rescales; it does not Gaussianize the distribution",
        },
        "sources": {
            "dataset_index": str(CCP_INDEX_PATH),
            "split": str(CCP_SPLIT_PATH),
            "scalers": str(CCP_SCALER_PATH),
        },
        **audit,
        "verification": verification,
        "display": {
            "histogram_bins": hist_bins,
            "raw_ranges_q001_q999": raw_display_ranges,
            "standardized_shared_range_q001_q999_union": list(z_bounds),
            "tails_outside_display_range": "not drawn; all samples were used to verify mean and std",
        },
    }
    return fig, metadata


def _load_icp_case() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    row = _row_for_case(ICP_INDEX_PATH, ICP_CASE_ID)
    field_path = ICP_DATA_ROOT / row["fields_npz"]
    structure_path = ICP_DATA_ROOT / row["structure_npz"]
    with np.load(field_path, allow_pickle=False) as data:
        fields = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in ("ne", "ni", "Te", "phi", "Br", "Bz")
        }
        fields["valid_B"] = (np.asarray(data["valid_Br"]) > 0) & (np.asarray(data["valid_Bz"]) > 0)
    with np.load(structure_path, allow_pickle=False) as data:
        structure = {
            "mask_plasma": np.asarray(data["mask_plasma"]) > 0.5,
            "valid_field_mask": np.asarray(data["valid_field_mask"]) > 0.5,
            "mask_coil": np.asarray(data["mask_coil"]) > 0.5,
            "part_mask_stack": np.asarray(data["part_mask_stack"], dtype=np.float64),
            "r_coords": np.asarray(data["r_coords"], dtype=np.float64),
            "z_coords": np.asarray(data["z_coords"], dtype=np.float64),
        }
        for index in range(1, 7):
            structure[f"sdf_coil_{index:02d}"] = np.asarray(
                data[f"sdf_coil_{index:02d}"], dtype=np.float64
            )
    expected_shape = (structure["z_coords"].size, structure["r_coords"].size)
    for name, values in fields.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape} != {expected_shape}")
    return fields, structure, row


def _read_icp_coils(case_id: str) -> list[dict[str, float | int]]:
    path = ICP_LAYOUT_ROOT / f"{case_id}__coil_layout.csv"
    coils: list[dict[str, float | int]] = []
    for row in _read_csv(path):
        if int(float(row.get("active", 0))) <= 0:
            continue
        coils.append(
            {
                "index": int(float(row["coil_index"])),
                "order": int(float(row["order"])),
                "r_min": float(row["r_min"]),
                "r_max": float(row["r_max"]),
                "z_min": float(row["z_min"]),
                "z_max": float(row["z_max"]),
            }
        )
    if not coils:
        raise ValueError(f"no active coils in {path}")
    return sorted(coils, key=lambda item: (int(item["order"]), int(item["index"])))


def _overlay_icp_geometry(ax: plt.Axes, coils: Iterable[dict[str, float | int]]) -> None:
    line = {"facecolor": "none", "linewidth": 0.8, "zorder": 5}
    ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, edgecolor=COLORS["ground"], **line))
    ax.add_patch(Rectangle((0.0, 0.0), 15.0, 2.0, edgecolor=COLORS["ground"], **line))
    ax.add_patch(Rectangle((15.0, 0.0), 5.0, 2.0, edgecolor=COLORS["window"], **line))
    ax.add_patch(Rectangle((0.0, 12.0), 30.0, 2.0, edgecolor=COLORS["window"], **line))
    for coil in coils:
        ax.add_patch(
            Rectangle(
                (float(coil["r_min"]), float(coil["z_min"])),
                float(coil["r_max"]) - float(coil["r_min"]),
                float(coil["z_max"]) - float(coil["z_min"]),
                facecolor="none",
                edgecolor=COLORS["coil"],
                linewidth=1.0,
                zorder=6,
            )
        )
    ax.plot([0.0, 0.0], [0.0, 22.0], color=COLORS["axis"], linewidth=0.7, linestyle=(0, (3.0, 2.2)), zorder=7)


def create_icp_physical_fields_figure() -> tuple[plt.Figure, dict[str, Any]]:
    fields, structure, row = _load_icp_case()
    coils = _read_icp_coils(ICP_CASE_ID)
    plasma = structure["mask_plasma"]
    valid_b = np.asarray(fields.pop("valid_B"), dtype=bool) & structure["valid_field_mask"]
    b_millitesla = 1.0e3 * np.sqrt(fields["Br"] ** 2 + fields["Bz"] ** 2)
    r = structure["r_coords"]
    z = structure["z_coords"]
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    extent = [float(r[0] - dr / 2.0), float(r[-1] + dr / 2.0), float(z[0] - dz / 2.0), float(z[-1] + dz / 2.0)]

    positive_ne = fields["ne"][plasma & (fields["ne"] > 0.0)]
    ne_log = np.log10(np.where(fields["ne"] > 0.0, fields["ne"], np.nan))
    ne_limits = (float(np.log10(np.min(positive_ne))), float(np.log10(np.max(positive_ne))))
    te_limits = tuple(float(value) for value in np.quantile(fields["Te"][plasma], [0.0, 1.0]))
    phi_limits = tuple(float(value) for value in np.quantile(fields["phi"][plasma], [0.0, 1.0]))
    b_max = float(np.max(b_millitesla[valid_b]))

    panels = [
        (ne_log, plasma, _masked_cmap("viridis"), Normalize(*ne_limits), r"$\log_{10}(n_e\;[\mathrm{m}^{-3}])$", "ne"),
        (fields["Te"], plasma, _masked_cmap("plasma"), Normalize(*te_limits), r"$T_e$ (eV)", "Te"),
        (fields["phi"], plasma, _masked_cmap("cividis"), Normalize(*phi_limits), r"$\phi$ (V)", "phi"),
        (b_millitesla, valid_b, _masked_cmap("magma"), PowerNorm(gamma=0.55, vmin=0.0, vmax=b_max), r"$|\mathbf{B}|$ (mT)", "B_magnitude"),
    ]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.1), constrained_layout=False)
        for panel_index, (ax, (display, mask, cmap, norm, title, _name)) in enumerate(zip(axes.flat, panels)):
            shown = np.ma.array(display, mask=(~mask) | (~np.isfinite(display)))
            image = ax.imshow(
                shown,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                aspect="equal",
                rasterized=True,
            )
            _overlay_icp_geometry(ax, coils)
            ax.set_xlim(0.0, 30.0)
            ax.set_ylim(0.0, 22.0)
            ax.set_xticks(np.arange(0.0, 30.1, 5.0))
            ax.set_yticks(np.arange(0.0, 20.1, 5.0))
            _style_scalar_axis(ax, xlabel=panel_index >= 2, ylabel=panel_index % 2 == 0)
            ax.set_title(title, loc="left", pad=5.0)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
            colorbar.ax.tick_params(labelsize=8.0, length=2.5)
            colorbar.outline.set_linewidth(0.6)
        fig.subplots_adjust(left=0.07, right=0.95, bottom=0.105, top=0.98, wspace=0.20, hspace=0.15)
        fig.text(
            0.51,
            0.024,
            (
                f"{ICP_CASE_ID}   |   pp={float(row['pp']):.3f}, pp0={float(row['pp0']):.5f}, "
                f"$N_{{coil}}$={int(float(row['nncoil']))}   |   simulation truth"
            ),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=COLORS["muted"],
        )

    metadata = {
        "figure": "GEC-ICP representative physical-field spatial distributions",
        "case_id": ICP_CASE_ID,
        "conditions_and_geometry_parameters": {
            key: float(row[key]) for key in ("pp", "pp0", "llcoil", "rrc", "nncoil", "rrce", "zzc")
        },
        "coordinate_system": "2D axisymmetric r-z",
        "coordinate_unit": "cm",
        "source_fields_npz": str(ICP_DATA_ROOT / row["fields_npz"]),
        "source_structure_npz": str(ICP_DATA_ROOT / row["structure_npz"]),
        "source_coil_layout": str(ICP_LAYOUT_ROOT / f"{ICP_CASE_ID}__coil_layout.csv"),
        "masks": {
            "ne_Te_phi": "mask_plasma",
            "B_magnitude": "valid_Br AND valid_Bz AND valid_field_mask",
        },
        "physical_field_statistics": {
            "ne_inside_plasma": _finite_masked_stats(fields["ne"], plasma),
            "Te_inside_plasma": _finite_masked_stats(fields["Te"], plasma),
            "phi_inside_plasma": _finite_masked_stats(fields["phi"], plasma),
            "B_magnitude_mT_inside_valid_field": _finite_masked_stats(b_millitesla, valid_b),
        },
        "display": {
            "ne": "display-only log10; source values remain linear m^-3",
            "Te_phi": "linear full case range",
            "B_magnitude": "linear values with power-law color normalization gamma=0.55",
            "outside_valid_mask": "light gray",
            "geometry_overlay": "fixed chamber/window/substrate outlines and case-specific coil-layout rectangles",
            "rendering": "scalar fields rasterized; geometry, axes, and text remain vector",
        },
    }
    return fig, metadata


def _draw_dimension_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "<->", "color": COLORS["text"], "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0},
        zorder=8,
    )
    ax.text(
        0.5 * (start[0] + end[0]) + offset[0],
        0.5 * (start[1] + end[1]) + offset[1],
        label,
        ha="center",
        va="center",
        fontsize=9.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
        zorder=9,
    )


def _draw_icp_dimension_panel(
    ax: plt.Axes,
    *,
    row: dict[str, str],
    coils: Sequence[dict[str, float | int]],
) -> None:
    ax.set_facecolor("#F8FAFC")
    ax.add_patch(Rectangle((0.0, 0.0), 30.0, 22.0, facecolor="white", edgecolor=COLORS["ground"], linewidth=1.0))
    ax.add_patch(Rectangle((0.0, 0.0), 15.0, 2.0, facecolor="#7B8794", edgecolor=COLORS["ground"], linewidth=0.8, alpha=0.8))
    ax.add_patch(Rectangle((15.0, 0.0), 5.0, 2.0, facecolor="#8CCBB5", edgecolor=COLORS["window"], linewidth=0.8, alpha=0.85))
    ax.add_patch(Rectangle((0.0, 12.0), 30.0, 2.0, facecolor="#76C8B2", edgecolor=COLORS["window"], linewidth=0.9, alpha=0.82))
    for coil in coils:
        ax.add_patch(
            Rectangle(
                (float(coil["r_min"]), float(coil["z_min"])),
                float(coil["r_max"]) - float(coil["r_min"]),
                float(coil["z_max"]) - float(coil["z_min"]),
                facecolor=COLORS["coil"],
                edgecolor=COLORS["coil_edge"],
                linewidth=0.8,
                zorder=5,
            )
        )

    r_c = float(row["rrc"])
    r_ce = float(row["rrce"])
    z_c = float(row["zzc"])
    l_c = float(row["llcoil"])
    n_c = int(float(row["nncoil"]))
    first = coils[0]
    pitch = (r_ce - r_c) / n_c

    ax.plot([r_c, r_c], [14.0, 19.3], color=COLORS["muted"], linewidth=0.65, linestyle=(0, (2.0, 2.0)))
    ax.plot([r_ce, r_ce], [14.0, 19.3], color=COLORS["muted"], linewidth=0.65, linestyle=(0, (2.0, 2.0)))
    _draw_dimension_arrow(ax, (0.0, 17.5), (r_c, 17.5), r"$r_c$", offset=(0.0, 0.45))
    _draw_dimension_arrow(ax, (r_c, 19.0), (r_ce, 19.0), r"$r_{ce}-r_c$", offset=(0.0, 0.42))
    # Brackets remain readable for sub-centimeter dimensions at the full-domain scale.
    l_bracket_x = float(first["r_max"]) + 0.55
    for x_value, y0, y1, label, label_dx in (
        (l_bracket_x, float(first["z_min"]), float(first["z_max"]), r"$l_c$", 0.75),
        (1.0, 14.0, 14.0 + z_c, r"$z_c$", 0.75),
    ):
        ax.plot([x_value, x_value], [y0, y1], color=COLORS["text"], linewidth=1.05, zorder=8)
        ax.plot([x_value - 0.18, x_value + 0.18], [y0, y0], color=COLORS["text"], linewidth=1.05, zorder=8)
        ax.plot([x_value - 0.18, x_value + 0.18], [y1, y1], color=COLORS["text"], linewidth=1.05, zorder=8)
        ax.text(
            x_value + label_dx,
            0.5 * (y0 + y1),
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
            zorder=9,
        )
    if len(coils) >= 2:
        _draw_dimension_arrow(
            ax,
            (float(coils[0]["r_min"]), 16.4),
            (float(coils[1]["r_min"]), 16.4),
            r"$p=(r_{ce}-r_c)/N_c$",
            offset=(0.0, 0.45),
        )
    ax.text(17.0, 17.45, rf"$N_c={n_c}$", ha="center", va="center", fontsize=10.0, weight="bold")
    ax.text(
        15.0,
        7.3,
        r"$\mathbf{g}=[l_c,\;r_c,\;N_c,\;r_{ce},\;z_c]$",
        ha="center",
        va="center",
        fontsize=12.0,
        weight="bold",
    )
    ax.text(
        15.0,
        5.4,
        rf"$[{l_c:.4f},\;{r_c:.4f},\;{n_c},\;{r_ce:.4f},\;{z_c:.4f}]$",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    ax.text(
        15.0,
        4.45,
        r"length entries: cm; $N_c$: dimensionless",
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
    )
    ax.text(
        15.0,
        3.55,
        r"train-fitted robust scaling $\rightarrow$ condition vector",
        ha="center",
        va="center",
        fontsize=8.4,
        color=COLORS["muted"],
    )
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(0.0, 22.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$r$ (cm)")
    ax.set_ylabel(r"$z$ (cm)")
    ax.set_xticks([0.0, 10.0, 20.0, 30.0])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0])
    ax.set_title("A   Dimension-parameter encoding", loc="left", pad=7.0, weight="bold")


def create_icp_geometry_features_figure() -> tuple[plt.Figure, dict[str, Any]]:
    _fields, structure, row = _load_icp_case()
    coils = _read_icp_coils(ICP_CASE_ID)
    r = structure["r_coords"]
    z = structure["z_coords"]
    dr = float(np.median(np.diff(r)))
    dz = float(np.median(np.diff(z)))
    if not np.isclose(dr, dz, atol=1.0e-8):
        raise ValueError("ICP SDF conversion to centimeters requires an isotropic grid")
    extent = [float(r[0] - dr / 2.0), float(r[-1] + dr / 2.0), float(z[0] - dz / 2.0), float(z[-1] + dz / 2.0)]
    sdf_px = [structure[f"sdf_coil_{index:02d}"] for index in range(1, 7)]
    sdf_cm = [sdf * dr for sdf in sdf_px]
    union_sdf_cm = np.minimum.reduce(sdf_cm)
    source_tau_px = 4.0
    source_sum = np.sum(
        np.stack([np.exp(-np.maximum(sdf, 0.0) / source_tau_px) for sdf in sdf_px], axis=0),
        axis=0,
    )
    union_mask = np.maximum.reduce(structure["part_mask_stack"] > 0.5)
    # The grid SDF uses an exact zero-valued boundary band on both sides of the
    # rasterized interface.  Zero is therefore intentionally ambiguous, while
    # strictly negative/positive values must agree with inside/outside.
    sign_contradictions = int(
        np.count_nonzero((union_sdf_cm < 0.0) & (~union_mask))
        + np.count_nonzero((union_sdf_cm > 0.0) & union_mask)
    )
    if sign_contradictions:
        raise ValueError(
            f"union SDF strict sign contradicts the coil mask at {sign_contradictions} cells"
        )
    zero_boundary_cells = int(np.count_nonzero(union_sdf_cm == 0.0))

    sdf_cmap = LinearSegmentedColormap.from_list(
        "sdf_conference",
        ["#2B59C3", "#F7F7F7", "#D55E00"],
        N=256,
    )
    sdf_cmap.set_bad(COLORS["mask"])
    sdf_norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=5.0)
    source_norm = Normalize(vmin=0.0, vmax=float(np.max(source_sum)))

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(18.0, 7.1))
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=(1.02, 2.15),
            left=0.045,
            right=0.982,
            bottom=0.16,
            top=0.91,
            wspace=0.11,
        )
        dimension_ax = fig.add_subplot(outer[0, 0])
        _draw_icp_dimension_panel(dimension_ax, row=row, coils=coils)

        maps_grid = outer[0, 1].subgridspec(2, 4, wspace=0.10, hspace=0.18)
        map_axes: list[plt.Axes] = []
        sdf_image = None
        for index in range(6):
            ax = fig.add_subplot(maps_grid[index // 3, index % 3])
            map_axes.append(ax)
            sdf_image = ax.imshow(
                sdf_cm[index],
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=sdf_cmap,
                norm=sdf_norm,
                aspect="equal",
                rasterized=True,
            )
            ax.contour(r, z, sdf_cm[index], levels=[0.0], colors=[COLORS["text"]], linewidths=0.55)
            ax.set_xlim(0.0, 30.0)
            ax.set_ylim(0.0, 22.0)
            ax.set_xticks([0.0, 15.0, 30.0])
            ax.set_yticks([0.0, 11.0, 22.0])
            ax.tick_params(labelsize=7.4, length=2.2)
            if index // 3 == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$r$", fontsize=8.0, labelpad=0.0)
            if index % 3 != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"$z$", fontsize=8.0, labelpad=0.0)
            ax.set_title(rf"$d_{{{index + 1}}}(r,z)$", pad=2.0, fontsize=9.2)

        union_ax = fig.add_subplot(maps_grid[0, 3])
        union_ax.imshow(
            union_sdf_cm,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=sdf_cmap,
            norm=sdf_norm,
            aspect="equal",
            rasterized=True,
        )
        union_ax.contour(r, z, union_sdf_cm, levels=[0.0], colors=[COLORS["text"]], linewidths=0.6)
        union_ax.set_title(r"$d_{\mathrm{union}}=\min_j d_j$", pad=2.0, fontsize=9.2)

        source_ax = fig.add_subplot(maps_grid[1, 3])
        source_image = source_ax.imshow(
            source_sum,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=_masked_cmap("cividis"),
            norm=source_norm,
            aspect="equal",
            rasterized=True,
        )
        source_ax.contour(r, z, union_sdf_cm, levels=[0.0], colors=["white"], linewidths=0.55)
        source_ax.set_title(
            r"$s=\sum_j e^{-\max(d_j,0)/(0.20\,\mathrm{cm})}$",
            pad=2.0,
            fontsize=8.2,
        )
        for ax in (union_ax, source_ax):
            ax.set_xlim(0.0, 30.0)
            ax.set_ylim(0.0, 22.0)
            ax.set_xticks([0.0, 15.0, 30.0])
            ax.set_yticks([0.0, 11.0, 22.0])
            ax.tick_params(labelsize=7.4, length=2.2, labelleft=False)
        union_ax.tick_params(labelbottom=False)
        source_ax.set_xlabel(r"$r$", fontsize=8.0, labelpad=0.0)

        if sdf_image is None:
            raise RuntimeError("SDF maps were not drawn")
        sdf_cax = fig.add_axes([0.43, 0.07, 0.36, 0.022])
        sdf_colorbar = fig.colorbar(sdf_image, cax=sdf_cax, orientation="horizontal", extend="max")
        sdf_colorbar.set_label("Signed distance (cm): negative inside, zero at coil boundary", fontsize=8.8)
        sdf_colorbar.ax.tick_params(labelsize=7.8, length=2.2)
        sdf_colorbar.outline.set_linewidth(0.55)
        source_cax = fig.add_axes([0.835, 0.07, 0.12, 0.022])
        source_colorbar = fig.colorbar(source_image, cax=source_cax, orientation="horizontal")
        source_colorbar.set_label(r"source sum ($\tau=0.20$ cm)", fontsize=8.8)
        source_colorbar.ax.tick_params(labelsize=7.8, length=2.2)
        source_colorbar.outline.set_linewidth(0.55)

        fig.text(0.655, 0.965, "B   Spatial structure encoding", ha="center", va="top", fontsize=11.5, weight="bold")
        fig.text(0.58, 0.925, "six slot-wise SDF channels", ha="center", va="top", fontsize=9.4, color=COLORS["muted"])
        fig.text(0.885, 0.925, "order-invariant trained profile", ha="center", va="top", fontsize=9.4, color=COLORS["muted"])

    dimension_values = {
        "llcoil_cm": float(row["llcoil"]),
        "rrc_cm": float(row["rrc"]),
        "nncoil": int(float(row["nncoil"])),
        "rrce_cm": float(row["rrce"]),
        "zzc_cm": float(row["zzc"]),
    }
    metadata = {
        "figure": "GEC-ICP dimension-parameter versus spatial SDF geometry encoding",
        "case_id": ICP_CASE_ID,
        "dimension_parameter_encoding": {
            "geometry_vector_order": ["llcoil", "rrc", "nncoil", "rrce", "zzc"],
            "values": dimension_values,
            "full_condition_vector": ["llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0"],
            "condition_scaler": "train-fitted robust scaler",
            "pitch_definition_cm": "(rrce - rrc) / nncoil",
            "pitch_value_cm": (float(row["rrce"]) - float(row["rrc"])) / float(row["nncoil"]),
            "source_config": str(ICP_DIMENSION_CONFIG),
        },
        "spatial_structure_encoding": {
            "slot_channels": [f"sdf_coil_{index:02d}" for index in range(1, 7)],
            "stored_sdf_unit": "grid pixels",
            "grid_spacing_cm": dr,
            "display_unit": "cm",
            "sign_convention": "negative inside coil, zero on boundary, positive outside",
            "display_clip": {"vmin_cm": -0.25, "vmax_cm": 5.0, "positive_values_above_vmax_saturated": True},
            "order_invariant_profile": {
                "part_sdf_union": "minimum over slot-wise signed distances",
                "part_source_sum": "sum_j exp(-max(d_j, 0) / tau)",
                "source_tau_px": source_tau_px,
                "source_tau_cm": source_tau_px * dr,
            },
            "trained_profile": "part_source_v1",
            "source_trained_config": str(ICP_TRAINED_CONFIG),
        },
        "sources": {
            "dataset_index": str(ICP_INDEX_PATH),
            "structure_npz": str(ICP_DATA_ROOT / row["structure_npz"]),
            "coil_layout_csv": str(ICP_LAYOUT_ROOT / f"{ICP_CASE_ID}__coil_layout.csv"),
        },
        "validation": {
            "active_coil_count": len(coils),
            "union_sdf_strict_sign_contradiction_cells": sign_contradictions,
            "union_sdf_zero_boundary_band_cells": zero_boundary_cells,
            "union_sdf_range_cm": [float(np.min(union_sdf_cm)), float(np.max(union_sdf_cm))],
            "source_sum_range": [float(np.min(source_sum)), float(np.max(source_sum))],
        },
        "interpretation_note": (
            "The dimension vector is a global compact descriptor. SDF/source channels retain where each "
            "structure lies in the r-z grid. The figure describes encodings and does not claim a completed "
            "dimension-versus-SDF performance comparison."
        ),
    }
    return fig, metadata


FIGURE_BUILDERS = {
    "ccp_fields": ("ccp_representative_physical_fields", create_ccp_physical_fields_figure),
    "ccp_zscore": ("ccp_linear_zscore_effect", create_ccp_zscore_figure),
    "icp_fields": ("icp_representative_physical_fields", create_icp_physical_fields_figure),
    "icp_sdf": ("icp_geometry_features_sdf", create_icp_geometry_features_figure),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=FORMATS,
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("all", *FIGURE_BUILDERS),
        default=("all",),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    selected = list(FIGURE_BUILDERS) if "all" in args.figures else list(dict.fromkeys(args.figures))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for key in selected:
        prefix, builder = FIGURE_BUILDERS[key]
        fig, metadata = builder()
        try:
            outputs = _save_figure(
                fig,
                out_dir=args.out_dir,
                prefix=prefix,
                formats=args.formats,
                dpi=args.dpi,
            )
        finally:
            plt.close(fig)
        metadata["output_files"] = [path.name for path in outputs]
        metadata_path = _write_metadata(args.out_dir, prefix, metadata)
        generated.extend((*outputs, metadata_path))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
