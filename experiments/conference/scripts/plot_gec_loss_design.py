#!/usr/bin/env python3
"""Create conference-ready GEC-CCP and GEC-ICP loss-design figures.

The figures summarize the effective training objectives recorded in completed
benchmark runs.  They intentionally separate the transformation space from the
physical output space and omit evaluation-only objectives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "gec_conference_materials"

CCP_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experimental"
    / "gec_ccp_nn_operator_comparison_v1"
    / "seed_412"
    / "n78"
    / "benchmark_ext0520_fno_n78.yaml"
)
CCP_EFFECTIVE_RUN = (
    REPO_ROOT
    / "runs"
    / "gec_ccp_nn_operator_comparison_v1"
    / "seed_412"
    / "n78"
    / "fno"
    / "manifest.json"
)
ICP_CONFIG = REPO_ROOT / "runs" / "icp_stage4_coil_structure_v1" / "unet" / "resolved_config.yaml"
ICP_EFFECTIVE_RUN = REPO_ROOT / "runs" / "icp_stage4_coil_structure_v1" / "unet" / "manifest.json"
ICP_LEADERBOARD = REPO_ROOT / "runs" / "icp_stage4_coil_structure_v1" / "unet" / "leaderboard.csv"

ICP_ALTERNATIVE_DIMENSION_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experimental"
    / "icp_stage4"
    / "generated_dimension_parameter_v1"
    / "benchmark_icp_stage4_dimension_parameter_v1_uno.yaml"
)
ICP_ALTERNATIVE_SDF_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experimental"
    / "icp_stage4"
    / "generated_axisymmetric_energy_weighted_v1"
    / "benchmark_icp_stage4_axisymmetric_energy_weighted_v1_uno.yaml"
)

FORMATS = ("png", "pdf", "svg")

COLORS = {
    "ink": "#17212B",
    "muted": "#5F6B76",
    "line": "#CBD2D9",
    "soft": "#F5F7F9",
    "blue": "#0072B2",
    "blue_soft": "#EAF4FA",
    "orange": "#E69F00",
    "orange_soft": "#FFF5DE",
    "green": "#009E73",
    "green_soft": "#E8F6F1",
    "purple": "#CC79A7",
    "purple_soft": "#FAEEF5",
    "red": "#D55E00",
}


def _configure_matplotlib() -> None:
    """Use publication-safe fonts and preserve editable vector text."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _assert_sources(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"loss-design source files are missing: {missing}")


def _canvas(figsize: tuple[float, float] = (11.8, 6.4)) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.985)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = COLORS["line"],
    linewidth: float = 1.0,
    radius: float = 0.012,
    zorder: float = 0.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _header(
    ax: Axes,
    *,
    title: str,
    subtitle: str,
    badge: str,
    badge_color: str,
    badge_face: str,
) -> None:
    ax.text(
        0.045,
        0.942,
        title,
        ha="left",
        va="top",
        fontsize=20.5,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.046,
        0.886,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.8,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    _box(
        ax,
        0.737,
        0.896,
        0.215,
        0.048,
        facecolor=badge_face,
        edgecolor=badge_color,
        linewidth=1.0,
        radius=0.018,
        zorder=1.0,
    )
    ax.text(
        0.8445,
        0.920,
        badge,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=badge_color,
        transform=ax.transAxes,
        zorder=2.0,
    )
    ax.plot(
        [0.045, 0.955],
        [0.855, 0.855],
        color=COLORS["line"],
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
    )


def _section_label(ax: Axes, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x,
        y,
        text.upper(),
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )


def _component(
    ax: Axes,
    x: float,
    *,
    color: str,
    heading: str,
    detail: str,
) -> None:
    ax.add_patch(
        Circle(
            (x, 0.122),
            0.0075,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.015,
        0.134,
        heading,
        ha="left",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.015,
        0.098,
        detail,
        ha="left",
        va="center",
        fontsize=8.6,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )


def build_ccp_metadata(output_stem: str = "gec_ccp_loss_design") -> dict[str, Any]:
    """Return the audited, machine-readable GEC-CCP loss contract."""

    return {
        "figure": "GEC-CCP training loss design",
        "status": "trained_comparison_contract",
        "target_space": {
            "physical_value_transform": "identity",
            "scaler": "zscore",
            "fit_split": "train_only",
            "fit_region": "plasma_only",
            "clip": "none",
            "formula": "y_tilde=(y-mu_train_plasma)/sigma_train_plasma",
        },
        "targets": ["ne", "ni", "Te", "phi"],
        "huber": {"delta": 1.0},
        "per_target_loss": {
            "point_weight": 1.0,
            "boundary_weight": 0.25,
            "boundary_band_px": 2.0,
            "boundary_distance_channels": ["distance_any", "part_sdf_nearest"],
            "gradient_weight": 0.10,
            "gradient_spacing_px": [1.0, 1.0],
            "multiscale_weight": 0.05,
            "multiscale_scales": [2, 4],
            "mask": "plasma_only",
            "normalization": "sample_mean",
        },
        "field_family_weighting": {
            "density": {"share": 1.0 / 3.0, "targets": {"ne": 1.0 / 6.0, "ni": 1.0 / 6.0}},
            "temperature": {"share": 1.0 / 3.0, "targets": {"Te": 1.0 / 3.0}},
            "electrostatic": {"share": 1.0 / 3.0, "targets": {"phi": 1.0 / 3.0}},
        },
        "physics_training_term": {"enabled": False, "weight": 0.0},
        "sources": {
            "config": _rel(CCP_CONFIG),
            "effective_run_manifest": _rel(CCP_EFFECTIVE_RUN),
            "loss_composer": "src/plasma_surrogate/train/loss_composer.py",
            "target_group_weighting": "src/plasma_surrogate/core/target_groups.py",
            "boundary_distance": "src/plasma_surrogate/core/boundary_distance.py",
        },
        "output_files": [f"{output_stem}.{suffix}" for suffix in FORMATS],
    }


def build_icp_metadata(output_stem: str = "gec_icp_loss_design") -> dict[str, Any]:
    """Return the trained ICP contract and a clearly separated untrained alternative."""

    return {
        "figure": "GEC-ICP training loss design",
        "status": "trained_part_source_v1_baseline",
        "model": "unet",
        "representation": "part_source_v1",
        "input_channels": ["x", "y", "distance_signed", "part_sdf_union", "part_source_sum"],
        "target_space": {
            "fit_split": "structure_holdout_train_only",
            "fit_region": "plasma_only",
            "ne": {
                "physical_clip": [1.0e8, 1.0e19],
                "value_transform": "log10_floor",
                "floor": 1.0e-30,
                "scaler": "zscore",
            },
            "ni": {
                "physical_clip": [1.0e8, 1.0e19],
                "value_transform": "log10_floor",
                "floor": 1.0e-30,
                "scaler": "zscore",
            },
            "Te": {
                "physical_clip": [0.0, 20.0],
                "value_transform": "log1p",
                "scaler": "robust",
            },
            "phi": {
                "physical_clip": [-50.0, 50.0],
                "value_transform": "signed_log1p",
                "scaler": "robust",
            },
        },
        "huber": {"delta": 1.0},
        "per_target_loss": {
            "point_weight": 1.0,
            "gradient_weight": 0.02,
            "gradient_normalization": "target_rms",
            "gradient_epsilon": 0.05,
            "multiscale_weight": 0.05,
            "multiscale_scales": [2, 4],
            "boundary_weight": 0.0,
            "mask": "plasma_only",
            "normalization": "sample_mean",
        },
        "target_weights": {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5},
        "physics_training_term": {"enabled": False, "weight": 0.0},
        "sources": {
            "resolved_config": _rel(ICP_CONFIG),
            "effective_run_manifest": _rel(ICP_EFFECTIVE_RUN),
            "trained_leaderboard": _rel(ICP_LEADERBOARD),
            "loss_composer": "src/plasma_surrogate/train/loss_composer.py",
        },
        "alternative_contract_not_plotted": {
            "status": "configuration_only_not_yet_trained",
            "purpose": "controlled dimension-vector versus coil-SDF/source representation study",
            "loss": "axisymmetric-volume and electron-density-weighted MSE; no auxiliary terms",
            "dimension_config": _rel(ICP_ALTERNATIVE_DIMENSION_CONFIG),
            "sdf_config": _rel(ICP_ALTERNATIVE_SDF_CONFIG),
            "reason_not_plotted": "the main figure prioritizes the completed trained baseline",
        },
        "output_files": [f"{output_stem}.{suffix}" for suffix in FORMATS],
    }


def plot_ccp_loss_design() -> Figure:
    """Draw the effective GEC-CCP plasma_surrogate_v3 objective."""

    fig, ax = _canvas()
    _header(
        ax,
        title="GEC–CCP training objective",
        subtitle="Linear physical targets → train-only plasma-region Z-score",
        badge="TRAINED COMPARISON CONTRACT",
        badge_color=COLORS["green"],
        badge_face=COLORS["green_soft"],
    )

    _section_label(ax, 0.055, 0.812, "model space", COLORS["blue"])
    ax.text(
        0.50,
        0.785,
        r"$\widetilde{y}_t=\dfrac{y_t-\mu_{t,\,\mathrm{train},\Omega_p}}"
        r"{\sigma_{t,\,\mathrm{train},\Omega_p}},\qquad "
        r"e_t=\widehat{\widetilde{y}}_t-\widetilde{y}_t$",
        ha="center",
        va="center",
        fontsize=18.5,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.945,
        0.810,
        r"$t\in\{n_e,n_i,T_e,\phi\}$",
        ha="right",
        va="center",
        fontsize=9.4,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    _box(ax, 0.055, 0.637, 0.355, 0.095, facecolor=COLORS["soft"], edgecolor=COLORS["line"])
    _section_label(ax, 0.075, 0.705, "robust point penalty", COLORS["muted"])
    ax.text(
        0.232,
        0.666,
        r"$H_1(e)=\frac{1}{2}e^2\ (|e|\leq1),\quad "
        r"H_1(e)=|e|-\frac{1}{2}\ (|e|>1)$",
        ha="center",
        va="center",
        fontsize=12.6,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )

    _box(
        ax,
        0.435,
        0.637,
        0.510,
        0.095,
        facecolor=COLORS["blue_soft"],
        edgecolor=COLORS["blue"],
        linewidth=1.1,
    )
    _section_label(ax, 0.455, 0.705, "per target", COLORS["blue"])
    ax.text(
        0.690,
        0.666,
        r"$\ell_t=\ell_t^{\mathrm{point}}"
        r"+0.25\,\ell_t^{\mathrm{boundary}}"
        r"+0.10\,\ell_t^{\nabla}"
        r"+0.05\,\ell_t^{\mathrm{multi}}$",
        ha="center",
        va="center",
        fontsize=17.0,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )

    _box(
        ax,
        0.055,
        0.405,
        0.890,
        0.165,
        facecolor=COLORS["orange_soft"],
        edgecolor=COLORS["orange"],
        linewidth=1.2,
    )
    _section_label(ax, 0.075, 0.538, "field-family balance", COLORS["red"])
    ax.text(
        0.50,
        0.479,
        r"$\mathcal{L}_{\mathrm{CCP}}="
        r"\frac{1}{6}(\ell_{n_e}+\ell_{n_i})"
        r"+\frac{1}{3}\ell_{T_e}+\frac{1}{3}\ell_{\phi}$",
        ha="center",
        va="center",
        fontsize=23.0,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.425,
        "equal share per field family; density share is split between electron and ion density",
        ha="center",
        va="center",
        fontsize=9.1,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    _component(ax, 0.075, color=COLORS["blue"], heading="POINT", detail="plasma pixels")
    _component(ax, 0.282, color=COLORS["orange"], heading="BOUNDARY", detail="nearest boundary ≤ 2 px")
    _component(ax, 0.530, color=COLORS["green"], heading="GRADIENT", detail="forward differences in r and z")
    _component(ax, 0.782, color=COLORS["purple"], heading="MULTISCALE", detail="masked pooling at 2× and 4×")

    ax.text(
        0.945,
        0.042,
        "sample mean on Ωp  ·  physics-residual training term: off",
        ha="right",
        va="center",
        fontsize=8.7,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    return fig


def plot_icp_loss_design() -> Figure:
    """Draw the trained part_source_v1 GEC-ICP objective."""

    fig, ax = _canvas()
    _header(
        ax,
        title="GEC–ICP training objective",
        subtitle="Transformed/scaled targets with case-varying coil SDF and source fields",
        badge="TRAINED part_source_v1 BASELINE",
        badge_color=COLORS["blue"],
        badge_face=COLORS["blue_soft"],
    )

    _section_label(ax, 0.055, 0.812, "target transforms", COLORS["blue"])
    _box(ax, 0.055, 0.650, 0.890, 0.125, facecolor=COLORS["soft"], edgecolor=COLORS["line"])

    transform_rows = (
        (0.075, 0.730, r"$n_e,\ n_i$", r"clip $[10^8,10^{19}]$ → $\log_{10}(\max(y,10^{-30}))$ → Z-score"),
        (0.075, 0.684, r"$T_e$", r"clip $[0,20]$ → $\log(1+y)$ → robust scale"),
        (0.515, 0.684, r"$\phi$", r"clip $[-50,50]$ → $\mathrm{sgn}(y)\log(1+|y|)$ → robust scale"),
    )
    for x, y, target, transform in transform_rows:
        ax.text(
            x,
            y,
            target,
            ha="left",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            color=COLORS["ink"],
            transform=ax.transAxes,
        )
        ax.text(
            x + (0.067 if x < 0.5 else 0.046),
            y,
            transform,
            ha="left",
            va="center",
            fontsize=9.3,
            color=COLORS["muted"],
            transform=ax.transAxes,
        )

    _box(ax, 0.055, 0.520, 0.355, 0.080, facecolor=COLORS["soft"], edgecolor=COLORS["line"])
    _section_label(ax, 0.075, 0.574, "penalty", COLORS["muted"])
    ax.text(
        0.232,
        0.542,
        r"$H_1(e)=\frac{1}{2}e^2\ (|e|\leq1),\quad |e|-\frac{1}{2}\ (|e|>1)$",
        ha="center",
        va="center",
        fontsize=11.7,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )

    _box(
        ax,
        0.435,
        0.520,
        0.510,
        0.080,
        facecolor=COLORS["blue_soft"],
        edgecolor=COLORS["blue"],
        linewidth=1.1,
    )
    _section_label(ax, 0.455, 0.574, "per target", COLORS["blue"])
    ax.text(
        0.690,
        0.542,
        r"$\ell_t=\ell_t^{\mathrm{point}}"
        r"+0.02\,\ell_t^{\nabla,\mathrm{RMS}}"
        r"+0.05\,\ell_t^{\mathrm{multi}}$",
        ha="center",
        va="center",
        fontsize=16.0,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )

    _box(
        ax,
        0.055,
        0.302,
        0.890,
        0.155,
        facecolor=COLORS["green_soft"],
        edgecolor=COLORS["green"],
        linewidth=1.2,
    )
    _section_label(ax, 0.075, 0.423, "fixed target weights", COLORS["green"])
    ax.text(
        0.50,
        0.365,
        r"$\mathcal{L}_{\mathrm{ICP}}="
        r"1.0\,\ell_{n_e}+1.0\,\ell_{n_i}+1.2\,\ell_{T_e}+0.5\,\ell_{\phi}$",
        ha="center",
        va="center",
        fontsize=22.0,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.320,
        "plasma-only sample mean  ·  no boundary or physics-residual training term",
        ha="center",
        va="center",
        fontsize=9.1,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    _component(ax, 0.095, color=COLORS["blue"], heading="POINT", detail="Huber in transformed space")
    _component(
        ax,
        0.395,
        color=COLORS["green"],
        heading="GRADIENT",
        detail="target-RMS norm; ε = 0.05",
    )
    _component(
        ax,
        0.700,
        color=COLORS["purple"],
        heading="MULTISCALE",
        detail="masked pooling at 2× and 4×",
    )

    ax.text(
        0.055,
        0.041,
        "part_source_v1 inputs:  r, z, chamber SDF, coil-union SDF, coil-source field",
        ha="left",
        va="center",
        fontsize=8.7,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    return fig


def _save_figure(fig: Figure, out_dir: Path, stem: str, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = out_dir / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def _write_metadata(out_dir: Path, stem: str, metadata: dict[str, Any]) -> Path:
    path = out_dir / f"{stem}_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def generate_loss_figures(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    only: str = "all",
    dpi: int = 300,
) -> dict[str, dict[str, Any]]:
    """Generate selected figures and return their sidecar metadata."""

    if only not in {"all", "ccp", "icp"}:
        raise ValueError("only must be one of: all, ccp, icp")
    if int(dpi) < 72:
        raise ValueError("dpi must be at least 72")

    _configure_matplotlib()
    _assert_sources((CCP_CONFIG, CCP_EFFECTIVE_RUN, ICP_CONFIG, ICP_EFFECTIVE_RUN, ICP_LEADERBOARD))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}

    if only in {"all", "ccp"}:
        stem = "gec_ccp_loss_design"
        metadata = build_ccp_metadata(stem)
        fig = plot_ccp_loss_design()
        outputs = _save_figure(fig, out_dir, stem, dpi)
        plt.close(fig)
        metadata_path = _write_metadata(out_dir, stem, metadata)
        result["ccp"] = {
            "outputs": [str(path.resolve()) for path in outputs],
            "metadata": str(metadata_path.resolve()),
            "contract": metadata,
        }

    if only in {"all", "icp"}:
        stem = "gec_icp_loss_design"
        metadata = build_icp_metadata(stem)
        fig = plot_icp_loss_design()
        outputs = _save_figure(fig, out_dir, stem, dpi)
        plt.close(fig)
        metadata_path = _write_metadata(out_dir, stem, metadata)
        result["icp"] = {
            "outputs": [str(path.resolve()) for path in outputs],
            "metadata": str(metadata_path.resolve()),
            "contract": metadata,
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: reports/gec_conference_materials).",
    )
    parser.add_argument("--only", choices=("all", "ccp", "icp"), default="all")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = generate_loss_figures(out_dir=args.out_dir, only=args.only, dpi=args.dpi)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
