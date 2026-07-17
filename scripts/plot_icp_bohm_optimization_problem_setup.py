"""Create conference-ready ICP Bohm-flux optimization problem diagrams."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


RUN = Path("runs/icp_stage4_bohm_flux_sdf_500trials_spacing_slope_v4")
OUTPUT = Path("reports/icp_conference_materials/bohm_representation_comparison")
GEOMETRY = Path("data/outputs_icp_stage4_enriched_360_csv_npz_causal_em_pabs_v1/geometry")
PROBLEM_STEM = "icp_bohm_optimization_problem_setup"
FORMULA_STEM = "icp_bohm_objective_definition"
SOURCE_TRIAL_ID = 281
E_CHARGE = 1.602176634e-19
ARGON_ION_MASS = 6.63352146325368e-26


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 240
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _trial() -> dict[str, str]:
    with (RUN / "trials.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return next(row for row in rows if int(row["trial_id"]) == SOURCE_TRIAL_ID)


def _profile() -> tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(RUN / "trial_bohm_profiles.npz") as stored:
        ids = np.asarray(stored["trial_ids"], dtype=int)
        index = int(np.flatnonzero(ids == SOURCE_TRIAL_ID)[0])
        radius = np.asarray(stored["radius"], dtype=float)
        profile = np.asarray(stored["profiles"], dtype=float)[index]
        deviation = float(np.asarray(stored["deviations"], dtype=float)[index])
        density = float(np.asarray(stored["mean_density"], dtype=float)[index])
    return radius, profile, deviation, density


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
        }
    )


def _problem_figure(row: dict[str, str], radius: np.ndarray, profile: np.ndarray, deviation: float, density: float) -> plt.Figure:
    r = np.load(GEOMETRY / "r_coords.npy")
    z = np.load(GEOMETRY / "z_coords.npy")
    plasma = np.load(GEOMETRY / "mask_plasma.npy") > 0.5
    r_grid, z_grid = np.meshgrid(r, z)
    first = np.asarray([np.flatnonzero(plasma[:, col])[0] for col in range(plasma.shape[1])])
    surface_row = int(np.max(first))
    wafer_columns = first == surface_row
    wafer_r_max = float(np.max(r[wafer_columns]))
    wafer_surface = float(z[surface_row])
    band_top = float(z[min(surface_row + 10, len(z) - 1)])

    fig = plt.figure(figsize=(13.0, 6.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18], wspace=0.24)
    ax_device = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.06, 0.94], hspace=0.34)
    ax_profile = fig.add_subplot(right[0, 0])
    ax_formula = fig.add_subplot(right[1, 0])

    ax_device.contourf(
        r_grid,
        z_grid,
        plasma.astype(float),
        levels=[0.5, 1.5],
        colors=["#b9d8eb"],
        alpha=0.72,
    )
    ax_device.contour(r_grid, z_grid, plasma.astype(float), levels=[0.5], colors="#3f3f3f", linewidths=1.1)
    ax_device.add_patch(
        Rectangle(
            (float(r.min()), float(z.min())),
            wafer_r_max - float(r.min()),
            wafer_surface - float(z.min()),
            facecolor="#737373",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.72,
        )
    )
    ax_device.add_patch(
        Rectangle(
            (float(r.min()), wafer_surface),
            wafer_r_max - float(r.min()),
            band_top - wafer_surface,
            facecolor="#ffb000",
            edgecolor="none",
            alpha=0.86,
        )
    )
    count = int(row["nncoil"])
    size = float(row["llcoil"])
    centres = []
    for slot in range(count):
        rc = float(row[f"r_center_{slot + 1:02d}"])
        zc = float(row[f"z_center_{slot + 1:02d}"])
        centres.append((rc, zc))
        ax_device.add_patch(
            Rectangle(
                (rc - 0.5 * size, zc - 0.5 * size),
                size,
                size,
                facecolor="#d62728",
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )
        )
    centres_array = np.asarray(centres)
    ax_device.plot(
        centres_array[:, 0],
        centres_array[:, 1],
        color="#d62728",
        linewidth=1.1,
        alpha=0.8,
        zorder=4,
    )
    ax_device.annotate(
        "Wafer-near evaluation region\n10 axial layers, 32 equal-area radial bins",
        xy=(0.58 * wafer_r_max, 0.5 * (wafer_surface + band_top)),
        xytext=(15.2, 5.2),
        ha="center",
        fontsize=9.3,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#777777", "alpha": 0.94},
    )
    ax_device.set(
        xlim=(float(r.min()), float(r.max())),
        ylim=(float(z.min()), float(z.max())),
        xlabel="Radius, r [dataset coordinate]",
        ylabel="Axial position, z [dataset coordinate]",
        title="ICP chamber and optimization geometry",
    )
    ax_device.set_aspect("equal", adjustable="box")
    ax_device.legend(
        handles=[
            Patch(facecolor="#b9d8eb", edgecolor="#3f3f3f", label="plasma / chamber"),
            Patch(facecolor="#737373", edgecolor="black", label="wafer"),
            Patch(facecolor="#ffb000", label="evaluation region"),
            Patch(facecolor="#d62728", edgecolor="black", label="optimized coils"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.95,
    )

    scaled = profile / 1.0e20
    mean_flux = float(np.mean(scaled))
    local_deviation = np.abs(scaled / mean_flux - 1.0)
    worst = int(np.argmax(local_deviation))
    ax_profile.plot(radius, scaled, color="#d55e00", linewidth=2.5, label="surrogate Bohm flux")
    ax_profile.axhline(mean_flux, color="black", linestyle="--", linewidth=1.2, label="radial mean")
    ax_profile.scatter(
        [radius[worst]],
        [scaled[worst]],
        marker="X",
        s=75,
        color="#cc79a7",
        edgecolor="black",
        zorder=5,
        label="maximum local deviation",
    )
    ax_profile.set(
        xlim=(float(radius.min()), float(radius.max())),
        ylim=(0.0, 4.5),
        xlabel="Wafer radius, r [dataset coordinate]",
        ylabel=r"Bohm ion flux [$10^{20}$ m$^{-2}$ s$^{-1}$]",
        title="Area-binned wafer-near Bohm-flux distribution",
    )
    ax_profile.grid(alpha=0.24)
    ax_profile.legend(frameon=False, ncol=2, loc="lower center")
    ax_profile.text(
        0.02,
        0.96,
        f"$D_{{max}}={100.0 * deviation:.2f}$%\n"
        + rf"$\langle n \rangle_W={density / 1.0e17:.3f}\times10^{{17}}$ m$^{{-3}}$",
        transform=ax_profile.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#999999", "alpha": 0.94},
    )

    ax_formula.axis("off")
    formula = (
        r"$\Gamma_B(r,z)=n_i(r,z)c_s(r,z),\qquad "
        r"c_s=\sqrt{\frac{eT_e(r,z)}{m_{\mathrm{Ar}^+}}}$"
        "\n\n"
        r"$\overline{\Gamma}_{B,k}="
        r"\frac{\sum_{(r,z)\in W_k} r\,\Gamma_B(r,z)}{\sum_{(r,z)\in W_k} r},\qquad "
        r"D_{\max}=\max_k\left|\frac{\overline{\Gamma}_{B,k}}{\langle\overline{\Gamma}_B\rangle}-1\right|$"
        "\n\n"
        r"$\underset{\theta}{\mathrm{minimize}}\;D_{\max}(\theta)\quad"
        r"\mathrm{subject\ to}\quad\langle n\rangle_W\geq10^{17}\;\mathrm{m}^{-3}$"
    )
    ax_formula.text(
        0.5,
        0.60,
        formula,
        ha="center",
        va="center",
        fontsize=12.0,
        bbox={"boxstyle": "round,pad=0.55", "fc": "#f7f7f7", "ec": "#7f7f7f"},
    )
    ax_formula.text(
        0.5,
        0.06,
        r"Design $\theta$: coil count $N=2\ldots5$, process $(PP,PP0)$, and coil geometry.  "
        r"Dimension: equal spacing/height; SDF: unequal spacing with constant-slope height.",
        ha="center",
        va="bottom",
        fontsize=9.6,
    )

    fig.suptitle("ICP coil optimization for uniform wafer-near Bohm flux", fontsize=15.0, y=0.985)
    fig.text(
        0.5,
        0.012,
        "Frozen U-Net surrogate: geometry/process conditions → (ni, Te) fields → Bohm-flux objective",
        ha="center",
        fontsize=10.3,
    )
    fig.subplots_adjust(top=0.91, bottom=0.10)
    return fig


def _formula_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    cards = [
        (
            "1. Local Bohm flux",
            r"$\Gamma_B=n_i c_s$" + "\n" + r"$c_s=\sqrt{eT_e/m_{\mathrm{Ar}^+}}$",
            "Ion density and electron temperature\nfrom the frozen surrogate",
        ),
        (
            "2. Wafer-near distribution",
            r"$\overline{\Gamma}_{B,k}=\frac{\sum_{W_k}r\Gamma_B}{\sum_{W_k}r}$" + "\n" +
            r"$D_{\max}=\max_k|\overline{\Gamma}_{B,k}/\langle\overline{\Gamma}_B\rangle-1|$",
            "10 axial layers\n32 equal-area radial bins",
        ),
        (
            "3. Constrained optimization",
            r"$\min_{\theta}\;D_{\max}(\theta)$" + "\n" +
            r"$\langle n\rangle_W\geq10^{17}\;\mathrm{m}^{-3}$",
            r"$\theta=\{N,PP,PP0,\mathrm{coil\ geometry}\}$",
        ),
    ]
    for index, (title, equation, note) in enumerate(cards):
        axis = axes[index]
        axis.axis("off")
        axis.text(0.5, 0.87, title, ha="center", va="center", fontsize=13.0, weight="bold")
        axis.text(
            0.5,
            0.53,
            equation,
            ha="center",
            va="center",
            fontsize=14.0 if index != 1 else 12.0,
            bbox={"boxstyle": "round,pad=0.6", "fc": ("#e8f3f8", "#fff3d6", "#f3eaf6")[index], "ec": "#777777"},
        )
        axis.text(0.5, 0.16, note, ha="center", va="center", fontsize=10.0)
        if index < 2:
            axis.annotate(
                "",
                xy=(1.08, 0.52),
                xytext=(0.93, 0.52),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#444444"},
                annotation_clip=False,
            )
    fig.suptitle("Bohm-flux objective used for ICP surrogate optimization", fontsize=15.0)
    return fig


def main() -> None:
    _style()
    row = _trial()
    radius, profile, deviation, density = _profile()
    problem = _problem_figure(row, radius, profile, deviation, density)
    problem_paths = _save(problem, PROBLEM_STEM)
    plt.close(problem)
    formula = _formula_figure()
    formula_paths = _save(formula, FORMULA_STEM)
    plt.close(formula)
    metadata = {
        "title": "ICP Bohm-flux optimization problem setup",
        "source_run": str(RUN),
        "source_trial_id": SOURCE_TRIAL_ID,
        "coil_count": int(row["nncoil"]),
        "wafer_layers": 10,
        "radial_bins": 32,
        "density_threshold_m3": 1.0e17,
        "bohm_flux": "Gamma_B = ni * sqrt(e * Te / m_Ar+)",
        "e_charge_C": E_CHARGE,
        "argon_ion_mass_kg": ARGON_ION_MASS,
        "objective": "maximum local deviation of equal-area-binned wafer-near Bohm flux",
        "outputs": [str(path) for path in problem_paths + formula_paths],
        "validation_status": "surrogate_only_high_fidelity_icp_validation_pending",
    }
    (OUTPUT / "icp_bohm_optimization_problem_setup_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(metadata["outputs"]))


if __name__ == "__main__":
    main()
