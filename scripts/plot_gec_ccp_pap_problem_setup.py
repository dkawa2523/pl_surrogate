"""Create a conference-ready diagram of the GEC-CCP pseudo-PAP inverse problem."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.data.geometry_provider import build_geometry_provider


DEFAULT_RUN = Path("runs/gec_ccp_ffno_pap_inverse_cmaes_target_at_mixed_best_gamma07_v1")
DEFAULT_CONFERENCE_DIR = Path("reports/gec_conference_materials/optimization_assets/pap_inverse_example")
OUTPUT_STEM = "ccp_pseudo_pap_problem_setup"


def _read_observation(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty PAP observation file: {path}")
    return {
        "r_mm": np.asarray([float(row["r_mm"]) for row in rows], dtype=np.float64),
        "z_mm": np.asarray([float(row["z_mm"]) for row in rows], dtype=np.float64),
        "truth_ne": np.asarray([float(row["truth_ne_m-3"]) for row in rows], dtype=np.float64),
        "observed_ne": np.asarray([float(row["observed_ne_m-3"]) for row in rows], dtype=np.float64),
        "relative_noise": np.asarray([float(row["relative_noise"]) for row in rows], dtype=np.float64),
    }


def _resolve_case(run_dir: Path, summary: dict[str, Any]) -> tuple[Any, dict[str, Any], Path]:
    run_root = Path(str(summary["run_root"]))
    with (run_root / "resolved_config.yaml").open("r", encoding="utf-8") as stream:
        cfg = dict(yaml.safe_load(stream) or {})
    dataset = load_dataset(cfg, run_root)
    case_id = str(summary["case_id"])
    try:
        case = next(case for case in dataset.cases if str(case["case_id"]) == case_id)
    except StopIteration as exc:
        raise KeyError(f"case not found in dataset: {case_id}") from exc
    return dataset, case, run_root


def _save_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            save_kwargs["dpi"] = 240
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def _copy_outputs(paths: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in paths:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--conference-dir", type=Path, default=DEFAULT_CONFERENCE_DIR)
    parser.add_argument("--no-conference-copy", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    observation = _read_observation(run_dir / "pap_observation.csv")
    dataset, case, run_root = _resolve_case(run_dir, summary)

    geometry = build_geometry_provider(Path(dataset.geometry_root), provider_mode="fixed").get(
        {"geom_id": str(case.get("geom_id", case.get("base_name", "base2")))}
    )
    r_mm = np.asarray(geometry.coord_grid[0], dtype=np.float64) * 1.0e3
    z_mm = np.asarray(geometry.coord_grid[1], dtype=np.float64) * 1.0e3
    plasma_mask = np.asarray(geometry.mask_plasma) > 0.5
    ne_scaled = np.asarray(case["y"]["ne"], dtype=np.float64) / 1.0e15
    ne_plot = np.ma.masked_where(~plasma_mask, ne_scaled)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.0,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
        }
    )
    fig, (ax_field, ax_profile) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.35),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.27},
    )

    field = ax_field.pcolormesh(
        r_mm,
        z_mm,
        ne_plot,
        shading="auto",
        cmap="viridis",
        rasterized=True,
        vmin=0.0,
        vmax=float(np.nanmax(ne_scaled[plasma_mask])),
    )
    ax_field.contour(r_mm, z_mm, plasma_mask.astype(float), levels=[0.5], colors="#323232", linewidths=0.65)
    scan_z = float(summary["z_mm"])
    scan_r = observation["r_mm"]
    ax_field.plot(
        [float(scan_r.min()), float(scan_r.max())],
        [scan_z, scan_z],
        color="white",
        linewidth=2.0,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    ax_field.scatter(
        scan_r,
        np.full_like(scan_r, scan_z),
        s=31,
        facecolors="#ff9f1c",
        edgecolors="white",
        linewidths=0.85,
        zorder=5,
    )
    ax_field.annotate(
        f"PAP radial scan  $z={scan_z:g}$ mm\n{scan_r.size} measurement positions",
        xy=(float(scan_r[-3]), scan_z),
        xytext=(54.0, 35.0),
        color="#111111",
        fontsize=10.0,
        ha="center",
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#777777", "alpha": 0.92},
        zorder=6,
    )
    ax_field.set_title("Spatial pseudo-truth and PAP scan")
    ax_field.set_xlabel("Radius, $r$ [mm]")
    ax_field.set_ylabel("Axial position, $z$ [mm]")
    ax_field.set_xlim(float(np.nanmin(r_mm)), float(np.nanmax(r_mm)))
    ax_field.set_ylim(float(np.nanmin(z_mm)), float(np.nanmax(z_mm)))
    ax_field.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(field, ax=ax_field, fraction=0.046, pad=0.025)
    cbar.set_label(r"Electron density, $n_e$ [$10^{15}$ m$^{-3}$]")

    ax_profile.plot(
        observation["r_mm"],
        observation["truth_ne"] / 1.0e15,
        color="#1f4e79",
        linewidth=2.5,
        label="COMSOL pseudo-truth",
        zorder=2,
    )
    ax_profile.scatter(
        observation["r_mm"],
        observation["observed_ne"] / 1.0e15,
        s=52,
        color="#ff9f1c",
        edgecolor="white",
        linewidth=0.9,
        label="Pseudo-PAP observations",
        zorder=3,
    )
    ax_profile.set_title(f"Measured radial distribution at $z={scan_z:g}$ mm")
    ax_profile.set_xlabel("Radius, $r$ [mm]")
    ax_profile.set_ylabel(r"Electron density, $n_e$ [$10^{15}$ m$^{-3}$]")
    ax_profile.set_xlim(float(scan_r.min()) - 2.0, float(scan_r.max()) + 2.0)
    ax_profile.set_ylim(bottom=0.0)
    ax_profile.grid(True, color="#d8d8d8", linewidth=0.75, alpha=0.8)
    ax_profile.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax_profile.text(
        0.03,
        0.05,
        "Frozen 5% multiplicative noise\n(clipped to $\\pm$10%)",
        transform=ax_profile.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#999999", "alpha": 0.94},
    )

    truth = dict(summary["truth_conditions"])
    fig.suptitle("GEC-CCP pseudo-PAP inverse problem", fontsize=15.0, y=0.995)
    fig.text(
        0.5,
        0.005,
        r"Find $\theta=(PP0,\,PA,\,\gamma)$ that minimizes the linear relative $L_2$ mismatch "
        + rf"at the PAP positions; $T_d={float(truth['Td']):g}$ and geometry are fixed.",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.90, bottom=0.14)

    run_paths = _save_figure(fig, run_dir)
    plt.close(fig)
    conference_paths: list[Path] = []
    if not args.no_conference_copy:
        conference_paths = _copy_outputs(run_paths, args.conference_dir)

    metadata = {
        "title": "GEC-CCP pseudo-PAP inverse problem",
        "source_run": str(run_dir),
        "model_run": str(run_root),
        "case_id": str(summary["case_id"]),
        "pseudo_truth_source": str(summary["pseudo_truth_source"]),
        "case_is_held_out_interp_test": bool(summary["case_is_held_out_interp_test"]),
        "presentation_scope": str(summary["case_selection"]),
        "measurement": {
            "quantity": "electron density",
            "height_mm": scan_z,
            "radial_positions_mm": observation["r_mm"].tolist(),
            "n_points": int(observation["r_mm"].size),
            "point_noise_rel": float(summary["point_noise_rel"]),
            "noise_clip_rel": float(summary["noise_clip_rel"]),
            "noise_seed": int(summary["noise_seed"]),
        },
        "inverse_variables": ["PP0", "PA", "gamma"],
        "fixed_conditions": dict(summary["fixed_conditions"]),
        "objective": str(summary["objective"]),
        "outputs": [str(path) for path in run_paths + conference_paths],
    }
    metadata_path = run_dir / f"{OUTPUT_STEM}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_conference_copy:
        shutil.copy2(metadata_path, args.conference_dir / metadata_path.name)

    print("\n".join(str(path) for path in run_paths + conference_paths + [metadata_path]))


if __name__ == "__main__":
    main()
