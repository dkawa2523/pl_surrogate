"""Build conference figures from the archived ICP FFNO optimization run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = (
    ROOT
    / "runs/icp_stage4_part_sdf_lite_v1_e80_primary"
    / "optimize_process_layout_mean_height_balanced_wide/ffno"
)
BASE_LAYOUT = (
    ROOT
    / "data/outputs_icp_stage4_enriched_360/structure/coil_layout"
    / "case_g002_op01__coil_layout.csv"
)
OUT_DIR = ROOT / "reports/icp_conference_materials/ffno_structure_optimization"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_coils(base_rows: list[dict[str, str]], best: dict[str, Any]) -> None:
    optimized = dict(best["best_geom_param"])
    base_specs = [
        {
            "index": int(row["coil_index"]),
            "r": float(row["r_center"]),
            "z": float(row["z_center"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
        }
        for row in base_rows
        if int(float(row.get("active", "1"))) == 1
    ]
    best_specs = [
        {
            "index": idx,
            "r": float(optimized[f"layout.coil_{idx:02d}.r_center"]),
            "z": float(optimized[f"layout.coil_{idx:02d}.z_center"]),
            "width": float(optimized[f"layout.coil_{idx:02d}.width"]),
            "height": float(optimized[f"layout.coil_{idx:02d}.height"]),
        }
        for idx in range(1, 7)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 2.7), sharex=True, sharey=True)
    for ax, specs, title, color in (
        (axes[0], base_specs, "Baseline coil layout", "#6b7280"),
        (axes[1], best_specs, "FFNO-optimized coil layout", "#1565c0"),
    ):
        for spec in specs:
            r0 = spec["r"] - spec["width"] / 2.0
            z0 = spec["z"] - spec["height"] / 2.0
            ax.add_patch(
                Rectangle(
                    (r0, z0),
                    spec["width"],
                    spec["height"],
                    facecolor=color,
                    edgecolor="#111827",
                    linewidth=1.0,
                    alpha=0.78,
                )
            )
            ax.text(spec["r"], spec["z"], str(spec["index"]), ha="center", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Radial position r")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Axial position z")
    all_specs = base_specs + best_specs
    axes[0].set_xlim(min(x["r"] - x["width"] for x in all_specs), max(x["r"] + x["width"] for x in all_specs))
    axes[0].set_ylim(min(x["z"] - x["height"] for x in all_specs), max(x["z"] + x["height"] for x in all_specs))
    fig.suptitle("ICP coil-structure optimization with the FFNO surrogate", fontsize=14, y=0.99)
    fig.tight_layout()
    _save_figure(fig, "icp_ffno_coil_layout_before_after")


def _plot_history(trials: list[dict[str, str]], base_qoi: dict[str, Any]) -> dict[str, float]:
    trial = np.asarray([int(row["trial"]) for row in trials], dtype=np.int64)
    value = np.asarray([float(row["value"]) for row in trials], dtype=np.float64)
    order = np.argsort(trial)
    trial = trial[order]
    value = value[order]
    best_so_far = np.minimum.accumulate(value)
    best_idx = int(np.argmin(value))
    baseline = float(base_qoi["uniformity"])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.scatter(trial + 1, value, s=13, alpha=0.32, color="#94a3b8", label="Trial objective")
    ax.plot(trial + 1, best_so_far, lw=2.2, color="#1565c0", label="Best-so-far")
    ax.axhline(baseline, color="#d97706", ls="--", lw=1.6, label=f"Baseline = {baseline:.3f}")
    ax.scatter([trial[best_idx] + 1], [value[best_idx]], s=55, color="#c62828", zorder=5)
    ax.annotate(
        f"best = {value[best_idx]:.3f}\ntrial {trial[best_idx] + 1}",
        (trial[best_idx] + 1, value[best_idx]),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Optimization trial")
    ax.set_ylabel("Uniformity objective (lower is better)")
    ax.set_yscale("log")
    ax.set_title("FFNO surrogate optimization history")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, "icp_ffno_optimization_history")
    return {
        "n_trials": int(value.size),
        "baseline_objective": baseline,
        "best_objective": float(value[best_idx]),
        "best_trial_one_based": int(trial[best_idx] + 1),
    }


def _plot_fields() -> None:
    with np.load(RUN_DIR / "base_best_fields.npz") as fields:
        arrays = {name: np.asarray(fields[name], dtype=np.float64) for name in fields.files}

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), constrained_layout=True)
    for row_idx, var in enumerate(("ne", "ni")):
        base = arrays[f"base_{var}"]
        best = arrays[f"best_{var}"]
        delta = best - base
        lo = float(min(np.nanmin(base), np.nanmin(best)))
        hi = float(max(np.nanmax(base), np.nanmax(best)))
        span = float(max(np.nanmax(np.abs(delta)), 1.0e-30))
        panels = (
            (base, f"Baseline {var}", "viridis", lo, hi),
            (best, f"Optimized {var}", "viridis", lo, hi),
            (delta, f"Optimized − baseline {var}", "coolwarm", -span, span),
        )
        for col_idx, (arr, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("r grid index")
            if col_idx == 0:
                ax.set_ylabel("z grid index")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("FFNO-predicted density fields before and after coil optimization", fontsize=14)
    _save_figure(fig, "icp_ffno_density_fields_before_after")


def _plot_qoi(base_qoi: dict[str, Any], best_qoi: dict[str, Any]) -> None:
    base_density = float(base_qoi["uniformity_mean_density"])
    best_density = float(best_qoi["uniformity_mean_density"])
    labels = ("Uniformity\nobjective", "Boundary flux\nCV", "Mean density\n/ baseline")
    baseline = np.asarray(
        [float(base_qoi["uniformity"]), float(base_qoi["boundary_gamma_uniformity"]), 1.0]
    )
    optimized = np.asarray(
        [float(best_qoi["uniformity"]), float(best_qoi["boundary_gamma_uniformity"]), best_density / base_density]
    )
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars_a = ax.bar(x - 0.19, baseline, 0.38, label="Baseline", color="#6b7280")
    bars_b = ax.bar(x + 0.19, optimized, 0.38, label="Optimized", color="#1565c0")
    ax.bar_label(bars_a, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars_b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Optimization effect predicted by the FFNO surrogate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, "icp_ffno_optimization_qoi")


def main() -> None:
    best = _load_json(RUN_DIR / "best.json")
    base_qoi = _load_json(RUN_DIR / "base_qoi.json")
    best_qoi = _load_json(RUN_DIR / "best_qoi.json")
    trials = _load_csv(RUN_DIR / "trials.csv")
    base_rows = _load_csv(BASE_LAYOUT)

    _plot_coils(base_rows, best)
    history = _plot_history(trials, base_qoi)
    _plot_fields()
    _plot_qoi(base_qoi, best_qoi)

    metadata = {
        "source_run": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "model": "ffno",
        "reference_case": "case_g002_op01",
        "optimization_space": "process_conditions_plus_six_coil_rectangles",
        "optimizer": "random_search",
        **history,
        "objective_improvement_percent": 100.0
        * (history["baseline_objective"] - history["best_objective"])
        / history["baseline_objective"],
        "best_mean_density_ratio": float(best_qoi["uniformity_mean_density"])
        / float(base_qoi["uniformity_mean_density"]),
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(OUT_DIR), **metadata}, indent=2))


if __name__ == "__main__":
    main()
