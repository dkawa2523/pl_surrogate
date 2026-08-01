"""Plot a CMA-ES trial trajectory on a fixed GEC-CCP PCA response map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import numpy as np


RESPONSE_COLOR_FLOOR_PERCENT = 1.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _project(
    rows: list[dict[str, Any]],
    *,
    variables: list[str],
    lower: np.ndarray,
    span: np.ndarray,
    mean_scaled: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        [[float(row[key]) for key in variables] for row in rows],
        dtype=np.float64,
    )
    return ((values - lower[None, :]) / span[None, :] - mean_scaled) @ components.T


def _project_point(
    cond: dict[str, Any],
    *,
    variables: list[str],
    lower: np.ndarray,
    span: np.ndarray,
    mean_scaled: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    values = np.asarray([float(cond[key]) for key in variables], dtype=np.float64)
    return (((values - lower) / span) - mean_scaled) @ components.T


def _response_surface(
    reference_scores: np.ndarray,
    loss_percent: np.ndarray,
    truth_score: np.ndarray,
    *,
    bandwidth: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    display_points = np.vstack([reference_scores, truth_score[None, :]])
    display_min = np.min(display_points, axis=0)
    display_max = np.max(display_points, axis=0)
    padding = 0.07 * np.maximum(display_max - display_min, 1.0e-6)
    grid_x, grid_y = np.meshgrid(
        np.linspace(display_min[0] - padding[0], display_max[0] + padding[0], 240),
        np.linspace(display_min[1] - padding[1], display_max[1] + padding[1], 240),
    )
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    score_scale = np.std(reference_scores, axis=0)
    normalized_scores = reference_scores / score_scale[None, :]
    normalized_grid = grid_points / score_scale[None, :]
    log_loss = np.log(loss_percent)
    smoothed = np.empty(normalized_grid.shape[0], dtype=np.float64)
    for start in range(0, normalized_grid.shape[0], 2048):
        stop = min(start + 2048, normalized_grid.shape[0])
        delta = normalized_grid[start:stop, None, :] - normalized_scores[None, :, :]
        distance_sq = np.sum(delta * delta, axis=2)
        distance_sq -= np.min(distance_sq, axis=1, keepdims=True)
        weights = np.exp(-0.5 * distance_sq / (bandwidth * bandwidth))
        smoothed[start:stop] = (weights @ log_loss) / np.sum(weights, axis=1)
    return grid_x, grid_y, np.exp(smoothed).reshape(grid_x.shape)


def _generation_centers(scores: np.ndarray, popsize: int) -> np.ndarray:
    centers = [scores[0]]
    for start in range(1, scores.shape[0], popsize):
        centers.append(np.mean(scores[start : start + popsize], axis=0))
    return np.asarray(centers, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-stem", default="pca_trial_trajectory")
    args = parser.parse_args()

    target_summary = json.loads((args.target_run / "summary.json").read_text(encoding="utf-8"))
    reference_summary = json.loads(
        (args.reference_run / "summary.json").read_text(encoding="utf-8")
    )
    target_rows = _read_csv(args.target_run / "trials.csv")
    reference_rows = _read_csv(args.reference_run / "trials.csv")
    reference_meta = reference_summary["pca_response_surface"]

    variables = list(reference_meta["variables"])
    search_space = reference_summary["search_space"]
    lower = np.asarray([float(search_space[key][0]) for key in variables])
    span = np.asarray(
        [float(search_space[key][1]) - float(search_space[key][0]) for key in variables]
    )
    mean_scaled = np.asarray(reference_meta["mean_scaled"], dtype=np.float64)
    components = np.asarray(reference_meta["components"], dtype=np.float64)
    explained = np.asarray(reference_meta["explained_variance_ratio"], dtype=np.float64)
    projection_kwargs = {
        "variables": variables,
        "lower": lower,
        "span": span,
        "mean_scaled": mean_scaled,
        "components": components,
    }
    reference_scores = _project(reference_rows, **projection_kwargs)
    target_scores = _project(target_rows, **projection_kwargs)
    truth_score = _project_point(target_summary["truth_conditions"], **projection_kwargs)
    initial_score = _project_point(target_summary["initial_conditions"], **projection_kwargs)
    best_score = _project_point(target_summary["best_conditions"], **projection_kwargs)
    reference_loss = 100.0 * np.asarray(
        [float(row["loss"]) for row in reference_rows], dtype=np.float64
    )
    grid_x, grid_y, surface = _response_surface(reference_scores, reference_loss, truth_score)

    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    vmin = RESPONSE_COLOR_FLOOR_PERCENT
    vmax = float(np.max(reference_loss))
    response = ax.contourf(
        grid_x,
        grid_y,
        surface,
        levels=np.geomspace(vmin, vmax, 22),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis_r",
        alpha=0.58,
        extend="max",
    )

    trial_number = np.arange(1, target_scores.shape[0] + 1, dtype=np.float64)
    trial_norm = Normalize(vmin=1.0, vmax=float(target_scores.shape[0]))
    segments = np.stack([target_scores[:-1], target_scores[1:]], axis=1)
    trajectory = LineCollection(
        segments,
        cmap="turbo",
        norm=trial_norm,
        linewidth=0.75,
        alpha=0.45,
        zorder=3,
    )
    trajectory.set_array(trial_number[1:])
    ax.add_collection(trajectory)
    ax.scatter(
        target_scores[:, 0],
        target_scores[:, 1],
        c=trial_number,
        cmap="turbo",
        norm=trial_norm,
        s=11,
        alpha=0.62,
        edgecolor="none",
        zorder=4,
    )

    popsize = int(target_summary.get("optimizer_config", {}).get("popsize", 9))
    centers = _generation_centers(target_scores, popsize)
    ax.plot(centers[:, 0], centers[:, 1], color="black", linewidth=2.8, alpha=0.78, zorder=5)
    ax.plot(centers[:, 0], centers[:, 1], color="white", linewidth=1.2, alpha=0.95, zorder=6)

    ax.scatter(
        initial_score[0], initial_score[1], marker="D", s=90, color="#56B4E9",
        edgecolor="black", linewidth=1.2, label="initial / trial 1", zorder=8,
    )
    ax.scatter(
        truth_score[0], truth_score[1], marker="*", s=190, color="white",
        edgecolor="black", linewidth=1.2, label="truth", zorder=8,
    )
    ax.scatter(
        best_score[0], best_score[1], marker="X", s=95, color="#D55E00",
        edgecolor="black", linewidth=1.0, label=f"best / trial {target_summary['best_trial']}", zorder=8,
    )

    label_offsets = {
        1: (6, 6),
        50: (6, -15),
        100: (6, 7),
        200: (7, -15),
        300: (7, 7),
    }
    final_trial = target_scores.shape[0]
    if final_trial != int(target_summary["best_trial"]):
        label_offsets[final_trial] = (7, 7)
    for number, offset in label_offsets.items():
        if number > target_scores.shape[0]:
            continue
        point = target_scores[number - 1]
        ax.annotate(
            f"T{number}",
            xy=point,
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
            zorder=9,
        )

    ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% variance)")
    ax.set_title("CMA-ES search trajectory in the fixed PCA space")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88, fontsize=9)
    response_cbar = fig.colorbar(response, ax=ax, pad=0.02)
    response_cbar.set_label("Reference PAP profile relative L2 [%]")
    response_ticks = [
        tick for tick in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0) if vmin <= tick <= vmax
    ]
    response_cbar.set_ticks(response_ticks)
    response_cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    trial_cbar = fig.colorbar(trajectory, ax=ax, orientation="horizontal", pad=0.14, fraction=0.055)
    trial_cbar.set_label("Trial number")
    fig.tight_layout()

    output = args.target_run / args.output_stem
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    projected_rows = []
    for row, score in zip(target_rows, target_scores, strict=True):
        projected_rows.append(
            {
                "trial": int(row["trial"]),
                "PC1": float(score[0]),
                "PC2": float(score[1]),
                **{key: float(row[key]) for key in variables},
                "loss": float(row["loss"]),
            }
        )
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(projected_rows[0]))
        writer.writeheader()
        writer.writerows(projected_rows)
    metadata = {
        "target_run": str(args.target_run),
        "reference_run": str(args.reference_run),
        "variables": variables,
        "explained_variance_ratio": explained.tolist(),
        "trial_count": len(target_rows),
        "population_size": popsize,
        "trajectory": "individual trials plus CMA-ES population-center path",
        "response_map": "fixed PCA basis and Gaussian-kernel surface from reference run",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
