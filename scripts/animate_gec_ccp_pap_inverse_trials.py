"""Animate every trial of a GEC-CCP FFNO PAP inverse run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
import torch
import yaml

import run_gec_ccp_ffno_pap_inverse_example as pap
from plot_gec_ccp_pap_pca_trajectory import (
    RESPONSE_COLOR_FLOOR_PERCENT,
    _project,
    _project_point,
    _response_surface,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_or_predict_profiles(
    run: Path,
    summary: dict,
    trial_rows: list[dict[str, str]],
    r_m: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    cache = run / "trial_profiles_ne.npz"
    variables = list(summary["search_space"])
    conditions = np.asarray(
        [[float(row[key]) for key in variables] for row in trial_rows], dtype=np.float64
    )
    if cache.exists():
        with np.load(cache) as stored:
            cached_conditions = np.asarray(stored["conditions"], dtype=np.float64)
            profiles = np.asarray(stored["profiles_ne_m3"], dtype=np.float64)
        if profiles.shape == (len(trial_rows), r_m.size) and np.array_equal(
            cached_conditions, conditions
        ):
            print(f"reusing profile cache: {cache}", flush=True)
            return profiles

    run_root = Path(summary["run_root"])
    cfg = yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset = pap.load_dataset(cfg, run_root)
    case = {str(item["case_id"]): item for item in dataset.cases}[summary["case_id"]]
    engine = pap._load_ffno_engine(run_root, cfg, Path(dataset.geometry_root), run)
    geom = pap._geom(case)
    axis = pap._axis(case)
    geom_ctx = engine._get_geom_ctx(engine._validate_geom_ref(geom), axis=axis)
    z_m = 1.0e-3 * float(summary["z_mm"])
    observation = pap.LinearNeProfileObservation(
        r_m=r_m,
        z_m=z_m,
        ne_obs=observed,
        density_scale=float(np.sqrt(np.mean(observed * observed))),
        covariance_scaled=np.eye(r_m.size, dtype=np.float64),
        case_id=summary["case_id"],
    )
    fixed = dict(summary["fixed_conditions"])
    fixed.pop("geometry", None)
    assimilation = pap.LinearNeProfileAssimilationEngine(
        engine,
        observation=observation,
        fixed_cond=fixed,
        setpoints={},
        nuisance_rel_sigma={},
    )
    profiles = np.empty((len(trial_rows), r_m.size), dtype=np.float64)
    for index, row in enumerate(trial_rows):
        cond = {key: float(row[key]) for key in variables}
        result = assimilation.single_run_aggregated(
            cond=cond, geom=geom, axis=axis, save_outputs=False
        )
        profiles[index] = pap._profile(result.fields_phys["ne"], geom_ctx, z_m, r_m)
        if (index + 1) % 50 == 0 or index == 0:
            print(f"predicted profiles: {index + 1}/{len(trial_rows)}", flush=True)
    np.savez_compressed(cache, conditions=conditions, profiles_ne_m3=profiles)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--cpu-threads", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    trial_rows = _read_csv(args.run / "trials.csv")
    observation_rows = _read_csv(args.run / "pap_observation.csv")
    r_mm = np.asarray([float(row["r_mm"]) for row in observation_rows], dtype=np.float64)
    r_m = 1.0e-3 * r_mm
    truth_profile = np.asarray(
        [float(row["truth_ne_m-3"]) for row in observation_rows], dtype=np.float64
    )
    observed = np.asarray(
        [float(row["observed_ne_m-3"]) for row in observation_rows], dtype=np.float64
    )
    profiles = _load_or_predict_profiles(args.run, summary, trial_rows, r_m, observed)

    variables = list(summary["pca_response_surface"]["variables"])
    search_space = summary["search_space"]
    lower = np.asarray([float(search_space[key][0]) for key in variables])
    span = np.asarray(
        [float(search_space[key][1]) - float(search_space[key][0]) for key in variables]
    )
    pca_meta = summary["pca_response_surface"]
    mean_scaled = np.asarray(pca_meta["mean_scaled"], dtype=np.float64)
    components = np.asarray(pca_meta["components"], dtype=np.float64)
    explained = np.asarray(pca_meta["explained_variance_ratio"], dtype=np.float64)
    projection_kwargs = {
        "variables": variables,
        "lower": lower,
        "span": span,
        "mean_scaled": mean_scaled,
        "components": components,
    }
    scores = _project(trial_rows, **projection_kwargs)
    truth_score = _project_point(summary["truth_conditions"], **projection_kwargs)
    losses = np.asarray([float(row["loss"]) for row in trial_rows], dtype=np.float64)
    loss_percent = 100.0 * losses
    grid_x, grid_y, surface = _response_surface(scores, loss_percent, truth_score)
    best_indices = np.empty(len(losses), dtype=np.int64)
    running_best = np.empty_like(losses)
    current_best_index = 0
    for index, value in enumerate(losses):
        if value < losses[current_best_index]:
            current_best_index = index
        best_indices[index] = current_best_index
        running_best[index] = losses[current_best_index]

    scale = 1.0e15
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75))
    ax_profile, ax_loss, ax_map = axes
    ax_profile.plot(
        r_mm,
        truth_profile / scale,
        color="black",
        linewidth=2.0,
        label="_nolegend_",
    )
    ax_profile.scatter(
        r_mm,
        observed / scale,
        s=22,
        facecolor="white",
        edgecolor="black",
        label="PAP pseudo-data",
        zorder=4,
    )
    prediction_line, = ax_profile.plot(
        r_mm,
        profiles[0] / scale,
        color="#D55E00",
        linewidth=2.0,
        label="FFNO prediction",
    )
    condition_text = ax_profile.text(
        0.03,
        0.04,
        "",
        transform=ax_profile.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    ax_profile.set_xlabel("Radius r [mm]")
    ax_profile.set_ylabel(r"Electron density $n_e$ [$10^{15}$ m$^{-3}$]")
    ax_profile.grid(alpha=0.22)
    ax_profile.legend(fontsize=8, frameon=False, loc="upper right")
    profile_min = min(float(np.min(profiles / scale)), float(np.min(observed / scale)))
    profile_max = max(float(np.max(profiles / scale)), float(np.max(observed / scale)))
    profile_pad = 0.05 * (profile_max - profile_min)
    ax_profile.set_ylim(profile_min - profile_pad, profile_max + profile_pad)

    completed_loss = ax_loss.scatter([], [], s=13, alpha=0.38, color="#0072B2")
    current_loss = ax_loss.scatter(
        [], [], marker="X", s=64, color="#CC79A7", edgecolor="black", zorder=6
    )
    best_line, = ax_loss.plot([], [], color="#D55E00", linewidth=2.0)
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, len(losses) + 8)
    ax_loss.set_ylim(max(0.8 * float(np.min(loss_percent)), 1.0e-3), 1.25 * float(np.max(loss_percent)))
    ax_loss.set_xlabel("FFNO evaluations")
    ax_loss.set_ylabel("PAP profile relative L2 [%]")
    ax_loss.grid(which="both", alpha=0.22)
    loss_text = ax_loss.text(
        0.98,
        0.96,
        "",
        transform=ax_loss.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.82},
    )

    vmin = RESPONSE_COLOR_FLOOR_PERCENT
    vmax = float(np.max(loss_percent))
    response = ax_map.contourf(
        grid_x,
        grid_y,
        surface,
        levels=np.geomspace(vmin, vmax, 22),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis_r",
        extend="max",
    )
    map_path, = ax_map.plot([], [], color="white", linewidth=0.9, alpha=0.68)
    map_history = ax_map.scatter([], [], s=9, color="white", alpha=0.34, edgecolor="none")
    map_current = ax_map.scatter(
        [], [], marker="X", s=68, color="#D55E00", edgecolor="black", zorder=7
    )
    ax_map.scatter(
        truth_score[0],
        truth_score[1],
        marker="*",
        s=155,
        color="white",
        edgecolor="black",
        zorder=8,
    )
    ax_map.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% variance)")
    ax_map.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% variance)")
    ax_map.grid(alpha=0.22)
    response_cbar = fig.colorbar(response, ax=ax_map, pad=0.02)
    response_cbar.set_label("PAP profile relative L2 [%]")
    response_ticks = [
        tick for tick in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0) if vmin <= tick <= vmax
    ]
    response_cbar.set_ticks(response_ticks)
    response_cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    title = fig.suptitle("", fontsize=12, y=1.01)
    fig.tight_layout()
    frames_dir = args.run / "trial_summary_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def update(index: int) -> None:
        upto = index + 1
        current_best = int(best_indices[index])
        prediction_line.set_ydata(profiles[index] / scale)
        cond = trial_rows[index]
        condition_text.set_text(
            f"T{upto}: PP0={float(cond['PP0']):.3f}\n"
            f"PA={float(cond['PA']):.4f}, γ={float(cond['gamma']):.4f}"
        )
        completed_loss.set_offsets(
            np.column_stack([np.arange(1, upto + 1), loss_percent[:upto]])
        )
        current_loss.set_offsets([[upto, loss_percent[index]]])
        best_line.set_data(np.arange(1, upto + 1), 100.0 * running_best[:upto])
        loss_text.set_text(
            f"current: {loss_percent[index]:.3f}%\n"
            f"best: {100.0 * running_best[index]:.3f}% (T{current_best + 1})"
        )
        map_path.set_data(scores[:upto, 0], scores[:upto, 1])
        map_history.set_offsets(scores[:upto])
        map_current.set_offsets([scores[index]])
        title.set_text(f"CMA-ES trial summary — T{upto}/{len(losses)}")

    frame_paths: list[Path] = []
    for index in range(len(losses)):
        update(index)
        frame_path = frames_dir / f"trial_{index + 1:04d}.png"
        fig.savefig(frame_path, dpi=int(args.dpi), bbox_inches="tight")
        frame_paths.append(frame_path)
        if (index + 1) % 50 == 0 or index == 0:
            print(f"rendered frames: {index + 1}/{len(losses)}", flush=True)
    plt.close(fig)

    gif_path = args.run / "trial_summary_animation.gif"
    duration_ms = max(20, int(round(1000.0 / float(args.fps))))
    with Image.open(frame_paths[0]) as first_raw:
        first = first_raw.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)

    def remaining_frames():
        for path in frame_paths[1:]:
            with Image.open(path) as image:
                yield image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)

    frame_durations = [duration_ms] * (len(frame_paths) - 1) + [1800]
    first.save(
        gif_path,
        save_all=True,
        append_images=remaining_frames(),
        duration=frame_durations,
        optimize=False,
        disposal=2,
    )
    metadata = {
        "run": str(args.run),
        "trial_count": len(losses),
        "fps": float(args.fps),
        "nominal_fps_duration_seconds": len(losses) / float(args.fps),
        "playback_duration_seconds": (len(losses) - 1) / float(args.fps) + 1.8,
        "loop": False,
        "final_frame_hold_ms": 1800,
        "response_color_floor_percent": RESPONSE_COLOR_FLOOR_PERCENT,
        "frames_dir": str(frames_dir),
        "gif": str(gif_path),
        "left_panel": "COMSOL truth line excluded from legend",
        "loss_panel": "completed trials, current trial, and best-so-far",
        "map_panel": "no legend; white history trajectory, current trial, and truth marker",
    }
    (args.run / "trial_summary_animation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"animation: {gif_path}", flush=True)


if __name__ == "__main__":
    main()
