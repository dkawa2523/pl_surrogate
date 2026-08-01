"""Create conference-ready ICP Bohm-profile optimization figures and animation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from experiments.icp_stage4.scripts.optimize_bohm_flux_representation_v1 import BohmMetrics
from experiments.icp_stage4.scripts.optimize_bohm_flux_sdf_500trials_v4 import _sdf_variable_height
from experiments.icp_stage4.scripts.optimize_regular_layout_representation_v1 import (
    DEFAULT_RUNS,
    RepresentationSurrogate,
)


DATASET_ROOT = Path("data/outputs_icp_stage4_enriched_360_csv_npz_causal_em_pabs_v1")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/icp_stage4_bohm_flux_sdf_500trials_spacing_slope_v4"),
    )
    parser.add_argument(
        "--representation",
        choices=("union_sdf", "dimension"),
        default="union_sdf",
    )
    parser.add_argument(
        "--trial-order",
        choices=("interleaved", "trial_id"),
        default="interleaved",
        help="Interleave coil counts for a fair truncated presentation sequence.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument(
        "--profile-ymax-scaled",
        type=float,
        default=None,
        help="Fixed profile upper limit in units of 1e20 m^-2 s^-1.",
    )
    parser.add_argument(
        "--stop-trial",
        type=int,
        default=None,
        help="Stop after this trial; the final frame shows the best case found by then.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _ordered_trials(rows: list[dict[str, str]], order: str) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["trial_id"]))
    if order == "trial_id":
        return ordered
    groups = {
        count: [row for row in ordered if int(row["nncoil"]) == count]
        for count in (2, 3, 4, 5)
    }
    output: list[dict[str, str]] = []
    for local_index in range(max(len(group) for group in groups.values())):
        for count in (2, 3, 4, 5):
            if local_index < len(groups[count]):
                output.append(groups[count][local_index])
    return output


def _tensor(rows: list[dict[str, str]], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor([float(row[key]) for row in rows], dtype=torch.float32, device=device)


def _load_or_predict(
    run: Path,
    rows: list[dict[str, str]],
    surrogate: RepresentationSurrogate,
    metric: BohmMetrics,
    *,
    batch_size: int,
    density_threshold: float,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = run / "trial_bohm_profiles.npz"
    trial_ids = np.asarray([int(row["trial_id"]) for row in rows], dtype=np.int64)
    if cache.exists():
        with np.load(cache) as stored:
            if np.array_equal(stored["trial_ids"], trial_ids):
                print(f"reusing Bohm profile cache: {cache}", flush=True)
                return tuple(
                    np.asarray(stored[key])
                    for key in ("radius", "profiles", "losses", "deviations", "mean_density")
                )

    radius = surrogate.radial_np.astype(np.float64)
    radial_weights = np.maximum(radius, 0.5 * surrogate.dr)
    bin_radius = np.asarray(
        [
            np.average(radius[mask.detach().cpu().numpy()], weights=radial_weights[mask.detach().cpu().numpy()])
            for mask in metric.bin_masks
        ],
        dtype=np.float64,
    )
    profiles = np.empty((len(rows), len(metric.bin_masks)), dtype=np.float64)
    losses = np.empty(len(rows), dtype=np.float64)
    deviations = np.empty(len(rows), dtype=np.float64)
    mean_density = np.empty(len(rows), dtype=np.float64)
    for count in (2, 3, 4, 5):
        indices = [index for index, row in enumerate(rows) if int(row["nncoil"]) == count]
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch_rows = [rows[index] for index in batch_indices]
            decoded = {
                key: _tensor(batch_rows, key, surrogate.device)
                for key in ("llcoil", "rrc", "rrce", "zzc", "pp", "pp0")
            }
            centers = torch.stack(
                [_tensor(batch_rows, f"r_center_{slot + 1:02d}", surrogate.device) for slot in range(count)],
                dim=1,
            )
            heights = torch.stack(
                [_tensor(batch_rows, f"z_center_{slot + 1:02d}", surrogate.device) for slot in range(count)],
                dim=1,
            )
            with torch.no_grad():
                if representation == "union_sdf":
                    mask, sdf = _sdf_variable_height(surrogate, decoded, centers, heights)
                    fields, _, _ = surrogate.predict_from_sdf(
                        decoded, count=count, mask=mask, sdf=sdf
                    )
                else:
                    fields, _, _ = surrogate.predict(decoded, count=count)
                values = metric(fields)
                deficit = torch.relu(
                    (float(density_threshold) - values["mean_density"]) / float(density_threshold)
                )
                loss = values["smooth_max_deviation"] + 20.0 * deficit.square()
            profiles[batch_indices] = values["bin_flux"].cpu().numpy()
            losses[batch_indices] = loss.cpu().numpy()
            deviations[batch_indices] = values["max_deviation"].cpu().numpy()
            mean_density[batch_indices] = values["mean_density"].cpu().numpy()
            completed = min(start + len(batch_indices), len(indices))
            if completed == len(indices) or completed % 40 == 0:
                print(f"predicted {count}-coil profiles: {completed}/{len(indices)}", flush=True)
    np.savez_compressed(
        cache,
        trial_ids=trial_ids,
        radius=bin_radius,
        profiles=profiles,
        losses=losses,
        deviations=deviations,
        mean_density=mean_density,
    )
    return bin_radius, profiles, losses, deviations, mean_density


def _running_best(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(losses.size, dtype=np.int64)
    values = np.empty(losses.size, dtype=np.float64)
    current = 0
    for index, value in enumerate(losses):
        if value < losses[current]:
            current = index
        indices[index] = current
        values[index] = losses[current]
    return indices, values


def main() -> None:
    args = _args()
    rows = _ordered_trials(_read_csv(args.run / "trials.csv"), str(args.trial_order))
    if not rows:
        raise ValueError("trials.csv is empty")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    representation_label = "SDF" if args.representation == "union_sdf" else "dimension"
    density_threshold = float(summary["density_threshold"])
    surrogate = RepresentationSurrogate(
        representation=str(args.representation),
        run_dir=DEFAULT_RUNS[str(args.representation)],
        dataset_root=DATASET_ROOT,
        protocol="structure_holdout",
        wafer_layers=int(summary["wafer_layers"]),
    )
    metric = BohmMetrics(surrogate, radial_bins=32)
    radius, profiles, losses, deviations, mean_density = _load_or_predict(
        args.run,
        rows,
        surrogate,
        metric,
        batch_size=int(args.batch_size),
        density_threshold=density_threshold,
        representation=str(args.representation),
    )
    best_indices, running_best = _running_best(losses)
    stop_trial = len(rows) if args.stop_trial is None else min(int(args.stop_trial), len(rows))
    if stop_trial < 1:
        raise ValueError("stop-trial must be at least one")
    profile_scale = 1.0e20
    profile_upper = (
        float(args.profile_ymax_scaled)
        if args.profile_ymax_scaled is not None
        else 1.08 * float(np.nanmax(profiles / profile_scale))
    )
    if profile_upper < float(np.nanmax(profiles / profile_scale)):
        raise ValueError("profile-ymax-scaled would clip at least one trial profile")
    loss_floor = max(0.75 * float(np.min(losses)), 1.0e-4)
    loss_upper = 1.25 * float(np.max(losses))
    r_min, r_max = float(np.min(surrogate.r_np)), float(np.max(surrogate.r_np))
    z_min, z_max = float(np.min(surrogate.z_np)), float(np.max(surrogate.z_np))
    extent = [r_min, r_max, z_min, z_max]
    plasma = np.asarray(surrogate.plasma_np, dtype=float)
    wafer_band = np.asarray(surrogate.selector_np, dtype=float)
    wafer_columns = np.any(wafer_band > 0.5, axis=0)
    wafer_r_max = float(np.max(surrogate.r_np[0, wafer_columns]))
    wafer_surface_z = float(np.min(surrogate.z_np[wafer_band > 0.5]))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.15), constrained_layout=True)
    ax_profile, ax_loss, ax_geometry = axes
    profile_line, = ax_profile.plot(radius, profiles[0] / profile_scale, color="#D55E00", linewidth=2.2)
    profile_mean = ax_profile.axhline(
        float(np.mean(profiles[0])) / profile_scale,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="radial mean",
    )
    profile_text = ax_profile.text(
        0.03,
        0.96,
        "",
        transform=ax_profile.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.84},
    )
    ax_profile.set(
        xlim=(float(radius.min()), float(radius.max())),
        ylim=(0.0, profile_upper),
        xlabel="Wafer radius r [dataset coordinate]",
        ylabel=r"Bohm ion flux [$10^{20}$ m$^{-2}$ s$^{-1}$]",
        title="Wafer-near Bohm flux (10 layers)",
    )
    ax_profile.grid(alpha=0.22)
    ax_profile.legend(frameon=False, fontsize=8, loc="lower left")

    completed_loss = ax_loss.scatter([], [], s=11, alpha=0.34, color="#0072B2")
    current_loss = ax_loss.scatter(
        [], [], marker="X", s=66, color="#CC79A7", edgecolor="black", zorder=6
    )
    best_line, = ax_loss.plot([], [], color="#D55E00", linewidth=2.0, label="best-so-far")
    loss_text = ax_loss.text(
        0.98,
        0.96,
        "",
        transform=ax_loss.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.84},
    )
    ax_loss.set_yscale("log")
    ax_loss.set(
        xlim=(0, stop_trial + 8),
        ylim=(loss_floor, loss_upper),
        xlabel=f"{representation_label} surrogate trials",
        ylabel="Bohm objective loss (log)",
        title="Joint search: 2-5 coils",
    )
    ax_loss.grid(which="both", alpha=0.22)
    ax_loss.legend(frameon=False, fontsize=8, loc="lower left")

    ax_geometry.imshow(
        np.ma.masked_where(plasma <= 0.5, plasma),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="Blues",
        vmin=0.0,
        vmax=1.6,
        alpha=0.38,
        interpolation="nearest",
    )
    ax_geometry.add_patch(
        Rectangle(
            (r_min, z_min),
            wafer_r_max - r_min,
            wafer_surface_z - z_min,
            facecolor="#737373",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
    )
    ax_geometry.contour(
        surrogate.r_np,
        surrogate.z_np,
        plasma,
        levels=[0.5],
        colors="#4D4D4D",
        linewidths=1.1,
    )
    ax_geometry.imshow(
        np.ma.masked_where(wafer_band <= 0.5, wafer_band),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="autumn",
        vmin=0.0,
        vmax=1.2,
        alpha=0.85,
        interpolation="nearest",
    )
    geometry_text = ax_geometry.text(
        0.03,
        0.97,
        "",
        transform=ax_geometry.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.84},
    )
    ax_geometry.set(
        xlim=(r_min, r_max),
        ylim=(z_min, z_max),
        xlabel="r [dataset coordinate]",
        ylabel="z [dataset coordinate]",
        title="Chamber, wafer and coil layout",
    )
    ax_geometry.legend(
        handles=[
            Patch(facecolor="#9ECAE1", edgecolor="#4D4D4D", label="plasma / chamber"),
            Patch(facecolor="#737373", edgecolor="black", label="wafer"),
            Patch(facecolor="#FFB000", edgecolor="none", label="wafer-near band"),
            Patch(facecolor="#D62728", edgecolor="black", label="coils"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="lower right",
    )
    title = fig.suptitle("", fontsize=12.5)
    coil_patches: list[Rectangle] = []

    def update(index: int, *, show_best: bool = False) -> None:
        shown = int(best_indices[index]) if show_best else index
        row = rows[shown]
        upto = index + 1
        current_profile = profiles[shown] / profile_scale
        profile_line.set_ydata(current_profile)
        profile_mean.set_ydata([float(np.mean(current_profile)), float(np.mean(current_profile))])
        profile_text.set_text(
            f"max deviation: {100.0 * deviations[shown]:.2f}%\n"
            f"mean density: {mean_density[shown] / 1.0e17:.2f} x 1e17 m^-3"
        )
        completed_loss.set_offsets(np.column_stack([np.arange(1, upto + 1), losses[:upto]]))
        current_loss.set_offsets([[shown + 1, losses[shown]]])
        best_line.set_data(np.arange(1, upto + 1), running_best[:upto])
        best_trial = int(best_indices[index])
        loss_text.set_text(
            f"current: {losses[shown]:.4f}\n"
            f"best: {running_best[index]:.4f} (T{best_trial + 1})"
        )
        for patch in coil_patches:
            patch.remove()
        coil_patches.clear()
        count = int(row["nncoil"])
        size = float(row["llcoil"])
        for slot in range(count):
            center_r = float(row[f"r_center_{slot + 1:02d}"])
            center_z = float(row[f"z_center_{slot + 1:02d}"])
            patch = Rectangle(
                (center_r - 0.5 * size, center_z - 0.5 * size),
                size,
                size,
                facecolor="#D62728",
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
            ax_geometry.add_patch(patch)
            coil_patches.append(patch)
        geometry_text.set_text(
            f"N={count}, slope={float(row['coil_line_slope']):+.3f}\n"
            f"PP={float(row['pp']):.0f}, PP0={float(row['pp0']):.4f}"
        )
        if show_best:
            title.set_text(
                f"ICP {representation_label} Bohm-flux optimization - T{upto}/{stop_trial} "
                f"(best found: T{shown + 1})"
            )
        else:
            title.set_text(
                f"ICP {representation_label} Bohm-flux optimization - "
                f"T{upto}/{stop_trial} (current trial)"
            )

    frames_dir = args.run / "bohm_trial_animation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_count = min(int(args.frames), stop_trial)
    frame_indices = np.linspace(0, stop_trial - 1, frame_count, dtype=int)
    frame_paths: list[Path] = []
    for frame_number, index in enumerate(frame_indices, start=1):
        update(int(index), show_best=(frame_number == frame_count))
        frame_path = frames_dir / f"trial_{frame_number:04d}.png"
        fig.savefig(frame_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        frame_paths.append(frame_path)
        if frame_number % 50 == 0 or frame_number == 1:
            print(f"rendered frames: {frame_number}/{frame_count}", flush=True)

    update(stop_trial - 1, show_best=True)
    stem = args.run / "bohm_optimization_conference_layout"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gif_path = args.run / "bohm_optimization_trial_animation.gif"
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
        "representation": str(args.representation),
        "trial_order": str(args.trial_order),
        "trial_count": len(rows),
        "animation_stop_trial": stop_trial,
        "frame_count": frame_count,
        "fps": float(args.fps),
        "playback_duration_seconds": (frame_count - 1) / float(args.fps) + 1.8,
        "fixed_profile_ylim_scaled_1e20": [0.0, profile_upper],
        "wafer_layers": int(summary["wafer_layers"]),
        "radial_bins": int(profiles.shape[1]),
        "gif": str(gif_path),
        "static_png": str(stem.with_suffix(".png")),
        "frames_dir": str(frames_dir),
        "left_panel": "current-trial wafer-near 1D Bohm profile with fixed y range",
        "middle_panel": "current loss and best-so-far across the joint 2-5 coil search",
        "right_panel": "fixed chamber/plasma and wafer context plus current-trial coils",
        "final_frame": f"best case found within trials 1-{stop_trial}",
        "interpretation_limit": summary.get("interpretation_limit"),
    }
    (args.run / "bohm_optimization_trial_animation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"animation: {gif_path}", flush=True)


if __name__ == "__main__":
    main()
