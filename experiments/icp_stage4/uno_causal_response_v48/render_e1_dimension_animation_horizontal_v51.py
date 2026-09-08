#!/usr/bin/env python
"""Restore the established landscape GEC/ICP optimization-animation layout.

This renderer does not rerun optimization or training.  It uses the validated
v49 search histories and the v52 current-trial spatial cache, then exports
synchronized 1D, 2D, and combined landscape animations per model.  In every
frame the profile, field, geometry, annotation, and current marker resolve to
one identical sampled trial.  All outputs remain surrogate-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "reports/icp_conference_materials/icp_uno_e1_dimension_conference_v49/optimization_seed1237"
REPRESENTATIONS = ("dimension", "e1")
DEFAULT_OUT = SOURCE / "horizontal_animation_v51"
LABELS = {"dimension": "Formal Dimension", "e1": "E1 causal EM"}
MODEL_COLORS = {"dimension": "#0072B2", "e1": "#D55E00"}
FEASIBLE_BLUE = "#0072B2"
BEST_ORANGE = "#D55E00"
CURRENT_PINK = "#CC79A7"
FUTURE_GREY = "#999999"
COIL_RED = "#D62728"
PLASMA_BLUE = "#9ECAE1"
WAFER_GREY = "#737373"
WAFER_YELLOW = "#FFB000"
INK = "#182230"
MID = "#667085"
GRID = "#D0D5DD"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--animation-frames", type=int, default=90)
    parser.add_argument("--animation-fps", type=float, default=12.0)
    parser.add_argument("--final-hold-seconds", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=105)
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source in raw:
        row: dict[str, Any] = dict(source)
        row["density_feasible"] = _as_bool(source["density_feasible"])
        row["geometry_feasible"] = _as_bool(source["geometry_feasible"])
        for key in (
            "rank",
            "bohm_max_deviation",
            "mean_bohm_flux",
            "mean_density",
            "llcoil",
            "rrc",
            "rrce",
            "zzc",
            "pp",
            "pp0",
        ):
            row[key] = float(source[key])
        row["nncoil"] = int(source["nncoil"])
        rows.append(row)
    return rows


def _frame_plan(
    histories: dict[str, list[dict[str, Any]]],
    frames: int,
    fps: float,
    hold_seconds: float,
) -> dict[str, Any]:
    frame_count = max(12, int(frames))
    progress = np.linspace(0.0, 1.0, frame_count)
    hold = max(1, int(round(float(hold_seconds) * float(fps))))
    progress = np.concatenate((progress, np.ones(hold)))
    positions_by_model: dict[str, np.ndarray] = {}
    for name, rows in histories.items():
        positions = np.asarray(
            [min(len(rows) - 1, int(round(float(fraction) * (len(rows) - 1)))) for fraction in progress],
            dtype=np.int64,
        )
        positions_by_model[name] = positions
    return {
        "progress": progress,
        "hold": hold,
        "positions": positions_by_model,
    }


def _load_payload(source: Path, representation: str, expected_rows: int) -> dict[str, np.ndarray]:
    path = source / f"{representation}_candidate_payload.npz"
    with np.load(path, allow_pickle=False) as pack:
        payload = {name: np.asarray(pack[name]) for name in pack.files}
    if payload["vectors"].shape[0] != expected_rows or payload["profiles"].shape[0] != expected_rows:
        raise ValueError(f"{representation} payload/history row mismatch")
    return payload


def _animation_metadata(path: Path, render_steps: int, fps: float, hold: int) -> dict[str, Any]:
    encoded = Image.open(path)
    durations: list[int] = []
    for frame in range(int(getattr(encoded, "n_frames", 1))):
        encoded.seek(frame)
        durations.append(int(encoded.info.get("duration", 0)))
    frame_count = int(getattr(encoded, "n_frames", 1))
    size = list(encoded.size)
    encoded.close()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "render_steps": int(render_steps),
        "encoded_frame_count": frame_count,
        "fps": float(fps),
        "final_hold_render_steps": int(hold),
        "encoded_duration_seconds": float(sum(durations) / 1000.0),
        "final_frame_duration_seconds": float(durations[-1] / 1000.0),
        "pixel_size": size,
    }


def _load_spatial_cache(source: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, np.ndarray]]:
    with np.load(source / "animation_trial_spatial_cache_v52.npz", allow_pickle=False) as pack:
        grid = {
            "r": np.asarray(pack["r"], dtype=np.float32),
            "z": np.asarray(pack["z"], dtype=np.float32),
            "plasma": np.asarray(pack["plasma"], dtype=np.uint8),
            "selector": np.asarray(pack["selector"], dtype=np.uint8),
        }
        fields: dict[str, dict[int, np.ndarray]] = {}
        for name in REPRESENTATIONS:
            indices = np.asarray(pack[f"{name}_history_indices"], dtype=np.int64)
            values = np.asarray(pack[f"{name}_normalized_bohm"], dtype=np.float32)
            fields[name] = {int(index): values[position] for position, index in enumerate(indices)}
    return fields, grid


def _running_feasible(history: list[dict[str, Any]]) -> np.ndarray:
    running = np.full(len(history), np.nan, dtype=np.float64)
    best = np.inf
    for index, row in enumerate(history):
        if bool(row["density_feasible"] and row["geometry_feasible"]):
            best = min(best, 100.0 * float(row["bohm_max_deviation"]))
        if np.isfinite(best):
            running[index] = best
    return running


def _shared_limits(
    histories: dict[str, list[dict[str, Any]]],
    payloads: dict[str, dict[str, np.ndarray]],
    plan: dict[str, Any],
    display_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_profiles = []
    for name in REPRESENTATIONS:
        means = np.asarray([float(row["mean_bohm_flux"]) for row in histories[name]], dtype=np.float64)
        normalized = 100.0 * (payloads[name]["profiles"] / np.maximum(means[:, None], 1.0) - 1.0)
        selected_profiles.append(normalized[np.unique(plan["positions"][name])])
    max_abs = float(np.max(np.abs(np.concatenate(selected_profiles, axis=0))))
    profile_limit = max(20.0, 10.0 * math.ceil(1.08 * max_abs / 10.0))
    deviations = np.concatenate([100.0 * payloads[name]["deviations"] for name in REPRESENTATIONS])
    positive = deviations[np.isfinite(deviations) & (deviations > 0.0)]
    loss_limits = (max(1.0, 0.75 * float(positive.min())), 1.25 * float(positive.max()))
    return {
        "profile_limit_percent": profile_limit,
        "loss_limits_percent": list(loss_limits),
        "spatial_limits": [
            float(display_summary["display"]["display_min"]),
            float(display_summary["display"]["display_max"]),
        ],
    }


def _spatial_axis(
    axis: plt.Axes,
    first: np.ndarray,
    grid: dict[str, np.ndarray],
    limits: tuple[float, float],
) -> tuple[Any, Any]:
    r, z = grid["r"], grid["z"]
    extent = (float(r.min()), float(r.max()), float(z.min()), float(z.max()))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = axis.imshow(
        np.ma.masked_invalid(first),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm=Normalize(vmin=limits[0], vmax=limits[1], clip=True),
        interpolation="nearest",
    )
    axis.contour(r, z, grid["plasma"], levels=[0.5], colors="#4D4D4D", linewidths=0.9)
    axis.contour(r, z, grid["selector"], levels=[0.5], colors="white", linewidths=1.0, linestyles=":")
    axis.set(
        xlim=(float(r.min()), float(r.max())),
        ylim=(float(z.min()), float(z.max())),
        xlabel="r (cm)",
        ylabel="z (cm)",
        title="Predicted 2D Bohm-flux field",
    )
    axis.spines[["top", "right"]].set_visible(False)
    text = axis.text(
        0.03,
        0.96,
        "",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#344054", "edgecolor": "none", "alpha": 0.78},
    )
    return image, text


def _geometry_axis(axis: plt.Axes, grid: dict[str, np.ndarray]) -> tuple[list[Rectangle], Any]:
    r, z = grid["r"], grid["z"]
    extent = (float(r.min()), float(r.max()), float(z.min()), float(z.max()))
    plasma = np.asarray(grid["plasma"], dtype=float)
    selector = np.asarray(grid["selector"], dtype=float)
    axis.imshow(
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
    wafer_columns = np.any(selector > 0.5, axis=0)
    wafer_r_max = float(np.max(r[wafer_columns]))
    wafer_surface_z = float(np.min(z[np.any(selector > 0.5, axis=1)]))
    axis.add_patch(
        Rectangle(
            (float(r.min()), float(z.min())),
            wafer_r_max - float(r.min()),
            wafer_surface_z - float(z.min()),
            facecolor=WAFER_GREY,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
    )
    axis.contour(r, z, plasma, levels=[0.5], colors="#4D4D4D", linewidths=1.0)
    axis.imshow(
        np.ma.masked_where(selector <= 0.5, selector),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="autumn",
        vmin=0.0,
        vmax=1.2,
        alpha=0.85,
        interpolation="nearest",
    )
    patches: list[Rectangle] = []
    for _ in range(6):
        patch = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            facecolor=COIL_RED,
            edgecolor="black",
            linewidth=0.9,
            visible=False,
            zorder=4,
        )
        axis.add_patch(patch)
        patches.append(patch)
    axis.set(
        xlim=(float(r.min()), float(r.max())),
        ylim=(float(z.min()), float(z.max())),
        xlabel="r (cm)",
        ylabel="z (cm)",
        title="Chamber, wafer and coil layout",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=PLASMA_BLUE, edgecolor="#4D4D4D", label="plasma / chamber"),
            Patch(facecolor=WAFER_GREY, edgecolor="black", label="wafer"),
            Patch(facecolor=WAFER_YELLOW, edgecolor="none", label="wafer-near band"),
            Patch(facecolor=COIL_RED, edgecolor="black", label="coils"),
        ],
        frameon=False,
        fontsize=6.8,
        loc="lower right",
    )
    text = axis.text(
        0.03,
        0.37,
        "",
        transform=axis.transAxes,
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        zorder=5,
    )
    return patches, text


def _update_geometry(patches: list[Rectangle], row: dict[str, Any]) -> None:
    count = int(row["nncoil"])
    for slot, patch in enumerate(patches):
        if slot >= count:
            patch.set_visible(False)
            continue
        size = float(row[f"coil_size_{slot + 1:02d}"])
        r = float(row[f"r_center_{slot + 1:02d}"])
        z = float(row[f"z_center_{slot + 1:02d}"])
        patch.set_xy((r - 0.5 * size, z - 0.5 * size))
        patch.set_width(size)
        patch.set_height(size)
        patch.set_visible(True)


def _loss_axis(
    axis: plt.Axes,
    history: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
    limits: tuple[float, float],
) -> dict[str, Any]:
    x = np.arange(1, len(history) + 1)
    deviations = 100.0 * payload["deviations"]
    representative = np.unique(np.linspace(0, len(history) - 1, min(300, len(history)), dtype=np.int64))
    axis.scatter(
        x[representative],
        deviations[representative],
        s=9,
        alpha=0.24,
        color=FUTURE_GREY,
        label="Representative candidates",
        rasterized=True,
    )
    completed = axis.scatter([], [], s=9, alpha=0.48, color=FEASIBLE_BLUE, label="Completed feasible")
    best_line = axis.plot([], [], color=BEST_ORANGE, linewidth=2.1, label="Best so far")[0]
    current = axis.scatter([], [], marker="X", s=65, color=CURRENT_PINK, edgecolor="black", zorder=6, label="Current evaluation")
    text = axis.text(
        0.97,
        0.96,
        "",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    axis.set_yscale("log")
    axis.set(
        xlim=(1, len(history)),
        ylim=limits,
        xlabel="Surrogate objective evaluation",
        ylabel="Maximum deviation Dmax (%)",
        title="Optimization history",
    )
    axis.grid(which="both", color=GRID, linewidth=0.65, alpha=0.65)
    axis.legend(frameon=False, fontsize=6.7, loc="lower left")
    axis.spines[["top", "right"]].set_visible(False)
    return {
        "completed": completed,
        "best": best_line,
        "current": current,
        "text": text,
        "representative_indices": representative,
    }


def _profile_axis(axis: plt.Axes, radius: np.ndarray, color: str, limit: float) -> tuple[Any, Any]:
    line = axis.plot(radius, np.zeros_like(radius), color=color, linewidth=2.3)[0]
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1, label="area-weighted mean = 0%")
    axis.set(
        xlim=(0.0, 20.0),
        ylim=(-limit, limit),
        xlabel="Wafer radius r (cm)",
        ylabel="Deviation from mean (%)",
        title="Wafer-near Bohm-flux deviation",
    )
    axis.grid(color=GRID, linewidth=0.65, alpha=0.65)
    axis.legend(frameon=False, fontsize=7.0, loc="lower left")
    axis.spines[["top", "right"]].set_visible(False)
    text = axis.text(
        0.03,
        0.96,
        "",
        transform=axis.transAxes,
        va="top",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    return line, text


def _render_one(
    *,
    representation: str,
    mode: str,
    history: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
    fields: dict[int, np.ndarray],
    grid: dict[str, np.ndarray],
    plan: dict[str, Any],
    shared: dict[str, Any],
    out_dir: Path,
    fps: float,
    dpi: int,
) -> dict[str, Any]:
    if mode == "combined":
        figsize = (17.4, 4.2)
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        profile_axis, spatial_axis, loss_axis, geometry_axis = axes
        fig.subplots_adjust(left=0.050, right=0.985, top=0.80, bottom=0.19, wspace=0.34)
        kind = "1d_2d_horizontal"
        layout = "1 x 4 landscape"
    elif mode == "two_dimensional":
        figsize = (13.4, 4.2)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        spatial_axis, loss_axis, geometry_axis = axes
        profile_axis = None
        fig.subplots_adjust(left=0.060, right=0.985, top=0.80, bottom=0.19, wspace=0.35)
        kind = "2d_horizontal"
        layout = "1 x 3 landscape"
    elif mode == "one_dimensional":
        figsize = (13.4, 4.2)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        profile_axis, loss_axis, geometry_axis = axes
        spatial_axis = None
        fig.subplots_adjust(left=0.060, right=0.985, top=0.80, bottom=0.19, wspace=0.35)
        kind = "1d_horizontal"
        layout = "1 x 3 landscape"
    else:
        raise ValueError(f"unsupported animation mode: {mode}")
    title = fig.suptitle("", fontsize=12.3, y=0.965)
    fig.text(
        0.5,
        0.885,
        "Each frame shows one sampled current trial; orange line is best-so-far. Same 7,040-evaluation history; surrogate-only.",
        ha="center",
        fontsize=8.2,
        color=MID,
    )
    positions = plan["positions"][representation]
    first_index = int(positions[0])
    image = spatial_text = None
    if spatial_axis is not None:
        image, spatial_text = _spatial_axis(
            spatial_axis,
            fields[first_index],
            grid,
            tuple(shared["spatial_limits"]),
        )
        colorbar = fig.colorbar(image, ax=spatial_axis, fraction=0.047, pad=0.035)
        colorbar.set_label(r"$\Gamma_B(r,z)/\langle\Gamma_B\rangle_{wafer}$", fontsize=7.6)
        colorbar.ax.tick_params(labelsize=7)
    loss_artists = _loss_axis(loss_axis, history, payload, tuple(shared["loss_limits_percent"]))
    coil_patches, geometry_text = _geometry_axis(geometry_axis, grid)
    means = np.asarray([float(row["mean_bohm_flux"]) for row in history], dtype=np.float64)
    profile_deviation = 100.0 * (payload["profiles"] / np.maximum(means[:, None], 1.0) - 1.0)
    profile_line = profile_text = None
    if profile_axis is not None:
        profile_line, profile_text = _profile_axis(
            profile_axis,
            payload["radial_centers"],
            MODEL_COLORS[representation],
            float(shared["profile_limit_percent"]),
        )
    feasible = payload["feasible"].astype(bool)
    deviation = 100.0 * payload["deviations"]
    running = _running_feasible(history)

    def update(frame: int):
        current_index = int(positions[frame])
        position = current_index
        row = history[current_index]
        dmax = 100.0 * float(row["bohm_max_deviation"])
        density = float(row["mean_density"]) / 1.0e17
        if image is not None and spatial_text is not None:
            image.set_data(np.ma.masked_invalid(fields[current_index]))
            spatial_text.set_text(f"current N={int(row['nncoil'])}\ntrial Dmax={dmax:.2f}%")
        _update_geometry(coil_patches, row)
        geometry_text.set_text(
            f"current N={int(row['nncoil'])}\n"
            f"height span={float(row.get('height_span_cm', 0.0)):.2f} cm\n"
            f"size span={float(row.get('size_span_cm', 0.0)):.2f} cm"
        )
        upto = position + 1
        representative_indices = loss_artists["representative_indices"]
        completed_indices = representative_indices[
            (representative_indices < upto) & feasible[representative_indices]
        ]
        if completed_indices.size:
            loss_artists["completed"].set_offsets(
                np.column_stack((completed_indices + 1, deviation[completed_indices]))
            )
        else:
            loss_artists["completed"].set_offsets(np.empty((0, 2)))
        finite = np.isfinite(running[:upto])
        loss_artists["best"].set_data(np.arange(1, upto + 1)[finite], running[:upto][finite])
        loss_artists["current"].set_offsets([[current_index + 1, deviation[current_index]]])
        best_value = float(running[position]) if np.isfinite(running[position]) else float("nan")
        loss_artists["text"].set_text(
            f"current trial: {current_index + 1}/{len(history)}\n"
            f"trial Dmax: {deviation[current_index]:.2f}%\n"
            f"best feasible: {best_value:.2f}%"
        )
        if profile_line is not None and profile_text is not None:
            profile_line.set_ydata(profile_deviation[current_index])
            profile_text.set_text(f"trial Dmax: {dmax:.2f}%\nmean density: {density:.2f} x 1e17 m^-3")
        displayed_frame = min(frame + 1, len(plan["progress"]) - int(plan["hold"]))
        displayed_total = len(plan["progress"]) - int(plan["hold"])
        title.set_text(
            f"ICP Bohm-flux optimization — {LABELS[representation]} — "
            f"{displayed_frame}/{displayed_total}"
        )
        artists: list[Any] = [
            title,
            *coil_patches,
            geometry_text,
            loss_artists["completed"],
            loss_artists["best"],
            loss_artists["current"],
            loss_artists["text"],
        ]
        if image is not None and spatial_text is not None:
            artists.extend([image, spatial_text])
        if profile_line is not None and profile_text is not None:
            artists.extend([profile_line, profile_text])
        return artists

    render_steps = len(plan["progress"])
    animation = FuncAnimation(fig, update, frames=render_steps, interval=1000.0 / fps, blit=False)
    rep_dir = out_dir / representation
    rep_dir.mkdir(parents=True, exist_ok=True)
    gif = rep_dir / f"{representation}_optimization_{kind}.gif"
    animation.save(gif, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": "white"})
    update(render_steps - 1)
    final_png = rep_dir / f"{representation}_optimization_{kind}_final.png"
    fig.savefig(final_png, dpi=190, facecolor="white")
    plt.close(fig)
    metadata = _animation_metadata(gif, render_steps, fps, int(plan["hold"]))
    metadata.update(
        {
            "final_png": final_png.relative_to(ROOT).as_posix(),
            "representation": representation,
            "layout": layout,
            "canvas_inches": list(figsize),
            "frame_state": "current sampled trial",
            "frame_index_invariant": "profile = spatial field = geometry = current marker",
            "best_so_far_usage": "orange optimization-history line only",
        }
    )
    return metadata


def main() -> int:
    args = _args()
    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    histories = {name: _read_history(source / f"{name}_optimization_history.csv") for name in REPRESENTATIONS}
    payloads = {name: _load_payload(source, name, len(histories[name])) for name in REPRESENTATIONS}
    if any(len(histories[name]) != 7040 for name in REPRESENTATIONS):
        raise ValueError("expected the validated 7,040-candidate v49 histories")
    plan = _frame_plan(
        histories,
        int(args.animation_frames),
        float(args.animation_fps),
        float(args.final_hold_seconds),
    )
    fields, grid = _load_spatial_cache(source)
    frame_index_qa: dict[str, dict[str, Any]] = {}
    for name in REPRESENTATIONS:
        expected = set(map(int, np.unique(plan["positions"][name])))
        actual = set(fields[name])
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"spatial cache is missing {name} history indices: {missing}")
        frame_index_qa[name] = {
            "displayed_unique_history_indices": len(expected),
            "cached_history_indices": len(actual),
            "exact_spatial_index_set": bool(expected == actual),
            "first_history_index": int(plan["positions"][name][0]),
            "middle_history_index": int(plan["positions"][name][int(args.animation_frames) // 2]),
            "last_history_index": int(plan["positions"][name][int(args.animation_frames) - 1]),
        }
    display_summary = json.loads((source / "animation_trial_spatial_cache_v52_summary.json").read_text(encoding="utf-8"))
    shared = _shared_limits(histories, payloads, plan, display_summary)
    outputs: dict[str, dict[str, Any]] = {}
    for name in REPRESENTATIONS:
        outputs[name] = {
            "one_dimensional": _render_one(
                representation=name,
                mode="one_dimensional",
                history=histories[name],
                payload=payloads[name],
                fields=fields[name],
                grid=grid,
                plan=plan,
                shared=shared,
                out_dir=out_dir,
                fps=float(args.animation_fps),
                dpi=int(args.dpi),
            ),
            "two_dimensional": _render_one(
                representation=name,
                mode="two_dimensional",
                history=histories[name],
                payload=payloads[name],
                fields=fields[name],
                grid=grid,
                plan=plan,
                shared=shared,
                out_dir=out_dir,
                fps=float(args.animation_fps),
                dpi=int(args.dpi),
            ),
            "combined_1d_2d": _render_one(
                representation=name,
                mode="combined",
                history=histories[name],
                payload=payloads[name],
                fields=fields[name],
                grid=grid,
                plan=plan,
                shared=shared,
                out_dir=out_dir,
                fps=float(args.animation_fps),
                dpi=int(args.dpi),
            ),
        }
    summary = {
        "schema": "icp-uno-e1-dimension-horizontal-current-trial-animation-v52",
        "reference_assets": {
            "icp": "reports/icp_conference_materials/axisymmetric_dimension_sdf_bohm_optimization_model_case_v1/optimization_animation_v1",
            "gec": "reports/gec_conference_materials/optimization_assets/field_loss_animation",
        },
        "history_rows": {name: len(histories[name]) for name in REPRESENTATIONS},
        "shared_limits": shared,
        "palette": {
            "dimension": MODEL_COLORS["dimension"],
            "e1": MODEL_COLORS["e1"],
            "spatial": "viridis",
            "completed_feasible": FEASIBLE_BLUE,
            "future_or_infeasible": FUTURE_GREY,
            "best_so_far": BEST_ORANGE,
            "current": CURRENT_PINK,
            "coils": COIL_RED,
            "wafer": WAFER_GREY,
            "wafer_band": WAFER_YELLOW,
        },
        "layout_fixed_between_frames": True,
        "model_stacking": "none; one landscape GIF per model",
        "frame_state": "current sampled trial; running best appears only in the orange history line",
        "displayed_trial_count": int(args.animation_frames),
        "full_history_rows_per_model": 7040,
        "frame_index_qa": frame_index_qa,
        "outputs": outputs,
        "surrogate_only_pending_COMSOL_validation": True,
        "validation_passed": bool(
            all(len(histories[name]) == 7040 for name in REPRESENTATIONS)
            and shared["spatial_limits"][1] > shared["spatial_limits"][0] >= 0.0
            and shared["profile_limit_percent"] > 0.0
            and shared["loss_limits_percent"][1] > shared["loss_limits_percent"][0] > 0.0
            and all(frame_index_qa[name]["exact_spatial_index_set"] for name in REPRESENTATIONS)
        ),
    }
    path = out_dir / "horizontal_animation_v51_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
