"""Plot every TPE trial's radial electron-density profile for fixed sensors.

Profiles are regenerated from the physical trial conditions stored by the
sensor-input optimization run.  Line colour is a continuous mapping of the
TPE trial number; the fixed noisy sensor observations are drawn as points.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml

import run_gec_ccp_ne_profile_assimilation as common
from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.infer.assimilation import (
    LinearNeProfileAssimilationEngine,
    LinearNeProfileObservation,
)


DEFAULT_OPTIMIZATION_ROOT = Path("runs/gec_ccp_multi_sensor_input_optimization_cpu")
DEFAULT_OUT_ROOT = DEFAULT_OPTIMIZATION_ROOT / "tpe_trial_profiles"
METHOD = "tpe"
DENSITY_DISPLAY_SCALE = 1.0e15


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _measurement_dirs(root: Path) -> list[Path]:
    paths = [path for path in root.glob("M*") if path.is_dir()]
    paths.sort(key=lambda path: int(path.name[1:]))
    if not paths:
        raise FileNotFoundError(f"no measurement directories found under {root}")
    return paths


def _load_observation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    r_m = np.asarray([float(row["r_m"]) for row in rows], dtype=np.float64)
    truth_ne = np.asarray([float(row["truth_ne"]) for row in rows], dtype=np.float64)
    sensor_ne = np.asarray([float(row["sensor_ne"]) for row in rows], dtype=np.float64)
    if r_m.size < 2 or np.any(np.diff(r_m) <= 0.0):
        raise ValueError(f"sensor radii must be strictly increasing: {path}")
    if np.any(sensor_ne <= 0.0) or not np.all(np.isfinite(sensor_ne)):
        raise ValueError(f"sensor electron density must be finite and positive: {path}")
    return r_m, truth_ne, sensor_ne


def _load_tpe_trials(path: Path) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if row["method"].strip().lower() == METHOD]
    rows.sort(key=lambda row: int(row["trial"]))
    expected = list(range(1, len(rows) + 1))
    actual = [int(row["trial"]) for row in rows]
    if not rows or actual != expected:
        raise ValueError(f"TPE trials must be contiguous from 1 in {path}; got={actual}")
    return rows


def _assimilation_engine(
    engine: Any,
    *,
    r_m: np.ndarray,
    z_m: float,
    sensor_ne: np.ndarray,
    case_id: str,
    td: float,
) -> LinearNeProfileAssimilationEngine:
    density_scale = float(np.sqrt(np.mean(sensor_ne * sensor_ne)))
    observation = LinearNeProfileObservation(
        r_m=r_m,
        z_m=z_m,
        ne_obs=sensor_ne,
        density_scale=density_scale,
        covariance_scaled=np.eye(r_m.size, dtype=np.float64),
        case_id=case_id,
    )
    return LinearNeProfileAssimilationEngine(
        engine,
        observation=observation,
        fixed_cond={"Td": td},
        setpoints={},
        nuisance_rel_sigma={},
    )


def _regenerate_profiles(
    *,
    engine: Any,
    case: dict[str, Any],
    metadata: dict[str, Any],
    trials: list[dict[str, str]],
    sensor_r_m: np.ndarray,
    plot_r_m: np.ndarray,
    sensor_ne: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    case_id = str(metadata["case_id"])
    z_m = float(metadata["z_m"])
    td = float(dict(metadata["fixed_cond"])["Td"])
    geom = common._case_geom(case)
    axis = common._case_axis(case)
    geom_ctx = common._geom_context(engine, geom, axis)
    assimilation = _assimilation_engine(
        engine,
        r_m=sensor_r_m,
        z_m=z_m,
        sensor_ne=sensor_ne,
        case_id=case_id,
        td=td,
    )
    truth_plot_ne = common._profile(case["y"]["ne"], geom_ctx, z_m, plot_r_m)
    profiles: list[np.ndarray] = []
    profile_rows: list[dict[str, Any]] = []
    max_loss_difference = 0.0
    for row in trials:
        cond = {key: float(row[key]) for key in ("PP0", "PA", "gamma")}
        result = assimilation.single_run_aggregated(
            cond=cond,
            geom=geom,
            axis=axis,
            save_outputs=False,
        )
        sensor_profile = common._profile(result.fields_phys["ne"], geom_ctx, z_m, sensor_r_m)
        plot_profile = common._profile(result.fields_phys["ne"], geom_ctx, z_m, plot_r_m)
        recomputed_loss = float(
            np.linalg.norm(sensor_profile - sensor_ne) / np.linalg.norm(sensor_ne)
        )
        stored_loss = float(row["loss"])
        difference = abs(recomputed_loss - stored_loss)
        max_loss_difference = max(max_loss_difference, difference)
        profiles.append(plot_profile)
        for radius, prediction, clean_truth in zip(
            plot_r_m, plot_profile, truth_plot_ne, strict=True
        ):
            sensor_matches = np.flatnonzero(np.isclose(sensor_r_m, radius, rtol=0.0, atol=1.0e-12))
            observation: float | str = (
                float(sensor_ne[int(sensor_matches[0])]) if sensor_matches.size == 1 else ""
            )
            profile_rows.append(
                {
                    "measurement_id": metadata.get("measurement_id", ""),
                    "case_id": case_id,
                    "optimizer_seed": int(metadata["optimizer_seed"]),
                    "method": METHOD,
                    "trial": int(row["trial"]),
                    "r_m": float(radius),
                    "predicted_ne": float(prediction),
                    "truth_ne": float(clean_truth),
                    "sensor_ne": observation,
                    "PP0": cond["PP0"],
                    "PA": cond["PA"],
                    "gamma": cond["gamma"],
                    "stored_loss": stored_loss,
                    "recomputed_loss": recomputed_loss,
                }
            )
    if max_loss_difference > 1.0e-8:
        raise RuntimeError(
            f"regenerated profile losses do not match stored TPE trials for {case_id}: "
            f"max_abs_difference={max_loss_difference:.3e}"
        )
    return truth_plot_ne, np.stack(profiles), profile_rows, max_loss_difference


def _draw_profiles(
    ax: Any,
    *,
    profile_r_m: np.ndarray,
    truth_ne: np.ndarray,
    sensor_r_m: np.ndarray,
    sensor_ne: np.ndarray,
    profiles: np.ndarray,
    trials: list[dict[str, str]],
    measurement_id: str,
    case_id: str,
    td: float,
    z_m: float,
    cmap: Any,
    norm: Normalize,
    include_xlabel: bool,
) -> None:
    profile_r_mm = 1000.0 * profile_r_m
    sensor_r_mm = 1000.0 * sensor_r_m
    for profile, row in zip(profiles, trials, strict=True):
        trial = int(row["trial"])
        ax.plot(
            profile_r_mm,
            profile / DENSITY_DISPLAY_SCALE,
            color=cmap(norm(trial)),
            linewidth=0.85,
            alpha=0.34,
            zorder=1,
        )
    best_index = int(np.argmin([float(row["loss"]) for row in trials]))
    best_trial = int(trials[best_index]["trial"])
    ax.plot(
        profile_r_mm,
        profiles[best_index] / DENSITY_DISPLAY_SCALE,
        color=cmap(norm(best_trial)),
        linewidth=2.6,
        alpha=1.0,
        label=f"best TPE trial {best_trial}",
        zorder=3,
    )
    ax.plot(
        profile_r_mm,
        truth_ne / DENSITY_DISPLAY_SCALE,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="clean COMSOL truth",
        zorder=2,
    )
    ax.scatter(
        sensor_r_mm,
        sensor_ne / DENSITY_DISPLAY_SCALE,
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1,
        label="target sensor points",
        zorder=4,
    )
    ax.set_title(
        f"{measurement_id}: {case_id}\nTd={td:.3g}, sensor z={1000.0 * z_m:.3g} mm",
        fontsize=10,
    )
    if include_xlabel:
        ax.set_xlabel("radius r [mm]")
    ax.set_ylabel(r"electron density $n_e$ [$10^{15}$ m$^{-3}$]")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8)


def _plot_case(
    out_path: Path,
    *,
    profile_r_m: np.ndarray,
    truth_ne: np.ndarray,
    sensor_r_m: np.ndarray,
    sensor_ne: np.ndarray,
    profiles: np.ndarray,
    trials: list[dict[str, str]],
    metadata: dict[str, Any],
    cmap: Any,
    norm: Normalize,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _draw_profiles(
        ax,
        profile_r_m=profile_r_m,
        truth_ne=truth_ne,
        sensor_r_m=sensor_r_m,
        sensor_ne=sensor_ne,
        profiles=profiles,
        trials=trials,
        measurement_id=str(metadata["measurement_id"]),
        case_id=str(metadata["case_id"]),
        td=float(dict(metadata["fixed_cond"])["Td"]),
        z_m=float(metadata["z_m"]),
        cmap=cmap,
        norm=norm,
        include_xlabel=True,
    )
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("TPE trial number")
    colorbar.set_ticks([1, 20, 40, 60, 80, 100])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_combined(
    out_path: Path,
    measurements: list[dict[str, Any]],
    *,
    cmap: Any,
    norm: Normalize,
) -> None:
    fig, axes = plt.subplots(len(measurements), 1, figsize=(8.4, 11.4), sharex=True)
    axes_array = np.atleast_1d(axes)
    for index, (ax, item) in enumerate(zip(axes_array, measurements, strict=True)):
        _draw_profiles(
            ax,
            profile_r_m=item["plot_r_m"],
            truth_ne=item["truth_ne"],
            sensor_r_m=item["sensor_r_m"],
            sensor_ne=item["sensor_ne"],
            profiles=item["profiles"],
            trials=item["trials"],
            measurement_id=str(item["metadata"]["measurement_id"]),
            case_id=str(item["metadata"]["case_id"]),
            td=float(dict(item["metadata"]["fixed_cond"])["Td"]),
            z_m=float(item["metadata"]["z_m"]),
            cmap=cmap,
            norm=norm,
            include_xlabel=index == len(measurements) - 1,
        )
    fig.suptitle("TPE trial-by-trial radial electron-density profiles", y=0.995)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.94, bottom=0.06, hspace=0.34)
    colorbar_axis = fig.add_axes([0.87, 0.08, 0.025, 0.84])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    colorbar.set_label("TPE trial number")
    colorbar.set_ticks([1, 20, 40, 60, 80, 100])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_index(
    out_root: Path,
    measurements: list[dict[str, Any]],
    wall_s: float,
    cmap_name: str,
) -> None:
    seed = int(measurements[0]["metadata"]["optimizer_seed"])
    lines = [
        "# TPE trial-by-trial radial electron-density profiles",
        "",
        f"- optimizer seed: `{seed}`",
        "- TPEの各trial条件を学習済FNOへ再入力し、半径方向電子密度profileを再生成しました。",
        f"- 線色はtrial番号1から100に対応する連続`{cmap_name}`カラーマップです。",
        "- 白抜き黒縁の点は、最適化ターゲットとして使用した固定ノイズ付きセンサー値です。",
        "- 破線はノイズ付加前のheld-out COMSOL真値です。",
        "- 太線は100 trial中の最小センサーlossを持つprofileです。",
        "",
        "## Graphs",
        "",
        "- [all measurements](tpe_trial_profiles_all_measurements.png)",
    ]
    for item in measurements:
        measurement_id = str(item["metadata"]["measurement_id"])
        lines.append(f"- [{measurement_id}]({measurement_id}_tpe_trial_profiles.png)")
    lines.extend(
        [
            "",
            "## Data",
            "",
            "- [tpe_trial_profiles.csv](tpe_trial_profiles.csv)",
            "- [profile_summary.csv](profile_summary.csv)",
            "- [run_metadata.json](run_metadata.json)",
            "",
            f"Profile regeneration and plotting wall time: {wall_s:.2f} s",
        ]
    )
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-root", type=Path, default=DEFAULT_OPTIMIZATION_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 20))
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only plot regeneration requested but CUDA remains visible")
    measurement_dirs = _measurement_dirs(args.optimization_root)
    first_metadata = _read_json(measurement_dirs[0] / "run_metadata.json")
    run_root = Path(str(first_metadata["run_root"]))
    with (run_root / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        cfg = dict(yaml.safe_load(handle) or {})
    dataset = load_dataset(cfg, run_root)
    cases = {str(case["case_id"]): case for case in dataset.cases}
    args.out_root.mkdir(parents=True, exist_ok=True)
    engine = common._load_engine(run_root, cfg, Path(dataset.geometry_root), args.out_root)
    measurements: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    trial_count: int | None = None
    optimizer_seed: int | None = None
    for measurement_dir in measurement_dirs:
        metadata = _read_json(measurement_dir / "run_metadata.json")
        metadata["measurement_id"] = measurement_dir.name
        if Path(str(metadata["run_root"])) != run_root:
            raise ValueError(f"run_root differs across measurements: {measurement_dir}")
        current_seed = int(metadata["optimizer_seed"])
        if optimizer_seed is not None and current_seed != optimizer_seed:
            raise ValueError("optimizer seed differs across measurement directories")
        optimizer_seed = current_seed
        case_id = str(metadata["case_id"])
        if case_id not in cases:
            raise KeyError(f"case not found in dataset: {case_id}")
        sensor_r_m, _, sensor_ne = _load_observation(measurement_dir / "observations.csv")
        expected_r = np.asarray(metadata["r_m"], dtype=np.float64)
        if not np.array_equal(sensor_r_m, expected_r):
            raise ValueError(f"sensor radii differ from metadata: {measurement_dir}")
        plot_points = int(round((sensor_r_m[-1] - sensor_r_m[0]) / 0.0005)) + 1
        plot_r_m = np.linspace(sensor_r_m[0], sensor_r_m[-1], plot_points, dtype=np.float64)
        trials = _load_tpe_trials(measurement_dir / "trials.csv")
        if trial_count is not None and len(trials) != trial_count:
            raise ValueError("TPE trial count differs across measurements")
        trial_count = len(trials)
        truth_ne, profiles, rows, max_loss_difference = _regenerate_profiles(
            engine=engine,
            case=cases[case_id],
            metadata=metadata,
            trials=trials,
            sensor_r_m=sensor_r_m,
            plot_r_m=plot_r_m,
            sensor_ne=sensor_ne,
        )
        long_rows.extend(rows)
        best_index = int(np.argmin([float(row["loss"]) for row in trials]))
        summary_rows.append(
            {
                "measurement_id": measurement_dir.name,
                "case_id": case_id,
                "optimizer_seed": current_seed,
                "n_trials": len(trials),
                "best_trial": int(trials[best_index]["trial"]),
                "best_loss": float(trials[best_index]["loss"]),
                "max_stored_vs_recomputed_loss_abs_difference": max_loss_difference,
            }
        )
        measurements.append(
            {
                "metadata": metadata,
                "sensor_r_m": sensor_r_m,
                "plot_r_m": plot_r_m,
                "truth_ne": truth_ne,
                "sensor_ne": sensor_ne,
                "trials": trials,
                "profiles": profiles,
            }
        )
    if trial_count is None or optimizer_seed is None:
        raise RuntimeError("no measurements were processed")
    cmap = plt.get_cmap(str(args.cmap))
    norm = Normalize(vmin=1, vmax=trial_count)
    for item in measurements:
        measurement_id = str(item["metadata"]["measurement_id"])
        _plot_case(
            args.out_root / f"{measurement_id}_tpe_trial_profiles.png",
            profile_r_m=item["plot_r_m"],
            truth_ne=item["truth_ne"],
            sensor_r_m=item["sensor_r_m"],
            sensor_ne=item["sensor_ne"],
            profiles=item["profiles"],
            trials=item["trials"],
            metadata=item["metadata"],
            cmap=cmap,
            norm=norm,
        )
    _plot_combined(
        args.out_root / "tpe_trial_profiles_all_measurements.png",
        measurements,
        cmap=cmap,
        norm=norm,
    )
    _write_csv(args.out_root / "tpe_trial_profiles.csv", long_rows)
    _write_csv(args.out_root / "profile_summary.csv", summary_rows)
    wall_s = time.perf_counter() - started
    metadata = {
        "optimization_root": str(args.optimization_root),
        "surrogate_run_root": str(run_root),
        "method": METHOD,
        "optimizer_seed": optimizer_seed,
        "measurement_ids": [str(item["metadata"]["measurement_id"]) for item in measurements],
        "case_ids": [str(item["metadata"]["case_id"]) for item in measurements],
        "trials_per_measurement": trial_count,
        "profile_evaluations": len(measurements) * trial_count,
        "sensor_points_per_measurement": int(measurements[0]["sensor_r_m"].size),
        "plot_points_per_profile": int(measurements[0]["plot_r_m"].size),
        "plot_radial_step_mm": 0.5,
        "colormap": str(args.cmap),
        "line_color_mapping": "continuous TPE trial number",
        "cpu_threads": int(args.cpu_threads),
        "wall_s": wall_s,
    }
    (args.out_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_index(args.out_root, measurements, wall_s, str(args.cmap))
    print(args.out_root / "index.md")


if __name__ == "__main__":
    main()
