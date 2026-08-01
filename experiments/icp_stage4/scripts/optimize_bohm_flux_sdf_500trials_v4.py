"""Large SDF search with unequal radial spacing and constant coil-line slope."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.icp_stage4.scripts.optimize_bohm_flux_representation_v1 import (
    BohmMetrics,
    _allocation_logits,
    _independent_centers,
)
from experiments.icp_stage4.scripts.optimize_regular_layout_representation_v1 import (
    DEFAULT_RUNS,
    RepresentationSurrogate,
    _fixed_seed_rows,
)
from experiments.icp_stage4.scripts.optimize_response_inventory_sdf_density import (
    Bounds,
    _decode,
    _encode,
    _load_structure_cases,
    _write_csv,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/icp_stage4_bohm_flux_sdf_500trials_spacing_slope_v4"),
    )
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--screen-steps", type=int, default=60)
    parser.add_argument("--total-refine-steps", type=int, default=200)
    parser.add_argument("--refine-per-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--density-threshold", type=float, default=1.0e17)
    parser.add_argument("--threshold-penalty", type=float, default=20.0)
    parser.add_argument("--trust-fraction", type=float, default=0.25)
    parser.add_argument("--max-height-offset", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _coil_heights(
    decoded: dict[str, torch.Tensor],
    centers: torch.Tensor,
    slope_logits: torch.Tensor,
    *,
    max_height_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put every coil centre on one straight line with a bounded slope."""

    ll = decoded["llcoil"]
    reference = 14.0 + decoded["zzc"] + 0.5 * ll
    lower = 14.0 + 0.5 * ll
    upper = 22.0 - 0.5 * ll
    margin = torch.minimum(reference - lower, upper - reference)
    amplitude = torch.minimum(
        torch.clamp(margin, min=0.0),
        torch.full_like(margin, float(max_height_offset)),
    ) * torch.tanh(slope_logits[:, 0])
    r_mean = torch.mean(centers, dim=1, keepdim=True)
    scale = torch.clamp(torch.max(torch.abs(centers - r_mean), dim=1, keepdim=True).values, min=1.0e-6)
    normalized_r = (centers - r_mean) / scale
    heights = reference[:, None] + amplitude[:, None] * normalized_r
    slope = amplitude / scale[:, 0]
    return heights, slope


def _sdf_variable_height(
    surrogate: RepresentationSurrogate,
    decoded: dict[str, torch.Tensor],
    centers: torch.Tensor,
    heights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count = centers.shape
    ll = decoded["llcoil"][:, None, None]
    sdf = torch.full((batch, surrogate.h, surrogate.w), 1.0e6, device=surrogate.device)
    mask = torch.zeros_like(sdf, dtype=torch.bool)
    for index in range(count):
        r_center = centers[:, index, None, None]
        z_center = heights[:, index, None, None]
        q_r = torch.abs(surrogate.r[None] - r_center) / surrogate.dr - 0.5 * ll / surrogate.dr
        q_z = torch.abs(surrogate.z[None] - z_center) / surrogate.dz - 0.5 * ll / surrogate.dz
        rectangle = torch.relu(q_r) + torch.relu(q_z) + torch.minimum(
            torch.maximum(q_r, q_z), torch.zeros_like(q_r)
        )
        sdf = torch.minimum(sdf, rectangle)
        mask |= (torch.abs(surrogate.r[None] - r_center) <= 0.5 * ll) & (
            torch.abs(surrogate.z[None] - z_center) <= 0.5 * ll
        )
    return mask.float(), sdf + 1.0


def _predict(
    surrogate: RepresentationSurrogate,
    decoded: dict[str, torch.Tensor],
    allocation: torch.Tensor,
    slope_logits: torch.Tensor,
    *,
    count: int,
    args: argparse.Namespace,
    bounds: Bounds,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = _independent_centers(
        surrogate, decoded, allocation, count=count,
        trust_fraction=float(args.trust_fraction), bounds=bounds,
    )
    heights, slope = _coil_heights(
        decoded, centers, slope_logits, max_height_offset=float(args.max_height_offset)
    )
    mask, sdf = _sdf_variable_height(surrogate, decoded, centers, heights)
    fields, _, _ = surrogate.predict_from_sdf(decoded, count=count, mask=mask, sdf=sdf)
    return fields, mask, centers, heights, slope


def _rank(mean_density: torch.Tensor, max_deviation: torch.Tensor, threshold: float) -> torch.Tensor:
    deficit = torch.relu((float(threshold) - mean_density) / float(threshold))
    return torch.where(mean_density >= float(threshold), max_deviation, 1000.0 + deficit)


def _make_trials(
    structures: dict[int, list[dict[str, Any]]],
    *,
    total: int,
    seed: int,
    bounds: Bounds,
) -> list[dict[str, Any]]:
    counts = (2, 3, 4, 5)
    if total % len(counts) != 0:
        raise ValueError("trials must be divisible by four for coil counts 2..5")
    per_count = total // len(counts)
    trials: list[dict[str, Any]] = []
    trial_id = 0
    for count in counts:
        rows, case_ids = _fixed_seed_rows(
            structures[count], process_center=(1753.353515625, 0.017528499476611614)
        )
        rng = np.random.default_rng(int(seed) + 1000 * count)
        for local_index in range(per_count):
            anchor = local_index % len(rows)
            row = dict(rows[anchor])
            regular = local_index < len(rows)
            allocation = _allocation_logits(row, count, bounds)
            if not regular:
                scale = (0.75, 1.5, 2.5, 4.0)[local_index % 4]
                noise = rng.normal(0.0, scale, size=count + 1).astype(np.float32)
                allocation = allocation + noise - np.mean(noise, dtype=np.float32)
            slope = 0.0 if regular else float(rng.normal(0.0, 1.5))
            trials.append(
                {
                    "trial_id": trial_id,
                    "nncoil": count,
                    "seed_case_id": case_ids[anchor],
                    "layout_start": "shared_regular" if regular else "asymmetric_sloped",
                    "base": _encode(row, count=count, bounds=bounds),
                    "allocation": allocation,
                    "slope": np.asarray([slope], dtype=np.float32),
                }
            )
            trial_id += 1
    return trials


def _optimize_batch(
    trials: list[dict[str, Any]],
    *,
    surrogate: RepresentationSurrogate,
    metric: BohmMetrics,
    bounds: Bounds,
    args: argparse.Namespace,
    start_step: int,
    steps: int,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    count = int(trials[0]["nncoil"])
    if any(int(trial["nncoil"]) != count for trial in trials):
        raise ValueError("one optimization batch must have one coil count")
    base = torch.nn.Parameter(torch.from_numpy(np.stack([trial["base"] for trial in trials])).to(surrogate.device))
    allocation = torch.nn.Parameter(
        torch.from_numpy(np.stack([trial["allocation"] for trial in trials])).to(surrogate.device)
    )
    slope_logits = torch.nn.Parameter(
        torch.from_numpy(np.stack([trial["slope"] for trial in trials])).to(surrogate.device)
    )
    parameters = [base, allocation, slope_logits]
    optimizer = torch.optim.Adam(parameters, lr=float(args.lr))
    best_rank = np.asarray([float(trial.get("best_rank", np.inf)) for trial in trials])
    best_base = base.detach().clone()
    best_allocation = allocation.detach().clone()
    best_slope = slope_logits.detach().clone()
    for local_step in range(int(steps)):
        step = int(start_step) + local_step
        optimizer.zero_grad(set_to_none=True)
        decoded = _decode(base, count=count, bounds=bounds)
        fields, _, centers, heights, slope = _predict(
            surrogate, decoded, allocation, slope_logits, count=count, args=args, bounds=bounds
        )
        values = metric(fields)
        deficit = torch.relu(
            (float(args.density_threshold) - values["mean_density"]) / float(args.density_threshold)
        )
        loss_each = values["smooth_max_deviation"] + float(args.threshold_penalty) * deficit.square()
        rank = _rank(values["mean_density"], values["max_deviation"], float(args.density_threshold))
        loss_each.mean().backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        with torch.no_grad():
            for index, rank_value in enumerate(rank.cpu().numpy()):
                if float(rank_value) < best_rank[index]:
                    best_rank[index] = float(rank_value)
                    best_base[index] = base.detach()[index]
                    best_allocation[index] = allocation.detach()[index]
                    best_slope[index] = slope_logits.detach()[index]
                history.append(
                    {
                        "trial_id": int(trials[index]["trial_id"]),
                        "nncoil": count,
                        "stage": "screen" if start_step == 0 else "refine",
                        "step": step,
                        "loss": float(loss_each[index].cpu()),
                        "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                        "mean_density": float(values["mean_density"][index].cpu()),
                        "density_feasible": bool(values["mean_density"][index].cpu() >= float(args.density_threshold)),
                        "spacing_std": float(torch.std(torch.diff(centers[index]), unbiased=False).cpu()) if count > 2 else 0.0,
                        "coil_line_slope": float(slope[index].cpu()),
                        "height_span": float((torch.max(heights[index]) - torch.min(heights[index])).cpu()),
                    }
                )
        optimizer.step()
    output: list[dict[str, Any]] = []
    decoded = _decode(best_base, count=count, bounds=bounds)
    with torch.no_grad():
        fields, mask, centers, heights, slope = _predict(
            surrogate, decoded, best_allocation, best_slope, count=count, args=args, bounds=bounds
        )
        values = metric(fields)
    for index, trial in enumerate(trials):
        updated = dict(trial)
        updated.update(
            {
                "base": best_base[index].cpu().numpy(),
                "allocation": best_allocation[index].cpu().numpy(),
                "slope": best_slope[index].cpu().numpy(),
                "best_rank": float(best_rank[index]),
                "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                "mean_bohm_flux": float(values["mean_flux"][index].cpu()),
                "mean_density": float(values["mean_density"][index].cpu()),
                "density_feasible": bool(values["mean_density"][index].cpu() >= float(args.density_threshold)),
                "spacing_std": float(torch.std(torch.diff(centers[index]), unbiased=False).cpu()) if count > 2 else 0.0,
                "coil_line_slope": float(slope[index].cpu()),
                "height_span": float((torch.max(heights[index]) - torch.min(heights[index])).cpu()),
                "centers": centers[index].cpu().numpy(),
                "heights": heights[index].cpu().numpy(),
                "bohm_field": values["bohm_field"][index].cpu().numpy(),
                "bin_flux": values["bin_flux"][index].cpu().numpy(),
                "mask": mask[index].cpu().numpy(),
            }
        )
        output.append(updated)
    return output


def _public_row(trial: dict[str, Any]) -> dict[str, Any]:
    count = int(trial["nncoil"])
    base = np.asarray(trial["base"], dtype=np.float32)
    decoded = _decode(torch.from_numpy(base)[None], count=count, bounds=Bounds())
    row: dict[str, Any] = {
        "trial_id": int(trial["trial_id"]),
        "nncoil": count,
        "seed_case_id": trial["seed_case_id"],
        "layout_start": trial["layout_start"],
        "bohm_max_deviation": trial["bohm_max_deviation"],
        "mean_bohm_flux": trial["mean_bohm_flux"],
        "mean_density": trial["mean_density"],
        "density_feasible": trial["density_feasible"],
        "spacing_std": trial["spacing_std"],
        "coil_line_slope": trial["coil_line_slope"],
        "height_span": trial["height_span"],
    }
    for name, value in decoded.items():
        row[name] = float(value[0])
    for slot in range(6):
        row[f"r_center_{slot + 1:02d}"] = float(trial["centers"][slot]) if slot < count else ""
        row[f"z_center_{slot + 1:02d}"] = float(trial["heights"][slot]) if slot < count else ""
    return row


def _plot_history(history: list[dict[str, Any]], out_dir: Path, threshold: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    steps = sorted({int(row["step"]) for row in history})
    best_loss, best_deviation, density = [], [], []
    for step in steps:
        rows = [row for row in history if int(row["step"]) == step]
        best_loss.append(min(float(row["loss"]) for row in rows))
        feasible = [row for row in rows if bool(row["density_feasible"])]
        chosen = min(feasible, key=lambda row: float(row["bohm_max_deviation"])) if feasible else max(
            rows, key=lambda row: float(row["mean_density"])
        )
        best_deviation.append(float(chosen["bohm_max_deviation"]) if feasible else np.nan)
        density.append(float(chosen["mean_density"]) / 1.0e17)
    axes[0].plot(steps, best_loss)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="gradient step", ylabel="best loss (log)", title="Optimization loss")
    axes[1].plot(steps, best_deviation)
    axes[1].set(xlabel="gradient step", ylabel="best feasible maximum deviation", title="Bohm uniformity")
    axes[2].plot(steps, density)
    axes[2].axhline(threshold / 1.0e17, color="black", linestyle="--", label="threshold")
    axes[2].set(xlabel="gradient step", ylabel="mean density [1e17 m^-3]", title="Density feasibility")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[2].legend()
    fig.savefig(out_dir / "optimization_loss_history.png", dpi=180)
    plt.close(fig)


def _plot_loss_history_by_coil_count(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Overlay the best loss trajectory for each discrete coil count."""
    fig, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for count in (2, 3, 4, 5):
        count_rows = [row for row in history if int(row["nncoil"]) == count]
        steps = sorted({int(row["step"]) for row in count_rows})
        best_loss = [
            min(float(row["loss"]) for row in count_rows if int(row["step"]) == step)
            for step in steps
        ]
        axis.plot(steps, best_loss, linewidth=1.8, label=f"{count} coils")
    refine_start = min(
        int(row["step"]) for row in history if str(row["stage"]) == "refine"
    )
    axis.axvline(refine_start, color="black", linestyle="--", linewidth=1.0, label="top-5 refinement")
    axis.set_yscale("log")
    axis.set(
        xlabel="gradient step",
        ylabel="best loss (log)",
        title="Optimization loss by coil count",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(out_dir / "optimization_loss_history_by_coil_count.png", dpi=180)
    plt.close(fig)


def _plot_joint_loss_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot one envelope trajectory with coil count treated as a joint variable."""
    steps = sorted({int(row["step"]) for row in history})
    best_loss = [
        min(float(row["loss"]) for row in history if int(row["step"]) == step)
        for step in steps
    ]
    refine_start = min(
        int(row["step"]) for row in history if str(row["stage"]) == "refine"
    )
    fig, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    axis.plot(steps, best_loss, color="tab:blue", linewidth=2.0)
    axis.axvline(refine_start, color="black", linestyle="--", linewidth=1.0, label="top-5/count refinement")
    axis.set_yscale("log")
    axis.set(
        xlabel="gradient step",
        ylabel="best loss across all coil counts (log)",
        title="Joint optimization loss (2-5 coils)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(out_dir / "optimization_loss_history_all_coil_counts.png", dpi=180)
    plt.close(fig)


def _plot_best(surrogate: RepresentationSurrogate, best: dict[str, Any], out_dir: Path) -> None:
    extent = [surrogate.r_np.min(), surrogate.r_np.max(), surrogate.z_np.min(), surrogate.z_np.max()]
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    field = np.where(surrogate.plasma_np, best["bohm_field"], np.nan)
    image = axis.imshow(field, origin="lower", extent=extent, aspect="auto", vmin=0)
    fig.colorbar(image, ax=axis, label="Bohm ion flux [m^-2 s^-1]")
    axis.set(xlabel="r", ylabel="z", title="Optimized Bohm flux")
    fig.savefig(out_dir / "bohm_flux_spatial_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.imshow(best["mask"], origin="lower", extent=extent, aspect="auto", cmap="gray_r")
    axis.scatter(best["centers"], best["heights"], color="tab:red", marker="x", s=55, label="coil centres")
    axis.plot(best["centers"], best["heights"], color="tab:red", linewidth=1, label="constant-slope line")
    axis.set(xlabel="r", ylabel="z", title="Optimized unequal-spacing sloped coil layout")
    axis.legend()
    fig.savefig(out_dir / "optimized_coil_layout.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    profile = best["bin_flux"] / np.mean(best["bin_flux"])
    axis.plot(np.arange(profile.size), profile)
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set(xlabel="equal-area radial bin", ylabel="Bohm flux / mean", title="Optimized wafer Bohm-flux profile")
    axis.grid(alpha=0.25)
    fig.savefig(out_dir / "wafer_bohm_flux_profile.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _args()
    if args.smoke:
        args.trials = 8
        args.screen_steps = 2
        args.total_refine_steps = 3
        args.refine_per_count = 1
        args.batch_size = 2
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((DEFAULT_RUNS["union_sdf"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    surrogate = RepresentationSurrogate(
        representation="union_sdf", run_dir=DEFAULT_RUNS["union_sdf"].resolve(),
        dataset_root=dataset_root, protocol="structure_holdout", wafer_layers=10,
    )
    metric = BohmMetrics(surrogate, 32)
    bounds = Bounds()
    trials = _make_trials(
        _load_structure_cases(dataset_root), total=int(args.trials), seed=int(args.seed), bounds=bounds
    )
    history: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for count in (2, 3, 4, 5):
        group = [trial for trial in trials if int(trial["nncoil"]) == count]
        for start in range(0, len(group), int(args.batch_size)):
            screened.extend(
                _optimize_batch(
                    group[start : start + int(args.batch_size)], surrogate=surrogate, metric=metric,
                    bounds=bounds, args=args, start_step=0, steps=int(args.screen_steps), history=history,
                )
            )
    refined_ids: set[int] = set()
    refined: list[dict[str, Any]] = []
    refine_steps = int(args.total_refine_steps) - int(args.screen_steps)
    for count in (2, 3, 4, 5):
        group = sorted(
            (trial for trial in screened if int(trial["nncoil"]) == count),
            key=lambda trial: float(trial["best_rank"]),
        )[: int(args.refine_per_count)]
        refined_ids.update(int(trial["trial_id"]) for trial in group)
        refined.extend(
            _optimize_batch(
                group, surrogate=surrogate, metric=metric, bounds=bounds, args=args,
                start_step=int(args.screen_steps), steps=refine_steps, history=history,
            )
        )
    final_by_id = {int(trial["trial_id"]): trial for trial in screened}
    final_by_id.update({int(trial["trial_id"]): trial for trial in refined})
    final = sorted(
        final_by_id.values(),
        key=lambda trial: (not bool(trial["density_feasible"]), float(trial["bohm_max_deviation"])),
    )
    best = final[0]
    public = [_public_row(trial) for trial in final]
    _write_csv(out_dir / "trials.csv", public)
    _write_csv(out_dir / "optimization_history.csv", history)
    _write_csv(out_dir / "high_fidelity_candidates_top5.csv", public[:5])
    best_by_count = [next(row for row in public if int(row["nncoil"]) == count) for count in (2, 3, 4, 5)]
    _write_csv(out_dir / "best_by_coil_count.csv", best_by_count)
    _plot_history(history, out_dir, float(args.density_threshold))
    _plot_loss_history_by_coil_count(history, out_dir)
    _plot_joint_loss_history(history, out_dir)
    _plot_best(surrogate, best, out_dir)

    dimension_path = Path(
        "runs/icp_stage4_bohm_flux_representation_validation_wafer10_asymmetric_v2/dimension/best_by_coil_count.csv"
    )
    with dimension_path.open(encoding="utf-8", newline="") as handle:
        dimension_rows = [row for row in csv.DictReader(handle) if 2 <= int(float(row["nncoil"])) <= 5]
    dimension_best = min(dimension_rows, key=lambda row: float(row["bohm_max_deviation"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(
        [int(float(row["nncoil"])) for row in dimension_rows],
        [float(row["bohm_max_deviation"]) for row in dimension_rows], marker="o", label="dimension fixed baseline",
    )
    axes[0].plot(
        [int(row["nncoil"]) for row in best_by_count],
        [float(row["bohm_max_deviation"]) for row in best_by_count], marker="o", label="SDF 500 trials",
    )
    axes[0].set(xlabel="coil count", ylabel="best feasible maximum deviation", title="Absolute Bohm uniformity")
    feasible = [row for row in public if bool(row["density_feasible"])]
    scatter = axes[1].scatter(
        [float(row["spacing_std"]) for row in feasible],
        [float(row["height_span"]) for row in feasible],
        c=[float(row["bohm_max_deviation"]) for row in feasible], cmap="viridis_r", s=24, alpha=0.7,
    )
    fig.colorbar(scatter, ax=axes[1], label="maximum Bohm deviation")
    axes[1].set(xlabel="spacing standard deviation", ylabel="coil height span", title="SDF geometry search coverage")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend() if axis is axes[0] else None
    fig.savefig(out_dir / "sdf_500trial_evaluation.png", dpi=180)
    plt.close(fig)

    relative = 100.0 * (
        float(best["bohm_max_deviation"]) - float(dimension_best["bohm_max_deviation"])
    ) / float(dimension_best["bohm_max_deviation"])
    dimension_by_count = {int(float(row["nncoil"])): row for row in dimension_rows}
    comparison_lines = [
        "| coils | SDF | dimension baseline | SDF improvement |",
        "|---:|---:|---:|---:|",
    ]
    for row in best_by_count:
        count = int(row["nncoil"])
        sdf_value = float(row["bohm_max_deviation"])
        dimension_value = float(dimension_by_count[count]["bohm_max_deviation"])
        improvement = 100.0 * (dimension_value - sdf_value) / dimension_value
        comparison_lines.append(
            f"| {count} | {sdf_value:.6f} | {dimension_value:.6f} | {improvement:+.2f}% |"
        )
    summary = {
        "device": str(surrogate.device),
        "trial_count": int(args.trials),
        "coil_counts": [2, 3, 4, 5],
        "screen_steps": int(args.screen_steps),
        "refined_trial_count": len(refined_ids),
        "total_refine_steps": int(args.total_refine_steps),
        "wafer_layers": 10,
        "density_threshold": float(args.density_threshold),
        "best_sdf": _public_row(best),
        "best_dimension_fixed_baseline": dimension_best,
        "sdf_relative_deviation_vs_dimension_percent": relative,
        "icp_validation_status": "not_run_no_high_fidelity_solver_in_repository",
        "interpretation_limit": "variable-height layouts are outside the equal-height training geometry distribution",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# SDF 500-trial unequal-spacing and constant-slope optimization", "",
        f"- trials: `{int(args.trials)}` (`{int(args.trials)//4}` per coil count)",
        f"- best SDF coil count: `{int(best['nncoil'])}`",
        f"- best SDF maximum deviation: `{float(best['bohm_max_deviation']):.6f}`",
        f"- best SDF mean density: `{float(best['mean_density']):.4e} m^-3`",
        f"- best SDF spacing standard deviation: `{float(best['spacing_std']):.6f}`",
        f"- best SDF height span: `{float(best['height_span']):.6f}`",
        f"- best SDF constant line slope: `{float(best['coil_line_slope']):.6f}`",
        f"- fixed dimension baseline maximum deviation (counts 2..5): `{float(dimension_best['bohm_max_deviation']):.6f}`",
        f"- SDF relative difference from dimension: `{relative:+.2f}%`", "",
        "The height constraint is `z_i = z_ref + slope * (r_i - mean(r))`; every coil centre therefore lies on one straight line.", "",
        "## Coil-count comparison", "", *comparison_lines, "",
        "## Figures", "", "- [Optimization history](optimization_loss_history.png)",
        "- [Joint loss history for all coil counts](optimization_loss_history_all_coil_counts.png)",
        "- [Loss history by coil count](optimization_loss_history_by_coil_count.png)",
        "- [Final Bohm spatial distribution](bohm_flux_spatial_distribution.png)",
        "- [Final unequal-spacing sloped layout](optimized_coil_layout.png)",
        "- [Final wafer profile](wafer_bohm_flux_profile.png)",
        "- [500-trial evaluation](sdf_500trial_evaluation.png)", "",
        "## Data", "", "- [All 500 trials](trials.csv)",
        "- [Best by coil count](best_by_coil_count.csv)",
        "- [Top five for ICP validation](high_fidelity_candidates_top5.csv)",
        "- [Summary](summary.json)", "",
        "## Interpretation limit", "",
        "The surrogate was trained on regular equal-height layouts. Variable-height layouts are therefore geometry extrapolation, even though the SDF input can represent them. The numerical advantage is a surrogate-screening result, not yet evidence of ICP-level improvement.", "",
        "ICP validation error and regret remain pending because no callable high-fidelity ICP solver is present in this repository.",
    ]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
