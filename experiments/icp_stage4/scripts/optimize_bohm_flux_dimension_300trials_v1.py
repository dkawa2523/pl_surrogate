"""Run a 300-trial regular-layout Bohm optimization with the dimension U-Net."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from experiments.icp_stage4.scripts.optimize_bohm_flux_representation_v1 import BohmMetrics
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
        default=Path("runs/icp_stage4_bohm_flux_dimension_300trials_v1"),
    )
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--screen-steps", type=int, default=60)
    parser.add_argument("--total-refine-steps", type=int, default=200)
    parser.add_argument("--refine-per-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--density-threshold", type=float, default=1.0e17)
    parser.add_argument("--threshold-penalty", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _make_trials(
    structures: dict[int, list[dict[str, Any]]],
    *,
    total: int,
    seed: int,
    bounds: Bounds,
) -> list[dict[str, Any]]:
    counts = (2, 3, 4, 5)
    if total % len(counts) != 0:
        raise ValueError("trials must be divisible by four")
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
            base = _encode(rows[anchor], count=count, bounds=bounds)
            randomized = local_index >= len(rows)
            if randomized:
                # Perturb unconstrained coordinates; decoding keeps every design
                # inside the same manufacturable bounds used by the SDF search.
                scale = (0.35, 0.7, 1.1)[local_index % 3]
                base = base + rng.normal(0.0, scale, size=base.shape).astype(np.float32)
            trials.append(
                {
                    "trial_id": trial_id,
                    "nncoil": count,
                    "seed_case_id": case_ids[anchor],
                    "layout_start": "shared_regular" if not randomized else "randomized_dimension",
                    "base": base.astype(np.float32),
                }
            )
            trial_id += 1
    return trials


def _centers(decoded: dict[str, torch.Tensor], count: int) -> torch.Tensor:
    pitch = (decoded["rrce"] - decoded["rrc"]) / float(count)
    return torch.stack(
        [
            decoded["rrc"] + float(index) * pitch + 0.5 * decoded["llcoil"]
            for index in range(count)
        ],
        dim=1,
    )


def _rank(mean_density: torch.Tensor, deviation: torch.Tensor, threshold: float) -> torch.Tensor:
    deficit = torch.relu((float(threshold) - mean_density) / float(threshold))
    return torch.where(mean_density >= float(threshold), deviation, 1000.0 + deficit)


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
    base = torch.nn.Parameter(
        torch.from_numpy(np.stack([trial["base"] for trial in trials])).to(surrogate.device)
    )
    optimizer = torch.optim.Adam([base], lr=float(args.lr))
    best_rank = np.asarray([float(trial.get("best_rank", np.inf)) for trial in trials])
    best_base = base.detach().clone()
    for local_step in range(int(steps)):
        step = int(start_step) + local_step
        optimizer.zero_grad(set_to_none=True)
        decoded = _decode(base, count=count, bounds=bounds)
        fields, _, _ = surrogate.predict(decoded, count=count)
        values = metric(fields)
        deficit = torch.relu(
            (float(args.density_threshold) - values["mean_density"]) / float(args.density_threshold)
        )
        loss_each = values["smooth_max_deviation"] + float(args.threshold_penalty) * deficit.square()
        rank = _rank(values["mean_density"], values["max_deviation"], float(args.density_threshold))
        loss_each.mean().backward()
        torch.nn.utils.clip_grad_norm_([base], 10.0)
        centers = _centers(decoded, count)
        with torch.no_grad():
            for index, rank_value in enumerate(rank.cpu().numpy()):
                if float(rank_value) < best_rank[index]:
                    best_rank[index] = float(rank_value)
                    best_base[index] = base.detach()[index]
                history.append(
                    {
                        "trial_id": int(trials[index]["trial_id"]),
                        "nncoil": count,
                        "stage": "screen" if start_step == 0 else "refine",
                        "step": step,
                        "loss": float(loss_each[index].cpu()),
                        "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                        "mean_density": float(values["mean_density"][index].cpu()),
                        "density_feasible": bool(
                            values["mean_density"][index].cpu() >= float(args.density_threshold)
                        ),
                        "spacing_std": float(torch.std(torch.diff(centers[index]), unbiased=False).cpu())
                        if count > 2
                        else 0.0,
                    }
                )
        optimizer.step()
    decoded = _decode(best_base, count=count, bounds=bounds)
    with torch.no_grad():
        fields, _, _ = surrogate.predict(decoded, count=count)
        values = metric(fields)
        centers = _centers(decoded, count)
    output: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        updated = dict(trial)
        updated.update(
            {
                "base": best_base[index].detach().cpu().numpy(),
                "best_rank": float(best_rank[index]),
                "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                "mean_bohm_flux": float(values["mean_flux"][index].cpu()),
                "mean_density": float(values["mean_density"][index].cpu()),
                "density_feasible": bool(
                    values["mean_density"][index].cpu() >= float(args.density_threshold)
                ),
                "centers": centers[index].detach().cpu().numpy(),
            }
        )
        output.append(updated)
    return output


def _public(trial: dict[str, Any], bounds: Bounds) -> dict[str, Any]:
    count = int(trial["nncoil"])
    decoded = _decode(torch.from_numpy(trial["base"])[None], count=count, bounds=bounds)
    values = {key: float(value[0]) for key, value in decoded.items()}
    z_center = 14.0 + values["zzc"] + 0.5 * values["llcoil"]
    row = {
        "trial_id": int(trial["trial_id"]),
        "nncoil": count,
        "seed_case_id": trial["seed_case_id"],
        "layout_start": trial["layout_start"],
        "bohm_max_deviation": trial["bohm_max_deviation"],
        "mean_bohm_flux": trial["mean_bohm_flux"],
        "mean_density": trial["mean_density"],
        "density_feasible": trial["density_feasible"],
        "spacing_std": 0.0,
        "coil_line_slope": 0.0,
        "height_span": 0.0,
        **values,
    }
    for slot in range(6):
        row[f"r_center_{slot + 1:02d}"] = float(trial["centers"][slot]) if slot < count else ""
        row[f"z_center_{slot + 1:02d}"] = z_center if slot < count else ""
    return row


def main() -> int:
    args = _args()
    if args.smoke:
        args.trials = 8
        args.screen_steps = 2
        args.total_refine_steps = 3
        args.refine_per_count = 1
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = yaml.safe_load((DEFAULT_RUNS["dimension"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    surrogate = RepresentationSurrogate(
        representation="dimension",
        run_dir=DEFAULT_RUNS["dimension"].resolve(),
        dataset_root=dataset_root,
        protocol="structure_holdout",
        wafer_layers=10,
    )
    metric = BohmMetrics(surrogate, radial_bins=32)
    bounds = Bounds()
    trials = _make_trials(
        _load_structure_cases(dataset_root), total=int(args.trials), seed=int(args.seed), bounds=bounds
    )
    history: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for count in (2, 3, 4, 5):
        subset = [trial for trial in trials if int(trial["nncoil"]) == count]
        for start in range(0, len(subset), int(args.batch_size)):
            screened.extend(
                _optimize_batch(
                    subset[start : start + int(args.batch_size)],
                    surrogate=surrogate,
                    metric=metric,
                    bounds=bounds,
                    args=args,
                    start_step=0,
                    steps=int(args.screen_steps),
                    history=history,
                )
            )
    refined_ids: set[int] = set()
    final_by_id = {int(trial["trial_id"]): trial for trial in screened}
    for count in (2, 3, 4, 5):
        subset = sorted(
            [trial for trial in screened if int(trial["nncoil"]) == count],
            key=lambda trial: float(trial["best_rank"]),
        )[: int(args.refine_per_count)]
        refined = _optimize_batch(
            subset,
            surrogate=surrogate,
            metric=metric,
            bounds=bounds,
            args=args,
            start_step=int(args.screen_steps),
            steps=int(args.total_refine_steps) - int(args.screen_steps),
            history=history,
        )
        for trial in refined:
            refined_ids.add(int(trial["trial_id"]))
            final_by_id[int(trial["trial_id"])] = trial
    final = sorted(
        final_by_id.values(),
        key=lambda trial: (not bool(trial["density_feasible"]), float(trial["bohm_max_deviation"])),
    )
    public = [_public(trial, bounds) for trial in final]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "trials.csv", public)
    _write_csv(out_dir / "optimization_history.csv", history)
    _write_csv(out_dir / "best_by_coil_count.csv", [
        next(row for row in public if int(row["nncoil"]) == count) for count in (2, 3, 4, 5)
    ])
    best = public[0]
    summary = {
        "representation": "dimension",
        "device": str(surrogate.device),
        "trial_count": int(args.trials),
        "coil_counts": [2, 3, 4, 5],
        "screen_steps": int(args.screen_steps),
        "refined_trial_count": len(refined_ids),
        "total_refine_steps": int(args.total_refine_steps),
        "wafer_layers": 10,
        "radial_bins": 32,
        "density_threshold": float(args.density_threshold),
        "best_dimension": best,
        "geometry_contract": "regular_equal_spacing_equal_height",
        "icp_validation_status": "not_run_no_high_fidelity_solver_in_repository",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    steps = sorted({int(row["step"]) for row in history})
    loss = [min(float(row["loss"]) for row in history if int(row["step"]) == step) for step in steps]
    axis.plot(steps, loss, linewidth=2)
    axis.set_yscale("log")
    axis.set(xlabel="gradient step", ylabel="best loss (log)", title="Dimension joint optimization (2-5 coils)")
    axis.grid(alpha=0.25)
    fig.savefig(out_dir / "optimization_loss_history.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
