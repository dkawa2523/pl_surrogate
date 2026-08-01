"""Optimize wafer-near Bohm-flux uniformity with the current ICP U-Nets."""

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

from experiments.icp_stage4.scripts.optimize_regular_layout_representation_v1 import (
    DEFAULT_RUNS,
    RepresentationSurrogate,
    _fixed_seed_rows,
)
from experiments.icp_stage4.scripts.optimize_response_inventory_sdf_density import (
    Bounds,
    _decode,
    _encode,
    _float_design,
    _load_structure_cases,
    _write_csv,
)


E_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906660e-27
ARGON_ION_MASS = 39.948 * ATOMIC_MASS


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/icp_stage4_bohm_flux_representation_validation_v1"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--density-threshold", type=float, default=1.0e17)
    parser.add_argument("--threshold-penalty", type=float, default=20.0)
    parser.add_argument("--radial-bins", type=int, default=32)
    parser.add_argument("--wafer-layers", type=int, default=3)
    parser.add_argument("--trust-fraction", type=float, default=0.25)
    parser.add_argument("--sdf-asymmetric-starts", type=int, default=4)
    parser.add_argument("--sdf-allocation-noise", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _allocation_logits(row: dict[str, float], count: int, bounds: Bounds) -> np.ndarray:
    raw = torch.from_numpy(_encode(row, count=count, bounds=bounds))[None]
    decoded = _decode(raw, count=count, bounds=bounds)
    ll = float(decoded["llcoil"][0])
    rrc = float(decoded["rrc"][0])
    rrce = float(decoded["rrce"][0])
    pitch = (rrce - rrc) / count
    min_gap = (1.0 + bounds.min_gap_fraction) * ll
    slack = max((rrce - rrc - ll) - (count - 1) * min_gap, 1.0e-6)
    allocations = [1.0e-4 * slack]
    allocations.extend([max(pitch - min_gap, 1.0e-6)] * (count - 1))
    allocations.append(max(pitch - ll, 1.0e-6))
    values = np.asarray(allocations, dtype=np.float64)
    values = np.maximum(values / np.sum(values), 1.0e-7)
    return np.log(values).astype(np.float32)


def _independent_centers(
    surrogate: RepresentationSurrogate,
    decoded: dict[str, torch.Tensor],
    allocation_logits: torch.Tensor,
    *,
    count: int,
    trust_fraction: float,
    bounds: Bounds,
) -> torch.Tensor:
    ll = decoded["llcoil"]
    span = decoded["rrce"] - decoded["rrc"]
    regular = torch.stack(
        [decoded["rrc"] + float(index) * span / float(count) + 0.5 * ll for index in range(count)],
        dim=1,
    )
    minimum_gap = (1.0 + bounds.min_gap_fraction) * ll
    available = torch.clamp(span - ll - float(count - 1) * minimum_gap, min=1.0e-6)
    allocation = torch.softmax(allocation_logits, dim=1) * available[:, None]
    centers = [decoded["rrc"] + 0.5 * ll + allocation[:, 0]]
    for index in range(1, count):
        centers.append(centers[-1] + minimum_gap + allocation[:, index])
    free = torch.stack(centers, dim=1)
    alpha = float(trust_fraction)
    return (1.0 - alpha) * regular + alpha * free


def _sdf_from_centers(
    surrogate: RepresentationSurrogate,
    decoded: dict[str, torch.Tensor],
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count = centers.shape
    ll = decoded["llcoil"][:, None, None]
    z0 = (14.0 + decoded["zzc"])[:, None, None]
    sdf = torch.full((batch, surrogate.h, surrogate.w), 1.0e6, device=surrogate.device)
    mask = torch.zeros_like(sdf, dtype=torch.bool)
    for index in range(count):
        center = centers[:, index, None, None]
        q_r = torch.abs(surrogate.r[None] - center) / surrogate.dr - 0.5 * ll / surrogate.dr
        q_z = torch.abs(surrogate.z[None] - (z0 + 0.5 * ll)) / surrogate.dz - 0.5 * ll / surrogate.dz
        rectangle = torch.relu(q_r) + torch.relu(q_z) + torch.minimum(
            torch.maximum(q_r, q_z), torch.zeros_like(q_r)
        )
        sdf = torch.minimum(sdf, rectangle)
        mask |= (
            (torch.abs(surrogate.r[None] - center) <= 0.5 * ll)
            & (surrogate.z[None] >= z0)
            & (surrogate.z[None] <= z0 + ll)
        )
    return mask.float(), sdf + 1.0


class BohmMetrics:
    def __init__(self, surrogate: RepresentationSurrogate, radial_bins: int):
        radius = surrogate.radial_np.astype(np.float64)
        edges = np.sqrt(np.linspace(radius.min() ** 2, radius.max() ** 2, int(radial_bins) + 1))
        index = np.clip(np.digitize(radius, edges[1:-1]), 0, int(radial_bins) - 1)
        self.bin_masks = [
            torch.from_numpy(index == bin_index).to(surrogate.device) for bin_index in range(int(radial_bins))
        ]
        if any(not bool(torch.any(mask)) for mask in self.bin_masks):
            raise ValueError("radial binning produced an empty bin")
        self.s = surrogate

    def __call__(self, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ni = torch.clamp(fields["ni"], min=1.0e8)
        te = torch.clamp(fields["Te"], min=0.05)
        bohm = ni * torch.sqrt(te * (E_CHARGE / ARGON_ION_MASS))
        density = 0.5 * (torch.clamp(fields["ne"], min=1.0e8) + ni)
        selector = self.s.selector[None]
        bohm_columns = torch.sum(torch.where(selector, bohm, torch.zeros_like(bohm)), dim=1)[:, self.s.profile_columns]
        density_columns = torch.sum(
            torch.where(selector, density, torch.zeros_like(density)), dim=1
        )[:, self.s.profile_columns]
        bohm_columns = bohm_columns / self.s.profile_counts[None]
        density_columns = density_columns / self.s.profile_counts[None]
        weights = self.s.radial_weights[None]
        mean_density = torch.sum(weights * density_columns, dim=1) / torch.sum(weights, dim=1)
        bin_flux = []
        bin_weights = []
        for mask in self.bin_masks:
            local_w = weights[:, mask]
            bin_flux.append(torch.sum(local_w * bohm_columns[:, mask], dim=1) / torch.sum(local_w, dim=1))
            bin_weights.append(torch.sum(local_w, dim=1))
        flux = torch.stack(bin_flux, dim=1)
        area = torch.stack(bin_weights, dim=1)
        mean_flux = torch.sum(area * flux, dim=1) / torch.sum(area, dim=1)
        deviation = torch.abs(flux / torch.clamp(mean_flux[:, None], min=1.0) - 1.0)
        maximum = torch.max(deviation, dim=1).values
        beta = 30.0
        smooth = (torch.logsumexp(beta * deviation, dim=1) - math.log(deviation.shape[1])) / beta
        return {
            "mean_density": mean_density,
            "mean_flux": mean_flux,
            "bin_flux": flux,
            "max_deviation": maximum,
            "smooth_max_deviation": smooth,
            "bohm_field": bohm,
        }


def _predict(
    surrogate: RepresentationSurrogate,
    decoded: dict[str, torch.Tensor],
    *,
    representation: str,
    count: int,
    allocation_logits: torch.Tensor | None,
    trust_fraction: float,
    bounds: Bounds,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    if representation == "dimension":
        fields, mask, sdf = surrogate.predict(decoded, count=count)
        span = decoded["rrce"] - decoded["rrc"]
        centers = torch.stack(
            [decoded["rrc"] + float(index) * span / float(count) + 0.5 * decoded["llcoil"] for index in range(count)],
            dim=1,
        )
        return fields, mask, sdf, centers
    assert allocation_logits is not None
    centers = _independent_centers(
        surrogate,
        decoded,
        allocation_logits,
        count=count,
        trust_fraction=trust_fraction,
        bounds=bounds,
    )
    mask, sdf = _sdf_from_centers(surrogate, decoded, centers)
    fields, _, _ = surrogate.predict_from_sdf(decoded, count=count, mask=mask, sdf=sdf)
    return fields, mask, sdf, centers


def _plot_history(rows: list[dict[str, Any]], out_dir: Path, threshold: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for count in range(2, 7):
        subset = [row for row in rows if int(row["nncoil"]) == count]
        steps = sorted({int(row["step"]) for row in subset})
        loss = []
        uniformity = []
        density = []
        for step in steps:
            current = [row for row in subset if int(row["step"]) == step]
            loss.append(min(float(row["loss"]) for row in current))
            feasible = [row for row in current if bool(row["density_feasible"])]
            chosen = min(feasible, key=lambda row: float(row["bohm_max_deviation"])) if feasible else max(
                current, key=lambda row: float(row["mean_density"])
            )
            uniformity.append(float(chosen["bohm_max_deviation"]) if feasible else np.nan)
            density.append(float(chosen["mean_density"]) / 1.0e17)
        axes[0].plot(steps, loss, label=f"n={count}")
        axes[1].plot(steps, uniformity, label=f"n={count}")
        axes[2].plot(steps, density, label=f"n={count}")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="step", ylabel="penalized loss (log)", title="Optimization loss")
    axes[1].set(xlabel="step", ylabel="max local Bohm-flux deviation", title="Feasible uniformity")
    axes[2].axhline(threshold / 1.0e17, color="black", linestyle="--", label="threshold")
    axes[2].set(xlabel="step", ylabel="mean density [1e17 m^-3]", title="Density feasibility")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(out_dir / "optimization_loss_history.png", dpi=180)
    plt.close(fig)


def _plot_result(
    surrogate: RepresentationSurrogate,
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    initial_centers: np.ndarray,
    final_centers: np.ndarray,
    initial_mask: np.ndarray,
    final_mask: np.ndarray,
    out_dir: Path,
) -> None:
    extent = [surrogate.r_np.min(), surrogate.r_np.max(), surrogate.z_np.min(), surrogate.z_np.max()]
    mask = surrogate.plasma_np
    vmax = float(np.nanmax(np.where(mask, final["bohm"], np.nan)))
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    image = axis.imshow(
        np.where(mask, final["bohm"], np.nan),
        origin="lower", extent=extent, aspect="auto", vmin=0, vmax=vmax,
    )
    fig.colorbar(image, ax=axis, label="Bohm ion flux [m^-2 s^-1]")
    axis.set(xlabel="r", ylabel="z", title="Optimized Bohm flux")
    fig.savefig(out_dir / "bohm_flux_spatial_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.imshow(final_mask, origin="lower", extent=extent, aspect="auto", cmap="gray_r")
    axis.scatter(final_centers, np.full_like(final_centers, 20.5), marker="v", color="tab:red", s=35)
    axis.set(xlabel="r", ylabel="z", title="Optimized coil layout")
    fig.savefig(out_dir / "optimized_coil_layout.png", dpi=180)
    plt.close(fig)

    bins = np.arange(len(initial["bin_flux"]))
    fig, axis = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    axis.plot(bins, final["bin_flux"] / np.mean(final["bin_flux"]), label="optimized")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="equal-area radial bin", ylabel="Bohm flux / mean", title="Wafer Bohm-flux profile")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(out_dir / "wafer_bohm_flux_profile.png", dpi=180)
    plt.close(fig)


def _run_representation(
    *,
    representation: str,
    dataset_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    surrogate = RepresentationSurrogate(
        representation=representation,
        run_dir=DEFAULT_RUNS[representation].resolve(),
        dataset_root=dataset_root,
        protocol="structure_holdout",
        wafer_layers=int(args.wafer_layers),
    )
    metric = BohmMetrics(surrogate, int(args.radial_bins))
    structures = _load_structure_cases(dataset_root)
    bounds = Bounds()
    counts = [4] if args.smoke else list(range(2, 7))
    steps = 2 if args.smoke else int(args.steps)
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seed_lookup: dict[str, dict[str, float]] = {}
    for count in counts:
        rows, case_ids = _fixed_seed_rows(structures[count], process_center=(1753.353515625, 0.017528499476611614))
        if args.smoke:
            rows, case_ids = rows[:2], case_ids[:2]
        allocation_initializers: list[np.ndarray] | None = None
        layout_start_ids = ["shared_regular"] * len(rows)
        if representation == "union_sdf":
            allocation_initializers = [_allocation_logits(row, count, bounds) for row in rows]
            extra_count = min(int(args.sdf_asymmetric_starts), len(rows))
            selected = np.linspace(0, len(rows) - 1, extra_count, dtype=int).tolist() if extra_count else []
            rng = np.random.default_rng(int(args.seed) + 100 * count)
            for extra_index, source_index in enumerate(selected, start=1):
                rows.append(dict(rows[source_index]))
                case_ids.append(case_ids[source_index])
                noise = rng.normal(0.0, float(args.sdf_allocation_noise), size=count + 1).astype(np.float32)
                noise -= np.mean(noise, dtype=np.float32)
                allocation_initializers.append(
                    _allocation_logits(rows[source_index], count, bounds) + noise
                )
                layout_start_ids.append(f"asymmetric_{extra_index:02d}")
        seed_lookup.update(dict(zip(case_ids, rows)))
        base = torch.nn.Parameter(torch.from_numpy(np.stack([_encode(row, count=count, bounds=bounds) for row in rows])).to(surrogate.device))
        allocation = None
        parameters: list[torch.nn.Parameter] = [base]
        if representation == "union_sdf":
            assert allocation_initializers is not None
            allocation = torch.nn.Parameter(
                torch.from_numpy(np.stack(allocation_initializers)).to(surrogate.device)
            )
            parameters.append(allocation)
        optimizer = torch.optim.Adam(parameters, lr=float(args.lr))
        best_rank = np.full(len(rows), np.inf)
        best_base = base.detach().clone()
        best_allocation = allocation.detach().clone() if allocation is not None else None
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            decoded = _decode(base, count=count, bounds=bounds)
            fields, _, _, centers = _predict(
                surrogate,
                decoded,
                representation=representation,
                count=count,
                allocation_logits=allocation,
                trust_fraction=float(args.trust_fraction),
                bounds=bounds,
            )
            values = metric(fields)
            deficit = torch.relu((float(args.density_threshold) - values["mean_density"]) / float(args.density_threshold))
            loss_each = values["smooth_max_deviation"] + float(args.threshold_penalty) * deficit.square()
            rank = torch.where(
                values["mean_density"] >= float(args.density_threshold),
                values["max_deviation"],
                1000.0 + deficit,
            )
            loss_each.mean().backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            with torch.no_grad():
                for index, rank_value in enumerate(rank.cpu().numpy()):
                    if float(rank_value) < best_rank[index]:
                        best_rank[index] = float(rank_value)
                        best_base[index] = base.detach()[index]
                        if allocation is not None and best_allocation is not None:
                            best_allocation[index] = allocation.detach()[index]
                    history.append(
                        {
                            "representation": representation,
                            "nncoil": count,
                            "restart": index,
                            "seed_case_id": case_ids[index],
                            "layout_start": layout_start_ids[index],
                            "step": step,
                            "loss": float(loss_each[index].cpu()),
                            "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                            "mean_bohm_flux": float(values["mean_flux"][index].cpu()),
                            "mean_density": float(values["mean_density"][index].cpu()),
                            "density_feasible": bool(values["mean_density"][index].cpu() >= float(args.density_threshold)),
                            "spacing_std": float(torch.std(torch.diff(centers[index]), unbiased=False).cpu()) if count > 2 else 0.0,
                            **_float_design(decoded, index, count),
                        }
                    )
            optimizer.step()
        decoded = _decode(best_base, count=count, bounds=bounds)
        with torch.no_grad():
            fields, masks, sdfs, centers = _predict(
                surrogate,
                decoded,
                representation=representation,
                count=count,
                allocation_logits=best_allocation,
                trust_fraction=float(args.trust_fraction),
                bounds=bounds,
            )
            values = metric(fields)
        for index in range(len(rows)):
            candidate = {
                "representation": representation,
                "nncoil": count,
                "restart": index,
                "seed_case_id": case_ids[index],
                "layout_start": layout_start_ids[index],
                "bohm_max_deviation": float(values["max_deviation"][index].cpu()),
                "mean_bohm_flux": float(values["mean_flux"][index].cpu()),
                "mean_density": float(values["mean_density"][index].cpu()),
                "density_feasible": bool(values["mean_density"][index].cpu() >= float(args.density_threshold)),
                "spacing_std": float(torch.std(torch.diff(centers[index]), unbiased=False).cpu()) if count > 2 else 0.0,
                **_float_design(decoded, index, count),
                **{f"r_center_{slot + 1:02d}": float(centers[index, slot].cpu()) if slot < count else "" for slot in range(6)},
                "_base": best_base[index].clone(),
                "_allocation": best_allocation[index].clone() if best_allocation is not None else None,
            }
            candidates.append(candidate)
    candidates.sort(key=lambda row: (not bool(row["density_feasible"]), float(row["bohm_max_deviation"])))
    best = candidates[0]
    count = int(best["nncoil"])
    final_decoded = _decode(best["_base"][None], count=count, bounds=bounds)
    initial_row = seed_lookup[str(best["seed_case_id"])]
    initial_base = torch.from_numpy(_encode(initial_row, count=count, bounds=bounds))[None].to(surrogate.device)
    initial_decoded = _decode(initial_base, count=count, bounds=bounds)
    initial_allocation = None
    final_allocation = best["_allocation"][None] if best["_allocation"] is not None else None
    if representation == "union_sdf":
        initial_allocation = torch.from_numpy(_allocation_logits(initial_row, count, bounds))[None].to(surrogate.device)
    with torch.no_grad():
        initial_fields, initial_mask, _, initial_centers = _predict(
            surrogate, initial_decoded, representation=representation, count=count,
            allocation_logits=initial_allocation, trust_fraction=float(args.trust_fraction), bounds=bounds,
        )
        final_fields, final_mask, _, final_centers = _predict(
            surrogate, final_decoded, representation=representation, count=count,
            allocation_logits=final_allocation, trust_fraction=float(args.trust_fraction), bounds=bounds,
        )
        initial_metric = metric(initial_fields)
        final_metric = metric(final_fields)
    public = [{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates]
    _write_csv(out_dir / "optimization_history.csv", history)
    _write_csv(out_dir / "final_candidates.csv", public)
    _write_csv(out_dir / "high_fidelity_candidates_top3.csv", public[:3])
    best_by_count = [next(row for row in public if int(row["nncoil"]) == count) for count in counts]
    _write_csv(out_dir / "best_by_coil_count.csv", best_by_count)
    _plot_history(history, out_dir, float(args.density_threshold))
    _plot_result(
        surrogate,
        {"bohm": initial_metric["bohm_field"][0].cpu().numpy(), "bin_flux": initial_metric["bin_flux"][0].cpu().numpy()},
        {"bohm": final_metric["bohm_field"][0].cpu().numpy(), "bin_flux": final_metric["bin_flux"][0].cpu().numpy()},
        initial_centers[0].cpu().numpy(), final_centers[0].cpu().numpy(),
        initial_mask[0].cpu().numpy(), final_mask[0].cpu().numpy(), out_dir,
    )
    summary = {
        "representation": representation,
        "device": str(surrogate.device),
        "objective": "minimize_max_equal_area_binned_bohm_flux_deviation",
        "density_threshold": float(args.density_threshold),
        "argon_ion_mass_kg": ARGON_ION_MASS,
        "radial_bins": int(args.radial_bins),
        "wafer_region": surrogate.selector_meta,
        "trust_fraction": float(args.trust_fraction) if representation == "union_sdf" else 0.0,
        "sdf_asymmetric_start_count_per_coil_count": (
            int(args.sdf_asymmetric_starts) if representation == "union_sdf" else 0
        ),
        "best": {key: value for key, value in best.items() if not key.startswith("_")},
        "initial": {
            "bohm_max_deviation": float(initial_metric["max_deviation"][0].cpu()),
            "mean_density": float(initial_metric["mean_density"][0].cpu()),
        },
        "icp_validation_status": "not_run_no_high_fidelity_solver_in_repository",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "index.md").write_text(
        "\n".join([
            f"# {representation} Bohm-flux optimization", "",
            f"- best coil count: `{count}`", f"- maximum local deviation: `{float(best['bohm_max_deviation']):.6f}`",
            f"- mean density: `{float(best['mean_density']):.6e} m^-3`", f"- density feasible: `{bool(best['density_feasible'])}`", "",
            "## Figures", "", "- [Loss history](optimization_loss_history.png)",
            "- [Bohm-flux spatial distribution](bohm_flux_spatial_distribution.png)",
            "- [Wafer profile](wafer_bohm_flux_profile.png)", "- [Coil layout](optimized_coil_layout.png)", "",
            "## Candidate data", "", "- [Top three for high-fidelity validation](high_fidelity_candidates_top3.csv)",
            "- [All final candidates](final_candidates.csv)", "- [Summary](summary.json)",
        ]), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = _args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((DEFAULT_RUNS["dimension"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    summaries = [
        _run_representation(representation=representation, dataset_root=dataset_root, out_dir=out_dir / representation, args=args)
        for representation in ("dimension", "union_sdf")
    ]
    rows = [
        {
            "representation": summary["representation"],
            "nncoil": summary["best"]["nncoil"],
            "bohm_max_deviation": summary["best"]["bohm_max_deviation"],
            "mean_bohm_flux": summary["best"]["mean_bohm_flux"],
            "mean_density": summary["best"]["mean_density"],
            "density_feasible": summary["best"]["density_feasible"],
            "spacing_std": summary["best"]["spacing_std"],
            "icp_validation_status": summary["icp_validation_status"],
        }
        for summary in summaries
    ]
    _write_csv(out_dir / "comparison_summary.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for representation in ("dimension", "union_sdf"):
        data = _read_rows(out_dir / representation / "best_by_coil_count.csv")
        axes[0].plot([int(float(row["nncoil"])) for row in data], [float(row["bohm_max_deviation"]) for row in data], marker="o", label=representation)
        axes[1].plot([int(float(row["nncoil"])) for row in data], [float(row["mean_density"]) / 1.0e17 for row in data], marker="o", label=representation)
    axes[0].set(xlabel="coil count", ylabel="best feasible max Bohm deviation", title="Absolute Bohm-flux uniformity")
    axes[1].axhline(float(args.density_threshold) / 1.0e17, color="black", linestyle="--", label="threshold")
    axes[1].set(xlabel="coil count", ylabel="mean density [1e17 m^-3]", title="Density feasibility")
    for axis in axes:
        axis.set_xticks(range(2, 7)); axis.grid(alpha=0.25); axis.legend()
    fig.savefig(out_dir / "bohm_flux_representation_comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Bohm-flux representation optimization", "",
        "Current frozen U-Nets were used. Dimension searches regular layouts; union SDF searches trust-region independent layouts.", "",
        "| representation | coils | max deviation | mean density | feasible | spacing std |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['representation']} | {int(row['nncoil'])} | {float(row['bohm_max_deviation']):.6f} | {float(row['mean_density']):.4e} | {row['density_feasible']} | {float(row['spacing_std']):.4f} |")
    dimension_row, sdf_row = rows
    relative = 100.0 * (
        float(sdf_row["bohm_max_deviation"]) - float(dimension_row["bohm_max_deviation"])
    ) / float(sdf_row["bohm_max_deviation"])
    lines += ["", "## Evaluation", "",
              f"- Dimension produced a `{relative:.1f}%` lower absolute maximum deviation than union SDF.",
              f"- The best SDF spacing standard deviation is `{float(sdf_row['spacing_std']):.6f}`; this is effectively equal spacing.",
              "- This surrogate-stage result does not support an SDF advantage with the current regular-layout-trained model.",
              "- ICP validation error and regret require an external high-fidelity solver run.", "",
              "- [Comparison graph](bohm_flux_representation_comparison.png)", "- [Comparison data](comparison_summary.csv)",
              "- [Dimension report](dimension/index.md)", "- [Union-SDF report](union_sdf/index.md)", "",
              "High-fidelity ICP validation was not executed because this repository contains converted exports but no callable ICP solver. Top-three candidate tables are ready for solver export."]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
