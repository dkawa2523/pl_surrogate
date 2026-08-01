"""Fair coil/process optimization with the dimension and union-SDF U-Nets.

Both frozen surrogates optimize the same manufacturable regular-coil generator
and the same scalar objective: the area-weighted harmonic mean of wafer-near
quasi-neutral density.  Only the representation delivered to the U-Net differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from plasma_surrogate.core.run_bundle import RunBundleLoader as _RunBundleLoader  # noqa: F401
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.preprocessing.spatial_features import (
    apply_coord_feature_scaling,
    apply_distance_transform,
)

from experiments.icp_stage4.scripts.optimize_response_inventory_sdf_density import (
    Bounds,
    _decode,
    _encode,
    _float_design,
    _load_structure_cases,
    _mask_hash,
    _plot_history,
    _plot_profiles,
    _plot_spatial,
    _plot_structure,
    _read_json,
    _write_csv,
)


DEFAULT_RUNS = {
    "dimension": Path("runs/icp_stage4_regular_layout_representation_v1_dimension"),
    "union_sdf": Path("runs/icp_stage4_regular_layout_representation_v1_union_sdf"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/icp_stage4_regular_layout_representation_v1_optimization"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--wafer-layers", type=int, default=3)
    parser.add_argument("--coil-counts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--protocol", default="structure_holdout")
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument(
        "--objective",
        choices=("harmonic", "uniformity_threshold"),
        default="harmonic",
    )
    parser.add_argument("--density-threshold", type=float, default=1.0e17)
    parser.add_argument("--threshold-penalty", type=float, default=20.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class RepresentationSurrogate:
    """Differentiable inference adapter preserving each training contract."""

    def __init__(
        self,
        *,
        representation: str,
        run_dir: Path,
        dataset_root: Path,
        protocol: str,
        wafer_layers: int,
    ) -> None:
        self.representation = representation
        checkpoint = run_dir / "models/unet/eval_protocol" / protocol / "checkpoints"
        self.model = load_checkpoint(checkpoint)
        self.model._ensure_net_device()
        self.device = torch.device(self.model.device)
        if self.device.type != "cuda":
            raise RuntimeError(f"GPU inference is required, got device={self.device}")
        self.model.net.eval()
        for parameter in self.model.net.parameters():
            parameter.requires_grad_(False)

        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        self.cond_columns = [str(value) for value in resolved["dataset"]["cond_columns"]]
        self.channels = [str(value) for value in self.model.input_feature_channels]
        self.output_keys = [str(value) for value in self.model.output_keys]
        expected = "part_sdf_union" in self.channels
        if expected != (representation == "union_sdf"):
            raise ValueError(f"checkpoint channels do not match representation={representation}: {self.channels}")

        scaler_dir = run_dir / "preprocessing/scalers/by_split" / protocol
        self.cond_scaler = _read_json(scaler_dir / "cond_scaler.json")
        self.coord_scaler = _read_json(scaler_dir / "coord_feature_scaler.json")
        self.y_scalers = _read_json(scaler_dir / "y_scalers.json")

        feature_dir = run_dir / "preprocessing/features"
        pack_path = feature_dir / "static_spatial_feature_pack.npz"
        if not pack_path.exists():
            pack_path = feature_dir / "coord_feature_pack.npz"
        pack = np.load(pack_path, allow_pickle=True)
        names = [str(value) for value in pack["channels"].tolist()]
        static = {name: np.asarray(pack["data"][i], dtype=np.float32) for i, name in enumerate(names)}
        self.r_np = static["x"]
        self.z_np = static["y"]
        self.h, self.w = self.r_np.shape
        self.dr = float(np.median(np.diff(self.r_np[0])))
        self.dz = float(np.median(np.diff(self.z_np[:, 0])))

        raw = np.zeros((self.h, self.w, len(self.channels)), dtype=np.float32)
        for index, name in enumerate(self.channels):
            if name in static:
                raw[..., index] = static[name]
        distance_cfg = dict(self.coord_scaler.get("distance_transform_effective", {}))
        flat, _ = apply_distance_transform(raw.reshape(-1, len(self.channels)), channels=self.channels, cfg=distance_cfg)
        flat, _, _ = apply_coord_feature_scaling(
            flat,
            channels=self.channels,
            coord_feature_scaler_artifact=self.coord_scaler,
            distance_transform_cfg=distance_cfg,
        )
        scaled = np.moveaxis(flat.reshape(self.h, self.w, -1), -1, 0).astype(np.float32)
        self.static_scaled = torch.from_numpy(scaled).to(self.device)
        self.sdf_index = self.channels.index("part_sdf_union") if expected else None

        self.r = torch.from_numpy(self.r_np).to(self.device)
        self.z = torch.from_numpy(self.z_np).to(self.device)
        plasma_path = dataset_root / "geometry/mask_plasma.npy"
        self.plasma_np = np.load(plasma_path).astype(np.float32) > 0.5
        self.plasma = torch.from_numpy(self.plasma_np).to(self.device)
        self.selector_np, self.selector_meta = self._wafer_surface_band(self.plasma_np, layers=wafer_layers)
        self.selector = torch.from_numpy(self.selector_np).to(self.device)
        columns = np.any(self.selector_np, axis=0)
        self.profile_columns_np = columns
        self.profile_columns = torch.from_numpy(columns).to(self.device)
        counts = np.sum(self.selector_np[:, columns], axis=0).astype(np.float32)
        self.profile_counts = torch.from_numpy(counts).to(self.device)
        self.radial_np = self.r_np[0, columns].astype(np.float32)
        self.radial_weights = torch.from_numpy(np.maximum(self.radial_np, 0.5 * self.dr)).to(self.device)

    @staticmethod
    def _wafer_surface_band(plasma: np.ndarray, *, layers: int) -> tuple[np.ndarray, dict[str, float]]:
        mask = np.asarray(plasma, dtype=bool)
        first = np.asarray([np.flatnonzero(mask[:, col])[0] if np.any(mask[:, col]) else -1 for col in range(mask.shape[1])])
        global_first = int(np.min(first[first >= 0]))
        raised = first > global_first
        surface_rows = np.unique(first[raised])
        if surface_rows.size != 1:
            raise ValueError(f"wafer surface is not a single grid row: {surface_rows.tolist()}")
        row0 = int(surface_rows[0])
        selector = np.zeros_like(mask, dtype=bool)
        for col in np.flatnonzero(raised):
            stop = min(row0 + max(1, int(layers)), mask.shape[0])
            selector[row0:stop, col] = mask[row0:stop, col]
        return selector, {
            "surface_row": float(row0),
            "layers": float(layers),
            "radial_column_count": float(np.count_nonzero(raised)),
        }

    def _inverse(self, value: torch.Tensor, name: str) -> torch.Tensor:
        scaler = dict(self.y_scalers[name])
        if str(scaler.get("type")) == "zscore":
            return value * float(scaler["std"][0]) + float(scaler["mean"][0])
        if str(scaler.get("type")) == "robust":
            return value * float(scaler["iqr"][0]) + float(scaler["median"][0])
        return value

    def _scaled_conditions(self, decoded: dict[str, torch.Tensor], count: int) -> torch.Tensor:
        values = []
        for name in self.cond_columns:
            if name == "nncoil":
                values.append(torch.full_like(decoded["pp"], float(count)))
            else:
                values.append(decoded[name])
        raw = torch.stack(values, dim=1)
        median = torch.as_tensor(self.cond_scaler["median"], dtype=raw.dtype, device=raw.device)
        iqr = torch.as_tensor(self.cond_scaler["iqr"], dtype=raw.dtype, device=raw.device)
        return (raw - median[None]) / iqr[None]

    def sdf_mask(self, decoded: dict[str, torch.Tensor], *, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard raster mask and differentiable training-compatible SDF.

        Training uses signed Manhattan distance in grid cells.  The +1 shift
        matches the outside-distance convention used by ``sdf_from_part_mask``.
        """

        batch = int(decoded["llcoil"].shape[0])
        ll = decoded["llcoil"][:, None, None]
        pitch = (decoded["rrce"] - decoded["rrc"]) / float(count)
        sdf = torch.full((batch, self.h, self.w), 1.0e6, dtype=torch.float32, device=self.device)
        hard = torch.zeros((batch, self.h, self.w), dtype=torch.bool, device=self.device)
        for index in range(count):
            r0 = (decoded["rrc"] + float(index) * pitch)[:, None, None]
            z0 = (14.0 + decoded["zzc"])[:, None, None]
            q_r = torch.abs(self.r[None] - (r0 + 0.5 * ll)) / self.dr - 0.5 * ll / self.dr
            q_z = torch.abs(self.z[None] - (z0 + 0.5 * ll)) / self.dz - 0.5 * ll / self.dz
            rectangle = torch.relu(q_r) + torch.relu(q_z) + torch.minimum(
                torch.maximum(q_r, q_z), torch.zeros_like(q_r)
            )
            sdf = torch.minimum(sdf, rectangle)
            hard |= (
                (self.r[None] >= r0)
                & (self.r[None] <= r0 + ll)
                & (self.z[None] >= z0)
                & (self.z[None] <= z0 + ll)
            )
        return hard.to(dtype=torch.float32), sdf + 1.0

    def predict(
        self, decoded: dict[str, torch.Tensor], *, count: int, tau_px: float = 0.5
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        del tau_px
        mask, sdf = self.sdf_mask(decoded, count=count)
        if self.sdf_index is not None:
            return self.predict_from_sdf(decoded, count=count, mask=mask, sdf=sdf)
        batch = int(mask.shape[0])
        spatial = self.static_scaled[None].expand(batch, -1, -1, -1)
        cond = self._scaled_conditions(decoded, count)[:, :, None, None].expand(-1, -1, self.h, self.w)
        raw = self.model._torch_forward(torch.cat([cond, spatial], dim=1))
        fields = {name: self._inverse(raw[:, index], name) for index, name in enumerate(self.output_keys)}
        return fields, mask, sdf

    def predict_from_sdf(
        self,
        decoded: dict[str, torch.Tensor],
        *,
        count: int,
        mask: torch.Tensor,
        sdf: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Infer from a caller-supplied differentiable union SDF."""

        if self.sdf_index is None:
            raise ValueError("predict_from_sdf is available only for a union-SDF checkpoint")
        batch = int(mask.shape[0])
        spatial = self.static_scaled[None].expand(batch, -1, -1, -1).clone()
        rule = dict(self.coord_scaler["channels"]["part_sdf_union"])
        spatial[:, self.sdf_index] = (sdf - float(rule["mean"][0])) / float(rule["std"][0])
        cond = self._scaled_conditions(decoded, count)[:, :, None, None].expand(-1, -1, self.h, self.w)
        raw = self.model._torch_forward(torch.cat([cond, spatial], dim=1))
        fields = {name: self._inverse(raw[:, index], name) for index, name in enumerate(self.output_keys)}
        return fields, mask, sdf

    def density_metrics(self, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        density = 0.5 * (fields["ne"] + fields["ni"])
        selected = torch.where(self.selector[None], density, torch.zeros_like(density))
        profile = torch.sum(selected, dim=1)[:, self.profile_columns] / self.profile_counts[None]
        positive = torch.clamp(profile, min=1.0e8)
        weights = self.radial_weights[None]
        weight_sum = torch.sum(weights, dim=1)
        mean = torch.sum(weights * positive, dim=1) / weight_sum
        variance = torch.sum(weights * (positive - mean[:, None]).square(), dim=1) / weight_sum
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        harmonic = weight_sum / torch.sum(weights / positive, dim=1)
        return {
            "profile": profile,
            "mean": mean,
            "std": std,
            "cv": std / torch.clamp(mean, min=1.0e8),
            "minimum": torch.min(profile, dim=1).values,
            "harmonic": harmonic,
        }


def _fixed_seed_rows(
    structures: list[dict[str, Any]], *, process_center: tuple[float, float]
) -> tuple[list[dict[str, float]], list[str]]:
    """Choose model-independent starts nearest the observed process centre."""

    rows: list[dict[str, float]] = []
    case_ids: list[str] = []
    pp_mid, pp0_mid = process_center
    pp_scale = max(pp_mid, 1.0)
    pp0_scale = max(pp0_mid, 1.0e-6)
    for structure in structures:
        operation = min(
            structure["operations"],
            key=lambda op: ((float(op["pp"]) - pp_mid) / pp_scale) ** 2
            + ((float(op["pp0"]) - pp0_mid) / pp0_scale) ** 2,
        )
        rows.append({**structure["geometry"], "pp": float(operation["pp"]), "pp0": float(operation["pp0"])})
        case_ids.append(str(operation["case_id"]))
    return rows, case_ids


def _plot_uniformity_history(
    rows: list[dict[str, Any]], out_dir: Path, *, density_threshold: float
) -> None:
    """Plot the optimized loss and the best feasible CV/mean trajectory."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for count in sorted({int(float(row["nncoil"])) for row in rows}):
        count_rows = [row for row in rows if int(float(row["nncoil"])) == count]
        steps = sorted({int(row["step"]) for row in count_rows})
        loss_values: list[float] = []
        cv_values: list[float] = []
        mean_values: list[float] = []
        for step in steps:
            step_rows = [row for row in count_rows if int(row["step"]) == step]
            loss_values.append(min(float(row["loss"]) for row in step_rows))
            feasible = [row for row in step_rows if bool(row["density_feasible"])]
            chosen = (
                min(feasible, key=lambda row: float(row["density_cv"]))
                if feasible
                else max(step_rows, key=lambda row: float(row["mean_density"]))
            )
            cv_values.append(float(chosen["density_cv"]) if feasible else np.nan)
            mean_values.append(float(chosen["mean_density"]) / 1.0e17)
        axes[0].plot(steps, loss_values, label=f"n={count}")
        axes[1].plot(steps, cv_values, label=f"n={count}")
        axes[2].plot(steps, mean_values, label=f"n={count}")
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="optimization step",
        ylabel="penalized loss (log scale)",
        title="Optimization loss",
    )
    axes[1].set(xlabel="optimization step", ylabel="best feasible density CV", title="Uniformity objective")
    axes[2].axhline(float(density_threshold) / 1.0e17, color="black", linestyle="--", label="threshold")
    axes[2].set(xlabel="optimization step", ylabel="mean density [1e17 m^-3]", title="Density feasibility")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(out_dir / "optimization_loss_history.png", dpi=180)
    plt.close(fig)


def _run_one(
    *,
    representation: str,
    run_dir: Path,
    out_dir: Path,
    dataset_root: Path,
    protocol: str,
    counts: list[int],
    steps: int,
    lr: float,
    wafer_layers: int,
    objective: str,
    density_threshold: float,
    threshold_penalty: float,
    smoke: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = Bounds()
    surrogate = RepresentationSurrogate(
        representation=representation,
        run_dir=run_dir,
        dataset_root=dataset_root,
        protocol=protocol,
        wafer_layers=wafer_layers,
    )
    structures_by_count = _load_structure_cases(dataset_root)
    process_center = (1753.353515625, 0.017528499476611614)

    training_masks: list[np.ndarray] = []
    training_hashes: set[str] = set()
    for count, structures in structures_by_count.items():
        seed_rows, _ = _fixed_seed_rows(structures, process_center=process_center)
        raw = torch.from_numpy(np.stack([_encode(row, count=count, bounds=bounds) for row in seed_rows])).to(surrogate.device)
        with torch.no_grad():
            masks, _ = surrogate.sdf_mask(_decode(raw, count=count, bounds=bounds), count=count)
        for mask in masks.cpu().numpy():
            training_masks.append(mask)
            training_hashes.add(_mask_hash(mask))

    history: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    initial_lookup: dict[str, dict[str, float]] = {}
    for count in counts:
        structures = structures_by_count[count]
        if smoke:
            structures = structures[:2]
        initial_rows, initial_case_ids = _fixed_seed_rows(structures, process_center=process_center)
        initial_lookup.update(dict(zip(initial_case_ids, initial_rows)))
        logits = torch.nn.Parameter(
            torch.from_numpy(np.stack([_encode(row, count=count, bounds=bounds) for row in initial_rows])).to(surrogate.device)
        )
        optimizer = torch.optim.Adam([logits], lr=lr)
        best_rank = np.full(len(initial_rows), np.inf)
        best_logits = logits.detach().clone()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            decoded = _decode(logits, count=count, bounds=bounds)
            fields, _, _ = surrogate.predict(decoded, count=count)
            metrics = surrogate.density_metrics(fields)
            density_deficit = torch.relu(
                (float(density_threshold) - metrics["mean"]) / float(density_threshold)
            )
            if objective == "uniformity_threshold":
                loss_each = metrics["cv"] + float(threshold_penalty) * density_deficit.square()
                # The rank implements the declared lexicographic decision:
                # feasible candidates first, then minimum CV.  Before reaching
                # feasibility, smaller density deficit takes priority.
                rank_each = torch.where(
                    metrics["mean"] >= float(density_threshold),
                    metrics["cv"],
                    1000.0 + density_deficit + metrics["cv"] * 1.0e-3,
                )
            else:
                loss_each = -torch.log(torch.clamp(metrics["harmonic"] / 1.0e17, min=1.0e-12))
                rank_each = -metrics["harmonic"]
            loss_each.mean().backward()
            torch.nn.utils.clip_grad_norm_([logits], max_norm=10.0)
            with torch.no_grad():
                ranks = rank_each.cpu().numpy()
                for index, rank in enumerate(ranks):
                    if float(rank) < best_rank[index]:
                        best_rank[index] = float(rank)
                        best_logits[index] = logits.detach()[index]
                    history.append(
                        {
                            "representation": representation,
                            "nncoil": count,
                            "restart": index,
                            "seed_case_id": initial_case_ids[index],
                            "step": step,
                            "loss": float(loss_each[index].cpu()),
                            "harmonic_density": float(metrics["harmonic"][index].cpu()),
                            "mean_density": float(metrics["mean"][index].cpu()),
                            "density_std": float(metrics["std"][index].cpu()),
                            "density_cv": float(metrics["cv"][index].cpu()),
                            "minimum_density": float(metrics["minimum"][index].cpu()),
                            "density_threshold": float(density_threshold),
                            "density_feasible": bool(metrics["mean"][index].cpu() >= float(density_threshold)),
                            "normalized_density_deficit": float(density_deficit[index].cpu()),
                            **_float_design(decoded, index, count),
                        }
                    )
            optimizer.step()

        with torch.no_grad():
            final_decoded = _decode(logits, count=count, bounds=bounds)
            final_fields, _, _ = surrogate.predict(final_decoded, count=count)
            final_metrics = surrogate.density_metrics(final_fields)
            final_deficit = torch.relu(
                (float(density_threshold) - final_metrics["mean"]) / float(density_threshold)
            )
            if objective == "uniformity_threshold":
                final_rank = torch.where(
                    final_metrics["mean"] >= float(density_threshold),
                    final_metrics["cv"],
                    1000.0 + final_deficit + final_metrics["cv"] * 1.0e-3,
                )
            else:
                final_rank = -final_metrics["harmonic"]
        for index, rank in enumerate(final_rank.cpu().numpy()):
            if float(rank) < best_rank[index]:
                best_rank[index] = float(rank)
                best_logits[index] = logits.detach()[index]

        decoded_best = _decode(best_logits, count=count, bounds=bounds)
        with torch.no_grad():
            fields, masks, _ = surrogate.predict(decoded_best, count=count)
            metrics = surrogate.density_metrics(fields)
        for index in range(len(initial_rows)):
            mask_np = masks[index].cpu().numpy()
            nearest_xor = min(float(np.mean((mask_np > 0.5) != (train > 0.5))) for train in training_masks)
            final_candidates.append(
                {
                    "representation": representation,
                    "nncoil": count,
                    "restart": index,
                    "seed_case_id": initial_case_ids[index],
                    "harmonic_density": float(metrics["harmonic"][index].cpu()),
                    "mean_density": float(metrics["mean"][index].cpu()),
                    "density_std": float(metrics["std"][index].cpu()),
                    "density_cv": float(metrics["cv"][index].cpu()),
                    "minimum_density": float(metrics["minimum"][index].cpu()),
                    "density_threshold": float(density_threshold),
                    "density_feasible": bool(metrics["mean"][index].cpu() >= float(density_threshold)),
                    "unique_structure": _mask_hash(mask_np) not in training_hashes,
                    "nearest_training_mask_xor_fraction": nearest_xor,
                    **_float_design(decoded_best, index, count),
                    "_logits": best_logits[index].detach().clone(),
                }
            )

    if objective == "uniformity_threshold":
        ranked = sorted(
            final_candidates,
            key=lambda row: (
                not bool(row["density_feasible"]),
                float(row["density_cv"])
                if bool(row["density_feasible"])
                else -float(row["mean_density"]),
            ),
        )
    else:
        ranked = sorted(final_candidates, key=lambda row: float(row["harmonic_density"]), reverse=True)
    unique = [row for row in ranked if bool(row["unique_structure"])]
    best = unique[0] if unique else ranked[0]
    best_count = int(best["nncoil"])
    final_decoded = _decode(best["_logits"][None], count=best_count, bounds=bounds)
    initial_row = initial_lookup[str(best["seed_case_id"])]
    initial_logits = torch.from_numpy(_encode(initial_row, count=best_count, bounds=bounds))[None].to(surrogate.device)
    initial_decoded = _decode(initial_logits, count=best_count, bounds=bounds)
    with torch.no_grad():
        initial_fields_t, initial_mask_t, _ = surrogate.predict(initial_decoded, count=best_count)
        final_fields_t, final_mask_t, final_sdf_t = surrogate.predict(final_decoded, count=best_count)
        initial_metrics = surrogate.density_metrics(initial_fields_t)
        final_metrics = surrogate.density_metrics(final_fields_t)

    public = [{key: value for key, value in row.items() if key != "_logits"} for row in ranked]
    _write_csv(out_dir / "optimization_history.csv", history)
    _write_csv(out_dir / "final_candidates.csv", public)
    if objective == "uniformity_threshold":
        best_by_count = [
            min(
                (row for row in public if int(row["nncoil"]) == count),
                key=lambda row: (
                    not bool(row["density_feasible"]),
                    float(row["density_cv"])
                    if bool(row["density_feasible"])
                    else -float(row["mean_density"]),
                ),
            )
            for count in counts
        ]
    else:
        best_by_count = [max((row for row in public if int(row["nncoil"]) == count), key=lambda row: float(row["harmonic_density"])) for count in counts]
    _write_csv(out_dir / "best_by_coil_count.csv", best_by_count)
    if objective == "uniformity_threshold":
        _plot_uniformity_history(history, out_dir, density_threshold=density_threshold)
    else:
        _plot_history(history, out_dir)
    _plot_spatial(
        surrogate,
        {key: value[0].cpu().numpy() for key, value in initial_fields_t.items()},
        {key: value[0].cpu().numpy() for key, value in final_fields_t.items()},
        out_dir,
    )
    _plot_structure(
        surrogate,
        initial_mask_t[0].cpu().numpy(),
        final_mask_t[0].cpu().numpy(),
        final_sdf_t[0].cpu().numpy(),
        out_dir,
    )
    _plot_profiles(
        surrogate,
        initial_metrics["profile"][0].cpu().numpy(),
        final_metrics["profile"][0].cpu().numpy(),
        out_dir,
    )

    summary = {
        "representation": representation,
        "run_dir": str(run_dir),
        "device": str(surrogate.device),
        "objective": (
            "minimize_area_weighted_density_cv_subject_to_mean_density_threshold"
            if objective == "uniformity_threshold"
            else "maximize_area_weighted_harmonic_mean_quasineutral_density"
        ),
        "density_threshold": float(density_threshold),
        "threshold_penalty": float(threshold_penalty),
        "wafer_region": surrogate.selector_meta,
        "best": {key: value for key, value in best.items() if key != "_logits"},
        "initial": {
            **_float_design(initial_decoded, 0, best_count),
            "seed_case_id": str(best["seed_case_id"]),
            "harmonic_density": float(initial_metrics["harmonic"][0].cpu()),
            "mean_density": float(initial_metrics["mean"][0].cpu()),
            "density_cv": float(initial_metrics["cv"][0].cpu()),
            "minimum_density": float(initial_metrics["minimum"][0].cpu()),
        },
        "improvement": {
            "harmonic_density_ratio": float(final_metrics["harmonic"][0].cpu() / initial_metrics["harmonic"][0].cpu()),
            "mean_density_ratio": float(final_metrics["mean"][0].cpu() / initial_metrics["mean"][0].cpu()),
            "cv_ratio": float(final_metrics["cv"][0].cpu() / initial_metrics["cv"][0].cpu()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "index.md").write_text(
        "\n".join(
            [
                f"# {representation} regular-layout optimization",
                "",
                f"- GPU: `{summary['device']}`",
                f"- best coil count: `{best_count}`",
                f"- harmonic density: `{float(best['harmonic_density']):.6e} m^-3`",
                f"- density CV: `{float(best['density_cv']):.6f}`",
                f"- density threshold feasible: `{bool(best['density_feasible'])}`",
                f"- unique raster structure: `{bool(best['unique_structure'])}`",
                "",
                "## Figures",
                "",
                "- [Optimization history](optimization_loss_history.png)",
                "- [Density spatial distribution](density_spatial_distribution.png)",
                "- [Optimized coil structure and SDF](optimized_coil_structure_sdf.png)",
                "- [Wafer-near density profile](wafer_density_radial_profile.png)",
                "",
                "## Data",
                "",
                "- [Summary](summary.json)",
                "- [Optimization history](optimization_history.csv)",
                "- [Final candidates](final_candidates.csv)",
                "- [Best by coil count](best_by_coil_count.csv)",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def _comparison_report(
    out_dir: Path, summaries: list[dict[str, Any]], *, objective: str
) -> None:
    rows = []
    for summary in summaries:
        best = summary["best"]
        initial = summary["initial"]
        rows.append(
            {
                "representation": summary["representation"],
                "best_nncoil": int(best["nncoil"]),
                "initial_harmonic_density": float(initial["harmonic_density"]),
                "optimized_harmonic_density": float(best["harmonic_density"]),
                "harmonic_improvement_ratio": float(summary["improvement"]["harmonic_density_ratio"]),
                "initial_mean_density": float(initial["mean_density"]),
                "initial_density_cv": float(initial["density_cv"]),
                "optimized_mean_density": float(best["mean_density"]),
                "optimized_density_cv": float(best["density_cv"]),
                "density_feasible": bool(best["density_feasible"]),
                "unique_structure": bool(best["unique_structure"]),
                "nearest_training_mask_xor_fraction": float(best["nearest_training_mask_xor_fraction"]),
                "llcoil": float(best["llcoil"]),
                "rrc": float(best["rrc"]),
                "rrce": float(best["rrce"]),
                "zzc": float(best["zzc"]),
                "pp": float(best["pp"]),
                "pp0": float(best["pp0"]),
            }
        )
    _write_csv(out_dir / "comparison_summary.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    labels = [row["representation"] for row in rows]
    axes[0].bar(labels, [row["optimized_mean_density"] / 1.0e17 for row in rows])
    axes[0].set(ylabel="predicted mean density [1e17 m^-3]", title="Optimized density level")
    axes[1].bar(labels, [row["optimized_density_cv"] for row in rows])
    axes[1].set(ylabel="predicted density CV", title="Wafer-near nonuniformity (diagnostic)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "representation_optimization_comparison.png", dpi=180)
    plt.close(fig)

    objective_text = (
        "minimum wafer-near density CV subject to the fixed mean-density threshold"
        if objective == "uniformity_threshold"
        else "area-weighted harmonic mean of wafer-near `(ne + ni) / 2`"
    )
    lines = [
        "# ICP regular-layout representation optimization",
        "",
        "Frozen dimension and union-SDF U-Nets were optimized independently with the same ",
        "regular-coil generator, bounds, starts, objective, optimizer and GPU protocol.",
        f"The decision objective is {objective_text}.",
        "",
        "| representation | coils | initial mean | optimized mean | initial CV | optimized CV | feasible |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation']} | {row['best_nncoil']} | {row['initial_mean_density']:.4e} | "
            f"{row['optimized_mean_density']:.4e} | {row['initial_density_cv']:.4f} | "
            f"{row['optimized_density_cv']:.4f} | {row['density_feasible']} |"
        )
    interpretation = [
        "- Compare representations by high-fidelity ICP validation, not by the two surrogate objective values directly: each model has its own calibration error.",
    ]
    if objective == "uniformity_threshold":
        interpretation.append(
            "- Feasibility uses predicted area-weighted mean density; local minimum density may be below the threshold."
        )
    else:
        interpretation.append(
            "- The harmonic mean rewards high density and penalizes low radial values, but it does not constrain CV. A density increase can therefore accompany a CV increase."
        )
    interpretation.append(
        "- `unique_structure=True` means a new raster mask, not proof of topology extrapolation. The XOR distance reports how far it is from the nearest observed structure."
    )
    lines += [
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Comparison",
        "",
        "- [Objective and CV comparison](representation_optimization_comparison.png)",
        "- [Comparison data](comparison_summary.csv)",
        "",
        "## Per-representation reports",
        "",
        "- [Dimension optimization](dimension/index.md)",
        "- [Union-SDF optimization](union_sdf/index.md)",
        "",
        "These are surrogate predictions. A high-fidelity ICP rerun is required before claiming a physically superior design.",
    ]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    first_cfg = yaml.safe_load((DEFAULT_RUNS["dimension"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset_root = Path(first_cfg["dataset"]["root"]).resolve()
    counts = [4] if args.smoke else [int(value) for value in args.coil_counts]
    steps = 2 if args.smoke else int(args.steps)
    summaries = []
    for representation, relative_run in DEFAULT_RUNS.items():
        print(f"optimizing {representation} on GPU", flush=True)
        summaries.append(
            _run_one(
                representation=representation,
                run_dir=relative_run.resolve(),
                out_dir=out_dir / representation,
                dataset_root=dataset_root,
                protocol=str(args.protocol),
                counts=counts,
                steps=steps,
                lr=float(args.lr),
                wafer_layers=int(args.wafer_layers),
                objective=str(args.objective),
                density_threshold=float(args.density_threshold),
                threshold_penalty=float(args.threshold_penalty),
                smoke=bool(args.smoke),
            )
        )
    _comparison_report(out_dir, summaries, objective=str(args.objective))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
