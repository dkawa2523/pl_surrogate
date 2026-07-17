"""Optimize ICP coil-series geometry for high wafer-wide density.

The frozen response-inventory U-Net receives the same hard ``mask_coil`` used
during training.  Gradients are supplied by a soft occupancy derived from the
analytic union SDF (straight-through estimator).  The only objective is the
area-weighted harmonic mean of wafer-near quasi-neutral density.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from plasma_surrogate.core.run_bundle import RunBundleLoader as _RunBundleLoader  # noqa: F401
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.preprocessing.spatial_features import (
    apply_coord_feature_scaling,
    apply_distance_transform,
)


@dataclass(frozen=True)
class Bounds:
    llcoil: tuple[float, float] = (0.55344, 1.44492)
    rrc: tuple[float, float] = (2.49330, 9.52599)
    rrce: tuple[float, float] = (20.59640, 29.38050)
    zzc: tuple[float, float] = (0.29864, 4.69076)
    pp: tuple[float, float] = (586.652, 2903.40)
    pp0: tuple[float, float] = (0.0038814, 0.0867448)
    min_gap_fraction: float = 0.20


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/icp_stage4_response_inventory_v1/unet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/icp_stage4_response_inventory_v1/sdf_density_uniformity_opt_v1"),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--protocol", default="structure_holdout")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--wafer-layers", type=int, default=3)
    parser.add_argument("--coil-counts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(value / (1.0 - value)).astype(np.float32)


def _unit(value: float, bounds: tuple[float, float]) -> float:
    return (float(value) - float(bounds[0])) / (float(bounds[1]) - float(bounds[0]))


def _decode(logits: torch.Tensor, *, count: int, bounds: Bounds) -> dict[str, torch.Tensor]:
    u = torch.sigmoid(logits)
    ll = bounds.llcoil[0] + u[:, 0] * (bounds.llcoil[1] - bounds.llcoil[0])
    rrc_max = torch.minimum(
        torch.full_like(ll, bounds.rrc[1]),
        torch.full_like(ll, bounds.rrce[1]) - float(count) * (1.0 + bounds.min_gap_fraction) * ll,
    )
    rrc = bounds.rrc[0] + u[:, 1] * (rrc_max - bounds.rrc[0])
    rrce_min = torch.maximum(
        torch.full_like(ll, bounds.rrce[0]),
        rrc + float(count) * (1.0 + bounds.min_gap_fraction) * ll,
    )
    rrce = rrce_min + u[:, 2] * (bounds.rrce[1] - rrce_min)
    zzc = bounds.zzc[0] + u[:, 3] * (bounds.zzc[1] - bounds.zzc[0])
    pp = bounds.pp[0] + u[:, 4] * (bounds.pp[1] - bounds.pp[0])
    pp0 = bounds.pp0[0] + u[:, 5] * (bounds.pp0[1] - bounds.pp0[0])
    return {"llcoil": ll, "rrc": rrc, "rrce": rrce, "zzc": zzc, "pp": pp, "pp0": pp0}


def _encode(row: dict[str, float], *, count: int, bounds: Bounds) -> np.ndarray:
    ll = float(row["llcoil"])
    rrc_max = min(bounds.rrc[1], bounds.rrce[1] - count * (1.0 + bounds.min_gap_fraction) * ll)
    rrce_min = max(bounds.rrce[0], float(row["rrc"]) + count * (1.0 + bounds.min_gap_fraction) * ll)
    values = np.asarray(
        [
            _unit(ll, bounds.llcoil),
            (float(row["rrc"]) - bounds.rrc[0]) / max(rrc_max - bounds.rrc[0], 1.0e-9),
            (float(row["rrce"]) - rrce_min) / max(bounds.rrce[1] - rrce_min, 1.0e-9),
            _unit(float(row["zzc"]), bounds.zzc),
            _unit(float(row["pp"]), bounds.pp),
            _unit(float(row["pp0"]), bounds.pp0),
        ],
        dtype=np.float64,
    )
    return _logit(values)


def _float_design(decoded: dict[str, torch.Tensor], index: int, count: int) -> dict[str, float]:
    return {
        "nncoil": float(count),
        **{key: float(value[index].detach().cpu()) for key, value in decoded.items()},
    }


def _load_structure_cases(dataset_root: Path) -> dict[int, list[dict[str, Any]]]:
    index_name = "index_ignited.csv" if (dataset_root / "index_ignited.csv").exists() else "index.csv"
    with (dataset_root / index_name).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[float, ...], list[dict[str, str]]] = {}
    keys = ("nncoil", "llcoil", "rrc", "rrce", "zzc")
    for row in rows:
        signature = tuple(float(row[key]) for key in keys)
        grouped.setdefault(signature, []).append(row)
    result: dict[int, list[dict[str, Any]]] = {}
    for signature, operations in grouped.items():
        count = int(round(signature[0]))
        result.setdefault(count, []).append(
            {
                "geometry": {key: float(operations[0][key]) for key in keys},
                "operations": [
                    {
                        "case_id": str(row["case_id"]),
                        "pp": float(row["pp"]),
                        "pp0": float(row["pp0"]),
                    }
                    for row in operations
                ],
            }
        )
    for structures in result.values():
        structures.sort(key=lambda item: tuple(item["geometry"][key] for key in keys))
    return result


class FrozenSurrogate:
    def __init__(self, *, run_dir: Path, dataset_root: Path, protocol: str, wafer_layers: int):
        checkpoint = run_dir / "models/unet/eval_protocol" / protocol / "checkpoints"
        self.model = load_checkpoint(checkpoint)
        self.model._ensure_net_device()
        self.device = torch.device(self.model.device)
        if self.device.type != "cuda":
            raise RuntimeError(f"GPU inference is required, got device={self.device}")
        self.model.net.eval()
        for parameter in self.model.net.parameters():
            parameter.requires_grad_(False)
        self.channels = [str(value) for value in self.model.input_feature_channels]
        if "mask_coil" not in self.channels:
            raise ValueError("checkpoint does not contain the case-varying mask_coil channel")
        self.output_keys = [str(value) for value in self.model.output_keys]
        self.cond_scaler = _read_json(run_dir / "preprocessing/scalers/by_split" / protocol / "cond_scaler.json")
        self.coord_scaler = _read_json(
            run_dir / "preprocessing/scalers/by_split" / protocol / "coord_feature_scaler.json"
        )
        self.y_scalers = _read_json(run_dir / "preprocessing/scalers/by_split" / protocol / "y_scalers.json")
        self.target_transforms = dict(self.cond_scaler.get("target_transforms", {}))

        pack = np.load(run_dir / "preprocessing/features/static_spatial_feature_pack.npz", allow_pickle=True)
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
        self.coil_index = self.channels.index("mask_coil")
        self.r = torch.from_numpy(self.r_np).to(self.device)
        self.z = torch.from_numpy(self.z_np).to(self.device)
        self.plasma_np = np.load(dataset_root / "geometry/mask_plasma.npy").astype(np.float32) > 0.5
        self.plasma = torch.from_numpy(self.plasma_np).to(self.device)
        self.selector_np, self.selector_meta = self._wafer_surface_band(self.plasma_np, layers=wafer_layers)
        self.selector = torch.from_numpy(self.selector_np).to(self.device)
        columns = np.any(self.selector_np, axis=0)
        self.profile_columns_np = columns
        self.profile_columns = torch.from_numpy(columns).to(self.device)
        counts = np.sum(self.selector_np[:, columns], axis=0).astype(np.float32)
        self.profile_counts = torch.from_numpy(counts).to(self.device)
        radial = self.r_np[0, columns].astype(np.float32)
        self.radial_np = radial
        self.radial_weights = torch.from_numpy(np.maximum(radial, 0.5 * self.dr)).to(self.device)

    @staticmethod
    def _wafer_surface_band(plasma: np.ndarray, *, layers: int) -> tuple[np.ndarray, dict[str, float]]:
        mask = np.asarray(plasma, dtype=bool)
        first = np.asarray([np.flatnonzero(mask[:, col])[0] if np.any(mask[:, col]) else -1 for col in range(mask.shape[1])])
        global_first = int(np.min(first[first >= 0]))
        raised = first > global_first
        if not np.any(raised):
            raise ValueError("could not identify the raised wafer surface from mask_plasma")
        surface_rows = np.unique(first[raised])
        if surface_rows.size != 1:
            raise ValueError(f"wafer surface is not a single grid row: {surface_rows.tolist()}")
        row0 = int(surface_rows[0])
        selector = np.zeros_like(mask, dtype=bool)
        for col in np.flatnonzero(raised):
            selector[row0 : min(row0 + max(1, int(layers)), mask.shape[0]), col] = mask[
                row0 : min(row0 + max(1, int(layers)), mask.shape[0]), col
            ]
        return selector, {
            "surface_row": float(row0),
            "layers": float(layers),
            "radial_column_count": float(np.count_nonzero(raised)),
        }

    def _cond_scaled(self, pp: torch.Tensor, pp0: torch.Tensor) -> torch.Tensor:
        raw = torch.stack([pp, pp0], dim=1)
        median = torch.as_tensor(self.cond_scaler["median"], dtype=raw.dtype, device=raw.device)
        iqr = torch.as_tensor(self.cond_scaler["iqr"], dtype=raw.dtype, device=raw.device)
        return (raw - median[None, :]) / iqr[None, :]

    def _inverse(self, value: torch.Tensor, name: str) -> torch.Tensor:
        scaler = dict(self.y_scalers[name])
        if str(scaler.get("type")) == "zscore":
            return value * float(scaler["std"][0]) + float(scaler["mean"][0])
        if str(scaler.get("type")) == "robust":
            return value * float(scaler["iqr"][0]) + float(scaler["median"][0])
        return value

    def _model_input(self, mask_coil: torch.Tensor, pp: torch.Tensor, pp0: torch.Tensor) -> torch.Tensor:
        batch = int(mask_coil.shape[0])
        spatial = self.static_scaled[None].expand(batch, -1, -1, -1).clone()
        spatial[:, self.coil_index] = mask_coil
        cond = self._cond_scaled(pp, pp0)[:, :, None, None].expand(-1, -1, self.h, self.w)
        return torch.cat([cond, spatial], dim=1)

    def _forward(self, mask_coil: torch.Tensor, pp: torch.Tensor, pp0: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.model._torch_forward(self._model_input(mask_coil, pp, pp0))
        return {name: self._inverse(raw[:, index], name) for index, name in enumerate(self.output_keys)}

    def sdf_mask(
        self,
        decoded: dict[str, torch.Tensor],
        *,
        count: int,
        tau_px: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(decoded["llcoil"].shape[0])
        ll = decoded["llcoil"][:, None, None]
        pitch = (decoded["rrce"] - decoded["rrc"]) / float(count)
        union_sdf = torch.full((batch, self.h, self.w), 1.0e6, dtype=torch.float32, device=self.device)
        hard_union = torch.zeros((batch, self.h, self.w), dtype=torch.bool, device=self.device)
        for index in range(count):
            r0 = (decoded["rrc"] + float(index) * pitch)[:, None, None]
            z0 = (14.0 + decoded["zzc"])[:, None, None]
            q_r = torch.abs(self.r[None] - (r0 + 0.5 * ll)) / self.dr - 0.5 * ll / self.dr
            q_z = torch.abs(self.z[None] - (z0 + 0.5 * ll)) / self.dz - 0.5 * ll / self.dz
            outside = torch.sqrt(torch.relu(q_r).square() + torch.relu(q_z).square() + 1.0e-12)
            sdf = outside + torch.minimum(torch.maximum(q_r, q_z), torch.zeros_like(q_r))
            union_sdf = torch.minimum(union_sdf, sdf)
            hard_union = hard_union | (
                (self.r[None] >= r0)
                & (self.r[None] <= r0 + ll)
                & (self.z[None] >= z0)
                & (self.z[None] <= z0 + ll)
            )
        soft = torch.sigmoid(-union_sdf / float(tau_px))
        hard = hard_union.to(dtype=soft.dtype)
        straight_through = soft + (hard - soft).detach()
        return straight_through, union_sdf

    def predict(
        self,
        decoded: dict[str, torch.Tensor],
        *,
        count: int,
        tau_px: float,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mask, sdf = self.sdf_mask(decoded, count=count, tau_px=tau_px)
        return self._forward(mask, decoded["pp"], decoded["pp0"]), mask, sdf

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


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask > 0.5, dtype=np.uint8).tobytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for count in sorted({int(row["nncoil"]) for row in rows}):
        count_rows = [row for row in rows if int(row["nncoil"]) == count]
        steps = sorted({int(row["step"]) for row in count_rows})
        best_h = [max(float(row["harmonic_density"]) for row in count_rows if int(row["step"]) == step) for step in steps]
        best_loss = [min(float(row["loss"]) for row in count_rows if int(row["step"]) == step) for step in steps]
        axes[0].plot(steps, best_loss, label=f"n={count}")
        axes[1].plot(steps, np.asarray(best_h) / 1.0e17, label=f"n={count}")
    axes[0].set(xlabel="optimization step", ylabel="loss = -ln(H / 1e17)", title="Optimization loss history")
    axes[1].set(xlabel="optimization step", ylabel="harmonic density H [1e17 m^-3]", title="Best harmonic density")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(out_dir / "optimization_loss_history.png", dpi=180)
    plt.close(fig)


def _plot_spatial(
    surrogate: FrozenSurrogate,
    initial_fields: dict[str, np.ndarray],
    final_fields: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    initial = 0.5 * (initial_fields["ne"] + initial_fields["ni"])
    final = 0.5 * (final_fields["ne"] + final_fields["ni"])
    mask = surrogate.plasma_np
    initial = np.where(mask, initial, np.nan)
    final = np.where(mask, final, np.nan)
    delta = final - initial
    vmax = float(np.nanmax([np.nanmax(initial), np.nanmax(final)]))
    vmin = float(np.nanmin([np.nanmin(initial), np.nanmin(final)]))
    dmax = float(np.nanmax(np.abs(delta)))
    extent = [float(np.min(surrogate.r_np)), float(np.max(surrogate.r_np)), float(np.min(surrogate.z_np)), float(np.max(surrogate.z_np))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, values, title in zip(axes[:2], (initial, final), ("initial predicted density", "optimized predicted density")):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(image, ax=axis, label="(ne + ni) / 2 [m^-3]")
        axis.set_title(title)
    image = axes[2].imshow(delta, origin="lower", extent=extent, aspect="auto", vmin=-dmax, vmax=dmax, cmap="coolwarm")
    fig.colorbar(image, ax=axes[2], label="optimized - initial [m^-3]")
    axes[2].set_title("density change")
    for axis in axes:
        axis.set(xlabel="r", ylabel="z")
    fig.savefig(out_dir / "density_spatial_distribution.png", dpi=180)
    plt.close(fig)


def _plot_structure(
    surrogate: FrozenSurrogate,
    initial_mask: np.ndarray,
    final_mask: np.ndarray,
    final_sdf: np.ndarray,
    out_dir: Path,
) -> None:
    extent = [float(np.min(surrogate.r_np)), float(np.max(surrogate.r_np)), float(np.min(surrogate.z_np)), float(np.max(surrogate.z_np))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].imshow(initial_mask, origin="lower", extent=extent, aspect="auto", cmap="gray_r")
    axes[0].set_title("initial coil mask")
    axes[1].imshow(final_mask, origin="lower", extent=extent, aspect="auto", cmap="gray_r")
    axes[1].set_title("optimized coil mask")
    clipped = np.clip(final_sdf, -5.0, 20.0)
    image = axes[2].imshow(clipped, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    axes[2].contour(surrogate.r_np, surrogate.z_np, final_sdf, levels=[0.0], colors="black", linewidths=0.8)
    fig.colorbar(image, ax=axes[2], label="union SDF [grid cells]")
    axes[2].set_title("optimized union SDF")
    for axis in axes:
        axis.set(xlabel="r", ylabel="z")
    fig.savefig(out_dir / "optimized_coil_structure_sdf.png", dpi=180)
    plt.close(fig)


def _plot_profiles(
    surrogate: FrozenSurrogate,
    initial_profile: np.ndarray,
    final_profile: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    axis.plot(surrogate.radial_np, initial_profile / 1.0e17, label="initial")
    axis.plot(surrogate.radial_np, final_profile / 1.0e17, label="optimized")
    axis.set(xlabel="wafer radius r", ylabel="wafer-near density [1e17 m^-3]", title="Wafer-near radial density profile")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(out_dir / "wafer_density_radial_profile.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    run_dir = args.run_dir.resolve()
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset_root = (args.dataset_root or Path(resolved["dataset"]["root"])).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = Bounds()
    surrogate = FrozenSurrogate(
        run_dir=run_dir,
        dataset_root=dataset_root,
        protocol=str(args.protocol),
        wafer_layers=int(args.wafer_layers),
    )
    structures_by_count = _load_structure_cases(dataset_root)
    counts = [int(value) for value in args.coil_counts if int(value) in structures_by_count]
    if args.smoke:
        counts = [4]
    steps = 2 if args.smoke else int(args.steps)

    training_masks: list[np.ndarray] = []
    training_hashes: set[str] = set()
    for count, structures in structures_by_count.items():
        for structure in structures:
            row = {**structure["geometry"], "pp": bounds.pp[0], "pp0": bounds.pp0[0]}
            logits = torch.from_numpy(_encode(row, count=count, bounds=bounds))[None].to(surrogate.device)
            decoded = _decode(logits, count=count, bounds=bounds)
            with torch.no_grad():
                mask, _ = surrogate.sdf_mask(decoded, count=count, tau_px=0.5)
            mask_np = mask[0].detach().cpu().numpy()
            training_masks.append(mask_np)
            training_hashes.add(_mask_hash(mask_np))

    history: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    for count in counts:
        structures = structures_by_count[count]
        if args.smoke:
            structures = structures[:2]
        initial_rows: list[dict[str, float]] = []
        initial_case_ids: list[str] = []
        # Model-guided seed selection: retain the best observed process pair
        # for every training structure of this coil count.
        for structure in structures:
            best: tuple[float, dict[str, float], str] | None = None
            for operation in structure["operations"]:
                row = {**structure["geometry"], "pp": operation["pp"], "pp0": operation["pp0"]}
                logits = torch.from_numpy(_encode(row, count=count, bounds=bounds))[None].to(surrogate.device)
                decoded = _decode(logits, count=count, bounds=bounds)
                with torch.no_grad():
                    fields, _, _ = surrogate.predict(decoded, count=count, tau_px=0.5)
                    score = float(surrogate.density_metrics(fields)["harmonic"][0].cpu())
                if best is None or score > best[0]:
                    best = (score, row, str(operation["case_id"]))
            assert best is not None
            initial_rows.append(best[1])
            initial_case_ids.append(best[2])

        logits = torch.nn.Parameter(
            torch.from_numpy(np.stack([_encode(row, count=count, bounds=bounds) for row in initial_rows])).to(
                surrogate.device
            )
        )
        optimizer = torch.optim.Adam([logits], lr=float(args.lr))
        best_score = np.full((len(initial_rows),), -np.inf, dtype=np.float64)
        best_logits = logits.detach().clone()
        for step in range(steps):
            fraction = 0.0 if steps <= 1 else float(step) / float(steps - 1)
            tau = 2.0 * (0.5 / 2.0) ** fraction
            optimizer.zero_grad(set_to_none=True)
            decoded = _decode(logits, count=count, bounds=bounds)
            fields, _, _ = surrogate.predict(decoded, count=count, tau_px=tau)
            metrics = surrogate.density_metrics(fields)
            loss_each = -torch.log(torch.clamp(metrics["harmonic"] / 1.0e17, min=1.0e-12))
            loss = torch.mean(loss_each)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], max_norm=10.0)
            current_logits = logits.detach().clone()
            with torch.no_grad():
                values = metrics["harmonic"].detach().cpu().numpy()
                for index, value in enumerate(values):
                    if float(value) > best_score[index]:
                        best_score[index] = float(value)
                        best_logits[index] = current_logits[index]
                    design = _float_design(decoded, index, count)
                    history.append(
                        {
                            "nncoil": count,
                            "restart": index,
                            "seed_case_id": initial_case_ids[index],
                            "step": step,
                            "tau_px": tau,
                            "loss": float(loss_each[index].cpu()),
                            "harmonic_density": float(metrics["harmonic"][index].cpu()),
                            "mean_density": float(metrics["mean"][index].cpu()),
                            "density_std": float(metrics["std"][index].cpu()),
                            "density_cv": float(metrics["cv"][index].cpu()),
                            "minimum_density": float(metrics["minimum"][index].cpu()),
                            **design,
                        }
                    )
            optimizer.step()

        # Include the state produced by the final optimizer update in the
        # candidate search even though no additional history row is needed.
        decoded_last = _decode(logits, count=count, bounds=bounds)
        with torch.no_grad():
            last_fields, _, _ = surrogate.predict(decoded_last, count=count, tau_px=0.5)
            last_metrics = surrogate.density_metrics(last_fields)
        for index, value in enumerate(last_metrics["harmonic"].detach().cpu().numpy()):
            if float(value) > best_score[index]:
                best_score[index] = float(value)
                best_logits[index] = logits.detach()[index]

        decoded_best = _decode(best_logits, count=count, bounds=bounds)
        with torch.no_grad():
            fields, masks, sdfs = surrogate.predict(decoded_best, count=count, tau_px=0.5)
            metrics = surrogate.density_metrics(fields)
        for index in range(len(initial_rows)):
            mask_np = masks[index].detach().cpu().numpy()
            nearest_xor = min(float(np.mean((mask_np > 0.5) != (train > 0.5))) for train in training_masks)
            design = _float_design(decoded_best, index, count)
            final_candidates.append(
                {
                    "nncoil": count,
                    "restart": index,
                    "seed_case_id": initial_case_ids[index],
                    "harmonic_density": float(metrics["harmonic"][index].cpu()),
                    "mean_density": float(metrics["mean"][index].cpu()),
                    "density_std": float(metrics["std"][index].cpu()),
                    "density_cv": float(metrics["cv"][index].cpu()),
                    "minimum_density": float(metrics["minimum"][index].cpu()),
                    "unique_structure": _mask_hash(mask_np) not in training_hashes,
                    "nearest_training_mask_xor_fraction": nearest_xor,
                    **design,
                    "_logits": best_logits[index].detach().clone(),
                }
            )

    ranked = sorted(final_candidates, key=lambda row: float(row["harmonic_density"]), reverse=True)
    unique = [row for row in ranked if bool(row["unique_structure"])]
    best = unique[0] if unique else ranked[0]
    best_count = int(best["nncoil"])
    final_logits = best["_logits"][None]
    final_decoded = _decode(final_logits, count=best_count, bounds=bounds)
    initial_row = next(
        {**structure["geometry"], "pp": operation["pp"], "pp0": operation["pp0"]}
        for structure in structures_by_count[best_count]
        for operation in structure["operations"]
        if str(operation["case_id"]) == str(best["seed_case_id"])
    )
    initial_logits = torch.from_numpy(_encode(initial_row, count=best_count, bounds=bounds))[None].to(surrogate.device)
    initial_decoded = _decode(initial_logits, count=best_count, bounds=bounds)
    initial_effective = _float_design(initial_decoded, 0, best_count)
    with torch.no_grad():
        initial_fields_t, initial_mask_t, _ = surrogate.predict(initial_decoded, count=best_count, tau_px=0.5)
        final_fields_t, final_mask_t, final_sdf_t = surrogate.predict(final_decoded, count=best_count, tau_px=0.5)
        initial_metrics = surrogate.density_metrics(initial_fields_t)
        final_metrics = surrogate.density_metrics(final_fields_t)
    initial_fields = {key: value[0].cpu().numpy() for key, value in initial_fields_t.items()}
    final_fields = {key: value[0].cpu().numpy() for key, value in final_fields_t.items()}
    initial_profile = initial_metrics["profile"][0].cpu().numpy()
    final_profile = final_metrics["profile"][0].cpu().numpy()

    public_candidates = [{key: value for key, value in row.items() if key != "_logits"} for row in ranked]
    _write_csv(out_dir / "optimization_history.csv", history)
    _write_csv(out_dir / "final_candidates.csv", public_candidates)
    best_by_count = []
    for count in counts:
        rows = [row for row in public_candidates if int(row["nncoil"]) == count]
        if rows:
            best_by_count.append(max(rows, key=lambda row: float(row["harmonic_density"])))
    _write_csv(out_dir / "best_by_coil_count.csv", best_by_count)
    _plot_history(history, out_dir)
    _plot_spatial(surrogate, initial_fields, final_fields, out_dir)
    _plot_structure(
        surrogate,
        initial_mask_t[0].cpu().numpy(),
        final_mask_t[0].cpu().numpy(),
        final_sdf_t[0].cpu().numpy(),
        out_dir,
    )
    _plot_profiles(surrogate, initial_profile, final_profile, out_dir)

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "device": str(surrogate.device),
        "objective": "maximize_area_weighted_harmonic_mean_quasineutral_density",
        "wafer_region": surrogate.selector_meta,
        "best": {key: value for key, value in best.items() if key != "_logits"},
        "initial": {
            **initial_effective,
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
    lines = [
        "# ICP SDF-guided harmonic-density optimization",
        "",
        f"- device: `{summary['device']}`",
        f"- best coil count: `{best_count}`",
        f"- harmonic density: `{float(best['harmonic_density']):.6e}`",
        f"- mean density: `{float(best['mean_density']):.6e}`",
        f"- density CV: `{float(best['density_cv']):.6f}`",
        f"- unique structure: `{bool(best['unique_structure'])}`",
        "",
        "## Figures",
        "",
        "- [Optimization loss history](optimization_loss_history.png)",
        "- [Density spatial distribution](density_spatial_distribution.png)",
        "- [Optimized coil structure and SDF](optimized_coil_structure_sdf.png)",
        "- [Wafer-near radial density profile](wafer_density_radial_profile.png)",
        "",
        "## Tables",
        "",
        "- [Optimization history](optimization_history.csv)",
        "- [Final candidates](final_candidates.csv)",
        "- [Best by coil count](best_by_coil_count.csv)",
        "- [Summary](summary.json)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
