from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.preprocessing.spatial_features import (
    apply_coord_feature_scaling,
    apply_distance_transform,
    part_sdf_summary_maps_from_stack,
)


STRUCTURE_CHANNELS = (
    "part_sdf_nearest",
    "part_sdf_second",
    "part_gap_proxy",
    "solid_proximity",
)


@dataclass(frozen=True)
class Bounds:
    llcoil: tuple[float, float] = (0.5, 1.5)
    rrc: tuple[float, float] = (2.0, 10.0)
    rrce: tuple[float, float] = (20.0, 30.0)
    zzc: tuple[float, float] = (0.0, 5.0)
    min_gap_frac: float = 0.2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent differentiable ICP coil-series optimization through one frozen grid model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--model", default="u_no", choices=("u_no", "unet"))
    parser.add_argument("--protocol", default="structure_holdout")
    parser.add_argument("--reference-case-id", default="case_g002_op01")
    parser.add_argument("--coil-counts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--density-ratio", type=float, default=0.95)
    parser.add_argument("--density-penalty", type=float, default=10.0)
    parser.add_argument("--mid-height-band-px", type=int, default=2)
    parser.add_argument("--radial-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_reference(dataset_root: Path, case_id: str) -> tuple[dict[str, float], dict[str, float]]:
    with (dataset_root / "index.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("case_id")) != case_id:
                continue
            cond = {"pp": float(row["pp"]), "pp0": float(row["pp0"])}
            geom = {key: float(row[key]) for key in ("nncoil", "llcoil", "rrc", "rrce", "zzc")}
            return cond, geom
    raise KeyError(f"reference case not found: {case_id}")


def _dilate_4(mask: np.ndarray, layers: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=np.float32) > 0.5
    for _ in range(max(1, int(layers))):
        p = np.pad(expanded, 1, mode="constant", constant_values=False)
        expanded = expanded | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return expanded


def _wafer_top_band(wafer_mask: np.ndarray, plasma_mask: np.ndarray, layers: int) -> np.ndarray:
    """Select the topmost in-plasma wafer cells in each radial column."""

    wafer = (np.asarray(wafer_mask, dtype=np.float32) > 0.5) & (np.asarray(plasma_mask, dtype=np.float32) > 0.5)
    out = np.zeros_like(wafer, dtype=bool)
    band = max(1, int(layers))
    for col in range(wafer.shape[1]):
        active = np.flatnonzero(wafer[:, col])
        if active.size == 0:
            continue
        top = int(active[-1])
        out[max(0, top - band + 1) : top + 1, col] = wafer[max(0, top - band + 1) : top + 1, col]
    return out


def _plasma_midplane_core_band(
    plasma_mask: np.ndarray,
    r_grid: np.ndarray,
    *,
    band_px: int,
    radial_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a fixed mid-height core band without relying on a wafer label."""

    plasma = np.asarray(plasma_mask, dtype=np.float32) > 0.5
    radius = np.asarray(r_grid, dtype=np.float64)
    if plasma.shape != radius.shape:
        raise ValueError("plasma mask and radial grid must have the same shape")
    active_rows = np.flatnonzero(np.any(plasma, axis=1))
    if active_rows.size == 0:
        raise ValueError("plasma mask is empty")
    fraction = float(radial_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("radial_fraction must be in (0, 1]")
    middle = 0.5 * (float(active_rows[0]) + float(active_rows[-1]))
    mid_row = int(active_rows[np.argmin(np.abs(active_rows.astype(np.float64) - middle))])
    half_band = max(0, int(band_px))
    y0 = max(int(active_rows[0]), mid_row - half_band)
    y1 = min(int(active_rows[-1]) + 1, mid_row + half_band + 1)
    max_radius = float(np.max(radius[plasma]))
    selector = np.zeros_like(plasma, dtype=bool)
    selector[y0:y1] = plasma[y0:y1] & (radius[y0:y1] <= fraction * max_radius)
    if not np.any(selector):
        raise ValueError("plasma mid-height core selector is empty")
    return selector, {
        "mid_row": float(mid_row),
        "row_start": float(y0),
        "row_end_exclusive": float(y1),
        "radial_fraction": fraction,
        "max_plasma_radius": max_radius,
        "sample_count": float(np.count_nonzero(selector)),
    }


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(clipped / (1.0 - clipped))


def decode_design(logits: torch.Tensor, *, count: int, bounds: Bounds) -> dict[str, torch.Tensor]:
    u = torch.sigmoid(logits)
    ll = bounds.llcoil[0] + u[0] * (bounds.llcoil[1] - bounds.llcoil[0])
    rrc = bounds.rrc[0] + u[1] * (bounds.rrc[1] - bounds.rrc[0])
    minimum_end = torch.maximum(
        torch.as_tensor(bounds.rrce[0], device=logits.device, dtype=logits.dtype),
        rrc + float(count) * (1.0 + bounds.min_gap_frac) * ll,
    )
    rrce = minimum_end + u[2] * (bounds.rrce[1] - minimum_end)
    zzc = bounds.zzc[0] + u[3] * (bounds.zzc[1] - bounds.zzc[0])
    return {"llcoil": ll, "rrc": rrc, "rrce": rrce, "zzc": zzc}


def encode_design(geom: dict[str, float], *, count: int, bounds: Bounds) -> np.ndarray:
    ll_u = (float(geom["llcoil"]) - bounds.llcoil[0]) / (bounds.llcoil[1] - bounds.llcoil[0])
    rrc_u = (float(geom["rrc"]) - bounds.rrc[0]) / (bounds.rrc[1] - bounds.rrc[0])
    minimum_end = max(bounds.rrce[0], float(geom["rrc"]) + count * (1.0 + bounds.min_gap_frac) * float(geom["llcoil"]))
    denom = max(bounds.rrce[1] - minimum_end, 1.0e-9)
    rrce_u = (float(geom["rrce"]) - minimum_end) / denom
    zzc_u = (float(geom["zzc"]) - bounds.zzc[0]) / (bounds.zzc[1] - bounds.zzc[0])
    return _logit(np.asarray([ll_u, rrc_u, rrce_u, zzc_u], dtype=np.float64)).astype(np.float32)


def _design_to_float(design: dict[str, torch.Tensor], *, count: int) -> dict[str, float]:
    return {"nncoil": float(count), **{key: float(value.detach().cpu()) for key, value in design.items()}}


def differentiable_part_features(
    *,
    r_grid: torch.Tensor,
    z_grid: torch.Tensor,
    dr: float,
    dz: float,
    design: dict[str, torch.Tensor],
    count: int,
    proximity_tau: float = 4.0,
) -> dict[str, torch.Tensor]:
    ll = design["llcoil"]
    rrc = design["rrc"]
    rrce = design["rrce"]
    zzc = design["zzc"]
    pitch = (rrce - rrc) / float(count)
    z_center = 14.0 + zzc + 0.5 * ll
    r0 = r_grid[0, 0]
    z0 = z_grid[0, 0]
    x_index = torch.arange(r_grid.shape[1], device=r_grid.device, dtype=r_grid.dtype)[None, :].expand_as(r_grid)
    y_index = torch.arange(z_grid.shape[0], device=z_grid.device, dtype=z_grid.dtype)[:, None].expand_as(z_grid)
    distances: list[torch.Tensor] = []
    for idx in range(int(count)):
        r_center = rrc + float(idx) * pitch + 0.5 * ll
        q_r = torch.abs(r_grid - r_center) / float(dr) - 0.5 * ll / float(dr)
        q_z = torch.abs(z_grid - z_center) / float(dz) - 0.5 * ll / float(dz)
        outside = torch.relu(q_r) + torch.relu(q_z)
        inside = torch.minimum(torch.relu(-q_r), torch.relu(-q_z))
        soft_distance = torch.where((q_r <= 0.0) & (q_z <= 0.0), inside, outside)

        # Match the inclusive rectangular raster and Manhattan SDF used by the
        # hard inference path in the forward pass. Gradients come from the
        # continuous analytic distance above (straight-through estimator).
        r_min = r_center - 0.5 * ll
        r_max = r_center + 0.5 * ll
        z_min = z_center - 0.5 * ll
        z_max = z_center + 0.5 * ll
        left = torch.ceil((r_min - r0) / float(dr))
        right = torch.floor((r_max - r0) / float(dr))
        bottom = torch.ceil((z_min - z0) / float(dz))
        top = torch.floor((z_max - z0) / float(dz))
        hard_outside = (
            torch.relu(left - x_index)
            + torch.relu(x_index - right)
            + torch.relu(bottom - y_index)
            + torch.relu(y_index - top)
        )
        hard_inside = torch.minimum(
            torch.minimum(x_index - left, right - x_index),
            torch.minimum(y_index - bottom, top - y_index),
        )
        hard_distance = torch.where(
            (x_index >= left) & (x_index <= right) & (y_index >= bottom) & (y_index <= top),
            hard_inside,
            hard_outside,
        )
        distances.append(soft_distance + (hard_distance - soft_distance).detach())
    ordered = torch.topk(torch.stack(distances, dim=0), k=2, dim=0, largest=False).values
    nearest = ordered[0]
    second = ordered[1]
    gap = torch.relu(second - nearest)
    proximity = torch.exp(-torch.relu(nearest) / float(proximity_tau))
    return {
        "part_sdf_nearest": nearest,
        "part_sdf_second": second,
        "part_gap_proxy": gap,
        "solid_proximity": proximity,
    }


def hard_part_features(
    *,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    geom: dict[str, float],
    count: int,
) -> dict[str, np.ndarray]:
    ll = float(geom["llcoil"])
    rrc = float(geom["rrc"])
    rrce = float(geom["rrce"])
    z_min = 14.0 + float(geom["zzc"])
    pitch = (rrce - rrc) / float(count)
    masks: list[np.ndarray] = []
    for idx in range(6):
        if idx >= count:
            masks.append(np.zeros_like(r_grid, dtype=np.float32))
            continue
        r_min = rrc + float(idx) * pitch
        mask = (
            (r_grid >= r_min)
            & (r_grid <= r_min + ll)
            & (z_grid >= z_min)
            & (z_grid <= z_min + ll)
        )
        masks.append(mask.astype(np.float32))
    return part_sdf_summary_maps_from_stack(np.stack(masks, axis=0))


class FrozenGridSurrogate:
    def __init__(
        self,
        *,
        run_dir: Path,
        model_name: str,
        protocol: str,
        dataset_root: Path,
        cond: dict[str, float],
        mid_height_band_px: int,
        radial_fraction: float,
    ):
        checkpoint_dir = run_dir / "models" / model_name / "eval_protocol" / protocol / "checkpoints"
        self.model = load_checkpoint(checkpoint_dir)
        if not hasattr(self.model, "net") or not hasattr(self.model, "torch"):
            raise TypeError("differentiable optimization requires a torch grid model exposing .net")
        self.model._ensure_net_device()
        self.device = self.model.device
        self.dtype = torch.float32
        self.model.net.eval()
        for parameter in self.model.net.parameters():
            parameter.requires_grad_(False)

        bundle = RunBundleLoader.load(run_dir, model=self.model)
        split_dir = run_dir / "preprocessing" / "scalers" / "by_split" / protocol
        self.cond_scaler = _load_json(split_dir / "cond_scaler.json")
        self.coord_scaler = _load_json(split_dir / "coord_feature_scaler.json")
        self.y_scalers = _load_json(split_dir / "y_scalers.json")
        self.target_transforms = dict(self.cond_scaler.get("target_transforms", {}))
        self.channels = list(self.model.input_feature_channels)
        self.output_keys = list(self.model.output_keys)

        static_pack = np.load(run_dir / "preprocessing" / "features" / "static_spatial_feature_pack.npz", allow_pickle=True)
        static_names = [str(v) for v in static_pack["channels"].tolist()]
        static_map = {name: np.asarray(static_pack["data"][i], dtype=np.float32) for i, name in enumerate(static_names)}
        self.r_grid_np = static_map["x"]
        self.z_grid_np = static_map["y"]
        self.h, self.w = self.r_grid_np.shape
        self.dr = float(np.median(np.diff(self.r_grid_np[0])))
        self.dz = float(np.median(np.diff(self.z_grid_np[:, 0])))

        raw = np.zeros((self.h, self.w, len(self.channels)), dtype=np.float32)
        for idx, name in enumerate(self.channels):
            if name in static_map:
                raw[..., idx] = static_map[name]
        distance_cfg = dict(self.coord_scaler.get("distance_transform_effective", {}))
        flat, _ = apply_distance_transform(raw.reshape(-1, len(self.channels)), channels=self.channels, cfg=distance_cfg)
        flat, _, _ = apply_coord_feature_scaling(
            flat,
            channels=self.channels,
            coord_feature_scaler_artifact=self.coord_scaler,
            distance_transform_cfg=distance_cfg,
        )
        self.static_scaled = torch.from_numpy(np.moveaxis(flat.reshape(self.h, self.w, -1), -1, 0)).to(self.device)
        self.structure_indices = {name: self.channels.index(name) for name in STRUCTURE_CHANNELS}
        self.structure_scalers = {name: dict(self.coord_scaler["channels"][name]) for name in STRUCTURE_CHANNELS}
        self.r_grid = torch.from_numpy(self.r_grid_np).to(self.device)
        self.z_grid = torch.from_numpy(self.z_grid_np).to(self.device)

        cond_raw = np.asarray([[cond["pp"], cond["pp0"]]], dtype=np.float32)
        median = np.asarray(self.cond_scaler["median"], dtype=np.float32)
        iqr = np.asarray(self.cond_scaler["iqr"], dtype=np.float32)
        cond_scaled = (cond_raw - median.reshape(1, -1)) / iqr.reshape(1, -1)
        self.cond_scaled = torch.from_numpy(cond_scaled).to(self.device)

        plasma = np.load(dataset_root / "geometry" / "mask_plasma.npy").astype(np.float32) > 0.5
        selector, region = _plasma_midplane_core_band(
            plasma,
            self.r_grid_np,
            band_px=mid_height_band_px,
            radial_fraction=radial_fraction,
        )
        self.evaluation_selector_np = selector
        self.evaluation_selector = torch.from_numpy(selector).to(self.device)
        self.evaluation_region = region
        selector_counts = selector.sum(axis=0).astype(np.float32)
        valid_columns = selector_counts > 0
        self.profile_counts = torch.from_numpy(selector_counts[valid_columns]).to(self.device)
        self.profile_columns_np = valid_columns
        self.profile_columns = torch.from_numpy(valid_columns).to(self.device)
        radial_weights = self.r_grid_np[int(region["mid_row"]), valid_columns].astype(np.float32)
        self.radial_weights_np = radial_weights.astype(np.float64)
        self.radial_weights = torch.from_numpy(radial_weights).to(self.device)
        self.plasma_np = plasma
        self.transform_bundle = bundle.transform_bundle(protocol, require_protocol=True)

    def _scaled_input(self, raw_structure: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial = self.static_scaled.clone()
        for name, raw in raw_structure.items():
            scaler = self.structure_scalers[name]
            if str(scaler.get("type", "none")) == "zscore":
                value = (raw - float(scaler["mean"][0])) / float(scaler["std"][0])
            else:
                value = raw
            spatial[self.structure_indices[name]] = value
        cond_map = self.cond_scaled[:, :, None, None].expand(1, -1, self.h, self.w)
        return torch.cat([cond_map, spatial[None, ...]], dim=1)

    def _inverse_target(self, raw: torch.Tensor, key: str) -> torch.Tensor:
        scaler = self.y_scalers[key]
        if str(scaler["type"]) == "zscore":
            transformed = raw * float(scaler["std"][0]) + float(scaler["mean"][0])
        elif str(scaler["type"]) == "robust":
            transformed = raw * float(scaler["iqr"][0]) + float(scaler["median"][0])
        else:
            transformed = raw
        spec = dict(self.target_transforms[key])
        clip = dict(spec.get("clip", {}))
        if str(clip.get("mode", "none")) == "quantile":
            transformed = torch.clamp(transformed, float(clip["clip_low"]), float(clip["clip_high"]))
        mode = str(spec.get("value_transform", "identity"))
        if mode in {"log10", "log10_floor"}:
            return torch.pow(torch.as_tensor(10.0, device=raw.device), transformed)
        if mode == "log1p":
            return torch.clamp(torch.expm1(transformed), min=float(spec.get("floor", 0.0) or 0.0))
        if mode == "signed_log1p":
            return torch.sign(transformed) * torch.expm1(torch.abs(transformed))
        return transformed

    def _forward_tensor(self, model_input: torch.Tensor) -> torch.Tensor:
        forward = getattr(self.model, "_torch_forward", None)
        if callable(forward):
            return forward(model_input)
        return self.model.net(model_input)

    def predict_soft(self, *, design: dict[str, torch.Tensor], count: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = differentiable_part_features(
            r_grid=self.r_grid,
            z_grid=self.z_grid,
            dr=self.dr,
            dz=self.dz,
            design=design,
            count=count,
        )
        raw = self._forward_tensor(self._scaled_input(features))[0]
        fields = {key: self._inverse_target(raw[idx], key) for idx, key in enumerate(self.output_keys)}
        return fields, features

    def predict_hard(self, *, geom: dict[str, float], count: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        features_np = hard_part_features(r_grid=self.r_grid_np, z_grid=self.z_grid_np, geom=geom, count=count)
        features = {key: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(self.device) for key, value in features_np.items()}
        with torch.no_grad():
            raw = self._forward_tensor(self._scaled_input(features))[0]
            fields = {
                key: self._inverse_target(raw[idx], key).detach().cpu().numpy().astype(np.float32)
                for idx, key in enumerate(self.output_keys)
            }
        return fields, features_np

    def _torch_radial_profile(self, field: torch.Tensor) -> torch.Tensor:
        selected = torch.where(self.evaluation_selector, field, torch.zeros_like(field))
        return torch.sum(selected, dim=0)[self.profile_columns] / self.profile_counts

    def radial_profile(self, fields: dict[str, np.ndarray], key: str = "ni") -> tuple[np.ndarray, np.ndarray]:
        field = np.asarray(fields[key], dtype=np.float64)
        selected = np.where(self.evaluation_selector_np, field, np.nan)
        with np.errstate(invalid="ignore"):
            profile = np.nanmean(selected[:, self.profile_columns_np], axis=0)
        row = int(self.evaluation_region["mid_row"])
        return self.r_grid_np[row, self.profile_columns_np].astype(np.float64), profile

    @staticmethod
    def _weighted_metrics(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        values64 = np.asarray(values, dtype=np.float64)
        weights64 = np.asarray(weights, dtype=np.float64)
        mean = float(np.sum(weights64 * values64) / np.sum(weights64))
        cv = float(np.sqrt(np.sum(weights64 * np.square(values64 - mean)) / np.sum(weights64)) / max(abs(mean), 1.0e-30))
        return mean, cv

    def objective(self, fields: dict[str, torch.Tensor], *, density_ref: torch.Tensor, density_ratio: float, penalty: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        profile = self._torch_radial_profile(fields["ni"])
        weights = self.radial_weights
        profile_scale = torch.clamp(torch.sum(weights * torch.abs(profile)).detach() / torch.sum(weights), min=1.0e-30)
        scaled = profile / profile_scale
        density_mean_scaled = torch.sum(weights * scaled) / torch.sum(weights)
        uniformity = torch.sqrt(torch.sum(weights * torch.square(scaled - density_mean_scaled)) / torch.sum(weights)) / (
            torch.abs(density_mean_scaled) + 1.0e-12
        )
        density_mean = density_mean_scaled * profile_scale
        ratio = density_mean / torch.clamp(density_ref, min=1.0e-30)
        violation = torch.relu(float(density_ratio) - ratio)
        loss = uniformity + float(penalty) * violation.square()
        return loss, {"uniformity": uniformity, "density_mean": density_mean, "density_ratio": ratio, "violation": violation}

    def hard_metrics(self, fields: dict[str, np.ndarray], *, density_ref: float) -> dict[str, float]:
        _, ni_profile = self.radial_profile(fields, "ni")
        _, te_profile = self.radial_profile(fields, "Te")
        density_mean, uniformity = self._weighted_metrics(ni_profile, self.radial_weights_np)
        temperature_mean, temperature_uniformity = self._weighted_metrics(te_profile, self.radial_weights_np)
        plasma_ni = np.asarray(fields["ni"], dtype=np.float64)[self.plasma_np]
        return {
            "uniformity": uniformity,
            "density_mean": density_mean,
            "density_ratio": density_mean / max(float(density_ref), 1.0e-30),
            "temperature_mean": temperature_mean,
            "temperature_uniformity": temperature_uniformity,
            "plasma_density_mean": float(np.mean(plasma_ni)),
        }


def _feature_agreement(soft: dict[str, torch.Tensor], hard: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in STRUCTURE_CHANNELS:
        a = soft[name].detach().cpu().numpy().astype(np.float64).reshape(-1)
        b = np.asarray(hard[name], dtype=np.float64).reshape(-1)
        out[f"{name}_mae"] = float(np.mean(np.abs(a - b)))
        out[f"{name}_corr"] = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return out


def _relative_rms(delta: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.asarray(delta, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        d = d[selected]
        ref = ref[selected]
    scale = max(float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30)
    return float(np.sqrt(np.mean(np.square(d))) / scale)


def _audit_structure_sensitivity(
    surrogate: FrozenGridSurrogate,
    *,
    reference_geom: dict[str, float],
    base_fields: dict[str, np.ndarray],
    base_features: dict[str, np.ndarray],
    bounds: Bounds,
) -> list[dict[str, Any]]:
    count = int(round(reference_geom["nncoil"]))
    encoded = encode_design(reference_geom, count=count, bounds=bounds)
    rows: list[dict[str, Any]] = []
    for index, parameter in enumerate(("llcoil", "rrc", "rrce", "zzc")):
        for level, unit_value in (("low", 0.1), ("high", 0.9)):
            varied = encoded.copy()
            varied[index] = float(_logit(np.asarray([unit_value]))[0])
            design = decode_design(torch.from_numpy(varied).to(surrogate.device), count=count, bounds=bounds)
            geom = _design_to_float(design, count=count)
            fields, features = surrogate.predict_hard(geom=geom, count=count)
            feature_delta = np.stack(
                [np.asarray(features[key], dtype=np.float64) - np.asarray(base_features[key], dtype=np.float64) for key in STRUCTURE_CHANNELS],
                axis=0,
            )
            feature_ref = np.stack([np.asarray(base_features[key], dtype=np.float64) for key in STRUCTURE_CHANNELS], axis=0)
            _, base_profile = surrogate.radial_profile(base_fields, "ni")
            _, profile = surrogate.radial_profile(fields, "ni")
            rows.append(
                {
                    "varied_parameter": parameter,
                    "level": level,
                    **geom,
                    "feature_relative_rms": _relative_rms(feature_delta, feature_ref),
                    "feature_changed_fraction": float(np.mean(np.any(np.abs(feature_delta) > 1.0e-6, axis=0))),
                    "plasma_ni_relative_rms": _relative_rms(fields["ni"] - base_fields["ni"], base_fields["ni"], surrogate.plasma_np),
                    "core_profile_ni_relative_rms": _relative_rms(profile - base_profile, base_profile),
                }
            )
    return rows


def _plot_structure_sensitivity(rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [f"{row['varied_parameter']}:{row['level']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)
    axes[0].bar(x, [float(row["feature_relative_rms"]) for row in rows])
    axes[0].set_ylabel("structure-feature relative RMS")
    axes[1].bar(x - 0.18, [float(row["plasma_ni_relative_rms"]) for row in rows], width=0.36, label="plasma field")
    axes[1].bar(x + 0.18, [float(row["core_profile_ni_relative_rms"]) for row in rows], width=0.36, label="core radial profile")
    axes[1].set_ylabel("predicted ni relative RMS")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "structure_sensitivity.png", dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    counts = sorted({int(row["nncoil"]) for row in history})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for count in counts:
        for restart in sorted({int(r["restart"]) for r in history if int(r["nncoil"]) == count}):
            rows = [r for r in history if int(r["nncoil"]) == count and int(r["restart"]) == restart]
            x = [int(r["step"]) for r in rows]
            axes[0].plot(x, [float(r["loss"]) for r in rows], alpha=0.7, label=f"n={count} r={restart}")
            axes[1].plot(x, [float(r["uniformity"]) for r in rows], alpha=0.7)
            axes[2].plot(x, [float(r["density_ratio"]) for r in rows], alpha=0.7)
    axes[0].set_title("total loss")
    axes[1].set_title("area-weighted radial ni CV")
    axes[2].set_title("area-weighted mean ni / baseline")
    axes[2].axhline(0.95, color="black", ls="--", lw=1)
    for ax in axes:
        ax.set_xlabel("Adam step")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.savefig(out_dir / "loss_history.png", dpi=180)
    plt.close(fig)


def _plot_parameter_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    keys = ("llcoil", "rrc", "rrce", "zzc")
    for ax, key in zip(axes.reshape(-1), keys, strict=True):
        for count in sorted({int(row["nncoil"]) for row in history}):
            best_restart = min(
                {int(r["restart"]) for r in history if int(r["nncoil"]) == count},
                key=lambda restart: min(float(r["loss"]) for r in history if int(r["nncoil"]) == count and int(r["restart"]) == restart),
            )
            rows = [r for r in history if int(r["nncoil"]) == count and int(r["restart"]) == best_restart]
            ax.plot([int(r["step"]) for r in rows], [float(r[key]) for r in rows], label=f"n={count}")
        ax.set_title(key)
        ax.set_xlabel("Adam step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(out_dir / "parameter_history.png", dpi=180)
    plt.close(fig)


def _plot_spatial(base: dict[str, np.ndarray], best: dict[str, np.ndarray], surrogate: FrozenGridSurrogate, out_dir: Path) -> None:
    plasma = surrogate.plasma_np
    fields = [
        ("ni", np.where(plasma, base["ni"], np.nan), np.where(plasma, best["ni"], np.nan)),
        ("ne", np.where(plasma, base["ne"], np.nan), np.where(plasma, best["ne"], np.nan)),
        ("Te", np.where(plasma, base["Te"], np.nan), np.where(plasma, best["Te"], np.nan)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.0), constrained_layout=True)
    extent = [float(surrogate.r_grid_np.min()), float(surrogate.r_grid_np.max()), float(surrogate.z_grid_np.min()), float(surrogate.z_grid_np.max())]
    for row, (name, before, after) in enumerate(fields):
        vmin = float(min(np.nanmin(before), np.nanmin(after)))
        vmax = float(max(np.nanmax(before), np.nanmax(after)))
        diff = after - before
        dmax = max(float(np.nanmax(np.abs(diff))), 1.0e-30)
        im0 = axes[row, 0].imshow(before, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(after, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        im2 = axes[row, 2].imshow(diff, origin="lower", extent=extent, aspect="auto", cmap="coolwarm", vmin=-dmax, vmax=dmax)
        axes[row, 0].set_ylabel(f"{name}\nz")
        axes[row, 0].set_title("baseline")
        axes[row, 1].set_title("optimized")
        axes[row, 2].set_title("optimized - baseline")
        fig.colorbar(im0, ax=axes[row, :2], shrink=0.75)
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.75)
    for ax in axes[-1]:
        ax.set_xlabel("r")
    fig.savefig(out_dir / "spatial_before_after.png", dpi=180)
    plt.close(fig)


def _plot_candidate_summary(base_metrics: dict[str, float], candidates: list[dict[str, Any]], out_dir: Path) -> None:
    ordered = sorted(candidates, key=lambda item: int(item["nncoil"]))
    counts = np.asarray([int(item["nncoil"]) for item in ordered], dtype=np.int64)
    uniformity = np.asarray([float(item["hard_uniformity"]) for item in ordered], dtype=np.float64)
    density_ratio = np.asarray([float(item["hard_density_ratio"]) for item in ordered], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    axes[0].plot(counts, uniformity, marker="o", lw=1.8)
    axes[0].axhline(float(base_metrics["uniformity"]), color="black", ls="--", label="baseline")
    axes[0].set_ylabel("hard midplane-core ni CV")
    axes[0].legend()
    axes[1].plot(counts, density_ratio, marker="o", lw=1.8)
    axes[1].axhline(0.95, color="black", ls="--", label="constraint")
    axes[1].axhline(1.0, color="gray", ls=":", label="baseline")
    axes[1].set_ylabel("hard mean ni / baseline")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("coil count")
        ax.set_xticks(counts)
        ax.grid(alpha=0.25)
    fig.savefig(out_dir / "best_by_coil_count.png", dpi=180)
    plt.close(fig)


def _plot_radial(base: dict[str, np.ndarray], candidates: list[dict[str, Any]], surrogate: FrozenGridSurrogate, out_dir: Path) -> None:
    r, p0 = surrogate.radial_profile(base, "ni")
    base_scale = surrogate._weighted_metrics(p0, surrogate.radial_weights_np)[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(r, p0, color="black", lw=2.2, label="baseline")
    axes[1].plot(r, p0 / base_scale, color="black", lw=2.2, label="baseline")
    for index, candidate in enumerate(candidates):
        rr, pp = surrogate.radial_profile(candidate["fields"], "ni")
        is_best = index == 0
        style = {"color": "tab:red", "lw": 2.3, "alpha": 1.0} if is_best else {"color": "0.65", "lw": 1.0, "alpha": 0.55}
        label = f"best n={int(candidate['nncoil'])}" if is_best else f"n={int(candidate['nncoil'])}"
        axes[0].plot(rr, pp, label=label, **style)
        axes[1].plot(rr, pp / base_scale, label=label, **style)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("ion density ni")
    axes[1].set_ylabel("ni / baseline area-weighted mean")
    for ax in axes:
        ax.set_xlabel("r")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("absolute midplane-core radial profiles")
    axes[1].set_title("baseline-normalized radial profiles")
    fig.savefig(out_dir / "radial_ion_density_profiles.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    dataset_root = Path(args.dataset_root or resolved["dataset"]["root"])
    cond, reference_geom = _read_reference(dataset_root, str(args.reference_case_id))
    bounds = Bounds()
    counts = sorted(set(int(v) for v in args.coil_counts if 2 <= int(v) <= 6))
    if not counts:
        raise ValueError("coil-counts must include at least one value in [2, 6]")
    restarts = 1 if args.smoke else max(1, int(args.restarts))
    steps = min(3, int(args.steps)) if args.smoke else max(1, int(args.steps))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    surrogate = FrozenGridSurrogate(
        run_dir=run_dir,
        model_name=str(args.model),
        protocol=str(args.protocol),
        dataset_root=dataset_root,
        cond=cond,
        mid_height_band_px=int(args.mid_height_band_px),
        radial_fraction=float(args.radial_fraction),
    )
    reference_count = int(round(reference_geom["nncoil"]))
    base_fields, base_features = surrogate.predict_hard(geom=reference_geom, count=reference_count)
    base_metrics_unscaled = surrogate.hard_metrics(base_fields, density_ref=1.0)
    base_ni_mean = float(base_metrics_unscaled["density_mean"])
    density_ref = torch.as_tensor(base_ni_mean, dtype=torch.float32, device=surrogate.device)
    base_metrics = surrogate.hard_metrics(base_fields, density_ref=base_ni_mean)
    sensitivity_rows = _audit_structure_sensitivity(
        surrogate,
        reference_geom=reference_geom,
        base_fields=base_fields,
        base_features=base_features,
        bounds=bounds,
    )

    rng = np.random.default_rng(int(args.seed))
    history: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for count in counts:
        best_for_count: dict[str, Any] | None = None
        for restart in range(restarts):
            if restart == 0 and count == reference_count:
                init = encode_design(reference_geom, count=count, bounds=bounds)
            elif restart == 0:
                init = _logit(np.full(4, 0.5, dtype=np.float64)).astype(np.float32)
            else:
                init = _logit(rng.uniform(0.12, 0.88, size=4)).astype(np.float32)
            logits = torch.nn.Parameter(torch.from_numpy(init).to(surrogate.device))
            optimizer = torch.optim.Adam([logits], lr=float(args.lr))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=float(args.lr) * 0.1)
            best_state: dict[str, Any] | None = None
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                design = decode_design(logits, count=count, bounds=bounds)
                fields, soft_features = surrogate.predict_soft(design=design, count=count)
                loss, metrics = surrogate.objective(
                    fields,
                    density_ref=density_ref,
                    density_ratio=float(args.density_ratio),
                    penalty=float(args.density_penalty),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss for count={count}, restart={restart}, step={step}")
                loss.backward()
                grad_norm = float(torch.linalg.vector_norm(logits.grad).detach().cpu())
                state_logits = logits.detach().clone()
                optimizer.step()
                scheduler.step()
                geom = _design_to_float(design, count=count)
                row = {
                    "nncoil": count,
                    "restart": restart,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "uniformity": float(metrics["uniformity"].detach().cpu()),
                    "density_mean": float(metrics["density_mean"].detach().cpu()),
                    "density_ratio": float(metrics["density_ratio"].detach().cpu()),
                    "violation": float(metrics["violation"].detach().cpu()),
                    "grad_norm": grad_norm,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **geom,
                }
                history.append(row)
                feasible = row["density_ratio"] >= float(args.density_ratio)
                rank = (0 if feasible else 1, row["uniformity"] if feasible else row["loss"])
                if best_state is None or rank < best_state["rank"]:
                    best_state = {
                        "rank": rank,
                        "logits": state_logits,
                        "soft_metrics": dict(row),
                    }
            assert best_state is not None
            best_design = decode_design(best_state["logits"], count=count, bounds=bounds)
            best_geom = _design_to_float(best_design, count=count)
            soft_fields, soft_features = surrogate.predict_soft(design=best_design, count=count)
            hard_fields, hard_features = surrogate.predict_hard(geom=best_geom, count=count)
            hard_metrics = surrogate.hard_metrics(hard_fields, density_ref=base_ni_mean)
            agreement = _feature_agreement(soft_features, hard_features)
            terminal = {
                "nncoil": count,
                "restart": restart,
                **best_geom,
                **{f"soft_{k}": v for k, v in best_state["soft_metrics"].items() if k in {"loss", "uniformity", "density_ratio"}},
                **{f"hard_{k}": v for k, v in hard_metrics.items()},
                **agreement,
            }
            terminal_rows.append(terminal)
            print(
                f"count={count} restart={restart} "
                f"soft_U={terminal['soft_uniformity']:.6g} hard_U={terminal['hard_uniformity']:.6g} "
                f"hard_density_ratio={terminal['hard_density_ratio']:.6g}",
                flush=True,
            )
            feasible_hard = hard_metrics["density_ratio"] >= float(args.density_ratio)
            hard_rank = (0 if feasible_hard else 1, hard_metrics["uniformity"] if feasible_hard else float(terminal["soft_loss"]))
            candidate = {**terminal, "rank": hard_rank, "fields": hard_fields, "features": hard_features}
            if best_for_count is None or hard_rank < best_for_count["rank"]:
                best_for_count = candidate
        assert best_for_count is not None
        final_candidates.append(best_for_count)

    final_candidates.sort(key=lambda item: item["rank"])
    best = final_candidates[0]
    elapsed = time.perf_counter() - started
    field_change = {
        key: {
            "plasma_relative_rms": _relative_rms(best["fields"][key] - base_fields[key], base_fields[key], surrogate.plasma_np),
            "plasma_mean_change_percent": 100.0
            * (float(np.mean(np.asarray(best["fields"][key])[surrogate.plasma_np])) / float(np.mean(np.asarray(base_fields[key])[surrogate.plasma_np])) - 1.0),
        }
        for key in ("ni", "ne", "Te")
    }
    _write_csv(out_dir / "loss_history.csv", history)
    _write_csv(out_dir / "restart_summary.csv", terminal_rows)
    _write_csv(out_dir / "best_by_coil_count.csv", [{k: v for k, v in c.items() if k not in {"rank", "fields", "features"}} for c in final_candidates])
    _write_csv(out_dir / "structure_sensitivity.csv", sensitivity_rows)
    summary = {
        "status": "completed",
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "model": str(args.model),
        "protocol": str(args.protocol),
        "device": str(surrogate.device),
        "reference_case_id": str(args.reference_case_id),
        "condition": cond,
        "reference_geometry": reference_geom,
        "baseline_metrics": base_metrics,
        "best": {k: v for k, v in best.items() if k not in {"rank", "fields", "features"}},
        "elapsed_seconds": float(elapsed),
        "counts": counts,
        "restarts": restarts,
        "steps": steps,
        "density_ratio_constraint": float(args.density_ratio),
        "evaluation_region": surrogate.evaluation_region,
        "objective_definition": "area-weighted radial ni CV in the plasma mid-height core band",
        "structure_sensitivity": {
            "max_feature_relative_rms": max(float(row["feature_relative_rms"]) for row in sensitivity_rows),
            "max_plasma_ni_relative_rms": max(float(row["plasma_ni_relative_rms"]) for row in sensitivity_rows),
            "max_core_profile_ni_relative_rms": max(float(row["core_profile_ni_relative_rms"]) for row in sensitivity_rows),
        },
        "field_change": field_change,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        out_dir / "best_fields.npz",
        **{f"baseline_{key}": value for key, value in base_fields.items()},
        **{f"optimized_{key}": value for key, value in best["fields"].items()},
        evaluation_selector=surrogate.evaluation_selector_np.astype(np.uint8),
        r_grid=surrogate.r_grid_np,
        z_grid=surrogate.z_grid_np,
    )
    _plot_history(history, out_dir)
    _plot_parameter_history(history, out_dir)
    _plot_candidate_summary(base_metrics, final_candidates, out_dir)
    _plot_spatial(base_fields, best["fields"], surrogate, out_dir)
    _plot_radial(base_fields, final_candidates, surrogate, out_dir)
    _plot_structure_sensitivity(sensitivity_rows, out_dir)

    lines = [
        "# Independent differentiable ICP shape optimization",
        "",
        f"- device: `{surrogate.device}`",
        f"- elapsed: `{elapsed:.1f} s`",
        f"- baseline uniformity: `{base_metrics['uniformity']:.6g}`",
        f"- best hard uniformity: `{best['hard_uniformity']:.6g}`",
        f"- uniformity improvement: `{100.0 * (base_metrics['uniformity'] - best['hard_uniformity']) / base_metrics['uniformity']:.2f}%`",
        f"- best hard density ratio: `{best['hard_density_ratio']:.6g}`",
        f"- whole-plasma ni relative RMS change: `{100.0 * field_change['ni']['plasma_relative_rms']:.2f}%`",
        f"- best coil count: `{int(best['nncoil'])}`",
        f"- evaluation: plasma mid-height rows `{int(surrogate.evaluation_region['row_start'])}:{int(surrogate.evaluation_region['row_end_exclusive'])}`, radial fraction `<= {surrogate.evaluation_region['radial_fraction']:.3g}`",
        "",
        "## Best dimensions",
        "",
        f"- llcoil: `{best['llcoil']:.6g}`",
        f"- rrc: `{best['rrc']:.6g}`",
        f"- rrce: `{best['rrce']:.6g}`",
        f"- zzc: `{best['zzc']:.6g}`",
        "",
        "## Figures",
        "",
        "- [Loss history](loss_history.png)",
        "- [Parameter history](parameter_history.png)",
        "- [Best by coil count](best_by_coil_count.png)",
        "- [Radial ion-density profiles](radial_ion_density_profiles.png)",
        "- [Spatial before/after](spatial_before_after.png)",
        "- [Structure sensitivity](structure_sensitivity.png)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
