"""Thin data/physics loss composition layer shared by numpy and torch trainers."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from plasma_surrogate.core.density_contract import resolve_density_key
from plasma_surrogate.core.spatial_regions import build_boundary_type_masks, build_region_masks
from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.losses import (
    build_signed_distance,
    is_effective_signed_distance,
    physics_loss_and_grad,
    sdf_continuous_weight_map,
)
from plasma_surrogate.train.physics_terms import resolve_numpy_terms, resolve_torch_terms
from plasma_surrogate.train.torch_losses import physics_terms_torch, sdf_continuous_weight_map_torch


_COMPONENT_KEYS = ("data", "physics", "poisson", "boundary", "boundary_operator", "rho")


def _empty_components() -> dict[str, float]:
    return {k: 0.0 for k in _COMPONENT_KEYS}


def _resolve_supervised_cfg(loss_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(loss_cfg or {})
    sup = dict(cfg.get("supervised", {}))
    mt = dict(cfg.get("multitask", {}))
    region_weighting_cfg = dict(sup.get("region_weighting", {}))
    region_balance_cfg = dict(sup.get("region_balance", {}))
    if region_weighting_cfg and region_balance_cfg:
        raise ValueError(
            "supervised.region_balance and supervised.region_weighting cannot be specified together; "
            "use supervised.region_balance (region_weighting is deprecated alias)"
        )
    if region_weighting_cfg:
        warnings.warn(
            "supervised.region_weighting is deprecated; use supervised.region_balance",
            DeprecationWarning,
            stacklevel=2,
        )
    if "coord_objective" in sup:
        raise ValueError("supervised.coord_objective is removed")
    if "sample_mean_group_scale" in sup:
        raise ValueError("supervised.sample_mean_group_scale is removed")
    return {
        "type": str(sup.get("type", "mse")).strip().lower(),
        "delta": float(sup.get("delta", 1.0)),
        "delta_by_var": dict(sup.get("delta_by_var", {})),
        "normalization": str(sup.get("normalization", "pixel_mean")).strip().lower(),
        "sample_mean_group_mode": str(sup.get("sample_mean_group_mode", "batch")).strip().lower(),
        "sample_mean_weight_denominator": str(sup.get("sample_mean_weight_denominator", "weighted")).strip().lower(),
        "weighting": str(mt.get("weighting", "fixed")).strip().lower(),
        "fixed_weights_by_var": dict(mt.get("fixed_weights_by_var", {})),
        "sigma_init": dict(mt.get("sigma_init", {})),
        "sigma_clamp": tuple(mt.get("sigma_clamp", [-3.0, 3.0])),
        "region_weighting": region_weighting_cfg,
        "robust_weighting": dict(sup.get("robust_weighting", {})),
        "nan_region_policy": str(sup.get("nan_region_policy", "mask_only")).strip().lower(),
        "sdf_weighting": dict(sup.get("sdf_weighting", {})),
        "chamber_weight_by_var": dict(sup.get("chamber_weight_by_var", {})),
        "global_target_region": str(sup.get("global_target_region", "")).strip().lower(),
        "label_clip_from_scaler": bool(sup.get("label_clip_from_scaler", False)),
        "robust_clip_stats": dict(sup.get("robust_clip_stats", {})),
        "sdf_distance_contract": str(sup.get("sdf_distance_contract", "off")).strip().lower(),
        "chamber_aux": dict(sup.get("chamber_aux", {})),
        "region_balance": region_balance_cfg,
        "boundary_type_weighting": dict(sup.get("boundary_type_weighting", {})),
        "boundary_profile_weighting": dict(sup.get("boundary_profile_weighting", {})),
        "density_positivity_penalty": dict(sup.get("density_positivity_penalty", {})),
        "density_relative_weighting": dict(sup.get("density_relative_weighting", {})),
        "spatial_consistency": dict(sup.get("spatial_consistency", {})),
    }


def _resolve_sigma_for_var(name: str, base_loss: float, cfg: dict[str, Any]) -> float:
    if cfg["weighting"] != "uncertainty":
        return 0.0
    init = cfg.get("sigma_init", {})
    raw = init.get(name, np.log(max(base_loss, 1e-8)))
    lo, hi = cfg.get("sigma_clamp", (-3.0, 3.0))
    return float(np.clip(float(raw), float(lo), float(hi)))


def _resolve_density_affine_payload(raw: Any, *, key_name: str, y_order: list[str]) -> dict[str, tuple[float, float]]:
    payload = dict(raw or {})
    unknown = sorted(set(str(k) for k in payload.keys()) - set(y_order))
    if unknown:
        raise ValueError(f"{key_name} contains unknown vars: {unknown}")
    out: dict[str, tuple[float, float]] = {}
    for var_name, stats in payload.items():
        stats_dict = dict(stats or {})
        mean = float(stats_dict.get("mean", 0.0))
        std = float(stats_dict.get("std", 1.0))
        if not np.isfinite(mean):
            raise ValueError(f"{key_name}[{var_name}].mean must be finite")
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"{key_name}[{var_name}].std must be > 0")
        out[str(var_name)] = (mean, std)
    return out


def _resolve_scale_payload(raw: Any, *, key_name: str, y_order: list[str]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    unknown = sorted(set(str(k) for k in raw.keys()) - set(y_order))
    if unknown:
        raise ValueError(f"{key_name} contains unknown vars: {unknown}")
    for key, value in raw.items():
        sval = float(value)
        if not np.isfinite(sval) or sval <= 0.0:
            raise ValueError(f"{key_name}[{key}] must be finite and > 0")
        out[str(key)] = sval
    return out


def _huber_loss_and_grad(err: np.ndarray, *, delta: float) -> tuple[np.ndarray, np.ndarray]:
    d = float(max(delta, 1e-8))
    abs_err = np.abs(err)
    quad = abs_err <= d
    loss = np.where(quad, 0.5 * err * err, d * (abs_err - 0.5 * d))
    grad = np.where(quad, err, d * np.sign(err))
    return loss.astype(np.float32), grad.astype(np.float32)


def _apply_chamber_override_numpy(
    sw: np.ndarray,
    *,
    mask: np.ndarray,
    var_name: str,
    chamber_weight_by_var: dict[str, Any],
) -> np.ndarray:
    if not chamber_weight_by_var:
        return sw
    if var_name not in chamber_weight_by_var:
        return sw
    chamber_w = float(chamber_weight_by_var[var_name])
    chamber = (mask <= 0.5).astype(np.float32)
    return sw * (1.0 - chamber) + chamber * chamber_w


def _weighted_reduce_numpy(
    loss_map: np.ndarray,
    grad_map: np.ndarray,
    sw: np.ndarray,
    *,
    normalization: str,
    weight_denominator: str = "weighted",
    group_ids: np.ndarray | None = None,
    group_mode: str = "batch",
) -> tuple[float, np.ndarray]:
    norm = normalization
    if norm not in {"pixel_mean", "sample_mean", "none"}:
        raise ValueError(f"Unsupported supervised.normalization: {norm}")
    if weight_denominator not in {"weighted", "count"}:
        raise ValueError("supervised.sample_mean_weight_denominator must be one of: weighted, count")
    if norm == "none":
        base_loss = float(np.sum(loss_map * sw))
        return base_loss, (grad_map * sw).astype(np.float32)
    if norm == "sample_mean":
        b = int(loss_map.shape[0])
        numer = np.sum(loss_map * sw, axis=(1, 2))
        if weight_denominator == "count":
            denom = np.maximum(np.sum((sw > 0.0).astype(np.float32), axis=(1, 2)), 1.0)
        else:
            denom = np.maximum(np.sum(sw, axis=(1, 2)), 1.0)
        per_sample = numer / denom
        grad = (grad_map * sw) / denom[:, None, None]
        if group_ids is None:
            base_loss = float(np.mean(per_sample))
            grad = grad / float(max(b, 1))
            return base_loss, grad.astype(np.float32)
        if group_mode != "batch":
            raise ValueError("supervised.sample_mean_group_mode must be one of: batch")
        gids = np.asarray(group_ids).reshape(-1)
        if gids.shape[0] != b:
            raise ValueError(f"group_ids length mismatch: expected {b}, got {gids.shape[0]}")
        _, inv = np.unique(gids, return_inverse=True)
        n_groups = int(np.max(inv) + 1) if inv.size > 0 else 0
        if n_groups <= 0:
            base_loss = float(np.mean(per_sample))
            grad = grad / float(max(b, 1))
            return base_loss, grad.astype(np.float32)
        group_means = np.zeros((n_groups,), dtype=np.float32)
        group_sizes = np.zeros((n_groups,), dtype=np.float32)
        for gi in range(n_groups):
            m = inv == gi
            if not np.any(m):
                continue
            group_means[gi] = float(np.mean(per_sample[m]))
            group_sizes[gi] = float(np.sum(m))
        base_loss = float(np.mean(group_means))
        group_factor = np.zeros((b,), dtype=np.float32)
        for gi in range(n_groups):
            m = inv == gi
            if not np.any(m):
                continue
            group_factor[m] = 1.0 / float(max(n_groups * group_sizes[gi], 1.0))
        grad = grad * group_factor[:, None, None]
        return base_loss, grad.astype(np.float32)
    denom = float(max(np.sum(sw), 1.0))
    base_loss = float(np.sum(loss_map * sw) / denom)
    grad = (grad_map * sw) / denom
    return base_loss, grad.astype(np.float32)


def _weighted_mean_count_numpy(
    loss_map: np.ndarray,
    grad_map: np.ndarray,
    sw: np.ndarray,
    region_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    reg = np.asarray(region_mask, dtype=np.float32)
    numer = np.sum(loss_map * sw * reg)
    denom = float(max(np.sum(reg > 0.0), 1.0))
    loss = float(numer / denom)
    grad = (grad_map * sw * reg) / denom
    return loss, grad.astype(np.float32)


def _resolve_region_balance_schedule_weights(
    *,
    region_balance_cfg: dict[str, Any],
    w_boundary: float,
    w_mid: float,
    w_deep: float,
    epoch_idx: int | None,
) -> tuple[float, float, float]:
    schedule_cfg = dict(region_balance_cfg.get("schedule", {}))
    if not bool(schedule_cfg.get("enabled", False)):
        return float(w_boundary), float(w_mid), float(w_deep)
    warmup = max(int(schedule_cfg.get("warmup_epochs", 0)), 0)
    ramp = max(int(schedule_cfg.get("ramp_epochs", 0)), 0)
    e = max(int(epoch_idx or 0), 0)
    if e <= warmup:
        t = 0.0
    elif ramp <= 0:
        t = 1.0
    else:
        t = float(np.clip((e - warmup) / float(ramp), 0.0, 1.0))
    b_start = float(schedule_cfg.get("boundary_start", w_boundary))
    b_end = float(schedule_cfg.get("boundary_end", w_boundary))
    m_start = float(schedule_cfg.get("mid_start", w_mid))
    m_end = float(schedule_cfg.get("mid_end", w_mid))
    d_start = float(schedule_cfg.get("deep_start", w_deep))
    d_end = float(schedule_cfg.get("deep_end", w_deep))
    return (
        float((1.0 - t) * b_start + t * b_end),
        float((1.0 - t) * m_start + t * m_end),
        float((1.0 - t) * d_start + t * d_end),
    )


def _spatial_consistency_grad_huber_numpy(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=np.float32)
    tgt_arr = np.asarray(target, dtype=np.float32)
    reg = np.asarray(region_mask, dtype=np.float32)
    grad_total = np.zeros_like(pred_arr, dtype=np.float32)
    loss_total = 0.0

    # Horizontal edges.
    err_x = (pred_arr[:, :, 1:] - pred_arr[:, :, :-1]) - (tgt_arr[:, :, 1:] - tgt_arr[:, :, :-1])
    mask_x = (reg[:, :, 1:] > 0.5).astype(np.float32) * (reg[:, :, :-1] > 0.5).astype(np.float32)
    if np.any(mask_x > 0.0):
        l_x, g_x = _huber_loss_and_grad(err_x, delta=delta)
        denom_x = float(max(np.sum(mask_x > 0.0), 1.0))
        g_x = (g_x * mask_x) / denom_x
        loss_total += float(np.sum(l_x * mask_x) / denom_x)
        grad_total[:, :, 1:] += g_x
        grad_total[:, :, :-1] -= g_x

    # Vertical edges.
    err_y = (pred_arr[:, 1:, :] - pred_arr[:, :-1, :]) - (tgt_arr[:, 1:, :] - tgt_arr[:, :-1, :])
    mask_y = (reg[:, 1:, :] > 0.5).astype(np.float32) * (reg[:, :-1, :] > 0.5).astype(np.float32)
    if np.any(mask_y > 0.0):
        l_y, g_y = _huber_loss_and_grad(err_y, delta=delta)
        denom_y = float(max(np.sum(mask_y > 0.0), 1.0))
        g_y = (g_y * mask_y) / denom_y
        loss_total += float(np.sum(l_y * mask_y) / denom_y)
        grad_total[:, 1:, :] += g_y
        grad_total[:, :-1, :] -= g_y

    return float(loss_total), grad_total.astype(np.float32)


def _avg_pool2d_bhw(arr: np.ndarray, scale: int) -> tuple[np.ndarray, tuple[int, int]]:
    if scale <= 1:
        return np.asarray(arr, dtype=np.float32), (int(arr.shape[-2]), int(arr.shape[-1]))
    h = int(arr.shape[-2])
    w = int(arr.shape[-1])
    h_eff = (h // scale) * scale
    w_eff = (w // scale) * scale
    if h_eff <= 0 or w_eff <= 0:
        return np.asarray(arr, dtype=np.float32), (h, w)
    cropped = np.asarray(arr[..., :h_eff, :w_eff], dtype=np.float32)
    pooled = cropped.reshape(cropped.shape[0], h_eff // scale, scale, w_eff // scale, scale).mean(axis=(2, 4))
    return pooled.astype(np.float32), (h_eff, w_eff)


def _avg_unpool2d_bhw(
    grad_small: np.ndarray,
    *,
    full_shape: tuple[int, int, int],
    crop_shape: tuple[int, int],
    scale: int,
) -> np.ndarray:
    b, h, w = int(full_shape[0]), int(full_shape[1]), int(full_shape[2])
    if scale <= 1:
        return np.asarray(grad_small, dtype=np.float32).reshape(b, h, w)
    h_eff, w_eff = int(crop_shape[0]), int(crop_shape[1])
    out = np.zeros((b, h, w), dtype=np.float32)
    if h_eff <= 0 or w_eff <= 0:
        return out
    grad = np.asarray(grad_small, dtype=np.float32)[:, : h_eff // scale, : w_eff // scale]
    tiled = np.repeat(np.repeat(grad, scale, axis=1), scale, axis=2) / float(scale * scale)
    out[:, :h_eff, :w_eff] = tiled.astype(np.float32)
    return out


def compose_supervised_numpy(
    pred_fields: dict[str, Any],
    target_fields: dict[str, Any],
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None = None,
    mask: np.ndarray | None = None,
    distance_any: np.ndarray | None = None,
    distance_signed: np.ndarray | None = None,
    bc_dir_mask: np.ndarray | None = None,
    wafer_mask: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    epoch_idx: int | None = None,
) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
    """Compose supervised loss/gradients for grid outputs with optional mask and uncertainty weighting."""

    cfg = _resolve_supervised_cfg(loss_cfg)
    mask_arr = None if mask is None else _as_bhw(mask, key="mask")
    distance_arr = None if distance_any is None else _as_bhw(distance_any, key="distance_any")
    region_cfg = dict(cfg.get("region_weighting", {}))
    use_region = bool(region_cfg.get("enabled", False))
    boundary_delta = float(region_cfg.get("boundary_delta", 2.0))
    w_bulk = float(region_cfg.get("w_bulk", 1.0))
    w_boundary = float(region_cfg.get("w_boundary", 3.0))
    robust_cfg = dict(cfg.get("robust_weighting", {}))
    use_robust = bool(robust_cfg.get("enabled", False))
    robust_vars = {str(v) for v in robust_cfg.get("vars", ["phi", "Te"])}
    robust_mad_scale = float(robust_cfg.get("mad_scale", 3.0))
    robust_min_weight = float(robust_cfg.get("min_weight", 0.2))
    normalization = str(cfg.get("normalization", "pixel_mean")).strip().lower()
    sample_mean_group_mode = str(cfg.get("sample_mean_group_mode", "batch")).strip().lower()
    sample_mean_weight_denominator = str(cfg.get("sample_mean_weight_denominator", "weighted")).strip().lower()
    if sample_mean_group_mode != "batch":
        raise ValueError("supervised.sample_mean_group_mode must be one of: batch")
    delta_by_var = dict(cfg.get("delta_by_var", {}))
    unknown_delta_vars = sorted(set(str(k) for k in delta_by_var.keys()) - set(y_order))
    if unknown_delta_vars:
        raise ValueError(f"supervised.delta_by_var contains unknown vars: {unknown_delta_vars}")
    for k, v in delta_by_var.items():
        if float(v) <= 0.0:
            raise ValueError(f"supervised.delta_by_var[{k}] must be > 0")
    fixed_weights_by_var = dict(cfg.get("fixed_weights_by_var", {}))
    if cfg["weighting"] != "fixed" and len(fixed_weights_by_var) > 0:
        raise ValueError("multitask.fixed_weights_by_var requires multitask.weighting=fixed")
    unknown_fixed_weight_vars = sorted(set(str(k) for k in fixed_weights_by_var.keys()) - set(y_order))
    if unknown_fixed_weight_vars:
        raise ValueError(f"multitask.fixed_weights_by_var contains unknown vars: {unknown_fixed_weight_vars}")
    nan_region_policy = str(cfg.get("nan_region_policy", "mask_only")).strip().lower()
    if nan_region_policy not in {"mask_only", "sdf_continuous"}:
        raise ValueError(f"Unsupported supervised.nan_region_policy: {nan_region_policy}")
    sdf_distance_contract = str(cfg.get("sdf_distance_contract", "off")).strip().lower()
    if sdf_distance_contract not in {"off", "warn", "error"}:
        raise ValueError("supervised.sdf_distance_contract must be one of: off, warn, error")
    global_target_region = str(cfg.get("global_target_region", "")).strip().lower()
    if global_target_region and global_target_region not in {"plasma_only", "sdf_continuous"}:
        raise ValueError(f"Unsupported supervised.global_target_region: {global_target_region}")
    effective_region_policy = nan_region_policy
    if global_target_region == "plasma_only":
        effective_region_policy = "mask_only"
    elif global_target_region == "sdf_continuous":
        effective_region_policy = "sdf_continuous"
    sdf_cfg = dict(cfg.get("sdf_weighting", {}))
    chamber_weight_by_var = dict(cfg.get("chamber_weight_by_var", {}))
    chamber_aux_cfg = dict(cfg.get("chamber_aux", {}))
    region_balance_cfg = dict(cfg.get("region_balance", {}))
    boundary_type_cfg = dict(cfg.get("boundary_type_weighting", {}))
    boundary_profile_cfg = dict(cfg.get("boundary_profile_weighting", {}))
    density_pos_cfg = dict(cfg.get("density_positivity_penalty", {}))
    density_rel_cfg = dict(cfg.get("density_relative_weighting", {}))
    spatial_consistency_cfg = dict(cfg.get("spatial_consistency", {}))
    boundary_type_enabled = bool(boundary_type_cfg.get("enabled", False))
    boundary_type_vars = {str(v) for v in boundary_type_cfg.get("vars", ["Te", "phi"])}
    boundary_type_band_px = float(boundary_type_cfg.get("band_px", 2.0))
    boundary_type_weights = dict(boundary_type_cfg.get("weights", {}))
    boundary_type_weight_interface = float(boundary_type_weights.get("interface", 1.0))
    boundary_type_weight_bc_dir = float(boundary_type_weights.get("bc_dir", 1.0))
    boundary_type_weight_wafer = float(boundary_type_weights.get("wafer", 1.0))
    if boundary_type_weight_interface < 0.0 or boundary_type_weight_bc_dir < 0.0 or boundary_type_weight_wafer < 0.0:
        raise ValueError("supervised.boundary_type_weighting.weights must be >= 0")
    if boundary_type_enabled:
        unknown_boundary_type_vars = sorted(boundary_type_vars - set(y_order))
        if unknown_boundary_type_vars:
            raise ValueError(
                f"supervised.boundary_type_weighting.vars contains unknown vars: {unknown_boundary_type_vars}"
            )
    boundary_profile_enabled = bool(boundary_profile_cfg.get("enabled", False))
    boundary_profile_vars = {str(v) for v in boundary_profile_cfg.get("vars", ["Te", "phi"])}
    boundary_profile_mode = str(boundary_profile_cfg.get("mode", "exp_decay")).strip().lower()
    if boundary_profile_mode not in {"exp_decay"}:
        raise ValueError("supervised.boundary_profile_weighting.mode must be: exp_decay")
    boundary_profile_band_px = float(boundary_profile_cfg.get("band_px", 2.0))
    boundary_profile_alpha = float(boundary_profile_cfg.get("alpha", 0.35))
    boundary_profile_tau_px = float(boundary_profile_cfg.get("tau_px", 0.8))
    boundary_profile_norm_plasma = bool(boundary_profile_cfg.get("normalize_plasma_mean_one", False))
    if boundary_profile_band_px <= 0.0:
        raise ValueError("supervised.boundary_profile_weighting.band_px must be > 0")
    if boundary_profile_tau_px <= 0.0:
        raise ValueError("supervised.boundary_profile_weighting.tau_px must be > 0")
    if boundary_profile_alpha < 0.0:
        raise ValueError("supervised.boundary_profile_weighting.alpha must be >= 0")
    if boundary_profile_enabled:
        unknown_boundary_profile_vars = sorted(boundary_profile_vars - set(y_order))
        if unknown_boundary_profile_vars:
            raise ValueError(
                f"supervised.boundary_profile_weighting.vars contains unknown vars: {unknown_boundary_profile_vars}"
            )
    boundary_profile_mid_px = float(boundary_profile_cfg.get("mid_plasma_px", boundary_profile_cfg.get("deep_plasma_px", 10.0)))
    boundary_profile_deep_px = float(boundary_profile_cfg.get("deep_plasma_px", 10.0))
    boundary_profile_parts_raw = dict(boundary_profile_cfg.get("parts", {}))
    valid_boundary_profile_parts = {"boundary_in", "plasma_mid", "plasma_deep"}
    unknown_boundary_profile_parts = sorted(set(str(k) for k in boundary_profile_parts_raw.keys()) - valid_boundary_profile_parts)
    if unknown_boundary_profile_parts:
        raise ValueError(
            "supervised.boundary_profile_weighting.parts contains unknown regions: "
            f"{unknown_boundary_profile_parts}"
        )
    boundary_profile_parts: dict[str, tuple[float, float]] = {}
    for part_name, part_cfg_raw in boundary_profile_parts_raw.items():
        part_cfg = dict(part_cfg_raw or {})
        part_alpha = float(part_cfg.get("alpha", boundary_profile_alpha))
        part_tau = float(part_cfg.get("tau_px", boundary_profile_tau_px))
        if part_alpha < 0.0:
            raise ValueError(f"supervised.boundary_profile_weighting.parts.{part_name}.alpha must be >= 0")
        if part_tau <= 0.0:
            raise ValueError(f"supervised.boundary_profile_weighting.parts.{part_name}.tau_px must be > 0")
        boundary_profile_parts[str(part_name)] = (part_alpha, part_tau)
    if not boundary_profile_parts:
        boundary_profile_parts = {"boundary_in": (boundary_profile_alpha, boundary_profile_tau_px)}
    boundary_profile_type_mult = dict(boundary_profile_cfg.get("type_multiplier", {}))
    boundary_profile_weight_interface = float(boundary_profile_type_mult.get("interface", 1.0))
    boundary_profile_weight_bc_dir = float(boundary_profile_type_mult.get("bc_dir", 1.0))
    boundary_profile_weight_wafer = float(boundary_profile_type_mult.get("wafer", 1.0))
    if (
        boundary_profile_weight_interface < 0.0
        or boundary_profile_weight_bc_dir < 0.0
        or boundary_profile_weight_wafer < 0.0
    ):
        raise ValueError("supervised.boundary_profile_weighting.type_multiplier values must be >= 0")
    density_pos_enabled = bool(density_pos_cfg.get("enabled", False))
    density_pos_vars = {str(v) for v in density_pos_cfg.get("vars", ["ne", "ni"])}
    density_pos_floor = float(density_pos_cfg.get("floor", 0.0))
    density_pos_lambda = float(density_pos_cfg.get("lambda", 0.0))
    density_pos_affine_by_var = _resolve_density_affine_payload(
        density_pos_cfg.get("affine_by_var", {}),
        key_name="supervised.density_positivity_penalty.affine_by_var",
        y_order=y_order,
    )
    if density_pos_lambda < 0.0:
        raise ValueError("supervised.density_positivity_penalty.lambda must be >= 0")
    if density_pos_enabled:
        unknown_density_pos_vars = sorted(density_pos_vars - set(y_order))
        if unknown_density_pos_vars:
            raise ValueError(
                "supervised.density_positivity_penalty.vars contains unknown vars: "
                f"{unknown_density_pos_vars}"
            )
    density_rel_enabled = bool(density_rel_cfg.get("enabled", False))
    density_rel_vars = {str(v) for v in density_rel_cfg.get("vars", ["ne", "ni"])}
    density_rel_lambda = float(density_rel_cfg.get("lambda", 0.0))
    density_rel_affine_by_var = _resolve_density_affine_payload(
        density_rel_cfg.get("affine_by_var", {}),
        key_name="supervised.density_relative_weighting.affine_by_var",
        y_order=y_order,
    )
    if density_rel_lambda < 0.0:
        raise ValueError("supervised.density_relative_weighting.lambda must be >= 0")
    density_rel_eps_default = float(density_rel_cfg.get("eps", 1.0e-6))
    density_rel_eps_min = float(density_rel_cfg.get("eps_min", 1.0e-8))
    if density_rel_eps_default <= 0.0:
        raise ValueError("supervised.density_relative_weighting.eps must be > 0")
    if density_rel_eps_min <= 0.0:
        raise ValueError("supervised.density_relative_weighting.eps_min must be > 0")
    density_rel_eps_by_var = {str(k): float(v) for k, v in dict(density_rel_cfg.get("eps_by_var", {})).items()}
    unknown_density_rel_eps_vars = sorted(set(density_rel_eps_by_var.keys()) - set(y_order))
    if unknown_density_rel_eps_vars:
        raise ValueError(
            "supervised.density_relative_weighting.eps_by_var contains unknown vars: "
            f"{unknown_density_rel_eps_vars}"
        )
    for key, value in density_rel_eps_by_var.items():
        if float(value) <= 0.0:
            raise ValueError(f"supervised.density_relative_weighting.eps_by_var[{key}] must be > 0")
    if density_rel_enabled:
        unknown_density_rel_vars = sorted(density_rel_vars - set(y_order))
        if unknown_density_rel_vars:
            raise ValueError(
                "supervised.density_relative_weighting.vars contains unknown vars: "
                f"{unknown_density_rel_vars}"
            )
    spatial_consistency_enabled = bool(spatial_consistency_cfg.get("enabled", False))
    spatial_consistency_mode = str(spatial_consistency_cfg.get("mode", "grad_huber")).strip().lower()
    if spatial_consistency_mode not in {"grad_huber"}:
        raise ValueError("supervised.spatial_consistency.mode must be: grad_huber")
    spatial_consistency_lambda = float(spatial_consistency_cfg.get("lambda", 0.05))
    if spatial_consistency_lambda < 0.0:
        raise ValueError("supervised.spatial_consistency.lambda must be >= 0")
    spatial_consistency_delta = float(spatial_consistency_cfg.get("delta", 1.0))
    if spatial_consistency_delta <= 0.0:
        raise ValueError("supervised.spatial_consistency.delta must be > 0")
    spatial_consistency_vars = {str(v) for v in spatial_consistency_cfg.get("vars", y_order)}
    unknown_spatial_consistency_vars = sorted(spatial_consistency_vars - set(y_order))
    if unknown_spatial_consistency_vars:
        raise ValueError(
            "supervised.spatial_consistency.vars contains unknown vars: "
            f"{unknown_spatial_consistency_vars}"
        )
    spatial_consistency_region = str(spatial_consistency_cfg.get("apply_region", "plasma_only")).strip().lower()
    if spatial_consistency_region not in {"plasma_only"}:
        raise ValueError("supervised.spatial_consistency.apply_region must be: plasma_only")
    spatial_consistency_normalize = bool(spatial_consistency_cfg.get("normalize_by_var_scale", False))
    spatial_consistency_scale_by_var = _resolve_scale_payload(
        spatial_consistency_cfg.get("scale_by_var", {}),
        key_name="supervised.spatial_consistency.scale_by_var",
        y_order=y_order,
    )
    spatial_consistency_ms_cfg = dict(spatial_consistency_cfg.get("multiscale", {}))
    spatial_consistency_ms_enabled = bool(spatial_consistency_ms_cfg.get("enabled", False))
    spatial_consistency_ms_scales_raw = spatial_consistency_ms_cfg.get("scales", [1, 2, 4])
    if not isinstance(spatial_consistency_ms_scales_raw, list) or len(spatial_consistency_ms_scales_raw) == 0:
        raise ValueError("supervised.spatial_consistency.multiscale.scales must be a non-empty list")
    spatial_consistency_ms_scales = [int(max(int(v), 1)) for v in spatial_consistency_ms_scales_raw]
    spatial_consistency_ms_weights_raw = spatial_consistency_ms_cfg.get("scale_weights", [1.0] * len(spatial_consistency_ms_scales))
    if not isinstance(spatial_consistency_ms_weights_raw, list) or len(spatial_consistency_ms_weights_raw) != len(spatial_consistency_ms_scales):
        raise ValueError("supervised.spatial_consistency.multiscale.scale_weights must match scales length")
    spatial_consistency_ms_weights = [float(v) for v in spatial_consistency_ms_weights_raw]
    if any((not np.isfinite(v) or v < 0.0) for v in spatial_consistency_ms_weights):
        raise ValueError("supervised.spatial_consistency.multiscale.scale_weights must be finite and >= 0")
    region_balance_enabled = bool(region_balance_cfg.get("enabled", False))
    region_balance_mode = str(region_balance_cfg.get("mode", "replace")).strip().lower()
    if region_balance_mode not in {"replace", "additive"}:
        raise ValueError("supervised.region_balance.mode must be one of: replace, additive")
    region_balance_lambda = float(region_balance_cfg.get("additive_lambda", 0.25))
    if region_balance_lambda < 0.0:
        raise ValueError("supervised.region_balance.additive_lambda must be >= 0")
    region_balance_boundary_px = float(region_balance_cfg.get("boundary_in_px", 2.0))
    region_balance_mid_px = float(region_balance_cfg.get("mid_plasma_px", region_balance_cfg.get("deep_plasma_px", 10.0)))
    region_balance_deep_px = float(region_balance_cfg.get("deep_plasma_px", 10.0))
    region_balance_vars = {str(v) for v in region_balance_cfg.get("vars", ["log_ne", "log_ni"])}
    region_balance_w_boundary = float(region_balance_cfg.get("weight_boundary_in", 0.6))
    region_balance_w_mid = float(region_balance_cfg.get("weight_plasma_mid", 0.0))
    region_balance_w_deep = float(region_balance_cfg.get("weight_deep_plasma", 0.4))
    region_balance_w_boundary, region_balance_w_mid, region_balance_w_deep = _resolve_region_balance_schedule_weights(
        region_balance_cfg=region_balance_cfg,
        w_boundary=region_balance_w_boundary,
        w_mid=region_balance_w_mid,
        w_deep=region_balance_w_deep,
        epoch_idx=epoch_idx,
    )
    region_balance_reduce = str(region_balance_cfg.get("reduce", "mean_count")).strip().lower()
    if region_balance_reduce not in {"mean_count"}:
        raise ValueError("supervised.region_balance.reduce must be: mean_count")
    chamber_aux_enabled = bool(chamber_aux_cfg.get("enabled", False))
    chamber_aux_scope = str(chamber_aux_cfg.get("scope", "band_only")).strip().lower()
    if chamber_aux_scope not in {"band_only", "full"}:
        raise ValueError("supervised.chamber_aux.scope must be one of: band_only, full")
    chamber_aux_band_px = float(chamber_aux_cfg.get("band_px", 2.0))
    chamber_aux_weight_by_var = dict(chamber_aux_cfg.get("weight_by_var", {}))
    chamber_aux_grad_clip_abs = float(chamber_aux_cfg.get("grad_clip_abs", 3.0))
    robust_clip_stats = dict(cfg.get("robust_clip_stats", {}))
    grads: dict[str, np.ndarray] = {}
    per_var_loss: dict[str, float] = {}
    total = 0.0
    chamber_aux_total = 0.0
    density_pos_total = 0.0
    density_rel_total = 0.0
    spatial_consistency_total = 0.0
    region_boundary_total = 0.0
    region_mid_total = 0.0
    region_deep_total = 0.0
    eps = 1e-8
    bc_dir_arr = None if bc_dir_mask is None else _as_bhw(bc_dir_mask, key="bc_dir_mask")
    wafer_arr = None if wafer_mask is None else _as_bhw(wafer_mask, key="wafer_mask")
    signed_arr = None if distance_signed is None else _as_bhw(distance_signed, key="distance_signed")
    profile_region_masks_cache: dict[str, np.ndarray] | None = None
    profile_type_masks_cache: dict[str, np.ndarray] | None = None

    for name in y_order:
        pred = _as_bhw(pred_fields[name], key=f"pred_{name}")
        tgt = _as_bhw(target_fields[name], key=f"target_{name}")
        if bool(cfg.get("label_clip_from_scaler", False)):
            clip_stats = dict(robust_clip_stats.get(name, {}))
            if "clip_q01" in clip_stats and "clip_q99" in clip_stats:
                q_lo = float(clip_stats["clip_q01"])
                q_hi = float(clip_stats["clip_q99"])
                tgt = np.clip(tgt, q_lo, q_hi).astype(np.float32)
        err = (pred - tgt).astype(np.float32)
        var_delta = float(delta_by_var.get(name, cfg["delta"])) if delta_by_var else float(cfg["delta"])
        if cfg["type"] == "huber":
            loss_map, grad_map = _huber_loss_and_grad(err, delta=var_delta)
        else:
            loss_map = 0.5 * err * err
            grad_map = err

        if mask_arr is None:
            sw = np.ones_like(loss_map, dtype=np.float32)
            base_loss, grad = _weighted_reduce_numpy(
                loss_map,
                grad_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                group_mode=sample_mean_group_mode,
            )
        else:
            m = mask_arr
            d_map = distance_arr
            d_signed = None
            if m.shape[0] == 1 and pred.shape[0] > 1:
                m = np.repeat(m, pred.shape[0], axis=0)
            if d_map is not None and d_map.shape[0] == 1 and pred.shape[0] > 1:
                d_map = np.repeat(d_map, pred.shape[0], axis=0)
            if signed_arr is not None and signed_arr.shape[0] == 1 and pred.shape[0] > 1:
                signed_arr = np.repeat(signed_arr, pred.shape[0], axis=0)
            if effective_region_policy == "sdf_continuous":
                if d_map is None:
                    raise ValueError("nan_region_policy=sdf_continuous requires distance_any")
                d_signed = build_signed_distance(m, d_map)
                if not is_effective_signed_distance(m, d_signed):
                    msg = (
                        "sdf_continuous contract violation: signed distance has no negative chamber-side values; "
                        "distance_any may be invalid for chamber supervision"
                    )
                    if sdf_distance_contract == "error":
                        raise ValueError(msg)
                    if sdf_distance_contract == "warn":
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                sw = sdf_continuous_weight_map(m, d_signed, sdf_cfg)
                if use_region:
                    boundary = (d_map <= boundary_delta).astype(np.float32) * m
                    bulk = np.clip(m - boundary, 0.0, 1.0)
                    region_mult = boundary * max(w_boundary / max(w_bulk, eps), 1.0) + bulk
                    outside = (1.0 - m).astype(np.float32)
                    sw = sw * (region_mult + outside)
            else:
                if use_region:
                    if d_map is None:
                        raise ValueError("region_weighting.enabled requires distance_any")
                    boundary = (d_map <= boundary_delta).astype(np.float32) * m
                    bulk = np.clip(m - boundary, 0.0, 1.0)
                    sw = boundary * w_boundary + bulk * w_bulk
                else:
                    sw = m
            if boundary_type_enabled and name in boundary_type_vars and d_map is not None:
                type_masks = build_boundary_type_masks(
                    mask_plasma=m,
                    distance_any=d_map,
                    band_px=boundary_type_band_px,
                    bc_dir_mask=bc_dir_arr,
                    wafer_mask=wafer_arr,
                )
                type_mult = np.ones_like(sw, dtype=np.float32)
                type_mult[type_masks["interface"]] = boundary_type_weight_interface
                type_mult[type_masks["bc_dir"]] = boundary_type_weight_bc_dir
                type_mult[type_masks["wafer"]] = boundary_type_weight_wafer
                sw = sw * type_mult
            if boundary_profile_enabled and name in boundary_profile_vars:
                if d_signed is None:
                    if d_map is not None:
                        d_signed = build_signed_distance(m, d_map)
                    else:
                        raise ValueError(
                            "supervised.boundary_profile_weighting requires distance_signed or distance_any"
                        )
                profile_mult = np.ones_like(sw, dtype=np.float32)
                if profile_region_masks_cache is None:
                    profile_region_masks_cache = build_region_masks(
                        mask_plasma=m,
                        distance_any=d_map,
                        distance_signed=d_signed,
                        boundary_in_px=boundary_profile_band_px,
                        mid_plasma_px=boundary_profile_mid_px,
                        deep_plasma_px=boundary_profile_deep_px,
                    )
                d_pos = np.clip(d_signed, 0.0, None).astype(np.float32)
                for part_name, (part_alpha, part_tau) in boundary_profile_parts.items():
                    if part_alpha <= 0.0:
                        continue
                    region_mask = np.asarray(profile_region_masks_cache.get(part_name, np.zeros_like(sw, dtype=bool)), dtype=bool)
                    if not np.any(region_mask):
                        continue
                    profile_mult[region_mask] = (
                        1.0 + float(part_alpha) * np.exp(-d_pos[region_mask] / float(part_tau))
                    )
                if (
                    boundary_profile_weight_interface != 1.0
                    or boundary_profile_weight_bc_dir != 1.0
                    or boundary_profile_weight_wafer != 1.0
                ):
                    if profile_type_masks_cache is None:
                        profile_type_masks_cache = build_boundary_type_masks(
                            mask_plasma=m,
                            distance_any=d_map if d_map is not None else np.abs(d_signed).astype(np.float32),
                            band_px=boundary_profile_band_px,
                            bc_dir_mask=bc_dir_arr,
                            wafer_mask=wafer_arr,
                        )
                    profile_mult[profile_type_masks_cache["interface"]] *= boundary_profile_weight_interface
                    profile_mult[profile_type_masks_cache["bc_dir"]] *= boundary_profile_weight_bc_dir
                    profile_mult[profile_type_masks_cache["wafer"]] *= boundary_profile_weight_wafer
                if boundary_profile_norm_plasma:
                    plasma_active = np.asarray(m > 0.5, dtype=bool)
                    if np.any(plasma_active):
                        prof_mean = float(np.mean(profile_mult[plasma_active]))
                        if np.isfinite(prof_mean) and prof_mean > 1.0e-12:
                            profile_mult = (profile_mult / prof_mean).astype(np.float32)
                sw = sw * profile_mult
            if global_target_region != "plasma_only":
                sw = _apply_chamber_override_numpy(
                    sw,
                    mask=m,
                    var_name=name,
                    chamber_weight_by_var=chamber_weight_by_var,
                )
            if use_robust and name in robust_vars:
                active = sw > 0.5
                if np.any(active):
                    tgt_active = tgt[active]
                    med = float(np.median(tgt_active))
                    mad = float(np.median(np.abs(tgt_active - med)))
                    scale = max(mad * robust_mad_scale, eps)
                    rw = 1.0 / (1.0 + np.abs(tgt - med) / scale)
                    rw = np.clip(rw, robust_min_weight, 1.0).astype(np.float32)
                    sw = sw * rw
            base_loss, grad = _weighted_reduce_numpy(
                loss_map,
                grad_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                group_mode=sample_mean_group_mode,
            )
            if (
                region_balance_enabled
                and name in region_balance_vars
                and d_map is not None
                and m is not None
            ):
                d_signed_eff = (
                    np.asarray(signed_arr, dtype=np.float32)
                    if signed_arr is not None
                    else np.where(m > 0.5, d_map, -d_map).astype(np.float32)
                )
                region_masks = build_region_masks(
                    mask_plasma=m,
                    distance_any=d_map,
                    distance_signed=d_signed_eff,
                    boundary_in_px=region_balance_boundary_px,
                    mid_plasma_px=region_balance_mid_px,
                    deep_plasma_px=region_balance_deep_px,
                )
                boundary_region = region_masks["boundary_in"]
                mid_region = region_masks["plasma_mid"]
                deep_region = region_masks["plasma_deep"]
                l_boundary, g_boundary = _weighted_mean_count_numpy(
                    loss_map,
                    grad_map,
                    sw,
                    boundary_region.astype(np.float32),
                )
                l_mid, g_mid = _weighted_mean_count_numpy(
                    loss_map,
                    grad_map,
                    sw,
                    mid_region.astype(np.float32),
                )
                l_deep, g_deep = _weighted_mean_count_numpy(
                    loss_map,
                    grad_map,
                    sw,
                    deep_region.astype(np.float32),
                )
                region_loss = float(
                    region_balance_w_boundary * l_boundary
                    + region_balance_w_mid * l_mid
                    + region_balance_w_deep * l_deep
                )
                region_grad = (
                    region_balance_w_boundary * g_boundary
                    + region_balance_w_mid * g_mid
                    + region_balance_w_deep * g_deep
                ).astype(np.float32)
                if region_balance_mode == "replace":
                    base_loss = region_loss
                    grad = region_grad
                else:
                    base_loss = float(base_loss + region_balance_lambda * region_loss)
                    grad = (grad + region_balance_lambda * region_grad).astype(np.float32)
                region_boundary_total += float(l_boundary)
                region_mid_total += float(l_mid)
                region_deep_total += float(l_deep)
            if chamber_aux_enabled and (name in chamber_aux_weight_by_var):
                aux_w = float(chamber_aux_weight_by_var.get(name, 0.0))
                if aux_w > 0.0:
                    if chamber_aux_scope == "full":
                        chamber_mask = (m <= 0.5).astype(np.float32)
                    else:
                        if d_signed is None:
                            if d_map is None:
                                raise ValueError("supervised.chamber_aux(scope=band_only) requires distance_any")
                            d_signed = build_signed_distance(m, d_map)
                        chamber_mask = np.logical_and(d_signed < 0.0, d_signed >= (-float(chamber_aux_band_px))).astype(
                            np.float32
                        )
                    sw_aux = chamber_mask * aux_w
                    aux_loss, aux_grad = _weighted_reduce_numpy(
                        loss_map,
                        grad_map,
                        sw_aux,
                        normalization=normalization,
                        weight_denominator=sample_mean_weight_denominator,
                        group_ids=group_ids,
                        group_mode=sample_mean_group_mode,
                    )
                    if chamber_aux_grad_clip_abs > 0.0:
                        aux_grad = np.clip(aux_grad, -chamber_aux_grad_clip_abs, chamber_aux_grad_clip_abs).astype(
                            np.float32
                        )
                    grad = grad + aux_grad
                    base_loss += float(aux_loss)
                    chamber_aux_total += float(aux_loss)

        if spatial_consistency_enabled and name in spatial_consistency_vars and spatial_consistency_lambda > 0.0:
            if mask_arr is None:
                raise ValueError("supervised.spatial_consistency requires supervised.mask=plasma_only")
            m_sp = m
            d_sp = d_map
            ds_sp = d_signed
            if d_sp is None and ds_sp is None:
                raise ValueError("supervised.spatial_consistency requires distance_any or distance_signed")
            if ds_sp is None:
                ds_sp = np.where(m_sp > 0.5, d_sp, -d_sp).astype(np.float32)
            region_masks_sc = build_region_masks(
                mask_plasma=m_sp,
                distance_any=d_sp,
                distance_signed=ds_sp,
                boundary_in_px=2.0,
                mid_plasma_px=10.0,
                deep_plasma_px=10.0,
            )
            sc_region = np.asarray(region_masks_sc["all_plasma"], dtype=np.float32)
            pred_sc = (
                pred / float(max(spatial_consistency_scale_by_var.get(name, 1.0), 1.0e-12))
                if spatial_consistency_normalize
                else pred
            )
            tgt_sc = (
                tgt / float(max(spatial_consistency_scale_by_var.get(name, 1.0), 1.0e-12))
                if spatial_consistency_normalize
                else tgt
            )
            full_shape = tuple(int(v) for v in pred_sc.shape)
            if spatial_consistency_ms_enabled:
                sc_loss = 0.0
                sc_grad = np.zeros_like(pred_sc, dtype=np.float32)
                for sc_scale, sc_weight in zip(spatial_consistency_ms_scales, spatial_consistency_ms_weights):
                    if sc_weight <= 0.0:
                        continue
                    pred_s, crop = _avg_pool2d_bhw(pred_sc, sc_scale)
                    tgt_s, _ = _avg_pool2d_bhw(tgt_sc, sc_scale)
                    reg_s, _ = _avg_pool2d_bhw(sc_region, sc_scale)
                    reg_s = (np.asarray(reg_s, dtype=np.float32) > 0.5).astype(np.float32)
                    l_s, g_s = _spatial_consistency_grad_huber_numpy(
                        pred=pred_s,
                        target=tgt_s,
                        region_mask=reg_s,
                        delta=spatial_consistency_delta,
                    )
                    sc_loss += float(sc_weight) * float(l_s)
                    sc_grad += float(sc_weight) * _avg_unpool2d_bhw(
                        g_s,
                        full_shape=full_shape,
                        crop_shape=crop,
                        scale=sc_scale,
                    )
            else:
                sc_loss, sc_grad = _spatial_consistency_grad_huber_numpy(
                    pred=pred_sc,
                    target=tgt_sc,
                    region_mask=sc_region,
                    delta=spatial_consistency_delta,
                )
            if spatial_consistency_normalize:
                scale_back = float(max(spatial_consistency_scale_by_var.get(name, 1.0), 1.0e-12))
                sc_grad = (sc_grad / scale_back).astype(np.float32)
            base_loss += float(spatial_consistency_lambda * sc_loss)
            grad = (grad + float(spatial_consistency_lambda) * sc_grad).astype(np.float32)
            spatial_consistency_total += float(spatial_consistency_lambda * sc_loss)

        if density_pos_enabled and name in density_pos_vars and density_pos_lambda > 0.0:
            mean_aff, std_aff = density_pos_affine_by_var.get(name, (0.0, 1.0))
            std_safe = float(max(abs(float(std_aff)), 1.0e-12))
            pred_phys = (pred * float(std_aff) + float(mean_aff)).astype(np.float32)
            violation_norm = ((float(density_pos_floor) - pred_phys) / std_safe).astype(np.float32)
            violation_pos = np.maximum(violation_norm, 0.0).astype(np.float32)
            pos_loss_map = (0.5 * np.square(violation_pos)).astype(np.float32)
            # d/d(pred_scaled) [0.5 * max(v_norm,0)^2], where v_norm=(floor-pred_phys)/std_safe.
            # For std_aff>0 this derivative is -max(v_norm,0); using std_safe keeps it bounded.
            pos_grad_map = np.where(violation_norm > 0.0, -violation_pos, 0.0).astype(np.float32)
            pos_loss, pos_grad = _weighted_reduce_numpy(
                pos_loss_map,
                pos_grad_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                group_mode=sample_mean_group_mode,
            )
            base_loss += float(density_pos_lambda * pos_loss)
            grad = (grad + float(density_pos_lambda) * pos_grad).astype(np.float32)
            density_pos_total += float(density_pos_lambda * pos_loss)

        if density_rel_enabled and name in density_rel_vars and density_rel_lambda > 0.0:
            mean_aff, std_aff = density_rel_affine_by_var.get(name, (0.0, 1.0))
            eps_var = float(density_rel_eps_by_var.get(name, density_rel_eps_default))
            eps_var = max(eps_var, density_rel_eps_min)
            pred_phys = (pred * float(std_aff) + float(mean_aff)).astype(np.float32)
            tgt_phys = (tgt * float(std_aff) + float(mean_aff)).astype(np.float32)
            err_phys = (pred_phys - tgt_phys).astype(np.float32)
            rel_denom = (np.abs(tgt_phys).astype(np.float32) + float(eps_var)).astype(np.float32)
            rel_loss_map = (np.abs(err_phys).astype(np.float32) / rel_denom).astype(np.float32)
            rel_grad_map = (np.sign(err_phys).astype(np.float32) / rel_denom).astype(np.float32)
            rel_grad_map = (rel_grad_map * float(std_aff)).astype(np.float32)
            rel_loss, rel_grad = _weighted_reduce_numpy(
                rel_loss_map,
                rel_grad_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                group_mode=sample_mean_group_mode,
            )
            base_loss += float(density_rel_lambda * rel_loss)
            grad = (grad + float(density_rel_lambda) * rel_grad).astype(np.float32)
            density_rel_total += float(density_rel_lambda * rel_loss)

        sigma = _resolve_sigma_for_var(name, base_loss, cfg)
        if cfg["weighting"] == "uncertainty":
            weighted = float(np.exp(-sigma) * base_loss + sigma)
            grad = grad * float(np.exp(-sigma))
        else:
            fixed_weight = float(fixed_weights_by_var.get(name, 1.0)) if fixed_weights_by_var else 1.0
            weighted = float(base_loss * fixed_weight)
            grad = (grad * fixed_weight).astype(np.float32)

        per_var_loss[name] = weighted
        grads[name] = grad.astype(np.float32)
        total += weighted

    if chamber_aux_enabled:
        per_var_loss["__chamber_aux__"] = float(chamber_aux_total)
    if density_pos_enabled and density_pos_lambda > 0.0:
        per_var_loss["__density_positivity_penalty__"] = float(density_pos_total)
    if density_rel_enabled and density_rel_lambda > 0.0:
        per_var_loss["__density_relative_weighting__"] = float(density_rel_total)
    if spatial_consistency_enabled and spatial_consistency_lambda > 0.0:
        per_var_loss["__spatial_consistency__"] = float(spatial_consistency_total)
    if region_balance_enabled:
        per_var_loss["__region_boundary_in__"] = float(region_boundary_total)
        per_var_loss["__region_mid_plasma__"] = float(region_mid_total)
        per_var_loss["__region_deep_plasma__"] = float(region_deep_total)
        per_var_loss["__region_balance_applied__"] = 1.0
    return float(total), grads, per_var_loss


def _as_bchw(x: Any, *, key: str, device: Any | None = None):
    torch = require_torch()
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 2:
        t = t[None, None, ...]
    if t.ndim == 3:
        t = t[:, None, ...]
    if t.ndim != 4 or int(t.shape[1]) != 1:
        raise ValueError(f"{key} must be [B,1,H,W] or [B,H,W], got {tuple(t.shape)}")
    return t


def _apply_chamber_override_torch(
    sw,
    *,
    mask,
    var_name: str,
    chamber_weight_by_var: dict[str, Any],
):
    if not chamber_weight_by_var:
        return sw
    if var_name not in chamber_weight_by_var:
        return sw
    chamber_w = float(chamber_weight_by_var[var_name])
    chamber = (mask <= 0.5).to(dtype=sw.dtype)
    return sw * (1.0 - chamber) + chamber * chamber_w


def _weighted_reduce_torch(base_map, sw, *, normalization: str, weight_denominator: str = "weighted"):
    torch = require_torch()
    norm = normalization
    if norm not in {"pixel_mean", "sample_mean", "none"}:
        raise ValueError(f"Unsupported supervised.normalization: {norm}")
    if weight_denominator not in {"weighted", "count"}:
        raise ValueError("supervised.sample_mean_weight_denominator must be one of: weighted, count")
    if norm == "none":
        return (base_map * sw).sum()
    if norm == "sample_mean":
        numer = (base_map * sw).sum(dim=(1, 2, 3))
        if weight_denominator == "count":
            denom = torch.clamp((sw > 0.0).to(dtype=sw.dtype).sum(dim=(1, 2, 3)), min=1.0)
        else:
            denom = torch.clamp(sw.sum(dim=(1, 2, 3)), min=1.0)
        return (numer / denom).mean()
    denom = torch.clamp(sw.sum(), min=1.0)
    return (base_map * sw).sum() / denom


def _weighted_mean_count_torch(base_map, sw, region_mask):
    torch = require_torch()
    reg = torch.as_tensor(region_mask, dtype=sw.dtype, device=sw.device)
    numer = (base_map * sw * reg).sum()
    denom = torch.clamp((reg > 0.0).to(dtype=sw.dtype).sum(), min=1.0)
    return numer / denom


def _resolve_torch_region_contract(
    *,
    cfg: dict[str, Any],
    y_order: list[str],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Resolve canonical region contract for torch supervised loss.

    Returns (region_balance_cfg, legacy_region_weighting_cfg, alias_used).
    """

    region_balance_cfg = dict(cfg.get("region_balance", {}))
    legacy_region_cfg = dict(cfg.get("region_weighting", {}))
    if region_balance_cfg and legacy_region_cfg:
        raise ValueError(
            "supervised.region_balance and supervised.region_weighting cannot be specified together; "
            "use supervised.region_balance (region_weighting is deprecated alias)"
        )
    alias_used = False
    if (not region_balance_cfg) and legacy_region_cfg:
        alias_used = True
        warnings.warn(
            "supervised.region_weighting is deprecated for torch path; "
            "use supervised.region_balance instead",
            DeprecationWarning,
            stacklevel=3,
        )
        # Keep legacy semantics when only alias is provided.
        return {}, legacy_region_cfg, alias_used
    if not region_balance_cfg:
        return {}, {}, alias_used
    region_balance_cfg = dict(region_balance_cfg)
    if "vars" not in region_balance_cfg:
        region_balance_cfg["vars"] = list(y_order)
    return region_balance_cfg, {}, alias_used


def compose_supervised_torch(
    pred_fields: dict[str, Any],
    target_fields: Any,
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None = None,
    mask: Any | None = None,
    distance_any: Any | None = None,
):
    """Compose supervised torch loss with optional mask and uncertainty weighting."""

    torch = require_torch()
    cfg = _resolve_supervised_cfg(loss_cfg)
    first_pred = pred_fields[str(y_order[0])] if y_order else None
    pred_device = torch.as_tensor(first_pred, dtype=torch.float32).device if first_pred is not None else None
    tgt = torch.as_tensor(target_fields, dtype=torch.float32, device=pred_device)
    if tgt.ndim != 4:
        raise ValueError(f"target_fields must be [B,C,H,W], got {tuple(tgt.shape)}")
    if int(tgt.shape[1]) != len(y_order):
        raise ValueError(f"target channel mismatch: expected {len(y_order)}, got {int(tgt.shape[1])}")

    if mask is None:
        m = None
        d_any = None
    else:
        m = _as_bchw(mask, key="mask", device=tgt.device)
        if int(m.shape[0]) == 1 and int(tgt.shape[0]) > 1:
            m = m.expand(int(tgt.shape[0]), -1, -1, -1)
        d_any = None
        if distance_any is not None:
            d_any = _as_bchw(distance_any, key="distance_any", device=tgt.device)
            if int(d_any.shape[0]) == 1 and int(tgt.shape[0]) > 1:
                d_any = d_any.expand(int(tgt.shape[0]), -1, -1, -1)

    region_cfg = dict(cfg.get("region_weighting", {}))
    use_region = bool(region_cfg.get("enabled", False))
    boundary_delta = float(region_cfg.get("boundary_delta", 2.0))
    w_bulk = float(region_cfg.get("w_bulk", 1.0))
    w_boundary = float(region_cfg.get("w_boundary", 3.0))
    robust_cfg = dict(cfg.get("robust_weighting", {}))
    use_robust = bool(robust_cfg.get("enabled", False))
    robust_vars = {str(v) for v in robust_cfg.get("vars", ["phi", "Te"])}
    robust_mad_scale = float(robust_cfg.get("mad_scale", 3.0))
    robust_min_weight = float(robust_cfg.get("min_weight", 0.2))
    region_balance_cfg, legacy_region_cfg, _ = _resolve_torch_region_contract(cfg=cfg, y_order=y_order)
    region_balance_enabled = bool(region_balance_cfg.get("enabled", False))
    region_balance_mode = str(region_balance_cfg.get("mode", "replace")).strip().lower()
    if region_balance_mode not in {"replace", "additive"}:
        raise ValueError("supervised.region_balance.mode must be one of: replace, additive")
    region_balance_lambda = float(region_balance_cfg.get("additive_lambda", 0.25))
    if region_balance_lambda < 0.0:
        raise ValueError("supervised.region_balance.additive_lambda must be >= 0")
    region_balance_boundary_px = float(region_balance_cfg.get("boundary_in_px", 2.0))
    region_balance_mid_px = float(region_balance_cfg.get("mid_plasma_px", region_balance_cfg.get("deep_plasma_px", 10.0)))
    region_balance_deep_px = float(region_balance_cfg.get("deep_plasma_px", 10.0))
    region_balance_vars = {str(v) for v in region_balance_cfg.get("vars", list(y_order))}
    region_balance_w_boundary = float(region_balance_cfg.get("weight_boundary_in", 0.6))
    region_balance_w_mid = float(region_balance_cfg.get("weight_plasma_mid", 0.0))
    region_balance_w_deep = float(region_balance_cfg.get("weight_deep_plasma", 0.4))
    region_balance_reduce = str(region_balance_cfg.get("reduce", "mean_count")).strip().lower()
    if region_balance_reduce not in {"mean_count"}:
        raise ValueError("supervised.region_balance.reduce must be: mean_count")
    legacy_region_enabled = bool(legacy_region_cfg.get("enabled", False))
    boundary_delta = float(legacy_region_cfg.get("boundary_delta", 2.0))
    w_bulk = float(legacy_region_cfg.get("w_bulk", 1.0))
    w_boundary = float(legacy_region_cfg.get("w_boundary", 3.0))
    normalization = str(cfg.get("normalization", "pixel_mean")).strip().lower()
    sample_mean_weight_denominator = str(cfg.get("sample_mean_weight_denominator", "weighted")).strip().lower()
    delta_by_var = dict(cfg.get("delta_by_var", {}))
    unknown_delta_vars = sorted(set(str(k) for k in delta_by_var.keys()) - set(y_order))
    if unknown_delta_vars:
        raise ValueError(f"supervised.delta_by_var contains unknown vars: {unknown_delta_vars}")
    for k, v in delta_by_var.items():
        if float(v) <= 0.0:
            raise ValueError(f"supervised.delta_by_var[{k}] must be > 0")
    fixed_weights_by_var = dict(cfg.get("fixed_weights_by_var", {}))
    if cfg["weighting"] != "fixed" and len(fixed_weights_by_var) > 0:
        raise ValueError("multitask.fixed_weights_by_var requires multitask.weighting=fixed")
    unknown_fixed_weight_vars = sorted(set(str(k) for k in fixed_weights_by_var.keys()) - set(y_order))
    if unknown_fixed_weight_vars:
        raise ValueError(f"multitask.fixed_weights_by_var contains unknown vars: {unknown_fixed_weight_vars}")
    nan_region_policy = str(cfg.get("nan_region_policy", "mask_only")).strip().lower()
    if nan_region_policy not in {"mask_only", "sdf_continuous"}:
        raise ValueError(f"Unsupported supervised.nan_region_policy: {nan_region_policy}")
    sdf_distance_contract = str(cfg.get("sdf_distance_contract", "off")).strip().lower()
    if sdf_distance_contract not in {"off", "warn", "error"}:
        raise ValueError("supervised.sdf_distance_contract must be one of: off, warn, error")
    sdf_cfg = dict(cfg.get("sdf_weighting", {}))
    chamber_weight_by_var = dict(cfg.get("chamber_weight_by_var", {}))
    density_pos_cfg = dict(cfg.get("density_positivity_penalty", {}))
    density_rel_cfg = dict(cfg.get("density_relative_weighting", {}))
    density_pos_enabled = bool(density_pos_cfg.get("enabled", False))
    density_pos_vars = {str(v) for v in density_pos_cfg.get("vars", ["ne", "ni"])}
    density_pos_floor = float(density_pos_cfg.get("floor", 0.0))
    density_pos_lambda = float(density_pos_cfg.get("lambda", 0.0))
    density_pos_affine_by_var = _resolve_density_affine_payload(
        density_pos_cfg.get("affine_by_var", {}),
        key_name="supervised.density_positivity_penalty.affine_by_var",
        y_order=y_order,
    )
    if density_pos_lambda < 0.0:
        raise ValueError("supervised.density_positivity_penalty.lambda must be >= 0")
    if density_pos_enabled:
        unknown_density_pos_vars = sorted(density_pos_vars - set(y_order))
        if unknown_density_pos_vars:
            raise ValueError(
                "supervised.density_positivity_penalty.vars contains unknown vars: "
                f"{unknown_density_pos_vars}"
            )
    density_rel_enabled = bool(density_rel_cfg.get("enabled", False))
    density_rel_vars = {str(v) for v in density_rel_cfg.get("vars", ["ne", "ni"])}
    density_rel_lambda = float(density_rel_cfg.get("lambda", 0.0))
    density_rel_affine_by_var = _resolve_density_affine_payload(
        density_rel_cfg.get("affine_by_var", {}),
        key_name="supervised.density_relative_weighting.affine_by_var",
        y_order=y_order,
    )
    if density_rel_lambda < 0.0:
        raise ValueError("supervised.density_relative_weighting.lambda must be >= 0")
    density_rel_eps_default = float(density_rel_cfg.get("eps", 1.0e-6))
    density_rel_eps_min = float(density_rel_cfg.get("eps_min", 1.0e-8))
    if density_rel_eps_default <= 0.0:
        raise ValueError("supervised.density_relative_weighting.eps must be > 0")
    if density_rel_eps_min <= 0.0:
        raise ValueError("supervised.density_relative_weighting.eps_min must be > 0")
    density_rel_eps_by_var = {str(k): float(v) for k, v in dict(density_rel_cfg.get("eps_by_var", {})).items()}
    unknown_density_rel_eps_vars = sorted(set(density_rel_eps_by_var.keys()) - set(y_order))
    if unknown_density_rel_eps_vars:
        raise ValueError(
            "supervised.density_relative_weighting.eps_by_var contains unknown vars: "
            f"{unknown_density_rel_eps_vars}"
        )
    for key, value in density_rel_eps_by_var.items():
        if float(value) <= 0.0:
            raise ValueError(f"supervised.density_relative_weighting.eps_by_var[{key}] must be > 0")
    if density_rel_enabled:
        unknown_density_rel_vars = sorted(density_rel_vars - set(y_order))
        if unknown_density_rel_vars:
            raise ValueError(
                "supervised.density_relative_weighting.vars contains unknown vars: "
                f"{unknown_density_rel_vars}"
            )
    total = torch.zeros((), dtype=torch.float32, device=tgt.device)
    per_var: dict[str, float] = {}
    region_masks_torch: dict[str, Any] | None = None
    if region_balance_enabled:
        if m is None or d_any is None:
            raise ValueError("supervised.region_balance.enabled requires supervised.mask=plasma_only and distance_any")
        d_signed_eff = torch.where(m > 0.5, d_any, -d_any).to(dtype=torch.float32)
        region_masks_np = build_region_masks(
            mask_plasma=np.asarray(m[:, 0].detach().cpu().numpy(), dtype=np.float32),
            distance_any=np.asarray(d_any[:, 0].detach().cpu().numpy(), dtype=np.float32),
            distance_signed=np.asarray(d_signed_eff[:, 0].detach().cpu().numpy(), dtype=np.float32),
            boundary_in_px=region_balance_boundary_px,
            mid_plasma_px=region_balance_mid_px,
            deep_plasma_px=region_balance_deep_px,
        )
        region_masks_torch = {
            str(k): torch.as_tensor(v.astype(np.float32), dtype=torch.float32, device=tgt.device)[:, None, ...]
            for k, v in region_masks_np.items()
        }

    for i, name in enumerate(y_order):
        pred = _as_bchw(pred_fields[name], key=f"pred_{name}", device=tgt.device)
        err = pred - tgt[:, i : i + 1]
        var_delta = float(delta_by_var.get(name, cfg["delta"])) if delta_by_var else float(cfg["delta"])
        if cfg["type"] == "huber":
            base_map = torch.nn.functional.huber_loss(
                pred,
                tgt[:, i : i + 1],
                delta=float(var_delta),
                reduction="none",
            )
            grad_scale = 1.0  # autograd handles exact derivative.
        else:
            base_map = 0.5 * err * err
            grad_scale = 1.0
        if m is not None:
            if nan_region_policy == "sdf_continuous":
                if d_any is None:
                    raise ValueError("nan_region_policy=sdf_continuous requires distance_any")
                d_signed = torch.where(m > 0.5, d_any, -d_any)
                if sdf_distance_contract != "off":
                    m_np = np.asarray(m.detach().cpu().numpy(), dtype=np.float32)
                    d_np = np.asarray(d_signed.detach().cpu().numpy(), dtype=np.float32)
                    if not is_effective_signed_distance(m_np[:, 0], d_np[:, 0]):
                        msg = (
                            "sdf_continuous contract violation: signed distance has no negative chamber-side values; "
                            "distance_any may be invalid for chamber supervision"
                        )
                        if sdf_distance_contract == "error":
                            raise ValueError(msg)
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                sw = sdf_continuous_weight_map_torch(m, d_signed, sdf_cfg).to(dtype=base_map.dtype)
                if legacy_region_enabled:
                    boundary = (d_any <= boundary_delta).to(dtype=base_map.dtype) * m
                    bulk = (m - boundary).clamp(min=0.0)
                    region_mult = boundary * max(w_boundary / max(w_bulk, 1e-8), 1.0) + bulk
                    outside = (1.0 - m).clamp(min=0.0)
                    sw = sw * (region_mult + outside)
            else:
                if legacy_region_enabled:
                    if d_any is None:
                        raise ValueError("region_weighting.enabled requires distance_any")
                    boundary = (d_any <= boundary_delta).to(dtype=base_map.dtype) * m
                    bulk = (m - boundary).clamp(min=0.0)
                    sw = boundary * w_boundary + bulk * w_bulk
                else:
                    sw = m
            sw = _apply_chamber_override_torch(
                sw,
                mask=m,
                var_name=name,
                chamber_weight_by_var=chamber_weight_by_var,
            )
            if use_robust and name in robust_vars:
                active = sw > 0.5
                if bool(torch.any(active)):
                    vals = tgt[:, i : i + 1]
                    vals_active = vals[active]
                    med = torch.median(vals_active)
                    mad = torch.median(torch.abs(vals_active - med))
                    scale = torch.clamp(mad * float(robust_mad_scale), min=1e-8)
                    rw = 1.0 / (1.0 + torch.abs(vals - med) / scale)
                    rw = torch.clamp(rw, min=float(robust_min_weight), max=1.0)
                    sw = sw * rw
            base = _weighted_reduce_torch(
                base_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
            )
        else:
            sw = torch.ones_like(base_map, dtype=base_map.dtype)
            base = _weighted_reduce_torch(
                base_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
            )

        if region_balance_enabled and name in region_balance_vars:
            if region_masks_torch is None:
                raise ValueError("supervised.region_balance.enabled requires region masks")
            boundary_region = region_masks_torch["boundary_in"].to(dtype=base_map.dtype)
            mid_region = region_masks_torch["plasma_mid"].to(dtype=base_map.dtype)
            deep_region = region_masks_torch["plasma_deep"].to(dtype=base_map.dtype)
            l_boundary = _weighted_mean_count_torch(base_map, sw, boundary_region)
            l_mid = _weighted_mean_count_torch(base_map, sw, mid_region)
            l_deep = _weighted_mean_count_torch(base_map, sw, deep_region)
            region_loss = (
                float(region_balance_w_boundary) * l_boundary
                + float(region_balance_w_mid) * l_mid
                + float(region_balance_w_deep) * l_deep
            )
            if region_balance_mode == "replace":
                base = region_loss
            else:
                base = base + float(region_balance_lambda) * region_loss

        if density_pos_enabled and name in density_pos_vars and density_pos_lambda > 0.0:
            mean_aff, std_aff = density_pos_affine_by_var.get(name, (0.0, 1.0))
            std_safe = float(max(abs(float(std_aff)), 1.0e-12))
            pred_phys = pred * float(std_aff) + float(mean_aff)
            violation_norm = (float(density_pos_floor) - pred_phys) / float(std_safe)
            violation = torch.clamp(violation_norm, min=0.0)
            pos_map = 0.5 * violation * violation
            pos_loss = _weighted_reduce_torch(
                pos_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
            )
            base = base + float(density_pos_lambda) * pos_loss

        if density_rel_enabled and name in density_rel_vars and density_rel_lambda > 0.0:
            mean_aff, std_aff = density_rel_affine_by_var.get(name, (0.0, 1.0))
            eps_var = float(density_rel_eps_by_var.get(name, density_rel_eps_default))
            eps_var = max(eps_var, density_rel_eps_min)
            pred_phys = pred * float(std_aff) + float(mean_aff)
            tgt_phys = tgt[:, i : i + 1] * float(std_aff) + float(mean_aff)
            rel_denom = torch.abs(tgt_phys) + float(eps_var)
            rel_map = torch.abs(pred_phys - tgt_phys) / rel_denom
            rel_loss = _weighted_reduce_torch(
                rel_map,
                sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
            )
            base = base + float(density_rel_lambda) * rel_loss

        base_loss = float(base.detach().cpu().item())
        sigma = _resolve_sigma_for_var(name, base_loss, cfg)
        if cfg["weighting"] == "uncertainty":
            weighted = torch.exp(torch.tensor(-sigma, dtype=torch.float32, device=tgt.device)) * base + float(sigma)
        else:
            fixed_weight = float(fixed_weights_by_var.get(name, 1.0)) if fixed_weights_by_var else 1.0
            weighted = base * float(grad_scale) * float(fixed_weight)
        per_var[name] = float(weighted.detach().cpu().item())
        total = total + weighted

    return total, per_var


def _as_bhw(arr: Any, *, key: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 2:
        out = out[None, ...]
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    if out.ndim != 3:
        raise ValueError(f"{key} must be [B,H,W] or [B,1,H,W], got shape={out.shape}")
    return out.astype(np.float32)


def _resolve_physics_symbol_keys(pred_fields: dict[str, Any], physics_cfg: dict[str, Any] | None) -> tuple[str, str, str]:
    keys = [str(k) for k in pred_fields.keys()]
    symbols = dict(dict(physics_cfg or {}).get("symbols", {}))

    def _resolve(symbol_name: str, *, fallback: str | None = None) -> str | None:
        raw = symbols.get(symbol_name, None)
        if raw is not None:
            key = str(raw)
            if key not in set(keys):
                raise ValueError(
                    f"physics.symbols.{symbol_name}={key} not found in prediction fields; available={keys}"
                )
            return key
        if fallback is not None and fallback in set(keys):
            return fallback
        return None

    density_key = _resolve("density")
    if density_key is None:
        density_key = resolve_density_key(keys, canonical="ne", prefer_linear=True)
    temperature_key = _resolve("temperature", fallback="Te")
    potential_key = _resolve("potential", fallback="phi")
    if density_key is None:
        raise ValueError(
            "physics enabled requires a density field. Provide physics.symbols.density or include ne/log_ne in targets."
        )
    if temperature_key is None:
        raise ValueError(
            "physics enabled requires a temperature field. Provide physics.symbols.temperature or include Te in targets."
        )
    if potential_key is None:
        raise ValueError(
            "physics enabled requires a potential field. Provide physics.symbols.potential or include phi in targets."
        )
    return str(density_key), str(temperature_key), str(potential_key)


def _resolve_numpy_physics_fields(
    pred_fields: dict[str, Any],
    physics_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density_key, temperature_key, potential_key = _resolve_physics_symbol_keys(pred_fields, physics_cfg)
    density = _as_bhw(pred_fields[density_key], key=density_key)
    log_density = density if density_key == "log_ne" else np.log10(np.maximum(density, np.float32(1.0e-30))).astype(np.float32)
    temperature = _as_bhw(pred_fields[temperature_key], key=temperature_key)
    potential = _as_bhw(pred_fields[potential_key], key=potential_key)
    return log_density, temperature, potential


def compose_numpy(
    pred_fields: dict[str, Any],
    physics_cfg: dict[str, Any] | None,
    resolved_terms: list[dict[str, Any]] | None = None,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Compose numpy physics terms using the existing finite-difference implementation."""
    if not physics_cfg or not bool(physics_cfg.get("enabled", False)):
        if "phi" in pred_fields:
            ref = _as_bhw(pred_fields.get("phi"), key="phi")
        elif len(pred_fields) > 0:
            first_key = str(next(iter(pred_fields.keys())))
            ref = _as_bhw(pred_fields[first_key], key=first_key)
        else:
            raise ValueError("compose_numpy requires at least one prediction field")
        phi = np.zeros_like(ref, dtype=np.float32)
        return 0.0, np.zeros_like(phi, dtype=np.float32), _empty_components()
    log_ne, te, phi = _resolve_numpy_physics_fields(pred_fields, physics_cfg)
    eff_cfg = dict(physics_cfg)
    if resolved_terms is not None:
        eff_cfg["resolved_terms"] = resolved_terms
    term_map = {s.name: s for s in resolve_numpy_terms(eff_cfg)}
    eff_cfg["lambda_poisson"] = (
        float(term_map["poisson"].weight) if bool(term_map["poisson"].enabled) else 0.0
    )
    eff_cfg["lambda_bc"] = (
        float(term_map["boundary"].weight) if bool(term_map["boundary"].enabled) else 0.0
    )
    bo_cfg = dict(eff_cfg.get("boundary_operator", {}))
    bo_cfg["lambda"] = (
        float(term_map["boundary_operator"].weight) if bool(term_map["boundary_operator"].enabled) else 0.0
    )
    bo_cfg["enabled"] = bool(bo_cfg.get("enabled", False)) and bool(term_map["boundary_operator"].enabled)
    eff_cfg["boundary_operator"] = bo_cfg

    phys_loss, grad_phi, terms = physics_loss_and_grad(phi=phi, cfg=eff_cfg, log_ne=log_ne, te=te)

    comps = _empty_components()
    comps["physics"] = float(phys_loss)
    comps["poisson"] = float(terms.get("poisson", 0.0))
    comps["boundary"] = float(terms.get("boundary", 0.0))
    comps["boundary_operator"] = float(terms.get("boundary_operator", 0.0))
    comps["rho"] = float(terms.get("rho", 0.0))
    return float(phys_loss), np.asarray(grad_phi, dtype=np.float32), comps


def compose_torch(
    pred_fields: dict[str, Any],
    cond_vec: Any,
    geom_ctx: Any,
    physics_cfg: dict[str, Any] | None,
    boundary_operator_model: Any | None = None,
    supervised_targets: dict[str, Any] | None = None,
    resolved_terms: list[dict[str, Any]] | None = None,
):
    """Compose torch physics terms using the existing autograd-compatible implementation."""

    torch = require_torch()
    if "phi" in pred_fields:
        ref = torch.as_tensor(pred_fields["phi"], dtype=torch.float32)
    elif len(pred_fields) > 0:
        first_key = str(next(iter(pred_fields.keys())))
        ref = torch.as_tensor(pred_fields[first_key], dtype=torch.float32)
    else:
        raise ValueError("compose_torch requires at least one prediction field")
    if not physics_cfg or not bool(physics_cfg.get("enabled", False)):
        return torch.zeros((), dtype=torch.float32, device=ref.device), _empty_components()
    density_key, temperature_key, potential_key = _resolve_physics_symbol_keys(pred_fields, physics_cfg)
    ref = torch.as_tensor(pred_fields[potential_key], dtype=torch.float32, device=ref.device)
    eff_cfg = dict(physics_cfg)
    if resolved_terms is not None:
        eff_cfg["resolved_terms"] = resolved_terms
    term_map = {s.name: s for s in resolve_torch_terms(eff_cfg)}
    eff_cfg["lambda_poisson"] = (
        float(term_map["poisson"].weight) if bool(term_map["poisson"].enabled) else 0.0
    )
    bo_cfg = dict(eff_cfg.get("boundary_operator", {}))
    bo_cfg["lambda"] = (
        float(term_map["boundary_operator"].weight) if bool(term_map["boundary_operator"].enabled) else 0.0
    )
    bo_cfg["enabled"] = bool(bo_cfg.get("enabled", False)) and bool(term_map["boundary_operator"].enabled)
    eff_cfg["boundary_operator"] = bo_cfg

    density_t = torch.as_tensor(pred_fields[density_key], dtype=torch.float32, device=ref.device)
    log_density_t = (
        density_t if density_key == "log_ne" else torch.log10(torch.clamp(density_t, min=1.0e-30))
    )
    physics_pred_fields = dict(pred_fields)
    physics_pred_fields["log_ne"] = log_density_t
    physics_pred_fields["Te"] = torch.as_tensor(pred_fields[temperature_key], dtype=torch.float32, device=ref.device)
    physics_pred_fields["phi"] = ref
    total, terms = physics_terms_torch(
        pred_fields=physics_pred_fields,
        cond_vec=cond_vec,
        geom_ctx=geom_ctx,
        cfg=eff_cfg,
        boundary_operator_model=boundary_operator_model,
        supervised_targets=supervised_targets,
    )

    comps = _empty_components()
    comps["physics"] = float(total.detach().cpu().item())
    comps["poisson"] = float(terms.get("poisson", 0.0))
    comps["boundary"] = float(terms.get("boundary", 0.0))
    comps["boundary_operator"] = float(terms.get("boundary_operator", 0.0))
    comps["rho"] = float(terms.get("rho", 0.0))
    return total, comps


__all__ = [
    "compose_numpy",
    "compose_torch",
    "compose_supervised_numpy",
    "compose_supervised_torch",
]
