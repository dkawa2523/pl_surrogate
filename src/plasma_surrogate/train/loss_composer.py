"""Product data/physics loss composition shared by numpy and torch trainers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from plasma_surrogate.core.target_groups import (
    TargetGroup,
    resolve_target_weight_multipliers,
)
from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys
from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.loss_contract import removed_supervised_keys
from plasma_surrogate.train.loss_protocols import (
    GROUP_WEIGHTING_MODES,
    GROUP_WEIGHTING_NONE,
)
from plasma_surrogate.train.losses import physics_loss_and_grad
from plasma_surrogate.train.physics_terms import resolve_numpy_terms, resolve_torch_terms
from plasma_surrogate.train.torch_losses import physics_terms_torch


_COMPONENT_KEYS = ("data", "physics", "poisson", "boundary", "boundary_operator", "rho")


def _empty_components() -> dict[str, float]:
    return {key: 0.0 for key in _COMPONENT_KEYS}


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _positive_finite_float(raw: Any, *, key_name: str) -> float:
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key_name} must be finite and > 0")
    return value


def _resolve_supervised_cfg(loss_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _dict_or_empty(loss_cfg)
    sup = _dict_or_empty(cfg.get("supervised"))
    mt = _dict_or_empty(cfg.get("multitask"))
    found_removed = removed_supervised_keys(sup)
    if found_removed:
        raise ValueError(
            "product supervised loss supports only the standard data-loss path; "
            f"removed supervised keys={found_removed}"
        )
    if "coord_objective" in sup:
        raise ValueError("supervised.coord_objective is removed")
    if "sample_mean_group_scale" in sup:
        raise ValueError("supervised.sample_mean_group_scale is removed")

    nan_region_policy = str(sup.get("nan_region_policy", "mask_only")).strip().lower()
    if nan_region_policy != "mask_only":
        raise ValueError("supervised.nan_region_policy supports only mask_only in the product loss path")

    fixed_weights_by_var = _dict_or_empty(mt.get("fixed_weights_by_var"))
    target_weights = _dict_or_empty(sup.get("target_weights"))
    if target_weights and not fixed_weights_by_var:
        fixed_weights_by_var = target_weights

    if "base" in sup:
        raise ValueError("supervised.base is removed; use supervised.type")
    if "delta" in sup:
        raise ValueError("supervised.delta is removed; use supervised.huber_delta")

    supervised_type = str(sup.get("type", "mse")).strip().lower()
    if supervised_type not in {"mse", "huber"}:
        raise ValueError("supervised.type must be one of: mse, huber")
    delta = _positive_finite_float(
        sup.get("huber_delta", 1.0),
        key_name="supervised.huber_delta",
    )
    spatial_raw = _dict_or_empty(sup.get("spatial"))
    gradient_weight = float(spatial_raw.get("gradient_weight", 0.0))
    multiscale_weight = float(spatial_raw.get("multiscale_weight", 0.0))
    boundary_weight = float(spatial_raw.get("boundary_weight", 0.0))
    if not np.isfinite(gradient_weight) or gradient_weight < 0.0:
        raise ValueError("supervised.spatial.gradient_weight must be finite and >= 0")
    if not np.isfinite(multiscale_weight) or multiscale_weight < 0.0:
        raise ValueError("supervised.spatial.multiscale_weight must be finite and >= 0")
    if not np.isfinite(boundary_weight) or boundary_weight < 0.0:
        raise ValueError("supervised.spatial.boundary_weight must be finite and >= 0")
    boundary_band_px = float(spatial_raw.get("boundary_band_px", 2.0))
    if not np.isfinite(boundary_band_px) or boundary_band_px <= 0.0:
        raise ValueError("supervised.spatial.boundary_band_px must be finite and > 0")
    raw_scales = spatial_raw.get("multiscale_scales", [2, 4])
    if not isinstance(raw_scales, (list, tuple)) or not raw_scales:
        raise ValueError("supervised.spatial.multiscale_scales must be a non-empty list")
    multiscale_scales = tuple(int(value) for value in raw_scales)
    if (
        any(isinstance(raw, bool) or float(raw) != float(value) for raw, value in zip(raw_scales, multiscale_scales))
        or any(value < 2 for value in multiscale_scales)
        or len(set(multiscale_scales)) != len(multiscale_scales)
    ):
        raise ValueError("supervised.spatial.multiscale_scales must contain unique integers >= 2")
    raw_spacing = spatial_raw.get("gradient_spacing", [1.0, 1.0])
    if not isinstance(raw_spacing, (list, tuple)) or len(raw_spacing) != 2:
        raise ValueError("supervised.spatial.gradient_spacing must be [dy, dx]")
    gradient_spacing = tuple(float(value) for value in raw_spacing)
    if any(not np.isfinite(value) or value <= 0.0 for value in gradient_spacing):
        raise ValueError("supervised.spatial.gradient_spacing values must be finite and > 0")
    gradient_normalization = str(spatial_raw.get("gradient_normalization", "none")).strip().lower()
    if gradient_normalization not in {"none", "target_rms"}:
        raise ValueError("supervised.spatial.gradient_normalization must be one of: none, target_rms")
    gradient_epsilon = _positive_finite_float(
        spatial_raw.get("gradient_epsilon", 0.05),
        key_name="supervised.spatial.gradient_epsilon",
    )
    physical_raw = _dict_or_empty(sup.get("physical_weighting"))
    axisymmetric_volume = bool(physical_raw.get("axisymmetric_volume", False))
    density_source = str(physical_raw.get("density_source", "ne")).strip()
    density_weighted_targets = tuple(
        str(name).strip() for name in physical_raw.get("density_weighted_targets", [])
    )
    if any(not name for name in density_weighted_targets):
        raise ValueError("supervised.physical_weighting.density_weighted_targets must contain names")
    if density_weighted_targets and not density_source:
        raise ValueError("supervised.physical_weighting.density_source must be non-empty")

    return {
        "type": supervised_type,
        "delta": delta,
        "normalization": str(sup.get("normalization", "pixel_mean")).strip().lower(),
        "sample_mean_group_mode": str(sup.get("sample_mean_group_mode", "batch")).strip().lower(),
        "sample_mean_weight_denominator": str(sup.get("sample_mean_weight_denominator", "weighted")).strip().lower(),
        "weighting": str(mt.get("weighting", "fixed")).strip().lower(),
        "fixed_weights_by_var": fixed_weights_by_var,
        "sigma_init": _dict_or_empty(mt.get("sigma_init")),
        "sigma_clamp": tuple(mt.get("sigma_clamp", [-3.0, 3.0])),
        "spatial": {
            "boundary_band_px": boundary_band_px,
            "boundary_weight": boundary_weight,
            "gradient_spacing": gradient_spacing,
            "gradient_normalization": gradient_normalization,
            "gradient_epsilon": gradient_epsilon,
            "gradient_weight": gradient_weight,
            "multiscale_weight": multiscale_weight,
            "multiscale_scales": multiscale_scales,
        },
        "physical_weighting": {
            "axisymmetric_volume": axisymmetric_volume,
            "density_source": density_source,
            "density_weighted_targets": density_weighted_targets,
        },
    }


def _physical_point_weights_numpy(
    *,
    name: str,
    target_fields: dict[str, Any],
    shape: tuple[int, int, int],
    cfg: dict[str, Any],
    point_weight: np.ndarray | None,
    target_affine: dict[str, dict[str, float]] | None,
) -> np.ndarray:
    """Build only the two physically defined point weights used by the product loss."""

    physical = dict(cfg.get("physical_weighting", {}))
    weights = np.ones(shape, dtype=np.float32)
    if bool(physical.get("axisymmetric_volume", False)):
        if point_weight is None:
            raise ValueError("axisymmetric_volume weighting requires a radial point_weight map")
        radial = _as_bhw(point_weight, key="axisymmetric radial point_weight")
        if radial.shape[0] == 1 and shape[0] > 1:
            radial = np.repeat(radial, shape[0], axis=0)
        if radial.shape != shape or np.any(~np.isfinite(radial)) or np.any(radial < 0.0):
            raise ValueError(f"invalid axisymmetric radial point_weight shape/values: {radial.shape}")
        weights *= radial

    if name in set(physical.get("density_weighted_targets", ())):
        source = str(physical.get("density_source", "ne"))
        if source not in target_fields:
            raise ValueError(f"density weighting source {source!r} is not present in targets")
        affine = dict((target_affine or {}).get(source, {}))
        if "mean" not in affine or "scale" not in affine:
            raise ValueError(f"density weighting requires linear inverse affine parameters for {source!r}")
        density_z = _as_bhw(target_fields[source], key=f"target_{source}")
        if density_z.shape != shape:
            raise ValueError(f"density weighting source shape mismatch: expected={shape}, got={density_z.shape}")
        density = density_z.astype(np.float64) * float(affine["scale"]) + float(affine["mean"])
        density = np.maximum(density, 0.0)
        if np.any(~np.isfinite(density)):
            raise ValueError(f"density weighting source {source!r} contains non-finite physical values")
        weights *= density.astype(np.float32)
    return weights


def _resolve_group_weighting_mode(loss_cfg: dict[str, Any] | None) -> str:
    cfg = _dict_or_empty(loss_cfg)
    group_weighting = _dict_or_empty(cfg.get("group_weighting"))
    mode = str(group_weighting.get("mode", GROUP_WEIGHTING_NONE)).strip().lower()
    if mode not in GROUP_WEIGHTING_MODES:
        raise ValueError(
            "train.loss.group_weighting.mode must be one of: "
            f"{', '.join(GROUP_WEIGHTING_MODES)}"
        )
    return mode


def _safe_group_loss_suffix(name: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    return suffix or "group"


def _resolve_loss_target_multipliers(
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None,
) -> tuple[str, dict[str, float], dict[str, TargetGroup]]:
    mode = _resolve_group_weighting_mode(loss_cfg)
    cfg = _dict_or_empty(loss_cfg)
    group_weighting = _dict_or_empty(cfg.get("group_weighting"))
    multipliers, groups = resolve_target_weight_multipliers(
        output_vars=[str(name) for name in y_order],
        target_role_schema=_dict_or_empty(cfg.get("target_role_schema")),
        mode=mode,
        group_weights=_dict_or_empty(group_weighting.get("weights")),
        context="train.loss.group_weighting",
    )
    return mode, multipliers, groups


def _append_group_loss_breakdown(
    per_var_loss: dict[str, float],
    *,
    groups: dict[str, TargetGroup],
) -> None:
    for group in groups.values():
        values = [float(per_var_loss[target]) for target in group.targets if target in per_var_loss]
        if not values:
            continue
        per_var_loss[f"loss_supervised_group_{_safe_group_loss_suffix(group.name)}"] = float(sum(values))


def _validate_var_payload(payload: dict[str, Any], *, key_name: str, y_order: list[str]) -> None:
    unknown = sorted(set(str(key) for key in payload.keys()) - set(y_order))
    if unknown:
        raise ValueError(f"{key_name} contains unknown vars: {unknown}")


def _resolve_sigma_for_var(name: str, base_loss: float, cfg: dict[str, Any]) -> float:
    if cfg["weighting"] != "uncertainty":
        return 0.0
    init = cfg.get("sigma_init", {})
    raw = init.get(name, np.log(max(base_loss, 1e-8)))
    lo, hi = cfg.get("sigma_clamp", (-3.0, 3.0))
    return float(np.clip(float(raw), float(lo), float(hi)))


def _huber_loss_and_grad(err: np.ndarray, *, delta: float) -> tuple[np.ndarray, np.ndarray]:
    d = _positive_finite_float(delta, key_name="supervised.huber_delta")
    abs_err = np.abs(err)
    quad = abs_err <= d
    loss = np.where(quad, 0.5 * err * err, d * (abs_err - 0.5 * d))
    grad = np.where(quad, err, d * np.sign(err))
    return loss.astype(np.float32), grad.astype(np.float32)


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
    if normalization not in {"pixel_mean", "sample_mean", "none"}:
        raise ValueError(f"Unsupported supervised.normalization: {normalization}")
    if weight_denominator not in {"weighted", "count"}:
        raise ValueError("supervised.sample_mean_weight_denominator must be one of: weighted, count")
    active = sw > 0.0
    safe_loss_map = np.where(active, loss_map, 0.0).astype(np.float32)
    safe_grad_map = np.where(active, grad_map, 0.0).astype(np.float32)
    weighted_loss = (safe_loss_map * sw).astype(np.float32)
    weighted_grad = (safe_grad_map * sw).astype(np.float32)
    if normalization == "none":
        return float(np.sum(weighted_loss)), weighted_grad
    if normalization == "sample_mean":
        batch = int(loss_map.shape[0])
        numer = np.sum(weighted_loss, axis=(1, 2))
        if weight_denominator == "count":
            denom = np.maximum(np.sum((sw > 0.0).astype(np.float32), axis=(1, 2)), 1.0)
        else:
            denom = np.maximum(np.sum(sw, axis=(1, 2)), 1.0e-12)
        per_sample = numer / denom
        grad = weighted_grad / denom[:, None, None]
        if group_ids is None:
            return float(np.mean(per_sample)), (grad / float(max(batch, 1))).astype(np.float32)
        if group_mode != "batch":
            raise ValueError("supervised.sample_mean_group_mode must be one of: batch")
        gids = np.asarray(group_ids).reshape(-1)
        if gids.shape[0] != batch:
            raise ValueError(f"group_ids length mismatch: expected {batch}, got {gids.shape[0]}")
        _, inv = np.unique(gids, return_inverse=True)
        n_groups = int(np.max(inv) + 1) if inv.size > 0 else 0
        if n_groups <= 0:
            return float(np.mean(per_sample)), (grad / float(max(batch, 1))).astype(np.float32)
        group_means = np.zeros((n_groups,), dtype=np.float32)
        group_sizes = np.zeros((n_groups,), dtype=np.float32)
        for gi in range(n_groups):
            active = inv == gi
            if not np.any(active):
                continue
            group_means[gi] = float(np.mean(per_sample[active]))
            group_sizes[gi] = float(np.sum(active))
        group_factor = np.zeros((batch,), dtype=np.float32)
        for gi in range(n_groups):
            active = inv == gi
            if np.any(active):
                group_factor[active] = 1.0 / float(max(n_groups * group_sizes[gi], 1.0))
        return float(np.mean(group_means)), (grad * group_factor[:, None, None]).astype(np.float32)
    denom = float(max(float(np.sum(sw)), 1.0e-12))
    return float(np.sum(weighted_loss) / denom), (weighted_grad / denom).astype(np.float32)


def _loss_map_and_grad_numpy(err: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if cfg["type"] == "huber":
        return _huber_loss_and_grad(err, delta=float(cfg["delta"]))
    return (0.5 * err * err).astype(np.float32), err.astype(np.float32)


def _sanitize_supervised_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    sw: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(sw, dtype=np.float32)
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"mask contains non-finite values for {name}")
    if np.any(weights < 0.0):
        raise ValueError(f"mask contains negative values for {name}")
    active = weights > 0.0
    invalid_active = active & (~np.isfinite(pred) | ~np.isfinite(target))
    if np.any(invalid_active):
        count = int(np.sum(invalid_active))
        raise ValueError(f"non-finite prediction/target on active mask for {name}: count={count}")
    pred_safe = np.where(active, pred, 0.0).astype(np.float32)
    target_safe = np.where(active, target, 0.0).astype(np.float32)
    return pred_safe, target_safe, weights


def _require_nonempty_case_weights_numpy(
    weights: np.ndarray,
    *,
    name: str,
    component: str,
) -> None:
    active_per_case = np.sum(np.asarray(weights) > 0.0, axis=(1, 2))
    empty = np.flatnonzero(active_per_case <= 0)
    if empty.size:
        raise ValueError(
            f"supervised {component} has no active samples for {name}: "
            f"case_indices={empty.astype(int).tolist()}"
        )


def _boundary_weights_numpy(
    distance_any: np.ndarray | None,
    sw: np.ndarray,
    *,
    band_px: float,
    name: str,
) -> np.ndarray:
    if distance_any is None:
        raise ValueError("supervised.spatial.boundary_weight > 0 requires distance_any")
    distance = _as_bhw(distance_any, key="distance_any")
    if distance.shape[0] == 1 and sw.shape[0] > 1:
        distance = np.repeat(distance, sw.shape[0], axis=0)
    if distance.shape != sw.shape:
        raise ValueError(f"distance_any shape mismatch for {name}: expected {sw.shape}, got {distance.shape}")
    active = sw > 0.0
    invalid_active = active & ~np.isfinite(distance)
    if np.any(invalid_active):
        count = int(np.sum(invalid_active))
        raise ValueError(f"distance_any is non-finite on active mask for {name}: count={count}")
    boundary = active & (np.abs(np.where(active, distance, 0.0)) <= float(band_px))
    boundary_weights = np.where(boundary, sw, 0.0).astype(np.float32)
    _require_nonempty_case_weights_numpy(
        boundary_weights,
        name=name,
        component="boundary band",
    )
    return boundary_weights


def _gradient_loss_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    sw: np.ndarray,
    *,
    cfg: dict[str, Any],
    normalization: str,
    weight_denominator: str,
    group_ids: np.ndarray | None,
    spacing: tuple[float, float],
    gradient_normalization: str,
    gradient_epsilon: float,
) -> tuple[float, np.ndarray]:
    edge_counts = np.zeros((int(pred.shape[0]),), dtype=np.int64)
    if int(pred.shape[1]) >= 2:
        edge_counts += np.sum(
            np.minimum(sw[:, 1:, :], sw[:, :-1, :]) > 0.0,
            axis=(1, 2),
        )
    if int(pred.shape[2]) >= 2:
        edge_counts += np.sum(
            np.minimum(sw[:, :, 1:], sw[:, :, :-1]) > 0.0,
            axis=(1, 2),
        )
    empty = np.flatnonzero(edge_counts <= 0)
    if empty.size:
        raise ValueError(
            "supervised spatial gradient has no active adjacent pixels: "
            f"case_indices={empty.astype(int).tolist()}"
        )
    losses: list[float] = []
    grads: list[np.ndarray] = []
    for axis, axis_spacing in zip((1, 2), spacing):
        if int(pred.shape[axis]) < 2:
            continue
        pred_diff = np.diff(pred, axis=axis).astype(np.float32)
        target_diff = np.diff(target, axis=axis).astype(np.float32)
        if axis == 1:
            pair_sw = np.minimum(sw[:, 1:, :], sw[:, :-1, :]).astype(np.float32)
        else:
            pair_sw = np.minimum(sw[:, :, 1:], sw[:, :, :-1]).astype(np.float32)
        scale = np.ones((int(pred.shape[0]), 1, 1), dtype=np.float32)
        if gradient_normalization == "target_rms":
            target_gradient = np.where(
                pair_sw > 0.0,
                target_diff / float(axis_spacing),
                0.0,
            ).astype(np.float32)
            weighted_energy = np.sum(pair_sw * target_gradient * target_gradient, axis=(1, 2))
            weight_sum = np.sum(pair_sw, axis=(1, 2))
            scale[:, 0, 0] = np.maximum(
                np.sqrt(weighted_energy / np.maximum(weight_sum, 1.0)),
                float(gradient_epsilon),
            )
        loss_map, grad_map = _loss_map_and_grad_numpy(
            (pred_diff - target_diff) / float(axis_spacing) / scale,
            cfg,
        )
        loss, edge_grad = _weighted_reduce_numpy(
            loss_map,
            grad_map,
            pair_sw,
            normalization=normalization,
            weight_denominator=weight_denominator,
            group_ids=group_ids,
        )
        grad = np.zeros_like(pred, dtype=np.float32)
        edge_grad = edge_grad / float(axis_spacing) / scale
        if axis == 1:
            grad[:, 1:, :] += edge_grad
            grad[:, :-1, :] -= edge_grad
        else:
            grad[:, :, 1:] += edge_grad
            grad[:, :, :-1] -= edge_grad
        losses.append(float(loss))
        grads.append(grad)
    if not losses:
        return 0.0, np.zeros_like(pred, dtype=np.float32)
    return float(np.mean(losses)), (np.sum(grads, axis=0) / float(len(grads))).astype(np.float32)


def _masked_pool_numpy(
    values: np.ndarray,
    sw: np.ndarray,
    *,
    scale: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, int, int]]:
    batch, height, width = values.shape
    out_h = (height + int(scale) - 1) // int(scale)
    out_w = (width + int(scale) - 1) // int(scale)
    padded_h = out_h * int(scale)
    padded_w = out_w * int(scale)
    pad_h = padded_h - height
    pad_w = padded_w - width
    values_pad = np.pad(values, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0.0)
    valid_pad = np.pad((sw > 0.0).astype(np.float32), ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0.0)
    value_blocks = values_pad.reshape(batch, out_h, scale, out_w, scale)
    valid_blocks = valid_pad.reshape(batch, out_h, scale, out_w, scale)
    valid_count = np.sum(valid_blocks, axis=(2, 4), keepdims=True).astype(np.float32)
    safe_count = np.maximum(valid_count, 1.0).astype(np.float32)
    pooled = (np.sum(value_blocks * valid_blocks, axis=(2, 4), keepdims=True) / safe_count)[:, :, 0, :, 0]
    valid_fraction = (valid_count[:, :, 0, :, 0] / float(scale * scale)).astype(np.float32)
    return pooled.astype(np.float32), valid_fraction, (valid_blocks, safe_count, height, width)


def _multiscale_loss_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    sw: np.ndarray,
    *,
    cfg: dict[str, Any],
    normalization: str,
    weight_denominator: str,
    group_ids: np.ndarray | None,
    scales: tuple[int, ...],
) -> tuple[float, np.ndarray]:
    losses: list[float] = []
    grads: list[np.ndarray] = []
    for scale in scales:
        pred_pool, pool_sw, cache = _masked_pool_numpy(pred, sw, scale=scale)
        target_pool, _, _ = _masked_pool_numpy(target, sw, scale=scale)
        loss_map, grad_map = _loss_map_and_grad_numpy(pred_pool - target_pool, cfg)
        loss, pool_grad = _weighted_reduce_numpy(
            loss_map,
            grad_map,
            pool_sw,
            normalization=normalization,
            weight_denominator=weight_denominator,
            group_ids=group_ids,
        )
        valid_blocks, safe_count, height, width = cache
        expanded = pool_grad[:, :, None, :, None] * valid_blocks / safe_count
        padded_grad = expanded.reshape(pred.shape[0], valid_blocks.shape[1] * scale, valid_blocks.shape[3] * scale)
        grad = padded_grad[:, :height, :width].astype(np.float32)
        losses.append(float(loss))
        grads.append(grad)
    return float(np.mean(losses)), (np.sum(grads, axis=0) / float(len(grads))).astype(np.float32)


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
    point_weight: np.ndarray | None = None,
    target_affine: dict[str, dict[str, float]] | None = None,
) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
    """Compose the product supervised loss for grid outputs."""

    cfg = _resolve_supervised_cfg(loss_cfg)
    mask_arr = None if mask is None else _as_bhw(mask, key="mask")
    normalization = str(cfg.get("normalization", "pixel_mean")).strip().lower()
    sample_mean_group_mode = str(cfg.get("sample_mean_group_mode", "batch")).strip().lower()
    sample_mean_weight_denominator = str(cfg.get("sample_mean_weight_denominator", "weighted")).strip().lower()
    if sample_mean_group_mode != "batch":
        raise ValueError("supervised.sample_mean_group_mode must be one of: batch")

    _mode, target_multipliers, target_groups = _resolve_loss_target_multipliers(
        y_order=y_order,
        loss_cfg=loss_cfg,
    )

    fixed_weights_by_var = _dict_or_empty(cfg.get("fixed_weights_by_var"))
    _validate_var_payload(fixed_weights_by_var, key_name="multitask.fixed_weights_by_var", y_order=y_order)
    if cfg["weighting"] != "fixed" and fixed_weights_by_var:
        raise ValueError("multitask.fixed_weights_by_var requires multitask.weighting=fixed")

    grads: dict[str, np.ndarray] = {}
    per_var_loss: dict[str, float] = {}
    total = 0.0
    for name in y_order:
        pred = _as_bhw(pred_fields[name], key=f"pred_{name}")
        target = _as_bhw(target_fields[name], key=f"target_{name}")
        if pred.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch for {name}: pred={pred.shape}, target={target.shape}")
        sw = np.ones_like(pred, dtype=np.float32)
        if mask_arr is not None:
            sw = mask_arr
            if sw.shape[0] == 1 and pred.shape[0] > 1:
                sw = np.repeat(sw, pred.shape[0], axis=0)
            if sw.shape != pred.shape:
                raise ValueError(f"mask shape mismatch for {name}: expected {pred.shape}, got {sw.shape}")
        sw = sw * _physical_point_weights_numpy(
            name=name,
            target_fields=target_fields,
            shape=pred.shape,
            cfg=cfg,
            point_weight=point_weight,
            target_affine=target_affine,
        )
        pred, target, sw = _sanitize_supervised_numpy(pred, target, sw, name=name)
        if normalization == "sample_mean":
            _require_nonempty_case_weights_numpy(
                sw,
                name=name,
                component="point loss",
            )
        loss_map, grad_map = _loss_map_and_grad_numpy((pred - target).astype(np.float32), cfg)

        point_loss, grad = _weighted_reduce_numpy(
            loss_map,
            grad_map,
            sw,
            normalization=normalization,
            weight_denominator=sample_mean_weight_denominator,
            group_ids=group_ids,
            group_mode=sample_mean_group_mode,
        )
        spatial_cfg = dict(cfg["spatial"])
        boundary_loss = 0.0
        gradient_loss = 0.0
        multiscale_loss = 0.0
        if float(spatial_cfg["boundary_weight"]) > 0.0:
            boundary_sw = _boundary_weights_numpy(
                distance_any,
                sw,
                band_px=float(spatial_cfg["boundary_band_px"]),
                name=name,
            )
            boundary_loss, boundary_grad = _weighted_reduce_numpy(
                loss_map,
                grad_map,
                boundary_sw,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                group_mode=sample_mean_group_mode,
            )
            grad = grad + float(spatial_cfg["boundary_weight"]) * boundary_grad
        if float(spatial_cfg["gradient_weight"]) > 0.0:
            gradient_loss, gradient_grad = _gradient_loss_numpy(
                pred,
                target,
                sw,
                cfg=cfg,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                spacing=tuple(spatial_cfg["gradient_spacing"]),
                gradient_normalization=str(spatial_cfg["gradient_normalization"]),
                gradient_epsilon=float(spatial_cfg["gradient_epsilon"]),
            )
            grad = grad + float(spatial_cfg["gradient_weight"]) * gradient_grad
        if float(spatial_cfg["multiscale_weight"]) > 0.0:
            multiscale_loss, multiscale_grad = _multiscale_loss_numpy(
                pred,
                target,
                sw,
                cfg=cfg,
                normalization=normalization,
                weight_denominator=sample_mean_weight_denominator,
                group_ids=group_ids,
                scales=tuple(spatial_cfg["multiscale_scales"]),
            )
            grad = grad + float(spatial_cfg["multiscale_weight"]) * multiscale_grad
        base_loss = float(
            point_loss
            + float(spatial_cfg["boundary_weight"]) * boundary_loss
            + float(spatial_cfg["gradient_weight"]) * gradient_loss
            + float(spatial_cfg["multiscale_weight"]) * multiscale_loss
        )
        sigma = _resolve_sigma_for_var(name, base_loss, cfg)
        if cfg["weighting"] == "uncertainty":
            weighted = float(np.exp(-sigma) * base_loss + sigma)
            grad = (grad * float(np.exp(-sigma))).astype(np.float32)
        else:
            fixed_weight = float(fixed_weights_by_var.get(name, 1.0)) if fixed_weights_by_var else 1.0
            weighted = float(base_loss * fixed_weight)
            grad = (grad * fixed_weight).astype(np.float32)
        target_multiplier = float(target_multipliers.get(str(name), 1.0))
        weighted = float(weighted * target_multiplier)
        grad = (grad * target_multiplier).astype(np.float32)
        component_factor = float(target_multiplier)
        if cfg["weighting"] == "uncertainty":
            component_factor *= float(np.exp(-sigma))
        elif fixed_weights_by_var:
            component_factor *= float(fixed_weights_by_var.get(name, 1.0))
        per_var_loss[name] = weighted
        spatial_enabled = any(
            float(spatial_cfg[key]) > 0.0
            for key in ("boundary_weight", "gradient_weight", "multiscale_weight")
        )
        if spatial_enabled:
            per_var_loss[f"loss_supervised_point_{name}"] = float(point_loss * component_factor)
            per_var_loss[f"loss_supervised_spatial_boundary_{name}"] = float(
                boundary_loss * float(spatial_cfg["boundary_weight"]) * component_factor
            )
            per_var_loss[f"loss_supervised_spatial_gradient_{name}"] = float(
                gradient_loss * float(spatial_cfg["gradient_weight"]) * component_factor
            )
            per_var_loss[f"loss_supervised_spatial_multiscale_{name}"] = float(
                multiscale_loss * float(spatial_cfg["multiscale_weight"]) * component_factor
            )
        grads[name] = grad.astype(np.float32)
        total += weighted
    _append_group_loss_breakdown(per_var_loss, groups=target_groups)
    return float(total), grads, per_var_loss


def _as_bchw(x: Any, *, key: str, device: Any | None = None):
    torch = require_torch()
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[None, None, ...]
    if tensor.ndim == 3:
        tensor = tensor[:, None, ...]
    if tensor.ndim != 4:
        raise ValueError(f"{key} must be [B,H,W] or [B,1,H,W], got shape={tuple(tensor.shape)}")
    return tensor


def _weighted_reduce_torch(base_map, sw, *, normalization: str, weight_denominator: str = "weighted"):
    torch = require_torch()
    if normalization not in {"pixel_mean", "sample_mean", "none"}:
        raise ValueError(f"Unsupported supervised.normalization: {normalization}")
    if weight_denominator not in {"weighted", "count"}:
        raise ValueError("supervised.sample_mean_weight_denominator must be one of: weighted, count")
    safe_base_map = torch.where(sw > 0.0, base_map, torch.zeros_like(base_map))
    weighted = safe_base_map * sw
    if normalization == "none":
        return torch.sum(weighted)
    if normalization == "sample_mean":
        numer = torch.sum(weighted, dim=(1, 2, 3))
        if weight_denominator == "count":
            denom = torch.clamp(torch.sum((sw > 0.0).to(dtype=base_map.dtype), dim=(1, 2, 3)), min=1.0)
        else:
            denom = torch.clamp(torch.sum(sw, dim=(1, 2, 3)), min=1.0e-12)
        return torch.mean(numer / denom)
    denom = torch.clamp(torch.sum(sw), min=1.0e-12)
    return torch.sum(weighted) / denom


def _loss_map_torch(pred, target, cfg: dict[str, Any]):
    torch = require_torch()
    if cfg["type"] == "huber":
        return torch.nn.functional.huber_loss(pred, target, delta=float(cfg["delta"]), reduction="none")
    return 0.5 * (pred - target) ** 2


def _sanitize_supervised_torch(pred, target, sw, *, name: str):
    torch = require_torch()
    if bool(torch.any(~torch.isfinite(sw)).detach().cpu().item()):
        raise ValueError(f"mask contains non-finite values for {name}")
    if bool(torch.any(sw < 0.0).detach().cpu().item()):
        raise ValueError(f"mask contains negative values for {name}")
    active = sw > 0.0
    invalid_active = active & (~torch.isfinite(pred) | ~torch.isfinite(target))
    if bool(torch.any(invalid_active).detach().cpu().item()):
        count = int(torch.sum(invalid_active).detach().cpu().item())
        raise ValueError(f"non-finite prediction/target on active mask for {name}: count={count}")
    pred_safe = torch.where(active, pred, torch.zeros_like(pred))
    target_safe = torch.where(active, target, torch.zeros_like(target))
    return pred_safe, target_safe, sw


def _require_nonempty_case_weights_torch(weights, *, name: str, component: str) -> None:
    torch = require_torch()
    active_per_case = torch.sum(weights > 0.0, dim=(1, 2, 3))
    empty = torch.nonzero(active_per_case <= 0, as_tuple=False).reshape(-1)
    if int(empty.numel()) > 0:
        raise ValueError(
            f"supervised {component} has no active samples for {name}: "
            f"case_indices={empty.detach().cpu().to(dtype=torch.int64).tolist()}"
        )


def _boundary_weights_torch(distance_any, sw, *, band_px: float, name: str):
    torch = require_torch()
    if distance_any is None:
        raise ValueError("supervised.spatial.boundary_weight > 0 requires distance_any")
    distance = _as_bchw(distance_any, key="distance_any", device=sw.device)
    if int(distance.shape[0]) == 1 and int(sw.shape[0]) > 1:
        distance = distance.expand(int(sw.shape[0]), -1, -1, -1)
    if tuple(distance.shape) != tuple(sw.shape):
        raise ValueError(
            f"distance_any shape mismatch for {name}: expected {tuple(sw.shape)}, got {tuple(distance.shape)}"
        )
    active = sw > 0.0
    invalid_active = active & ~torch.isfinite(distance)
    if bool(torch.any(invalid_active).detach().cpu().item()):
        count = int(torch.sum(invalid_active).detach().cpu().item())
        raise ValueError(f"distance_any is non-finite on active mask for {name}: count={count}")
    distance_safe = torch.where(active, distance, torch.zeros_like(distance))
    boundary = active & (torch.abs(distance_safe) <= float(band_px))
    boundary_weights = torch.where(boundary, sw, torch.zeros_like(sw))
    _require_nonempty_case_weights_torch(
        boundary_weights,
        name=name,
        component="boundary band",
    )
    return boundary_weights


def _gradient_loss_torch(
    pred,
    target,
    sw,
    *,
    cfg: dict[str, Any],
    normalization: str,
    weight_denominator: str,
    spacing: tuple[float, float],
    gradient_normalization: str,
    gradient_epsilon: float,
):
    torch = require_torch()
    edge_counts = torch.zeros((int(pred.shape[0]),), dtype=torch.int64, device=pred.device)
    if int(pred.shape[-2]) >= 2:
        edge_counts = edge_counts + torch.sum(
            torch.minimum(sw[:, :, 1:, :], sw[:, :, :-1, :]) > 0.0,
            dim=(1, 2, 3),
        )
    if int(pred.shape[-1]) >= 2:
        edge_counts = edge_counts + torch.sum(
            torch.minimum(sw[:, :, :, 1:], sw[:, :, :, :-1]) > 0.0,
            dim=(1, 2, 3),
        )
    empty = torch.nonzero(edge_counts <= 0, as_tuple=False).reshape(-1)
    if int(empty.numel()) > 0:
        raise ValueError(
            "supervised spatial gradient has no active adjacent pixels: "
            f"case_indices={empty.detach().cpu().to(dtype=torch.int64).tolist()}"
        )
    losses = []
    if int(pred.shape[-2]) >= 2:
        pred_diff = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) / float(spacing[0])
        target_diff = (target[:, :, 1:, :] - target[:, :, :-1, :]) / float(spacing[0])
        pair_sw = torch.minimum(sw[:, :, 1:, :], sw[:, :, :-1, :])
        scale = torch.ones((int(pred.shape[0]), 1, 1, 1), dtype=pred.dtype, device=pred.device)
        if gradient_normalization == "target_rms":
            target_active = torch.where(pair_sw > 0.0, target_diff, torch.zeros_like(target_diff))
            energy = torch.sum(pair_sw * target_active.square(), dim=(1, 2, 3), keepdim=True)
            weight = torch.sum(pair_sw, dim=(1, 2, 3), keepdim=True)
            scale = torch.clamp(torch.sqrt(energy / torch.clamp(weight, min=1.0)), min=float(gradient_epsilon))
        losses.append(
            _weighted_reduce_torch(
                _loss_map_torch(pred_diff / scale, target_diff / scale, cfg),
                pair_sw,
                normalization=normalization,
                weight_denominator=weight_denominator,
            )
        )
    if int(pred.shape[-1]) >= 2:
        pred_diff = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) / float(spacing[1])
        target_diff = (target[:, :, :, 1:] - target[:, :, :, :-1]) / float(spacing[1])
        pair_sw = torch.minimum(sw[:, :, :, 1:], sw[:, :, :, :-1])
        scale = torch.ones((int(pred.shape[0]), 1, 1, 1), dtype=pred.dtype, device=pred.device)
        if gradient_normalization == "target_rms":
            target_active = torch.where(pair_sw > 0.0, target_diff, torch.zeros_like(target_diff))
            energy = torch.sum(pair_sw * target_active.square(), dim=(1, 2, 3), keepdim=True)
            weight = torch.sum(pair_sw, dim=(1, 2, 3), keepdim=True)
            scale = torch.clamp(torch.sqrt(energy / torch.clamp(weight, min=1.0)), min=float(gradient_epsilon))
        losses.append(
            _weighted_reduce_torch(
                _loss_map_torch(pred_diff / scale, target_diff / scale, cfg),
                pair_sw,
                normalization=normalization,
                weight_denominator=weight_denominator,
            )
        )
    if not losses:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return torch.stack(losses).mean()


def _masked_pool_torch(values, sw, *, scale: int):
    torch = require_torch()
    height, width = int(values.shape[-2]), int(values.shape[-1])
    pad_h = (-height) % int(scale)
    pad_w = (-width) % int(scale)
    values = torch.nn.functional.pad(values, (0, pad_w, 0, pad_h), value=0.0)
    valid = torch.nn.functional.pad(
        (sw > 0.0).to(dtype=values.dtype),
        (0, pad_w, 0, pad_h),
        value=0.0,
    )
    valid_fraction = torch.nn.functional.avg_pool2d(valid, kernel_size=int(scale), stride=int(scale))
    weighted_mean = torch.nn.functional.avg_pool2d(
        values * valid,
        kernel_size=int(scale),
        stride=int(scale),
    )
    pooled = weighted_mean / torch.clamp(valid_fraction, min=1.0e-12)
    return pooled, valid_fraction


def _multiscale_loss_torch(
    pred,
    target,
    sw,
    *,
    cfg: dict[str, Any],
    normalization: str,
    weight_denominator: str,
    scales: tuple[int, ...],
):
    torch = require_torch()
    losses = []
    for scale in scales:
        pred_pool, pool_sw = _masked_pool_torch(pred, sw, scale=int(scale))
        target_pool, _ = _masked_pool_torch(target, sw, scale=int(scale))
        losses.append(
            _weighted_reduce_torch(
                _loss_map_torch(pred_pool, target_pool, cfg),
                pool_sw,
                normalization=normalization,
                weight_denominator=weight_denominator,
            )
        )
    return torch.stack(losses).mean()


def compose_supervised_torch(
    pred_fields: dict[str, Any],
    target_fields: Any,
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None = None,
    mask: Any | None = None,
    distance_any: Any | None = None,
    point_weight: Any | None = None,
    target_affine: dict[str, dict[str, float]] | None = None,
):
    """Compose the product supervised torch loss."""

    torch = require_torch()
    cfg = _resolve_supervised_cfg(loss_cfg)
    first_pred = pred_fields[str(y_order[0])] if y_order else None
    pred_device = torch.as_tensor(first_pred, dtype=torch.float32).device if first_pred is not None else None
    target = torch.as_tensor(target_fields, dtype=torch.float32, device=pred_device)
    if target.ndim != 4:
        raise ValueError(f"target_fields must be [B,C,H,W], got {tuple(target.shape)}")
    if int(target.shape[1]) != len(y_order):
        raise ValueError(f"target channel mismatch: expected {len(y_order)}, got {int(target.shape[1])}")

    mask_tensor = None if mask is None else _as_bchw(mask, key="mask", device=target.device)
    if mask_tensor is not None and int(mask_tensor.shape[0]) == 1 and int(target.shape[0]) > 1:
        mask_tensor = mask_tensor.expand(int(target.shape[0]), -1, -1, -1)

    physical_cfg = dict(cfg.get("physical_weighting", {}))
    radial_tensor = None
    if bool(physical_cfg.get("axisymmetric_volume", False)):
        if point_weight is None:
            raise ValueError("axisymmetric_volume weighting requires a radial point_weight map")
        radial_tensor = _as_bchw(point_weight, key="axisymmetric radial point_weight", device=target.device)
        if int(radial_tensor.shape[0]) == 1 and int(target.shape[0]) > 1:
            radial_tensor = radial_tensor.expand(int(target.shape[0]), -1, -1, -1)
        if tuple(radial_tensor.shape) != (int(target.shape[0]), 1, int(target.shape[2]), int(target.shape[3])):
            raise ValueError(f"axisymmetric radial point_weight shape mismatch: {tuple(radial_tensor.shape)}")
        if not bool(torch.all(torch.isfinite(radial_tensor))) or bool(torch.any(radial_tensor < 0.0)):
            raise ValueError("axisymmetric radial point_weight must be finite and non-negative")

    normalization = str(cfg.get("normalization", "pixel_mean")).strip().lower()
    weight_denominator = str(cfg.get("sample_mean_weight_denominator", "weighted")).strip().lower()
    _mode, target_multipliers, target_groups = _resolve_loss_target_multipliers(
        y_order=y_order,
        loss_cfg=loss_cfg,
    )

    fixed_weights_by_var = _dict_or_empty(cfg.get("fixed_weights_by_var"))
    _validate_var_payload(fixed_weights_by_var, key_name="multitask.fixed_weights_by_var", y_order=y_order)
    if cfg["weighting"] != "fixed" and fixed_weights_by_var:
        raise ValueError("multitask.fixed_weights_by_var requires multitask.weighting=fixed")

    total = torch.zeros((), dtype=torch.float32, device=target.device)
    per_var: dict[str, float] = {}
    for idx, name in enumerate(y_order):
        pred = _as_bchw(pred_fields[name], key=f"pred_{name}", device=target.device)
        if pred.shape != target[:, idx : idx + 1].shape:
            raise ValueError(
                f"prediction shape mismatch for {name}: "
                f"expected {tuple(target[:, idx : idx + 1].shape)}, got {tuple(pred.shape)}"
            )
        target_i = target[:, idx : idx + 1]
        sw = torch.ones_like(pred, dtype=pred.dtype)
        if mask_tensor is not None:
            sw = mask_tensor.to(dtype=pred.dtype)
            if tuple(sw.shape) != tuple(pred.shape):
                raise ValueError(f"mask shape mismatch for {name}: expected {tuple(pred.shape)}, got {tuple(sw.shape)}")
        if radial_tensor is not None:
            sw = sw * radial_tensor.to(dtype=pred.dtype)
        if name in set(physical_cfg.get("density_weighted_targets", ())):
            source = str(physical_cfg.get("density_source", "ne"))
            if source not in y_order:
                raise ValueError(f"density weighting source {source!r} is not present in targets")
            affine = dict((target_affine or {}).get(source, {}))
            if "mean" not in affine or "scale" not in affine:
                raise ValueError(f"density weighting requires linear inverse affine parameters for {source!r}")
            source_idx = y_order.index(source)
            density = target[:, source_idx : source_idx + 1] * float(affine["scale"]) + float(affine["mean"])
            density = torch.clamp(density, min=0.0)
            if not bool(torch.all(torch.isfinite(density))):
                raise ValueError(f"density weighting source {source!r} contains non-finite physical values")
            sw = sw * density.to(dtype=pred.dtype)
        pred, target_i, sw = _sanitize_supervised_torch(pred, target_i, sw, name=name)
        if normalization == "sample_mean":
            _require_nonempty_case_weights_torch(
                sw,
                name=name,
                component="point loss",
            )
        base_map = _loss_map_torch(pred, target_i, cfg)

        point = _weighted_reduce_torch(base_map, sw, normalization=normalization, weight_denominator=weight_denominator)
        spatial_cfg = dict(cfg["spatial"])
        boundary = torch.zeros((), dtype=point.dtype, device=point.device)
        gradient = torch.zeros((), dtype=point.dtype, device=point.device)
        multiscale = torch.zeros((), dtype=point.dtype, device=point.device)
        if float(spatial_cfg["boundary_weight"]) > 0.0:
            boundary_sw = _boundary_weights_torch(
                distance_any,
                sw,
                band_px=float(spatial_cfg["boundary_band_px"]),
                name=name,
            )
            boundary = _weighted_reduce_torch(
                base_map,
                boundary_sw,
                normalization=normalization,
                weight_denominator=weight_denominator,
            )
        if float(spatial_cfg["gradient_weight"]) > 0.0:
            gradient = _gradient_loss_torch(
                pred,
                target_i,
                sw,
                cfg=cfg,
                normalization=normalization,
                weight_denominator=weight_denominator,
                spacing=tuple(spatial_cfg["gradient_spacing"]),
                gradient_normalization=str(spatial_cfg["gradient_normalization"]),
                gradient_epsilon=float(spatial_cfg["gradient_epsilon"]),
            )
        if float(spatial_cfg["multiscale_weight"]) > 0.0:
            multiscale = _multiscale_loss_torch(
                pred,
                target_i,
                sw,
                cfg=cfg,
                normalization=normalization,
                weight_denominator=weight_denominator,
                scales=tuple(spatial_cfg["multiscale_scales"]),
            )
        base = (
            point
            + float(spatial_cfg["boundary_weight"]) * boundary
            + float(spatial_cfg["gradient_weight"]) * gradient
            + float(spatial_cfg["multiscale_weight"]) * multiscale
        )
        base_loss = float(base.detach().cpu().item())
        sigma = _resolve_sigma_for_var(name, base_loss, cfg)
        if cfg["weighting"] == "uncertainty":
            weighted = torch.exp(torch.tensor(-sigma, dtype=torch.float32, device=target.device)) * base + float(sigma)
        else:
            fixed_weight = float(fixed_weights_by_var.get(name, 1.0)) if fixed_weights_by_var else 1.0
            weighted = base * float(fixed_weight)
        weighted = weighted * float(target_multipliers.get(str(name), 1.0))
        component_factor = float(target_multipliers.get(str(name), 1.0))
        if cfg["weighting"] == "uncertainty":
            component_factor *= float(np.exp(-sigma))
        elif fixed_weights_by_var:
            component_factor *= float(fixed_weights_by_var.get(name, 1.0))
        per_var[name] = float(weighted.detach().cpu().item())
        spatial_enabled = any(
            float(spatial_cfg[key]) > 0.0
            for key in ("boundary_weight", "gradient_weight", "multiscale_weight")
        )
        if spatial_enabled:
            per_var[f"loss_supervised_point_{name}"] = float(point.detach().cpu().item()) * component_factor
            per_var[f"loss_supervised_spatial_boundary_{name}"] = (
                float(boundary.detach().cpu().item())
                * float(spatial_cfg["boundary_weight"])
                * component_factor
            )
            per_var[f"loss_supervised_spatial_gradient_{name}"] = (
                float(gradient.detach().cpu().item())
                * float(spatial_cfg["gradient_weight"])
                * component_factor
            )
            per_var[f"loss_supervised_spatial_multiscale_{name}"] = (
                float(multiscale.detach().cpu().item())
                * float(spatial_cfg["multiscale_weight"])
                * component_factor
            )
        total = total + weighted
    _append_group_loss_breakdown(per_var, groups=target_groups)
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
    keys = [str(key) for key in pred_fields.keys()]
    cfg = dict(physics_cfg or {})
    role_schema = cfg.get("target_role_schema")
    if role_schema is None:
        role_schema = cfg.get("target_roles")
    resolved = resolve_physics_symbol_keys(
        keys,
        symbols=dict(cfg.get("symbols", {}) or {}),
        target_role_schema=dict(role_schema or {}),
        context="training physics",
    )
    return resolved["density"], resolved["temperature"], resolved["potential"]


def _resolve_numpy_physics_fields(
    pred_fields: dict[str, Any],
    physics_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density_key, temperature_key, potential_key = _resolve_physics_symbol_keys(pred_fields, physics_cfg)
    density = _as_bhw(pred_fields[density_key], key=density_key)
    temperature = _as_bhw(pred_fields[temperature_key], key=temperature_key)
    potential = _as_bhw(pred_fields[potential_key], key=potential_key)
    return density, temperature, potential


def compose_numpy(
    pred_fields: dict[str, Any],
    physics_cfg: dict[str, Any] | None,
    resolved_terms: list[dict[str, Any]] | None = None,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Compose numpy physics terms using the finite-difference implementation."""

    if not physics_cfg or not bool(physics_cfg.get("enabled", False)):
        if pred_fields:
            first_key = str(next(iter(pred_fields.keys())))
            ref = _as_bhw(pred_fields[first_key], key=first_key)
        else:
            raise ValueError("compose_numpy requires at least one prediction field")
        return 0.0, np.zeros_like(ref, dtype=np.float32), _empty_components()

    density, temperature, potential = _resolve_numpy_physics_fields(pred_fields, physics_cfg)
    eff_cfg = dict(physics_cfg)
    if resolved_terms is not None:
        eff_cfg["resolved_terms"] = resolved_terms
    term_map = {term.name: term for term in resolve_numpy_terms(eff_cfg)}
    eff_cfg["poisson_weight"] = (
        float(term_map["poisson"].weight) if bool(term_map["poisson"].enabled) else 0.0
    )
    eff_cfg["boundary_weight"] = (
        float(term_map["boundary"].weight) if bool(term_map["boundary"].enabled) else 0.0
    )
    bo_cfg = dict(eff_cfg.get("boundary_operator", {}))
    bo_cfg["weight"] = (
        float(term_map["boundary_operator"].weight) if bool(term_map["boundary_operator"].enabled) else 0.0
    )
    bo_cfg["enabled"] = bool(bo_cfg.get("enabled", False)) and bool(term_map["boundary_operator"].enabled)
    eff_cfg["boundary_operator"] = bo_cfg

    phys_loss, grad_phi, terms = physics_loss_and_grad(phi=potential, cfg=eff_cfg, density=density, te=temperature)
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
    """Compose torch physics terms using the autograd-compatible implementation."""

    torch = require_torch()
    if pred_fields:
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
    term_map = {term.name: term for term in resolve_torch_terms(eff_cfg)}
    unsupported = [
        name
        for name in ("boundary", "rho")
        if bool(term_map[name].enabled) and float(term_map[name].weight) > 0.0
    ]
    if unsupported:
        raise ValueError(
            "Torch physics loss currently supports only poisson and boundary_operator terms; "
            f"unsupported enabled terms: {unsupported}"
        )
    eff_cfg["poisson_weight"] = (
        float(term_map["poisson"].weight) if bool(term_map["poisson"].enabled) else 0.0
    )
    bo_cfg = dict(eff_cfg.get("boundary_operator", {}))
    bo_cfg["weight"] = (
        float(term_map["boundary_operator"].weight) if bool(term_map["boundary_operator"].enabled) else 0.0
    )
    bo_cfg["enabled"] = bool(bo_cfg.get("enabled", False)) and bool(term_map["boundary_operator"].enabled)
    eff_cfg["boundary_operator"] = bo_cfg

    density_t = torch.as_tensor(pred_fields[density_key], dtype=torch.float32, device=ref.device)
    physics_pred_fields = dict(pred_fields)
    physics_pred_fields["density"] = density_t
    physics_pred_fields["temperature"] = torch.as_tensor(pred_fields[temperature_key], dtype=torch.float32, device=ref.device)
    physics_pred_fields["potential"] = ref
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
