"""Product data/physics loss composition shared by numpy and torch trainers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups
from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys
from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.loss_contract import removed_supervised_keys
from plasma_surrogate.train.loss_protocols import (
    GROUP_WEIGHTING_MODES,
    GROUP_WEIGHTING_NONE,
    GROUP_WEIGHTING_UNIFORM_BY_TARGET,
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
    }


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


def _filter_target_role_schema_for_outputs(
    target_role_schema: dict[str, Any],
    *,
    y_order: list[str],
) -> dict[str, Any]:
    raw_targets = target_role_schema.get("targets", [])
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("train.loss.group_weighting.mode=uniform_by_group requires target_role_schema.targets")
    y_set = set(str(name) for name in y_order)
    filtered_targets: list[dict[str, Any]] = []
    filtered_ids: set[str] = set()
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        target_id = str(raw.get("id", "")).strip()
        if target_id in y_set:
            filtered_targets.append(dict(raw))
            filtered_ids.add(target_id)
    missing = [str(name) for name in y_order if str(name) not in filtered_ids]
    if missing:
        raise ValueError(
            "train.loss.group_weighting.mode=uniform_by_group missing target_role_schema targets: "
            f"{missing}"
        )
    filtered = dict(target_role_schema)
    filtered["targets"] = filtered_targets
    return filtered


def _safe_group_loss_suffix(name: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    return suffix or "group"


def _resolve_loss_target_multipliers(
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None,
) -> tuple[str, dict[str, float], dict[str, TargetGroup]]:
    mode = _resolve_group_weighting_mode(loss_cfg)
    target_names = [str(name) for name in y_order]
    if mode == GROUP_WEIGHTING_NONE:
        return mode, {name: 1.0 for name in target_names}, {}
    if mode == GROUP_WEIGHTING_UNIFORM_BY_TARGET:
        scale = 1.0 / float(max(len(target_names), 1))
        return mode, {name: scale for name in target_names}, {}

    cfg = _dict_or_empty(loss_cfg)
    target_role_schema = _dict_or_empty(cfg.get("target_role_schema"))
    filtered_schema = _filter_target_role_schema_for_outputs(target_role_schema, y_order=target_names)
    groups = resolve_target_groups(
        output_vars=target_names,
        target_role_schema=filtered_schema,
        mode="field_family",
        strict=True,
    )
    if not groups:
        raise ValueError("train.loss.group_weighting.mode=uniform_by_group resolved no target groups")
    multipliers: dict[str, float] = {}
    group_count = float(len(groups))
    for group in groups.values():
        group_size = float(max(len(group.targets), 1))
        scale = 1.0 / float(group_count * group_size)
        for target in group.targets:
            multipliers[str(target)] = scale
    missing = [name for name in target_names if name not in multipliers]
    if missing:
        raise ValueError(f"train.loss.group_weighting did not assign targets: {missing}")
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
    if normalization == "none":
        return float(np.sum(loss_map * sw)), (grad_map * sw).astype(np.float32)
    if normalization == "sample_mean":
        batch = int(loss_map.shape[0])
        numer = np.sum(loss_map * sw, axis=(1, 2))
        if weight_denominator == "count":
            denom = np.maximum(np.sum((sw > 0.0).astype(np.float32), axis=(1, 2)), 1.0)
        else:
            denom = np.maximum(np.sum(sw, axis=(1, 2)), 1.0)
        per_sample = numer / denom
        grad = (grad_map * sw) / denom[:, None, None]
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
    denom = float(max(np.sum(sw), 1.0))
    return float(np.sum(loss_map * sw) / denom), ((grad_map * sw) / denom).astype(np.float32)


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
        err = (pred - target).astype(np.float32)
        if cfg["type"] == "huber":
            loss_map, grad_map = _huber_loss_and_grad(err, delta=float(cfg["delta"]))
        else:
            loss_map = (0.5 * err * err).astype(np.float32)
            grad_map = err

        sw = np.ones_like(loss_map, dtype=np.float32)
        if mask_arr is not None:
            sw = mask_arr
            if sw.shape[0] == 1 and pred.shape[0] > 1:
                sw = np.repeat(sw, pred.shape[0], axis=0)
            if sw.shape != pred.shape:
                raise ValueError(f"mask shape mismatch for {name}: expected {pred.shape}, got {sw.shape}")

        base_loss, grad = _weighted_reduce_numpy(
            loss_map,
            grad_map,
            sw,
            normalization=normalization,
            weight_denominator=sample_mean_weight_denominator,
            group_ids=group_ids,
            group_mode=sample_mean_group_mode,
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
        per_var_loss[name] = weighted
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
    if normalization == "none":
        return torch.sum(base_map * sw)
    if normalization == "sample_mean":
        numer = torch.sum(base_map * sw, dim=(1, 2, 3))
        if weight_denominator == "count":
            denom = torch.clamp(torch.sum((sw > 0.0).to(dtype=base_map.dtype), dim=(1, 2, 3)), min=1.0)
        else:
            denom = torch.clamp(torch.sum(sw, dim=(1, 2, 3)), min=1.0)
        return torch.mean(numer / denom)
    denom = torch.clamp(torch.sum(sw), min=1.0)
    return torch.sum(base_map * sw) / denom


def compose_supervised_torch(
    pred_fields: dict[str, Any],
    target_fields: Any,
    *,
    y_order: list[str],
    loss_cfg: dict[str, Any] | None = None,
    mask: Any | None = None,
    distance_any: Any | None = None,
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
        err = pred - target_i
        if cfg["type"] == "huber":
            base_map = torch.nn.functional.huber_loss(pred, target_i, delta=float(cfg["delta"]), reduction="none")
        else:
            base_map = 0.5 * err * err

        sw = torch.ones_like(base_map, dtype=base_map.dtype)
        if mask_tensor is not None:
            sw = mask_tensor.to(dtype=base_map.dtype)
            if tuple(sw.shape) != tuple(base_map.shape):
                raise ValueError(f"mask shape mismatch for {name}: expected {tuple(base_map.shape)}, got {tuple(sw.shape)}")

        base = _weighted_reduce_torch(base_map, sw, normalization=normalization, weight_denominator=weight_denominator)
        base_loss = float(base.detach().cpu().item())
        sigma = _resolve_sigma_for_var(name, base_loss, cfg)
        if cfg["weighting"] == "uncertainty":
            weighted = torch.exp(torch.tensor(-sigma, dtype=torch.float32, device=target.device)) * base + float(sigma)
        else:
            fixed_weight = float(fixed_weights_by_var.get(name, 1.0)) if fixed_weights_by_var else 1.0
            weighted = base * float(fixed_weight)
        weighted = weighted * float(target_multipliers.get(str(name), 1.0))
        per_var[name] = float(weighted.detach().cpu().item())
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
