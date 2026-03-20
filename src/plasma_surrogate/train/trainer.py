"""Cycle1/M3 trainers for baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.deeponet_contract import allvars_plasma_balance_score, masked_r2_score
from plasma_surrogate.core.density_contract import resolve_density_key, resolve_density_pair
from plasma_surrogate.core.physics_contract import resolve_epoch_scaled_physics
from plasma_surrogate.core.spatial_regions import align_bhw_batch, build_region_masks
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.train.loss_composer import compose_numpy, compose_supervised_numpy
from plasma_surrogate.train.losses import laplacian2d


@dataclass
class TrainOutput:
    history: list[dict[str, float]]
    model: Any


def _mse_grad(x: np.ndarray, pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    err = pred - target
    loss = float(np.mean(err**2))
    grad_y = (2.0 / pred.shape[0]) * err
    grad_w = x.T @ grad_y
    grad_b = np.sum(grad_y, axis=0)
    return grad_w.astype(np.float32), grad_b.astype(np.float32), loss


def _forward_model(model: Any, x: np.ndarray, *, training: bool) -> np.ndarray:
    try:
        return model.forward(x, training=training)
    except TypeError:
        return model.forward(x)


def _backward_model(model: Any, grad_output: np.ndarray, *, lr: float, weight_decay: float = 0.0) -> bool:
    if hasattr(model, "backward"):
        model.backward(grad_output, lr=lr, weight_decay=weight_decay)
        return True
    return False


def _resolve_log_ne_field(pred_fields: np.ndarray, y_vars: list[str]) -> np.ndarray:
    density_ne_key = resolve_density_key(list(y_vars), canonical="ne", prefer_linear=True)
    if density_ne_key is None:
        raise ValueError("physics-aware training requires one of ne/log_ne in model output_keys")
    ne_arr = np.asarray(pred_fields[:, y_vars.index(density_ne_key)], dtype=np.float32)
    if density_ne_key == "log_ne":
        return ne_arr
    return np.log10(np.maximum(ne_arr, np.float32(1.0e-30))).astype(np.float32)


def _clone_model_state_numpy(model: Any) -> dict[str, np.ndarray]:
    if not hasattr(model, "state_dict_numpy"):
        raise TypeError("model does not support state_dict_numpy")
    state = model.state_dict_numpy()
    return {str(k): np.asarray(v, dtype=np.float32).copy() for k, v in dict(state).items()}


def _restore_model_state_numpy(model: Any, state: dict[str, np.ndarray]) -> None:
    if not hasattr(model, "load_state_dict_numpy"):
        raise TypeError("model does not support load_state_dict_numpy")
    model.load_state_dict_numpy({str(k): np.asarray(v, dtype=np.float32).copy() for k, v in state.items()})


def _masked_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return masked_r2_score(y_true, y_pred, mask)


def _unet_allvars_boundary_balance_score(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    y_vars: list[str],
    mask: np.ndarray | None,
    distance_any: np.ndarray | None,
    weights: dict[str, float],
    boundary_band_px: float,
    boundary_bonus_weight: float = 0.25,
) -> tuple[float, dict[str, float]]:
    batch_size = int(pred.shape[0])
    if distance_any is None:
        if mask is None:
            plasma_mask = np.ones((batch_size, pred.shape[2], pred.shape[3]), dtype=bool)
        else:
            m = align_bhw_batch(mask, batch_size=batch_size, key="supervised_mask")
            plasma_mask = m > 0.5
        boundary_mask = np.zeros_like(plasma_mask, dtype=bool)
    else:
        if mask is None:
            mask_for_regions = np.ones((batch_size, pred.shape[2], pred.shape[3]), dtype=np.float32)
        else:
            mask_for_regions = align_bhw_batch(mask, batch_size=batch_size, key="supervised_mask")
        distance_for_regions = align_bhw_batch(distance_any, batch_size=batch_size, key="supervised_distance")
        region_masks = build_region_masks(
            mask_plasma=mask_for_regions,
            distance_any=distance_for_regions,
            distance_signed=None,
            boundary_in_px=float(boundary_band_px),
            mid_plasma_px=10.0,
            deep_plasma_px=10.0,
        )
        plasma_mask = np.asarray(region_masks["all_plasma"], dtype=bool)
        boundary_mask = np.asarray(region_masks["boundary_in"], dtype=bool)
    parts: dict[str, float] = {}
    density_keys = resolve_density_pair(list(y_vars), prefer_linear=True)
    score_names = list(y_vars)
    deep_mask = np.zeros_like(plasma_mask, dtype=bool)
    if distance_any is not None:
        deep_mask = np.asarray(region_masks["plasma_deep"], dtype=bool)
    for name in score_names:
        if name not in y_vars:
            continue
        idx = y_vars.index(name)
        r2_plasma = _masked_r2(target[:, idx], pred[:, idx], plasma_mask)
        r2_deep = _masked_r2(target[:, idx], pred[:, idx], deep_mask)
        parts[f"r2_{name}_plasma"] = float(r2_plasma)
        parts[f"r2_{name}_plasma_deep"] = float(r2_deep)
    base_score, base_parts = allvars_plasma_balance_score(
        pred=pred,
        target=target,
        y_vars=list(y_vars),
        weights=dict(weights),
        plasma_mask=plasma_mask,
    )
    parts.update(base_parts)
    for name in density_keys:
        if name not in y_vars:
            continue
        idx = y_vars.index(name)
        density_pred = np.asarray(pred[:, idx], dtype=np.float32)
        plasma_vals = density_pred[plasma_mask]
        if plasma_vals.size == 0:
            parts[f"neg_ratio_{name}_plasma"] = float("nan")
        else:
            parts[f"neg_ratio_{name}_plasma"] = float(np.mean((plasma_vals < 0.0).astype(np.float32)))
    if not np.isfinite(base_score):
        return float("nan"), parts
    boundary_total = 0.0
    boundary_weight = 0.0
    for name in score_names:
        if name not in y_vars:
            continue
        idx = y_vars.index(name)
        r2_boundary = _masked_r2(target[:, idx], pred[:, idx], boundary_mask)
        parts[f"r2_{name}_boundary"] = float(r2_boundary)
        w = float(weights.get(name, 0.0))
        if w > 0.0 and np.isfinite(r2_boundary):
            boundary_total += w * float(r2_boundary)
            boundary_weight += w
    if boundary_weight > 0.0 and float(boundary_bonus_weight) > 0.0:
        boundary_score = boundary_total / boundary_weight
        return float(base_score + float(boundary_bonus_weight) * float(boundary_score)), parts
    return float(base_score), parts


def _resolve_unet_optimizer_cfg(unet_optimizer_cfg: dict[str, Any] | None, *, default_lr: float) -> dict[str, Any]:
    cfg = dict(unet_optimizer_cfg or {})
    opt_type = str(cfg.get("type", "adamw")).strip().lower()
    if opt_type != "adamw":
        raise ValueError("train.unet.optimizer.type must be: adamw")
    lr = float(cfg.get("lr", default_lr))
    if lr <= 0.0:
        raise ValueError("train.unet.optimizer.lr must be > 0")
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if weight_decay < 0.0:
        raise ValueError("train.unet.optimizer.weight_decay must be >= 0")
    betas_raw = cfg.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ValueError("train.unet.optimizer.betas must be [beta1, beta2]")
    beta1 = float(betas_raw[0])
    beta2 = float(betas_raw[1])
    eps = float(cfg.get("eps", 1.0e-8))
    if eps <= 0.0:
        raise ValueError("train.unet.optimizer.eps must be > 0")
    schedule = str(cfg.get("schedule", "none")).strip().lower()
    if schedule not in {"none", "cosine"}:
        raise ValueError("train.unet.optimizer.schedule must be one of: none, cosine")
    warmup_epochs = int(max(int(cfg.get("warmup_epochs", 0)), 0))
    return {
        "type": opt_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
        "schedule": schedule,
        "warmup_epochs": warmup_epochs,
    }


def _set_unet_optimizer_epoch_lr(optimizer: Any, *, base_lr: float, epoch: int, epochs: int, schedule: str, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        scale = float(epoch + 1) / float(max(warmup_epochs, 1))
    elif schedule == "cosine":
        tail = max(int(epochs) - int(warmup_epochs), 1)
        t = float(max(epoch - warmup_epochs, 0)) / float(max(tail - 1, 1))
        scale = 0.5 * (1.0 + float(np.cos(np.pi * t)))
    else:
        scale = 1.0
    lr_now = float(base_lr) * float(scale)
    for group in optimizer.param_groups:
        group["lr"] = lr_now
    return lr_now


def _resolve_global_grad_scale(
    grad_scale_cfg: dict[str, Any] | None,
    *,
    mask: np.ndarray | None,
    grid_shape: tuple[int, int],
) -> float:
    cfg = dict(grad_scale_cfg or {})
    raw_mode = cfg.get("mode", "off")
    if isinstance(raw_mode, bool):
        mode = "auto" if raw_mode else "off"
    else:
        mode = str(raw_mode).strip().lower()
    if mode == "on":
        mode = "auto"
    if mode == "off":
        return 1.0
    if mode == "fixed":
        return float(max(float(cfg.get("fixed_value", 1.0)), 0.0))
    if mode == "auto":
        auto_ref = str(cfg.get("auto_ref", "active_pixels")).strip().lower()
        auto_power = float(cfg.get("auto_power", 0.5))
        if auto_ref not in {"active_pixels", "total_pixels"}:
            raise ValueError("global_mlp.grad_scale.auto_ref must be one of: active_pixels, total_pixels")
        if auto_ref == "active_pixels" and mask is not None:
            m = np.asarray(mask, dtype=np.float32)
            if m.ndim == 3 and int(m.shape[0]) == 1:
                m = m[0]
            if m.ndim == 2:
                ref = float(np.sum(m > 0.5))
            else:
                ref = float(grid_shape[0] * grid_shape[1])
        else:
            ref = float(grid_shape[0] * grid_shape[1])
        return float(max(ref, 1.0) ** auto_power)
    raise ValueError("global_mlp.grad_scale.mode must be one of: off, auto, fixed")


def _clip_grad_global_norm(grad: np.ndarray, clip_norm: float) -> np.ndarray:
    c = float(clip_norm)
    if c <= 0.0:
        return grad
    nrm = float(np.linalg.norm(grad.reshape(-1)))
    if nrm <= c or nrm <= 0.0:
        return grad
    return grad * (c / nrm)


def _clip_grad_per_sample_norm(grad: np.ndarray, clip_norm: float) -> np.ndarray:
    c = float(clip_norm)
    if c <= 0.0:
        return grad
    g = np.asarray(grad, dtype=np.float32)
    flat = g.reshape(g.shape[0], -1)
    nrm = np.linalg.norm(flat, axis=1)
    scale = np.ones_like(nrm, dtype=np.float32)
    active = nrm > c
    scale[active] = c / np.maximum(nrm[active], 1e-12)
    scale_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
    return g * scale.reshape(scale_shape)


def _resolve_global_clip(
    *,
    grad_clip_cfg: dict[str, Any] | None,
    grad_clip_norm: float,
    mask: np.ndarray | None,
    grid_shape: tuple[int, int],
    out_channels: int,
) -> tuple[str, float]:
    cfg = dict(grad_clip_cfg or {})
    legacy_default_mode = "global" if float(grad_clip_norm) > 0.0 else "off"
    raw_mode = cfg.get("mode", legacy_default_mode)
    if isinstance(raw_mode, bool):
        mode = "global" if raw_mode else "off"
    else:
        mode = str(raw_mode).strip().lower()
    if mode == "on":
        mode = "global"
    if mode not in {"off", "global", "per_sample"}:
        raise ValueError("global_mlp.grad_clip.mode must be one of: off, global, per_sample")
    base_norm = float(cfg.get("norm", grad_clip_norm))
    if mode == "off" or base_norm <= 0.0:
        return "off", 0.0
    adaptive = bool(cfg.get("adaptive_by_dim", False))
    if not adaptive:
        return mode, float(base_norm)

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3 and int(m.shape[0]) == 1:
            m = m[0]
        active_pixels = float(np.sum(m > 0.5)) if m.ndim == 2 else float(grid_shape[0] * grid_shape[1])
    else:
        active_pixels = float(grid_shape[0] * grid_shape[1])
    factor = np.sqrt(max(active_pixels * float(max(out_channels, 1)), 1.0))
    return mode, float(base_norm) * float(factor)


def _resolve_clip_contract(
    optimizer_contract: dict[str, Any] | None,
    *,
    default_mode: str = "off",
    default_norm: float = 0.0,
    adaptive_ref_count: float = 1.0,
    out_channels: int = 1,
    mode_error_prefix: str = "train.optimizer_contract.grad_clip",
) -> tuple[str, float]:
    raw = dict(optimizer_contract or {})
    cfg = dict(raw.get("grad_clip", raw))
    raw_mode = cfg.get("mode", default_mode)
    if isinstance(raw_mode, bool):
        mode = "global" if raw_mode else "off"
    else:
        mode = str(raw_mode).strip().lower()
    if mode == "on":
        mode = "global"
    if mode not in {"off", "global", "per_sample"}:
        raise ValueError(f"{mode_error_prefix}.mode must be one of: off, global, per_sample")
    norm = float(cfg.get("norm", default_norm))
    if mode == "off" or norm <= 0.0:
        return "off", 0.0
    if bool(cfg.get("adaptive_by_dim", False)):
        factor = np.sqrt(max(float(adaptive_ref_count) * float(max(out_channels, 1)), 1.0))
        norm = norm * float(factor)
    return mode, float(norm)


def _is_effective_clip(
    *,
    clip_ratio: float,
    threshold: float,
    direction: str,
) -> bool:
    mode = str(direction).strip().lower()
    if mode == "lt":
        return float(clip_ratio) < float(threshold)
    if mode == "gt":
        return float(clip_ratio) > float(threshold)
    raise ValueError("train.optimizer_contract.diagnostics.alert.clip_effective_direction must be one of: lt, gt")


def train_one_epoch_global(
    model: GlobalMLP,
    cond_matrix: np.ndarray,
    y_flat: np.ndarray,
    lr: float = 1e-3,
) -> float:
    pred = _forward_model(model, cond_matrix, training=True)
    err = pred - y_flat
    loss = float(np.mean(err**2))
    grad_y = (2.0 / float(max(pred.size, 1))) * err
    if not _backward_model(model, grad_y.astype(np.float32), lr=lr):
        grad_w, grad_b, _ = _mse_grad(cond_matrix, pred, y_flat)
        model.W -= lr * grad_w
        model.b -= lr * grad_b
    return loss


def train_one_epoch_global_physics(
    model: GlobalMLP,
    cond_matrix: np.ndarray,
    y_field: np.ndarray,
    lr: float = 1e-3,
    physics_cfg: dict[str, Any] | None = None,
    resolved_terms: list[dict[str, Any]] | None = None,
    loss_cfg: dict[str, Any] | None = None,
    supervised_mask: np.ndarray | None = None,
    supervised_distance: np.ndarray | None = None,
    batch_size_cases: int = 0,
    shuffle_cases: bool = True,
    rng: np.random.Generator | None = None,
    grad_scale_cfg: dict[str, Any] | None = None,
    layer_lr_multiplier: dict[str, float] | None = None,
    grad_clip_norm: float = 0.0,
    grad_clip_cfg: dict[str, Any] | None = None,
    epoch_idx: int | None = None,
) -> dict[str, float]:
    n_cases = int(cond_matrix.shape[0])
    if n_cases == 0:
        raise ValueError("global training batch is empty")
    bsz = n_cases if int(batch_size_cases) <= 0 else min(int(batch_size_cases), n_cases)
    order = np.arange(n_cases, dtype=np.int64)
    if shuffle_cases and n_cases > 1:
        (rng or np.random.default_rng(0)).shuffle(order)

    h, w = model.grid_shape
    y_vars = list(getattr(model, "output_keys", ["ne", "Te", "phi"]))
    grad_diag_keys = ["phi", "Te"]
    density_ne_key = resolve_density_key(y_vars, canonical="ne", prefer_linear=True)
    density_ni_key = resolve_density_key(y_vars, canonical="ni", prefer_linear=True)
    if density_ne_key is not None:
        grad_diag_keys.append(density_ne_key)
    if density_ni_key is not None:
        grad_diag_keys.append(density_ni_key)
    accum = {
        "total": 0.0,
        "data": 0.0,
        "physics": 0.0,
        "poisson": 0.0,
        "boundary": 0.0,
        "boundary_operator": 0.0,
        "rho": 0.0,
        "grad_l2_total": 0.0,
        "grad_l2_phi": 0.0,
        "grad_l2_Te": 0.0,
        "grad_l2_ne": 0.0,
        "grad_l2_ni": 0.0,
        "grad_l2_log_ne": 0.0,
        "grad_l2_log_ni": 0.0,
        "active_weight_ratio": 0.0,
        "step_rel_hidden_mean": 0.0,
        "step_rel_output": 0.0,
        "step_rel_output_to_hidden": 0.0,
        "grad_scale_applied": 0.0,
        "grad_norm_pre_scale": 0.0,
        "grad_norm_post_scale": 0.0,
        "grad_norm_post_clip": 0.0,
        "clip_ratio": 0.0,
    }
    total_seen = 0
    grad_scale = _resolve_global_grad_scale(
        grad_scale_cfg,
        mask=supervised_mask,
        grid_shape=(h, w),
    )
    clip_mode, clip_norm = _resolve_global_clip(
        grad_clip_cfg=grad_clip_cfg,
        grad_clip_norm=grad_clip_norm,
        mask=supervised_mask,
        grid_shape=(h, w),
        out_channels=int(model.out_channels),
    )

    for start in range(0, n_cases, bsz):
        sl = order[start : start + bsz]
        x_batch = cond_matrix[sl]
        y_batch = y_field[sl]
        pred = _forward_model(model, x_batch, training=True)
        pred_fields = pred.reshape(pred.shape[0], model.out_channels, h, w)
        pred_dict = {name: pred_fields[:, i] for i, name in enumerate(y_vars)}
        tgt_dict = {name: np.asarray(y_batch[:, i], dtype=np.float32) for i, name in enumerate(y_vars)}
        data_loss, grad_by_var, _ = compose_supervised_numpy(
            pred_dict,
            tgt_dict,
            y_order=y_vars,
            loss_cfg=loss_cfg,
            mask=supervised_mask,
            distance_any=supervised_distance,
            epoch_idx=epoch_idx,
        )
        grad_fields = np.stack([grad_by_var[name] for name in y_vars], axis=1).astype(np.float32)
        grad_y = grad_fields.reshape(pred.shape)
        total_loss = float(data_loss)
        phys_loss = 0.0
        terms = {"poisson": 0.0, "boundary": 0.0, "boundary_operator": 0.0, "rho": 0.0}

        if physics_cfg and bool(physics_cfg.get("enabled", False)):
            if "phi" not in y_vars or "Te" not in y_vars:
                raise ValueError("physics-aware training requires ne/log_ne, Te, phi in model output_keys")
            phi_idx = y_vars.index("phi")
            phys_loss, grad_phi, terms = compose_numpy(
                {
                    "log_ne": _resolve_log_ne_field(pred_fields, y_vars),
                    "Te": pred_fields[:, y_vars.index("Te")],
                    "phi": pred_fields[:, phi_idx],
                },
                physics_cfg=physics_cfg,
                resolved_terms=resolved_terms,
            )
            if phys_loss > 0.0:
                grad_phys_fields = np.zeros_like(pred_fields, dtype=np.float32)
                grad_phys_fields[:, phi_idx] = grad_phi
                grad_y = grad_y + grad_phys_fields.reshape(pred.shape)
            total_loss = float(data_loss + phys_loss)

        grad_norm_pre = float(np.mean(np.linalg.norm(grad_y.reshape(grad_y.shape[0], -1), axis=1)))
        grad_y = grad_y * float(grad_scale)
        grad_norm_post_scale = float(np.mean(np.linalg.norm(grad_y.reshape(grad_y.shape[0], -1), axis=1)))
        if clip_mode == "global":
            grad_y = _clip_grad_global_norm(grad_y, float(clip_norm))
        elif clip_mode == "per_sample":
            grad_y = _clip_grad_per_sample_norm(grad_y, float(clip_norm))
        grad_norm_post_clip = float(np.mean(np.linalg.norm(grad_y.reshape(grad_y.shape[0], -1), axis=1)))
        clip_ratio = grad_norm_post_clip / max(grad_norm_post_scale, 1e-12)

        back_diag = {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}
        if hasattr(model, "backward"):
            ret = model.backward(
                grad_y.astype(np.float32),
                lr=lr,
                layer_lr_multipliers=layer_lr_multiplier,
            )
            if isinstance(ret, dict):
                back_diag["step_rel_hidden_mean"] = float(ret.get("step_rel_hidden_mean", 0.0))
                back_diag["step_rel_output"] = float(ret.get("step_rel_output", 0.0))
        else:
            grad_w = x_batch.T @ grad_y
            grad_b = np.sum(grad_y, axis=0)
            model.W -= lr * grad_w
            model.b -= lr * grad_b

        seen = int(sl.size)
        total_seen += seen
        accum["total"] += float(total_loss) * seen
        accum["data"] += float(data_loss) * seen
        accum["physics"] += float(phys_loss) * seen
        accum["poisson"] += float(terms.get("poisson", 0.0)) * seen
        accum["boundary"] += float(terms.get("boundary", 0.0)) * seen
        accum["boundary_operator"] += float(terms.get("boundary_operator", 0.0)) * seen
        accum["rho"] += float(terms.get("rho", 0.0)) * seen
        accum["step_rel_hidden_mean"] += float(back_diag.get("step_rel_hidden_mean", 0.0)) * seen
        accum["step_rel_output"] += float(back_diag.get("step_rel_output", 0.0)) * seen
        accum["step_rel_output_to_hidden"] += (
            float(back_diag.get("step_rel_output", 0.0))
            / max(float(back_diag.get("step_rel_hidden_mean", 0.0)), 1e-12)
        ) * seen
        accum["grad_scale_applied"] += float(grad_scale) * seen
        accum["grad_norm_pre_scale"] += float(grad_norm_pre) * seen
        accum["grad_norm_post_scale"] += float(grad_norm_post_scale) * seen
        accum["grad_norm_post_clip"] += float(grad_norm_post_clip) * seen
        accum["clip_ratio"] += float(clip_ratio) * seen

        grad_flat = grad_y.reshape(seen, -1)
        accum["grad_l2_total"] += float(np.mean(np.linalg.norm(grad_flat, axis=1))) * seen
        for key in grad_diag_keys:
            if key in grad_by_var:
                gflat = grad_by_var[key].reshape(seen, -1)
                target_metric_key = f"grad_l2_{key}"
                if key == density_ne_key:
                    target_metric_key = "grad_l2_log_ne"
                elif key == density_ni_key:
                    target_metric_key = "grad_l2_log_ni"
                accum[target_metric_key] += float(np.mean(np.linalg.norm(gflat, axis=1))) * seen
        accum["active_weight_ratio"] += float(np.mean(np.abs(grad_y) > 0.0)) * seen

    denom = float(max(total_seen, 1))
    return {k: float(v / denom) for k, v in accum.items()}


def train_one_epoch_unet(
    model: UNetBaseline,
    cond_matrix: np.ndarray,
    y_field: np.ndarray,
    lr: float = 1e-3,
    physics_cfg: dict[str, Any] | None = None,
    resolved_terms: list[dict[str, Any]] | None = None,
    loss_cfg: dict[str, Any] | None = None,
    supervised_mask: np.ndarray | None = None,
    supervised_distance: np.ndarray | None = None,
    supervised_distance_signed: np.ndarray | None = None,
    supervised_bc_dir_mask: np.ndarray | None = None,
    supervised_wafer_mask: np.ndarray | None = None,
    optimizer_contract: dict[str, Any] | None = None,
    torch_optimizer: Any | None = None,
    batch_size_cases: int = 0,
    shuffle_cases: bool = True,
    rng: np.random.Generator | None = None,
    epoch_idx: int | None = None,
) -> dict[str, float]:
    n_cases = int(cond_matrix.shape[0])
    if n_cases == 0:
        raise ValueError("unet-like training batch is empty")
    bsz = n_cases if int(batch_size_cases) <= 0 else min(int(batch_size_cases), n_cases)
    order = np.arange(n_cases, dtype=np.int64)
    if shuffle_cases and n_cases > 1:
        (rng or np.random.default_rng(0)).shuffle(order)

    y_vars = list(getattr(model, "output_keys", ["ne", "Te", "phi"]))
    accum = {
        "total": 0.0,
        "data": 0.0,
        "physics": 0.0,
        "rho": 0.0,
        "poisson": 0.0,
        "boundary": 0.0,
        "boundary_operator": 0.0,
        "grad_l2_total": 0.0,
        "grad_l2_phi": 0.0,
        "grad_l2_Te": 0.0,
        "grad_l2_ne": 0.0,
        "grad_l2_ni": 0.0,
        "active_weight_ratio": 0.0,
        "step_rel_hidden_mean": 0.0,
        "step_rel_output": 0.0,
        "step_rel_output_to_hidden": 0.0,
        "grad_norm_post_clip": 0.0,
        "clip_ratio": 0.0,
    }
    total_seen = 0

    for start in range(0, n_cases, bsz):
        sl = order[start : start + bsz]
        cond_batch = cond_matrix[sl]
        y_batch = y_field[sl]
        raw_pred = model.forward_raw(cond_batch, training=True)
        pred = raw_pred[:, : model.out_channels]
        pred_dict = {name: pred[:, i] for i, name in enumerate(y_vars)}
        tgt_dict = {name: np.asarray(y_batch[:, i], dtype=np.float32) for i, name in enumerate(y_vars)}
        data_loss, grad_by_var, _ = compose_supervised_numpy(
            pred_dict,
            tgt_dict,
            y_order=y_vars,
            loss_cfg=loss_cfg,
            mask=supervised_mask,
            distance_any=supervised_distance,
            distance_signed=supervised_distance_signed,
            bc_dir_mask=supervised_bc_dir_mask,
            wafer_mask=supervised_wafer_mask,
            epoch_idx=epoch_idx,
        )
        grad_y = np.stack([grad_by_var[name] for name in y_vars], axis=1).astype(np.float32)
        raw_grad = np.zeros_like(raw_pred, dtype=np.float32)
        raw_grad[:, : model.out_channels] = grad_y
        total_loss = float(data_loss)
        phys_loss = 0.0
        terms = {"poisson": 0.0, "boundary": 0.0, "boundary_operator": 0.0, "rho": 0.0}
        rho_loss = 0.0

        if physics_cfg and bool(physics_cfg.get("enabled", False)):
            if "phi" not in y_vars or "Te" not in y_vars:
                raise ValueError("physics-aware training requires ne/log_ne, Te, phi in model output_keys")
            phi_idx = y_vars.index("phi")
            phys_loss, grad_phi, terms = compose_numpy(
                {
                    "log_ne": _resolve_log_ne_field(pred, y_vars),
                    "Te": pred[:, y_vars.index("Te")],
                    "phi": pred[:, phi_idx],
                },
                physics_cfg=physics_cfg,
                resolved_terms=resolved_terms,
            )
            raw_grad[:, phi_idx] += grad_phi
            total_loss = float(data_loss + phys_loss)

        if getattr(model, "with_rho_eff_head", False) and raw_pred.shape[1] > model.out_channels:
            phi_idx = y_vars.index("phi")
            rho_pred = raw_pred[:, model.out_channels : model.out_channels + 1]
            rho_target = -laplacian2d(y_batch[:, phi_idx]).astype(np.float32)[:, None]
            rho_err = rho_pred - rho_target
            rho_weight = float(physics_cfg.get("lambda_rho", 0.05)) if physics_cfg else 0.05
            rho_loss = float(np.mean(rho_err**2))
            total_loss += rho_weight * rho_loss
            raw_grad[:, model.out_channels : model.out_channels + 1] += (
                rho_weight * (2.0 / float(rho_pred.size)) * rho_err
            )

        if supervised_mask is not None:
            m = np.asarray(supervised_mask, dtype=np.float32)
            if m.ndim == 3 and int(m.shape[0]) == 1:
                m = m[0]
            active_pixels = float(np.sum(m > 0.5)) if m.ndim == 2 else float(raw_pred.shape[-2] * raw_pred.shape[-1])
        else:
            active_pixels = float(raw_pred.shape[-2] * raw_pred.shape[-1])
        clip_mode, clip_norm = _resolve_clip_contract(
            optimizer_contract,
            default_mode="off",
            default_norm=0.0,
            adaptive_ref_count=active_pixels,
            out_channels=int(raw_pred.shape[1]),
        )
        grad_norm_pre = float(np.mean(np.linalg.norm(raw_grad.reshape(raw_grad.shape[0], -1), axis=1)))
        if clip_mode == "global":
            raw_grad = _clip_grad_global_norm(raw_grad, clip_norm)
        elif clip_mode == "per_sample":
            raw_grad = _clip_grad_per_sample_norm(raw_grad, clip_norm)
        grad_norm_post_clip = float(np.mean(np.linalg.norm(raw_grad.reshape(raw_grad.shape[0], -1), axis=1)))
        clip_ratio = grad_norm_post_clip / max(grad_norm_pre, 1e-12)

        back_diag = {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}
        if hasattr(model, "backward_raw"):
            use_torch_optimizer = bool(getattr(model, "backend", "numpy") == "torch" and torch_optimizer is not None)
            if use_torch_optimizer:
                torch = model.torch
                w_hidden, w_out = model._torch_step_reference()
                with torch.no_grad():
                    w_hidden_prev = w_hidden.detach().clone() if w_hidden is not None else None
                    w_out_prev = w_out.detach().clone() if w_out is not None else None
                torch_optimizer.zero_grad(set_to_none=True)
                _ = model.backward_raw(raw_grad.astype(np.float32), lr=lr, apply_step=False)
                torch_optimizer.step()
                with torch.no_grad():
                    w_hidden_cur, w_out_cur = model._torch_step_reference()
                    if w_hidden_prev is not None and w_hidden_cur is not None:
                        dh = w_hidden_cur.detach() - w_hidden_prev
                        back_diag["step_rel_hidden_mean"] = float(
                            torch.linalg.norm(dh) / max(float(torch.linalg.norm(w_hidden_prev)), 1e-12)
                        )
                    if w_out_prev is not None and w_out_cur is not None:
                        do = w_out_cur.detach() - w_out_prev
                        back_diag["step_rel_output"] = float(
                            torch.linalg.norm(do) / max(float(torch.linalg.norm(w_out_prev)), 1e-12)
                        )
            else:
                ret = model.backward_raw(raw_grad.astype(np.float32), lr=lr)
                if isinstance(ret, dict):
                    back_diag["step_rel_hidden_mean"] = float(ret.get("step_rel_hidden_mean", 0.0))
                    back_diag["step_rel_output"] = float(ret.get("step_rel_output", 0.0))
        else:
            xmat = model.feature_matrix(cond_batch)
            grad_y_mat = np.moveaxis(raw_grad, 1, -1).reshape(-1, raw_pred.shape[1])
            grad_w = xmat.T @ grad_y_mat
            grad_b = np.sum(grad_y_mat, axis=0)
            w_prev = np.asarray(model.W, dtype=np.float32).copy()
            model.W -= lr * grad_w.astype(np.float32)
            model.b -= lr * grad_b.astype(np.float32)
            back_diag["step_rel_output"] = float(np.linalg.norm(lr * grad_w) / max(np.linalg.norm(w_prev), 1e-12))

        seen = int(sl.size)
        total_seen += seen
        accum["total"] += float(total_loss) * seen
        accum["data"] += float(data_loss) * seen
        accum["physics"] += float(phys_loss) * seen
        accum["rho"] += float(rho_loss) * seen
        accum["poisson"] += float(terms.get("poisson", 0.0)) * seen
        accum["boundary"] += float(terms.get("boundary", 0.0)) * seen
        accum["boundary_operator"] += float(terms.get("boundary_operator", 0.0)) * seen
        accum["grad_l2_total"] += float(np.mean(np.linalg.norm(raw_grad.reshape(seen, -1), axis=1))) * seen
        if "phi" in y_vars:
            accum["grad_l2_phi"] += float(np.mean(np.abs(raw_grad[:, y_vars.index("phi")]))) * seen
        if "Te" in y_vars:
            accum["grad_l2_Te"] += float(np.mean(np.abs(raw_grad[:, y_vars.index("Te")]))) * seen
        density_ne_key = resolve_density_key(y_vars, canonical="ne", prefer_linear=True)
        density_ni_key = resolve_density_key(y_vars, canonical="ni", prefer_linear=True)
        if density_ne_key is not None and density_ne_key in y_vars:
            ne_grad = float(np.mean(np.abs(raw_grad[:, y_vars.index(density_ne_key)]))) * seen
            accum["grad_l2_ne"] += ne_grad
        if density_ni_key is not None and density_ni_key in y_vars:
            ni_grad = float(np.mean(np.abs(raw_grad[:, y_vars.index(density_ni_key)]))) * seen
            accum["grad_l2_ni"] += ni_grad
        accum["active_weight_ratio"] += float(np.mean(np.abs(raw_grad) > 0.0)) * seen
        accum["step_rel_hidden_mean"] += float(back_diag.get("step_rel_hidden_mean", 0.0)) * seen
        accum["step_rel_output"] += float(back_diag.get("step_rel_output", 0.0)) * seen
        accum["step_rel_output_to_hidden"] += (
            float(back_diag.get("step_rel_output", 0.0))
            / max(float(back_diag.get("step_rel_hidden_mean", 0.0)), 1e-12)
        ) * seen
        accum["grad_norm_post_clip"] += float(grad_norm_post_clip) * seen
        accum["clip_ratio"] += float(clip_ratio) * seen

    denom = float(max(total_seen, 1))
    return {k: float(v / denom) for k, v in accum.items()}


class Trainer:
    """Minimal trainer for cycle1/1.1/3 integration tests and benchmark."""

    def __init__(self, output_dir: str | Path):
        self.store = ArtifactStore(output_dir)

    def _save_physics_terms(self, history: list[dict[str, float]]) -> None:
        if not history:
            return
        header = [
            "epoch",
            "train_phys_loss",
            "train_poisson_loss",
            "train_boundary_loss",
            "train_boundary_operator_loss",
            "train_rho_loss",
        ]
        rows = [
            [
                h.get("epoch", 0.0),
                h.get("train_phys_loss", 0.0),
                h.get("train_poisson_loss", 0.0),
                h.get("train_boundary_loss", 0.0),
                h.get("train_boundary_operator_loss", 0.0),
                h.get("train_rho_loss", 0.0),
            ]
            for h in history
        ]
        self.store.save_csv("scalars/physics_terms.csv", header, rows)

    def _save_optimization_diagnostics(self, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        base_header = [
            "epoch",
            "grad_l2_total",
            "active_weight_ratio",
            "step_rel_hidden_mean",
            "step_rel_output",
            "step_rel_output_to_hidden",
            "grad_scale_applied",
            "grad_norm_pre_scale",
            "grad_norm_post_scale",
            "grad_norm_post_clip",
            "clip_ratio",
            "effective_clip_flag",
            "stagnation_flag",
            "head_refresh_applied",
            "head_design_cond",
            "chamber_band_effective_ratio",
            "outside_boundary_sample_ratio",
            "boundary_inside_sample_ratio",
            "boundary_weight_effective_ratio",
            "distance_transform_mode",
            "loss_boundary_in",
            "loss_deep_plasma",
            "region_balance_applied",
            "field_reg_loss_inside",
            "field_reg_loss_boundary",
            "field_reg_points_inside",
            "field_reg_points_boundary",
            "selection_valid_flag",
            "selected_epoch_score",
            "selected_epoch_flag",
        ]
        dynamic_grad = sorted(
            {
                str(k)
                for row in rows
                for k in row.keys()
                if str(k).startswith("grad_l2_") and str(k) != "grad_l2_total"
            }
        )
        dynamic_selection = sorted(
            {
                str(k)
                for row in rows
                for k in row.keys()
                if str(k).startswith("selection_score_")
            }
        )
        header = (
            ["epoch", "grad_l2_total"]
            + dynamic_grad
            + [k for k in base_header if k not in {"epoch", "grad_l2_total"}]
            + [k for k in dynamic_selection if k not in set(base_header)]
        )
        data = [[r.get(k, 0.0) for k in header] for r in rows]
        self.store.save_csv("scalars/optimization_diagnostics.csv", header, data)

    def _save_resolved_physics(self, physics_cfg: dict[str, Any] | None) -> None:
        cfg = dict(physics_cfg or {})
        self.store.save_json(
            "resolved_physics.json",
            {
                "enabled": bool(cfg.get("enabled", False)),
                "resolved_terms": list(cfg.get("resolved_terms", [])),
                "lambda_poisson": float(cfg.get("lambda_poisson", 0.0)),
                "lambda_bc": float(cfg.get("lambda_bc", 0.0)),
                "boundary_operator": {
                    "enabled": bool(cfg.get("boundary_operator", {}).get("enabled", False)),
                    "lambda": float(cfg.get("boundary_operator", {}).get("lambda", 0.0)),
                    "mode": str(cfg.get("boundary_operator", {}).get("mode", "proxy")),
                    "primary_qoi_key": str(cfg.get("boundary_operator", {}).get("primary_qoi_key", "Gamma_i")),
                },
            },
        )

    def run_global(
        self,
        model: GlobalMLP,
        cond_train: np.ndarray,
        y_train: np.ndarray,
        cond_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        physics_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        curriculum_cfg: dict[str, Any] | None = None,
        supervised_mask: np.ndarray | None = None,
        supervised_distance: np.ndarray | None = None,
        batch_size_cases: int = 0,
        shuffle_cases: bool = True,
        seed: int = 0,
        grad_scale_cfg: dict[str, Any] | None = None,
        layer_lr_multiplier: dict[str, float] | None = None,
        grad_clip_norm: float = 0.0,
        grad_clip_cfg: dict[str, Any] | None = None,
        output_head_refresh_cfg: dict[str, Any] | None = None,
    ) -> TrainOutput:
        self._save_resolved_physics(physics_cfg)
        y_train_field = y_train.reshape(y_train.shape[0], model.out_channels, *model.grid_shape).astype(np.float32)
        y_val_field = y_val.reshape(y_val.shape[0], model.out_channels, *model.grid_shape).astype(np.float32)
        history: list[dict[str, float]] = []
        diagnostics_rows: list[dict[str, float]] = []
        rng = np.random.default_rng(int(seed))
        refresh_cfg = dict(output_head_refresh_cfg or {})
        refresh_enabled = bool(refresh_cfg.get("enabled", False))
        refresh_every = int(max(int(refresh_cfg.get("every_n_epochs", 5)), 1))
        refresh_ridge = float(max(float(refresh_cfg.get("ridge", 1e-4)), 0.0))
        for epoch in range(epochs):
            epoch_physics_cfg = resolve_epoch_scaled_physics(
                physics_cfg,
                epoch=epoch,
                curriculum_cfg=curriculum_cfg,
            )
            resolved_terms = list(epoch_physics_cfg.get("resolved_terms", []))
            stats = train_one_epoch_global_physics(
                model,
                cond_train,
                y_train_field,
                lr=lr,
                physics_cfg=epoch_physics_cfg,
                resolved_terms=resolved_terms,
                loss_cfg=loss_cfg,
                supervised_mask=supervised_mask,
                supervised_distance=supervised_distance,
                batch_size_cases=batch_size_cases,
                shuffle_cases=shuffle_cases,
                rng=rng,
                grad_scale_cfg=grad_scale_cfg,
                layer_lr_multiplier=layer_lr_multiplier,
                grad_clip_norm=grad_clip_norm,
                grad_clip_cfg=grad_clip_cfg,
                epoch_idx=epoch,
            )
            head_refresh_applied = 0.0
            head_design_cond = 0.0
            if refresh_enabled and ((int(epoch) + 1) % refresh_every == 0):
                hidden = model.extract_hidden(cond_train)
                head_design_cond = float(model.fit_output_layer_ridge(hidden, y_train, ridge=refresh_ridge))
                head_refresh_applied = 1.0
            val_pred = _forward_model(model, cond_val, training=False)
            val_fields = val_pred.reshape(val_pred.shape[0], model.out_channels, *model.grid_shape)
            y_vars = list(getattr(model, "output_keys", ["ne", "Te", "phi"]))
            val_data_loss, _, _ = compose_supervised_numpy(
                {name: val_fields[:, i] for i, name in enumerate(y_vars)},
                {name: y_val_field[:, i] for i, name in enumerate(y_vars)},
                y_order=y_vars,
                loss_cfg=loss_cfg,
                mask=supervised_mask,
                distance_any=supervised_distance,
                epoch_idx=epoch,
            )
            val_phys_loss = 0.0
            if epoch_physics_cfg and bool(epoch_physics_cfg.get("enabled", False)):
                if "phi" not in y_vars or "Te" not in y_vars:
                    raise ValueError("physics-aware training requires ne/log_ne, Te, phi in model output_keys")
                val_phys_loss = float(
                    compose_numpy(
                        {
                            "log_ne": _resolve_log_ne_field(val_fields, y_vars),
                            "Te": val_fields[:, y_vars.index("Te")],
                            "phi": val_fields[:, y_vars.index("phi")],
                        },
                        physics_cfg=epoch_physics_cfg,
                        resolved_terms=resolved_terms,
                    )[0]
                )
            val_loss = val_data_loss + val_phys_loss
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": stats["total"],
                    "train_data_loss": stats["data"],
                    "train_phys_loss": stats["physics"],
                    "train_poisson_loss": stats["poisson"],
                    "train_boundary_loss": stats["boundary"],
                    "train_boundary_operator_loss": stats["boundary_operator"],
                    "train_physics_scale": float(epoch_physics_cfg.get("physics_ramp_scale", 1.0)),
                    "val_loss": float(val_loss),
                    "val_data_loss": float(val_data_loss),
                    "val_phys_loss": float(val_phys_loss),
                }
            )
            diagnostics_rows.append(
                {
                    "epoch": float(epoch),
                    "grad_l2_total": float(stats.get("grad_l2_total", 0.0)),
                    "grad_l2_phi": float(stats.get("grad_l2_phi", 0.0)),
                    "grad_l2_Te": float(stats.get("grad_l2_Te", 0.0)),
                    "grad_l2_log_ne": float(stats.get("grad_l2_log_ne", 0.0)),
                    "grad_l2_log_ni": float(stats.get("grad_l2_log_ni", 0.0)),
                    "active_weight_ratio": float(stats.get("active_weight_ratio", 0.0)),
                    "step_rel_hidden_mean": float(stats.get("step_rel_hidden_mean", 0.0)),
                    "step_rel_output": float(stats.get("step_rel_output", 0.0)),
                    "step_rel_output_to_hidden": float(stats.get("step_rel_output_to_hidden", 0.0)),
                    "grad_scale_applied": float(stats.get("grad_scale_applied", 1.0)),
                    "grad_norm_pre_scale": float(stats.get("grad_norm_pre_scale", 0.0)),
                    "grad_norm_post_scale": float(stats.get("grad_norm_post_scale", 0.0)),
                    "grad_norm_post_clip": float(stats.get("grad_norm_post_clip", 0.0)),
                    "clip_ratio": float(stats.get("clip_ratio", 1.0)),
                    "effective_clip_flag": 0.0,
                    "stagnation_flag": 0.0,
                    "head_refresh_applied": float(head_refresh_applied),
                    "head_design_cond": float(head_design_cond),
                }
            )

        header = list(history[0].keys()) if history else ["epoch", "train_loss", "val_loss"]
        rows = [[h[k] for k in header] for h in history]
        self.store.save_csv("scalars/metrics.csv", header, rows)
        self._save_physics_terms(history)
        self._save_optimization_diagnostics(diagnostics_rows)
        return TrainOutput(history=history, model=model)

    def run_unet(
        self,
        model: UNetBaseline,
        cond_train: np.ndarray,
        y_train: np.ndarray,
        cond_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        physics_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        curriculum_cfg: dict[str, Any] | None = None,
        supervised_mask: np.ndarray | None = None,
        supervised_distance: np.ndarray | None = None,
        supervised_distance_signed: np.ndarray | None = None,
        supervised_bc_dir_mask: np.ndarray | None = None,
        supervised_wafer_mask: np.ndarray | None = None,
        optimizer_contract: dict[str, Any] | None = None,
        unet_optimizer_cfg: dict[str, Any] | None = None,
        batch_size_cases: int = 0,
        shuffle_cases: bool = True,
        seed: int = 0,
        selection_cfg: dict[str, Any] | None = None,
        selection_target_override: np.ndarray | None = None,
        selection_pred_additive: np.ndarray | None = None,
    ) -> TrainOutput:
        self._save_resolved_physics(physics_cfg)
        history: list[dict[str, float]] = []
        diagnostics_rows: list[dict[str, float]] = []
        contract = dict(optimizer_contract or {})
        diag_cfg = dict(contract.get("diagnostics", {}))
        diagnostics_enabled = bool(diag_cfg.get("enabled", True))
        alert_cfg = dict(diag_cfg.get("alert", {}))
        alert_enabled = bool(alert_cfg.get("enabled", False))
        min_effective_clip_ratio = float(alert_cfg.get("min_effective_clip_ratio", 0.99))
        clip_effective_direction = str(alert_cfg.get("clip_effective_direction", "lt")).strip().lower()
        min_grad_l2_total = float(alert_cfg.get("min_grad_l2_total", 1.0e-5))
        fail_fast_cfg = dict(diag_cfg.get("fail_fast", {}))
        fail_fast_enabled = bool(fail_fast_cfg.get("enabled", False))
        min_step_ratio = float(fail_fast_cfg.get("min_step_ratio", 1.0e-4))
        patience_epochs = int(max(int(fail_fast_cfg.get("patience_epochs", 10)), 1))
        selection_cfg = dict(selection_cfg or {})
        selection_mode = str(selection_cfg.get("mode", "last")).strip().lower()
        if selection_mode not in {"last", "best_val_allvars_balance"}:
            raise ValueError(
                "train.unet.selection.mode must be one of: last, best_val_allvars_balance"
            )
        selection_eval_every = int(max(int(selection_cfg.get("eval_every_n_epochs", 2)), 1))
        selection_warmup = int(max(int(selection_cfg.get("warmup_epochs", 5)), 0))
        selection_band_px = float(selection_cfg.get("boundary_band_px", 2.0))
        y_vars = [str(v) for v in list(getattr(model, "output_keys", []))]
        if len(y_vars) == 0:
            y_vars = ["ne", "Te", "phi"]
        raw_weights = dict(selection_cfg.get("weights", {}))
        if len(raw_weights) == 0:
            uniform = 1.0 / float(max(len(y_vars), 1))
            selection_allvars_weights = {str(name): float(uniform) for name in y_vars}
        else:
            unknown = sorted(set(str(k) for k in raw_weights.keys()) - set(str(v) for v in y_vars))
            if unknown:
                raise ValueError(
                    "train.unet.selection.weights contains unknown vars: "
                    f"{unknown}; expected={list(y_vars)}"
                )
            selection_allvars_weights = {str(name): float(raw_weights.get(name, 0.0)) for name in y_vars}
            total_w = 0.0
            for name in y_vars:
                w = float(selection_allvars_weights[name])
                if not np.isfinite(w) or w < 0.0:
                    raise ValueError(f"train.unet.selection.weights[{name}] must be finite and >= 0; got={w}")
                total_w += w
            if total_w <= 0.0:
                raise ValueError("train.unet.selection.weights must sum to > 0")
        torch_optimizer = None
        unet_opt_effective = {"type": "none", "lr": float(lr), "schedule": "none", "warmup_epochs": 0}
        if str(getattr(model, "backend", "numpy")).strip().lower() == "torch":
            unet_opt = _resolve_unet_optimizer_cfg(unet_optimizer_cfg, default_lr=float(lr))
            torch = model.torch
            torch_optimizer = torch.optim.AdamW(
                model.net.parameters(),
                lr=float(unet_opt["lr"]),
                betas=tuple(unet_opt["betas"]),
                eps=float(unet_opt["eps"]),
                weight_decay=float(unet_opt["weight_decay"]),
            )
            unet_opt_effective = dict(unet_opt)
        best_state = None
        best_score = float("-inf")
        best_epoch = -1
        selection_valid = selection_mode == "last"
        best_score_parts: dict[str, float] = {}
        stale = 0
        rng = np.random.default_rng(int(seed))
        for epoch in range(epochs):
            if torch_optimizer is not None:
                lr = _set_unet_optimizer_epoch_lr(
                    torch_optimizer,
                    base_lr=float(unet_opt_effective["lr"]),
                    epoch=int(epoch),
                    epochs=int(epochs),
                    schedule=str(unet_opt_effective.get("schedule", "none")),
                    warmup_epochs=int(unet_opt_effective.get("warmup_epochs", 0)),
                )
            epoch_physics_cfg = resolve_epoch_scaled_physics(
                physics_cfg,
                epoch=epoch,
                curriculum_cfg=curriculum_cfg,
            )
            resolved_terms = list(epoch_physics_cfg.get("resolved_terms", []))
            stats = train_one_epoch_unet(
                model,
                cond_train,
                y_train,
                lr=lr,
                physics_cfg=epoch_physics_cfg,
                resolved_terms=resolved_terms,
                loss_cfg=loss_cfg,
                supervised_mask=supervised_mask,
                supervised_distance=supervised_distance,
                supervised_distance_signed=supervised_distance_signed,
                supervised_bc_dir_mask=supervised_bc_dir_mask,
                supervised_wafer_mask=supervised_wafer_mask,
                optimizer_contract=contract,
                torch_optimizer=torch_optimizer,
                batch_size_cases=batch_size_cases,
                shuffle_cases=shuffle_cases,
                rng=rng,
                epoch_idx=epoch,
            )
            val_pred = model.forward(cond_val)
            val_data_loss, _, _ = compose_supervised_numpy(
                {name: val_pred[:, i] for i, name in enumerate(y_vars)},
                {name: y_val[:, i] for i, name in enumerate(y_vars)},
                y_order=y_vars,
                loss_cfg=loss_cfg,
                mask=supervised_mask,
                distance_any=supervised_distance,
                distance_signed=supervised_distance_signed,
                bc_dir_mask=supervised_bc_dir_mask,
                wafer_mask=supervised_wafer_mask,
                epoch_idx=epoch,
            )
            val_phys_loss = 0.0
            if epoch_physics_cfg and bool(epoch_physics_cfg.get("enabled", False)):
                if "phi" not in y_vars or "Te" not in y_vars:
                    raise ValueError("physics-aware training requires ne/log_ne, Te, phi in model output_keys")
                val_phys_loss = float(
                    compose_numpy(
                        {
                            "log_ne": _resolve_log_ne_field(val_pred, y_vars),
                            "Te": val_pred[:, y_vars.index("Te")],
                            "phi": val_pred[:, y_vars.index("phi")],
                        },
                        physics_cfg=epoch_physics_cfg,
                        resolved_terms=resolved_terms,
                    )[0]
                )
            val_total = float(val_data_loss + val_phys_loss)
            val_balance_score = float("nan")
            selection_score_te = float("nan")
            selection_score_phi = float("nan")
            should_eval_selection = (
                selection_mode == "best_val_allvars_balance"
                and epoch >= selection_warmup
                and ((epoch - selection_warmup) % selection_eval_every == 0)
            )
            if should_eval_selection:
                target_for_selection = (
                    np.asarray(selection_target_override, dtype=np.float32)
                    if selection_target_override is not None
                    else np.asarray(y_val, dtype=np.float32)
                )
                pred_for_selection = np.asarray(val_pred, dtype=np.float32)
                if selection_pred_additive is not None:
                    pred_for_selection = pred_for_selection + np.asarray(selection_pred_additive, dtype=np.float32)
                score, parts = _unet_allvars_boundary_balance_score(
                    pred=pred_for_selection,
                    target=target_for_selection,
                    y_vars=y_vars,
                    mask=supervised_mask,
                    distance_any=supervised_distance,
                    weights=selection_allvars_weights,
                    boundary_band_px=selection_band_px,
                    boundary_bonus_weight=0.0,
                )
                val_balance_score = float(score)
                selection_score_te = float(parts.get("r2_Te_plasma", float("nan")))
                selection_score_phi = float(parts.get("r2_phi_plasma", float("nan")))
                if np.isfinite(val_balance_score) and val_balance_score > best_score:
                    best_score = float(val_balance_score)
                    best_epoch = int(epoch)
                    best_state = _clone_model_state_numpy(model)
                    selection_valid = True
                    best_score_parts = dict(parts)

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": stats["total"],
                    "train_data_loss": stats["data"],
                    "train_phys_loss": stats["physics"],
                    "train_rho_loss": stats["rho"],
                    "train_poisson_loss": stats["poisson"],
                    "train_boundary_loss": stats["boundary"],
                    "train_boundary_operator_loss": stats["boundary_operator"],
                    "train_physics_scale": float(epoch_physics_cfg.get("physics_ramp_scale", 1.0)),
                    "val_loss": val_total,
                    "val_data_loss": float(val_data_loss),
                    "val_phys_loss": float(val_phys_loss),
                    "val_balance_score": float(val_balance_score),
                    "selection_score_Te": float(selection_score_te if np.isfinite(selection_score_te) else 0.0),
                    "selection_score_phi": float(selection_score_phi if np.isfinite(selection_score_phi) else 0.0),
                    "selection_valid_flag": 1.0 if selection_valid else 0.0,
                    "train_lr": float(lr),
                }
            )
            if diagnostics_enabled:
                diagnostics_rows.append(
                    {
                        "epoch": float(epoch),
                        "grad_l2_total": float(stats.get("grad_l2_total", 0.0)),
                        "grad_l2_phi": float(stats.get("grad_l2_phi", 0.0)),
                        "grad_l2_Te": float(stats.get("grad_l2_Te", 0.0)),
                        "grad_l2_ne": float(stats.get("grad_l2_ne", 0.0)),
                        "grad_l2_ni": float(stats.get("grad_l2_ni", 0.0)),
                        "active_weight_ratio": float(stats.get("active_weight_ratio", 0.0)),
                        "step_rel_hidden_mean": float(stats.get("step_rel_hidden_mean", 0.0)),
                        "step_rel_output": float(stats.get("step_rel_output", 0.0)),
                        "step_rel_output_to_hidden": float(stats.get("step_rel_output_to_hidden", 0.0)),
                        "grad_scale_applied": 1.0,
                        "grad_norm_pre_scale": 0.0,
                        "grad_norm_post_scale": 0.0,
                        "grad_norm_post_clip": float(stats.get("grad_norm_post_clip", 0.0)),
                        "clip_ratio": float(stats.get("clip_ratio", 1.0)),
                        "effective_clip_flag": float(
                            1.0
                            if (
                                alert_enabled
                                and str(contract.get("grad_clip", {}).get("mode", "off")).strip().lower() != "off"
                                and _is_effective_clip(
                                    clip_ratio=float(stats.get("clip_ratio", 1.0)),
                                    threshold=min_effective_clip_ratio,
                                    direction=clip_effective_direction,
                                )
                            )
                            else 0.0
                        ),
                        "stagnation_flag": float(
                            1.0
                            if (alert_enabled and float(stats.get("grad_l2_total", 0.0)) <= min_grad_l2_total)
                            else 0.0
                        ),
                        "head_refresh_applied": 0.0,
                        "head_design_cond": 0.0,
                        "selection_score_Te": float(selection_score_te if np.isfinite(selection_score_te) else 0.0),
                        "selection_score_phi": float(selection_score_phi if np.isfinite(selection_score_phi) else 0.0),
                        "selection_valid_flag": 1.0 if selection_valid else 0.0,
                        "selected_epoch_score": float(val_balance_score if np.isfinite(val_balance_score) else 0.0),
                        "selected_epoch_flag": 0.0,
                    }
                )
            if fail_fast_enabled:
                if float(stats.get("grad_l2_total", 0.0)) <= min_step_ratio:
                    stale += 1
                else:
                    stale = 0
                if stale >= patience_epochs:
                    break

        selected_epoch_effective = int(best_epoch if best_epoch >= 0 else (len(history) - 1))
        if selection_mode == "best_val_allvars_balance" and best_state is not None:
            _restore_model_state_numpy(model, best_state)
        for h in history:
            h["selected_epoch_flag"] = 1.0 if int(h["epoch"]) == selected_epoch_effective else 0.0
            h["selected_epoch_score"] = float(best_score if best_epoch >= 0 else 0.0)
            h["selection_valid_flag"] = 1.0 if selection_valid else 0.0
            h["selection_mode_effective"] = selection_mode
            h["unet_optimizer_effective"] = str(unet_opt_effective.get("type", "none"))
            if int(h["epoch"]) == selected_epoch_effective and best_epoch >= 0:
                h["selection_score_Te"] = float(best_score_parts.get("r2_Te_plasma", 0.0))
                h["selection_score_phi"] = float(best_score_parts.get("r2_phi_plasma", 0.0))
        for row in diagnostics_rows:
            row["selected_epoch_flag"] = 1.0 if int(row["epoch"]) == selected_epoch_effective else 0.0
            row["selection_valid_flag"] = 1.0 if selection_valid else 0.0
            row["selected_epoch_score"] = float(best_score if best_epoch >= 0 else 0.0)
            if int(row["epoch"]) == selected_epoch_effective and best_epoch >= 0:
                row["selection_score_Te"] = float(best_score_parts.get("r2_Te_plasma", 0.0))
                row["selection_score_phi"] = float(best_score_parts.get("r2_phi_plasma", 0.0))

        header = list(history[0].keys()) if history else ["epoch", "train_loss", "val_loss"]
        rows = [[h[k] for k in header] for h in history]
        self.store.save_csv("scalars/metrics.csv", header, rows)
        self._save_physics_terms(history)
        if diagnostics_enabled:
            self._save_optimization_diagnostics(diagnostics_rows)
        return TrainOutput(history=history, model=model)


__all__ = [
    "TrainOutput",
    "Trainer",
    "train_one_epoch_global",
    "train_one_epoch_unet",
]
