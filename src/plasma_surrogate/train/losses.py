"""Physics-aware loss helpers for surrogate training."""

from __future__ import annotations

from typing import Any

import numpy as np


def masked_huber_loss(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    delta: float = 1.0,
) -> float:
    """Mean Huber loss on active mask region (all entries if mask is None)."""

    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"masked_huber_loss shape mismatch: pred={p.shape}, target={t.shape}")
    err = p - t
    d = float(max(delta, 1e-8))
    abs_err = np.abs(err)
    loss_map = np.where(abs_err <= d, 0.5 * err * err, d * (abs_err - 0.5 * d)).astype(np.float32)
    if mask is None:
        return float(np.mean(loss_map))
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    if m.shape[0] == 1 and p.shape[0] > 1:
        m = np.repeat(m, p.shape[0], axis=0)
    if m.shape != p.shape:
        raise ValueError(f"masked_huber_loss mask mismatch: expected {p.shape}, got {m.shape}")
    denom = max(float(np.sum(m)), 1.0)
    return float(np.sum(loss_map * m) / denom)


def masked_region_huber_loss(
    pred: np.ndarray,
    target: np.ndarray,
    mask_plasma: np.ndarray,
    distance_any: np.ndarray,
    *,
    delta: float = 1.0,
    boundary_delta: float = 2.0,
    w_bulk: float = 1.0,
    w_boundary: float = 3.0,
) -> float:
    """Huber loss weighted by plasma-region classes (bulk/boundary)."""

    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"masked_region_huber_loss shape mismatch: pred={p.shape}, target={t.shape}")

    m = np.asarray(mask_plasma, dtype=np.float32)
    d_any = np.asarray(distance_any, dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    if d_any.ndim == 2:
        d_any = d_any[None, ...]
    if m.shape[0] == 1 and p.shape[0] > 1:
        m = np.repeat(m, p.shape[0], axis=0)
    if d_any.shape[0] == 1 and p.shape[0] > 1:
        d_any = np.repeat(d_any, p.shape[0], axis=0)
    if m.shape != p.shape or d_any.shape != p.shape:
        raise ValueError(
            "masked_region_huber_loss mask/distance mismatch: "
            f"pred={p.shape}, mask={m.shape}, distance={d_any.shape}"
        )

    err = p - t
    d = float(max(delta, 1e-8))
    abs_err = np.abs(err)
    loss_map = np.where(abs_err <= d, 0.5 * err * err, d * (abs_err - 0.5 * d)).astype(np.float32)

    active = (m > 0.5).astype(np.float32)
    boundary = (d_any <= float(boundary_delta)).astype(np.float32) * active
    bulk = (active - boundary).astype(np.float32)
    weights = boundary * float(w_boundary) + bulk * float(w_bulk)
    denom = max(float(np.sum(weights)), 1.0)
    return float(np.sum(loss_map * weights) / denom)


def build_signed_distance(mask_plasma: np.ndarray, distance_any: np.ndarray) -> np.ndarray:
    """Build signed distance where plasma region is positive and chamber region is negative."""

    m = np.asarray(mask_plasma, dtype=np.float32)
    d = np.asarray(distance_any, dtype=np.float32)
    if m.shape != d.shape:
        raise ValueError(f"build_signed_distance shape mismatch: mask={m.shape}, distance={d.shape}")
    return np.where(m > 0.5, d, -d).astype(np.float32)


def is_effective_signed_distance(
    mask_plasma: np.ndarray,
    distance_signed: np.ndarray,
    *,
    min_negative_ratio: float = 1e-6,
) -> bool:
    """Return True if chamber-side signed distances contain sufficient negative values."""

    m = np.asarray(mask_plasma, dtype=np.float32)
    ds = np.asarray(distance_signed, dtype=np.float32)
    if m.shape != ds.shape:
        raise ValueError(f"is_effective_signed_distance shape mismatch: mask={m.shape}, signed={ds.shape}")
    outside = m <= 0.5
    if not np.any(outside):
        return True
    ratio = float(np.mean((ds[outside] < 0.0).astype(np.float32)))
    return ratio >= float(min_negative_ratio)


def sdf_continuous_weight_map(
    mask_plasma: np.ndarray,
    distance_signed: np.ndarray,
    cfg: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Continuous supervision weights over plasma/chamber using signed distance.

    - inside plasma (d>=0): weight decays from w_boundary -> w_inner
    - chamber side (d<0): weight decays from w_boundary -> w_chamber_floor
    """

    c = dict(cfg or {})
    tau_in = float(max(float(c.get("tau_in", 2.0)), 1e-6))
    tau_out = float(max(float(c.get("tau_out", 1.5)), 1e-6))
    w_boundary = float(c.get("w_boundary", 3.0))
    w_inner = float(c.get("w_inner", 1.0))
    w_chamber_floor = float(c.get("w_chamber_floor", 0.08))

    m = np.asarray(mask_plasma, dtype=np.float32)
    ds = np.asarray(distance_signed, dtype=np.float32)
    if m.shape != ds.shape:
        raise ValueError(f"sdf_continuous_weight_map shape mismatch: mask={m.shape}, signed={ds.shape}")
    inside = ds >= 0.0
    out_abs = np.abs(ds)
    w_in = w_inner + (w_boundary - w_inner) * np.exp(-np.maximum(ds, 0.0) / tau_in)
    w_out = w_chamber_floor + (w_boundary - w_chamber_floor) * np.exp(-out_abs / tau_out)
    w = np.where(inside, w_in, w_out).astype(np.float32)
    w = np.where(m > 0.5, w, np.maximum(w, w_chamber_floor)).astype(np.float32)
    return np.clip(w, min(w_inner, w_chamber_floor, w_boundary), max(w_inner, w_chamber_floor, w_boundary)).astype(
        np.float32
    )


def laplacian2d(phi: np.ndarray) -> np.ndarray:
    """Return 2D Laplacian for [B,H,W] tensor using edge-replicate padding."""

    arr = np.asarray(phi, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"laplacian2d expects [B,H,W], got shape={arr.shape}")
    p = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
    center = p[:, 1:-1, 1:-1]
    up = p[:, :-2, 1:-1]
    down = p[:, 2:, 1:-1]
    left = p[:, 1:-1, :-2]
    right = p[:, 1:-1, 2:]
    return (up + down + left + right - 4.0 * center).astype(np.float32)


def poisson_residual(phi: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    lap = laplacian2d(phi)
    if rhs is None:
        return lap
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    if rhs_arr.ndim == 2:
        rhs_arr = rhs_arr[None, ...]
    if rhs_arr.shape[0] == 1 and lap.shape[0] > 1:
        rhs_arr = np.repeat(rhs_arr, lap.shape[0], axis=0)
    if rhs_arr.shape != lap.shape:
        raise ValueError(f"rhs shape mismatch: rhs={rhs_arr.shape}, phi={lap.shape}")
    return (lap - rhs_arr).astype(np.float32)


def poisson_residual_loss(phi: np.ndarray, rhs: np.ndarray | None = None) -> float:
    res = poisson_residual(phi, rhs=rhs)
    return float(np.mean(res**2))


def poisson_residual_grad(phi: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    """Gradient of mean((Lap(phi)-rhs)^2) wrt phi (using symmetric Laplacian approximation)."""

    res = poisson_residual(phi, rhs=rhs)
    grad = (2.0 / float(res.size)) * laplacian2d(res)
    return grad.astype(np.float32)


def boundary_loss(phi: np.ndarray, bc_mask: np.ndarray, bc_value: np.ndarray | float = 0.0) -> float:
    mask = np.asarray(bc_mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.shape[0] == 1 and phi.shape[0] > 1:
        mask = np.repeat(mask, phi.shape[0], axis=0)
    value = np.asarray(bc_value, dtype=np.float32)
    if value.ndim == 0:
        value = np.full_like(mask, float(value), dtype=np.float32)
    elif value.ndim == 2:
        value = value[None, ...]
    if value.shape[0] == 1 and phi.shape[0] > 1:
        value = np.repeat(value, phi.shape[0], axis=0)
    if mask.shape != phi.shape or value.shape != phi.shape:
        raise ValueError(f"boundary shape mismatch: phi={phi.shape}, mask={mask.shape}, value={value.shape}")
    diff = (phi - value) * mask
    return float(np.mean(diff**2))


def boundary_grad(phi: np.ndarray, bc_mask: np.ndarray, bc_value: np.ndarray | float = 0.0) -> np.ndarray:
    mask = np.asarray(bc_mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.shape[0] == 1 and phi.shape[0] > 1:
        mask = np.repeat(mask, phi.shape[0], axis=0)
    value = np.asarray(bc_value, dtype=np.float32)
    if value.ndim == 0:
        value = np.full_like(mask, float(value), dtype=np.float32)
    elif value.ndim == 2:
        value = value[None, ...]
    if value.shape[0] == 1 and phi.shape[0] > 1:
        value = np.repeat(value, phi.shape[0], axis=0)
    grad = (2.0 / float(phi.size)) * (phi - value) * mask
    return grad.astype(np.float32)


def _as_batch(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    if a.shape[0] == 1 and ref.shape[0] > 1:
        a = np.repeat(a, ref.shape[0], axis=0)
    if a.shape != ref.shape:
        raise ValueError(f"shape mismatch: expected {ref.shape}, got {a.shape}")
    return a.astype(np.float32)


def boundary_operator_target(
    density: np.ndarray,
    te: np.ndarray,
    phi: np.ndarray | None = None,
    mode: str = "proxy",
    target_coeffs: dict[str, float] | None = None,
    prior_coeffs: dict[str, float] | None = None,
    operator_handle: Any | None = None,
    target_clamp: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Proxy target for boundary operator prior.

    This keeps the product path lightweight while exposing a stable contract:
    target_phi = c_density * density + c_Te * Te + bias
    """

    mode_norm = str(mode).strip().lower()
    coeff = target_coeffs or {}
    dens = np.asarray(density, dtype=np.float32)
    tt = np.asarray(te, dtype=np.float32)
    if mode_norm == "proxy":
        c_density = float(coeff.get("density", 0.10))
        c_te = float(coeff.get("Te", 0.05))
        bias = float(coeff.get("bias", 0.0))
        target = c_density * dens + c_te * tt + bias
    elif mode_norm == "operator_prior":
        if operator_handle is not None:
            if phi is None:
                raise ValueError("boundary_operator_target(mode=operator_prior) with operator_handle requires phi")
            p = np.asarray(phi, dtype=np.float32)
            if hasattr(operator_handle, "predict_target"):
                target = operator_handle.predict_target(dens, tt, p)
            elif callable(operator_handle):
                target = operator_handle(dens, tt, p)
            else:
                raise TypeError("operator_handle must be callable or implement predict_target(density, te, phi)")
            target = np.asarray(target, dtype=np.float32)
            if target.shape != dens.shape:
                raise ValueError(f"operator_handle target shape mismatch: expected {dens.shape}, got {target.shape}")
        else:
            if phi is None:
                raise ValueError("boundary_operator_target(mode=operator_prior) requires phi")
            p = np.asarray(phi, dtype=np.float32)
            if p.ndim != 3:
                raise ValueError(f"phi must be [B,H,W], got {p.shape}")
            gy, gx = np.gradient(p, axis=(-2, -1), edge_order=1)
            e_n = np.sqrt(gx**2 + gy**2).astype(np.float32)
            prior = prior_coeffs or {}
            c_density = float(prior.get("density", 0.08))
            c_te = float(prior.get("Te", 0.06))
            c_en = float(prior.get("E_n", 0.04))
            bias = float(prior.get("bias", 0.0))
            target = c_density * dens + c_te * tt + c_en * e_n + bias
    else:
        raise ValueError(f"Unknown boundary operator mode: {mode}")

    if target_clamp is not None:
        lo = float(target_clamp[0])
        hi = float(target_clamp[1])
        target = np.clip(target, lo, hi)
    return target.astype(np.float32)


def boundary_operator_loss(
    phi: np.ndarray,
    density: np.ndarray,
    te: np.ndarray,
    mask_band: np.ndarray,
    mode: str = "proxy",
    target_coeffs: dict[str, float] | None = None,
    prior_coeffs: dict[str, float] | None = None,
    operator_handle: Any | None = None,
    target_clamp: tuple[float, float] | None = None,
) -> float:
    p = np.asarray(phi, dtype=np.float32)
    dens = _as_batch(np.asarray(density, dtype=np.float32), p)
    tt = _as_batch(np.asarray(te, dtype=np.float32), p)
    m = _as_batch(np.asarray(mask_band, dtype=np.float32), p)
    t = boundary_operator_target(
        dens,
        tt,
        phi=p,
        mode=mode,
        target_coeffs=target_coeffs,
        prior_coeffs=prior_coeffs,
        operator_handle=operator_handle,
        target_clamp=target_clamp,
    )
    denom = max(float(np.sum(m)), 1.0)
    diff = (p - t) * m
    return float(np.sum(diff**2) / denom)


def boundary_operator_grad(
    phi: np.ndarray,
    density: np.ndarray,
    te: np.ndarray,
    mask_band: np.ndarray,
    mode: str = "proxy",
    target_coeffs: dict[str, float] | None = None,
    prior_coeffs: dict[str, float] | None = None,
    operator_handle: Any | None = None,
    target_clamp: tuple[float, float] | None = None,
) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float32)
    dens = _as_batch(np.asarray(density, dtype=np.float32), p)
    tt = _as_batch(np.asarray(te, dtype=np.float32), p)
    m = _as_batch(np.asarray(mask_band, dtype=np.float32), p)
    t = boundary_operator_target(
        dens,
        tt,
        phi=p,
        mode=mode,
        target_coeffs=target_coeffs,
        prior_coeffs=prior_coeffs,
        operator_handle=operator_handle,
        target_clamp=target_clamp,
    )
    denom = max(float(np.sum(m)), 1.0)
    grad = (2.0 / denom) * (p - t) * m
    return grad.astype(np.float32)


def physics_loss_and_grad(
    phi: np.ndarray,
    cfg: dict[str, Any] | None = None,
    density: np.ndarray | None = None,
    te: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return (loss, grad_phi, components) for enabled physics terms."""

    if not cfg or not bool(cfg.get("enabled", False)):
        z = np.zeros_like(phi, dtype=np.float32)
        return 0.0, z, {"poisson": 0.0, "boundary": 0.0, "boundary_operator": 0.0}

    poisson_weight = float(cfg.get("poisson_weight", 0.0))
    boundary_weight = float(cfg.get("boundary_weight", 0.0))
    rhs = cfg.get("rhs")
    bc_mask = cfg.get("bc_mask")
    bc_value = cfg.get("bc_value", 0.0)

    grad = np.zeros_like(phi, dtype=np.float32)
    poisson_term = 0.0
    boundary_term = 0.0
    boundary_operator_term = 0.0

    if poisson_weight > 0.0:
        p_loss = poisson_residual_loss(phi, rhs=rhs)
        p_grad = poisson_residual_grad(phi, rhs=rhs)
        poisson_term = poisson_weight * p_loss
        grad += poisson_weight * p_grad

    if boundary_weight > 0.0 and bc_mask is not None:
        b_loss = boundary_loss(phi, bc_mask=bc_mask, bc_value=bc_value)
        b_grad = boundary_grad(phi, bc_mask=bc_mask, bc_value=bc_value)
        boundary_term = boundary_weight * b_loss
        grad += boundary_weight * b_grad

    bo_cfg = cfg.get("boundary_operator", {})
    bo_enabled = bool(bo_cfg.get("enabled", False))
    boundary_operator_weight = float(bo_cfg.get("weight", 0.0))
    if bo_enabled and boundary_operator_weight > 0.0:
        if density is None or te is None:
            raise ValueError("boundary_operator requires density and te tensors")
        mask_band = bo_cfg.get("mask_band")
        if mask_band is None:
            raise ValueError("boundary_operator requires mask_band")
        bo_loss = boundary_operator_loss(
            phi=phi,
            density=density,
            te=te,
            mask_band=np.asarray(mask_band, dtype=np.float32),
            mode=str(bo_cfg.get("mode", "proxy")),
            target_coeffs=bo_cfg.get("target_coeffs"),
            prior_coeffs=bo_cfg.get("prior_coeffs"),
            operator_handle=bo_cfg.get("operator_handle"),
            target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
        )
        bo_grad = boundary_operator_grad(
            phi=phi,
            density=density,
            te=te,
            mask_band=np.asarray(mask_band, dtype=np.float32),
            mode=str(bo_cfg.get("mode", "proxy")),
            target_coeffs=bo_cfg.get("target_coeffs"),
            prior_coeffs=bo_cfg.get("prior_coeffs"),
            operator_handle=bo_cfg.get("operator_handle"),
            target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
        )
        boundary_operator_term = boundary_operator_weight * bo_loss
        grad += boundary_operator_weight * bo_grad

    total = poisson_term + boundary_term + boundary_operator_term
    return float(total), grad.astype(np.float32), {
        "poisson": float(poisson_term),
        "boundary": float(boundary_term),
        "boundary_operator": float(boundary_operator_term),
    }
