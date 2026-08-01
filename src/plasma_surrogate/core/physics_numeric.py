"""Small NumPy physics helpers shared outside training."""

from __future__ import annotations

from typing import Any

import numpy as np


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
    """Proxy or operator-prior target for boundary diagnostics/losses."""

    mode_norm = str(mode).strip().lower()
    coeff = target_coeffs or {}
    dens = np.asarray(density, dtype=np.float32)
    tt = np.asarray(te, dtype=np.float32)
    if mode_norm == "proxy":
        c_density = float(coeff.get("density", 0.10))
        c_te = float(coeff.get("temperature", 0.05))
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
            c_te = float(prior.get("temperature", 0.06))
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


__all__ = [
    "boundary_operator_loss",
    "boundary_operator_target",
    "laplacian2d",
    "poisson_residual",
    "poisson_residual_loss",
]
