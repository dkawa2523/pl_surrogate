"""Evaluation metrics for cycle1."""

from __future__ import annotations

import numpy as np


def _broadcast_mask_to_shape(mask: np.ndarray, target: np.ndarray, *, name: str) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    t = np.asarray(target)
    if t.ndim == 2:
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.ndim != 2:
            raise ValueError(f"{name} mask mismatch: expected {t.shape}, got {m.shape}")
        return m
    if t.ndim == 3:
        if m.ndim == 2:
            m = m[None, ...]
        elif m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        if m.ndim == 3 and m.shape[0] == 1 and t.shape[0] > 1:
            m = np.repeat(m, t.shape[0], axis=0)
        if m.shape != t.shape:
            raise ValueError(f"{name} mask mismatch: expected {t.shape}, got {m.shape}")
        return m
    if t.ndim == 4:
        if m.ndim == 2:
            m = m[None, None, ...]
        elif m.ndim == 3:
            m = m[:, None, ...]
        if m.ndim != 4:
            raise ValueError(f"{name} mask mismatch: expected {t.shape}, got {m.shape}")
        if m.shape[0] == 1 and t.shape[0] > 1:
            m = np.repeat(m, t.shape[0], axis=0)
        if m.shape[1] == 1 and t.shape[1] > 1:
            m = np.repeat(m, t.shape[1], axis=1)
        if m.shape != t.shape:
            raise ValueError(f"{name} mask mismatch: expected {t.shape}, got {m.shape}")
        return m
    raise ValueError(f"{name} only supports 2D/3D/4D tensors, got {t.shape}")


def _finite_pair_or_nan(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"metric shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return None
    return yt, yp


def finite_pair_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Count finite/non-finite paired values used by evaluation metrics."""

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"finite_pair_stats shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if mask is None:
        active = np.ones(yt.shape, dtype=bool)
    else:
        active = _broadcast_mask_to_shape(mask, yt, name="finite_pair_stats") > 0.5
    n_active = int(np.sum(active))
    if n_active <= 0:
        return {"n_active": 0.0, "n_finite": 0.0, "n_nonfinite": 0.0, "finite_ratio": 0.0}
    finite = np.isfinite(yt) & np.isfinite(yp)
    n_finite = int(np.sum(active & finite))
    n_nonfinite = int(n_active - n_finite)
    return {
        "n_active": float(n_active),
        "n_finite": float(n_finite),
        "n_nonfinite": float(n_nonfinite),
        "finite_ratio": float(n_finite / float(n_active)),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pair = _finite_pair_or_nan(y_true, y_pred)
    if pair is None:
        return float("nan")
    yt, yp = pair
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pair = _finite_pair_or_nan(y_true, y_pred)
    if pair is None:
        return float("nan")
    yt, yp = pair
    return float(np.mean(np.abs(yt - yp)))


def rmse_by_var(y_true: dict[str, np.ndarray], y_pred: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: rmse(y_true[k], y_pred[k]) for k in y_true.keys()}


def rmse_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"rmse_masked shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    m = _broadcast_mask_to_shape(mask, yt, name="rmse_masked")
    active = m > 0.5
    if not np.any(active):
        return 0.0
    if not (np.all(np.isfinite(yt[active])) and np.all(np.isfinite(yp[active]))):
        return float("nan")
    diff = yt[active] - yp[active]
    return float(np.sqrt(np.mean(diff * diff)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"r2 shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-18:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def r2_by_var(y_true: dict[str, np.ndarray], y_pred: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: r2(y_true[k], y_pred[k]) for k in y_true.keys()}


def r2_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"r2_masked shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    m = _broadcast_mask_to_shape(mask, yt, name="r2_masked")
    active = m > 0.5
    if not np.any(active):
        return 0.0
    yta = yt[active]
    ypa = yp[active]
    if not (np.all(np.isfinite(yta)) and np.all(np.isfinite(ypa))):
        return float("nan")
    ss_res = float(np.sum((yta - ypa) ** 2))
    ss_tot = float(np.sum((yta - np.mean(yta)) ** 2))
    if ss_tot <= 1e-18:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def uniformity(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return float("nan")
    if not np.all(np.isfinite(vals)):
        return float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if (not np.isfinite(mean)) or (not np.isfinite(std)):
        return float("nan")
    return float(std / (abs(mean) + 1e-12))


def poisson_residual_norm(phi: np.ndarray, eps: np.ndarray | None = None) -> float:
    f = np.asarray(phi, dtype=np.float32)
    if f.ndim == 2:
        f = f[None, ...]
    p = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode="edge")
    lap = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4.0 * p[:, 1:-1, 1:-1]
    if eps is not None:
        e = np.asarray(eps, dtype=np.float32)
        if e.ndim == 2:
            e = e[None, ...]
        if e.shape[0] == 1 and lap.shape[0] > 1:
            e = np.repeat(e, lap.shape[0], axis=0)
        lap = lap * e
    return float(np.mean(np.abs(lap)))


def bc_mae(phi: np.ndarray, bc_mask: np.ndarray, bc_value: np.ndarray | float = 0.0) -> float:
    p = np.asarray(phi, dtype=np.float32)
    m = np.asarray(bc_mask, dtype=np.float32)
    if p.ndim == 3:
        p = p[0]
    if np.isscalar(bc_value):
        v = np.full_like(m, float(bc_value), dtype=np.float32)
    else:
        v = np.asarray(bc_value, dtype=np.float32)
    active = m > 0.5
    if not np.any(active):
        return 0.0
    return float(np.mean(np.abs(p[active] - v[active])))


def region_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    t = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(y_pred, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32) > 0.5
    if not np.any(m):
        return 0.0
    return float(np.sqrt(np.mean((t[m] - p[m]) ** 2)))


def boundary_band_mask(
    mask_plasma: np.ndarray,
    distance_any: np.ndarray,
    delta_edge: float = 1.5,
    wafer_mask: np.ndarray | None = None,
    wafer_only: bool = False,
) -> np.ndarray:
    m = (np.asarray(mask_plasma, dtype=np.float32) > 0.5) & (np.asarray(distance_any, dtype=np.float32) <= float(delta_edge))
    if wafer_only and wafer_mask is not None:
        m = m & (np.asarray(wafer_mask, dtype=np.float32) > 0.5)
    return m.astype(np.float32)


def boundary_gamma_proxy(
    log_ne: np.ndarray,
    te: np.ndarray,
    phi: np.ndarray,
    mask_band: np.ndarray,
) -> np.ndarray:
    ln = np.asarray(log_ne, dtype=np.float32)
    tt = np.asarray(te, dtype=np.float32)
    p = np.asarray(phi, dtype=np.float32)
    m = np.asarray(mask_band, dtype=np.float32) > 0.5
    gy, gx = np.gradient(p, edge_order=1)
    e_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)
    ne = np.power(10.0, np.clip(ln, -6.0, 6.0)).astype(np.float32)
    gamma = ne * np.sqrt(np.maximum(tt, 0.0) + 1e-6) * e_mag
    return gamma[m].astype(np.float32)
