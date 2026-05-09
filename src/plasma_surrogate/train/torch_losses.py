"""Torch losses for DeepONet PDE coupling."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.torch_backend import require_torch


def _as_bchw(x, *, batch_size: int | None = None, device: Any | None = None):
    torch = require_torch()
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 2:
        t = t[None, None, ...]
    elif t.ndim == 3:
        t = t[:, None, ...]
    elif t.ndim != 4:
        raise ValueError(f"expected [B,1,H,W] or [B,H,W], got {tuple(t.shape)}")
    if int(t.shape[1]) != 1:
        raise ValueError(f"expected single-channel tensor, got channel_dim={int(t.shape[1])}")
    if batch_size is not None:
        bs = int(batch_size)
        if int(t.shape[0]) not in {1, bs}:
            raise ValueError(f"expected batch dim 1 or {bs}, got {int(t.shape[0])}")
        if int(t.shape[0]) == 1 and bs > 1:
            t = t.expand(bs, -1, -1, -1)
    return t


def _gather_points(x, sample_idx):
    torch = require_torch()
    t = _as_bchw(x)
    idx = torch.as_tensor(sample_idx, dtype=torch.int64, device=t.device).reshape(-1)
    flat = t.reshape(t.shape[0], 1, -1).permute(0, 2, 1)
    return flat[:, idx, :]


def _as_scale_tensor(scale, ref):
    torch = require_torch()
    if scale is None:
        return None
    s = torch.as_tensor(scale, dtype=torch.float32, device=ref.device)
    if s.ndim == 0:
        s = s.reshape(1, 1, 1, 1)
    elif s.ndim == 1 and s.numel() == ref.shape[0]:
        s = s.reshape(ref.shape[0], 1, 1, 1)
    elif s.ndim == 2:
        s = s[None, None, ...]
    s = torch.clamp(s, min=1e-8)
    return s


def masked_huber_loss_torch(pred, target, mask=None, *, delta: float = 1.0):
    torch = require_torch()
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32, device=p.device)
    if p.shape != t.shape:
        raise ValueError(f"masked_huber_loss_torch shape mismatch: pred={tuple(p.shape)}, target={tuple(t.shape)}")
    loss_map = torch.nn.functional.huber_loss(p, t, delta=float(delta), reduction="none")
    if mask is None:
        return loss_map.mean()
    m = _as_bchw(mask, batch_size=int(p.shape[0]), device=p.device)
    if p.ndim == 3:
        p = p[:, None, ...]
    if p.ndim != 4 or int(p.shape[1]) != 1:
        raise ValueError(f"masked_huber_loss_torch expects [B,H,W] or [B,1,H,W], got {tuple(p.shape)}")
    if int(loss_map.ndim) == 3:
        loss_map = loss_map[:, None, ...]
    if tuple(loss_map.shape) != tuple(m.shape):
        raise ValueError(f"masked_huber_loss_torch mask mismatch: loss={tuple(loss_map.shape)} mask={tuple(m.shape)}")
    denom = torch.clamp(m.sum(), min=1.0)
    return (loss_map * m).sum() / denom


def masked_region_huber_loss_torch(
    pred,
    target,
    mask_plasma,
    distance_any,
    *,
    delta: float = 1.0,
    boundary_delta: float = 2.0,
    w_bulk: float = 1.0,
    w_boundary: float = 3.0,
):
    """Huber loss weighted by bulk/boundary classes over plasma mask."""

    torch = require_torch()
    p = _as_bchw(pred)
    t = _as_bchw(target, batch_size=int(p.shape[0]), device=p.device)
    m = _as_bchw(mask_plasma, batch_size=int(p.shape[0]), device=p.device)
    d_any = _as_bchw(distance_any, batch_size=int(p.shape[0]), device=p.device)
    loss_map = torch.nn.functional.huber_loss(p, t, delta=float(delta), reduction="none")
    active = (m > 0.5).to(dtype=loss_map.dtype)
    boundary = ((d_any <= float(boundary_delta)).to(dtype=loss_map.dtype)) * active
    bulk = (active - boundary).clamp(min=0.0)
    weights = boundary * float(w_boundary) + bulk * float(w_bulk)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (loss_map * weights).sum() / denom


def sdf_continuous_weight_map_torch(mask_plasma, distance_signed, cfg: dict[str, float] | None = None):
    """Torch equivalent of continuous SDF weighting across plasma/chamber."""

    torch = require_torch()
    c = dict(cfg or {})
    tau_in = float(max(float(c.get("tau_in", 2.0)), 1e-6))
    tau_out = float(max(float(c.get("tau_out", 1.5)), 1e-6))
    w_boundary = float(c.get("w_boundary", 3.0))
    w_inner = float(c.get("w_inner", 1.0))
    w_chamber_floor = float(c.get("w_chamber_floor", 0.08))

    m = _as_bchw(mask_plasma)
    ds = _as_bchw(distance_signed, batch_size=int(m.shape[0]))
    inside = ds >= 0.0
    w_in = w_inner + (w_boundary - w_inner) * torch.exp(-torch.clamp(ds, min=0.0) / tau_in)
    w_out = w_chamber_floor + (w_boundary - w_chamber_floor) * torch.exp(-torch.abs(ds) / tau_out)
    w = torch.where(inside, w_in, w_out)
    w = torch.where(m > 0.5, w, torch.maximum(w, torch.tensor(w_chamber_floor, dtype=w.dtype, device=w.device)))
    lo = float(min(w_inner, w_chamber_floor, w_boundary))
    hi = float(max(w_inner, w_chamber_floor, w_boundary))
    return torch.clamp(w, min=lo, max=hi)


def poisson_residual_fd_torch(phi, rhs=None, eps=None, mask=None, scale=None):
    """Finite-difference Poisson residual on [B,1,H,W] tensors."""
    torch = require_torch()
    p = _as_bchw(phi)
    pad = torch.nn.functional.pad(p, (1, 1, 1, 1), mode="replicate")
    lap = (
        pad[:, :, :-2, 1:-1]
        + pad[:, :, 2:, 1:-1]
        + pad[:, :, 1:-1, :-2]
        + pad[:, :, 1:-1, 2:]
        - 4.0 * pad[:, :, 1:-1, 1:-1]
    )
    res = lap
    if rhs is not None:
        res = res - _as_bchw(rhs, batch_size=int(p.shape[0]), device=p.device)
    if eps is not None:
        e = _as_bchw(eps, batch_size=int(p.shape[0]), device=p.device)
        res = res * e
    if mask is not None:
        m = _as_bchw(mask, batch_size=int(p.shape[0]), device=p.device)
        res = res * m
    s = _as_scale_tensor(scale, res)
    if s is not None:
        res = res / s
    return res


def boundary_operator_loss_torch(
    pred_fields: dict[str, Any],
    cond_vec: Any,
    geom_ctx: Any,
    operator_model: Any,
    mode: str = "operator_prior",
    primary_qoi_key: str = "Gamma_i",
    mask_band: Any | None = None,
    supervised_targets: dict[str, Any] | None = None,
    sample_idx: Any | None = None,
):
    """
    Boundary-operator loss.

    If `sample_idx` is None, this uses dense `mask_band` on the full grid.
    If provided, loss is evaluated only on the selected flattened points.
    """
    torch = require_torch()
    ln = _as_bchw(pred_fields["log_ne"])
    te = _as_bchw(pred_fields["Te"])
    phi = _as_bchw(pred_fields["phi"])
    if mask_band is None:
        m = torch.ones_like(phi)
    else:
        m = _as_bchw(mask_band, batch_size=int(phi.shape[0]), device=phi.device)
    if sample_idx is not None:
        m = _gather_points(m, sample_idx)
    pred = operator_model.predict_target(
        log_ne=ln,
        te=te,
        phi=phi,
        cond=cond_vec,
        geom_ctx=geom_ctx,
        primary_qoi_key=primary_qoi_key,
        sample_idx=sample_idx,
    )[primary_qoi_key]
    if mode == "supervised":
        if supervised_targets is None or primary_qoi_key not in supervised_targets:
            raise ValueError("supervised boundary operator loss requires supervised_targets[primary_qoi_key]")
        tgt = _as_bchw(supervised_targets[primary_qoi_key], batch_size=int(phi.shape[0]), device=phi.device)
        if sample_idx is not None:
            tgt = _gather_points(tgt, sample_idx)
    else:
        # operator-prior: compare against a simple Bohm-like proxy
        if sample_idx is not None:
            tgt = 0.10 * _gather_points(ln, sample_idx) + 0.05 * _gather_points(te, sample_idx)
        else:
            tgt = 0.10 * ln + 0.05 * te
    diff = (pred - tgt) * m
    denom = torch.clamp(m.sum(), min=1.0)
    return (diff * diff).sum() / denom


def physics_terms_torch(
    pred_fields: dict[str, Any],
    cond_vec: Any,
    geom_ctx: Any,
    cfg: dict[str, Any] | None,
    boundary_operator_model: Any | None = None,
    supervised_targets: dict[str, Any] | None = None,
):
    torch = require_torch()
    ref_device = None
    if pred_fields:
        ref_device = torch.as_tensor(next(iter(pred_fields.values())), dtype=torch.float32).device
    if not cfg or not bool(cfg.get("enabled", False)):
        z = torch.zeros((), dtype=torch.float32, device=ref_device)
        return z, {"poisson": 0.0, "boundary_operator": 0.0}

    phi = _as_bchw(pred_fields["phi"])
    lambda_poisson = float(cfg.get("lambda_poisson", 0.0))
    rhs = cfg.get("rhs")
    if rhs is None and bool(cfg.get("use_pred_rho_eff", True)) and ("rho_eff" in pred_fields):
        # Poisson form: lap(phi) + rho_eff = 0  => lap(phi) - (-rho_eff) = 0
        rhs = -_as_bchw(pred_fields["rho_eff"])
    mask = cfg.get("mask")
    if mask is None and (geom_ctx is not None) and hasattr(geom_ctx, "mask_plasma"):
        mask = getattr(geom_ctx, "mask_plasma")
    eps = cfg.get("eps")
    if eps is None and (geom_ctx is not None) and hasattr(geom_ctx, "eps"):
        eps = getattr(geom_ctx, "eps")
    scale_rho = cfg.get("scale_rho", cfg.get("scale"))
    poisson = torch.zeros((), dtype=torch.float32, device=phi.device)
    if lambda_poisson > 0.0:
        res = poisson_residual_fd_torch(phi, rhs=rhs, eps=eps, mask=mask, scale=scale_rho)
        clamp_val = cfg.get("poisson_clamp")
        if clamp_val is not None:
            cv = float(clamp_val)
            res = torch.clamp(res, min=-cv, max=cv)
        poisson = float(lambda_poisson) * (res * res).mean()

    bo_term = torch.zeros((), dtype=torch.float32, device=phi.device)
    bo_cfg = cfg.get("boundary_operator", {})
    if (
        boundary_operator_model is not None
        and bool(bo_cfg.get("enabled", False))
        and float(bo_cfg.get("lambda", 0.0)) > 0.0
    ):
        bo = boundary_operator_loss_torch(
            pred_fields=pred_fields,
            cond_vec=cond_vec,
            geom_ctx=geom_ctx,
            operator_model=boundary_operator_model,
            mode=str(bo_cfg.get("mode", "operator_prior")),
            primary_qoi_key=str(bo_cfg.get("primary_qoi_key", "Gamma_i")),
            mask_band=bo_cfg.get("mask_band"),
            supervised_targets=supervised_targets,
            sample_idx=bo_cfg.get("sample_idx"),
        )
        bo_term = float(bo_cfg.get("lambda", 0.0)) * bo

    total = poisson + bo_term
    return total, {"poisson": float(poisson.detach().cpu().item()), "boundary_operator": float(bo_term.detach().cpu().item())}
