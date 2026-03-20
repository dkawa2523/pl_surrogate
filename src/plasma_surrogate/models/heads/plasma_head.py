"""Shared plasma field head for phi_mode handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.train.losses import laplacian2d


def _to_numpy_f32(x: Any) -> np.ndarray:
    v = x
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    return np.asarray(v, dtype=np.float32)


def _expand_batch(arr: np.ndarray, bsz: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    if a.shape[0] == 1 and bsz > 1:
        a = np.repeat(a, bsz, axis=0)
    return a.astype(np.float32)


def _harmonic_extension(
    bc_mask: np.ndarray,
    bc_value: np.ndarray,
    mask_active: np.ndarray,
    n_iters: int = 64,
) -> np.ndarray:
    """Simple harmonic extension used for hard BC embedding and phi0 init."""

    bsz, h, w = bc_mask.shape
    phi = np.zeros((bsz, h, w), dtype=np.float32)
    phi = (1.0 - bc_mask) * phi + bc_mask * bc_value
    active = (mask_active > 0.5).astype(np.float32)

    for _ in range(max(1, int(n_iters))):
        padded = np.pad(phi, ((0, 0), (1, 1), (1, 1)), mode="edge")
        neigh = (
            padded[:, :-2, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2]
            + padded[:, 1:-1, 2:]
        ) * 0.25
        phi = active * ((1.0 - bc_mask) * neigh + bc_mask * bc_value)
    return phi.astype(np.float32)


def _jacobi_poisson(
    rho_eff: np.ndarray,
    eps: np.ndarray,
    bc_mask: np.ndarray,
    bc_value: np.ndarray,
    phi0: np.ndarray,
    mask_active: np.ndarray,
    n_iters: int,
) -> np.ndarray:
    """Differentiability is not required in this numpy path; deterministic solve is enough."""

    bsz, h, w = rho_eff.shape
    e = _expand_batch(eps, bsz)
    active = (mask_active > 0.5).astype(np.float32)
    phi = np.asarray(phi0, dtype=np.float32).copy()
    phi = active * ((1.0 - bc_mask) * phi + bc_mask * bc_value)

    # face-centered eps (harmonic-like arithmetic average for v1)
    ex_p = np.zeros_like(e)
    ex_m = np.zeros_like(e)
    ey_p = np.zeros_like(e)
    ey_m = np.zeros_like(e)
    ex_p[:, :, :-1] = 0.5 * (e[:, :, :-1] + e[:, :, 1:])
    ex_p[:, :, -1] = e[:, :, -1]
    ex_m[:, :, 1:] = 0.5 * (e[:, :, 1:] + e[:, :, :-1])
    ex_m[:, :, 0] = e[:, :, 0]
    ey_p[:, :-1, :] = 0.5 * (e[:, :-1, :] + e[:, 1:, :])
    ey_p[:, -1, :] = e[:, -1, :]
    ey_m[:, 1:, :] = 0.5 * (e[:, 1:, :] + e[:, :-1, :])
    ey_m[:, 0, :] = e[:, 0, :]
    denom = np.maximum(ex_p + ex_m + ey_p + ey_m, 1e-6)

    for _ in range(max(1, int(n_iters))):
        p = np.pad(phi, ((0, 0), (1, 1), (1, 1)), mode="edge")
        pxp = p[:, 1:-1, 2:]
        pxm = p[:, 1:-1, :-2]
        pyp = p[:, 2:, 1:-1]
        pym = p[:, :-2, 1:-1]
        num = ex_p * pxp + ex_m * pxm + ey_p * pyp + ey_m * pym + rho_eff
        nxt = num / denom
        phi = active * ((1.0 - bc_mask) * nxt + bc_mask * bc_value)
    return phi.astype(np.float32)


@dataclass
class PlasmaHead:
    mode: str = "direct"
    jacobi_iters: int = 3
    hard_bc_extension_iters: int = 64

    def _phi_bc_ext(self, geom_ctx: GeometryContext, bsz: int) -> np.ndarray:
        if "phi_bc_ext" in geom_ctx.regions:
            return _expand_batch(np.asarray(geom_ctx.regions["phi_bc_ext"], dtype=np.float32), bsz)
        if "bc_basis" in geom_ctx.regions and "bc_coeffs" in geom_ctx.regions:
            basis = np.asarray(geom_ctx.regions["bc_basis"], dtype=np.float32)  # [G,H,W]
            coeffs = np.asarray(geom_ctx.regions["bc_coeffs"], dtype=np.float32).reshape(-1, 1, 1)
            ext = np.sum(basis * coeffs, axis=0)
            return _expand_batch(ext, bsz)

        bc_mask = np.zeros_like(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx.bc_dir_mask is None else geom_ctx.bc_dir_mask
        bc_val = np.zeros_like(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx.bc_dir_value is None else geom_ctx.bc_dir_value
        m = _expand_batch(bc_mask, bsz)
        v = _expand_batch(bc_val, bsz)
        active = _expand_batch(geom_ctx.mask_plasma, bsz)
        return _harmonic_extension(m, v, active, n_iters=self.hard_bc_extension_iters)

    def apply(
        self,
        fields_phys: dict[str, np.ndarray],
        geom_ctx: GeometryContext,
        refine_iters: int = 0,
        deeponet_head: Any | None = None,
        cond_vec: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply phi mode and return (fields, aux)."""

        out = {k: _to_numpy_f32(v).copy() for k, v in fields_phys.items()}
        phi_in = np.asarray(out["phi"], dtype=np.float32)[:, 0]  # [B,H,W]
        bsz = phi_in.shape[0]

        active = _expand_batch(geom_ctx.mask_plasma, bsz)
        bc_mask = _expand_batch(
            np.zeros_like(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx.bc_dir_mask is None else geom_ctx.bc_dir_mask,
            bsz,
        )
        bc_val = _expand_batch(
            np.zeros_like(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx.bc_dir_value is None else geom_ctx.bc_dir_value,
            bsz,
        )
        eps = _expand_batch(geom_ctx.eps, bsz)
        phi_bc_ext = self._phi_bc_ext(geom_ctx, bsz)
        aux: dict[str, np.ndarray] = {"phi_bc_ext": phi_bc_ext}

        if self.mode == "direct":
            phi = phi_in
        elif self.mode == "hard_bc_embed":
            dist0 = _expand_batch(geom_ctx.dist0, bsz)
            phi = phi_bc_ext + dist0 * phi_in
            phi = active * ((1.0 - bc_mask) * phi + bc_mask * bc_val)
        elif self.mode == "poisson_hybrid":
            rho_eff = out.get("rho_eff")
            if rho_eff is None:
                # fallback keeps compatibility with direct-only backbones
                rho_eff = -laplacian2d(phi_in)
            else:
                rho_eff = np.asarray(rho_eff, dtype=np.float32)[:, 0]
            aux["rho_eff"] = rho_eff[:, None]
            phi = _jacobi_poisson(
                rho_eff=rho_eff,
                eps=eps,
                bc_mask=bc_mask,
                bc_value=bc_val,
                phi0=phi_bc_ext,
                mask_active=active,
                n_iters=self.jacobi_iters + max(0, int(refine_iters)),
            )
        elif self.mode == "deeponet_poisson":
            if deeponet_head is None:
                raise NotImplementedError("deeponet_poisson head is not implemented in Cycle 1.4 without deeponet_head")
            if cond_vec is None:
                raise ValueError("deeponet_poisson requires cond_vec for deeponet_head prediction")
            cond_arr = np.asarray(cond_vec, dtype=np.float32)
            if cond_arr.ndim == 1:
                cond_arr = cond_arr[None, :]

            phi0: np.ndarray
            if hasattr(deeponet_head, "predict_phi"):
                rho_eff_in = out.get("rho_eff")
                if rho_eff_in is None:
                    rho_eff_in = -laplacian2d(phi_in)[:, None]
                try:
                    phi_pred = deeponet_head.predict_phi(
                        rho_eff=rho_eff_in,
                        cond_vec=cond_arr,
                        geom_ctx=geom_ctx,
                        refine_iters=max(0, int(refine_iters)),
                    )
                except TypeError:
                    phi_pred = deeponet_head.predict_phi(rho_eff_in, cond_arr, geom_ctx)
                phi0 = _to_numpy_f32(phi_pred)
                if phi0.ndim == 4:
                    phi0 = phi0[:, 0]
                elif phi0.ndim == 3:
                    pass
                else:
                    raise ValueError(f"deeponet_head.predict_phi returned invalid shape: {phi0.shape}")
            else:
                if not hasattr(deeponet_head, "predict_fields"):
                    raise TypeError("deeponet_head must implement predict_phi(...) or predict_fields(cond)")
                try:
                    pred = deeponet_head.predict_fields(cond_arr, grid_shape=geom_ctx.mask_plasma.shape)
                except TypeError:
                    pred = deeponet_head.predict_fields(cond_arr)
                if not isinstance(pred, dict) or "phi" not in pred:
                    raise ValueError("deeponet_head.predict_fields must return dict containing 'phi'")
                phi0 = np.asarray(pred["phi"], dtype=np.float32)
                if phi0.ndim == 4:
                    phi0 = phi0[:, 0]
                elif phi0.ndim == 3:
                    pass
                elif phi0.ndim == 2:
                    phi0 = phi0[None, ...]
                else:
                    raise ValueError(f"deeponet_head phi has invalid shape: {phi0.shape}")

            if phi0.shape[0] == 1 and bsz > 1:
                phi0 = np.repeat(phi0, bsz, axis=0)
            if phi0.shape != phi_in.shape:
                raise ValueError(f"deeponet_head phi shape mismatch: expected {phi_in.shape}, got {phi0.shape}")

            rho_eff = out.get("rho_eff")
            if rho_eff is None:
                rho_eff = -laplacian2d(phi0)
            else:
                rho_eff = np.asarray(rho_eff, dtype=np.float32)[:, 0]
            aux["rho_eff"] = rho_eff[:, None]
            phi = _jacobi_poisson(
                rho_eff=rho_eff,
                eps=eps,
                bc_mask=bc_mask,
                bc_value=bc_val,
                phi0=phi0,
                mask_active=active,
                n_iters=self.jacobi_iters + max(0, int(refine_iters)),
            )
        else:
            raise ValueError(f"Unsupported phi mode: {self.mode}")

        out["phi"] = phi[:, None].astype(np.float32)
        return out, aux
