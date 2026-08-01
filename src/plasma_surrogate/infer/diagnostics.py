"""Physics diagnostics for single-case inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.eval.metrics import bc_mae, poisson_residual_norm
from plasma_surrogate.core.physics_numeric import (
    boundary_operator_loss,
    boundary_operator_target,
    poisson_residual,
)


def collect_inference_diagnostics(
    *,
    fields_phys: dict[str, np.ndarray],
    geom_ctx: GeometryContext,
    potential_field: np.ndarray | None,
    band_mask: np.ndarray,
    op_inputs: dict[str, np.ndarray] | None,
    bo_cfg: dict[str, Any],
    include_maps: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    diagnostics: dict[str, Any] = {
        "n_active": int(np.sum(geom_ctx.mask_plasma > 0.5)),
        "boundary_band_n": int(np.sum(band_mask > 0.5)),
        "coord_split_merge_effective": False,
        "physics_diagnostics_available": potential_field is not None,
        "field_count": len(fields_phys),
    }
    for name, value in sorted(fields_phys.items()):
        arr = np.asarray(value, dtype=np.float32)
        finite = np.isfinite(arr)
        diagnostics[f"finite_ratio_{name}"] = float(np.mean(finite)) if arr.size else float("nan")
        if np.any(finite):
            diagnostics[f"min_{name}"] = float(np.min(arr[finite]))
            diagnostics[f"max_{name}"] = float(np.max(arr[finite]))
        else:
            diagnostics[f"min_{name}"] = float("nan")
            diagnostics[f"max_{name}"] = float("nan")

    if potential_field is not None:
        diagnostics["poisson_residual_norm"] = float(poisson_residual_norm(potential_field[0], eps=geom_ctx.eps))
        diagnostics["bc_potential_mae"] = float(
            bc_mae(
                potential_field[0],
                bc_mask=np.zeros_like(geom_ctx.mask_plasma) if geom_ctx.bc_dir_mask is None else geom_ctx.bc_dir_mask,
                bc_value=np.zeros_like(geom_ctx.mask_plasma) if geom_ctx.bc_dir_value is None else geom_ctx.bc_dir_value,
            )
        )
    if op_inputs is not None:
        diagnostics["boundary_operator_proxy_loss"] = float(
            boundary_operator_loss(
                phi=op_inputs["potential"],
                density=op_inputs["density"],
                te=op_inputs["temperature"],
                mask_band=band_mask,
                mode=str(bo_cfg.get("mode", "proxy")),
                target_coeffs=bo_cfg.get("target_coeffs"),
                prior_coeffs=bo_cfg.get("prior_coeffs"),
                operator_handle=bo_cfg.get("operator_handle"),
                target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
            )
        )

    diagnostics_maps: dict[str, np.ndarray] = {}
    if include_maps:
        diagnostics_maps["boundary_band_mask"] = band_mask.astype(np.float32)
    if include_maps and potential_field is not None:
        rhs_map = None
        if "rho_eff" in fields_phys:
            rhs_map = -np.asarray(fields_phys["rho_eff"], dtype=np.float32)
        poisson_map = poisson_residual(potential_field, rhs=rhs_map)[0].astype(np.float32)
        bo_target = np.zeros_like(potential_field[0], dtype=np.float32)
        if op_inputs is not None:
            bo_target = boundary_operator_target(
                density=op_inputs["density"],
                te=op_inputs["temperature"],
                phi=op_inputs["potential"],
                mode=str(bo_cfg.get("mode", "proxy")),
                target_coeffs=bo_cfg.get("target_coeffs"),
                prior_coeffs=bo_cfg.get("prior_coeffs"),
                operator_handle=bo_cfg.get("operator_handle"),
                target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
            )[0].astype(np.float32)
        bo_residual = ((potential_field[0] - bo_target) * band_mask).astype(np.float32)
        diagnostics["poisson_residual_map_l2"] = float(np.sqrt(np.mean(poisson_map**2)))
        diagnostics["boundary_operator_residual_map_l2"] = float(np.sqrt(np.mean(bo_residual**2)))
        diagnostics_maps.update(
            {
                "poisson_residual_map": poisson_map,
                "boundary_operator_target": bo_target,
                "boundary_operator_residual_map": bo_residual,
            }
        )
    return diagnostics, diagnostics_maps


__all__ = ["collect_inference_diagnostics"]
