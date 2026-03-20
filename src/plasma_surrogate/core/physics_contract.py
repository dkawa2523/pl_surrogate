"""Shared physics configuration builder for train and benchmark paths."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext


def normalize_physics_terms(raw_cfg: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize physics-term weights with backward-compatible key precedence."""

    cfg = dict(raw_cfg or {})
    terms_raw = cfg.get("terms", {})
    if isinstance(terms_raw, list):
        terms_cfg = {
            str(item.get("name")): {k: v for k, v in dict(item).items() if k != "name"}
            for item in terms_raw
            if isinstance(item, dict) and item.get("name") is not None
        }
    else:
        terms_cfg = dict(terms_raw or {})
    bo_cfg = dict(cfg.get("boundary_operator", {}))

    def _normalize(name: str, *, legacy_key: str, legacy_weight: float) -> dict[str, Any]:
        term_cfg = dict(terms_cfg.get(name, {}))
        weight_raw = term_cfg.get("weight")
        if weight_raw is None:
            weight = float(legacy_weight)
            source_key = legacy_key
        else:
            weight = float(weight_raw)
            source_key = "terms"
        enabled_raw = term_cfg.get("enabled")
        enabled = bool((weight > 0.0) if enabled_raw is None else enabled_raw)
        return {"enabled": enabled, "weight": weight, "source_key": source_key}

    return {
        "poisson": _normalize(
            "poisson",
            legacy_key="lambda_poisson",
            legacy_weight=float(cfg.get("lambda_poisson", 0.0)),
        ),
        "boundary": _normalize(
            "boundary",
            legacy_key="lambda_bc",
            legacy_weight=float(cfg.get("lambda_bc", 0.0)),
        ),
        "boundary_operator": _normalize(
            "boundary_operator",
            legacy_key="boundary_operator.lambda",
            legacy_weight=float(bo_cfg.get("lambda", 0.0)),
        ),
        "rho": _normalize(
            "rho",
            legacy_key="lambda_rho",
            legacy_weight=float(cfg.get("lambda_rho", 0.0)),
        ),
    }


def _resolved_terms_payload(terms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ["poisson", "boundary", "boundary_operator", "rho"]:
        row = dict(terms.get(name, {}))
        out.append(
            {
                "name": name,
                "weight": float(row.get("weight", 0.0)),
                "enabled": bool(row.get("enabled", False)),
                "source_key": str(row.get("source_key", "unknown")),
            }
        )
    return out


def build_physics_cfg(
    raw_cfg: dict[str, Any] | None,
    geom_ctx: GeometryContext,
    *,
    default_enabled: bool = False,
    default_primary_qoi_key: str = "Gamma_i",
    default_lambda_poisson: float = 0.05,
    default_lambda_bc: float = 0.02,
) -> dict[str, Any]:
    """Build normalized physics config contract from raw user config."""

    cfg = dict(raw_cfg or {})
    if not bool(cfg.get("enabled", default_enabled)):
        return {"enabled": False, "resolved_terms": []}
    terms = normalize_physics_terms(cfg)

    bc_mask = geom_ctx.bc_dir_mask
    if bc_mask is None or float(np.sum(bc_mask)) <= 0.0:
        bc_mask = (geom_ctx.mask_plasma <= 0.5).astype(np.float32)
    bc_value = geom_ctx.bc_dir_value if geom_ctx.bc_dir_value is not None else np.zeros_like(bc_mask, dtype=np.float32)

    bo_cfg = dict(cfg.get("boundary_operator", {}))
    boundary_operator_cfg: dict[str, Any] = {"enabled": False}
    bo_enabled = bool(bo_cfg.get("enabled", False)) or bool(terms["boundary_operator"]["enabled"])
    if bo_enabled:
        delta_edge = float(bo_cfg.get("delta_edge", 1.5))
        mask_plasma = np.asarray(geom_ctx.mask_plasma, dtype=np.float32)
        distance_any = np.asarray(geom_ctx.distance_any, dtype=np.float32)
        mask_band = ((mask_plasma > 0.5) & (distance_any <= delta_edge)).astype(np.float32)
        if bool(bo_cfg.get("wafer_only", False)):
            wafer = geom_ctx.regions.get("wafer_mask")
            if wafer is not None:
                mask_band = mask_band * (np.asarray(wafer, dtype=np.float32) > 0.5).astype(np.float32)

        bo_mode = str(bo_cfg.get("mode", "proxy"))
        if bo_mode == "external_operator":
            raise ValueError("physics.boundary_operator.mode=external_operator is removed from mainline")
        boundary_operator_cfg = {
            "enabled": True,
            "lambda": float(terms["boundary_operator"]["weight"]),
            "mask_band": mask_band.astype(np.float32),
            "mode": bo_mode,
            "primary_qoi_key": str(bo_cfg.get("primary_qoi_key", default_primary_qoi_key)),
            "sample_idx_source": str(
                bo_cfg.get("sample_idx_source", "deeponet_task:boundary_operator.query_indices")
            ),
            "supervised_targets_npz": bo_cfg.get("supervised_targets_npz"),
            "target_coeffs": bo_cfg.get("target_coeffs", {"log_ne": 0.10, "Te": 0.05, "bias": 0.0}),
            "prior_coeffs": bo_cfg.get("prior_coeffs"),
            "operator_handle": None,
            "external_operator_handle": None,
            "target_clamp": bo_cfg.get("target_clamp"),
        }

    lambda_poisson = terms["poisson"]["weight"]
    if "terms" not in cfg and "lambda_poisson" not in cfg:
        lambda_poisson = float(default_lambda_poisson)
        terms["poisson"]["source_key"] = "default_lambda_poisson"
    terms["poisson"]["weight"] = float(lambda_poisson)
    lambda_bc = terms["boundary"]["weight"]
    if "terms" not in cfg and "lambda_bc" not in cfg:
        lambda_bc = float(default_lambda_bc)
        terms["boundary"]["source_key"] = "default_lambda_bc"
    terms["boundary"]["weight"] = float(lambda_bc)
    terms["boundary_operator"]["weight"] = float(terms["boundary_operator"]["weight"])
    terms["rho"]["weight"] = float(terms["rho"]["weight"])

    return {
        "enabled": True,
        "lambda_poisson": float(lambda_poisson),
        "lambda_bc": float(lambda_bc),
        "rhs": None,
        "bc_mask": np.asarray(bc_mask, dtype=np.float32),
        "bc_value": np.asarray(bc_value, dtype=np.float32),
        "boundary_operator": boundary_operator_cfg,
        "terms": terms,
        "resolved_terms": _resolved_terms_payload(terms),
    }


def resolve_epoch_scaled_physics(
    physics_cfg: dict[str, Any] | None,
    *,
    epoch: int,
    curriculum_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve epoch-dependent lambda scaling from train.curriculum.physics_lambda_ramp."""

    cfg = copy.deepcopy(dict(physics_cfg or {}))
    if not bool(cfg.get("enabled", False)):
        return cfg
    cur = dict(curriculum_cfg or {})
    ramp = dict(cur.get("physics_lambda_ramp", {}))
    if not ramp:
        return cfg

    start = int(ramp.get("start_epoch", 0))
    end = int(ramp.get("end_epoch", start))
    frm = float(ramp.get("scale_from", 1.0))
    to = float(ramp.get("scale_to", 1.0))
    ep = int(epoch)
    if ep <= start:
        scale = frm
    elif ep >= end:
        scale = to
    else:
        r = float(ep - start) / float(max(1, end - start))
        scale = frm + (to - frm) * r
    scale = float(max(scale, 0.0))

    def _scaled(value: float | int | None) -> float:
        return float(scale * float(value if value is not None else 0.0))

    cfg["lambda_poisson"] = _scaled(cfg.get("lambda_poisson", 0.0))
    cfg["lambda_bc"] = _scaled(cfg.get("lambda_bc", 0.0))
    bo = dict(cfg.get("boundary_operator", {}))
    if bo:
        bo["lambda"] = _scaled(bo.get("lambda", 0.0))
        cfg["boundary_operator"] = bo
    terms = normalize_physics_terms(cfg)
    cfg["terms"] = terms
    cfg["resolved_terms"] = _resolved_terms_payload(terms)
    cfg["physics_ramp_scale"] = scale
    return cfg
