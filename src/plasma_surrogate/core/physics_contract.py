"""Shared physics configuration builder for train and benchmark paths."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext


REMOVED_PHYSICS_KEYS = ("lambda_poisson", "lambda_bc", "lambda_rho")
REGISTERED_PHYSICS_TERMS = ("poisson", "boundary", "boundary_operator", "rho")


def _reject_removed_physics_keys(cfg: dict[str, Any]) -> None:
    removed = [key for key in REMOVED_PHYSICS_KEYS if key in cfg]
    bo_cfg = dict(cfg.get("boundary_operator", {}) or {})
    if "lambda" in bo_cfg:
        removed.append("boundary_operator.lambda")
    if removed:
        raise ValueError(f"removed physics keys: {removed}; use physics.terms[].weight")


def normalize_physics_terms(raw_cfg: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize physics-term weights from physics.terms only."""

    cfg = dict(raw_cfg or {})
    _reject_removed_physics_keys(cfg)
    terms_raw = cfg.get("terms", {})
    if terms_raw is None:
        terms_cfg = {}
    elif isinstance(terms_raw, list):
        terms_cfg: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(terms_raw):
            if not isinstance(item, dict):
                raise ValueError(f"physics.terms[{index}] must be a mapping with name, enabled, and weight")
            if "name" not in item:
                raise ValueError(f"physics.terms[{index}].name is required")
            missing = [key for key in ("enabled", "weight") if key not in item]
            if missing:
                raise ValueError(f"physics.terms[{index}] is missing required keys: {missing}")
            name = str(item.get("name", "")).strip().lower()
            if name not in REGISTERED_PHYSICS_TERMS:
                raise ValueError(f"unsupported physics term {name!r}; registered terms={list(REGISTERED_PHYSICS_TERMS)}")
            if "symbols" in item:
                raise ValueError(
                    f"physics.terms[{index}].symbols is not supported; use top-level physics.symbols "
                    "or target_role_schema role/family metadata"
                )
            terms_cfg[name] = {k: v for k, v in dict(item).items() if k != "name"}
    elif isinstance(terms_raw, dict):
        terms_cfg = {}
        for raw_name, raw_payload in dict(terms_raw or {}).items():
            name = str(raw_name).strip().lower()
            if name not in REGISTERED_PHYSICS_TERMS:
                raise ValueError(f"unsupported physics term {name!r}; registered terms={list(REGISTERED_PHYSICS_TERMS)}")
            payload = dict(raw_payload or {})
            if "symbols" in payload:
                raise ValueError(
                    f"physics.terms.{name}.symbols is not supported; use top-level physics.symbols "
                    "or target_role_schema role/family metadata"
                )
            terms_cfg[name] = payload
    else:
        raise ValueError("physics.terms must be a mapping or a list of term objects")

    def _normalize(name: str) -> dict[str, Any]:
        term_cfg = dict(terms_cfg.get(name, {}))
        weight_raw = term_cfg.get("weight")
        weight = 0.0 if weight_raw is None else float(weight_raw)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"physics.terms.{name}.weight must be finite and >= 0")
        enabled_raw = term_cfg.get("enabled")
        enabled = bool((weight > 0.0) if enabled_raw is None else enabled_raw)
        return {"enabled": enabled, "weight": weight, "source_key": "terms"}

    return {
        "poisson": _normalize("poisson"),
        "boundary": _normalize("boundary"),
        "boundary_operator": _normalize("boundary_operator"),
        "rho": _normalize("rho"),
    }


def _resolved_terms_payload(terms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in REGISTERED_PHYSICS_TERMS:
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


def _terms_from_resolved_payload(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        out[name] = {
            "weight": float(row.get("weight", 0.0)),
            "enabled": bool(row.get("enabled", float(row.get("weight", 0.0)) > 0.0)),
        }
    return out


def build_physics_cfg(
    raw_cfg: dict[str, Any] | None,
    geom_ctx: GeometryContext,
    *,
    default_enabled: bool = False,
    default_primary_qoi_key: str = "Gamma_i",
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
            "weight": float(terms["boundary_operator"]["weight"]),
            "mask_band": mask_band.astype(np.float32),
            "mode": bo_mode,
            "primary_qoi_key": str(bo_cfg.get("primary_qoi_key", default_primary_qoi_key)),
            "sample_idx_source": str(
                bo_cfg.get("sample_idx_source", "deeponet_task:boundary_operator.query_indices")
            ),
            "supervised_targets_npz": bo_cfg.get("supervised_targets_npz"),
            "target_coeffs": bo_cfg.get("target_coeffs", {"density": 0.10, "temperature": 0.05, "bias": 0.0}),
            "prior_coeffs": bo_cfg.get("prior_coeffs"),
            "operator_handle": None,
            "target_clamp": bo_cfg.get("target_clamp"),
        }

    poisson_weight = float(terms["poisson"]["weight"])
    boundary_weight = float(terms["boundary"]["weight"])
    terms["poisson"]["weight"] = float(poisson_weight)
    terms["boundary"]["weight"] = float(boundary_weight)
    terms["boundary_operator"]["weight"] = float(terms["boundary_operator"]["weight"])
    terms["rho"]["weight"] = float(terms["rho"]["weight"])

    return {
        "enabled": True,
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
    """Resolve epoch-dependent physics-term scaling."""

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

    terms_raw = cfg.get("terms")
    if not terms_raw:
        terms_raw = _terms_from_resolved_payload(cfg.get("resolved_terms"))
    terms = normalize_physics_terms({"terms": terms_raw})
    for name, term in terms.items():
        term["weight"] = _scaled(term.get("weight", 0.0))
        term["enabled"] = bool(term.get("enabled", False)) and float(term["weight"]) > 0.0
    bo = dict(cfg.get("boundary_operator", {}))
    if bo:
        bo["weight"] = float(terms["boundary_operator"]["weight"])
        cfg["boundary_operator"] = bo
    cfg["terms"] = terms
    cfg["resolved_terms"] = _resolved_terms_payload(terms)
    cfg["physics_ramp_scale"] = scale
    return cfg
