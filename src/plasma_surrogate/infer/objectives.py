"""Objective registry for inference optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


LEGACY_OBJECTIVE_KEYS = {
    "main",
    "negative_penalty",
    "physics_penalty",
    "boundary_penalty",
    "floor_penalty",
    "reward",
    "numerator_key",
    "denominator_key",
    "denominator_reference",
}


@dataclass
class ObjectiveEvaluation:
    objective_value: float
    search_value: float
    feasible: bool
    violated_constraints: list[str]
    objective_mode: str
    constraint_violation_total: float = 0.0
    term_values: dict[str, float] = field(default_factory=dict)
    term_contributions: dict[str, float] = field(default_factory=dict)
    constraint_values: dict[str, float] = field(default_factory=dict)
    parts: dict[str, float] = field(default_factory=dict)


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _scalar_maps(result: Any) -> tuple[dict[str, float], dict[str, float]]:
    qoi_raw = dict(getattr(result, "qoi", {}) or {})
    diag_raw = dict(getattr(result, "diagnostics", {}) or {})
    qoi = {str(k): float(v) for k, v in qoi_raw.items() if _float_or_none(v) is not None}
    diagnostics = {str(k): float(v) for k, v in diag_raw.items() if _float_or_none(v) is not None}
    return qoi, diagnostics


def _metric_value(result: Any, key: str) -> float:
    name = str(key).strip()
    if not name:
        raise ValueError("objective term key must be non-empty")
    qoi, diagnostics = _scalar_maps(result)
    if name in qoi:
        return float(qoi[name])
    if name in diagnostics:
        return float(diagnostics[name])
    raise ValueError(f"objective metric key not found or non-finite: {name!r}")


def _reject_legacy_keys(cfg: dict[str, Any]) -> None:
    legacy = sorted(k for k in cfg.keys() if k in LEGACY_OBJECTIVE_KEYS)
    if legacy:
        raise ValueError(
            "legacy objective keys were removed from product foundation; "
            f"use objective.mode=weighted_sum and objective.terms[]. got={legacy}"
        )


def objective_mode_from_config(objective_cfg: dict[str, Any] | None = None) -> str:
    """Return the validated product objective mode."""

    cfg = _dict_or_empty(objective_cfg)
    _reject_legacy_keys(cfg)
    mode = str(cfg.get("mode", "weighted_sum")).strip().lower() or "weighted_sum"
    if mode != "weighted_sum":
        raise ValueError("objective.mode must be: weighted_sum")
    return mode


def _term_transform(value: float, transform: str) -> float:
    mode = str(transform or "identity").strip().lower()
    if mode == "identity":
        return float(value)
    if mode == "log1p_abs":
        return float(np.log1p(abs(float(value))))
    raise ValueError("objective.terms[].transform must be one of: identity, log1p_abs")


def _weighted_sum_terms(objective_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    cfg = _dict_or_empty(objective_cfg)
    raw_terms = cfg.get("terms")
    if raw_terms is None:
        raw_terms = [{"key": "uniformity", "direction": "min", "weight": 1.0}]
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("objective.terms must be a non-empty list")
    terms: list[dict[str, Any]] = []
    for raw in raw_terms:
        item = dict(raw or {})
        key = str(item.get("key", "")).strip()
        if not key:
            raise ValueError("objective.terms[].key must be non-empty")
        direction = str(item.get("direction", "min")).strip().lower()
        if direction not in {"min", "max"}:
            raise ValueError("objective.terms[].direction must be one of: min, max")
        weight = float(item.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("objective.terms[].weight must be finite and >= 0")
        scale = float(item.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("objective.terms[].scale must be finite and > 0")
        transform = str(item.get("transform", "identity") or "identity").strip().lower()
        if transform not in {"identity", "log1p_abs"}:
            raise ValueError("objective.terms[].transform must be one of: identity, log1p_abs")
        terms.append(
            {
                "key": key,
                "direction": direction,
                "weight": float(weight),
                "scale": float(scale),
                "transform": transform,
            }
        )
    return terms


def _constraint_scale(bound: float) -> float:
    magnitude = abs(float(bound))
    return magnitude if magnitude > 1.0e-12 else 1.0


def _evaluate_constraints(result: Any, constraints_cfg: Any) -> tuple[bool, list[str], dict[str, float], float]:
    if constraints_cfg is None:
        constraints: list[Any] = []
    elif isinstance(constraints_cfg, list):
        constraints = constraints_cfg
    else:
        raise ValueError("inference.optimize.constraints must be a list")
    violated: list[str] = []
    values: dict[str, float] = {}
    violation_total = 0.0
    for raw in constraints:
        item = dict(raw or {})
        key = str(item.get("key", "")).strip()
        if not key:
            raise ValueError("constraints[].key must be non-empty")
        value = _metric_value(result, key)
        values[key] = float(value)
        has_upper = item.get("upper") is not None
        has_lower = item.get("lower") is not None
        if not (has_upper or has_lower):
            raise ValueError(f"constraint for {key!r} requires upper or lower")
        if has_upper:
            upper = float(item["upper"])
            if not np.isfinite(upper):
                raise ValueError(f"constraint upper must be finite for {key!r}")
            if value > upper:
                violated.append(key)
                violation_total += (value - upper) / _constraint_scale(upper)
        if has_lower:
            lower = float(item["lower"])
            if not np.isfinite(lower):
                raise ValueError(f"constraint lower must be finite for {key!r}")
            if value < lower:
                if key not in violated:
                    violated.append(key)
                violation_total += (lower - value) / _constraint_scale(lower)
    return len(violated) == 0, violated, values, float(violation_total)


def evaluate_objective(
    result: Any,
    objective_cfg: dict[str, Any] | None = None,
    constraints_cfg: Any = None,
) -> ObjectiveEvaluation:
    """Evaluate a lower-better optimization objective from inference outputs."""

    cfg = _dict_or_empty(objective_cfg)
    _reject_legacy_keys(cfg)
    mode = str(cfg.get("mode", "weighted_sum")).strip().lower()
    if mode != "weighted_sum":
        raise ValueError("objective.mode must be: weighted_sum")

    term_values: dict[str, float] = {}
    term_contributions: dict[str, float] = {}
    objective_value = 0.0
    terms = _weighted_sum_terms(cfg)
    for term in terms:
        key = str(term["key"])
        raw_value = _metric_value(result, key)
        transformed = _term_transform(raw_value, str(term["transform"]))
        normalized = transformed / float(term["scale"])
        sign = 1.0 if str(term["direction"]) == "min" else -1.0
        contribution = sign * float(term["weight"]) * normalized
        term_values[key] = float(normalized)
        term_contributions[key] = float(contribution)
        objective_value += contribution

    feasible, violated, constraint_values, violation_total = _evaluate_constraints(result, constraints_cfg)
    search_value = float(objective_value) + 1.0e6 * float(violation_total)
    parts = {
        "objective_total": float(objective_value),
    }
    for key, contribution in term_contributions.items():
        parts[f"objective_term_{key}"] = float(contribution)
    return ObjectiveEvaluation(
        objective_value=float(objective_value),
        search_value=float(search_value),
        feasible=feasible,
        violated_constraints=violated,
        objective_mode="weighted_sum",
        constraint_violation_total=float(violation_total),
        term_values=term_values,
        term_contributions=term_contributions,
        constraint_values=constraint_values,
        parts=parts,
    )


__all__ = ["ObjectiveEvaluation", "evaluate_objective", "objective_mode_from_config"]
