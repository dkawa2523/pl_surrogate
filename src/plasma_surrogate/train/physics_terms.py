"""Minimal physics-term registry for numpy/torch loss composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicsTermSpec:
    name: str
    weight: float
    enabled: bool
    source_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TERM_ORDER = ("poisson", "boundary", "boundary_operator", "rho")
_REGISTERED_TERM_NAMES = tuple(_TERM_ORDER)
_REMOVED_PHYSICS_KEYS = ("lambda_poisson", "lambda_bc", "lambda_rho")


def _reject_removed_keys(cfg: dict[str, Any]) -> None:
    removed = [key for key in _REMOVED_PHYSICS_KEYS if key in cfg]
    bo_cfg = dict(cfg.get("boundary_operator", {}) or {})
    if "lambda" in bo_cfg:
        removed.append("boundary_operator.lambda")
    if removed and "resolved_terms" not in cfg:
        raise ValueError(f"removed physics keys: {removed}; use physics.terms[].weight")


def _canonical_term_name(name: str) -> str:
    key = str(name).strip().lower()
    return key


def _canonical_terms(terms_cfg: Any) -> dict[str, dict[str, Any]]:
    if isinstance(terms_cfg, dict):
        out: dict[str, dict[str, Any]] = {}
        for k, v in terms_cfg.items():
            out[_canonical_term_name(str(k))] = dict(v or {})
        return out
    if isinstance(terms_cfg, list):
        out: dict[str, dict[str, Any]] = {}
        for item in terms_cfg:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name is None:
                continue
            payload = dict(item)
            payload.pop("name", None)
            out[_canonical_term_name(str(name))] = payload
        return out
    return {}


def _from_resolved_terms(raw: Any) -> dict[str, PhysicsTermSpec]:
    out: dict[str, PhysicsTermSpec] = {}
    if not isinstance(raw, list):
        return out
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        name = _canonical_term_name(name)
        out[name] = PhysicsTermSpec(
            name=name,
            weight=float(row.get("weight", 0.0)),
            enabled=bool(row.get("enabled", float(row.get("weight", 0.0)) > 0.0)),
            source_key=str(row.get("source_key", "resolved_terms")),
        )
    return out


def _terms_specs(normalized_cfg: dict[str, Any]) -> dict[str, PhysicsTermSpec]:
    cfg = dict(normalized_cfg or {})
    terms_cfg = _canonical_terms(cfg.get("terms", {}))
    out: dict[str, PhysicsTermSpec] = {}
    for name in _TERM_ORDER:
        term_cfg = dict(terms_cfg.get(name, {}))
        weight = float(term_cfg.get("weight", 0.0))
        enabled_raw = term_cfg.get("enabled")
        enabled = bool(weight > 0.0) if enabled_raw is None else bool(enabled_raw)
        out[name] = PhysicsTermSpec(name=name, weight=float(weight), enabled=enabled, source_key="terms")
    return out


def _resolve_terms(normalized_cfg: dict[str, Any]) -> list[PhysicsTermSpec]:
    cfg = dict(normalized_cfg or {})
    _reject_removed_keys(cfg)
    resolved = _from_resolved_terms(cfg.get("resolved_terms"))
    terms = _terms_specs(normalized_cfg)
    merged = {**terms, **resolved}
    return [merged[name] for name in _TERM_ORDER]


def resolve_numpy_terms(normalized_cfg: dict[str, Any]) -> list[PhysicsTermSpec]:
    return _resolve_terms(normalized_cfg)


def resolve_torch_terms(normalized_cfg: dict[str, Any]) -> list[PhysicsTermSpec]:
    return _resolve_terms(normalized_cfg)


__all__ = [
    "PhysicsTermSpec",
    "resolve_numpy_terms",
    "resolve_torch_terms",
    "_REGISTERED_TERM_NAMES",
]
