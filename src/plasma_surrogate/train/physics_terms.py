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
_TERM_ALIASES = {
    "pinn_residual": "poisson",
    "pino_operator": "boundary_operator",
}
_REGISTERED_TERM_NAMES = tuple(sorted(set(_TERM_ORDER) | set(_TERM_ALIASES.keys())))


def _canonical_term_name(name: str) -> str:
    key = str(name).strip().lower()
    return _TERM_ALIASES.get(key, key)


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


def _fallback_specs(normalized_cfg: dict[str, Any]) -> dict[str, PhysicsTermSpec]:
    cfg = dict(normalized_cfg or {})
    terms_cfg = _canonical_terms(cfg.get("terms", {}))
    bo_cfg = dict(cfg.get("boundary_operator", {}))
    legacy = {
        "poisson": ("lambda_poisson", float(cfg.get("lambda_poisson", 0.0))),
        "boundary": ("lambda_bc", float(cfg.get("lambda_bc", 0.0))),
        "boundary_operator": ("boundary_operator.lambda", float(bo_cfg.get("lambda", 0.0))),
        "rho": ("lambda_rho", float(cfg.get("lambda_rho", 0.0))),
    }
    out: dict[str, PhysicsTermSpec] = {}
    for name in _TERM_ORDER:
        term_cfg = dict(terms_cfg.get(name, {}))
        weight_raw = term_cfg.get("weight")
        if weight_raw is None:
            source_key, legacy_weight = legacy[name]
            weight = float(legacy_weight)
        else:
            source_key = "terms"
            weight = float(weight_raw)
        enabled_raw = term_cfg.get("enabled")
        enabled = bool(weight > 0.0) if enabled_raw is None else bool(enabled_raw)
        out[name] = PhysicsTermSpec(name=name, weight=float(weight), enabled=enabled, source_key=source_key)
    return out


def _resolve_terms(normalized_cfg: dict[str, Any]) -> list[PhysicsTermSpec]:
    resolved = _from_resolved_terms(dict(normalized_cfg or {}).get("resolved_terms"))
    fallback = _fallback_specs(normalized_cfg)
    merged = {**fallback, **resolved}
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
