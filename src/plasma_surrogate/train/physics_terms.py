"""Minimal physics-term registry for numpy/torch loss composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plasma_surrogate.core.physics_contract import REGISTERED_PHYSICS_TERMS, normalize_physics_terms


@dataclass(frozen=True)
class PhysicsTermSpec:
    name: str
    weight: float
    enabled: bool
    source_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TERM_ORDER = tuple(REGISTERED_PHYSICS_TERMS)


def _to_spec(name: str, payload: dict[str, Any], *, source_key: str | None = None) -> PhysicsTermSpec:
    return PhysicsTermSpec(
        name=str(name),
        weight=float(payload.get("weight", 0.0)),
        enabled=bool(payload.get("enabled", False)),
        source_key=str(source_key or payload.get("source_key", "terms")),
    )


def _from_resolved_terms(raw: Any) -> dict[str, PhysicsTermSpec]:
    if not isinstance(raw, list):
        return {}
    terms_cfg: dict[str, dict[str, Any]] = {}
    source_keys: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip().lower()
        if not name:
            continue
        terms_cfg[name] = dict(row)
        source_keys[name] = str(row.get("source_key", "resolved_terms"))
    if not terms_cfg:
        return {}
    normalized = normalize_physics_terms({"terms": terms_cfg})
    return {name: _to_spec(name, normalized[name], source_key=source_keys[name]) for name in terms_cfg}


def _terms_specs(normalized_cfg: dict[str, Any]) -> dict[str, PhysicsTermSpec]:
    terms = normalize_physics_terms(normalized_cfg)
    return {name: _to_spec(name, dict(terms.get(name, {}))) for name in _TERM_ORDER}


def _resolve_terms(normalized_cfg: dict[str, Any]) -> list[PhysicsTermSpec]:
    cfg = dict(normalized_cfg or {})
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
]
