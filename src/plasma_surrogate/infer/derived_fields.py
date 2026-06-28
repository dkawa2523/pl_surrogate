"""Derived field helpers for inference-time diagnostics.

Learned targets stay in ``fields_phys``.  This module computes optional fields
from those physical-space predictions without feeding them back into training.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys


def _as_chw(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"derived field {name!r} must be 2D or 3D; got shape={arr.shape}")
    if arr.shape[-2] < 2 or arr.shape[-1] < 2:
        raise ValueError(f"derived field {name!r} requires at least a 2x2 spatial grid; got shape={arr.shape}")
    return arr


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("derived field config contains an empty string")
    return text


def _finite_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def negative_gradient(source: np.ndarray, *, output_id: str = "electric_field") -> dict[str, np.ndarray]:
    """Return ``-grad(source)`` components over the last two spatial axes."""

    arr = _as_chw(source, name="negative_gradient.source")
    grad_y = np.empty_like(arr, dtype=np.float32)
    grad_x = np.empty_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        gy, gx = np.gradient(arr[idx], edge_order=1)
        grad_y[idx] = -_finite_float32(gy)
        grad_x[idx] = -_finite_float32(gx)
    field_id = _clean_text(output_id)
    return {
        f"{field_id}_x": grad_x.astype(np.float32, copy=False),
        f"{field_id}_y": grad_y.astype(np.float32, copy=False),
    }


def vector_magnitude(sources: Iterable[np.ndarray]) -> np.ndarray:
    """Return Euclidean magnitude for same-shaped vector component fields."""

    arrays = [_as_chw(source, name="vector_magnitude.source") for source in sources]
    if not arrays:
        raise ValueError("vector_magnitude requires at least one source field")
    shape = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != shape:
            raise ValueError(f"vector_magnitude source shape mismatch: expected={shape}, got={arr.shape}")
    total = np.zeros(shape, dtype=np.float32)
    for arr in arrays:
        total = total + arr.astype(np.float32, copy=False) ** 2
    return np.sqrt(total).astype(np.float32, copy=False)


def compute_electric_field_magnitude(potential: np.ndarray) -> np.ndarray:
    """Compute the existing ``E_mag`` equivalent from a scalar potential."""

    components = negative_gradient(potential, output_id="electric_field")
    return vector_magnitude([components["electric_field_x"], components["electric_field_y"]])


def _resolve_potential_source(
    fields_phys: dict[str, np.ndarray],
    *,
    symbols: dict[str, Any] | None,
    target_role_schema: dict[str, Any] | None,
    strict: bool,
) -> str | None:
    try:
        resolved = resolve_physics_symbol_keys(
            [str(key) for key in fields_phys.keys()],
            symbols=dict(symbols or {}),
            target_role_schema=dict(target_role_schema or {}),
            required=("potential",),
            context="inference derived fields",
        )
    except ValueError:
        if strict:
            raise
        return None
    return resolved["potential"]


def compute_default_derived_fields(
    fields_phys: dict[str, np.ndarray],
    *,
    symbols: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the backwards-compatible default derived fields.

    The default path only emits ``E_mag`` and only when a unique potential field
    can be resolved from explicit physics symbols or target-role metadata.
    """

    source_key = _resolve_potential_source(
        fields_phys,
        symbols=symbols,
        target_role_schema=target_role_schema,
        strict=strict,
    )
    if source_key is None:
        return {}
    return {"E_mag": compute_electric_field_magnitude(fields_phys[source_key])}


def _lookup_field(
    available: dict[str, np.ndarray],
    source: Any,
    *,
    operator: str,
    strict: bool,
) -> np.ndarray | None:
    if source is None:
        raise ValueError(f"derived field operator {operator!r} requires source")
    key = _clean_text(source)
    if key in available:
        return available[key]
    if strict:
        raise ValueError(f"derived field operator {operator!r} source not found: {key!r}")
    return None


def _config_sources(raw: Any, *, operator: str) -> list[str]:
    if raw is None:
        raise ValueError(f"derived field operator {operator!r} requires sources")
    if isinstance(raw, str):
        return [_clean_text(raw)]
    try:
        values = list(raw)
    except TypeError as exc:
        raise ValueError(f"derived field operator {operator!r} sources must be a list") from exc
    return [_clean_text(value) for value in values]


def compute_configured_derived_fields(
    fields_phys: dict[str, np.ndarray],
    configs: Iterable[dict[str, Any]],
    *,
    symbols: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Compute derived fields from a small inference-time registry.

    Supported operators are ``negative_gradient``, ``vector_magnitude``, and
    ``electric_field_magnitude``.  Configured fields are evaluated in order, so
    later entries may reference fields produced by earlier entries.
    """

    available: dict[str, np.ndarray] = {str(key): np.asarray(value, dtype=np.float32) for key, value in fields_phys.items()}
    derived: dict[str, np.ndarray] = {}
    for raw in configs:
        cfg = dict(raw or {})
        operator = _clean_text(cfg.get("operator"))
        if operator == "negative_gradient":
            source = _lookup_field(available, cfg.get("source"), operator=operator, strict=strict)
            if source is None:
                continue
            output_id = _clean_text(cfg.get("id", "electric_field"))
            out = negative_gradient(source, output_id=output_id)
        elif operator == "vector_magnitude":
            source_keys = _config_sources(cfg.get("sources", cfg.get("source")), operator=operator)
            sources: list[np.ndarray] = []
            missing = False
            for key in source_keys:
                source = _lookup_field(available, key, operator=operator, strict=strict)
                if source is None:
                    missing = True
                    break
                sources.append(source)
            if missing:
                continue
            output_id = _clean_text(cfg.get("id", "vector_magnitude"))
            out = {output_id: vector_magnitude(sources)}
        elif operator == "electric_field_magnitude":
            source = None
            if cfg.get("source") is not None:
                source = _lookup_field(available, cfg.get("source"), operator=operator, strict=strict)
                if source is None:
                    continue
            else:
                source_key = _resolve_potential_source(
                    available,
                    symbols=symbols,
                    target_role_schema=target_role_schema,
                    strict=strict,
                )
                if source_key is None:
                    continue
                source = available[source_key]
            output_id = _clean_text(cfg.get("id", "E_mag"))
            out = {output_id: compute_electric_field_magnitude(source)}
        else:
            raise ValueError(f"unsupported derived field operator: {operator!r}")
        derived.update(out)
        available.update(out)
    return derived


__all__ = [
    "compute_configured_derived_fields",
    "compute_default_derived_fields",
    "compute_electric_field_magnitude",
    "negative_gradient",
    "vector_magnitude",
]
