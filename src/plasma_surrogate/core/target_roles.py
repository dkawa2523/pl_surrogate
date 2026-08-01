"""Target role schema and physics-symbol resolution helpers."""

from __future__ import annotations

from typing import Any


TARGET_ROLE_SCHEMA_VERSION = 1

TARGET_METADATA_KEYS = (
    "source_key",
    "value_transform",
    "units",
    "dtype",
    "role",
    "positive",
    "field_family",
    "default_region",
)

PHYSICS_SYMBOL_ROLES: dict[str, tuple[str, ...]] = {
    "density": ("density_electron", "electron_density", "density"),
    "temperature": ("temperature_electron", "electron_temperature", "temperature"),
    "potential": ("potential", "electrostatic_potential", "potential_electrostatic"),
}

PHYSICS_SYMBOL_FAMILIES: dict[str, tuple[str, ...]] = {
    "density": ("density",),
    "temperature": ("temperature",),
    "potential": ("electrostatic", "potential"),
}


def _clean_optional_string(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _clean_optional_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "1", "yes", "on"}:
            return True
        if value in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"target positive metadata must be boolean-compatible, got={raw!r}")


def normalize_target_metadata(
    targets: list[dict[str, Any]] | None,
    *,
    y_vars: list[str],
) -> list[dict[str, Any]]:
    """Return target metadata aligned to output_layout.vars order."""

    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(targets or []):
        if not isinstance(raw, dict):
            raise ValueError(f"target metadata entry {index} must be an object")
        target_id = _clean_optional_string(raw.get("id"))
        if target_id is None:
            raise ValueError(f"target metadata entry {index} requires id")
        if target_id in by_id:
            raise ValueError(f"target metadata has duplicated id: {target_id}")
        entry: dict[str, Any] = {"id": target_id}
        for key in TARGET_METADATA_KEYS:
            if key not in raw:
                continue
            if key == "positive":
                value = _clean_optional_bool(raw.get(key))
            else:
                value = _clean_optional_string(raw.get(key))
            if value is not None:
                entry[key] = value
        by_id[target_id] = entry

    out: list[dict[str, Any]] = []
    for name in y_vars:
        target_id = str(name)
        entry = dict(by_id.get(target_id, {"id": target_id}))
        entry["id"] = target_id
        out.append(entry)
    return out


def build_target_role_schema(
    targets: list[dict[str, Any]] | None,
    *,
    y_vars: list[str],
) -> dict[str, Any]:
    """Build the persisted target role schema artifact."""

    normalized = normalize_target_metadata(targets, y_vars=y_vars)
    role_to_targets: dict[str, list[str]] = {}
    family_to_targets: dict[str, list[str]] = {}
    positive_targets: list[str] = []
    for entry in normalized:
        target_id = str(entry["id"])
        role = _clean_optional_string(entry.get("role"))
        family = _clean_optional_string(entry.get("field_family"))
        if role:
            role_to_targets.setdefault(role, []).append(target_id)
        if family:
            family_to_targets.setdefault(family, []).append(target_id)
        if entry.get("positive") is True:
            positive_targets.append(target_id)
    return {
        "version": TARGET_ROLE_SCHEMA_VERSION,
        "vars": [str(v) for v in y_vars],
        "targets": normalized,
        "role_to_targets": role_to_targets,
        "field_family_to_targets": family_to_targets,
        "positive_targets": positive_targets,
    }


def _schema_targets(schema: dict[str, Any] | None) -> list[dict[str, Any]]:
    targets = dict(schema or {}).get("targets", [])
    if not isinstance(targets, list):
        return []
    return [dict(t) for t in targets if isinstance(t, dict) and _clean_optional_string(t.get("id"))]


def _resolve_configured_symbol(
    *,
    symbol_name: str,
    configured: Any,
    available: set[str],
) -> str | None:
    if configured is None:
        return None
    candidates = configured if isinstance(configured, list) else [configured]
    missing: list[str] = []
    for raw in candidates:
        key = _clean_optional_string(raw)
        if key is None:
            continue
        if key in available:
            return key
        missing.append(key)
    raise ValueError(
        f"physics.symbols.{symbol_name} did not match available target fields; "
        f"configured={missing}; available={sorted(available)}"
    )


def resolve_target_by_role(
    *,
    symbol_name: str,
    available: list[str],
    target_role_schema: dict[str, Any] | None,
) -> str | None:
    """Resolve one physics symbol by role/family only when the match is unique."""

    available_set = {str(k) for k in available}
    roles = set(PHYSICS_SYMBOL_ROLES.get(str(symbol_name), ()))
    families = set(PHYSICS_SYMBOL_FAMILIES.get(str(symbol_name), ()))
    role_matches: list[str] = []
    family_matches: list[str] = []
    for entry in _schema_targets(target_role_schema):
        target_id = str(entry["id"])
        if target_id not in available_set:
            continue
        role = _clean_optional_string(entry.get("role"))
        family = _clean_optional_string(entry.get("field_family"))
        if role and role in roles:
            role_matches.append(target_id)
        elif family and family in families:
            family_matches.append(target_id)
    matches = role_matches if role_matches else family_matches
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(
            f"physics symbol {symbol_name!r} is ambiguous in target_role_schema: "
            f"matches={unique}. Set physics.symbols.{symbol_name} explicitly."
        )
    return None


def resolve_physics_symbol_keys(
    available: list[str],
    *,
    symbols: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    required: tuple[str, ...] = ("density", "temperature", "potential"),
    context: str = "physics",
) -> dict[str, str]:
    """Resolve physics symbol keys using symbols first, then unique role metadata."""

    available_set = {str(k) for k in available}
    symbols_cfg = dict(symbols or {})
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for symbol_name in required:
        configured = _resolve_configured_symbol(
            symbol_name=str(symbol_name),
            configured=symbols_cfg.get(symbol_name),
            available=available_set,
        )
        key = configured
        if key is None:
            key = resolve_target_by_role(
                symbol_name=str(symbol_name),
                available=list(available_set),
                target_role_schema=target_role_schema,
            )
        if key is None:
            missing.append(str(symbol_name))
        else:
            resolved[str(symbol_name)] = key
    if missing:
        raise ValueError(
            f"{context} symbols unresolved: missing={missing}; available={sorted(available_set)}. "
            "Set physics.symbols explicitly or provide unique dataset.targets[].role metadata."
        )
    return resolved


def target_role_schema_from_config_targets(
    targets: list[dict[str, Any]] | None,
    *,
    y_vars: list[str],
) -> dict[str, Any]:
    return build_target_role_schema(targets, y_vars=y_vars)


__all__ = [
    "PHYSICS_SYMBOL_FAMILIES",
    "PHYSICS_SYMBOL_ROLES",
    "TARGET_ROLE_SCHEMA_VERSION",
    "build_target_role_schema",
    "normalize_target_metadata",
    "resolve_physics_symbol_keys",
    "resolve_target_by_role",
    "target_role_schema_from_config_targets",
]
