"""Shared helpers for density-key contracts.

Supports legacy log-density keys and v2 linear-density keys.
"""

from __future__ import annotations

from typing import Any


DENSITY_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "ne": ("ne", "log_ne"),
    "ni": ("ni", "log_ni"),
}

FIELD_KEYS: tuple[str, ...] = ("Te", "phi")


def density_aliases(canonical: str) -> tuple[str, ...]:
    return DENSITY_ALIAS_GROUPS.get(str(canonical), ())


def canonical_density_key(name: str) -> str | None:
    value = str(name)
    for canonical, aliases in DENSITY_ALIAS_GROUPS.items():
        if value in aliases:
            return canonical
    return None


def resolve_density_key(
    available: list[str],
    *,
    canonical: str,
    prefer_linear: bool = True,
) -> str | None:
    aliases = list(density_aliases(canonical))
    if not aliases:
        return None
    present = [k for k in aliases if k in set(available)]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    linear_name = aliases[0]
    legacy_name = aliases[1]
    return linear_name if prefer_linear else legacy_name


def resolve_density_pair(available: list[str], *, prefer_linear: bool = True) -> list[str]:
    out: list[str] = []
    for canonical in ("ne", "ni"):
        resolved = resolve_density_key(available, canonical=canonical, prefer_linear=prefer_linear)
        if resolved is not None:
            out.append(resolved)
    return out


def resolve_present_density_vars(available: list[str]) -> list[str]:
    out: list[str] = []
    present = set(str(v) for v in available)
    for canonical in ("ne", "ni"):
        aliases = density_aliases(canonical)
        for name in aliases:
            if name in present:
                out.append(name)
                break
    return out


def resolve_density_metric_key(row: dict[str, Any], *, canonical: str, suffix: str = "") -> str | None:
    aliases = density_aliases(canonical)
    candidates: list[str] = []
    for name in aliases:
        if suffix:
            candidates.append(f"test_{suffix}_{name}")
        else:
            candidates.append(name)
    for key in candidates:
        if key in row:
            return key
    return None


def resolve_allvars_order(available: list[str], *, prefer_linear: bool = True) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in resolve_density_pair(available, prefer_linear=prefer_linear):
        if name not in seen:
            out.append(name)
            seen.add(name)
    for name in FIELD_KEYS:
        if name in set(available) and name not in seen:
            out.append(name)
            seen.add(name)
    for name in available:
        if name not in seen:
            out.append(str(name))
            seen.add(str(name))
    return out


def resolve_family_vars(
    *,
    family: str,
    available: list[str],
    prefer_linear: bool = True,
) -> list[str]:
    family_norm = str(family).strip().lower()
    if family_norm == "allvars":
        return resolve_allvars_order(available, prefer_linear=prefer_linear)
    if family_norm == "field":
        return [name for name in FIELD_KEYS if name in set(available)]
    if family_norm == "logpair":
        return resolve_density_pair(available, prefer_linear=prefer_linear)
    raise ValueError(f"Unsupported target family: {family}")


def is_density_linear_v2_policy(raw: Any) -> bool:
    return str(raw if raw is not None else "").strip().lower() == "density_linear_v2"
