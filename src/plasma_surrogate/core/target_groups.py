"""Resolve target groups from output layout and target role metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


FIELD_FAMILY_MODE = "field_family"
DEFAULT_TARGET_GROUP = "default"
STANDARD_TARGET_GROUP_NAMES = (
    "density",
    "temperature",
    "electrostatic",
    "electric_field",
    "flux",
    "source",
    DEFAULT_TARGET_GROUP,
)


@dataclass(frozen=True)
class TargetGroup:
    name: str
    targets: tuple[str, ...]
    source: str
    field_family: str | None = None
    roles: tuple[str, ...] = ()


def _clean_optional_string(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _normalize_output_vars(output_vars: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, raw in enumerate(output_vars):
        target = _clean_optional_string(raw)
        if target is None:
            raise ValueError(f"output_vars[{index}] must be a non-empty target id")
        if target in seen:
            duplicates.append(target)
            continue
        seen.add(target)
        out.append(target)
    if duplicates:
        raise ValueError(f"output_vars contains duplicated targets: {duplicates}")
    return tuple(out)


def _normalize_schema_targets(
    target_role_schema: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, dict[str, Any]]:
    if not isinstance(target_role_schema, Mapping):
        raise ValueError("target_role_schema must be a mapping")

    raw_targets = target_role_schema.get("targets", [])
    if raw_targets is None:
        raw_targets = []
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ValueError("target_role_schema.targets must be a list of target metadata objects")

    targets: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            if strict:
                raise ValueError(f"target_role_schema.targets[{index}] must be an object")
            continue
        target_id = _clean_optional_string(raw.get("id"))
        if target_id is None:
            if strict:
                raise ValueError(f"target_role_schema.targets[{index}].id is required")
            continue
        if target_id in targets:
            raise ValueError(f"target_role_schema.targets has duplicated id: {target_id}")
        entry = dict(raw)
        entry["id"] = target_id
        targets[target_id] = entry
    return targets


def _normalize_custom_group_targets(group_name: str, raw_targets: Any) -> tuple[str, ...]:
    if raw_targets is None:
        return ()
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ValueError(f"custom_groups.{group_name} must be a list of target ids")

    targets: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, raw in enumerate(raw_targets):
        target = _clean_optional_string(raw)
        if target is None:
            raise ValueError(f"custom_groups.{group_name}[{index}] must be a non-empty target id")
        if target in seen:
            duplicates.append(target)
            continue
        seen.add(target)
        targets.append(target)
    if duplicates:
        raise ValueError(f"custom_groups.{group_name} contains duplicated targets: {duplicates}")
    return tuple(targets)


def _unique_roles(targets: tuple[str, ...], target_metadata: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    roles: list[str] = []
    seen: set[str] = set()
    for target in targets:
        role = _clean_optional_string(target_metadata.get(target, {}).get("role"))
        if role and role not in seen:
            seen.add(role)
            roles.append(role)
    return tuple(roles)


def _build_group(
    *,
    name: str,
    targets: tuple[str, ...],
    source: str,
    target_metadata: dict[str, dict[str, Any]],
    field_family: str | None = None,
) -> TargetGroup:
    return TargetGroup(
        name=name,
        targets=targets,
        source=source,
        field_family=field_family,
        roles=_unique_roles(targets, target_metadata),
    )


def resolve_target_groups(
    *,
    output_vars: list[str],
    target_role_schema: dict[str, Any],
    mode: str = FIELD_FAMILY_MODE,
    custom_groups: dict[str, Any] | None = None,
    strict: bool = True,
    custom_groups_missing: str = "field_family",
    allow_empty_custom_groups: bool = True,
) -> dict[str, TargetGroup]:
    """Resolve output target groups while preserving output layout order."""

    if mode != FIELD_FAMILY_MODE:
        raise ValueError(f"unsupported target group mode: {mode!r}")
    missing_policy = str(custom_groups_missing).strip().lower()
    if missing_policy not in {"field_family", "default", "error"}:
        raise ValueError("custom_groups_missing must be one of: field_family, default, error")

    ordered_targets = _normalize_output_vars(output_vars)
    output_set = set(ordered_targets)
    target_metadata = _normalize_schema_targets(target_role_schema, strict=strict)

    if strict:
        schema_only = sorted(set(target_metadata) - output_set)
        if schema_only:
            raise ValueError(f"target_role_schema.targets contains targets outside output_vars: {schema_only}")
        missing_metadata = [target for target in ordered_targets if target not in target_metadata]
        if missing_metadata:
            raise ValueError(f"target_role_schema.targets is missing output_vars: {missing_metadata}")

    groups: dict[str, TargetGroup] = {}
    assigned_targets: dict[str, str] = {}

    if custom_groups is not None:
        if not isinstance(custom_groups, Mapping):
            raise ValueError("custom_groups must be a mapping of group name to target id list")
        for raw_group_name, raw_targets in custom_groups.items():
            group_name = _clean_optional_string(raw_group_name)
            if group_name is None:
                raise ValueError("custom_groups contains an empty group name")
            custom_targets = _normalize_custom_group_targets(group_name, raw_targets)
            if not custom_targets and not bool(allow_empty_custom_groups):
                raise ValueError(f"custom_groups.{group_name} must contain at least one target")
            unknown = [target for target in custom_targets if target not in output_set]
            if unknown:
                raise ValueError(f"custom_groups.{group_name} contains unknown targets: {unknown}")
            duplicates = [target for target in custom_targets if target in assigned_targets]
            if duplicates:
                previous = {target: assigned_targets[target] for target in duplicates}
                raise ValueError(
                    f"custom_groups.{group_name} contains targets already assigned to another group: {previous}"
                )
            custom_target_set = set(custom_targets)
            ordered_group_targets = tuple(target for target in ordered_targets if target in custom_target_set)
            if not ordered_group_targets:
                continue
            groups[group_name] = _build_group(
                name=group_name,
                targets=ordered_group_targets,
                source="custom",
                target_metadata=target_metadata,
            )
            for target in ordered_group_targets:
                assigned_targets[target] = group_name

    missing_custom_targets = [target for target in ordered_targets if target not in assigned_targets]
    if custom_groups is not None and missing_policy == "error" and missing_custom_targets:
        raise ValueError(f"custom_groups is missing output_vars: {missing_custom_targets}")

    auto_group_targets: dict[str, list[str]] = {}
    auto_group_families: dict[str, str | None] = {}
    for target in ordered_targets:
        if target in assigned_targets:
            continue
        if custom_groups is not None and missing_policy == "default":
            family = None
            group_name = DEFAULT_TARGET_GROUP
        else:
            family = _clean_optional_string(target_metadata.get(target, {}).get("field_family"))
            group_name = family or DEFAULT_TARGET_GROUP
        if group_name in groups:
            raise ValueError(f"automatic target group {group_name!r} conflicts with a custom group name")
        auto_group_targets.setdefault(group_name, []).append(target)
        auto_group_families.setdefault(group_name, family)

    for group_name, targets in auto_group_targets.items():
        if not targets:
            continue
        groups[group_name] = _build_group(
            name=group_name,
            targets=tuple(targets),
            source=FIELD_FAMILY_MODE,
            field_family=auto_group_families[group_name],
            target_metadata=target_metadata,
        )

    return groups


__all__ = [
    "DEFAULT_TARGET_GROUP",
    "FIELD_FAMILY_MODE",
    "STANDARD_TARGET_GROUP_NAMES",
    "TargetGroup",
    "resolve_target_groups",
]
