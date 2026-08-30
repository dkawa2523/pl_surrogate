"""Role/group-aware lightweight output heads for grid field models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups


OUTPUT_HEAD_MODE_SHARED = "shared"
OUTPUT_HEAD_MODE_ROLE_GROUPED = "role_grouped"
OUTPUT_HEAD_MODE_CUSTOM_GROUPS = "custom_groups"
OUTPUT_HEAD_MODE_CAUSAL_EM = "causal_em"
GROUPED_OUTPUT_HEAD_MODES = frozenset({OUTPUT_HEAD_MODE_ROLE_GROUPED, OUTPUT_HEAD_MODE_CUSTOM_GROUPS})
OUTPUT_GROUP_HEAD_DEFAULT = "default"
OUTPUT_GROUP_HEAD_SPATIAL_REFINE = "spatial_refine"
OUTPUT_GROUP_HEAD_POISSON_HYBRID = "poisson_hybrid"
OUTPUT_GROUP_HEAD_TYPES = frozenset({OUTPUT_GROUP_HEAD_DEFAULT, OUTPUT_GROUP_HEAD_SPATIAL_REFINE})
ROLE_GROUPED_OUTPUT_HEAD_MODELS = frozenset(
    {
        "fno",
        "ffno",
        "unet",
        "unetpp",
        "unetpp_attn",
        "u_no",
        "cno",
    }
)


def _clean_string(raw: Any) -> str:
    return str(raw if raw is not None else "").strip()


def _bool_from_config(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"output_heads.strict must be a boolean, got={raw!r}")


def is_grouped_output_head_mode(mode: Any) -> bool:
    return str(mode if mode is not None else "").strip().lower() in GROUPED_OUTPUT_HEAD_MODES


def output_head_strict_from_config(output_heads: dict[str, Any] | None, *, default: bool = True) -> bool:
    return _bool_from_config(dict(output_heads or {}).get("strict"), default=bool(default))


def target_groups_to_metadata(groups: Mapping[str, TargetGroup]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups.values():
        out.append(
            {
                "name": str(group.name),
                "targets": [str(v) for v in group.targets],
                "source": str(group.source),
                "field_family": None if group.field_family is None else str(group.field_family),
                "roles": [str(v) for v in group.roles],
            }
        )
    return out


def target_groups_from_metadata(raw_groups: Any) -> dict[str, TargetGroup]:
    if raw_groups is None:
        return {}
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise ValueError("output_heads.target_groups must be a list of target group metadata")
    groups: dict[str, TargetGroup] = {}
    for index, raw in enumerate(raw_groups):
        if not isinstance(raw, Mapping):
            raise ValueError(f"output_heads.target_groups[{index}] must be an object")
        name = _clean_string(raw.get("name"))
        if not name:
            raise ValueError(f"output_heads.target_groups[{index}].name is required")
        if name in groups:
            raise ValueError(f"output_heads.target_groups has duplicated group name: {name}")
        targets = tuple(_clean_string(v) for v in list(raw.get("targets", []) or []))
        if not targets or any(not v for v in targets):
            raise ValueError(f"output_heads.target_groups[{index}].targets must contain target ids")
        roles = tuple(_clean_string(v) for v in list(raw.get("roles", []) or []) if _clean_string(v))
        field_family_raw = raw.get("field_family")
        groups[name] = TargetGroup(
            name=name,
            targets=targets,
            source=_clean_string(raw.get("source")) or OUTPUT_HEAD_MODE_ROLE_GROUPED,
            field_family=None if field_family_raw is None else (_clean_string(field_family_raw) or None),
            roles=roles,
        )
    return groups


def validate_output_head_group_options(
    output_heads: dict[str, Any] | None,
    *,
    target_groups: Mapping[str, TargetGroup],
    cfg_prefix: str,
) -> dict[str, dict[str, str]]:
    head_cfg = dict(output_heads or {})
    raw_options = head_cfg.get("group_options")
    if raw_options is None:
        return {}
    if not isinstance(raw_options, Mapping):
        raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.group_options must be a mapping")
    normalized: dict[str, dict[str, str]] = {}
    for raw_group_name, raw_spec in raw_options.items():
        group_name = _clean_string(raw_group_name)
        if not group_name:
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.group_options contains an empty group name")
        if group_name not in target_groups:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.group_options.{group_name} does not match a resolved target group"
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.group_options.{group_name} must be a mapping")
        unsupported = sorted(set(str(key) for key in raw_spec.keys()) - {"head"})
        if unsupported:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.group_options.{group_name} has unsupported keys: {unsupported}"
            )
        head = _clean_string(raw_spec.get("head", OUTPUT_GROUP_HEAD_DEFAULT)).lower() or OUTPUT_GROUP_HEAD_DEFAULT
        if head == OUTPUT_GROUP_HEAD_POISSON_HYBRID:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.group_options.{group_name}.head={head!r} "
                "is a planned optional experiment lane but is not implemented yet; use head='default'"
            )
        if head not in OUTPUT_GROUP_HEAD_TYPES:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.group_options.{group_name}.head must be "
                f"one of {sorted(OUTPUT_GROUP_HEAD_TYPES)}; "
                f"{OUTPUT_GROUP_HEAD_POISSON_HYBRID!r} is planned but unavailable"
            )
        normalized[group_name] = {"head": head}
    return normalized


def custom_groups_from_output_head_config(
    output_heads: dict[str, Any] | None,
    *,
    cfg_prefix: str,
) -> dict[str, list[str]]:
    head_cfg = dict(output_heads or {})
    if "groups" in head_cfg:
        raw_groups = head_cfg.get("groups")
        groups_key = "groups"
        nested_targets = True
    else:
        raw_groups = head_cfg.get("custom_groups")
        groups_key = "custom_groups"
        nested_targets = False
    if not isinstance(raw_groups, Mapping):
        raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.{groups_key} must be a mapping")

    groups: dict[str, list[str]] = {}
    for raw_name, raw_spec in raw_groups.items():
        name = _clean_string(raw_name)
        if not name:
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.{groups_key} contains an empty group name")
        if name in groups:
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.{groups_key} has duplicated group name: {name}")
        if nested_targets:
            if not isinstance(raw_spec, Mapping):
                raise ValueError(
                    f"{cfg_prefix}.model_cfg.output_heads.groups.{name} must be an object with a targets list"
                )
            unsupported = sorted(set(str(key) for key in raw_spec.keys()) - {"targets"})
            if unsupported:
                raise ValueError(
                    f"{cfg_prefix}.model_cfg.output_heads.groups.{name} has unsupported keys: {unsupported}"
                )
            raw_targets = raw_spec.get("targets")
        else:
            raw_targets = raw_spec
        if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.{groups_key}.{name}.targets must be a list")
        targets = [_clean_string(target) for target in raw_targets]
        if not targets or any(not target for target in targets):
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.{groups_key}.{name}.targets must contain target ids"
            )
        groups[name] = targets
    if not groups:
        raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.{groups_key} must define at least one group")
    return groups


def resolve_output_head_groups(
    *,
    output_keys: list[str],
    target_role_schema: dict[str, Any] | None,
    output_heads: dict[str, Any] | None,
    cfg_prefix: str,
) -> dict[str, TargetGroup]:
    head_cfg = dict(output_heads or {})
    mode = str(head_cfg.get("mode", OUTPUT_HEAD_MODE_SHARED)).strip().lower()
    if "target_groups" in head_cfg:
        groups = target_groups_from_metadata(head_cfg.get("target_groups"))
    elif mode == OUTPUT_HEAD_MODE_CUSTOM_GROUPS:
        strict = _bool_from_config(head_cfg.get("strict"), default=True)
        groups = resolve_target_groups(
            output_vars=[str(v) for v in output_keys],
            target_role_schema=dict(target_role_schema or {}),
            mode="field_family",
            custom_groups=custom_groups_from_output_head_config(head_cfg, cfg_prefix=cfg_prefix),
            strict=strict,
            custom_groups_missing="error" if strict else "default",
            allow_empty_custom_groups=False,
        )
    else:
        groups = resolve_target_groups(
            output_vars=[str(v) for v in output_keys],
            target_role_schema=dict(target_role_schema or {}),
            mode=str(head_cfg.get("group_mode", "field_family")).strip().lower(),
            custom_groups=dict(head_cfg.get("custom_groups", {}) or {}) or None,
            strict=True,
        )
    ordered = [str(v) for v in output_keys]
    output_set = set(ordered)
    assigned: dict[str, str] = {}
    for group in groups.values():
        unknown = [target for target in group.targets if target not in output_set]
        if unknown:
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads contains unknown targets: {unknown}")
        for target in group.targets:
            if target in assigned:
                raise ValueError(
                    f"{cfg_prefix}.model_cfg.output_heads target {target!r} appears in both "
                    f"{assigned[target]!r} and {group.name!r}"
                )
            assigned[target] = group.name
    missing = [target for target in ordered if target not in assigned]
    if missing:
        raise ValueError(f"{cfg_prefix}.model_cfg.output_heads is missing targets: {missing}")
    return groups


def configure_output_head_metadata(
    model: Any,
    *,
    output_heads: dict[str, Any] | None,
    output_keys: list[str],
    target_role_schema: dict[str, Any] | None,
    cfg_prefix: str,
) -> str:
    head_cfg = dict(output_heads or {})
    mode = str(head_cfg.get("mode", OUTPUT_HEAD_MODE_SHARED)).strip().lower()
    allowed = {OUTPUT_HEAD_MODE_SHARED, OUTPUT_HEAD_MODE_CAUSAL_EM, *GROUPED_OUTPUT_HEAD_MODES}
    if mode not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.mode must be one of: {allowed_text}")
    model.output_heads = head_cfg
    model.output_heads_mode = mode
    if is_grouped_output_head_mode(mode):
        groups = resolve_output_head_groups(
            output_keys=[str(v) for v in output_keys],
            target_role_schema=dict(target_role_schema or {}),
            output_heads=head_cfg,
            cfg_prefix=cfg_prefix,
        )
        group_options = validate_output_head_group_options(
            head_cfg,
            target_groups=groups,
            cfg_prefix=cfg_prefix,
        )
    else:
        if "group_options" in head_cfg:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.group_options requires "
                "output_heads.mode='role_grouped' or 'custom_groups'"
            )
        groups = {}
        group_options = {}
    metadata = target_groups_to_metadata(groups)
    model.target_groups = groups
    model.target_groups_metadata = metadata
    effective: dict[str, Any] = {"mode": mode}
    if mode == OUTPUT_HEAD_MODE_CAUSAL_EM:
        allowed_keys = {
            "mode",
            "driver_targets",
            "response_targets",
            "hidden_channels",
            "detach_driver",
        }
        unsupported = sorted(set(head_cfg) - allowed_keys)
        if unsupported:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads has unsupported causal_em keys: {unsupported}"
            )
        effective.update({key: value for key, value in head_cfg.items() if key != "mode"})
    if mode == OUTPUT_HEAD_MODE_CUSTOM_GROUPS:
        effective["strict"] = _bool_from_config(head_cfg.get("strict"), default=True)
        if "groups" in head_cfg or "custom_groups" in head_cfg:
            effective["groups"] = {
                name: {"targets": list(targets)}
                for name, targets in custom_groups_from_output_head_config(head_cfg, cfg_prefix=cfg_prefix).items()
            }
        else:
            effective["groups"] = {
                group.name: {"targets": [str(target) for target in group.targets]}
                for group in groups.values()
            }
    if group_options:
        effective["group_options"] = dict(group_options)
    model.output_head_group_options = dict(group_options)
    model.output_heads_effective = effective
    return mode


def build_role_grouped_conv2d_head(
    *,
    torch: Any,
    in_channels: int,
    output_keys: list[str],
    target_groups: Mapping[str, TargetGroup],
    group_options: Mapping[str, Mapping[str, str]] | None = None,
    with_rho_eff_head: bool,
):
    nn = torch.nn
    output_order = [str(v) for v in output_keys]
    group_items = list(target_groups.items())
    group_options_by_name = {str(name): dict(value) for name, value in dict(group_options or {}).items()}
    group_indices: list[list[int]] = []
    for _, group in group_items:
        group_indices.append([output_order.index(str(target)) for target in group.targets])

    def _build_group_head(group_name: str, out_channels: int):
        head = str(
            group_options_by_name.get(str(group_name), {}).get("head", OUTPUT_GROUP_HEAD_DEFAULT)
        ).strip().lower() or OUTPUT_GROUP_HEAD_DEFAULT
        if head == OUTPUT_GROUP_HEAD_SPATIAL_REFINE:
            return nn.Sequential(
                nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1),
            )
        return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)

    class _RoleGroupedConv2dHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_heads = nn.ModuleDict(
                {
                    f"group_{idx}": _build_group_head(group_name, len(indices))
                    for idx, ((group_name, _group), indices) in enumerate(zip(group_items, group_indices))
                    if indices
                }
            )
            self.rho_head = nn.Conv2d(int(in_channels), 1, kernel_size=1) if bool(with_rho_eff_head) else None

        def forward(self, feat):
            out_ch = len(output_order) + (1 if self.rho_head is not None else 0)
            out = feat.new_empty((feat.shape[0], out_ch, feat.shape[-2], feat.shape[-1]))
            for idx, indices in enumerate(group_indices):
                if not indices:
                    continue
                group_out = self.group_heads[f"group_{idx}"](feat)
                for src_idx, dst_idx in enumerate(indices):
                    out[:, dst_idx : dst_idx + 1] = group_out[:, src_idx : src_idx + 1]
            if self.rho_head is not None:
                out[:, len(output_order) : len(output_order) + 1] = self.rho_head(feat)
            return out

        def step_reference(self):
            if self.group_heads:
                first_key = next(iter(self.group_heads.keys()))
                head = self.group_heads[first_key]
                if hasattr(head, "weight"):
                    return head.weight
                for module in reversed(list(head.modules())):
                    if module is not head and hasattr(module, "weight"):
                        return module.weight
            if self.rho_head is not None:
                return self.rho_head.weight
            return None

    return _RoleGroupedConv2dHead()


__all__ = [
    "OUTPUT_HEAD_MODE_CAUSAL_EM",
    "OUTPUT_HEAD_MODE_ROLE_GROUPED",
    "OUTPUT_HEAD_MODE_CUSTOM_GROUPS",
    "OUTPUT_HEAD_MODE_SHARED",
    "OUTPUT_GROUP_HEAD_DEFAULT",
    "OUTPUT_GROUP_HEAD_SPATIAL_REFINE",
    "OUTPUT_GROUP_HEAD_POISSON_HYBRID",
    "OUTPUT_GROUP_HEAD_TYPES",
    "GROUPED_OUTPUT_HEAD_MODES",
    "ROLE_GROUPED_OUTPUT_HEAD_MODELS",
    "build_role_grouped_conv2d_head",
    "configure_output_head_metadata",
    "custom_groups_from_output_head_config",
    "is_grouped_output_head_mode",
    "output_head_strict_from_config",
    "resolve_output_head_groups",
    "target_groups_from_metadata",
    "target_groups_to_metadata",
    "validate_output_head_group_options",
]
