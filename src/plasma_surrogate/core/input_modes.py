"""Runtime input-mode contract helpers for train/infer/benchmark entrypoints."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

from plasma_surrogate.features.structure_feature_registry import (
    list_descriptor_profiles,
    list_feature_profiles,
    list_latent_profiles,
    normalize_descriptor_profile_name,
    normalize_feature_profile_name,
    normalize_latent_profile_name,
    validate_descriptor_profile_name,
    validate_feature_profile_name,
    validate_latent_profile_name,
)


TABLE_ONLY = "table_only"
TABLE_PLUS_STRUCTURE = "table_plus_structure"
INPUT_MODES: tuple[str, str] = (TABLE_ONLY, TABLE_PLUS_STRUCTURE)
DEFAULT_INPUT_MODE = TABLE_PLUS_STRUCTURE

STRICT_INPUT_MODE_VALUES: tuple[str, ...] = ("error", "warn", "off")
STRUCTURE_FEATURE_PROFILES: tuple[str, ...] = ("none",) + tuple(list_feature_profiles())
STRUCTURE_DESCRIPTOR_PROFILES: tuple[str, ...] = ("none",) + tuple(list_descriptor_profiles())
STRUCTURE_LATENT_PROFILES: tuple[str, ...] = ("none",) + tuple(list_latent_profiles())
STRUCTURE_ADAPTER_MODES: tuple[str, ...] = (
    "auto",
    "none",
    "grid_pack",
    "coord_pack",
    "descriptor_branch",
    "hybrid_pack_descriptor",
)
STRUCTURE_PROVIDER_MODES: tuple[str, ...] = ("fixed", "parametric_parts")

INPUT_MODE_EFFECTIVE_KEY = "input_mode_effective"
STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY = "structure_feature_profile_effective"
STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY = "structure_descriptor_profile_effective"
STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY = "structure_latent_profile_effective"
STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY = "structure_adapter_mode_effective"
GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY = "geometry_provider_mode_effective"
HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY = "has_structure_inputs_effective"
DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY = "deeponet_pod_descriptor_dim_effective"
DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY = "deeponet_pod_descriptor_profile_effective"
DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY = "deeponet_pod_latent_profile_effective"
DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY = "deeponet_pod_latent_hook_effective"
GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY = "geom_deeponet_siren_descriptor_dim_effective"
GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY = "geom_deeponet_siren_descriptor_profile_effective"
INPUT_MODE_EFFECTIVE_METADATA_KEYS: tuple[str, ...] = (
    INPUT_MODE_EFFECTIVE_KEY,
    STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY,
    STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY,
)
DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS: tuple[str, ...] = (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
)
EFFECTIVE_RUNTIME_METADATA_KEYS: tuple[str, ...] = (
    *INPUT_MODE_EFFECTIVE_METADATA_KEYS,
    *DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS,
)
INPUT_MODE_FALLBACK_APPLIED_KEY = "input_mode_fallback_applied"


def _norm_text(value: Any, *, default: str) -> str:
    if value is None:
        return default
    txt = str(value).strip().lower()
    return txt if txt else default


def _norm_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    txt = str(value).strip().lower()
    if txt in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def normalize_strict_input_mode(value: Any, *, default: str = "error") -> str:
    mode = _norm_text(value, default=default)
    if mode not in set(STRICT_INPUT_MODE_VALUES):
        raise ValueError(
            "strict_input_mode must be one of: "
            f"{list(STRICT_INPUT_MODE_VALUES)}; got={mode!r}"
        )
    return mode


def apply_strict_input_mode_policy(
    *,
    strict_input_mode: str,
    message: str,
    category: type[Warning] = RuntimeWarning,
) -> bool:
    """Apply strict-input-mode policy for a contract violation.

    Returns True when execution may continue, False when the caller should stop.
    """

    mode = normalize_strict_input_mode(strict_input_mode)
    if mode == "error":
        raise ValueError(message)
    if mode == "warn":
        warnings.warn(message, category=category, stacklevel=2)
    return True


def normalize_input_mode_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized copy of config with runtime.input_mode defaults."""

    out = copy.deepcopy(dict(cfg or {}))
    runtime = dict(out.get("runtime", {}))
    runtime["input_mode"] = _norm_text(runtime.get("input_mode"), default=DEFAULT_INPUT_MODE)
    runtime["strict_input_mode"] = normalize_strict_input_mode(runtime.get("strict_input_mode"), default="error")
    runtime["allow_mode_fallback"] = _norm_bool(runtime.get("allow_mode_fallback"), default=False)

    structure = dict(runtime.get("structure", {}))
    feature_profile = _norm_text(structure.get("feature_profile"), default="none")
    descriptor_profile = _norm_text(structure.get("descriptor_profile"), default="none")
    latent_profile = _norm_text(structure.get("latent_profile"), default="none")
    if feature_profile != "none":
        feature_profile = normalize_feature_profile_name(feature_profile)
    if descriptor_profile != "none":
        descriptor_profile = normalize_descriptor_profile_name(descriptor_profile)
    if latent_profile != "none":
        latent_profile = normalize_latent_profile_name(latent_profile)
    structure["feature_profile"] = feature_profile
    structure["descriptor_profile"] = descriptor_profile
    structure["latent_profile"] = latent_profile
    structure["adapter_mode"] = _norm_text(structure.get("adapter_mode"), default="auto")
    structure["provider_mode"] = _norm_text(structure.get("provider_mode"), default="fixed")
    runtime["structure"] = structure
    out["runtime"] = runtime
    return out


def validate_input_mode_cfg(cfg: dict[str, Any] | None) -> None:
    """Validate runtime.input_mode foundation rules for phase 0."""

    norm = normalize_input_mode_cfg(cfg)
    runtime = dict(norm.get("runtime", {}))
    mode = str(runtime.get("input_mode", DEFAULT_INPUT_MODE))
    strict_mode = normalize_strict_input_mode(runtime.get("strict_input_mode", "error"), default="error")
    if mode not in set(INPUT_MODES):
        raise ValueError(
            "runtime.input_mode must be one of: "
            f"{list(INPUT_MODES)}; got={mode!r}"
        )
    if not isinstance(runtime.get("allow_mode_fallback"), bool):
        raise ValueError("runtime.allow_mode_fallback must be boolean")

    structure = dict(runtime.get("structure", {}))
    feature_profile = str(structure.get("feature_profile", "none"))
    descriptor_profile = str(structure.get("descriptor_profile", "none"))
    latent_profile = str(structure.get("latent_profile", "none"))
    adapter_mode = str(structure.get("adapter_mode", "auto"))
    provider_mode = str(structure.get("provider_mode", "fixed"))
    if feature_profile != "none":
        validate_feature_profile_name(feature_profile)
    if descriptor_profile != "none":
        validate_descriptor_profile_name(descriptor_profile)
    if latent_profile != "none":
        validate_latent_profile_name(latent_profile)
    if adapter_mode not in set(STRUCTURE_ADAPTER_MODES):
        raise ValueError(
            "runtime.structure.adapter_mode must be one of: "
            f"{list(STRUCTURE_ADAPTER_MODES)}; got={adapter_mode!r}"
        )
    if provider_mode not in set(STRUCTURE_PROVIDER_MODES):
        raise ValueError(
            "runtime.structure.provider_mode must be one of: "
            f"{list(STRUCTURE_PROVIDER_MODES)}; got={provider_mode!r}"
        )
    if mode == TABLE_ONLY:
        non_none_profiles = [
            ("feature_profile", feature_profile),
            ("descriptor_profile", descriptor_profile),
            ("latent_profile", latent_profile),
        ]
        bad = [name for name, value in non_none_profiles if value != "none"]
        if bad:
            raise ValueError(
                "runtime.input_mode=table_only requires structure profiles to be 'none'; "
                f"got non-none keys={bad}"
            )
        if provider_mode != "fixed":
            raise ValueError(
                "runtime.input_mode=table_only requires runtime.structure.provider_mode='fixed'; "
                f"got={provider_mode!r}"
            )
    if mode == TABLE_PLUS_STRUCTURE and feature_profile == "none":
        raise ValueError(
            "runtime.input_mode=table_plus_structure requires runtime.structure.feature_profile; "
            "none/empty is not allowed"
        )


def build_input_mode_effective_metadata(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Build canonical effective metadata payload for runtime input mode."""

    norm = normalize_input_mode_cfg(cfg)
    validate_input_mode_cfg(norm)
    runtime = dict(norm.get("runtime", {}))
    structure = dict(runtime.get("structure", {}))
    mode = str(runtime.get("input_mode", DEFAULT_INPUT_MODE))
    return {
        INPUT_MODE_EFFECTIVE_KEY: mode,
        STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY: str(structure.get("feature_profile", "none")),
        STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(structure.get("descriptor_profile", "none")),
        STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY: str(structure.get("latent_profile", "none")),
        STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY: str(structure.get("adapter_mode", "auto")),
        GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY: str(structure.get("provider_mode", "fixed")),
        HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY: bool(mode == TABLE_PLUS_STRUCTURE),
    }


def resolve_runtime_controls(cfg: dict[str, Any] | None) -> tuple[str, bool]:
    """Resolve effective strict/fallback runtime controls from cfg."""

    norm = normalize_input_mode_cfg(cfg)
    runtime = dict(norm.get("runtime", {}))
    strict_mode = normalize_strict_input_mode(runtime.get("strict_input_mode", "error"), default="error")
    allow_fallback = bool(runtime.get("allow_mode_fallback", False))
    return strict_mode, allow_fallback


def resolve_benchmark_runtime_controls(cfg: dict[str, Any] | None) -> tuple[str, bool]:
    """Resolve runtime controls for benchmark runs under strict-fairness policy."""

    strict_mode, allow_fallback = resolve_runtime_controls(cfg)
    if strict_mode != "error":
        raise ValueError(
            "benchmark runtime requires runtime.strict_input_mode='error'; "
            f"got={strict_mode!r}"
        )
    if allow_fallback:
        raise ValueError(
            "benchmark runtime forbids runtime.allow_mode_fallback=true "
            "(mode fallback is disabled for fair comparison)"
        )
    return strict_mode, allow_fallback


def resolve_input_mode_metadata_contract(
    *,
    request_meta: dict[str, Any] | None,
    checkpoint_meta: dict[str, Any] | None,
    strict_input_mode: str,
    allow_mode_fallback: bool,
    keys: tuple[str, ...] | None = None,
    context: str = "input_mode_metadata",
) -> tuple[dict[str, Any], list[str], bool]:
    """Resolve request/checkpoint metadata contract under strict/fallback controls."""

    strict_mode = normalize_strict_input_mode(strict_input_mode, default="error")
    keyset = tuple(keys or input_mode_metadata_keys())
    req = dict(request_meta or {})
    ckpt = dict(checkpoint_meta or {})
    warnings_out: list[str] = []
    fallback_applied = False

    def _handle_violation(msg: str) -> None:
        nonlocal warnings_out
        if strict_mode == "error":
            raise ValueError(msg)
        if strict_mode == "warn":
            warnings.warn(msg, category=RuntimeWarning, stacklevel=3)
            warnings_out.append(msg)

    if not ckpt:
        return req, warnings_out, fallback_applied

    for key in keyset:
        if key not in ckpt:
            _handle_violation(
                f"{context}: input-mode metadata mismatch: "
                f"checkpoint metadata missing required key {key!r}"
            )
            continue
        if key not in req:
            if allow_mode_fallback:
                req[key] = ckpt[key]
                fallback_applied = True
                msg = f"{context}: request metadata missing key {key!r}; fallback to checkpoint value"
                warnings_out.append(msg)
                if strict_mode == "warn":
                    warnings.warn(msg, category=RuntimeWarning, stacklevel=3)
                continue
            _handle_violation(
                f"{context}: input-mode metadata mismatch: "
                f"request metadata missing required key {key!r}"
            )
            continue
        if req[key] != ckpt[key]:
            if allow_mode_fallback:
                msg = (
                    f"{context}: request/checkpoint mismatch for {key!r}; "
                    f"request={req[key]!r}, checkpoint={ckpt[key]!r}; fallback to checkpoint value"
                )
                req[key] = ckpt[key]
                fallback_applied = True
                warnings_out.append(msg)
                if strict_mode == "warn":
                    warnings.warn(msg, category=RuntimeWarning, stacklevel=3)
                continue
            _handle_violation(
                f"{context}: input-mode metadata mismatch: "
                f"key={key!r}, request={req[key]!r}, checkpoint={ckpt[key]!r}"
            )

    return req, warnings_out, fallback_applied


def input_mode_metadata_keys() -> tuple[str, ...]:
    """Return canonical metadata keys for checkpoints/benchmark/infer summaries."""

    return INPUT_MODE_EFFECTIVE_METADATA_KEYS


def descriptor_latent_metadata_keys() -> tuple[str, ...]:
    """Return canonical descriptor/latent lane metadata keys."""

    return DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS


def merge_effective_runtime_metadata(
    *,
    runtime_meta: dict[str, Any] | None,
    dispatch_meta: dict[str, Any] | None,
    keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Merge runtime/dispatch effective metadata and reject conflicting values."""

    merged = dict(runtime_meta or {})
    source = dict(dispatch_meta or {})
    check_keys = tuple(keys or EFFECTIVE_RUNTIME_METADATA_KEYS)
    for key in check_keys:
        if key not in source:
            continue
        raw_value = source.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            value: Any = raw_value.strip()
            if not value:
                continue
        else:
            value = raw_value
        current = merged.get(key)
        if current is not None and current != value:
            raise ValueError(
                "dispatch/runtime effective metadata mismatch: "
                f"key={key!r}, runtime={current!r}, dispatch={value!r}"
            )
        merged[key] = value
    return merged


def extract_input_mode_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Extract canonical input-mode metadata subset from an arbitrary payload."""

    raw = dict(payload or {})
    return {key: raw[key] for key in INPUT_MODE_EFFECTIVE_METADATA_KEYS if key in raw}


def extract_descriptor_latent_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Extract descriptor/latent metadata subset from an arbitrary payload."""

    raw = dict(payload or {})
    return {key: raw[key] for key in DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS if key in raw}


def load_checkpoint_metadata_with_input_mode(meta_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load checkpoint meta.json and extract canonical input-mode metadata keys."""

    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint metadata under {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, extract_input_mode_metadata(payload)


__all__ = [
    "DEFAULT_INPUT_MODE",
    "DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY",
    "DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY",
    "DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY",
    "DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS",
    "EFFECTIVE_RUNTIME_METADATA_KEYS",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY",
    "HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY",
    "INPUT_MODE_EFFECTIVE_KEY",
    "INPUT_MODE_EFFECTIVE_METADATA_KEYS",
    "INPUT_MODE_FALLBACK_APPLIED_KEY",
    "INPUT_MODES",
    "STRICT_INPUT_MODE_VALUES",
    "STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY",
    "STRUCTURE_ADAPTER_MODES",
    "STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "STRUCTURE_DESCRIPTOR_PROFILES",
    "STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY",
    "STRUCTURE_FEATURE_PROFILES",
    "STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY",
    "STRUCTURE_LATENT_PROFILES",
    "STRUCTURE_PROVIDER_MODES",
    "TABLE_ONLY",
    "TABLE_PLUS_STRUCTURE",
    "build_input_mode_effective_metadata",
    "descriptor_latent_metadata_keys",
    "extract_descriptor_latent_metadata",
    "extract_input_mode_metadata",
    "input_mode_metadata_keys",
    "load_checkpoint_metadata_with_input_mode",
    "merge_effective_runtime_metadata",
    "apply_strict_input_mode_policy",
    "normalize_strict_input_mode",
    "normalize_input_mode_cfg",
    "resolve_input_mode_metadata_contract",
    "resolve_runtime_controls",
    "resolve_benchmark_runtime_controls",
    "validate_input_mode_cfg",
]
