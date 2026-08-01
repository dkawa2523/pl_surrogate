"""Runtime input-mode contract helpers for train/infer/benchmark entrypoints."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from plasma_surrogate.core.contracts import validate_runtime_metadata_pair
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
STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY = "structure_adapter_mode_effective"
GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY = "geometry_provider_mode_effective"
TARGET_SCHEMA_HASH_KEY = "target_schema_hash"
FEATURE_SCHEMA_HASH_KEY = "feature_schema_hash"
DESCRIPTOR_PROFILE_KEY = "descriptor_profile"
LATENT_PROFILE_KEY = "latent_profile"
RUNTIME_SCHEMA_HASH_PENDING = "pending"

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
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    TARGET_SCHEMA_HASH_KEY,
    FEATURE_SCHEMA_HASH_KEY,
)
OPTIONAL_RUNTIME_METADATA_KEYS: tuple[str, ...] = (
    DESCRIPTOR_PROFILE_KEY,
    LATENT_PROFILE_KEY,
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
    *OPTIONAL_RUNTIME_METADATA_KEYS,
    *DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS,
)

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


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_value(raw: Any) -> str:
    value = str(raw if raw is not None else "").strip().lower()
    return value if value else RUNTIME_SCHEMA_HASH_PENDING


def build_runtime_schema_hashes(schemas: dict[str, Any] | None) -> dict[str, str]:
    """Build product runtime schema hashes from loaded preprocessing schemas."""

    raw = dict(schemas or {})
    target_payload = {
        "output_layout": dict(raw.get("output_layout", {}) or {}),
        "target_role_schema": dict(raw.get("target_role_schema", {}) or {}),
    }
    feature_payload = {
        "channel_map": dict(raw.get("channel_map", {}) or {}),
        "coord_feature_pack_meta": dict(raw.get("coord_feature_pack_meta", {}) or {}),
        "static_spatial_feature_pack_meta": dict(raw.get("static_spatial_feature_pack_meta", {}) or {}),
        "case_structure_feature_pack_meta": dict(raw.get("case_structure_feature_pack_meta", {}) or {}),
    }
    descriptor_meta = dict(raw.get("structure_descriptor_pack_meta", {}) or {})
    if descriptor_meta:
        feature_payload["structure_descriptor_pack_meta"] = descriptor_meta
    latent_meta = dict(raw.get("latent_feature_pack_meta", {}) or {})
    if latent_meta:
        feature_payload["latent_feature_pack_meta"] = latent_meta
    return {
        TARGET_SCHEMA_HASH_KEY: _stable_hash(target_payload),
        FEATURE_SCHEMA_HASH_KEY: _stable_hash(feature_payload),
    }


def attach_runtime_schema_hashes(
    meta: dict[str, Any] | None,
    *,
    schemas: dict[str, Any] | None = None,
    schema_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metadata with preprocessing-derived runtime schema hashes attached."""

    out = dict(meta or {})
    hashes = dict(schema_hashes or {})
    if schemas is not None and not hashes:
        hashes = build_runtime_schema_hashes(schemas)
    for key in (TARGET_SCHEMA_HASH_KEY, FEATURE_SCHEMA_HASH_KEY):
        value = _hash_value(hashes.get(key, out.get(key)))
        out[key] = value
    return out


def normalize_input_mode_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized copy of config with product runtime.input_mode defaults."""

    out = copy.deepcopy(dict(cfg or {}))
    runtime = dict(out.get("runtime", {}))
    runtime["input_mode"] = _norm_text(runtime.get("input_mode"), default=DEFAULT_INPUT_MODE)

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
    """Validate product runtime.input_mode foundation rules."""

    norm = normalize_input_mode_cfg(cfg)
    runtime = dict(norm.get("runtime", {}))
    if "allow_mode_fallback" in runtime:
        raise ValueError("runtime.allow_mode_fallback is removed; runtime contracts are strict")
    mode = str(runtime.get("input_mode", DEFAULT_INPUT_MODE))
    if mode not in set(INPUT_MODES):
        raise ValueError(
            "runtime.input_mode must be one of: "
            f"{list(INPUT_MODES)}; got={mode!r}"
        )

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
        bad = [
            name
            for name, value in (
                ("feature_profile", feature_profile),
                ("descriptor_profile", descriptor_profile),
                ("latent_profile", latent_profile),
            )
            if value != "none"
        ]
        if bad:
            raise ValueError(f"runtime.input_mode=table_only: config=runtime.structure, got={bad}")
        if adapter_mode not in {"auto", "none"}:
            raise ValueError(
                "runtime.input_mode=table_only: "
                f"config=runtime.structure.adapter_mode, expected='none', got={adapter_mode!r}"
            )
        if provider_mode != "fixed":
            raise ValueError(
                "runtime.input_mode=table_only: "
                f"config=runtime.structure.provider_mode, expected='fixed', got={provider_mode!r}"
            )
    if mode == TABLE_PLUS_STRUCTURE and feature_profile == "none":
        raise ValueError(
            "runtime.input_mode=table_plus_structure: "
            "config=runtime.structure.feature_profile, expected=non-none, got='none'"
        )


def build_input_mode_effective_metadata(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Build product runtime metadata from config."""

    norm = normalize_input_mode_cfg(cfg)
    validate_input_mode_cfg(norm)
    runtime = dict(norm.get("runtime", {}))
    structure = dict(runtime.get("structure", {}))
    mode = str(runtime.get("input_mode", DEFAULT_INPUT_MODE))
    adapter_mode = str(structure.get("adapter_mode", "auto"))
    if mode == TABLE_ONLY and adapter_mode == "auto":
        adapter_mode = "none"
    meta: dict[str, Any] = {
        INPUT_MODE_EFFECTIVE_KEY: mode,
        STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY: str(structure.get("feature_profile", "none")),
        STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY: adapter_mode,
        GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY: str(structure.get("provider_mode", "fixed")),
        TARGET_SCHEMA_HASH_KEY: _hash_value(runtime.get(TARGET_SCHEMA_HASH_KEY)),
        FEATURE_SCHEMA_HASH_KEY: _hash_value(runtime.get(FEATURE_SCHEMA_HASH_KEY)),
    }
    descriptor_profile = str(structure.get("descriptor_profile", "none"))
    latent_profile = str(structure.get("latent_profile", "none"))
    if descriptor_profile != "none":
        meta[DESCRIPTOR_PROFILE_KEY] = descriptor_profile
    if latent_profile != "none":
        meta[LATENT_PROFILE_KEY] = latent_profile
    return meta


def input_mode_metadata_keys() -> tuple[str, ...]:
    """Return required product runtime metadata keys."""

    return INPUT_MODE_EFFECTIVE_METADATA_KEYS


def optional_runtime_metadata_keys() -> tuple[str, ...]:
    """Return optional product runtime metadata keys."""

    return OPTIONAL_RUNTIME_METADATA_KEYS


def runtime_metadata_keys() -> tuple[str, ...]:
    """Return required plus optional product runtime metadata keys."""

    return (*INPUT_MODE_EFFECTIVE_METADATA_KEYS, *OPTIONAL_RUNTIME_METADATA_KEYS)


def descriptor_latent_metadata_keys() -> tuple[str, ...]:
    """Return model-internal descriptor/latent lane metadata keys."""

    return DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS


def validate_runtime_metadata_contract(
    *,
    request_meta: dict[str, Any] | None,
    checkpoint_meta: dict[str, Any] | None,
    keys: tuple[str, ...] | None = None,
    context: str = "runtime_metadata",
) -> dict[str, Any]:
    """Validate request/checkpoint runtime metadata and return normalized request metadata."""

    return validate_runtime_metadata_pair(
        request_meta=request_meta,
        checkpoint_meta=checkpoint_meta,
        keys=keys,
        context=context,
    )


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
                f"runtime metadata mismatch: key={key!r}, expected={current!r}, got={value!r}"
            )
        merged[key] = value
    return merged


def extract_input_mode_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Extract product runtime metadata subset from an arbitrary payload."""

    raw = dict(payload or {})
    return {key: raw[key] for key in runtime_metadata_keys() if key in raw}


def extract_descriptor_latent_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Extract optional/model-internal descriptor/latent metadata subset."""

    raw = dict(payload or {})
    keys = (*OPTIONAL_RUNTIME_METADATA_KEYS, *DESCRIPTOR_LATENT_EFFECTIVE_METADATA_KEYS)
    return {key: raw[key] for key in keys if key in raw}


def load_checkpoint_metadata_with_input_mode(meta_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load checkpoint meta.json and extract product runtime metadata keys."""

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
    "DESCRIPTOR_PROFILE_KEY",
    "EFFECTIVE_RUNTIME_METADATA_KEYS",
    "FEATURE_SCHEMA_HASH_KEY",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY",
    "HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY",
    "INPUT_MODE_EFFECTIVE_KEY",
    "INPUT_MODE_EFFECTIVE_METADATA_KEYS",
    "INPUT_MODES",
    "LATENT_PROFILE_KEY",
    "OPTIONAL_RUNTIME_METADATA_KEYS",
    "RUNTIME_SCHEMA_HASH_PENDING",
    "STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY",
    "STRUCTURE_ADAPTER_MODES",
    "STRUCTURE_DESCRIPTOR_PROFILES",
    "STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY",
    "STRUCTURE_FEATURE_PROFILES",
    "STRUCTURE_LATENT_PROFILES",
    "STRUCTURE_PROVIDER_MODES",
    "TABLE_ONLY",
    "TABLE_PLUS_STRUCTURE",
    "TARGET_SCHEMA_HASH_KEY",
    "attach_runtime_schema_hashes",
    "build_input_mode_effective_metadata",
    "build_runtime_schema_hashes",
    "descriptor_latent_metadata_keys",
    "extract_descriptor_latent_metadata",
    "extract_input_mode_metadata",
    "input_mode_metadata_keys",
    "load_checkpoint_metadata_with_input_mode",
    "merge_effective_runtime_metadata",
    "normalize_input_mode_cfg",
    "optional_runtime_metadata_keys",
    "runtime_metadata_keys",
    "validate_input_mode_cfg",
    "validate_runtime_metadata_contract",
]
