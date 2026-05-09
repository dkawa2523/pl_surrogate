"""Model/input-mode support registry and adapter validation helpers."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.input_modes import (
    INPUT_MODE_EFFECTIVE_KEY,
    INPUT_MODES,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.model_specs import (
    ADAPTER_AUTO,
    ADAPTER_COORD_PACK,
    ADAPTER_DESCRIPTOR_BRANCH,
    ADAPTER_GRID_PACK,
    ADAPTER_HYBRID_PACK_DESCRIPTOR,
    ADAPTER_MODES,
    ADAPTER_NONE,
    MODEL_SPECS,
    normalize_model_name,
)


MODEL_SUPPORTED_INPUT_MODES: dict[str, tuple[str, ...]] = {
    name: spec.supported_input_modes for name, spec in MODEL_SPECS.items()
}
MODEL_ALLOWED_ADAPTER_MODES: dict[str, tuple[str, ...]] = {
    name: spec.allowed_adapter_modes for name, spec in MODEL_SPECS.items()
}
MODEL_AUTO_ADAPTER_MODE: dict[str, str] = {name: spec.auto_adapter_mode for name, spec in MODEL_SPECS.items()}
STRUCTURE_PACK_REQUIRED_MODELS: tuple[str, ...] = tuple(
    name for name, spec in MODEL_SPECS.items() if spec.requires_structure_pack
)


def _normalize_model_name(model_name: Any) -> str:
    return normalize_model_name(model_name)


def _normalize_input_mode(input_mode: Any) -> str:
    mode = str(input_mode).strip().lower()
    if mode not in set(INPUT_MODES):
        raise ValueError(f"input_mode must be one of: {list(INPUT_MODES)}; got={input_mode!r}")
    return mode


def _normalize_adapter_mode(adapter_mode: Any) -> str:
    mode = str(adapter_mode if adapter_mode is not None else ADAPTER_AUTO).strip().lower()
    if not mode:
        mode = ADAPTER_AUTO
    if mode not in set(ADAPTER_MODES):
        raise ValueError(f"adapter_mode must be one of: {list(ADAPTER_MODES)}; got={adapter_mode!r}")
    return mode


def resolve_supported_input_modes(model_name: Any) -> tuple[str, ...]:
    """Resolve supported input modes for the given model name."""

    name = _normalize_model_name(model_name)
    modes = MODEL_SUPPORTED_INPUT_MODES.get(name)
    if modes is None:
        raise ValueError(
            f"Unsupported model for input-mode policy: {model_name!r}; "
            f"known={sorted(MODEL_SUPPORTED_INPUT_MODES.keys())}"
        )
    return tuple(modes)


def resolve_allowed_adapter_modes(model_name: Any) -> tuple[str, ...]:
    """Resolve allowed adapter modes for the given model name."""

    name = _normalize_model_name(model_name)
    adapter_modes = MODEL_ALLOWED_ADAPTER_MODES.get(name)
    if adapter_modes is None:
        raise ValueError(
            f"Unsupported model for adapter policy: {model_name!r}; "
            f"known={sorted(MODEL_ALLOWED_ADAPTER_MODES.keys())}"
        )
    return tuple(adapter_modes)


def validate_model_input_mode(model_name: Any, input_mode: Any) -> None:
    """Validate model/input_mode compatibility."""

    mode = _normalize_input_mode(input_mode)
    supported = set(resolve_supported_input_modes(model_name))
    if mode not in supported:
        raise ValueError(
            f"model/input_mode mismatch: model={model_name!r} does not support input_mode={mode!r}; "
            f"supported={sorted(supported)}"
        )


def resolve_effective_adapter_mode(model_name: Any, input_mode: Any, adapter_mode: Any) -> str:
    """Resolve concrete adapter mode and validate model/input_mode/adapter coherence."""

    name = _normalize_model_name(model_name)
    mode = _normalize_input_mode(input_mode)
    validate_model_input_mode(model_name, input_mode)
    adapter_mode_raw = _normalize_adapter_mode(adapter_mode)
    allowed = set(resolve_allowed_adapter_modes(name))
    if adapter_mode_raw == ADAPTER_AUTO:
        resolved = MODEL_AUTO_ADAPTER_MODE.get(name)
        if resolved is None:
            raise ValueError(
                f"adapter_mode=auto is not resolvable for model={model_name!r}; "
                "explicit runtime.structure.adapter_mode is required"
            )
    else:
        resolved = adapter_mode_raw
    if mode == TABLE_ONLY and resolved != ADAPTER_NONE:
        raise ValueError(
            "runtime.input_mode=table_only requires adapter_mode_effective='none'; "
            f"model={name!r}, resolved_adapter_mode={resolved!r}"
        )
    if mode == TABLE_PLUS_STRUCTURE and name in set(STRUCTURE_PACK_REQUIRED_MODELS) and resolved == ADAPTER_NONE:
        raise ValueError(
            "runtime.input_mode=table_plus_structure requires a structure-aware adapter_mode; "
            f"model={name!r}, resolved_adapter_mode={resolved!r}"
        )
    if resolved not in allowed:
        raise ValueError(
            f"model/adapter mismatch: model={model_name!r} does not allow adapter_mode={resolved!r}; "
            f"allowed={sorted(allowed)}"
        )
    return resolved


def validate_model_mode_adapter_policy(model_name: Any, input_mode: Any, adapter_mode: Any) -> str:
    """Validate model/input_mode/adapter policy as a single gate and return effective adapter mode."""

    return resolve_effective_adapter_mode(
        model_name=model_name,
        input_mode=input_mode,
        adapter_mode=adapter_mode,
    )


def validate_adapter_mode(model_name: Any, input_mode: Any, adapter_mode: Any) -> None:
    """Validate adapter mode against model policy and input mode."""

    resolve_effective_adapter_mode(
        model_name=model_name,
        input_mode=input_mode,
        adapter_mode=adapter_mode,
    )


def resolve_effective_input_mode_metadata_for_model(
    *,
    model_name: Any,
    input_mode_meta: dict[str, Any],
) -> dict[str, Any]:
    """Return input-mode metadata with model-specific effective adapter mode resolved."""

    meta = dict(input_mode_meta or {})
    mode = _normalize_input_mode(meta.get(INPUT_MODE_EFFECTIVE_KEY))
    meta[STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY] = resolve_effective_adapter_mode(
        model_name=model_name,
        input_mode=mode,
        adapter_mode=meta.get(STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY),
    )
    return meta


__all__ = [
    "ADAPTER_AUTO",
    "ADAPTER_COORD_PACK",
    "ADAPTER_DESCRIPTOR_BRANCH",
    "ADAPTER_GRID_PACK",
    "ADAPTER_HYBRID_PACK_DESCRIPTOR",
    "ADAPTER_MODES",
    "ADAPTER_NONE",
    "MODEL_AUTO_ADAPTER_MODE",
    "MODEL_ALLOWED_ADAPTER_MODES",
    "MODEL_SUPPORTED_INPUT_MODES",
    "resolve_allowed_adapter_modes",
    "resolve_effective_input_mode_metadata_for_model",
    "resolve_effective_adapter_mode",
    "resolve_supported_input_modes",
    "validate_model_mode_adapter_policy",
    "validate_adapter_mode",
    "validate_model_input_mode",
]
