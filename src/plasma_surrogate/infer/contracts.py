"""Inference-time contract validation helpers."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.input_modes import (
    HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY,
    TABLE_ONLY,
    runtime_metadata_keys,
    validate_runtime_metadata_contract,
)
from plasma_surrogate.core.model_input_policy import validate_model_input_mode
from plasma_surrogate.core.target_groups import resolve_target_groups
from plasma_surrogate.models.heads.role_grouped import (
    OUTPUT_HEAD_MODE_CUSTOM_GROUPS,
    OUTPUT_HEAD_MODE_ROLE_GROUPED,
    custom_groups_from_output_head_config,
    is_grouped_output_head_mode,
    output_head_strict_from_config,
    target_groups_from_metadata,
    target_groups_to_metadata,
)


def normalize_effective_meta_value(*, key: str, value: Any) -> Any:
    if key == HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"invalid boolean value for {key}: {value!r}")
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"empty value for {key} is not allowed")
    return text


def normalize_effective_meta_dict(raw: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in runtime_metadata_keys():
        if raw is None or key not in raw:
            continue
        out[key] = normalize_effective_meta_value(key=key, value=raw[key])
    return out


def validate_inference_runtime_metadata(
    *,
    request_meta: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    context: str = "inference runtime_metadata",
) -> dict[str, Any]:
    resolved_meta = validate_runtime_metadata_contract(
        request_meta=request_meta,
        checkpoint_meta=checkpoint_meta,
        context=context,
    )
    return normalize_effective_meta_dict(resolved_meta)


def resolve_model_type_name(model: Any) -> str:
    if hasattr(model, "to_meta"):
        meta = model.to_meta()
        if not isinstance(meta, dict):
            raise TypeError(f"{type(model).__name__}.to_meta() must return a dict")
        name = str(meta.get("model_type", "")).strip().lower()
        if name:
            return name
    model_type_attr = str(getattr(model, "model_type", "")).strip().lower()
    if model_type_attr:
        return model_type_attr
    class_name = str(type(model).__name__).strip().lower()
    aliases = {
        "globalmlp": "global_mlp",
        "unetbaseline": "unet",
        "unetppbaseline": "unetpp",
        "fnobaseline": "fno",
        "ffnobaseline": "ffno",
        "unobaseline": "u_no",
        "cnobaseline": "cno",
        "cnooperatorunet": "cno_operator_unet",
        "coordmlptorch": "coord_mlp_fourier",
        "poddeeponettorch": "deeponet_pod",
        "geomdeeponetsiren": "geom_deeponet_siren",
    }
    return aliases.get(class_name, class_name)


def validate_model_input_mode_contract(*, model: Any, input_mode: str) -> None:
    if str(input_mode) != TABLE_ONLY:
        return
    model_type = resolve_model_type_name(model)
    try:
        validate_model_input_mode(model_type, TABLE_ONLY)
    except ValueError as exc:
        msg = str(exc)
        if "Unsupported model for input-mode policy" in msg:
            return
        raise ValueError(
            "inference model/input_mode contract violation: "
            f"model_type={model_type!r}, input_mode={TABLE_ONLY!r}; detail={msg}"
        ) from exc


def validate_output_head_metadata_contract(
    *,
    model: Any,
    checkpoint_meta: dict[str, Any] | None,
    target_role_schema: dict[str, Any] | None,
) -> None:
    meta = dict(checkpoint_meta or {})
    if not meta:
        return
    meta_head_cfg = dict(meta.get("output_heads", {}) or {})
    checkpoint_mode = str(
        meta.get("output_heads_mode_effective", meta_head_cfg.get("mode", "shared"))
    ).strip().lower() or "shared"
    model_mode = str(getattr(model, "output_heads_mode", checkpoint_mode)).strip().lower() or "shared"
    if checkpoint_mode != model_mode:
        raise ValueError(
            "inference output head contract mismatch: "
            f"checkpoint mode={checkpoint_mode!r}, model mode={model_mode!r}"
        )
    if not is_grouped_output_head_mode(checkpoint_mode):
        return

    checkpoint_groups = list(meta.get("target_groups") or meta_head_cfg.get("target_groups", []) or [])
    if not checkpoint_groups:
        raise ValueError("inference grouped output head requires checkpoint target_groups metadata")
    checkpoint_group_metadata = target_groups_to_metadata(target_groups_from_metadata(checkpoint_groups))
    model_group_metadata = list(getattr(model, "target_groups_metadata", []) or [])
    if not model_group_metadata and getattr(model, "target_groups", None):
        model_group_metadata = target_groups_to_metadata(getattr(model, "target_groups"))
    if not model_group_metadata:
        raise ValueError("inference grouped output head requires model target_groups metadata")
    if model_group_metadata != checkpoint_group_metadata:
        raise ValueError(
            "inference output head contract mismatch: "
            "checkpoint target_groups differ from model target_groups"
        )

    schema = dict(target_role_schema or {})
    if not isinstance(schema.get("targets"), list) or not schema.get("targets"):
        return
    output_keys = [str(v) for v in list(getattr(model, "output_keys", []))]
    if checkpoint_mode == OUTPUT_HEAD_MODE_ROLE_GROUPED:
        runtime_groups = resolve_target_groups(output_vars=output_keys, target_role_schema=schema)
    elif checkpoint_mode == OUTPUT_HEAD_MODE_CUSTOM_GROUPS:
        strict = output_head_strict_from_config(meta_head_cfg, default=True)
        runtime_groups = resolve_target_groups(
            output_vars=output_keys,
            target_role_schema=schema,
            custom_groups=custom_groups_from_output_head_config(meta_head_cfg, cfg_prefix="checkpoint"),
            strict=strict,
            custom_groups_missing="error" if strict else "default",
            allow_empty_custom_groups=False,
        )
    else:
        return
    runtime_group_metadata = target_groups_to_metadata(runtime_groups)
    if checkpoint_group_metadata != runtime_group_metadata:
        raise ValueError(
            "inference output head contract mismatch: "
            "checkpoint target_groups differ from runtime target_role_schema"
        )


__all__ = [
    "normalize_effective_meta_dict",
    "resolve_model_type_name",
    "validate_inference_runtime_metadata",
    "validate_model_input_mode_contract",
    "validate_output_head_metadata_contract",
]
