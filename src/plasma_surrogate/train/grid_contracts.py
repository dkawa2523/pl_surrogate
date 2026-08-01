"""Grid-family validation and effective contract builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.input_modes import (
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.model_families import (
    COORD_MLP_FAMILY_MODELS,
    GRID_TORCH_MODELS,
    MAINLINE_GEOM_PACK_MODELS,
)
from plasma_surrogate.core.model_input_policy import ADAPTER_AUTO, ADAPTER_HYBRID_PACK_DESCRIPTOR
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.features.structure_feature_registry import FEATURE_PROFILE_CHANNELS
from plasma_surrogate.models.heads.role_grouped import (
    GROUPED_OUTPUT_HEAD_MODES,
    ROLE_GROUPED_OUTPUT_HEAD_MODELS,
    is_grouped_output_head_mode,
)
from plasma_surrogate.train.target_contracts import validate_mainline_selection_contract


def _normalize_profile_name(value: Any, *, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    return text if text else default


def validate_unet_like_mainline_contract(
    *,
    model_name: str,
    model_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    require_shared_output_head: bool,
    input_features_mode: str | None = None,
    input_feature_channels: list[str] | None = None,
) -> None:
    del loss_cfg
    model_key = str(model_name).strip().lower()
    if model_key not in GRID_TORCH_MODELS:
        raise ValueError(f"unsupported unet-like model for mainline validation: {model_name}")
    cfg_prefix = f"train.{model_key}"
    if str(target_family).strip().lower() != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for mainline; got={target_family}")
    unknown = [name for name in target_vars if name not in set(y_vars)]
    if unknown:
        raise ValueError(f"{cfg_prefix}.target_vars contains unknown vars: {unknown}; available={y_vars}")
    if require_shared_output_head:
        head_mode = str(dict(dict(model_cfg or {}).get("output_heads", {})).get("mode", "shared")).strip().lower()
        allowed_head_modes = {"shared", *GROUPED_OUTPUT_HEAD_MODES}
        if head_mode not in allowed_head_modes:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.mode must be shared, role_grouped, or custom_groups for mainline"
            )
        if is_grouped_output_head_mode(head_mode) and model_key not in ROLE_GROUPED_OUTPUT_HEAD_MODELS:
            raise ValueError(
                f"{cfg_prefix}.model_cfg.output_heads.mode={head_mode} is supported only for "
                f"{sorted(ROLE_GROUPED_OUTPUT_HEAD_MODELS)}"
            )
        operator_head_mode_raw = dict(dict(model_cfg or {}).get("unet_operator_v2_cfg", {})).get("head_mode")
        if operator_head_mode_raw is not None:
            operator_head_mode = str(operator_head_mode_raw).strip().lower()
            if operator_head_mode != "shared":
                raise ValueError(f"{cfg_prefix}.model_cfg.unet_operator_v2_cfg.head_mode must be shared for mainline")
    if model_key in MAINLINE_GEOM_PACK_MODELS:
        mode = str(input_features_mode or "").strip().lower()
        if mode != "geom_feature_pack":
            raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack for mainline")
        channels = [str(v) for v in list(input_feature_channels or [])]
        matching_profiles = [
            name
            for name, profile_channels in FEATURE_PROFILE_CHANNELS.items()
            if tuple(channels) == tuple(profile_channels)
        ]
        if not matching_profiles:
            raise ValueError(
                f"{cfg_prefix}.input_features.features must exactly match one registered "
                f"structure feature profile; profiles={list(FEATURE_PROFILE_CHANNELS)}, got={channels}"
            )
    validate_mainline_selection_contract(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )


def validate_coord_mlp_experimental_contract(
    *,
    model_name: str,
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    input_features_cfg: dict[str, Any],
    input_feature_channels: list[str],
) -> None:
    model_key = str(model_name).strip().lower()
    if model_key not in COORD_MLP_FAMILY_MODELS:
        raise ValueError(f"unsupported coord-mlp model: {model_name}")
    cfg_prefix = f"train.{model_key}"
    if str(target_family).strip().lower() != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for coord-mlp experimental")
    expected = list(y_vars)
    if list(target_vars) != expected:
        raise ValueError(
            f"{cfg_prefix}.target_vars must match target_family=allvars output_layout.vars order: "
            f"expected={expected}, got={target_vars}"
        )
    mode = str(dict(input_features_cfg).get("mode", "")).strip().lower()
    if mode != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack")
    if len(set(str(v) for v in input_feature_channels)) != len(list(input_feature_channels)):
        raise ValueError(f"{cfg_prefix}.input_features.features must not contain duplicates")


def validate_geom_deeponet_siren_experimental_contract(
    *,
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    input_features_cfg: dict[str, Any],
    input_feature_channels: list[str],
) -> None:
    cfg_prefix = "train.geom_deeponet_siren"
    if str(target_family).strip().lower() != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars")
    expected = list(y_vars)
    if list(target_vars) != expected:
        raise ValueError(f"{cfg_prefix}.target_vars must match output_layout.vars order: expected={expected}, got={target_vars}")
    mode = str(dict(input_features_cfg).get("mode", "")).strip().lower()
    if mode != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack")
    if len(set(str(v) for v in input_feature_channels)) != len(list(input_feature_channels)):
        raise ValueError(f"{cfg_prefix}.input_features.features must not contain duplicates")


def resolve_geom_deeponet_siren_descriptor_contract(
    *,
    input_mode: str,
    adapter_mode: str,
    descriptor_profile: str,
    descriptor_pack: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = _normalize_profile_name(input_mode, default=TABLE_PLUS_STRUCTURE)
    adapter = _normalize_profile_name(adapter_mode, default=ADAPTER_AUTO)
    desc_profile = _normalize_profile_name(descriptor_profile, default="none")
    if mode != TABLE_PLUS_STRUCTURE:
        raise ValueError("geom_deeponet_siren requires runtime.input_mode=table_plus_structure")
    if adapter != ADAPTER_HYBRID_PACK_DESCRIPTOR:
        raise ValueError(
            "geom_deeponet_siren requires runtime.structure.adapter_mode_effective='hybrid_pack_descriptor'"
        )
    if desc_profile == "none":
        raise ValueError("geom_deeponet_siren requires runtime.structure.descriptor_profile != none")
    payload = dict(descriptor_pack or {})
    descriptor_scope = "static_provider_geometry"
    if "vectors" in payload:
        descriptor_input = np.asarray(payload["vectors"], dtype=np.float32)
        if descriptor_input.ndim != 2 or min(descriptor_input.shape) < 1:
            raise ValueError(
                "structure_descriptor_pack vectors must be a non-empty [N,D] matrix; "
                f"got shape={descriptor_input.shape}"
            )
        if not np.all(np.isfinite(descriptor_input)):
            raise ValueError("structure_descriptor_pack vectors must contain finite values")
        descriptor_names = [
            str(value)
            for value in np.asarray(payload.get("feature_names", [])).reshape(-1).tolist()
        ]
        if descriptor_names and len(descriptor_names) != int(descriptor_input.shape[1]):
            raise ValueError(
                "structure_descriptor_pack feature_names length mismatch: "
                f"names={len(descriptor_names)}, dim={int(descriptor_input.shape[1])}"
            )
        descriptor_case_ids = [
            str(value) for value in np.asarray(payload.get("case_ids", [])).reshape(-1).tolist()
        ]
        if len(descriptor_case_ids) != int(descriptor_input.shape[0]):
            raise ValueError(
                "case-specific structure_descriptor_pack requires one case_id per row: "
                f"case_ids={len(descriptor_case_ids)}, rows={int(descriptor_input.shape[0])}"
            )
        if len(set(descriptor_case_ids)) != len(descriptor_case_ids):
            raise ValueError("case-specific structure_descriptor_pack case_ids must be unique")
        descriptor_scope = "case_specific"
    else:
        descriptor_input, descriptor_names = load_vector_from_pack(
            pack=payload,
            pack_name="structure_descriptor_pack",
        )
    return descriptor_input, {
        GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(desc_profile),
        GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY: int(descriptor_input.shape[-1]),
        "descriptor_feature_names_effective": list(descriptor_names),
        "descriptor_scope_effective": descriptor_scope,
        "adapter_mode_effective": str(adapter),
    }


def build_spectral_contract(
    *,
    prefix: str,
    cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    input_features_mode: str,
    input_feature_channels: list[str],
    target_family: str,
    target_vars: list[str],
    loss_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    del loss_cfg
    spectral_cfg = dict(dict(cfg.get("model_cfg", {})).get("spectral_cfg", {}))
    out = {
        f"{prefix}_backend_effective": "torch",
        f"{prefix}_input_channels_effective": list(input_feature_channels),
        f"{prefix}_input_features_mode_effective": str(input_features_mode),
        f"{prefix}_selection_mode_effective": str(selection_cfg.get("mode", "last")).strip().lower(),
        "selection_weights_effective": dict(selection_cfg.get("weights", {})),
        f"{prefix}_optimizer_effective": {
            "type": str(optimizer_cfg.get("type", "adamw")).strip().lower(),
            "lr": float(optimizer_cfg.get("lr", cfg.get("lr", train_cfg.get("lr", 1e-3)))),
            "schedule": str(optimizer_cfg.get("schedule", "none")).strip().lower(),
            "warmup_epochs": int(optimizer_cfg.get("warmup_epochs", 0)),
        },
        "feature": {
            "input_features_mode": str(input_features_mode),
            "input_feature_channels": list(input_feature_channels),
            "distance_transform_mode": str(
                dict(dict(cfg.get("input_features", {})).get("distance_transform", {})).get("mode", "raw")
            ).strip().lower(),
        },
        "spectral": {
            "n_modes": int(dict(cfg.get("model_cfg", {})).get("n_modes", dict(cfg.get("model_cfg", {})).get("fno_n_modes", 2))),
            "dealias_ratio": float(spectral_cfg.get("dealias_ratio", 1.0)),
            "taper_alpha": float(spectral_cfg.get("taper_alpha", 0.0)),
            "skip_filter": str(spectral_cfg.get("skip_filter", "none")).strip().lower(),
        },
        "target_family_effective": str(target_family),
        "target_vars_effective": list(target_vars),
        "output_heads_mode_effective": (
            str(dict(dict(cfg.get("model_cfg", {})).get("output_heads", {})).get("mode", "shared"))
            .strip()
            .lower()
        ),
    }
    factorized_cfg = dict(spectral_cfg.get("factorized_cfg", {}))
    if prefix == "ffno":
        out["spectral"]["factorized_cfg"] = {
            "enabled": bool(factorized_cfg.get("enabled", True)),
            "mode": str(factorized_cfg.get("mode", "separable_1d")).strip().lower(),
            "share_weights": bool(factorized_cfg.get("share_weights", False)),
        }
        local_skip_cfg = dict(spectral_cfg.get("local_skip_cfg", {}))
        out["spectral"]["local_skip_cfg"] = {
            "enabled": bool(local_skip_cfg.get("enabled", False)),
            "init_scale": float(local_skip_cfg.get("init_scale", 0.0)),
        }
        axis_mix_cfg = dict(spectral_cfg.get("axis_mix_cfg", {}))
        out["spectral"]["axis_mix_cfg"] = {
            "enabled": bool(axis_mix_cfg.get("enabled", False)),
            "init_h": float(axis_mix_cfg.get("init_h", 1.0)),
            "init_w": float(axis_mix_cfg.get("init_w", 1.0)),
        }
    return out


def build_unet_contract_effective(
    *,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    model: Any,
    grid_backend_effective: str,
    grid_feature_channels: list[str],
    grid_selection_cfg: dict[str, Any],
    grid_optimizer_cfg: dict[str, Any],
    grid_input_features_mode: str,
    grid_target_family: str,
    grid_target_vars: list[str],
    grid_loss_cfg: dict[str, Any],
) -> dict[str, Any]:
    del grid_loss_cfg
    model_cfg = dict(cfg.get("model_cfg", {}))
    conv_cfg = dict(model_cfg.get("conv_cfg", {}))
    operator_cfg = dict(model_cfg.get("unet_operator_v2_cfg", {}))
    input_features_cfg = dict(cfg.get("input_features", {}))
    distance_transform_cfg = dict(input_features_cfg.get("distance_transform", {}))
    return {
        "unet_backend_effective": str(grid_backend_effective),
        "unet_input_channels_effective": list(grid_feature_channels),
        "unet_selection_mode_effective": str(grid_selection_cfg.get("mode", "last")).strip().lower(),
        "selection_weights_effective": dict(grid_selection_cfg.get("weights", {})),
        "boundary_bonus_weight_effective": float(grid_selection_cfg.get("boundary_bonus_weight", 0.0)),
        "unet_optimizer_effective": {
            "type": str(grid_optimizer_cfg.get("type", "adamw")).strip().lower()
            if grid_backend_effective == "torch"
            else "none",
            "lr": (
                float(grid_optimizer_cfg.get("lr", cfg.get("lr", train_cfg.get("lr", 1e-3))))
                if grid_backend_effective == "torch"
                else float(cfg.get("lr", train_cfg.get("lr", 1e-3)))
            ),
            "weight_decay": float(grid_optimizer_cfg.get("weight_decay", 0.0)) if grid_backend_effective == "torch" else 0.0,
            "schedule": str(grid_optimizer_cfg.get("schedule", "none")).strip().lower()
            if grid_backend_effective == "torch"
            else "none",
            "warmup_epochs": int(grid_optimizer_cfg.get("warmup_epochs", 0)) if grid_backend_effective == "torch" else 0,
        },
        "unet_output_heads_mode_effective": str(getattr(model, "output_heads_mode", "shared")).strip().lower(),
        "target_groups": list(getattr(model, "target_groups_metadata", []) or []),
        "feature": {
            "input_features_mode": str(grid_input_features_mode),
            "input_feature_channels": list(grid_feature_channels),
            "upsample_mode": str(
                conv_cfg.get(
                    "upsample_mode",
                    operator_cfg.get("upsample", model_cfg.get("upsample_mode", "deconv")),
                )
            ).strip().lower(),
            "distance_transform_mode": str(distance_transform_cfg.get("mode", "raw")).strip().lower(),
        },
        "operator_v2": {
            "enabled": str(getattr(model, "model_type", "")).strip().lower() == "unet_operator_v2",
            "depth": int(operator_cfg.get("depth", getattr(model, "_torch_depth", 0) or 0)),
            "width": int(operator_cfg.get("width", 0)),
            "blocks_per_level": int(operator_cfg.get("blocks_per_level", getattr(model, "_torch_blocks_per_level", 0) or 0)),
            "downsample": str(operator_cfg.get("downsample", getattr(model, "_torch_downsample", ""))).strip().lower(),
            "use_film": bool(operator_cfg.get("use_film", getattr(model, "_torch_use_film", False))),
            "head_mode": str(operator_cfg.get("head_mode", getattr(model, "_torch_head_mode", "shared"))).strip().lower(),
        },
        "merge_role": "single",
        "target_family_effective": str(grid_target_family),
        "target_vars_effective": list(grid_target_vars),
    }


def build_coord_mlp_contract_effective(
    *,
    model_name: str,
    grid_backend_effective: str,
    grid_target_family: str,
    grid_target_vars: list[str],
    grid_input_features_mode: str,
    grid_feature_channels: list[str],
    grid_feature_source: str,
    cfg: dict[str, Any],
    grid_selection_cfg: dict[str, Any],
    coord_model_cfg: dict[str, Any],
    basis_rank_by_var: dict[str, int] | None = None,
) -> dict[str, Any]:
    out = {
        "model_type_effective": str(model_name),
        "backend_effective": str(grid_backend_effective),
        "target_family_effective": str(grid_target_family),
        "target_vars_effective": list(grid_target_vars),
        "input_features_mode": str(grid_input_features_mode),
        "input_feature_channels": list(grid_feature_channels),
        "feature_source_effective": str(grid_feature_source),
        "selection_mode_effective": str(grid_selection_cfg.get("mode", "last")).strip().lower(),
        "selection_weights_effective": dict(grid_selection_cfg.get("weights", {})),
        "embedding": dict(coord_model_cfg.get("embedding", {})),
        "siren": dict(coord_model_cfg.get("siren", {})),
    }
    if str(model_name).strip().lower() == "coord_mlp_pod_residual":
        out["pod_residual"] = {
            "basis": dict(coord_model_cfg.get("basis", {})),
            "basis_rank_by_var": {str(k): int(v) for k, v in dict(basis_rank_by_var or {}).items()},
            "coeff_loss_weight": float(coord_model_cfg.get("coeff_loss_weight", 0.0)),
            "residual_scale_init": float(coord_model_cfg.get("residual_scale_init", 0.0)),
            "point_encoder": dict(coord_model_cfg.get("point_encoder", {})),
        }
    return out


def build_geom_deeponet_siren_contract_effective(
    *,
    model_name: str,
    grid_target_family: str,
    grid_target_vars: list[str],
    grid_input_features_mode: str,
    grid_feature_channels: list[str],
    grid_selection_cfg: dict[str, Any],
    extra_artifacts: dict[str, Any],
    structure_adapter_mode_effective: str,
) -> dict[str, Any]:
    return {
        "model_type_effective": str(model_name),
        "target_family_effective": str(grid_target_family),
        "target_vars_effective": list(grid_target_vars),
        "input_features_mode": str(grid_input_features_mode),
        "input_feature_channels": list(grid_feature_channels),
        "selection_mode_effective": str(grid_selection_cfg.get("mode", "last")).strip().lower(),
        "selection_weights_effective": dict(grid_selection_cfg.get("weights", {})),
        "descriptor_profile_effective": str(
            extra_artifacts.get(GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY, "none")
        ),
        "descriptor_dim_effective": int(
            extra_artifacts.get(GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY, 0)
        ),
        "adapter_mode_effective": str(structure_adapter_mode_effective),
    }


__all__ = [
    "build_coord_mlp_contract_effective",
    "build_geom_deeponet_siren_contract_effective",
    "build_spectral_contract",
    "build_unet_contract_effective",
    "resolve_geom_deeponet_siren_descriptor_contract",
    "validate_coord_mlp_experimental_contract",
    "validate_geom_deeponet_siren_experimental_contract",
    "validate_unet_like_mainline_contract",
]
