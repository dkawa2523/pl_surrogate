"""DeepONet-family training defaults and contract validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.contracts import validate_pod_descriptor_latent_contract
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.model_families import POD_DEEPONET_FAMILY_MODELS
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.features.structure_feature_registry import (
    GEOM_V1_MAINLINE_CHANNELS,
    validate_coord_feature_channels,
)
from plasma_surrogate.models.deeponet.pod_deeponet_torch import normalize_pod_deeponet_model_cfg
from plasma_surrogate.train.selection import SPATIAL_SELECTION_MODE, resolve_spatial_selection_config
from plasma_surrogate.train.target_contracts import validate_mainline_selection_contract


def _normalize_profile_name(value: Any, *, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    return text if text else default


def _load_descriptor_input_from_pack(
    *,
    pack: dict[str, Any] | None,
) -> tuple[np.ndarray, str]:
    payload = dict(pack or {})
    if "vectors" not in payload:
        vector, _ = load_vector_from_pack(
            pack=payload,
            pack_name="structure_descriptor_pack",
        )
        return vector.astype(np.float32), "static_provider_geometry"

    rows = np.asarray(payload["vectors"], dtype=np.float32)
    if rows.ndim != 2 or int(rows.shape[0]) < 1 or int(rows.shape[1]) < 1:
        raise ValueError(
            "structure_descriptor_pack vectors must be a non-empty [N,D] matrix; "
            f"got shape={rows.shape}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("structure_descriptor_pack vectors must contain finite values")
    if "vector" in payload:
        static_vector, _ = load_vector_from_pack(
            pack=payload,
            pack_name="structure_descriptor_pack",
        )
        if int(static_vector.shape[0]) != int(rows.shape[1]):
            raise ValueError(
                "structure_descriptor_pack vector/vectors dimension mismatch: "
                f"vector={static_vector.shape[0]}, vectors={rows.shape[1]}"
            )
    if "feature_names" in payload:
        names = [str(v) for v in np.asarray(payload["feature_names"]).reshape(-1).tolist()]
        if names and len(names) != int(rows.shape[1]):
            raise ValueError(
                "structure_descriptor_pack feature_names length mismatch: "
                f"len(names)={len(names)}, dim={int(rows.shape[1])}"
            )
    if "case_ids" not in payload:
        raise ValueError("case-specific structure_descriptor_pack must include case_ids")
    case_ids = [str(v) for v in np.asarray(payload["case_ids"]).reshape(-1).tolist()]
    if len(case_ids) != int(rows.shape[0]):
        raise ValueError(
            "structure_descriptor_pack case_ids length mismatch: "
            f"len(case_ids)={len(case_ids)}, rows={int(rows.shape[0])}"
        )
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("structure_descriptor_pack case_ids must be unique")
    return rows.astype(np.float32), "case_specific"


def resolve_pod_descriptor_and_latent_contract(
    *,
    input_mode: str,
    adapter_mode: str,
    descriptor_profile: str,
    latent_profile: str,
    descriptor_pack: dict[str, Any] | None,
    latent_pack: dict[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    meta = validate_pod_descriptor_latent_contract(
        input_mode=input_mode,
        adapter_mode=adapter_mode,
        descriptor_profile=descriptor_profile,
        latent_profile=latent_profile,
    )
    desc_profile = str(meta[DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY])
    lat_profile = str(meta[DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY])
    mode = _normalize_profile_name(input_mode, default=TABLE_PLUS_STRUCTURE)

    if mode == TABLE_ONLY:
        return None, meta

    descriptor_input: np.ndarray | None = None
    descriptor_dim = 0
    if desc_profile != "none":
        descriptor_input, descriptor_scope = _load_descriptor_input_from_pack(pack=descriptor_pack)
        descriptor_dim = int(descriptor_input.shape[-1])
    else:
        descriptor_scope = "none"

    latent_hook = False
    if lat_profile != "none":
        load_vector_from_pack(
            pack=latent_pack,
            pack_name="latent_feature_pack",
        )
        latent_hook = True

    return descriptor_input, {
        DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: int(descriptor_dim),
        DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(desc_profile),
        DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: str(lat_profile),
        DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: bool(latent_hook),
        "deeponet_pod_descriptor_scope_effective": str(descriptor_scope),
    }


def validate_pod_deeponet_experimental_contract(
    *,
    model_name: str,
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    model_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    model_type = str(model_name).strip().lower()
    cfg_prefix = f"train.{model_type}"
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for pod_deeponet family")
    expected = list(y_vars)
    if list(target_vars) != expected:
        raise ValueError(f"{cfg_prefix}.target_vars must match output_layout.vars order: expected={expected}, got={target_vars}")
    normalized_cfg = normalize_pod_deeponet_model_cfg(model_cfg, model_type=model_type)
    basis_cfg = dict(normalized_cfg.get("basis", {}))
    rank = int(basis_cfg.get("rank", 32))
    if rank < 1:
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.rank must be >= 1")
    fit_scope = str(basis_cfg.get("fit_scope", "train_only")).strip().lower()
    if fit_scope != "train_only":
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.fit_scope must be train_only")
    per_var = bool(basis_cfg.get("per_var", True))
    if not per_var:
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.per_var must be true")
    center = bool(basis_cfg.get("center", True))
    selection_mode = str(dict(selection_cfg or {}).get("mode", "best_val_allvars_balance")).strip().lower()
    if selection_mode not in {"best_val_allvars_balance", SPATIAL_SELECTION_MODE, "best_val_loss"}:
        raise ValueError(
            f"{cfg_prefix}.selection.mode must be one of: best_val_allvars_balance, "
            "best_val_spatial_objective, best_val_loss"
        )
    if selection_mode == SPATIAL_SELECTION_MODE:
        resolve_spatial_selection_config(selection_cfg)
    return {
        "rank": int(rank),
        "min_rank": int(basis_cfg.get("min_rank", 1)),
        "energy_threshold": basis_cfg.get("energy_threshold"),
        "coeff_std_floor_rel": float(basis_cfg.get("coeff_std_floor_rel", 0.0)),
        "fit_scope": fit_scope,
        "per_var": per_var,
        "center": center,
    }


def resolve_deeponet_pod_training_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg or {})
    optimizer_cfg = dict(out.get("optimizer", {}))
    optimizer_cfg.setdefault("type", "adamw")
    optimizer_cfg.setdefault("schedule", "cosine")
    optimizer_cfg.setdefault("warmup_epochs", 10)
    out["optimizer"] = optimizer_cfg
    selection_cfg = dict(out.get("selection", {}))
    selection_cfg.setdefault("mode", "best_val_allvars_balance")
    out["selection"] = selection_cfg
    return out


def _resolve_coord_mlp_siren_training_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg or {})
    out.setdefault("epochs", 160)
    out.setdefault("lr", 3e-4)
    optimizer_cfg = dict(out.get("optimizer", {}))
    optimizer_cfg.setdefault("schedule", "cosine")
    optimizer_cfg.setdefault("warmup_epochs", 20)
    out["optimizer"] = optimizer_cfg
    selection_cfg = dict(out.get("selection", {}))
    selection_cfg.setdefault("mode", "best_val_allvars_balance")
    out["selection"] = selection_cfg
    return out


def _resolve_coord_mlp_pod_residual_training_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg or {})
    out.setdefault("epochs", 100)
    out.setdefault("lr", 6e-4)
    optimizer_cfg = dict(out.get("optimizer", {}))
    optimizer_cfg.setdefault("type", "adamw")
    optimizer_cfg.setdefault("schedule", "cosine")
    optimizer_cfg.setdefault("warmup_epochs", 10)
    out["optimizer"] = optimizer_cfg
    selection_cfg = dict(out.get("selection", {}))
    selection_cfg.setdefault("mode", "best_val_allvars_balance")
    out["selection"] = selection_cfg
    return out


def resolve_family_train_cfg(train_cfg: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    cfg = dict(train_cfg.get(model_name, {}))
    if model_name in POD_DEEPONET_FAMILY_MODELS:
        cfg = resolve_deeponet_pod_training_defaults(cfg)
    if model_name == "coord_mlp_siren":
        cfg = _resolve_coord_mlp_siren_training_defaults(cfg)
    if model_name == "coord_mlp_pod_residual":
        cfg = _resolve_coord_mlp_pod_residual_training_defaults(cfg)
    return cfg


def validate_deeponet_mainline_contract(
    *,
    deeponet_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    input_features_mode: str,
    input_feature_channels: list[str],
) -> None:
    del loss_cfg
    cfg_prefix = "train.deeponet_plasma"
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for mainline")
    expected = list(y_vars)
    if list(target_vars) != list(expected):
        raise ValueError(f"{cfg_prefix}.target_vars must match allvars order: expected={expected}, got={target_vars}")
    if str(input_features_mode).strip().lower() != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack for mainline")
    required_channels = list(GEOM_V1_MAINLINE_CHANNELS)
    channels = list(validate_coord_feature_channels(input_feature_channels))
    if channels[: len(required_channels)] != required_channels:
        raise ValueError(
            f"{cfg_prefix}.input_features.features must start with the ordered mainline channels "
            f"{required_channels}; additional registered structure channels are allowed; got={channels}"
        )
    input_features_cfg = dict(deeponet_cfg.get("input_features", {}))
    distance_transform_cfg = dict(input_features_cfg.get("distance_transform", {}))
    dt_mode = str(distance_transform_cfg.get("mode", "raw")).strip().lower()
    if dt_mode not in {"raw", "bounded_auto"}:
        raise ValueError(
            f"{cfg_prefix}.input_features.distance_transform.mode must be one of: raw, bounded_auto for mainline"
        )
    operator_mode = str(deeponet_cfg.get("operator_mode", "plain")).strip().lower()
    if operator_mode != "plain":
        raise ValueError(f"{cfg_prefix}.operator_mode must be plain for mainline")
    model_cfg = dict(deeponet_cfg.get("model_cfg", {}))
    trunk_mode = str(model_cfg.get("trunk_input_mode", "geom_feature_pack")).strip().lower()
    if trunk_mode != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_input_mode must be geom_feature_pack for mainline")
    branch_mode = str(model_cfg.get("branch_mode", "moments")).strip().lower()
    if branch_mode not in {"cond_only", "set_mlp_pool"}:
        raise ValueError(
            f"{cfg_prefix}.model_cfg.branch_mode must be one of: cond_only, set_mlp_pool"
        )
    sensor_pool_mode = str(model_cfg.get("sensor_pool_mode", "moments")).strip().lower()
    expected_pool_mode = "moments" if branch_mode == "cond_only" else "set_mlp_pool"
    if sensor_pool_mode != expected_pool_mode:
        raise ValueError(
            f"{cfg_prefix}.model_cfg.sensor_pool_mode must be {expected_pool_mode} "
            f"when branch_mode={branch_mode}"
        )
    output_path_cfg = dict(model_cfg.get("output_path", {}))
    output_path_mode = str(output_path_cfg.get("mode", "dot")).strip().lower()
    if output_path_mode not in {"dot", "fused"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.output_path.mode must be one of: dot, fused")
    output_path_dot_skip = float(output_path_cfg.get("dot_skip", 0.25))
    if not np.isfinite(output_path_dot_skip):
        raise ValueError(f"{cfg_prefix}.model_cfg.output_path.dot_skip must be finite")
    output_path_dot_skip_mode = str(output_path_cfg.get("dot_skip_mode", "fixed")).strip().lower()
    if output_path_dot_skip_mode not in {"fixed", "learned_per_var"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.output_path.dot_skip_mode must be one of: fixed, learned_per_var")
    output_path_global_hidden = int(output_path_cfg.get("global_hidden_dim", 64))
    if output_path_global_hidden < 1:
        raise ValueError(f"{cfg_prefix}.model_cfg.output_path.global_hidden_dim must be >= 1")
    residual_cfg = dict(model_cfg.get("residual_head", {}))
    if bool(residual_cfg.get("enabled", False)):
        raise ValueError(f"{cfg_prefix}.model_cfg.residual_head.enabled is removed from deeponet mainline")
    if bool(model_cfg.get("latent_layer_norm", False)):
        raise ValueError(f"{cfg_prefix}.model_cfg.latent_layer_norm is removed from deeponet mainline")
    trunk_fourier_n_freq = int(model_cfg.get("trunk_fourier_n_freq", 1))
    if trunk_fourier_n_freq < 1:
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_fourier_n_freq must be >= 1")
    trunk_fourier_mode = str(model_cfg.get("trunk_fourier_mode", "symmetric")).strip().lower()
    if trunk_fourier_mode != "symmetric":
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_fourier_mode must be symmetric")
    trunk_cond_modulation = str(model_cfg.get("trunk_cond_modulation", "none")).strip().lower()
    if trunk_cond_modulation not in {"none", "film"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_cond_modulation must be one of: none, film")
    missing_geom_feature_policy = str(model_cfg.get("missing_geom_feature_policy", "error")).strip().lower()
    if missing_geom_feature_policy != "error":
        raise ValueError(f"{cfg_prefix}.model_cfg.missing_geom_feature_policy must be error")
    validate_mainline_selection_contract(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )


def build_deeponet_contract_effective(
    *,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    deeponet_target_family: str,
    deeponet_target_vars: list[str],
    deeponet_input_mode: str,
    deeponet_feature_channels: list[str],
    deeponet_feature_source: str,
    deeponet_selection_cfg: dict[str, Any],
    deeponet_optimizer_cfg: dict[str, Any],
    operator_mode: str,
    deeponet_distance_transform_effective: dict[str, Any],
    strict_mainline: bool,
) -> dict[str, Any]:
    model_cfg = dict(cfg.get("model_cfg", {}))
    residual_cfg = dict(model_cfg.get("residual_head", {}))
    output_path_cfg = dict(model_cfg.get("output_path", {}))
    output_path_global_local_cfg = dict(output_path_cfg.get("global_local", {}))
    return {
        "target_family_effective": str(deeponet_target_family),
        "target_vars_effective": list(deeponet_target_vars),
        "feature": {
            "input_features_mode": str(deeponet_input_mode),
            "input_feature_channels": list(deeponet_feature_channels),
            "feature_source_effective": str(deeponet_feature_source),
        },
        "selection_mode_effective": str(deeponet_selection_cfg.get("mode", "last")).strip().lower(),
        "selection_weights_effective": dict(deeponet_selection_cfg.get("weights", {})),
        "optimizer_effective": {
            "type": str(deeponet_optimizer_cfg.get("type", "adamw")).strip().lower(),
            "lr": float(deeponet_optimizer_cfg.get("lr", cfg.get("lr", train_cfg.get("lr", 1e-3)))),
            "weight_decay": float(deeponet_optimizer_cfg.get("weight_decay", 0.0)),
            "schedule": str(deeponet_optimizer_cfg.get("schedule", "none")).strip().lower(),
            "warmup_epochs": int(max(int(deeponet_optimizer_cfg.get("warmup_epochs", 0)), 0)),
        },
        "operator_mode_effective": str(operator_mode),
        "trunk_input_mode_effective": str(model_cfg.get("trunk_input_mode", "geom_feature_pack")),
        "trunk_fourier_n_freq_effective": int(model_cfg.get("trunk_fourier_n_freq", 1)),
        "trunk_fourier_mode_effective": str(model_cfg.get("trunk_fourier_mode", "symmetric")),
        "trunk_cond_modulation_effective": str(model_cfg.get("trunk_cond_modulation", "none")),
        "trunk_cond_mod_hidden_effective": int(model_cfg.get("trunk_cond_mod_hidden", 64)),
        "residual_head_enabled_effective": bool(residual_cfg.get("enabled", False)),
        "residual_head_hidden_dim_effective": int(residual_cfg.get("hidden_dim", 64)),
        "residual_head_scale_init_effective": float(residual_cfg.get("scale_init", 0.0)),
        "residual_head_gain_mode_effective": str(residual_cfg.get("gain_mode", "learned")),
        "residual_head_gain_value_effective": float(residual_cfg.get("gain_value", 1.0)),
        "output_path_mode_effective": str(output_path_cfg.get("mode", "dot")),
        "output_path_dot_skip_effective": float(output_path_cfg.get("dot_skip", 0.25)),
        "output_path_dot_skip_mode_effective": str(output_path_cfg.get("dot_skip_mode", "fixed")).strip().lower(),
        "output_path_fused_hidden_dim_effective": int(output_path_cfg.get("fused_hidden_dim", 96)),
        "output_path_global_local_enabled_effective": bool(output_path_global_local_cfg.get("enabled", False)),
        "output_path_global_hidden_dim_effective": int(output_path_cfg.get("global_hidden_dim", 64)),
        "sensor_pool_mode_effective": str(model_cfg.get("sensor_pool_mode", "moments")),
        "branch_mode_effective": str(model_cfg.get("branch_mode", "moments")),
        "missing_geom_feature_policy_effective": str(model_cfg.get("missing_geom_feature_policy", "error")).strip().lower(),
        "latent_layer_norm_effective": bool(model_cfg.get("latent_layer_norm", False)),
        "distance_transform_effective": dict(deeponet_distance_transform_effective),
        "strict_mainline_effective": bool(strict_mainline),
    }


__all__ = [
    "build_deeponet_contract_effective",
    "resolve_family_train_cfg",
    "resolve_pod_descriptor_and_latent_contract",
    "validate_deeponet_mainline_contract",
    "validate_pod_deeponet_experimental_contract",
]
