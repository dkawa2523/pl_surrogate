"""Shared model train/predict dispatch for workflow and benchmark paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plasma_surrogate.core.contracts import validate_pod_descriptor_latent_contract
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    INPUT_MODES,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.core.model_input_policy import (
    ADAPTER_AUTO,
    resolve_effective_adapter_mode,
    validate_model_input_mode,
)
from plasma_surrogate.core.model_specs import normalize_model_name
from plasma_surrogate.core.model_families import POD_DEEPONET_FAMILY_MODELS
from plasma_surrogate.core.deeponet_contract import (
    load_supervised_boundary_targets,
    resolve_sample_idx_source,
    validate_deeponet_task_meta,
)
from plasma_surrogate.eval.metrics import r2_by_var, rmse_by_var
from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.deeponet.pod_deeponet_torch import fit_pod_basis_from_targets, normalize_pod_deeponet_model_cfg
from plasma_surrogate.models.deeponet.poisson_head_torch import DeepONetPoissonHeadTorch
from plasma_surrogate.models.checkpoint import build_model_from_name
from plasma_surrogate.train.grid_training import run_grid_torch_train_predict
from plasma_surrogate.train.model_adapters import (
    TRAIN_ADAPTER_DEEPONET_PLASMA,
    TRAIN_ADAPTER_GLOBAL_MLP,
    TRAIN_ADAPTER_GRID_TORCH,
    TRAIN_ADAPTER_POD_DEEPONET,
    resolve_train_model_adapter,
)
from plasma_surrogate.train.deeponet_stages import resolve_deeponet_stages
from plasma_surrogate.train.loss_protocols import resolve_loss_protocol
from plasma_surrogate.train.spatial_features import (
    apply_coord_feature_scaling as _apply_coord_feature_scaling,
    apply_distance_transform as _apply_distance_transform,
    build_coord_feature_rows as _build_coord_feature_rows,
    resolve_coord_feature_channels as _resolve_coord_feature_channels,
    resolve_distance_transform_cfg as _resolve_distance_transform_cfg,
    resolve_distance_transform_effective as _resolve_distance_transform_effective,
)
from plasma_surrogate.train.target_contracts import (
    resolve_allvars_target_family as _resolve_unet_target_family,
    resolve_allvars_target_vars_for_family as _resolve_unet_target_vars_for_family,
    resolve_mainline_selection_weights as _resolve_mainline_selection_weights,
    resolve_target_vars as _resolve_target_vars,
    to_true_eval as _to_true_eval,
)
from plasma_surrogate.train.torch_trainer import TorchTrainer
from plasma_surrogate.train.trainer import Trainer


@dataclass
class TrainDispatchContext:
    run_cfg: dict[str, Any]
    profile_lock: dict[str, Any]
    model_idx: int
    model_name: str
    model_dir: Path
    global_seed: int
    n_cases: int
    h: int
    w: int
    y_vars: list[str]
    cond_scaled: np.ndarray
    y: np.ndarray
    y_scaled: np.ndarray
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    transforms: Any
    physics_cfg: dict[str, Any]
    geom_ctx: Any
    deeponet_index: dict[str, Any]
    deeponet_index_meta: dict[str, Any]
    deeponet_poisson_index: dict[str, Any]
    deeponet_poisson_meta: dict[str, Any]
    deeponet_boundary_index: dict[str, Any]
    deeponet_boundary_meta: dict[str, Any]
    config_base_dir: Path | None = None
    loss_cfg: dict[str, Any] | None = None
    curriculum_cfg: dict[str, Any] | None = None
    supervised_mask: np.ndarray | None = None
    supervised_distance: np.ndarray | None = None
    coord_scaler: dict[str, Any] | None = None
    coord_feature_pack: dict[str, Any] | None = None
    case_spatial_feature_pack: dict[str, Any] | None = None
    static_spatial_feature_pack: dict[str, Any] | None = None
    case_structure_feature_pack: dict[str, Any] | None = None
    coord_feature_scaler: dict[str, Any] | None = None
    coord_distance_transform_stats: dict[str, Any] | None = None
    structure_descriptor_pack: dict[str, Any] | None = None
    latent_feature_pack: dict[str, Any] | None = None
    input_mode_effective: str | None = None
    structure_adapter_mode_effective: str | None = None
    structure_descriptor_profile_effective: str | None = None
    structure_latent_profile_effective: str | None = None


@dataclass
class TrainDispatchResult:
    model: Any
    history: list[dict[str, float]]
    pred_eval: dict[str, np.ndarray]
    true_eval: dict[str, np.ndarray]
    metrics: dict[str, float]
    r2_scores: dict[str, float]
    extra_artifacts: dict[str, Any]


@dataclass
class TrainLaneRuntime:
    ctx: TrainDispatchContext
    adapter: Any
    trainer: Trainer
    train_cfg: dict[str, Any]
    optimizer_contract: dict[str, Any]
    model_name: str
    input_mode_effective: str
    structure_adapter_mode_effective: str
    loss_cfg: dict[str, Any]
    supervised_mask: np.ndarray | None
    supervised_distance: np.ndarray | None
    extra_artifacts: dict[str, Any]

    @property
    def h(self) -> int:
        return int(self.ctx.h)

    @property
    def w(self) -> int:
        return int(self.ctx.w)


@dataclass
class TrainLaneResult:
    model: Any
    history: list[dict[str, float]]
    pred_eval: dict[str, np.ndarray]
    true_eval: dict[str, np.ndarray]
    eval_vars: list[str]


TrainLaneHandler = Callable[[TrainLaneRuntime], TrainLaneResult]


def _record_model_contract(extra_artifacts: dict[str, Any], model_name: str, contract: dict[str, Any]) -> None:
    model_contracts = dict(extra_artifacts.get("model_contracts", {}) or {})
    model_contracts[str(model_name)] = dict(contract)
    extra_artifacts["model_contracts"] = model_contracts


@dataclass
class DeeponetRuntime:
    model: DeepONetPlasmaOperatorTorch
    supervised_targets: dict[str, np.ndarray] | None


def _resolve_runtime_input_mode_for_model(*, input_mode_effective: Any) -> str:
    mode = str(input_mode_effective).strip().lower()
    if mode in set(INPUT_MODES):
        return mode
    raise ValueError(f"input_mode_effective must be one of {list(INPUT_MODES)}; got={input_mode_effective!r}")


def validate_runtime_model_policy(*, model_name: Any, input_mode_effective: Any) -> str:
    name = normalize_model_name(model_name)
    mode = _resolve_runtime_input_mode_for_model(
        input_mode_effective=input_mode_effective,
    )
    validate_model_input_mode(name, mode)
    return name


def resolve_runtime_model_adapter_policy(
    *,
    model_name: Any,
    input_mode_effective: Any,
    structure_adapter_mode_effective: Any,
) -> tuple[str, str, str]:
    name = validate_runtime_model_policy(
        model_name=model_name,
        input_mode_effective=input_mode_effective,
    )
    mode = _resolve_runtime_input_mode_for_model(
        input_mode_effective=input_mode_effective,
    )
    requested_adapter_mode = (
        str(structure_adapter_mode_effective).strip().lower()
        if structure_adapter_mode_effective is not None
        else ADAPTER_AUTO
    )
    effective_adapter_mode = resolve_effective_adapter_mode(
        model_name=name,
        input_mode=mode,
        adapter_mode=requested_adapter_mode,
    )
    return name, mode, effective_adapter_mode


def _normalize_profile_name(value: Any, *, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    return text if text else default


def _resolve_pod_descriptor_and_latent_contract(
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

    descriptor_vector: np.ndarray | None = None
    descriptor_dim = 0
    if desc_profile != "none":
        descriptor_vector, _ = load_vector_from_pack(
            pack=descriptor_pack,
            pack_name="structure_descriptor_pack",
        )
        descriptor_dim = int(descriptor_vector.shape[0])

    latent_hook = False
    if lat_profile != "none":
        load_vector_from_pack(
            pack=latent_pack,
            pack_name="latent_feature_pack",
        )
        latent_hook = True

    return descriptor_vector, {
        DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: int(descriptor_dim),
        DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(desc_profile),
        DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: str(lat_profile),
        DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: bool(latent_hook),
    }


def _validate_pod_deeponet_experimental_contract(
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
    if selection_mode not in {"best_val_allvars_balance", "best_val_loss"}:
        raise ValueError(f"{cfg_prefix}.selection.mode must be one of: best_val_allvars_balance, best_val_loss")
    return {
        "rank": int(rank),
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


def _resolve_family_train_cfg(train_cfg: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    cfg = dict(train_cfg.get(model_name, {}))
    if model_name in POD_DEEPONET_FAMILY_MODELS:
        cfg = resolve_deeponet_pod_training_defaults(cfg)
    if model_name == "coord_mlp_siren":
        cfg = _resolve_coord_mlp_siren_training_defaults(cfg)
    if model_name == "coord_mlp_pod_residual":
        cfg = _resolve_coord_mlp_pod_residual_training_defaults(cfg)
    return cfg


def _validate_deeponet_mainline_contract(
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
    cfg_prefix = "train.deeponet_plasma"
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for mainline")
    expected = list(y_vars)
    if list(target_vars) != list(expected):
        raise ValueError(f"{cfg_prefix}.target_vars must match allvars order: expected={expected}, got={target_vars}")
    if str(input_features_mode).strip().lower() != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack for mainline")
    required_channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    if [str(v) for v in list(input_feature_channels)] != required_channels:
        raise ValueError(
            f"{cfg_prefix}.input_features.features must be {required_channels} for mainline; got={input_feature_channels}"
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
    if branch_mode != "cond_only":
        raise ValueError(f"{cfg_prefix}.model_cfg.branch_mode must be cond_only for mainline")
    sensor_pool_mode = str(model_cfg.get("sensor_pool_mode", "moments")).strip().lower()
    if sensor_pool_mode != "moments":
        raise ValueError(f"{cfg_prefix}.model_cfg.sensor_pool_mode must be moments for cond_only mainline")
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
    sel_mode = str(selection_cfg.get("mode", "last")).strip().lower()
    if sel_mode not in {"best_val_allvars_balance", "best_val_loss"}:
        raise ValueError(f"{cfg_prefix}.selection.mode must be one of: best_val_allvars_balance, best_val_loss for mainline")
    if "boundary_bonus_weight" in selection_cfg and float(selection_cfg.get("boundary_bonus_weight", 0.0)) != 0.0:
        raise ValueError(f"{cfg_prefix}.selection.boundary_bonus_weight must be 0.0 for mainline")
    if "density_guard" in selection_cfg:
        raise ValueError(f"{cfg_prefix}.selection.density_guard is removed from deeponet mainline")
    _resolve_mainline_selection_weights(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )

def _build_deeponet_contract_effective(
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


def resolve_deeponet_runtime(
    ctx: TrainDispatchContext,
    deeponet_plasma_cfg: dict[str, Any],
    *,
    output_keys: list[str] | None = None,
    sensor_feature_names: list[str] | None = None,
    operator_mode: str = "pde_coupled",
) -> DeeponetRuntime:
    h, w = ctx.h, ctx.w
    model_cfg = dict(deeponet_plasma_cfg.get("model_cfg", deeponet_plasma_cfg))
    branch_mode = str(model_cfg.get("branch_mode", "moments")).strip().lower()
    missing_geom_feature_policy = str(model_cfg.get("missing_geom_feature_policy", "error")).strip().lower()
    residual_head_cfg = dict(model_cfg.get("residual_head", {}))
    output_path_cfg = dict(model_cfg.get("output_path", {}))
    mode = str(operator_mode).strip().lower()
    if mode not in {"plain", "pde_coupled"}:
        raise ValueError("train.deeponet_plasma.operator_mode must be one of: plain, pde_coupled")
    plain_output_keys = list(output_keys or ctx.y_vars)
    if mode == "plain":
        sensor_idx = None
        query_idx = None
        flatten_order = "C"
        if branch_mode != "cond_only" and isinstance(ctx.deeponet_index, dict) and ctx.deeponet_index:
            sensor_idx_arr = np.asarray(ctx.deeponet_index.get("sensor_indices", []), dtype=np.int64).reshape(-1)
            query_idx_arr = np.asarray(ctx.deeponet_index.get("query_indices", []), dtype=np.int64).reshape(-1)
            sensor_idx = sensor_idx_arr if sensor_idx_arr.size > 0 else None
            query_idx = query_idx_arr if query_idx_arr.size > 0 else None
            flatten_order = str(dict(ctx.deeponet_index_meta or {}).get("flatten_order", "C"))
        model = DeepONetPlasmaOperatorTorch(
            cond_dim=ctx.cond_scaled.shape[1],
            grid_shape=(h, w),
            output_keys=plain_output_keys,
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            sensor_indices=sensor_idx,
            query_indices=query_idx,
            flatten_order=flatten_order,
            sensor_feature_names=list(sensor_feature_names or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]),
            trunk_input_mode=str(model_cfg.get("trunk_input_mode", "geom_feature_pack")),
            sensor_pool_mode=str(model_cfg.get("sensor_pool_mode", "moments")),
            sensor_embed_dim=int(model_cfg.get("sensor_embed_dim", 32)),
            branch_mode=branch_mode,
            trunk_fourier_n_freq=int(model_cfg.get("trunk_fourier_n_freq", 1)),
            trunk_fourier_mode=str(model_cfg.get("trunk_fourier_mode", "symmetric")),
            trunk_cond_modulation=str(model_cfg.get("trunk_cond_modulation", "none")),
            trunk_cond_mod_hidden=int(model_cfg.get("trunk_cond_mod_hidden", 64)),
            residual_head_enabled=bool(residual_head_cfg.get("enabled", False)),
            residual_head_hidden_dim=int(residual_head_cfg.get("hidden_dim", 64)),
            residual_head_scale_init=float(residual_head_cfg.get("scale_init", 0.0)),
            residual_head_gain_mode=str(residual_head_cfg.get("gain_mode", "learned")),
            residual_head_gain_value=float(residual_head_cfg.get("gain_value", 1.0)),
            latent_layer_norm=bool(model_cfg.get("latent_layer_norm", False)),
            output_path_mode=str(output_path_cfg.get("mode", "dot")),
            output_path_dot_skip=float(output_path_cfg.get("dot_skip", 0.25)),
            output_path_dot_skip_mode=str(output_path_cfg.get("dot_skip_mode", "fixed")),
            output_path_fused_hidden_dim=int(output_path_cfg.get("fused_hidden_dim", 96)),
            output_path_global_local_enabled=bool(output_path_cfg.get("global_local", {}).get("enabled", False)),
            output_path_global_hidden_dim=int(output_path_cfg.get("global_hidden_dim", 64)),
            missing_geom_feature_policy=missing_geom_feature_policy,
            seed=ctx.global_seed + ctx.model_idx,
        )
        return DeeponetRuntime(model=model, supervised_targets=None)

    if not ctx.deeponet_poisson_index or not ctx.deeponet_boundary_index:
        raise ValueError("deeponet_plasma requires task artifacts for poisson_head and boundary_operator")
    validate_deeponet_task_meta("poisson_head", ctx.deeponet_poisson_meta, expected_shape=(h, w))
    validate_deeponet_task_meta(
        "boundary_operator",
        ctx.deeponet_boundary_meta,
        expected_shape=(h, w),
        expected_order=str(ctx.deeponet_poisson_meta.get("flatten_order", "C")),
    )
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=ctx.cond_scaled.shape[1],
        grid_shape=(h, w),
        output_keys=plain_output_keys + ["rho_eff"],
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        sensor_indices=np.asarray(ctx.deeponet_poisson_index["sensor_indices"], dtype=np.int64),
        query_indices=np.asarray(ctx.deeponet_poisson_index["query_indices"], dtype=np.int64),
        flatten_order=str(ctx.deeponet_poisson_meta.get("flatten_order", "C")),
        sensor_feature_names=list(sensor_feature_names or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]),
        trunk_input_mode=str(model_cfg.get("trunk_input_mode", "geom_feature_pack")),
        sensor_pool_mode=str(model_cfg.get("sensor_pool_mode", "moments")),
        sensor_embed_dim=int(model_cfg.get("sensor_embed_dim", 32)),
        branch_mode=branch_mode,
        trunk_fourier_n_freq=int(model_cfg.get("trunk_fourier_n_freq", 1)),
        trunk_fourier_mode=str(model_cfg.get("trunk_fourier_mode", "symmetric")),
        trunk_cond_modulation=str(model_cfg.get("trunk_cond_modulation", "none")),
        trunk_cond_mod_hidden=int(model_cfg.get("trunk_cond_mod_hidden", 64)),
        residual_head_enabled=bool(residual_head_cfg.get("enabled", False)),
        residual_head_hidden_dim=int(residual_head_cfg.get("hidden_dim", 64)),
        residual_head_scale_init=float(residual_head_cfg.get("scale_init", 0.0)),
        residual_head_gain_mode=str(residual_head_cfg.get("gain_mode", "learned")),
        residual_head_gain_value=float(residual_head_cfg.get("gain_value", 1.0)),
        latent_layer_norm=bool(model_cfg.get("latent_layer_norm", False)),
        output_path_mode=str(output_path_cfg.get("mode", "dot")),
        output_path_dot_skip=float(output_path_cfg.get("dot_skip", 0.25)),
        output_path_dot_skip_mode=str(output_path_cfg.get("dot_skip_mode", "fixed")),
        output_path_fused_hidden_dim=int(output_path_cfg.get("fused_hidden_dim", 96)),
        output_path_global_local_enabled=bool(output_path_cfg.get("global_local", {}).get("enabled", False)),
        output_path_global_hidden_dim=int(output_path_cfg.get("global_hidden_dim", 64)),
        missing_geom_feature_policy=missing_geom_feature_policy,
        seed=ctx.global_seed + ctx.model_idx,
    )
    poisson_cfg = dict(model_cfg.get("poisson_head", {}))
    poisson_residual_cfg = dict(poisson_cfg.get("residual_head", residual_head_cfg) or {})
    poisson_net = DeepONetPlasmaOperatorTorch(
        cond_dim=ctx.cond_scaled.shape[1],
        grid_shape=(h, w),
        output_keys=["phi"],
        latent_dim=int(poisson_cfg.get("latent_dim", model_cfg.get("latent_dim", 32))),
        hidden_dim=int(poisson_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 64))),
        sensor_indices=np.asarray(ctx.deeponet_poisson_index["sensor_indices"], dtype=np.int64),
        query_indices=np.asarray(ctx.deeponet_poisson_index["query_indices"], dtype=np.int64),
        flatten_order=str(ctx.deeponet_poisson_meta.get("flatten_order", "C")),
        sensor_feature_names=list(sensor_feature_names or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]),
        trunk_input_mode=str(poisson_cfg.get("trunk_input_mode", model_cfg.get("trunk_input_mode", "geom_feature_pack"))),
        sensor_pool_mode=str(poisson_cfg.get("sensor_pool_mode", model_cfg.get("sensor_pool_mode", "moments"))),
        sensor_embed_dim=int(poisson_cfg.get("sensor_embed_dim", model_cfg.get("sensor_embed_dim", 32))),
        branch_mode=str(poisson_cfg.get("branch_mode", model_cfg.get("branch_mode", "moments"))),
        trunk_fourier_n_freq=int(poisson_cfg.get("trunk_fourier_n_freq", model_cfg.get("trunk_fourier_n_freq", 1))),
        trunk_fourier_mode=str(poisson_cfg.get("trunk_fourier_mode", model_cfg.get("trunk_fourier_mode", "symmetric"))),
        trunk_cond_modulation=str(poisson_cfg.get("trunk_cond_modulation", model_cfg.get("trunk_cond_modulation", "none"))),
        trunk_cond_mod_hidden=int(poisson_cfg.get("trunk_cond_mod_hidden", model_cfg.get("trunk_cond_mod_hidden", 64))),
        residual_head_enabled=bool(poisson_residual_cfg.get("enabled", False)),
        residual_head_hidden_dim=int(poisson_residual_cfg.get("hidden_dim", 64)),
        residual_head_scale_init=float(poisson_residual_cfg.get("scale_init", 0.0)),
        residual_head_gain_mode=str(poisson_residual_cfg.get("gain_mode", residual_head_cfg.get("gain_mode", "learned"))),
        residual_head_gain_value=float(poisson_residual_cfg.get("gain_value", residual_head_cfg.get("gain_value", 1.0))),
        latent_layer_norm=bool(poisson_cfg.get("latent_layer_norm", model_cfg.get("latent_layer_norm", False))),
        output_path_mode=str(poisson_cfg.get("output_path_mode", output_path_cfg.get("mode", "dot"))),
        output_path_dot_skip=float(poisson_cfg.get("output_path_dot_skip", output_path_cfg.get("dot_skip", 0.25))),
        output_path_dot_skip_mode=str(
            poisson_cfg.get(
                "output_path_dot_skip_mode",
                output_path_cfg.get("dot_skip_mode", "fixed"),
            )
        ),
        output_path_fused_hidden_dim=int(poisson_cfg.get("output_path_fused_hidden_dim", output_path_cfg.get("fused_hidden_dim", 96))),
        output_path_global_local_enabled=bool(
            poisson_cfg.get(
                "output_path_global_local_enabled",
                output_path_cfg.get("global_local", {}).get("enabled", False),
            )
        ),
        output_path_global_hidden_dim=int(
            poisson_cfg.get(
                "output_path_global_hidden_dim",
                output_path_cfg.get("global_hidden_dim", 64),
            )
        ),
        missing_geom_feature_policy=str(
            poisson_cfg.get(
                "missing_geom_feature_policy",
                missing_geom_feature_policy,
            )
        ),
        seed=int(poisson_cfg.get("seed", ctx.global_seed + 1000 + ctx.model_idx)),
    )
    poisson_head = DeepONetPoissonHeadTorch.from_cache(
        deeponet_poisson=poisson_net,
        sensor_idx=np.asarray(ctx.deeponet_poisson_index["sensor_indices"], dtype=np.int64),
        query_idx=np.asarray(ctx.deeponet_poisson_index["query_indices"], dtype=np.int64),
        flatten_order=str(ctx.deeponet_poisson_meta.get("flatten_order", "C")),
        grid_shape=(h, w),
        freeze=bool(poisson_cfg.get("freeze", True)),
    )
    bo_cfg = dict(model_cfg.get("boundary_operator", {}))
    boundary_operator = BoundaryOperatorTorch(
        primary_qoi_key=str(ctx.profile_lock.get("primary_qoi_key", bo_cfg.get("primary_qoi_key", "Gamma_i"))),
        freeze=bool(bo_cfg.get("freeze", True)),
    )
    model.attach_poisson_head(poisson_head)
    model.attach_boundary_operator(boundary_operator)

    bo_phys_cfg = ctx.physics_cfg.get("boundary_operator", {})
    supervised_targets = None
    if bool(bo_phys_cfg.get("enabled", False)):
        sample_idx = resolve_sample_idx_source(
            bo_phys_cfg.get("sample_idx_source"),
            boundary_sensor_idx=np.asarray(ctx.deeponet_boundary_index["sensor_indices"], dtype=np.int64),
            boundary_query_idx=np.asarray(ctx.deeponet_boundary_index["query_indices"], dtype=np.int64),
            poisson_sensor_idx=np.asarray(ctx.deeponet_poisson_index["sensor_indices"], dtype=np.int64),
            poisson_query_idx=np.asarray(ctx.deeponet_poisson_index["query_indices"], dtype=np.int64),
        )
        bo_phys_cfg["sample_idx"] = sample_idx
        if str(bo_phys_cfg.get("mode", "operator_prior")) == "supervised":
            supervised_targets = load_supervised_boundary_targets(
                bo_phys_cfg.get("supervised_targets_npz"),
                primary_qoi_key=str(bo_phys_cfg.get("primary_qoi_key", ctx.profile_lock.get("primary_qoi_key", "Gamma_i"))),
                expected_grid_shape=(h, w),
                base_dir=ctx.config_base_dir,
            )
    return DeeponetRuntime(model=model, supervised_targets=supervised_targets)


def run_model_train_predict(ctx: TrainDispatchContext) -> TrainDispatchResult:
    return resolve_train_model_adapter(ctx.model_name).run(ctx)


def _run_global_mlp_train_predict(
    ctx: TrainDispatchContext,
    *,
    adapter: Any,
) -> TrainDispatchResult:
    return _run_train_lane(ctx, adapter=adapter, handler=_run_global_mlp_lane)


def _run_pod_deeponet_train_predict(
    ctx: TrainDispatchContext,
    *,
    adapter: Any,
) -> TrainDispatchResult:
    return _run_train_lane(ctx, adapter=adapter, handler=_run_pod_deeponet_lane)


def _run_grid_torch_adapter_train_predict(
    ctx: TrainDispatchContext,
    *,
    adapter: Any,
) -> TrainDispatchResult:
    return _run_train_lane(ctx, adapter=adapter, handler=_run_grid_torch_lane)


def _run_deeponet_plasma_train_predict(
    ctx: TrainDispatchContext,
    *,
    adapter: Any,
) -> TrainDispatchResult:
    return _run_train_lane(ctx, adapter=adapter, handler=_run_deeponet_plasma_lane)


def _build_train_lane_runtime(
    ctx: TrainDispatchContext,
    *,
    expected_adapter: Any,
) -> TrainLaneRuntime:
    trainer = Trainer(ctx.model_dir / "train")
    train_cfg = dict(ctx.run_cfg.get("train", {}))
    optimizer_contract = dict(train_cfg.get("optimizer_contract", {}))
    model_name, input_mode_effective, structure_adapter_mode_effective = resolve_runtime_model_adapter_policy(
        model_name=ctx.model_name,
        input_mode_effective=ctx.input_mode_effective,
        structure_adapter_mode_effective=ctx.structure_adapter_mode_effective,
    )
    model_adapter = resolve_train_model_adapter(model_name)
    if model_adapter.name != expected_adapter.name:
        raise ValueError(
            f"Train adapter mismatch for {model_name!r}: "
            f"resolved={model_adapter.name!r}, expected={expected_adapter.name!r}"
        )
    raw_loss_cfg = ctx.loss_cfg if ctx.loss_cfg is not None else train_cfg.get("loss", {})
    role_schema = dict(dict(ctx.physics_cfg or {}).get("target_role_schema", {}) or {})
    loss_cfg = resolve_loss_protocol(copy.deepcopy(raw_loss_cfg or {}), target_role_schema=role_schema)
    supervised_mask = ctx.supervised_mask
    supervised_distance = ctx.supervised_distance
    if (
        str(dict(loss_cfg.get("supervised", {})).get("mask", "none")).strip().lower() == "plasma_only"
        and ctx.geom_ctx is not None
    ):
        if supervised_mask is None and getattr(ctx.geom_ctx, "mask_plasma", None) is not None:
            supervised_mask = np.asarray(ctx.geom_ctx.mask_plasma, dtype=np.float32)
        if supervised_distance is None and getattr(ctx.geom_ctx, "distance_any", None) is not None:
            supervised_distance = np.asarray(ctx.geom_ctx.distance_any, dtype=np.float32)
    extra_artifacts: dict[str, Any] = {
        "input_mode_effective": str(input_mode_effective),
        "structure_adapter_mode_effective": str(structure_adapter_mode_effective),
        "loss_protocol_effective": str(loss_cfg.get("protocol_effective", "none")),
    }
    return TrainLaneRuntime(
        ctx=ctx,
        adapter=expected_adapter,
        trainer=trainer,
        train_cfg=train_cfg,
        optimizer_contract=optimizer_contract,
        model_name=model_name,
        input_mode_effective=input_mode_effective,
        structure_adapter_mode_effective=structure_adapter_mode_effective,
        loss_cfg=loss_cfg,
        supervised_mask=supervised_mask,
        supervised_distance=supervised_distance,
        extra_artifacts=extra_artifacts,
    )


def _run_train_lane(
    ctx: TrainDispatchContext,
    *,
    adapter: Any,
    handler: TrainLaneHandler,
) -> TrainDispatchResult:
    runtime = _build_train_lane_runtime(ctx, expected_adapter=adapter)
    lane = handler(runtime)
    true_eval_metrics = {k: lane.true_eval[k] for k in lane.eval_vars if k in lane.true_eval and k in lane.pred_eval}
    pred_eval_metrics = {k: lane.pred_eval[k] for k in lane.eval_vars if k in lane.true_eval and k in lane.pred_eval}
    return TrainDispatchResult(
        model=lane.model,
        history=lane.history,
        pred_eval=lane.pred_eval,
        true_eval=lane.true_eval,
        metrics=rmse_by_var(true_eval_metrics, pred_eval_metrics),
        r2_scores=r2_by_var(true_eval_metrics, pred_eval_metrics),
        extra_artifacts=runtime.extra_artifacts,
    )


def _run_model_train_predict_for_adapter(
    ctx: TrainDispatchContext,
    *,
    expected_adapter: Any,
) -> TrainDispatchResult:
    handlers: dict[str, TrainLaneHandler] = {
        TRAIN_ADAPTER_GLOBAL_MLP: _run_global_mlp_lane,
        TRAIN_ADAPTER_POD_DEEPONET: _run_pod_deeponet_lane,
        TRAIN_ADAPTER_GRID_TORCH: _run_grid_torch_lane,
        TRAIN_ADAPTER_DEEPONET_PLASMA: _run_deeponet_plasma_lane,
    }
    try:
        handler = handlers[str(expected_adapter.name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported train adapter: {expected_adapter.name}") from exc
    return _run_train_lane(ctx, adapter=expected_adapter, handler=handler)


def _run_global_mlp_lane(rt: TrainLaneRuntime) -> TrainLaneResult:
    ctx = rt.ctx
    train_cfg = rt.train_cfg
    cfg = dict(train_cfg.get("global_mlp", {}))
    batch_size_cases = int(cfg.get("batch_size_cases", 0))
    shuffle_cases = bool(cfg.get("shuffle_cases", True))
    grad_scale_cfg = dict(cfg.get("grad_scale", {}))
    layer_lr_multiplier = dict(cfg.get("layer_lr_multiplier", {}))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    grad_clip_cfg = dict(cfg.get("grad_clip", {}))
    output_head_refresh_cfg = dict(cfg.get("output_head_refresh", {}))
    global_loss_cfg = copy.deepcopy(rt.loss_cfg or {})
    sup = dict(global_loss_cfg.get("supervised", {}))
    if bool(sup.get("label_clip_from_scaler", False)):
        transform_specs = dict(getattr(ctx.transforms, "target_transforms", {}) or {})
        clip_stats: dict[str, dict[str, float]] = {}
        for var, spec in transform_specs.items():
            clip_cfg = dict(dict(spec).get("clip", {}))
            if str(clip_cfg.get("mode", "none")).strip().lower() != "quantile":
                continue
            if "clip_low" in clip_cfg and "clip_high" in clip_cfg:
                clip_stats[str(var)] = {
                    "clip_low": float(clip_cfg["clip_low"]),
                    "clip_high": float(clip_cfg["clip_high"]),
                }
        if clip_stats:
            sup["robust_clip_stats"] = clip_stats
    global_loss_cfg["supervised"] = sup
    model = build_model_from_name(
        model_name="global_mlp",
        input_dim=ctx.cond_scaled.shape[1],
        grid_shape=(rt.h, rt.w),
        model_cfg=dict(cfg.get("model_cfg", {})),
        seed=ctx.global_seed + ctx.model_idx,
        phi_mode=str(ctx.profile_lock.get("phi_mode", "direct")),
        out_channels=len(ctx.y_vars),
        output_keys=ctx.y_vars,
    )
    y_flat = ctx.y_scaled.reshape(ctx.n_cases, -1)
    out = rt.trainer.run_global(
        model,
        ctx.cond_scaled[ctx.tr],
        y_flat[ctx.tr],
        ctx.cond_scaled[ctx.va],
        y_flat[ctx.va],
        epochs=int(cfg.get("epochs", train_cfg.get("epochs", 20))),
        lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
        physics_cfg=ctx.physics_cfg,
        loss_cfg=global_loss_cfg,
        curriculum_cfg=ctx.curriculum_cfg,
        supervised_mask=rt.supervised_mask,
        supervised_distance=rt.supervised_distance,
        batch_size_cases=batch_size_cases,
        shuffle_cases=shuffle_cases,
        seed=ctx.global_seed + ctx.model_idx,
        grad_scale_cfg=grad_scale_cfg,
        layer_lr_multiplier=layer_lr_multiplier,
        grad_clip_norm=grad_clip_norm,
        grad_clip_cfg=grad_clip_cfg,
        output_head_refresh_cfg=output_head_refresh_cfg,
        selection_cfg=dict(cfg.get("selection", {})),
    )
    history = out.history
    steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, batch_size_cases))) if batch_size_cases > 0 else 1
    rt.extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
    return TrainLaneResult(
        model=model,
        history=history,
        pred_eval=ctx.transforms.inverse_field_dict(model.predict_fields(ctx.cond_scaled[ctx.te])),
        true_eval=_to_true_eval(ctx.y, ctx.te, ctx.y_vars),
        eval_vars=list(ctx.y_vars),
    )


def _run_pod_deeponet_lane(rt: TrainLaneRuntime) -> TrainLaneResult:
    ctx = rt.ctx
    train_cfg = rt.train_cfg
    cfg = _resolve_family_train_cfg(train_cfg, model_name=rt.model_name)
    pod_model_type = str(rt.model_name).strip().lower()
    train_key = f"train.{pod_model_type}"
    pod_target_family = _resolve_unet_target_family(
        cfg.get("target_family", "allvars"),
        cfg_key=f"{train_key}.target_family",
    )
    pod_target_vars = _resolve_unet_target_vars_for_family(
        family=pod_target_family,
        raw_target_vars=cfg.get("target_vars"),
        available=ctx.y_vars,
        cfg_key=f"{train_key}.target_vars",
    )
    basis_cfg_effective = _validate_pod_deeponet_experimental_contract(
        model_name=pod_model_type,
        y_vars=ctx.y_vars,
        target_family=pod_target_family,
        target_vars=pod_target_vars,
        model_cfg=dict(cfg.get("model_cfg", {})),
        selection_cfg=dict(cfg.get("selection", {})),
    )
    pod_descriptor_vec, pod_descriptor_meta = _resolve_pod_descriptor_and_latent_contract(
        input_mode=rt.input_mode_effective,
        adapter_mode=rt.structure_adapter_mode_effective,
        descriptor_profile=ctx.structure_descriptor_profile_effective,
        latent_profile=ctx.structure_latent_profile_effective,
        descriptor_pack=ctx.structure_descriptor_pack,
        latent_pack=ctx.latent_feature_pack,
    )
    rt.extra_artifacts.update(dict(pod_descriptor_meta))
    pod_target_indices = [ctx.y_vars.index(v) for v in pod_target_vars]
    cond_train = np.asarray(ctx.cond_scaled[ctx.tr], dtype=np.float32)
    cond_val = np.asarray(ctx.cond_scaled[ctx.va], dtype=np.float32)
    cond_test = np.asarray(ctx.cond_scaled[ctx.te], dtype=np.float32)
    if pod_descriptor_vec is not None:
        desc_train = np.repeat(pod_descriptor_vec.reshape(1, -1), cond_train.shape[0], axis=0).astype(np.float32)
        desc_val = np.repeat(pod_descriptor_vec.reshape(1, -1), cond_val.shape[0], axis=0).astype(np.float32)
        desc_test = np.repeat(pod_descriptor_vec.reshape(1, -1), cond_test.shape[0], axis=0).astype(np.float32)
        cond_train = np.concatenate([cond_train, desc_train], axis=1).astype(np.float32)
        cond_val = np.concatenate([cond_val, desc_val], axis=1).astype(np.float32)
        cond_test = np.concatenate([cond_test, desc_test], axis=1).astype(np.float32)
    pod_y_train = np.asarray(ctx.y_scaled[ctx.tr][:, pod_target_indices], dtype=np.float32)
    pod_basis_bundle = fit_pod_basis_from_targets(
        pod_y_train,
        output_keys=list(pod_target_vars),
        requested_rank=int(basis_cfg_effective["rank"]),
        center=bool(basis_cfg_effective["center"]),
        per_var=bool(basis_cfg_effective["per_var"]),
    )
    pod_model_cfg = normalize_pod_deeponet_model_cfg(
        dict(cfg.get("model_cfg", {})),
        model_type=pod_model_type,
    )
    model = build_model_from_name(
        model_name=rt.model_name,
        input_dim=int(cond_train.shape[1]),
        grid_shape=(rt.h, rt.w),
        model_cfg=pod_model_cfg,
        seed=ctx.global_seed + ctx.model_idx,
        phi_mode=str(ctx.profile_lock.get("phi_mode", "direct")),
        out_channels=len(pod_target_vars),
        output_keys=pod_target_vars,
        pod_basis_bundle=pod_basis_bundle,
    )
    pod_selection_cfg = dict(cfg.get("selection", {}))
    pod_selection_cfg["weights"] = _resolve_mainline_selection_weights(
        selection_cfg=pod_selection_cfg,
        target_vars=pod_target_vars,
        cfg_prefix=train_key,
    )
    pod_optimizer_cfg = dict(cfg.get("optimizer", {}))
    grid_like_cfg = dict(train_cfg.get("unet_like", {}))
    batch_size_cases = int(grid_like_cfg.get("batch_size_cases", 0))
    shuffle_cases = bool(grid_like_cfg.get("shuffle_cases", True))
    out = rt.trainer.run_unet(
        model,
        cond_train,
        pod_y_train,
        cond_val,
        np.asarray(ctx.y_scaled[ctx.va][:, pod_target_indices], dtype=np.float32),
        epochs=int(cfg.get("epochs", train_cfg.get("epochs", 20))),
        lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
        physics_cfg={"enabled": False},
        loss_cfg=copy.deepcopy(rt.loss_cfg or {}),
        curriculum_cfg=ctx.curriculum_cfg,
        supervised_mask=rt.supervised_mask,
        supervised_distance=rt.supervised_distance,
        optimizer_contract=rt.optimizer_contract,
        unet_optimizer_cfg=pod_optimizer_cfg,
        batch_size_cases=batch_size_cases,
        shuffle_cases=shuffle_cases,
        seed=ctx.global_seed + ctx.model_idx,
        selection_cfg=pod_selection_cfg,
    )
    history = out.history
    steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, batch_size_cases))) if batch_size_cases > 0 else 1
    rt.extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
    _record_model_contract(
        rt.extra_artifacts,
        "deeponet_pod",
        {
            "model_type_effective": pod_model_type,
            "target_family_effective": str(pod_target_family),
            "target_vars_effective": list(pod_target_vars),
            "basis_rank_by_var": {str(k): int(v) for k, v in pod_basis_bundle.rank_by_var.items()},
            "coeff_std_by_var": {
                str(k): np.asarray(v, dtype=np.float32).reshape(-1).tolist()
                for k, v in pod_basis_bundle.coeff_std_by_var.items()
            },
            "basis_fit_scope_effective": str(basis_cfg_effective["fit_scope"]),
            "basis_center_effective": bool(basis_cfg_effective["center"]),
            "basis_per_var_effective": bool(basis_cfg_effective["per_var"]),
            "selection_mode_effective": str(pod_selection_cfg.get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(pod_selection_cfg.get("weights", {})),
            "optimizer_effective": {
                "type": str(pod_optimizer_cfg.get("type", "adamw")).strip().lower(),
                "lr": float(pod_optimizer_cfg.get("lr", cfg.get("lr", train_cfg.get("lr", 1e-3)))),
                "schedule": str(pod_optimizer_cfg.get("schedule", "none")).strip().lower(),
                "warmup_epochs": int(max(int(pod_optimizer_cfg.get("warmup_epochs", 0)), 0)),
            },
            "model_cfg_effective": dict(pod_model_cfg),
            DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: int(
                pod_descriptor_meta[DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY]
            ),
            DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(
                pod_descriptor_meta[DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY]
            ),
            DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: str(
                pod_descriptor_meta[DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY]
            ),
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: bool(
                pod_descriptor_meta[DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY]
            ),
        },
    )
    pred_features = model.forward_features(cond_test)
    return TrainLaneResult(
        model=model,
        history=history,
        pred_eval=ctx.transforms.inverse_field_dict(
            {name: np.asarray(pred_features[name], dtype=np.float32) for name in pod_target_vars}
        ),
        true_eval=_to_true_eval(ctx.y, ctx.te, pod_target_vars, source_y_vars=ctx.y_vars),
        eval_vars=list(pod_target_vars),
    )


def _run_grid_torch_lane(rt: TrainLaneRuntime) -> TrainLaneResult:
    ctx = rt.ctx
    grid_ctx = replace(
        ctx,
        loss_cfg=rt.loss_cfg,
        supervised_mask=rt.supervised_mask,
        supervised_distance=rt.supervised_distance,
    )
    grid_result = run_grid_torch_train_predict(
        ctx=grid_ctx,
        trainer=rt.trainer,
        train_cfg=rt.train_cfg,
        optimizer_contract=rt.optimizer_contract,
        model_name=rt.model_name,
        cfg=_resolve_family_train_cfg(rt.train_cfg, model_name=rt.model_name),
        input_mode_effective=rt.input_mode_effective,
        structure_adapter_mode_effective=rt.structure_adapter_mode_effective,
        extra_artifacts=rt.extra_artifacts,
    )
    return TrainLaneResult(
        model=grid_result.model,
        history=grid_result.history,
        pred_eval=grid_result.pred_eval,
        true_eval=grid_result.true_eval,
        eval_vars=grid_result.eval_vars,
    )


def _run_deeponet_plasma_lane(rt: TrainLaneRuntime) -> TrainLaneResult:
    ctx = rt.ctx
    train_cfg = rt.train_cfg
    cfg = dict(train_cfg.get("deeponet_plasma", {}))
    train_key = "train.deeponet_plasma"
    deeponet_target_family = _resolve_unet_target_family(
        cfg.get("target_family", "allvars"),
        cfg_key=f"{train_key}.target_family",
    )
    default_deeponet_vars = list(ctx.y_vars)
    deeponet_target_vars = (
        list(default_deeponet_vars)
        if cfg.get("target_vars") is None
        else _resolve_target_vars(
            cfg.get("target_vars"),
            available=ctx.y_vars,
            cfg_key=f"{train_key}.target_vars",
        )
    )
    deeponet_target_indices = [ctx.y_vars.index(v) for v in deeponet_target_vars]
    input_features_cfg = dict(cfg.get("input_features", {}))
    deeponet_input_mode = str(input_features_cfg.get("mode", "geom_feature_pack")).strip().lower()
    if deeponet_input_mode != "geom_feature_pack":
        raise ValueError(f"{train_key}.input_features.mode must be geom_feature_pack")
    deeponet_feature_channels = _resolve_coord_feature_channels(input_features_cfg.get("features"))
    deeponet_selection_cfg = dict(cfg.get("selection", {}))
    deeponet_optimizer_cfg = dict(cfg.get("optimizer", {}))
    deeponet_batch_size_cases = int(cfg.get("batch_size_cases", 0))
    deeponet_shuffle_cases = bool(cfg.get("shuffle_cases", True))
    strict_mainline = bool(cfg.get("strict_mainline", False))
    if strict_mainline:
        _validate_deeponet_mainline_contract(
            deeponet_cfg=cfg,
            selection_cfg=deeponet_selection_cfg,
            loss_cfg=dict(rt.loss_cfg or {}),
            y_vars=ctx.y_vars,
            target_family=deeponet_target_family,
            target_vars=deeponet_target_vars,
            input_features_mode=deeponet_input_mode,
            input_feature_channels=deeponet_feature_channels,
        )
        if bool(dict(ctx.physics_cfg or {}).get("enabled", False)):
            raise ValueError("train.physics.enabled must be false for deeponet mainline plain operator")
    deeponet_selection_cfg["weights"] = _resolve_mainline_selection_weights(
        selection_cfg=deeponet_selection_cfg,
        target_vars=deeponet_target_vars,
        cfg_prefix=train_key,
    )
    operator_mode = str(cfg.get("operator_mode", "pde_coupled")).strip().lower()
    runtime = resolve_deeponet_runtime(
        ctx,
        cfg,
        output_keys=deeponet_target_vars,
        sensor_feature_names=deeponet_feature_channels,
        operator_mode=operator_mode,
    )
    model = runtime.model
    deeponet_feature_source = "geom_feature_pack"
    deeponet_distance_transform_effective: dict[str, Any] = {"mode": "raw"}
    rows, source = _build_coord_feature_rows(
        channels=deeponet_feature_channels,
        pack=ctx.coord_feature_pack,
        geom_ctx=ctx.geom_ctx,
        h=rt.h,
        w=rt.w,
    )
    deeponet_feature_source = str(source)
    if source != "preprocess_pack":
        raise ValueError(
            "deeponet input-feature contract requires preprocessing coord_feature_pack; "
            f"effective_source={source}"
        )
    distance_transform_cfg = _resolve_distance_transform_cfg(
        dict(input_features_cfg.get("distance_transform") or {})
    )
    distance_transform_cfg_effective, _ = _resolve_distance_transform_effective(
        distance_transform_cfg,
        stats=ctx.coord_distance_transform_stats,
    )
    rows, deeponet_distance_transform_effective = _apply_distance_transform(
        rows.astype(np.float32),
        channels=deeponet_feature_channels,
        cfg=distance_transform_cfg_effective,
    )
    rows, _, _ = _apply_coord_feature_scaling(
        rows.astype(np.float32),
        channels=deeponet_feature_channels,
        coord_feature_scaler_artifact=ctx.coord_feature_scaler,
    )
    if hasattr(model, "set_static_spatial_features"):
        model.set_static_spatial_features(rows.astype(np.float32), channels=list(deeponet_feature_channels))
    ttrainer = TorchTrainer(ctx.model_dir / "train")
    stages = resolve_deeponet_stages(
        cfg,
        default_epochs=max(1, int(cfg.get("epochs", train_cfg.get("epochs", 2)))),
        default_lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
    )
    deeponet_y_train = ctx.y_scaled[ctx.tr][:, deeponet_target_indices]
    deeponet_y_val = ctx.y_scaled[ctx.va][:, deeponet_target_indices]
    out_t = ttrainer.run_deeponet(
        model=model,
        cond_train=ctx.cond_scaled[ctx.tr],
        y_train=deeponet_y_train,
        cond_val=ctx.cond_scaled[ctx.va],
        y_val=deeponet_y_val,
        geom_ctx=ctx.geom_ctx,
        y_vars=deeponet_target_vars,
        stages=stages,
        physics_cfg=ctx.physics_cfg,
        supervised_targets=runtime.supervised_targets,
        loss_cfg=rt.loss_cfg,
        curriculum_cfg=ctx.curriculum_cfg,
        supervised_mask=rt.supervised_mask,
        supervised_distance=rt.supervised_distance,
        optimizer_contract={**dict(rt.optimizer_contract), "deeponet_optimizer": dict(deeponet_optimizer_cfg)},
        selection_cfg=deeponet_selection_cfg,
        batch_size_cases=deeponet_batch_size_cases,
        shuffle_cases=deeponet_shuffle_cases,
        seed=ctx.global_seed + ctx.model_idx,
    )
    history = out_t.history
    steps_per_epoch = (
        int(np.ceil(len(ctx.tr) / max(1, deeponet_batch_size_cases))) if deeponet_batch_size_cases > 0 else 1
    )
    rt.extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
    pred = model.predict_fields(ctx.cond_scaled[ctx.te], geom_ctx=ctx.geom_ctx)
    pred_eval = ctx.transforms.inverse_field_dict({k: v for k, v in pred.items() if k in set(deeponet_target_vars)})
    if "rho_eff" in pred:
        pred_eval["rho_eff"] = np.asarray(pred["rho_eff"], dtype=np.float32)
    rt.extra_artifacts["supervised_targets_enabled"] = bool(runtime.supervised_targets is not None)
    _record_model_contract(
        rt.extra_artifacts,
        "deeponet",
        _build_deeponet_contract_effective(
            cfg=cfg,
            train_cfg=train_cfg,
            deeponet_target_family=deeponet_target_family,
            deeponet_target_vars=deeponet_target_vars,
            deeponet_input_mode=deeponet_input_mode,
            deeponet_feature_channels=deeponet_feature_channels,
            deeponet_feature_source=deeponet_feature_source,
            deeponet_selection_cfg=deeponet_selection_cfg,
            deeponet_optimizer_cfg=deeponet_optimizer_cfg,
            operator_mode=operator_mode,
            deeponet_distance_transform_effective=deeponet_distance_transform_effective,
            strict_mainline=strict_mainline,
        ),
    )
    return TrainLaneResult(
        model=model,
        history=history,
        pred_eval=pred_eval,
        true_eval=_to_true_eval(ctx.y, ctx.te, deeponet_target_vars, source_y_vars=ctx.y_vars),
        eval_vars=list(deeponet_target_vars),
    )


__all__ = [
    "DeeponetRuntime",
    "TrainDispatchContext",
    "TrainDispatchResult",
    "normalize_model_name",
    "resolve_runtime_model_adapter_policy",
    "resolve_deeponet_runtime",
    "run_model_train_predict",
    "validate_runtime_model_policy",
]
