"""Shared model train/predict dispatch for workflow and benchmark paths."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.density_contract import (
    resolve_allvars_order,
    resolve_family_vars,
)
from plasma_surrogate.core.model_families import (
    COORD_MLP_FAMILY_MODELS,
    GRID_TORCH_MODELS,
    MAINLINE_GEOM_PACK_MODELS,
    POD_DEEPONET_FAMILY_MODELS,
    SPECTRAL_FAMILY_MODELS,
    UNET_FAMILY_MODELS,
)
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
from plasma_surrogate.models.mlp.io import (
    build_model_from_name,
    resolve_deeponet_stages,
)
from plasma_surrogate.models.mlp.coord_mlp_torch import _normalize_coord_mlp_model_cfg
from plasma_surrogate.preprocessing.scalers import ScalerFactory
from plasma_surrogate.train.torch_trainer import TorchTrainer
from plasma_surrogate.train.trainer import Trainer

_ALLOWED_COORD_FEATURE_CHANNELS = (
    "x",
    "y",
    "distance_signed",
    "distance_any",
    "mask_plasma",
    "normal_x",
    "normal_y",
    "curvature_proxy",
)


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
    coord_feature_scaler: dict[str, Any] | None = None
    coord_distance_transform_stats: dict[str, Any] | None = None


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
class DeeponetRuntime:
    model: DeepONetPlasmaOperatorTorch
    supervised_targets: dict[str, np.ndarray] | None


def _to_true_eval(
    y: np.ndarray,
    idx: np.ndarray,
    y_vars: list[str],
    *,
    source_y_vars: list[str] | None = None,
) -> dict[str, np.ndarray]:
    source = list(source_y_vars or y_vars)
    pos = {name: i for i, name in enumerate(source)}
    return {name: y[idx, pos[name] : pos[name] + 1] for name in y_vars}


def _resolve_target_vars(raw: Any, *, available: list[str], cfg_key: str) -> list[str]:
    if raw is None:
        return list(available)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{cfg_key} must be a non-empty list")
    target = [str(v) for v in raw]
    unknown = [v for v in target if v not in set(available)]
    if unknown:
        raise ValueError(f"{cfg_key} contains unknown vars: {unknown}; available={available}")
    if len(set(target)) != len(target):
        raise ValueError(f"{cfg_key} must not contain duplicates")
    return target


def _resolve_unet_target_family(raw: Any, *, cfg_key: str) -> str:
    family = str(raw if raw is not None else "allvars").strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_key} must be allvars for mainline")
    return family


def _resolve_unet_target_vars_for_family(
    *,
    family: str,
    raw_target_vars: Any,
    available: list[str],
    cfg_key: str,
) -> list[str]:
    default = list(available)
    if raw_target_vars is None:
        return default
    target = _resolve_target_vars(raw_target_vars, available=available, cfg_key=cfg_key)
    if target != default:
        raise ValueError(f"{cfg_key} must match target_family=allvars: expected={default}, got={target}")
    return target


def _resolve_mainline_selection_weights(
    *,
    selection_cfg: dict[str, Any],
    target_vars: list[str],
    cfg_prefix: str,
) -> dict[str, float]:
    raw_weights = dict(selection_cfg.get("weights", {}))
    if len(raw_weights) == 0:
        uniform = 1.0 / float(max(len(target_vars), 1))
        return {str(name): float(uniform) for name in target_vars}
    expected = set(str(v) for v in target_vars)
    unknown = sorted(set(str(k) for k in raw_weights.keys()) - expected)
    missing = sorted(expected - set(str(k) for k in raw_weights.keys()))
    if unknown:
        raise ValueError(
            f"{cfg_prefix}.selection.weights contains unknown vars: {unknown}; expected={sorted(expected)}"
        )
    if missing:
        raise ValueError(
            f"{cfg_prefix}.selection.weights missing vars: {missing}; expected={sorted(expected)}"
        )
    out: dict[str, float] = {}
    total = 0.0
    for name in target_vars:
        w = float(raw_weights.get(name, 0.0))
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"{cfg_prefix}.selection.weights[{name}] must be finite and >= 0; got={w}")
        out[str(name)] = float(w)
        total += float(w)
    if total <= 0.0:
        raise ValueError(f"{cfg_prefix}.selection.weights must sum to > 0")
    return out


def _validate_unet_like_mainline_contract(
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
    model_key = str(model_name).strip().lower()
    if model_key not in GRID_TORCH_MODELS:
        raise ValueError(f"unsupported unet-like model for mainline validation: {model_name}")
    cfg_prefix = f"train.{model_key}"
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(
            f"{cfg_prefix}.target_family must be allvars for mainline; "
            f"got={target_family}"
        )
    expected = list(y_vars)
    if list(target_vars) != list(expected):
        raise ValueError(f"{cfg_prefix}.target_vars must match allvars order: expected={expected}, got={target_vars}")
    if require_shared_output_head:
        head_mode = str(dict(dict(model_cfg or {}).get("output_heads", {})).get("mode", "shared")).strip().lower()
        if head_mode != "shared":
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.mode must be shared for mainline")
    if model_key in MAINLINE_GEOM_PACK_MODELS:
        mode = str(input_features_mode or "").strip().lower()
        if mode != "geom_feature_pack":
            raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack for mainline")
        required_channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
        channels = [str(v) for v in list(input_feature_channels or [])]
        if channels != required_channels:
            raise ValueError(
                f"{cfg_prefix}.input_features.features must be {required_channels} for mainline; got={channels}"
            )
    sel_mode = str(selection_cfg.get("mode", "last")).strip().lower()
    if sel_mode != "best_val_allvars_balance":
        raise ValueError(
            f"{cfg_prefix}.selection.mode must be best_val_allvars_balance for mainline"
        )
    _resolve_mainline_selection_weights(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )
    if "density_guard" in selection_cfg:
        raise ValueError(f"{cfg_prefix}.selection.density_guard is removed from mainline")
    if "boundary_bonus_weight" in selection_cfg and float(selection_cfg.get("boundary_bonus_weight", 0.0)) != 0.0:
        raise ValueError(f"{cfg_prefix}.selection.boundary_bonus_weight must be 0.0 for mainline")
    sup_cfg = dict(loss_cfg.get("supervised", {}))
    for forbidden_key in ("density_positivity_penalty", "density_relative_weighting"):
        if forbidden_key in sup_cfg:
            raise ValueError(f"train.loss.supervised.{forbidden_key} is removed from {model_key} mainline")


def _validate_coord_mlp_experimental_contract(
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
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for coord-mlp experimental")
    expected = list(y_vars)
    if list(target_vars) != expected:
        raise ValueError(f"{cfg_prefix}.target_vars must match output_layout.vars order: expected={expected}, got={target_vars}")
    mode = str(dict(input_features_cfg).get("mode", "")).strip().lower()
    if mode != "geom_feature_pack":
        raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack")
    if len(set(str(v) for v in input_feature_channels)) != len(list(input_feature_channels)):
        raise ValueError(f"{cfg_prefix}.input_features.features must not contain duplicates")


def _validate_pod_deeponet_experimental_contract(
    *,
    y_vars: list[str],
    target_family: str,
    target_vars: list[str],
    model_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg_prefix = "train.deeponet_pod"
    family = str(target_family).strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for deeponet_pod experimental")
    expected = list(y_vars)
    if list(target_vars) != expected:
        raise ValueError(f"{cfg_prefix}.target_vars must match output_layout.vars order: expected={expected}, got={target_vars}")
    normalized_cfg = normalize_pod_deeponet_model_cfg(model_cfg, model_type="deeponet_pod")
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
    if selection_mode != "best_val_allvars_balance":
        raise ValueError(f"{cfg_prefix}.selection.mode must be best_val_allvars_balance")
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


def _grid_contract_warning_key(model_name: str) -> str:
    model_key = str(model_name).strip().lower()
    if model_key in COORD_MLP_FAMILY_MODELS:
        return "coord_mlp_contract_warnings"
    if model_key in SPECTRAL_FAMILY_MODELS:
        return "grid_contract_warnings"
    return "unet_contract_warnings"


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


def _resolve_family_train_cfg(train_cfg: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    cfg = dict(train_cfg.get(model_name, {}))
    if model_name in POD_DEEPONET_FAMILY_MODELS:
        cfg = resolve_deeponet_pod_training_defaults(cfg)
    if model_name == "coord_mlp_siren":
        cfg = _resolve_coord_mlp_siren_training_defaults(cfg)
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
    trunk_mode = str(model_cfg.get("trunk_input_mode", "legacy_xy_fourier")).strip().lower()
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
    trunk_fourier_mode = str(model_cfg.get("trunk_fourier_mode", "legacy")).strip().lower()
    if trunk_fourier_mode not in {"legacy", "symmetric"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_fourier_mode must be one of: legacy, symmetric")
    trunk_cond_modulation = str(model_cfg.get("trunk_cond_modulation", "none")).strip().lower()
    if trunk_cond_modulation not in {"none", "film"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.trunk_cond_modulation must be one of: none, film")
    sel_mode = str(selection_cfg.get("mode", "last")).strip().lower()
    if sel_mode != "best_val_allvars_balance":
        raise ValueError(f"{cfg_prefix}.selection.mode must be best_val_allvars_balance for mainline")
    if "boundary_bonus_weight" in selection_cfg and float(selection_cfg.get("boundary_bonus_weight", 0.0)) != 0.0:
        raise ValueError(f"{cfg_prefix}.selection.boundary_bonus_weight must be 0.0 for mainline")
    if "density_guard" in selection_cfg:
        raise ValueError(f"{cfg_prefix}.selection.density_guard is removed from deeponet mainline")
    _resolve_mainline_selection_weights(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )
    sup_cfg = dict(loss_cfg.get("supervised", {}))
    for forbidden_key in ("density_positivity_penalty", "density_relative_weighting"):
        if forbidden_key in sup_cfg:
            raise ValueError(f"train.loss.supervised.{forbidden_key} is removed from deeponet mainline")


def _build_spectral_contract_effective(
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
        f"{prefix}_feature_contract_effective": {
            "input_features_mode": str(input_features_mode),
            "input_feature_channels": list(input_feature_channels),
            "distance_transform_mode": str(
                dict(dict(cfg.get("input_features", {})).get("distance_transform", {})).get("mode", "raw")
            ).strip().lower(),
        },
        f"{prefix}_spectral_contract_effective": {
            "n_modes": int(dict(cfg.get("model_cfg", {})).get("n_modes", dict(cfg.get("model_cfg", {})).get("fno_n_modes", 2))),
            "dealias_ratio": float(spectral_cfg.get("dealias_ratio", 1.0)),
            "taper_alpha": float(spectral_cfg.get("taper_alpha", 0.0)),
            "skip_filter": str(spectral_cfg.get("skip_filter", "none")).strip().lower(),
        },
        "target_family_effective": str(target_family),
        "target_vars_effective": list(target_vars),
        f"{prefix}_spatial_consistency_effective": bool(
            dict((loss_cfg or {}).get("supervised", {}).get("spatial_consistency", {})).get("enabled", False)
        ),
    }
    factorized_cfg = dict(spectral_cfg.get("factorized_cfg", {}))
    if prefix == "ffno":
        out[f"{prefix}_spectral_contract_effective"]["factorized_cfg"] = {
            "enabled": bool(factorized_cfg.get("enabled", True)),
            "mode": str(factorized_cfg.get("mode", "separable_1d")).strip().lower(),
            "share_weights": bool(factorized_cfg.get("share_weights", False)),
        }
        local_skip_cfg = dict(spectral_cfg.get("local_skip_cfg", {}))
        out[f"{prefix}_spectral_contract_effective"]["local_skip_cfg"] = {
            "enabled": bool(local_skip_cfg.get("enabled", False)),
            "init_scale": float(local_skip_cfg.get("init_scale", 0.0)),
        }
        axis_mix_cfg = dict(spectral_cfg.get("axis_mix_cfg", {}))
        out[f"{prefix}_spectral_contract_effective"]["axis_mix_cfg"] = {
            "enabled": bool(axis_mix_cfg.get("enabled", False)),
            "init_h": float(axis_mix_cfg.get("init_h", 1.0)),
            "init_w": float(axis_mix_cfg.get("init_w", 1.0)),
        }
    return out


def _coord_xy_rows_from_geom(geom_ctx: Any, *, h: int, w: int) -> tuple[np.ndarray, str]:
    if geom_ctx is None:
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1), "normalized"
    raw_obj = getattr(geom_ctx, "coord_grid", None)
    if raw_obj is None:
        raise ValueError("coord_source=geom_ctx requires geom_ctx.coord_grid")
    raw_grid = np.asarray(raw_obj, dtype=np.float32)
    if raw_grid.shape == (2, h, w):
        return np.stack([raw_grid[0].reshape(-1), raw_grid[1].reshape(-1)], axis=1).astype(np.float32), "geom_ctx"
    if raw_grid.shape == (h, w, 2):
        return raw_grid.reshape(-1, 2).astype(np.float32), "geom_ctx"
    raise ValueError(f"geom_ctx.coord_grid shape mismatch: got {raw_grid.shape}, expected (2,{h},{w})")


def _resolve_coord_feature_channels(raw: Any) -> list[str]:
    if raw is None:
        return ["x", "y", "distance_signed", "distance_any", "mask_plasma"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("train.input_features.features must be a non-empty list")
    channels = [str(v) for v in raw]
    unknown = [v for v in channels if v not in _ALLOWED_COORD_FEATURE_CHANNELS]
    if unknown:
        raise ValueError(
            "train.input_features.features contains unsupported channels: "
            f"{unknown}; allowed={list(_ALLOWED_COORD_FEATURE_CHANNELS)}"
        )
    if len(set(channels)) != len(channels):
        raise ValueError("train.input_features.features must not contain duplicates")
    return channels


def _derive_geom_feature_maps(
    *,
    coord_xy: np.ndarray,
    distance_signed: np.ndarray,
    distance_any: np.ndarray,
    mask_plasma: np.ndarray,
    h: int,
    w: int,
) -> dict[str, np.ndarray]:
    signed_2d = np.asarray(distance_signed, dtype=np.float32).reshape(h, w)
    gy, gx = np.gradient(signed_2d, edge_order=1)
    gnorm = np.sqrt(gx**2 + gy**2).astype(np.float32)
    gnorm = np.where(gnorm < 1e-6, 1e-6, gnorm).astype(np.float32)
    normal_x = (gx / gnorm).reshape(-1).astype(np.float32)
    normal_y = (gy / gnorm).reshape(-1).astype(np.float32)
    dnx_dy, dnx_dx = np.gradient(gx / gnorm, edge_order=1)
    dny_dy, dny_dx = np.gradient(gy / gnorm, edge_order=1)
    curvature_proxy = (dnx_dx + dny_dy).reshape(-1).astype(np.float32)
    return {
        "x": coord_xy[:, 0],
        "y": coord_xy[:, 1],
        "distance_signed": np.asarray(distance_signed, dtype=np.float32).reshape(-1),
        "distance_any": np.asarray(distance_any, dtype=np.float32).reshape(-1),
        "mask_plasma": np.asarray(mask_plasma, dtype=np.float32).reshape(-1),
        "normal_x": normal_x,
        "normal_y": normal_y,
        "curvature_proxy": curvature_proxy,
    }


def _apply_coord_feature_scaling(
    rows: np.ndarray,
    *,
    channels: list[str],
    coord_feature_scaler_artifact: dict[str, Any] | None,
) -> tuple[np.ndarray, str, bool]:
    raw = dict(coord_feature_scaler_artifact or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return np.asarray(rows, dtype=np.float32), "none", False
    ch_scalers = dict(raw.get("channels", {}))
    if len(ch_scalers) == 0:
        return np.asarray(rows, dtype=np.float32), "none", False
    out = np.asarray(rows, dtype=np.float32).copy()
    for i, name in enumerate(channels):
        payload = ch_scalers.get(name)
        if not isinstance(payload, dict) or len(payload) == 0:
            continue
        scaler = ScalerFactory.from_dict(payload)
        out[:, i : i + 1] = scaler.transform(out[:, i : i + 1]).astype(np.float32)
    return out.astype(np.float32), str(raw.get("mode", "custom")), True


def _resolve_distance_transform_cfg(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    mode = str(cfg.get("mode", "raw")).strip().lower()
    if mode not in {"raw", "bounded", "bounded_auto"}:
        raise ValueError(
            "train.input_features.distance_transform.mode must be one of: raw, bounded, bounded_auto"
        )
    signed_tanh_tau = float(cfg.get("signed_tanh_tau", 8.0))
    proximity_tau = float(cfg.get("proximity_tau", 6.0))
    if signed_tanh_tau <= 0.0:
        raise ValueError("distance_transform.signed_tanh_tau must be > 0")
    if proximity_tau <= 0.0:
        raise ValueError("distance_transform.proximity_tau must be > 0")
    return {
        "mode": mode,
        "signed_tanh_tau": signed_tanh_tau,
        "proximity_tau": proximity_tau,
        "replace_distance_any": bool(cfg.get("replace_distance_any", True)),
    }


def _resolve_distance_transform_effective(
    cfg: dict[str, Any],
    *,
    stats: dict[str, Any] | None,
    warnings_out: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    mode = str(cfg.get("mode", "raw")).strip().lower()
    out = dict(cfg)
    if mode != "bounded_auto":
        return out, "config"
    raw_stats = dict(stats or {})
    s_tau = float(raw_stats.get("signed_tanh_tau_auto", 0.0))
    p_tau = float(raw_stats.get("proximity_tau_auto", 0.0))
    if s_tau > 0.0 and p_tau > 0.0:
        out["signed_tanh_tau"] = s_tau
        out["proximity_tau"] = p_tau
        out["signed_quantile"] = float(raw_stats.get("signed_quantile", 0.75))
        out["proximity_quantile"] = float(raw_stats.get("proximity_quantile", 0.50))
        return out, "artifact"
    if warnings_out is not None:
        warnings_out.append(
            "bounded_auto fallback: preprocessing/scalers/distance_transform_stats.json is missing "
            "or invalid; using configured tau values"
        )
    return out, "fallback_default"


def _apply_distance_transform(
    rows: np.ndarray,
    *,
    channels: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(rows, dtype=np.float32).copy()
    mode = str(cfg.get("mode", "raw")).strip().lower()
    effective = {
        "mode": mode,
        "signed_tanh_tau": float(cfg.get("signed_tanh_tau", 8.0)),
        "proximity_tau": float(cfg.get("proximity_tau", 6.0)),
        "replace_distance_any": bool(cfg.get("replace_distance_any", True)),
        "signed_quantile": float(cfg.get("signed_quantile", 0.75)),
        "proximity_quantile": float(cfg.get("proximity_quantile", 0.50)),
    }
    if mode == "raw":
        return out, effective

    signed_tau = max(float(effective["signed_tanh_tau"]), 1e-6)
    prox_tau = max(float(effective["proximity_tau"]), 1e-6)
    if "distance_signed" in channels:
        idx = channels.index("distance_signed")
        out[:, idx] = np.tanh(out[:, idx] / signed_tau).astype(np.float32)
    if "distance_any" in channels and bool(effective["replace_distance_any"]):
        idx = channels.index("distance_any")
        out[:, idx] = np.exp(-np.maximum(out[:, idx], 0.0) / prox_tau).astype(np.float32)
    return out.astype(np.float32), effective


def _build_coord_feature_rows(
    *,
    channels: list[str],
    pack: dict[str, Any] | None,
    geom_ctx: Any,
    h: int,
    w: int,
) -> tuple[np.ndarray, str]:
    if pack:
        data = np.asarray(pack.get("data"), dtype=np.float32) if "data" in pack else None
        raw_channels = pack.get("channels")
        if data is not None and raw_channels is not None:
            pack_channels = [str(v) for v in np.asarray(raw_channels).reshape(-1).tolist()]
            if data.ndim == 3 and data.shape[1:] == (h, w):
                pack_map = {name: data[i] for i, name in enumerate(pack_channels) if i < data.shape[0]}
                if all(name in pack_map for name in channels):
                    stacked = np.stack([pack_map[name] for name in channels], axis=0).astype(np.float32)
                    return stacked.reshape(len(channels), -1).T.astype(np.float32), "preprocess_pack"
    coord_xy, _ = _coord_xy_rows_from_geom(geom_ctx, h=h, w=w)
    if geom_ctx is None:
        distance_any = np.ones((h * w,), dtype=np.float32)
        distance_signed = np.ones((h * w,), dtype=np.float32)
        mask_plasma = np.ones((h * w,), dtype=np.float32)
    else:
        distance_any = np.asarray(getattr(geom_ctx, "distance_any"), dtype=np.float32).reshape(-1)
        raw_signed = getattr(geom_ctx, "distance_signed", None)
        if raw_signed is None:
            mask_tmp = np.asarray(getattr(geom_ctx, "mask_plasma"), dtype=np.float32).reshape(-1)
            distance_signed = np.where(mask_tmp > 0.5, distance_any, -distance_any).astype(np.float32)
        else:
            distance_signed = np.asarray(raw_signed, dtype=np.float32).reshape(-1)
        mask_plasma = np.asarray(getattr(geom_ctx, "mask_plasma"), dtype=np.float32).reshape(-1)
    mapping = _derive_geom_feature_maps(
        coord_xy=coord_xy,
        distance_signed=distance_signed,
        distance_any=distance_any,
        mask_plasma=mask_plasma,
        h=h,
        w=w,
    )
    feats = np.stack([np.asarray(mapping[name], dtype=np.float32) for name in channels], axis=1).astype(np.float32)
    return feats, "runtime_geom"


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
            trunk_input_mode=str(model_cfg.get("trunk_input_mode", "legacy_xy_fourier")),
            sensor_pool_mode=str(model_cfg.get("sensor_pool_mode", "moments")),
            sensor_embed_dim=int(model_cfg.get("sensor_embed_dim", 32)),
            branch_mode=branch_mode,
            trunk_fourier_n_freq=int(model_cfg.get("trunk_fourier_n_freq", 1)),
            trunk_fourier_mode=str(model_cfg.get("trunk_fourier_mode", "legacy")),
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
            output_path_fused_hidden_dim=int(output_path_cfg.get("fused_hidden_dim", 96)),
            output_path_global_local_enabled=bool(output_path_cfg.get("global_local", {}).get("enabled", False)),
            output_path_global_hidden_dim=int(output_path_cfg.get("global_hidden_dim", 64)),
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
        trunk_input_mode=str(model_cfg.get("trunk_input_mode", "legacy_xy_fourier")),
        sensor_pool_mode=str(model_cfg.get("sensor_pool_mode", "moments")),
        sensor_embed_dim=int(model_cfg.get("sensor_embed_dim", 32)),
        branch_mode=branch_mode,
        trunk_fourier_n_freq=int(model_cfg.get("trunk_fourier_n_freq", 1)),
        trunk_fourier_mode=str(model_cfg.get("trunk_fourier_mode", "legacy")),
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
        output_path_fused_hidden_dim=int(output_path_cfg.get("fused_hidden_dim", 96)),
        output_path_global_local_enabled=bool(output_path_cfg.get("global_local", {}).get("enabled", False)),
        output_path_global_hidden_dim=int(output_path_cfg.get("global_hidden_dim", 64)),
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
        trunk_input_mode=str(poisson_cfg.get("trunk_input_mode", model_cfg.get("trunk_input_mode", "legacy_xy_fourier"))),
        sensor_pool_mode=str(poisson_cfg.get("sensor_pool_mode", model_cfg.get("sensor_pool_mode", "moments"))),
        sensor_embed_dim=int(poisson_cfg.get("sensor_embed_dim", model_cfg.get("sensor_embed_dim", 32))),
        branch_mode=str(poisson_cfg.get("branch_mode", model_cfg.get("branch_mode", "moments"))),
        trunk_fourier_n_freq=int(poisson_cfg.get("trunk_fourier_n_freq", model_cfg.get("trunk_fourier_n_freq", 1))),
        trunk_fourier_mode=str(poisson_cfg.get("trunk_fourier_mode", model_cfg.get("trunk_fourier_mode", "legacy"))),
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
    trainer = Trainer(ctx.model_dir / "train")
    train_cfg = dict(ctx.run_cfg.get("train", {}))
    optimizer_contract = dict(train_cfg.get("optimizer_contract", {}))
    model_name = ctx.model_name
    h, w = ctx.h, ctx.w
    history: list[dict[str, float]] = []
    extra_artifacts: dict[str, Any] = {}
    eval_vars = list(ctx.y_vars)

    if model_name == "global_mlp":
        cfg = dict(train_cfg.get("global_mlp", {}))
        batch_size_cases = int(cfg.get("batch_size_cases", 0))
        shuffle_cases = bool(cfg.get("shuffle_cases", True))
        grad_scale_cfg = dict(cfg.get("grad_scale", {}))
        layer_lr_multiplier = dict(cfg.get("layer_lr_multiplier", {}))
        grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
        grad_clip_cfg = dict(cfg.get("grad_clip", {}))
        output_head_refresh_cfg = dict(cfg.get("output_head_refresh", {}))
        global_loss_cfg = copy.deepcopy(ctx.loss_cfg or {})
        sup = dict(global_loss_cfg.get("supervised", {}))
        sup.setdefault("global_target_region", "plasma_only")
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
            grid_shape=(h, w),
            model_cfg=dict(cfg.get("model_cfg", {})),
            seed=ctx.global_seed + ctx.model_idx,
            phi_mode=str(ctx.profile_lock.get("phi_mode", "direct")),
            out_channels=len(ctx.y_vars),
            output_keys=ctx.y_vars,
        )
        y_flat = ctx.y_scaled.reshape(ctx.n_cases, -1)
        out = trainer.run_global(
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
            supervised_mask=ctx.supervised_mask,
            supervised_distance=ctx.supervised_distance,
            batch_size_cases=batch_size_cases,
            shuffle_cases=shuffle_cases,
            seed=ctx.global_seed + ctx.model_idx,
            grad_scale_cfg=grad_scale_cfg,
            layer_lr_multiplier=layer_lr_multiplier,
            grad_clip_norm=grad_clip_norm,
            grad_clip_cfg=grad_clip_cfg,
            output_head_refresh_cfg=output_head_refresh_cfg,
        )
        history = out.history
        steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, batch_size_cases))) if batch_size_cases > 0 else 1
        extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
        pred_eval = ctx.transforms.inverse_field_dict(model.predict_fields(ctx.cond_scaled[ctx.te]))
        true_eval = _to_true_eval(ctx.y, ctx.te, ctx.y_vars)
    elif model_name in POD_DEEPONET_FAMILY_MODELS:
        cfg = _resolve_family_train_cfg(train_cfg, model_name=model_name)
        train_key = "train.deeponet_pod"
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
            y_vars=ctx.y_vars,
            target_family=pod_target_family,
            target_vars=pod_target_vars,
            model_cfg=dict(cfg.get("model_cfg", {})),
            selection_cfg=dict(cfg.get("selection", {})),
        )
        pod_target_indices = [ctx.y_vars.index(v) for v in pod_target_vars]
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
            model_type="deeponet_pod",
        )
        model = build_model_from_name(
            model_name=model_name,
            input_dim=ctx.cond_scaled.shape[1],
            grid_shape=(h, w),
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
        out = trainer.run_unet(
            model,
            ctx.cond_scaled[ctx.tr],
            pod_y_train,
            ctx.cond_scaled[ctx.va],
            np.asarray(ctx.y_scaled[ctx.va][:, pod_target_indices], dtype=np.float32),
            epochs=int(cfg.get("epochs", train_cfg.get("epochs", 20))),
            lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
            physics_cfg={"enabled": False},
            loss_cfg=copy.deepcopy(ctx.loss_cfg or {}),
            curriculum_cfg=ctx.curriculum_cfg,
            supervised_mask=ctx.supervised_mask,
            supervised_distance=ctx.supervised_distance,
            optimizer_contract=optimizer_contract,
            unet_optimizer_cfg=pod_optimizer_cfg,
            batch_size_cases=batch_size_cases,
            shuffle_cases=shuffle_cases,
            seed=ctx.global_seed + ctx.model_idx,
            selection_cfg=pod_selection_cfg,
        )
        history = out.history
        steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, batch_size_cases))) if batch_size_cases > 0 else 1
        extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
        extra_artifacts["deeponet_pod_contract_effective"] = {
            "model_type_effective": "deeponet_pod",
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
        }
        pred_features = model.forward_features(ctx.cond_scaled[ctx.te])
        pred_eval = ctx.transforms.inverse_field_dict(
            {name: np.asarray(pred_features[name], dtype=np.float32) for name in pod_target_vars}
        )
        true_eval = _to_true_eval(ctx.y, ctx.te, pod_target_vars, source_y_vars=ctx.y_vars)
        eval_vars = list(pod_target_vars)
    elif model_name in GRID_TORCH_MODELS:
        cfg = _resolve_family_train_cfg(train_cfg, model_name=model_name)
        grid_like_cfg = dict(train_cfg.get("unet_like", {}))
        grid_loss_cfg = copy.deepcopy(ctx.loss_cfg or {})
        grid_batch_size_cases = int(grid_like_cfg.get("batch_size_cases", 0))
        grid_shuffle_cases = bool(grid_like_cfg.get("shuffle_cases", True))
        grid_feature_channels = ["x", "y"]
        grid_input_features_mode = "legacy_xy"
        grid_selection_cfg: dict[str, Any] = {}
        grid_optimizer_cfg: dict[str, Any] = {}
        grid_backend_effective = str(dict(cfg.get("model_cfg", cfg)).get("backend", "numpy")).strip().lower()
        grid_target_family = "allvars"
        grid_target_vars = list(ctx.y_vars)
        grid_target_indices = list(range(len(ctx.y_vars)))
        grid_supervised_distance_signed = None
        grid_supervised_bc_dir_mask = None
        grid_supervised_wafer_mask = None
        if model_name in GRID_TORCH_MODELS:
            train_key = f"train.{model_name}"
            grid_target_family = _resolve_unet_target_family(
                cfg.get("target_family", "allvars"),
                cfg_key=f"{train_key}.target_family",
            )
            grid_target_vars = _resolve_unet_target_vars_for_family(
                family=grid_target_family,
                raw_target_vars=cfg.get("target_vars"),
                available=ctx.y_vars,
                cfg_key=f"{train_key}.target_vars",
            )
            grid_target_indices = [ctx.y_vars.index(v) for v in grid_target_vars]
            input_features_cfg = dict(cfg.get("input_features", {}))
            default_input_mode = "legacy_xy" if model_name == "unet" else "geom_feature_pack"
            grid_input_features_mode = str(input_features_cfg.get("mode", default_input_mode)).strip().lower()
            if grid_input_features_mode not in {"legacy_xy", "geom_feature_pack"}:
                raise ValueError(f"{train_key}.input_features.mode must be one of: legacy_xy, geom_feature_pack")
            grid_feature_channels = _resolve_coord_feature_channels(input_features_cfg.get("features"))
            if grid_input_features_mode == "legacy_xy":
                grid_feature_channels = ["x", "y"]
            grid_selection_cfg = dict(cfg.get("selection", {}))
            if model_name in MAINLINE_GEOM_PACK_MODELS:
                _validate_unet_like_mainline_contract(
                    model_name=model_name,
                    model_cfg=dict(cfg.get("model_cfg", {})),
                    selection_cfg=grid_selection_cfg,
                    loss_cfg=grid_loss_cfg,
                    y_vars=ctx.y_vars,
                    target_family=grid_target_family,
                    target_vars=grid_target_vars,
                    require_shared_output_head=True,
                    input_features_mode=grid_input_features_mode,
                    input_feature_channels=grid_feature_channels,
                )
            elif model_name in COORD_MLP_FAMILY_MODELS:
                _validate_coord_mlp_experimental_contract(
                    model_name=model_name,
                    y_vars=ctx.y_vars,
                    target_family=grid_target_family,
                    target_vars=grid_target_vars,
                    input_features_cfg=input_features_cfg,
                    input_feature_channels=grid_feature_channels,
                )
            grid_optimizer_cfg = dict(cfg.get("optimizer", {}))
            if model_name in UNET_FAMILY_MODELS:
                field_contract_mode = str(dict(cfg.get("field_contract", {})).get("mode", "plain")).strip().lower()
                if field_contract_mode != "plain":
                    raise ValueError(f"{train_key}.field_contract.mode=frozen_global_residual is removed in S2.45")
                supervised_cfg = dict((grid_loss_cfg or {}).get("supervised", {}))
                if bool(dict(supervised_cfg.get("chamber_aux", {})).get("enabled", False)):
                    raise ValueError(
                        f"train.loss.supervised.chamber_aux is not supported for {model_name} mainline; use region_balance"
                    )
                _validate_unet_like_mainline_contract(
                    model_name=model_name,
                    model_cfg=dict(cfg.get("model_cfg", {})),
                    selection_cfg=grid_selection_cfg,
                    loss_cfg=grid_loss_cfg,
                    y_vars=ctx.y_vars,
                    target_family=grid_target_family,
                    target_vars=grid_target_vars,
                    require_shared_output_head=True,
                    input_features_mode=grid_input_features_mode,
                    input_feature_channels=grid_feature_channels,
                )
            if model_name not in COORD_MLP_FAMILY_MODELS:
                grid_selection_cfg["weights"] = _resolve_mainline_selection_weights(
                    selection_cfg=grid_selection_cfg,
                    target_vars=grid_target_vars,
                    cfg_prefix=train_key,
                )
            # Spatial-consistency normalization uses train y-scaler stats when enabled.
            sc_cfg = dict(dict(grid_loss_cfg.get("supervised", {})).get("spatial_consistency", {}))
            if bool(sc_cfg.get("enabled", False)):
                scale_by_var = dict(sc_cfg.get("scale_by_var", {}))
                for name in grid_target_vars:
                    scaler_obj = dict(getattr(ctx.transforms, "y_scalers", {}) or {}).get(name, None)
                    if scaler_obj is None:
                        continue
                    try:
                        scaler_dict = scaler_obj.to_dict()
                    except Exception:
                        scaler_dict = {}
                    std_raw = scaler_dict.get("std", None)
                    if isinstance(std_raw, list) and std_raw:
                        std_val = float(std_raw[0])
                    elif std_raw is not None:
                        std_val = float(std_raw)
                    else:
                        std_val = 1.0
                    if np.isfinite(std_val) and std_val > 0.0:
                        scale_by_var[str(name)] = float(std_val)
                sc_cfg["scale_by_var"] = scale_by_var
                sup_cfg_mut = dict(grid_loss_cfg.get("supervised", {}))
                sup_cfg_mut["spatial_consistency"] = sc_cfg
                grid_loss_cfg["supervised"] = sup_cfg_mut
        if model_name in MAINLINE_GEOM_PACK_MODELS or model_name in COORD_MLP_FAMILY_MODELS:
            grid_backend_effective = "torch"
        model = build_model_from_name(
            model_name=model_name,
            input_dim=ctx.cond_scaled.shape[1],
            grid_shape=(h, w),
            model_cfg=dict(cfg.get("model_cfg", cfg)),
            seed=ctx.global_seed + ctx.model_idx,
            phi_mode=str(ctx.profile_lock.get("phi_mode", "direct")),
            out_channels=len(grid_target_vars),
            output_keys=grid_target_vars,
            unet_feature_channels=grid_feature_channels,
        )
        if model_name in GRID_TORCH_MODELS:
            require_pack = str(dict(cfg.get("input_features", {})).get("require_pack", "warn")).strip().lower()
            if require_pack not in {"off", "warn", "error"}:
                raise ValueError(f"train.{model_name}.input_features.require_pack must be one of: off, warn, error")
            warning_bucket = extra_artifacts.setdefault(_grid_contract_warning_key(model_name), [])
            grid_feature_source = "legacy_xy"
            if grid_input_features_mode == "geom_feature_pack":
                rows, source = _build_coord_feature_rows(
                    channels=grid_feature_channels,
                    pack=ctx.coord_feature_pack,
                    geom_ctx=ctx.geom_ctx,
                    h=h,
                    w=w,
                )
                grid_feature_source = str(source)
                if source != "preprocess_pack":
                    msg = (
                        f"{model_name} input-feature contract: preprocess coord_feature_pack was not used; "
                        f"effective_source={source}"
                    )
                    if model_name in COORD_MLP_FAMILY_MODELS or require_pack == "error":
                        raise ValueError(msg)
                    if require_pack == "warn":
                        warning_bucket.append(msg)
                distance_transform_cfg = _resolve_distance_transform_cfg(
                    dict(dict(cfg.get("input_features", {})).get("distance_transform") or {})
                )
                distance_transform_cfg_effective, _ = _resolve_distance_transform_effective(
                    distance_transform_cfg,
                    stats=ctx.coord_distance_transform_stats,
                    warnings_out=warning_bucket,
                )
                rows, _ = _apply_distance_transform(
                    rows.astype(np.float32),
                    channels=grid_feature_channels,
                    cfg=distance_transform_cfg_effective,
                )
                rows, _, scaling_applied = _apply_coord_feature_scaling(
                    rows.astype(np.float32),
                    channels=grid_feature_channels,
                    coord_feature_scaler_artifact=ctx.coord_feature_scaler,
                )
                if model_name in COORD_MLP_FAMILY_MODELS and not bool(scaling_applied):
                    raise ValueError(
                        f"train.{model_name} requires preprocessing.coord_features.scaling.enabled=true"
                    )
            else:
                yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
                xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
                yv, xv = np.meshgrid(yy, xx, indexing="ij")
                rows = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)
            spatial = rows.reshape(h, w, len(grid_feature_channels)).astype(np.float32)
            if hasattr(model, "set_static_spatial_features"):
                model.set_static_spatial_features(spatial)
            if ctx.geom_ctx is not None:
                raw_signed = getattr(ctx.geom_ctx, "distance_signed", None)
                if raw_signed is not None:
                    grid_supervised_distance_signed = np.asarray(raw_signed, dtype=np.float32)
                raw_bc_dir = getattr(ctx.geom_ctx, "bc_dir_mask", None)
                if raw_bc_dir is not None:
                    grid_supervised_bc_dir_mask = np.asarray(raw_bc_dir, dtype=np.float32)
                raw_wafer = getattr(ctx.geom_ctx, "regions", {}).get("wafer_mask") if hasattr(ctx.geom_ctx, "regions") else None
                if raw_wafer is not None:
                    grid_supervised_wafer_mask = np.asarray(raw_wafer, dtype=np.float32)
        unet_y_train = ctx.y_scaled[ctx.tr][:, grid_target_indices]
        unet_y_val = ctx.y_scaled[ctx.va][:, grid_target_indices]
        out = trainer.run_unet(
            model,
            ctx.cond_scaled[ctx.tr],
            unet_y_train,
            ctx.cond_scaled[ctx.va],
            unet_y_val,
            epochs=int(cfg.get("epochs", train_cfg.get("epochs", 20))),
            lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
            physics_cfg=ctx.physics_cfg,
            loss_cfg=grid_loss_cfg if model_name in GRID_TORCH_MODELS else ctx.loss_cfg,
            curriculum_cfg=ctx.curriculum_cfg,
            supervised_mask=ctx.supervised_mask,
            supervised_distance=ctx.supervised_distance,
            supervised_distance_signed=grid_supervised_distance_signed if model_name in GRID_TORCH_MODELS else None,
            supervised_bc_dir_mask=grid_supervised_bc_dir_mask if model_name in GRID_TORCH_MODELS else None,
            supervised_wafer_mask=grid_supervised_wafer_mask if model_name in GRID_TORCH_MODELS else None,
            optimizer_contract=optimizer_contract,
            unet_optimizer_cfg=grid_optimizer_cfg if model_name in GRID_TORCH_MODELS else None,
            batch_size_cases=grid_batch_size_cases,
            shuffle_cases=grid_shuffle_cases,
            seed=ctx.global_seed + ctx.model_idx,
            selection_cfg=grid_selection_cfg if model_name in GRID_TORCH_MODELS else None,
            selection_target_override=None,
            selection_pred_additive=None,
        )
        history = out.history
        steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, grid_batch_size_cases))) if grid_batch_size_cases > 0 else 1
        extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
        if model_name in UNET_FAMILY_MODELS:
            supervised_cfg = dict((grid_loss_cfg or {}).get("supervised", {}))
            bt_cfg = dict(supervised_cfg.get("boundary_type_weighting", {}))
            bp_cfg = dict(supervised_cfg.get("boundary_profile_weighting", {}))
            rb_cfg = dict(supervised_cfg.get("region_balance", {}))
            rb_schedule_cfg = dict(rb_cfg.get("schedule", {}))
            extra_artifacts["unet_contract_effective"] = {
                "unet_backend_effective": grid_backend_effective,
                "unet_input_channels_effective": list(grid_feature_channels),
                "unet_selection_mode_effective": str(grid_selection_cfg.get("mode", "last")).strip().lower(),
                "selection_weights_effective": dict(grid_selection_cfg.get("weights", {})),
                "boundary_bonus_weight_effective": float(grid_selection_cfg.get("boundary_bonus_weight", 0.0)),
                "unet_optimizer_effective": {
                    "type": str(grid_optimizer_cfg.get("type", "adamw")).strip().lower() if grid_backend_effective == "torch" else "none",
                    "lr": float(grid_optimizer_cfg.get("lr", cfg.get("lr", train_cfg.get("lr", 1e-3)))) if grid_backend_effective == "torch" else float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
                    "schedule": str(grid_optimizer_cfg.get("schedule", "none")).strip().lower() if grid_backend_effective == "torch" else "none",
                    "warmup_epochs": int(grid_optimizer_cfg.get("warmup_epochs", 0)) if grid_backend_effective == "torch" else 0,
                },
                "unet_output_heads_mode_effective": str(getattr(model, "output_heads_mode", "shared")).strip().lower(),
                "unet_feature_contract_effective": {
                    "input_features_mode": str(grid_input_features_mode),
                    "input_feature_channels": list(grid_feature_channels),
                    "upsample_mode": str(
                        dict(dict(cfg.get("model_cfg", {})).get("conv_cfg", {})).get(
                            "upsample_mode",
                            dict(cfg.get("model_cfg", {})).get("upsample_mode", "deconv"),
                        )
                    ).strip().lower(),
                    "distance_transform_mode": str(
                        dict(dict(cfg.get("input_features", {})).get("distance_transform", {})).get("mode", "raw")
                    ).strip().lower(),
                },
                "merge_role": "single",
                "target_family_effective": str(grid_target_family),
                "target_vars_effective": list(grid_target_vars),
                "unet_spatial_consistency_effective": bool(
                    dict((grid_loss_cfg or {}).get("supervised", {}).get("spatial_consistency", {})).get("enabled", False)
                ),
                "boundary_type_weighting_effective": {
                    "enabled": bool(bt_cfg.get("enabled", False)),
                    "vars": [str(v) for v in bt_cfg.get("vars", ["Te", "phi"])],
                    "band_px": float(bt_cfg.get("band_px", 2.0)),
                    "weights": {
                        "interface": float(dict(bt_cfg.get("weights", {})).get("interface", 1.0)),
                        "bc_dir": float(dict(bt_cfg.get("weights", {})).get("bc_dir", 1.0)),
                        "wafer": float(dict(bt_cfg.get("weights", {})).get("wafer", 1.0)),
                    },
                },
                "boundary_profile_weighting_effective": {
                    "enabled": bool(bp_cfg.get("enabled", False)),
                    "vars": [str(v) for v in bp_cfg.get("vars", ["Te", "phi"])],
                    "band_px": float(bp_cfg.get("band_px", 2.0)),
                    "mode": str(bp_cfg.get("mode", "exp_decay")).strip().lower(),
                    "alpha": float(bp_cfg.get("alpha", 0.35)),
                    "tau_px": float(bp_cfg.get("tau_px", 0.8)),
                },
                "region_balance_bands_effective": {
                    "enabled": bool(rb_cfg.get("enabled", False)),
                    "boundary_in_px": float(rb_cfg.get("boundary_in_px", 2.0)),
                    "mid_plasma_px": float(rb_cfg.get("mid_plasma_px", rb_cfg.get("deep_plasma_px", 10.0))),
                    "deep_plasma_px": float(rb_cfg.get("deep_plasma_px", 10.0)),
                    "weight_boundary_in": float(rb_cfg.get("weight_boundary_in", 0.6)),
                    "weight_plasma_mid": float(rb_cfg.get("weight_plasma_mid", 0.0)),
                    "weight_deep_plasma": float(rb_cfg.get("weight_deep_plasma", 0.4)),
                },
                "region_balance_schedule_effective": {
                    "enabled": bool(rb_schedule_cfg.get("enabled", False)),
                    "warmup_epochs": int(max(int(rb_schedule_cfg.get("warmup_epochs", 0)), 0)),
                    "ramp_epochs": int(max(int(rb_schedule_cfg.get("ramp_epochs", 0)), 0)),
                },
            }
            eval_vars = list(grid_target_vars)
        elif model_name in SPECTRAL_FAMILY_MODELS:
            prefix = "fno" if model_name == "fno" else "ffno"
            extra_artifacts[f"{prefix}_contract_effective"] = _build_spectral_contract_effective(
                prefix=prefix,
                cfg=cfg,
                selection_cfg=grid_selection_cfg,
                optimizer_cfg=grid_optimizer_cfg,
                input_features_mode=grid_input_features_mode,
                input_feature_channels=grid_feature_channels,
                target_family=grid_target_family,
                target_vars=grid_target_vars,
                loss_cfg=grid_loss_cfg,
                train_cfg=train_cfg,
            )
            eval_vars = list(grid_target_vars)
        elif model_name in COORD_MLP_FAMILY_MODELS:
            _, coord_model_cfg = _normalize_coord_mlp_model_cfg(
                model_name=str(model_name),
                raw_cfg=dict(cfg.get("model_cfg", {})),
            )
            extra_artifacts["coord_mlp_contract_effective"] = {
                "model_type_effective": str(model_name),
                "backend_effective": str(grid_backend_effective),
                "target_family_effective": str(grid_target_family),
                "target_vars_effective": list(grid_target_vars),
                "input_features_mode": str(grid_input_features_mode),
                "input_feature_channels": list(grid_feature_channels),
                "feature_source_effective": str(locals().get("grid_feature_source", "legacy_xy")),
                "require_pack_effective": str(dict(cfg.get("input_features", {})).get("require_pack", "warn")).strip().lower(),
                "selection_mode_effective": str(grid_selection_cfg.get("mode", "last")).strip().lower(),
                "selection_weights_effective": dict(grid_selection_cfg.get("weights", {})),
                "embedding": dict(coord_model_cfg.get("embedding", {})),
                "siren": dict(coord_model_cfg.get("siren", {})),
            }
            eval_vars = list(grid_target_vars)
        pred_features = model.forward_features(ctx.cond_scaled[ctx.te])
        pred_eval = ctx.transforms.inverse_field_dict(
            {name: np.asarray(pred_features[name], dtype=np.float32) for name in grid_target_vars}
        )
        if "rho_eff" in pred_features:
            pred_eval["rho_eff"] = np.asarray(pred_features["rho_eff"], dtype=np.float32)
        true_eval = _to_true_eval(ctx.y, ctx.te, grid_target_vars, source_y_vars=ctx.y_vars)
    elif model_name == "deeponet_plasma":
        cfg = dict(train_cfg.get("deeponet_plasma", {}))
        train_key = "train.deeponet_plasma"
        deeponet_target_family = _resolve_unet_target_family(
            cfg.get("target_family", "allvars"),
            cfg_key=f"{train_key}.target_family",
        )
        default_deeponet_vars = resolve_allvars_order(list(ctx.y_vars), prefer_linear=True)
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
        if deeponet_input_mode not in {"legacy_xy", "geom_feature_pack"}:
            raise ValueError(f"{train_key}.input_features.mode must be one of: legacy_xy, geom_feature_pack")
        deeponet_feature_channels = _resolve_coord_feature_channels(input_features_cfg.get("features"))
        if deeponet_input_mode == "legacy_xy":
            deeponet_feature_channels = ["x", "y"]
        deeponet_selection_cfg = dict(cfg.get("selection", {}))
        deeponet_optimizer_cfg = dict(cfg.get("optimizer", {}))
        deeponet_batch_size_cases = int(cfg.get("batch_size_cases", 0))
        deeponet_shuffle_cases = bool(cfg.get("shuffle_cases", True))
        strict_mainline = bool(cfg.get("strict_mainline", False))
        if strict_mainline:
            _validate_deeponet_mainline_contract(
                deeponet_cfg=cfg,
                selection_cfg=deeponet_selection_cfg,
                loss_cfg=dict(ctx.loss_cfg or {}),
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
        deeponet_feature_source = "legacy_xy"
        deeponet_distance_transform_effective: dict[str, Any] = {"mode": "raw"}
        if deeponet_input_mode == "geom_feature_pack":
            require_pack = str(input_features_cfg.get("require_pack", "warn")).strip().lower()
            if require_pack not in {"off", "warn", "error"}:
                raise ValueError(f"{train_key}.input_features.require_pack must be one of: off, warn, error")
            rows, source = _build_coord_feature_rows(
                channels=deeponet_feature_channels,
                pack=ctx.coord_feature_pack,
                geom_ctx=ctx.geom_ctx,
                h=h,
                w=w,
            )
            deeponet_feature_source = str(source)
            if source != "preprocess_pack":
                msg = (
                    "deeponet input-feature contract: preprocess coord_feature_pack was not used; "
                    f"effective_source={source}"
                )
                if strict_mainline or require_pack == "error":
                    raise ValueError(msg)
                if require_pack == "warn":
                    extra_artifacts.setdefault("deeponet_contract_warnings", []).append(msg)
            distance_transform_cfg = _resolve_distance_transform_cfg(
                dict(input_features_cfg.get("distance_transform") or {})
            )
            distance_transform_cfg_effective, _ = _resolve_distance_transform_effective(
                distance_transform_cfg,
                stats=ctx.coord_distance_transform_stats,
                warnings_out=extra_artifacts.setdefault("deeponet_contract_warnings", []),
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
            loss_cfg=ctx.loss_cfg,
            curriculum_cfg=ctx.curriculum_cfg,
            supervised_mask=ctx.supervised_mask,
            supervised_distance=ctx.supervised_distance,
            optimizer_contract={**dict(optimizer_contract), "deeponet_optimizer": dict(deeponet_optimizer_cfg)},
            selection_cfg=deeponet_selection_cfg,
            batch_size_cases=deeponet_batch_size_cases,
            shuffle_cases=deeponet_shuffle_cases,
            seed=ctx.global_seed + ctx.model_idx,
        )
        history = out_t.history
        steps_per_epoch = (
            int(np.ceil(len(ctx.tr) / max(1, deeponet_batch_size_cases))) if deeponet_batch_size_cases > 0 else 1
        )
        extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))
        pred = model.predict_fields(ctx.cond_scaled[ctx.te], geom_ctx=ctx.geom_ctx)
        pred_eval = ctx.transforms.inverse_field_dict({k: v for k, v in pred.items() if k in set(deeponet_target_vars)})
        if "rho_eff" in pred:
            pred_eval["rho_eff"] = np.asarray(pred["rho_eff"], dtype=np.float32)
        true_eval = _to_true_eval(ctx.y, ctx.te, deeponet_target_vars, source_y_vars=ctx.y_vars)
        extra_artifacts["supervised_targets_enabled"] = bool(runtime.supervised_targets is not None)
        extra_artifacts["deeponet_contract_effective"] = {
            "target_family_effective": str(deeponet_target_family),
            "target_vars_effective": list(deeponet_target_vars),
            "feature_contract_effective": {
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
            "trunk_input_mode_effective": str(dict(cfg.get("model_cfg", {})).get("trunk_input_mode", "legacy_xy_fourier")),
            "trunk_fourier_n_freq_effective": int(dict(cfg.get("model_cfg", {})).get("trunk_fourier_n_freq", 1)),
            "trunk_fourier_mode_effective": str(dict(cfg.get("model_cfg", {})).get("trunk_fourier_mode", "legacy")),
            "trunk_cond_modulation_effective": str(dict(cfg.get("model_cfg", {})).get("trunk_cond_modulation", "none")),
            "trunk_cond_mod_hidden_effective": int(dict(cfg.get("model_cfg", {})).get("trunk_cond_mod_hidden", 64)),
            "residual_head_enabled_effective": bool(dict(dict(cfg.get("model_cfg", {})).get("residual_head", {})).get("enabled", False)),
            "residual_head_hidden_dim_effective": int(dict(dict(cfg.get("model_cfg", {})).get("residual_head", {})).get("hidden_dim", 64)),
            "residual_head_scale_init_effective": float(dict(dict(cfg.get("model_cfg", {})).get("residual_head", {})).get("scale_init", 0.0)),
            "residual_head_gain_mode_effective": str(
                dict(dict(cfg.get("model_cfg", {})).get("residual_head", {})).get("gain_mode", "learned")
            ),
            "residual_head_gain_value_effective": float(
                dict(dict(cfg.get("model_cfg", {})).get("residual_head", {})).get("gain_value", 1.0)
            ),
            "output_path_mode_effective": str(
                dict(dict(cfg.get("model_cfg", {})).get("output_path", {})).get("mode", "dot")
            ),
            "output_path_dot_skip_effective": float(
                dict(dict(cfg.get("model_cfg", {})).get("output_path", {})).get("dot_skip", 0.25)
            ),
            "output_path_fused_hidden_dim_effective": int(
                dict(dict(cfg.get("model_cfg", {})).get("output_path", {})).get("fused_hidden_dim", 96)
            ),
            "output_path_global_local_enabled_effective": bool(
                dict(dict(dict(cfg.get("model_cfg", {})).get("output_path", {})).get("global_local", {})).get(
                    "enabled", False
                )
            ),
            "output_path_global_hidden_dim_effective": int(
                dict(dict(cfg.get("model_cfg", {})).get("output_path", {})).get("global_hidden_dim", 64)
            ),
            "sensor_pool_mode_effective": str(dict(cfg.get("model_cfg", {})).get("sensor_pool_mode", "moments")),
            "branch_mode_effective": str(dict(cfg.get("model_cfg", {})).get("branch_mode", "moments")),
            "latent_layer_norm_effective": bool(dict(cfg.get("model_cfg", {})).get("latent_layer_norm", False)),
            "distance_transform_effective": dict(deeponet_distance_transform_effective),
            "strict_mainline_effective": bool(strict_mainline),
        }
        eval_vars = list(deeponet_target_vars)
    else:
        raise ValueError(f"Unsupported model in dispatch: {model_name}")

    true_eval_metrics = {k: true_eval[k] for k in eval_vars if k in true_eval and k in pred_eval}
    pred_eval_metrics = {k: pred_eval[k] for k in eval_vars if k in true_eval and k in pred_eval}
    metrics = rmse_by_var(true_eval_metrics, pred_eval_metrics)
    r2_scores = r2_by_var(true_eval_metrics, pred_eval_metrics)
    return TrainDispatchResult(
        model=model,
        history=history,
        pred_eval=pred_eval,
        true_eval=true_eval,
        metrics=metrics,
        r2_scores=r2_scores,
        extra_artifacts=extra_artifacts,
    )


__all__ = [
    "DeeponetRuntime",
    "TrainDispatchContext",
    "TrainDispatchResult",
    "resolve_deeponet_runtime",
    "run_model_train_predict",
]
