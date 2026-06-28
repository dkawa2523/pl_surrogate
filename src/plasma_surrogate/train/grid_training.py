"""Grid-family train/eval lane for shared model dispatch."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.core.input_modes import (
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.model_families import (
    COORD_MLP_FAMILY_MODELS,
    GEOM_DEEPONET_SIREN_FAMILY_MODELS,
    GRID_TORCH_MODELS,
    MAINLINE_GEOM_PACK_MODELS,
    SPECTRAL_FAMILY_MODELS,
    UNET_FAMILY_MODELS,
)
from plasma_surrogate.core.model_input_policy import ADAPTER_AUTO, ADAPTER_HYBRID_PACK_DESCRIPTOR
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.models.checkpoint import build_model_from_name
from plasma_surrogate.models.deeponet.pod_deeponet_torch import fit_pod_basis_from_targets
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import normalize_coord_mlp_pod_residual_cfg
from plasma_surrogate.models.mlp.coord_mlp_torch import _normalize_coord_mlp_model_cfg
from plasma_surrogate.train.loss_contract import removed_supervised_keys
from plasma_surrogate.train.spatial_features import (
    ICP_PART_SDF_CHANNELS,
    PART_SDF_SUMMARY_CHANNELS,
    apply_coord_feature_scaling,
    apply_distance_transform,
    build_case_spatial_features,
    build_coord_feature_rows,
    materialize_case_spatial_batch,
    resolve_coord_feature_channels,
    resolve_distance_transform_cfg,
    resolve_distance_transform_effective,
)
from plasma_surrogate.train.target_contracts import (
    resolve_allvars_target_family,
    resolve_allvars_target_vars_for_family,
    resolve_mainline_selection_weights,
    to_true_eval,
)


@dataclass
class GridTorchTrainResult:
    model: Any
    history: list[dict[str, float]]
    pred_eval: dict[str, np.ndarray]
    true_eval: dict[str, np.ndarray]
    eval_vars: list[str]


def _record_model_contract(extra_artifacts: dict[str, Any], model_name: str, contract: dict[str, Any]) -> None:
    model_contracts = dict(extra_artifacts.get("model_contracts", {}) or {})
    model_contracts[str(model_name)] = dict(contract)
    extra_artifacts["model_contracts"] = model_contracts


def _predict_features_batched(
    model: Any,
    cond: np.ndarray,
    *,
    spatial_features: Any | None,
    batch_size_cases: int,
) -> dict[str, np.ndarray]:
    n_cases = int(np.asarray(cond).shape[0])
    batch_size = n_cases if int(batch_size_cases) <= 0 else min(max(1, int(batch_size_cases)), n_cases)
    chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, n_cases, batch_size):
        stop = min(start + batch_size, n_cases)
        spatial_batch = materialize_case_spatial_batch(spatial_features, np.arange(start, stop, dtype=np.int64))
        raw = model.forward_features(cond[start:stop], spatial_features=spatial_batch)
        chunks.append({str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()})
    keys = sorted({key for chunk in chunks for key in chunk})
    return {
        key: np.concatenate([chunk[key] for chunk in chunks if key in chunk], axis=0).astype(np.float32)
        for key in keys
    }


def _normalize_profile_name(value: Any, *, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    return text if text else default


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
    if str(target_family).strip().lower() != "allvars":
        raise ValueError(f"{cfg_prefix}.target_family must be allvars for mainline; got={target_family}")
    unknown = [name for name in target_vars if name not in set(y_vars)]
    if unknown:
        raise ValueError(f"{cfg_prefix}.target_vars contains unknown vars: {unknown}; available={y_vars}")
    if require_shared_output_head:
        head_mode = str(dict(dict(model_cfg or {}).get("output_heads", {})).get("mode", "shared")).strip().lower()
        allowed_head_modes = {"shared"}
        if head_mode not in allowed_head_modes:
            raise ValueError(f"{cfg_prefix}.model_cfg.output_heads.mode must be shared for mainline")
        operator_head_mode_raw = dict(dict(model_cfg or {}).get("unet_operator_v2_cfg", {})).get("head_mode")
        if operator_head_mode_raw is not None:
            operator_head_mode = str(operator_head_mode_raw).strip().lower()
            if operator_head_mode != "shared":
                raise ValueError(f"{cfg_prefix}.model_cfg.unet_operator_v2_cfg.head_mode must be shared for mainline")
    if model_key in MAINLINE_GEOM_PACK_MODELS:
        mode = str(input_features_mode or "").strip().lower()
        if mode != "geom_feature_pack":
            raise ValueError(f"{cfg_prefix}.input_features.mode must be geom_feature_pack for mainline")
        required_channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
        part_lite_channels = required_channels + [
            "normal_x",
            "normal_y",
            "curvature_proxy",
            "boundary_band",
            *PART_SDF_SUMMARY_CHANNELS,
        ]
        icp_struct_channels = required_channels + ["mask_coil", "distance_coil", "coil_proximity"]
        icp_part_sdf_lite_channels = icp_struct_channels + list(ICP_PART_SDF_CHANNELS)
        channels = [str(v) for v in list(input_feature_channels or [])]
        if channels not in (required_channels, part_lite_channels, icp_struct_channels, icp_part_sdf_lite_channels):
            raise ValueError(
                f"{cfg_prefix}.input_features.features must be {required_channels} "
                f"(or {part_lite_channels} for part_lite_v1, "
                f"or {icp_struct_channels} for icp_struct_spatial_v1, "
                f"or {icp_part_sdf_lite_channels} for icp_part_sdf_lite_v1); got={channels}"
            )
    sel_mode = str(selection_cfg.get("mode", "last")).strip().lower()
    if sel_mode not in {"best_val_allvars_balance", "best_val_loss"}:
        raise ValueError(f"{cfg_prefix}.selection.mode must be one of: best_val_allvars_balance, best_val_loss for mainline")
    resolve_mainline_selection_weights(
        selection_cfg=selection_cfg,
        target_vars=target_vars,
        cfg_prefix=cfg_prefix,
    )
    if "density_guard" in selection_cfg:
        raise ValueError(f"{cfg_prefix}.selection.density_guard is removed from mainline")
    if "boundary_bonus_weight" in selection_cfg and float(selection_cfg.get("boundary_bonus_weight", 0.0)) != 0.0:
        raise ValueError(f"{cfg_prefix}.selection.boundary_bonus_weight must be 0.0 for mainline")

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


def _validate_geom_deeponet_siren_experimental_contract(
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


def _resolve_geom_deeponet_siren_descriptor_contract(
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
    descriptor_vector, descriptor_names = load_vector_from_pack(
        pack=descriptor_pack,
        pack_name="structure_descriptor_pack",
    )
    return descriptor_vector, {
        GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: str(desc_profile),
        GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY: int(descriptor_vector.shape[0]),
        "descriptor_feature_names_effective": list(descriptor_names),
        "adapter_mode_effective": str(adapter),
    }


def _build_spectral_contract(
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


def _build_unet_contract_effective(
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


def _build_coord_mlp_contract_effective(
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


def _build_geom_deeponet_siren_contract_effective(
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


def run_grid_torch_train_predict(
    *,
    ctx: Any,
    trainer: Any,
    train_cfg: dict[str, Any],
    optimizer_contract: dict[str, Any],
    model_name: str,
    cfg: dict[str, Any],
    input_mode_effective: str,
    structure_adapter_mode_effective: str,
    extra_artifacts: dict[str, Any],
) -> GridTorchTrainResult:
    h, w = int(ctx.h), int(ctx.w)
    grid_like_cfg = dict(train_cfg.get("unet_like", {}))
    grid_loss_cfg = copy.deepcopy(ctx.loss_cfg or {})
    grid_batch_size_cases = int(grid_like_cfg.get("batch_size_cases", 0))
    grid_shuffle_cases = bool(grid_like_cfg.get("shuffle_cases", True))
    grid_cond_train = np.asarray(ctx.cond_scaled[ctx.tr], dtype=np.float32)
    grid_cond_val = np.asarray(ctx.cond_scaled[ctx.va], dtype=np.float32)
    grid_cond_test = np.asarray(ctx.cond_scaled[ctx.te], dtype=np.float32)
    grid_backend_effective = str(dict(cfg.get("model_cfg", cfg)).get("backend", "numpy")).strip().lower()
    grid_supervised_distance_signed = None
    grid_supervised_bc_dir_mask = None
    grid_supervised_wafer_mask = None
    train_key = f"train.{model_name}"

    grid_target_family = resolve_allvars_target_family(
        cfg.get("target_family", "allvars"),
        cfg_key=f"{train_key}.target_family",
    )
    grid_target_vars = resolve_allvars_target_vars_for_family(
        family=grid_target_family,
        raw_target_vars=cfg.get("target_vars"),
        available=ctx.y_vars,
        cfg_key=f"{train_key}.target_vars",
    )
    grid_target_indices = [ctx.y_vars.index(v) for v in grid_target_vars]
    input_features_cfg = dict(cfg.get("input_features", {}))
    grid_input_features_mode = str(input_features_cfg.get("mode", "geom_feature_pack")).strip().lower()
    if grid_input_features_mode != "geom_feature_pack":
        raise ValueError(f"{train_key}.input_features.mode must be geom_feature_pack")
    grid_feature_channels = resolve_coord_feature_channels(input_features_cfg.get("features"))
    grid_selection_cfg = dict(cfg.get("selection", {}))

    if model_name in GEOM_DEEPONET_SIREN_FAMILY_MODELS:
        _validate_geom_deeponet_siren_experimental_contract(
            y_vars=ctx.y_vars,
            target_family=grid_target_family,
            target_vars=grid_target_vars,
            input_features_cfg=input_features_cfg,
            input_feature_channels=grid_feature_channels,
        )
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
        supervised_cfg = dict((grid_loss_cfg or {}).get("supervised", {}))
        removed_keys = removed_supervised_keys(supervised_cfg)
        if removed_keys:
            raise ValueError(
                f"{model_name} mainline supports only the product supervised loss path; "
                f"move research loss extensions to experiments. got={removed_keys}"
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
        grid_selection_cfg["weights"] = resolve_mainline_selection_weights(
            selection_cfg=grid_selection_cfg,
            target_vars=grid_target_vars,
            cfg_prefix=train_key,
        )

    if model_name in GEOM_DEEPONET_SIREN_FAMILY_MODELS:
        grid_descriptor_vec, geom_descriptor_meta = _resolve_geom_deeponet_siren_descriptor_contract(
            input_mode=input_mode_effective,
            adapter_mode=structure_adapter_mode_effective,
            descriptor_profile=ctx.structure_descriptor_profile_effective,
            descriptor_pack=ctx.structure_descriptor_pack,
        )
        extra_artifacts.update(dict(geom_descriptor_meta))
        desc_train = np.repeat(grid_descriptor_vec.reshape(1, -1), grid_cond_train.shape[0], axis=0).astype(np.float32)
        desc_val = np.repeat(grid_descriptor_vec.reshape(1, -1), grid_cond_val.shape[0], axis=0).astype(np.float32)
        desc_test = np.repeat(grid_descriptor_vec.reshape(1, -1), grid_cond_test.shape[0], axis=0).astype(np.float32)
        grid_cond_train = np.concatenate([grid_cond_train, desc_train], axis=1).astype(np.float32)
        grid_cond_val = np.concatenate([grid_cond_val, desc_val], axis=1).astype(np.float32)
        grid_cond_test = np.concatenate([grid_cond_test, desc_test], axis=1).astype(np.float32)
    if model_name in MAINLINE_GEOM_PACK_MODELS or model_name in COORD_MLP_FAMILY_MODELS:
        grid_backend_effective = "torch"

    model_cfg_for_build = dict(cfg.get("model_cfg", cfg))
    pod_basis_bundle = None
    coord_pod_basis_rank_by_var: dict[str, int] = {}
    if model_name == "coord_mlp_pod_residual":
        coord_pod_cfg = normalize_coord_mlp_pod_residual_cfg(model_cfg_for_build)
        basis_cfg = dict(coord_pod_cfg.get("basis", {}))
        pod_basis_bundle = fit_pod_basis_from_targets(
            ctx.y_scaled[ctx.tr][:, grid_target_indices],
            output_keys=list(grid_target_vars),
            requested_rank=int(basis_cfg.get("rank", 32)),
            center=bool(basis_cfg.get("center", True)),
            per_var=bool(basis_cfg.get("per_var", True)),
        )
        coord_pod_basis_rank_by_var = {str(k): int(v) for k, v in pod_basis_bundle.rank_by_var.items()}
        extra_artifacts["coord_mlp_pod_residual_basis_rank_by_var"] = dict(coord_pod_basis_rank_by_var)

    model = build_model_from_name(
        model_name=model_name,
        input_dim=int(grid_cond_train.shape[1]),
        grid_shape=(h, w),
        model_cfg=model_cfg_for_build,
        seed=ctx.global_seed + ctx.model_idx,
        phi_mode=str(ctx.profile_lock.get("phi_mode", "direct")),
        out_channels=len(grid_target_vars),
        output_keys=grid_target_vars,
        unet_feature_channels=grid_feature_channels,
        pod_basis_bundle=pod_basis_bundle,
    )

    grid_feature_source = "geom_feature_pack"
    case_spatial_source: Any | None = None
    scaling_applied = False
    distance_transform_cfg = resolve_distance_transform_cfg(
        dict(dict(cfg.get("input_features", {})).get("distance_transform") or {})
    )
    distance_transform_cfg_effective, _ = resolve_distance_transform_effective(
        distance_transform_cfg,
        stats=ctx.coord_distance_transform_stats,
    )
    case_spatial_source, source = build_case_spatial_features(
        channels=grid_feature_channels,
        h=h,
        w=w,
        pack=getattr(ctx, "case_spatial_feature_pack", None),
        static_pack=getattr(ctx, "static_spatial_feature_pack", None),
        case_pack=getattr(ctx, "case_structure_feature_pack", None),
        distance_transform_cfg=distance_transform_cfg_effective,
        coord_feature_scaler_artifact=ctx.coord_feature_scaler,
    )
    grid_feature_source = str(source)
    if case_spatial_source is not None:
        scaling_applied = bool(getattr(case_spatial_source, "scaling_applied", False))
        extra_artifacts["case_spatial_pack_used"] = True
        extra_artifacts["case_spatial_pack_storage"] = str(source)
        extra_artifacts["case_spatial_feature_channels"] = list(grid_feature_channels)
        extra_artifacts["case_spatial_feature_shape"] = [int(v) for v in case_spatial_source.shape]
    else:
        static_supported = {
            "x",
            "y",
            "distance_signed",
            "distance_any",
            "mask_plasma",
            "normal_x",
            "normal_y",
            "curvature_proxy",
        }
        unsupported_static = sorted(set(grid_feature_channels) - static_supported)
        if unsupported_static:
            raise ValueError(
                f"{model_name} input-feature contract: case-specific spatial pack was not used; "
                f"effective_source={source}; missing_static_channels={unsupported_static}"
            )
        rows, source = build_coord_feature_rows(
            channels=grid_feature_channels,
            pack=ctx.coord_feature_pack,
            geom_ctx=ctx.geom_ctx,
            h=h,
            w=w,
        )
        grid_feature_source = str(source)
        if source != "preprocess_pack":
            raise ValueError(
                f"{model_name} input-feature contract: preprocessing coord_feature_pack is required; "
                f"effective_source={source}"
            )
        rows, _ = apply_distance_transform(
            rows.astype(np.float32),
            channels=grid_feature_channels,
            cfg=distance_transform_cfg_effective,
        )
        rows, _, scaling_applied = apply_coord_feature_scaling(
            rows.astype(np.float32),
            channels=grid_feature_channels,
            coord_feature_scaler_artifact=ctx.coord_feature_scaler,
        )
    if model_name in COORD_MLP_FAMILY_MODELS and not bool(scaling_applied):
        raise ValueError(f"train.{model_name} requires preprocessing.coord_features.scaling.enabled=true")

    spatial_train = case_spatial_source.subset(ctx.tr) if case_spatial_source is not None else None
    spatial_val = case_spatial_source.subset(ctx.va) if case_spatial_source is not None else None
    spatial_test = case_spatial_source.subset(ctx.te) if case_spatial_source is not None else None
    spatial = None if case_spatial_source is not None else rows.reshape(h, w, len(grid_feature_channels)).astype(np.float32)
    if spatial is not None and hasattr(model, "set_static_spatial_features"):
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

    out = trainer.run_unet(
        model,
        grid_cond_train,
        ctx.y_scaled[ctx.tr][:, grid_target_indices],
        grid_cond_val,
        ctx.y_scaled[ctx.va][:, grid_target_indices],
        epochs=int(cfg.get("epochs", train_cfg.get("epochs", 20))),
        lr=float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
        physics_cfg=ctx.physics_cfg,
        loss_cfg=grid_loss_cfg,
        curriculum_cfg=ctx.curriculum_cfg,
        supervised_mask=ctx.supervised_mask,
        supervised_distance=ctx.supervised_distance,
        supervised_distance_signed=grid_supervised_distance_signed,
        supervised_bc_dir_mask=grid_supervised_bc_dir_mask,
        supervised_wafer_mask=grid_supervised_wafer_mask,
        optimizer_contract=optimizer_contract,
        unet_optimizer_cfg=grid_optimizer_cfg,
        spatial_train=spatial_train,
        spatial_val=spatial_val,
        batch_size_cases=grid_batch_size_cases,
        shuffle_cases=grid_shuffle_cases,
        seed=ctx.global_seed + ctx.model_idx,
        selection_cfg=grid_selection_cfg,
        selection_target_override=None,
        selection_pred_additive=None,
    )

    history = out.history
    steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, grid_batch_size_cases))) if grid_batch_size_cases > 0 else 1
    extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))

    eval_vars = list(grid_target_vars)
    if model_name in UNET_FAMILY_MODELS:
        _record_model_contract(
            extra_artifacts,
            "unet",
            _build_unet_contract_effective(
                cfg=cfg,
                train_cfg=train_cfg,
                model=model,
                grid_backend_effective=grid_backend_effective,
                grid_feature_channels=grid_feature_channels,
                grid_selection_cfg=grid_selection_cfg,
                grid_optimizer_cfg=grid_optimizer_cfg,
                grid_input_features_mode=grid_input_features_mode,
                grid_target_family=grid_target_family,
                grid_target_vars=grid_target_vars,
                grid_loss_cfg=grid_loss_cfg,
            ),
        )
    elif model_name in SPECTRAL_FAMILY_MODELS:
        prefix = "fno" if model_name == "fno" else "ffno"
        _record_model_contract(
            extra_artifacts,
            prefix,
            _build_spectral_contract(
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
            ),
        )
    elif model_name in COORD_MLP_FAMILY_MODELS:
        if model_name == "coord_mlp_pod_residual":
            coord_model_cfg = normalize_coord_mlp_pod_residual_cfg(dict(cfg.get("model_cfg", {})))
        else:
            _, coord_model_cfg = _normalize_coord_mlp_model_cfg(
                model_name=str(model_name),
                raw_cfg=dict(cfg.get("model_cfg", {})),
            )
        _record_model_contract(
            extra_artifacts,
            "coord_mlp",
            _build_coord_mlp_contract_effective(
                model_name=model_name,
                grid_backend_effective=grid_backend_effective,
                grid_target_family=grid_target_family,
                grid_target_vars=grid_target_vars,
                grid_input_features_mode=grid_input_features_mode,
                grid_feature_channels=grid_feature_channels,
                grid_feature_source=grid_feature_source,
                cfg=cfg,
                grid_selection_cfg=grid_selection_cfg,
                coord_model_cfg=coord_model_cfg,
                basis_rank_by_var=coord_pod_basis_rank_by_var,
            ),
        )
    elif model_name in GEOM_DEEPONET_SIREN_FAMILY_MODELS:
        _record_model_contract(
            extra_artifacts,
            "geom_deeponet_siren",
            _build_geom_deeponet_siren_contract_effective(
                model_name=model_name,
                grid_target_family=grid_target_family,
                grid_target_vars=grid_target_vars,
                grid_input_features_mode=grid_input_features_mode,
                grid_feature_channels=grid_feature_channels,
                grid_selection_cfg=grid_selection_cfg,
                extra_artifacts=extra_artifacts,
                structure_adapter_mode_effective=structure_adapter_mode_effective,
            ),
        )

    pred_features = _predict_features_batched(
        model,
        grid_cond_test,
        spatial_features=spatial_test,
        batch_size_cases=grid_batch_size_cases,
    )
    pred_eval = ctx.transforms.inverse_field_dict(
        {name: np.asarray(pred_features[name], dtype=np.float32) for name in grid_target_vars}
    )
    if "rho_eff" in pred_features:
        pred_eval["rho_eff"] = np.asarray(pred_features["rho_eff"], dtype=np.float32)
    true_eval = to_true_eval(ctx.y, ctx.te, grid_target_vars, source_y_vars=ctx.y_vars)
    return GridTorchTrainResult(
        model=model,
        history=history,
        pred_eval=pred_eval,
        true_eval=true_eval,
        eval_vars=eval_vars,
    )


__all__ = [
    "GridTorchTrainResult",
    "run_grid_torch_train_predict",
]
