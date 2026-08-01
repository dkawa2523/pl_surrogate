"""Grid-family train/eval lane for shared model dispatch."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.core.boundary_distance import boundary_distance_channels_from_loss_cfg
from plasma_surrogate.core.model_families import (
    COORD_MLP_FAMILY_MODELS,
    GEOM_DEEPONET_SIREN_FAMILY_MODELS,
    MAINLINE_GEOM_PACK_MODELS,
    SPECTRAL_FAMILY_MODELS,
    UNET_FAMILY_MODELS,
)
from plasma_surrogate.models.checkpoint import build_model_from_name
from plasma_surrogate.models.deeponet.pod_deeponet_torch import fit_pod_basis_from_targets
from plasma_surrogate.models.heads.role_grouped import (
    is_grouped_output_head_mode,
)
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import normalize_coord_mlp_pod_residual_cfg
from plasma_surrogate.models.mlp.coord_mlp_torch import _normalize_coord_mlp_model_cfg
from plasma_surrogate.train.grid_contracts import (
    build_coord_mlp_contract_effective,
    build_geom_deeponet_siren_contract_effective,
    build_spectral_contract,
    build_unet_contract_effective,
    resolve_geom_deeponet_siren_descriptor_contract,
    validate_coord_mlp_experimental_contract,
    validate_geom_deeponet_siren_experimental_contract,
    validate_unet_like_mainline_contract,
)
from plasma_surrogate.train.loss_contract import removed_supervised_keys
from plasma_surrogate.train.model_artifacts import record_model_contract
from plasma_surrogate.train.selection import GROUP_BALANCE_SELECTION_MODE, SPATIAL_SELECTION_MODE
from plasma_surrogate.train.spatial_supervision import (
    materialize_supervised_geometry,
    required_raw_supervision_channels,
)
from plasma_surrogate.preprocessing.spatial_features import (
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


def _resolve_physical_point_weighting(
    *,
    ctx: Any,
    loss_cfg: dict[str, Any],
    target_vars: list[str],
) -> tuple[np.ndarray | None, dict[str, dict[str, float]] | None]:
    supervised = dict(loss_cfg.get("supervised", {}) or {})
    physical = dict(supervised.get("physical_weighting", {}) or {})
    axisymmetric = bool(physical.get("axisymmetric_volume", False))
    density_targets = [str(name) for name in physical.get("density_weighted_targets", [])]
    if not axisymmetric and not density_targets:
        return None, None

    point_weight = None
    if axisymmetric:
        if ctx.geom_ctx is None:
            raise ValueError("axisymmetric_volume weighting requires GeometryContext")
        coord = np.asarray(ctx.geom_ctx.coord_grid, dtype=np.float32)
        if coord.shape != (2, int(ctx.h), int(ctx.w)):
            raise ValueError(
                "axisymmetric_volume weighting requires coord_grid [2,H,W], "
                f"got={coord.shape}"
            )
        point_weight = np.asarray(coord[0], dtype=np.float32)
        if np.any(~np.isfinite(point_weight)) or np.any(point_weight < 0.0) or not np.any(point_weight > 0.0):
            raise ValueError("axisymmetric radial coordinates must be finite, non-negative, and non-zero")

    target_affine: dict[str, dict[str, float]] = {}
    if density_targets:
        density_source = str(physical.get("density_source", "ne"))
        if density_source not in target_vars:
            raise ValueError(f"density weighting source {density_source!r} must be a trained target")
        transform_spec = dict(ctx.transforms.target_transforms.get(density_source, {}) or {})
        if str(transform_spec.get("value_transform", "identity")).strip().lower() != "identity":
            raise ValueError("density weighting requires a linear identity target transform")
        scaler = dict(ctx.transforms.y_scalers[density_source].to_dict())
        if str(scaler.get("type", "")).strip().lower() != "zscore":
            raise ValueError("density weighting requires a zscore density scaler")
        mean = np.asarray(scaler.get("mean"), dtype=np.float64).reshape(-1)
        std = np.asarray(scaler.get("std"), dtype=np.float64).reshape(-1)
        if mean.size != 1 or std.size != 1 or not np.isfinite(mean[0]) or not np.isfinite(std[0]) or std[0] <= 0.0:
            raise ValueError("density zscore scaler must contain one finite mean and positive std")
        target_affine[density_source] = {"mean": float(mean[0]), "scale": float(std[0])}
    return point_weight, (target_affine or None)


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
    grid_point_weight, grid_target_affine = _resolve_physical_point_weighting(
        ctx=ctx,
        loss_cfg=grid_loss_cfg,
        target_vars=grid_target_vars,
    )
    input_features_cfg = dict(cfg.get("input_features", {}))
    grid_input_features_mode = str(input_features_cfg.get("mode", "geom_feature_pack")).strip().lower()
    if grid_input_features_mode != "geom_feature_pack":
        raise ValueError(f"{train_key}.input_features.mode must be geom_feature_pack")
    grid_feature_channels = resolve_coord_feature_channels(input_features_cfg.get("features"))
    grid_selection_cfg = dict(cfg.get("selection", {}))
    if str(grid_selection_cfg.get("mode", "")).strip().lower() in {
        GROUP_BALANCE_SELECTION_MODE,
        SPATIAL_SELECTION_MODE,
    }:
        grid_selection_cfg.setdefault(
            "target_role_schema",
            dict(dict(ctx.physics_cfg or {}).get("target_role_schema", {}) or {}),
        )

    if model_name in GEOM_DEEPONET_SIREN_FAMILY_MODELS:
        validate_geom_deeponet_siren_experimental_contract(
            y_vars=ctx.y_vars,
            target_family=grid_target_family,
            target_vars=grid_target_vars,
            input_features_cfg=input_features_cfg,
            input_feature_channels=grid_feature_channels,
        )
    if model_name in MAINLINE_GEOM_PACK_MODELS:
        validate_unet_like_mainline_contract(
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
        validate_coord_mlp_experimental_contract(
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
        validate_unet_like_mainline_contract(
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
        grid_descriptor_input, geom_descriptor_meta = resolve_geom_deeponet_siren_descriptor_contract(
            input_mode=input_mode_effective,
            adapter_mode=structure_adapter_mode_effective,
            descriptor_profile=ctx.structure_descriptor_profile_effective,
            descriptor_pack=ctx.structure_descriptor_pack,
        )
        extra_artifacts.update(dict(geom_descriptor_meta))
        descriptor_input = np.asarray(grid_descriptor_input, dtype=np.float32)
        if descriptor_input.ndim == 1:
            desc_train = np.repeat(descriptor_input.reshape(1, -1), grid_cond_train.shape[0], axis=0)
            desc_val = np.repeat(descriptor_input.reshape(1, -1), grid_cond_val.shape[0], axis=0)
            desc_test = np.repeat(descriptor_input.reshape(1, -1), grid_cond_test.shape[0], axis=0)
        elif descriptor_input.ndim == 2:
            if int(descriptor_input.shape[0]) != int(ctx.n_cases):
                raise ValueError(
                    "geom_deeponet_siren case descriptor row count mismatch: "
                    f"rows={int(descriptor_input.shape[0])}, n_cases={int(ctx.n_cases)}"
                )
            descriptor_case_ids = [
                str(value)
                for value in np.asarray(dict(ctx.structure_descriptor_pack or {}).get("case_ids", [])).reshape(-1).tolist()
            ]
            runtime_case_ids = [str(value) for value in list(ctx.case_ids or [])]
            if runtime_case_ids and descriptor_case_ids != runtime_case_ids:
                raise ValueError(
                    "geom_deeponet_siren descriptor case_ids are not aligned with runtime dataset case_ids"
                )
            desc_train = descriptor_input[np.asarray(ctx.tr, dtype=np.int64)]
            desc_val = descriptor_input[np.asarray(ctx.va, dtype=np.int64)]
            desc_test = descriptor_input[np.asarray(ctx.te, dtype=np.int64)]
        else:
            raise ValueError(
                "geom_deeponet_siren descriptor input must be [D] or [N,D]; "
                f"got shape={descriptor_input.shape}"
            )
        desc_train = np.asarray(desc_train, dtype=np.float32)
        desc_val = np.asarray(desc_val, dtype=np.float32)
        desc_test = np.asarray(desc_test, dtype=np.float32)
        grid_cond_train = np.concatenate([grid_cond_train, desc_train], axis=1).astype(np.float32)
        grid_cond_val = np.concatenate([grid_cond_val, desc_val], axis=1).astype(np.float32)
        grid_cond_test = np.concatenate([grid_cond_test, desc_test], axis=1).astype(np.float32)
    if model_name in MAINLINE_GEOM_PACK_MODELS or model_name in COORD_MLP_FAMILY_MODELS:
        grid_backend_effective = "torch"

    model_cfg_for_build = dict(cfg.get("model_cfg", cfg))
    if is_grouped_output_head_mode(dict(model_cfg_for_build.get("output_heads", {})).get("mode", "shared")):
        model_cfg_for_build["target_role_schema"] = dict(dict(ctx.physics_cfg or {}).get("target_role_schema", {}) or {})

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
        static_pack=getattr(ctx, "static_spatial_feature_pack", None),
        case_pack=getattr(ctx, "case_structure_feature_pack", None),
        distance_transform_cfg=distance_transform_cfg_effective,
        coord_feature_scaler_artifact=ctx.coord_feature_scaler,
        expected_case_ids=getattr(ctx, "case_ids", None),
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
            "boundary_band",
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
            distance_transform_cfg=distance_transform_cfg_effective,
        )
    if model_name in COORD_MLP_FAMILY_MODELS and not bool(scaling_applied):
        raise ValueError(f"train.{model_name} requires preprocessing.coord_features.scaling.enabled=true")

    spatial_train = case_spatial_source.subset(ctx.tr) if case_spatial_source is not None else None
    spatial_val = case_spatial_source.subset(ctx.va) if case_spatial_source is not None else None
    spatial_test = case_spatial_source.subset(ctx.te) if case_spatial_source is not None else None
    spatial = None if case_spatial_source is not None else rows.reshape(h, w, len(grid_feature_channels)).astype(np.float32)

    supervision_channels = required_raw_supervision_channels(
        loss_cfg=grid_loss_cfg,
        selection_cfg=grid_selection_cfg,
    )
    supervision_source: Any | None = None
    supervision_source_name = "static_fallback"
    if supervision_channels:
        supervision_source, supervision_source_name = build_case_spatial_features(
            channels=supervision_channels,
            h=h,
            w=w,
            static_pack=getattr(ctx, "static_spatial_feature_pack", None),
            case_pack=getattr(ctx, "case_structure_feature_pack", None),
            distance_transform_cfg={"mode": "raw"},
            coord_feature_scaler_artifact=None,
            expected_case_ids=getattr(ctx, "case_ids", None),
        )
        custom_boundary_channels = set(boundary_distance_channels_from_loss_cfg(grid_loss_cfg)) - {
            "distance_any"
        }
        if supervision_source is None and custom_boundary_channels:
            raise ValueError(
                f"{train_key} raw boundary supervision is missing configured channels: "
                f"{sorted(custom_boundary_channels)}; source={supervision_source_name}"
            )
    supervision_train = supervision_source.subset(ctx.tr) if supervision_source is not None else None
    supervision_val = supervision_source.subset(ctx.va) if supervision_source is not None else None
    if supervision_source is not None:
        extra_artifacts["case_supervision_pack_used"] = True
        extra_artifacts["case_supervision_pack_storage"] = str(supervision_source_name)
        extra_artifacts["case_supervision_channels"] = list(supervision_channels)

    pod_basis_bundle = None
    coord_pod_basis_rank_by_var: dict[str, int] = {}
    if model_name == "coord_mlp_pod_residual":
        coord_pod_cfg = normalize_coord_mlp_pod_residual_cfg(model_cfg_for_build)
        basis_cfg = dict(coord_pod_cfg.get("basis", {}))
        basis_geometry = materialize_supervised_geometry(
            supervision_train,
            np.arange(int(len(ctx.tr)), dtype=np.int64),
            mask=ctx.supervised_mask,
            distance_any=ctx.supervised_distance,
        )
        pod_basis_bundle = fit_pod_basis_from_targets(
            ctx.y_scaled[ctx.tr][:, grid_target_indices],
            output_keys=list(grid_target_vars),
            requested_rank=int(basis_cfg.get("rank", 32)),
            center=bool(basis_cfg.get("center", True)),
            per_var=bool(basis_cfg.get("per_var", True)),
            active_mask=basis_geometry["mask"],
        )
        coord_pod_basis_rank_by_var = {
            str(k): int(v) for k, v in pod_basis_bundle.rank_by_var.items()
        }
        extra_artifacts["coord_mlp_pod_residual_basis_rank_by_var"] = dict(
            coord_pod_basis_rank_by_var
        )

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
        supervision_train=supervision_train,
        supervision_val=supervision_val,
        supervised_point_weight=grid_point_weight,
        target_affine=grid_target_affine,
    )

    history = out.history
    steps_per_epoch = int(np.ceil(len(ctx.tr) / max(1, grid_batch_size_cases))) if grid_batch_size_cases > 0 else 1
    extra_artifacts["effective_steps"] = int(max(steps_per_epoch, 1) * len(history))

    eval_vars = list(grid_target_vars)
    if model_name in UNET_FAMILY_MODELS:
        record_model_contract(
            extra_artifacts,
            "unet",
            build_unet_contract_effective(
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
        record_model_contract(
            extra_artifacts,
            prefix,
            build_spectral_contract(
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
        record_model_contract(
            extra_artifacts,
            "coord_mlp",
            build_coord_mlp_contract_effective(
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
        record_model_contract(
            extra_artifacts,
            "geom_deeponet_siren",
            build_geom_deeponet_siren_contract_effective(
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
