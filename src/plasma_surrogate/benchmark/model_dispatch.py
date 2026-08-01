"""Benchmark-side adapter for shared train model dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.input_modes import (
    DESCRIPTOR_PROFILE_KEY,
    LATENT_PROFILE_KEY,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    build_input_mode_effective_metadata,
)
from plasma_surrogate.core.model_input_policy import resolve_effective_input_mode_metadata_for_model
from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict


@dataclass
class BenchmarkModelContext:
    benchmark_cfg: dict[str, Any]
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
    loss_cfg: dict[str, Any] | None
    curriculum_cfg: dict[str, Any] | None
    supervised_mask: np.ndarray | None
    supervised_distance: np.ndarray | None
    geom_ctx: Any
    deeponet_index: dict[str, Any]
    deeponet_index_meta: dict[str, Any]
    deeponet_poisson_index: dict[str, Any]
    deeponet_poisson_meta: dict[str, Any]
    deeponet_boundary_index: dict[str, Any]
    deeponet_boundary_meta: dict[str, Any]
    coord_feature_scaler: dict[str, Any] | None = None
    coord_feature_pack: dict[str, Any] | None = None
    static_spatial_feature_pack: dict[str, Any] | None = None
    case_structure_feature_pack: dict[str, Any] | None = None
    coord_distance_transform_stats: dict[str, Any] | None = None
    structure_descriptor_pack: dict[str, Any] | None = None
    latent_feature_pack: dict[str, Any] | None = None
    case_ids: list[str] | None = None
    input_mode_meta: dict[str, Any] | None = None


def run_model_train_eval(ctx: BenchmarkModelContext) -> dict[str, Any]:
    train_cfg = dict(ctx.benchmark_cfg.get("train", {}))
    input_mode_meta = dict(ctx.input_mode_meta or {})
    if not input_mode_meta:
        input_mode_meta = resolve_effective_input_mode_metadata_for_model(
            model_name=ctx.model_name,
            input_mode_meta=build_input_mode_effective_metadata(ctx.benchmark_cfg),
        )
    if "fno_n_modes" in ctx.benchmark_cfg:
        n_modes = int(ctx.benchmark_cfg["fno_n_modes"])
        fno_cfg = dict(train_cfg.get("fno", train_cfg.get("fno_baseline", {})))
        fno_model_cfg = dict(fno_cfg.get("model_cfg", {}))
        fno_model_cfg["fno_n_modes"] = n_modes
        fno_cfg["model_cfg"] = fno_model_cfg
        train_cfg["fno"] = fno_cfg
    if "ffno_n_modes" in ctx.benchmark_cfg:
        n_modes = int(ctx.benchmark_cfg["ffno_n_modes"])
        ffno_cfg = dict(train_cfg.get("ffno", {}))
        ffno_model_cfg = dict(ffno_cfg.get("model_cfg", {}))
        ffno_model_cfg["fno_n_modes"] = n_modes
        ffno_cfg["model_cfg"] = ffno_model_cfg
        train_cfg["ffno"] = ffno_cfg

    dispatch = run_model_train_predict(
        TrainDispatchContext(
            run_cfg={"train": train_cfg, **ctx.benchmark_cfg},
            profile_lock=ctx.profile_lock,
            model_idx=ctx.model_idx,
            model_name=ctx.model_name,
            model_dir=ctx.model_dir,
            global_seed=ctx.global_seed,
            n_cases=ctx.n_cases,
            h=ctx.h,
            w=ctx.w,
            y_vars=ctx.y_vars,
            cond_scaled=ctx.cond_scaled,
            y=ctx.y,
            y_scaled=ctx.y_scaled,
            tr=ctx.tr,
            va=ctx.va,
            te=ctx.te,
            transforms=ctx.transforms,
            physics_cfg=ctx.physics_cfg,
            geom_ctx=ctx.geom_ctx,
            deeponet_index=ctx.deeponet_index,
            deeponet_index_meta=ctx.deeponet_index_meta,
            deeponet_poisson_index=ctx.deeponet_poisson_index,
            deeponet_poisson_meta=ctx.deeponet_poisson_meta,
            deeponet_boundary_index=ctx.deeponet_boundary_index,
            deeponet_boundary_meta=ctx.deeponet_boundary_meta,
            config_base_dir=None,
            loss_cfg=ctx.loss_cfg,
            curriculum_cfg=ctx.curriculum_cfg,
            supervised_mask=ctx.supervised_mask,
            supervised_distance=ctx.supervised_distance,
            coord_feature_scaler=dict(ctx.coord_feature_scaler or {}),
            coord_feature_pack=ctx.coord_feature_pack,
            static_spatial_feature_pack=ctx.static_spatial_feature_pack,
            case_structure_feature_pack=ctx.case_structure_feature_pack,
            coord_distance_transform_stats=dict(ctx.coord_distance_transform_stats or {}),
            structure_descriptor_pack=ctx.structure_descriptor_pack,
            latent_feature_pack=ctx.latent_feature_pack,
            case_ids=list(ctx.case_ids) if ctx.case_ids is not None else None,
            input_mode_effective=str(input_mode_meta.get("input_mode_effective", "")),
            structure_adapter_mode_effective=str(input_mode_meta.get(STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY, "")),
            structure_descriptor_profile_effective=str(
                input_mode_meta.get(DESCRIPTOR_PROFILE_KEY, "none")
            ),
            structure_latent_profile_effective=str(
                input_mode_meta.get(LATENT_PROFILE_KEY, "none")
            ),
        )
    )
    return {
        "model": dispatch.model,
        "history": dispatch.history,
        "pred_eval": dispatch.pred_eval,
        "true_eval": dispatch.true_eval,
        "metrics": dispatch.metrics,
        "r2_scores": dispatch.r2_scores,
        "extra_artifacts": dispatch.extra_artifacts,
    }
