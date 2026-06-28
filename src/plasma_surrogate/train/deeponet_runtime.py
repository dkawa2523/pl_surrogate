"""DeepONet train-time runtime builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.core.deeponet_contract import (
    load_supervised_boundary_targets,
    resolve_sample_idx_source,
    validate_deeponet_task_meta,
)
from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.deeponet.poisson_head_torch import (
    POISSON_POTENTIAL_OUTPUT_KEY,
    DeepONetPoissonHeadTorch,
)


@dataclass
class DeeponetRuntime:
    model: DeepONetPlasmaOperatorTorch
    supervised_targets: dict[str, np.ndarray] | None


def resolve_deeponet_runtime(
    ctx: Any,
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
        output_keys=[POISSON_POTENTIAL_OUTPUT_KEY],
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


__all__ = ["DeeponetRuntime", "resolve_deeponet_runtime"]
