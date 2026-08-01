"""Checkpoint serializer for the DeepONet plasma operator family."""

from __future__ import annotations

from typing import Any

import numpy as np

DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES = frozenset(
    {
        "deeponet_plasma",
        # Read compatibility for checkpoints written before the product model
        # id was aligned with the canonical registry key.
        "deeponet_plasma_torch",
    }
)
DEFAULT_SENSOR_FEATURE_NAMES = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]

__all__ = [
    "DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES",
    "is_deeponet_plasma_checkpoint_model",
    "load_deeponet_plasma_checkpoint_model",
    "load_deeponet_plasma_checkpoint_weights",
    "make_deeponet_plasma_checkpoint_meta",
]


def is_deeponet_plasma_checkpoint_model(model: Any) -> bool:
    to_meta = getattr(model, "to_meta", None)
    if not callable(to_meta):
        return False
    meta = to_meta()
    return (
        isinstance(meta, dict)
        and str(meta.get("model_type", "")).strip().lower()
        in DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES
    )


def make_deeponet_plasma_checkpoint_meta(model: Any) -> dict[str, Any] | None:
    if not is_deeponet_plasma_checkpoint_model(model):
        return None
    return dict(model.to_meta())


def _get_meta_value(meta: dict[str, Any], parent: dict[str, Any], key: str, default: Any) -> Any:
    return meta.get(key, parent.get(key, default))


def _build_plasma_operator_from_meta(
    meta: dict[str, Any],
    *,
    parent: dict[str, Any] | None = None,
    output_keys: list[str] | None = None,
) -> Any:
    from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch

    root = parent or {}
    return DeepONetPlasmaOperatorTorch(
        cond_dim=int(root.get("cond_dim", meta.get("cond_dim"))),
        grid_shape=tuple(_get_meta_value(meta, root, "grid_shape", meta.get("grid_shape"))),
        output_keys=list(output_keys if output_keys is not None else meta.get("output_keys", ["target_0", "target_1", "target_2", "target_3"])),
        latent_dim=int(_get_meta_value(meta, root, "latent_dim", 32)),
        hidden_dim=int(_get_meta_value(meta, root, "hidden_dim", 64)),
        sensor_indices=np.asarray(_get_meta_value(meta, root, "sensor_indices", []), dtype=np.int64),
        query_indices=np.asarray(_get_meta_value(meta, root, "query_indices", []), dtype=np.int64),
        flatten_order=str(_get_meta_value(meta, root, "flatten_order", "C")),
        sensor_feature_names=list(_get_meta_value(meta, root, "sensor_feature_names", DEFAULT_SENSOR_FEATURE_NAMES)),
        query_feature_names=(
            list(_get_meta_value(meta, root, "query_feature_names", []))
            if _get_meta_value(meta, root, "query_feature_names", None) is not None
            else None
        ),
        trunk_input_mode=str(_get_meta_value(meta, root, "trunk_input_mode", "geom_feature_pack")),
        sensor_pool_mode=str(_get_meta_value(meta, root, "sensor_pool_mode", "moments")),
        sensor_embed_dim=int(_get_meta_value(meta, root, "sensor_embed_dim", 32)),
        branch_mode=str(_get_meta_value(meta, root, "branch_mode", "moments")),
        trunk_fourier_n_freq=int(_get_meta_value(meta, root, "trunk_fourier_n_freq", 1)),
        trunk_fourier_mode=str(_get_meta_value(meta, root, "trunk_fourier_mode", "symmetric")),
        trunk_cond_modulation=str(_get_meta_value(meta, root, "trunk_cond_modulation", "none")),
        trunk_cond_mod_hidden=int(_get_meta_value(meta, root, "trunk_cond_mod_hidden", 64)),
        residual_head_enabled=bool(_get_meta_value(meta, root, "residual_head_enabled", False)),
        residual_head_hidden_dim=int(_get_meta_value(meta, root, "residual_head_hidden_dim", 64)),
        residual_head_scale_init=float(_get_meta_value(meta, root, "residual_head_scale_init", 0.0)),
        residual_head_gain_mode=str(_get_meta_value(meta, root, "residual_head_gain_mode", "learned")),
        residual_head_gain_value=float(_get_meta_value(meta, root, "residual_head_gain_value", 1.0)),
        latent_layer_norm=bool(_get_meta_value(meta, root, "latent_layer_norm", False)),
        output_path_mode=str(_get_meta_value(meta, root, "output_path_mode", "dot")),
        output_path_dot_skip=float(_get_meta_value(meta, root, "output_path_dot_skip", 0.25)),
        output_path_dot_skip_mode=str(_get_meta_value(meta, root, "output_path_dot_skip_mode", "fixed")),
        output_path_fused_hidden_dim=int(_get_meta_value(meta, root, "output_path_fused_hidden_dim", 96)),
        output_path_global_local_enabled=bool(_get_meta_value(meta, root, "output_path_global_local_enabled", False)),
        output_path_global_hidden_dim=int(_get_meta_value(meta, root, "output_path_global_hidden_dim", 64)),
        missing_geom_feature_policy=str(_get_meta_value(meta, root, "missing_geom_feature_policy", "error")),
    )


def load_deeponet_plasma_checkpoint_model(meta: dict[str, Any]) -> Any | None:
    model_type = str(meta.get("model_type", "")).strip().lower()
    if model_type not in DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES:
        return None

    from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch
    from plasma_surrogate.models.deeponet.poisson_head_torch import (
        POISSON_POTENTIAL_OUTPUT_KEY,
        DeepONetPoissonHeadTorch,
    )

    model = _build_plasma_operator_from_meta(meta)

    ph_meta = meta.get("poisson_head")
    if isinstance(ph_meta, dict):
        ph_net = _build_plasma_operator_from_meta(ph_meta, parent=meta, output_keys=[POISSON_POTENTIAL_OUTPUT_KEY])
        ph = DeepONetPoissonHeadTorch.from_cache(
            deeponet_poisson=ph_net,
            sensor_idx=np.asarray(ph_meta.get("sensor_indices", []), dtype=np.int64),
            query_idx=np.asarray(ph_meta.get("query_indices", []), dtype=np.int64),
            flatten_order=str(ph_meta.get("flatten_order", "C")),
            grid_shape=tuple(ph_meta.get("grid_shape", meta["grid_shape"])),
            freeze=bool(ph_meta.get("freeze", True)),
        )
        model.attach_poisson_head(ph)

    bo_meta = meta.get("boundary_operator")
    if isinstance(bo_meta, dict):
        bo = BoundaryOperatorTorch(
            primary_qoi_key=str(bo_meta.get("primary_qoi_key", "Gamma_i")),
            w_density=float(bo_meta.get("w_density", 0.08)),
            w_te=float(bo_meta.get("w_te", 0.06)),
            w_en=float(bo_meta.get("w_en", 0.04)),
            bias=float(bo_meta.get("bias", 0.0)),
            clamp=tuple(bo_meta["clamp"]) if bo_meta.get("clamp") is not None else None,
            freeze=bool(bo_meta.get("freeze", True)),
        )
        model.attach_boundary_operator(bo)

    return model


def load_deeponet_plasma_checkpoint_weights(model: Any, weights: Any) -> None:
    model.load_state_dict_numpy({k: np.asarray(weights[k], dtype=np.float32) for k in weights.files})

    ph = getattr(model, "poisson_head", None)
    if ph is not None:
        ph_weights = {
            k.split("poisson_head::", 1)[1]: np.asarray(weights[k], dtype=np.float32)
            for k in weights.files
            if k.startswith("poisson_head::")
        }
        if ph_weights:
            ph.load_state_dict_numpy(ph_weights)

    bo = getattr(model, "boundary_operator", None)
    if bo is not None:
        bo_weights = {
            k.split("boundary_operator::", 1)[1]: np.asarray(weights[k], dtype=np.float32)
            for k in weights.files
            if k.startswith("boundary_operator::")
        }
        if bo_weights:
            bo.load_state_dict_numpy(bo_weights)
