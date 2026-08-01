"""Shared helpers for deterministic InferenceEngine construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_surrogate.core.input_modes import load_checkpoint_metadata_with_input_mode
from plasma_surrogate.infer.engine import InferenceEngine


def build_inference_engine(
    *,
    model: Any,
    cond_schema: Any,
    axis_schema: Any,
    geometry_provider: Any,
    output_dir: Path,
    transform_bundle: Any,
    cond_stats: dict[str, Any] | None,
    phi_mode: str,
    phi_hybrid_steps: int,
    poisson_refine_iters: int,
    ood_cfg: dict[str, Any] | None,
    feature_store: Any,
    coord_scaler: dict[str, Any] | None,
    coord_feature_scaler: dict[str, Any] | None,
    coord_feature_pack: dict[str, Any] | None,
    coord_distance_transform_stats: dict[str, Any] | None,
    input_mode_meta: dict[str, Any],
    checkpoint_meta_path: Path | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
    checkpoint_input_mode_meta: dict[str, Any] | None = None,
    target_role_schema: dict[str, Any] | None = None,
    deeponet_head: Any | None = None,
    grid_input_features_cfg: dict[str, Any] | None = None,
    structure_descriptor_pack: dict[str, Any] | None = None,
    latent_feature_pack: dict[str, Any] | None = None,
) -> InferenceEngine:
    if checkpoint_meta_path is not None:
        if checkpoint_meta is not None or checkpoint_input_mode_meta is not None:
            raise ValueError(
                "build_inference_engine expects either checkpoint_meta_path or "
                "checkpoint_meta/checkpoint_input_mode_meta, not both"
            )
        checkpoint_meta, checkpoint_input_mode_meta = load_checkpoint_metadata_with_input_mode(checkpoint_meta_path)
    effective_checkpoint_meta = dict(checkpoint_meta or {})
    effective_checkpoint_input_mode_meta = dict(checkpoint_input_mode_meta or {})

    return InferenceEngine(
        model=model,
        cond_schema=cond_schema,
        axis_schema=axis_schema,
        geometry_provider=geometry_provider,
        output_dir=output_dir,
        transform_bundle=transform_bundle,
        cond_stats=dict(cond_stats or {}),
        phi_mode=phi_mode,
        phi_hybrid_steps=int(phi_hybrid_steps),
        poisson_refine_iters=int(poisson_refine_iters),
        ood_cfg=dict(ood_cfg or {}),
        feature_store=feature_store,
        coord_scaler=dict(coord_scaler or {}),
        coord_feature_scaler=dict(coord_feature_scaler or {}),
        coord_feature_pack=dict(coord_feature_pack or {}),
        coord_distance_transform_stats=dict(coord_distance_transform_stats or {}),
        input_mode=str(input_mode_meta.get("input_mode_effective", "")),
        input_mode_meta=dict(input_mode_meta),
        checkpoint_input_mode_meta=effective_checkpoint_input_mode_meta,
        checkpoint_meta=effective_checkpoint_meta,
        target_role_schema=dict(target_role_schema or {}),
        deeponet_head=deeponet_head,
        grid_input_features_cfg=dict(grid_input_features_cfg or {}),
        structure_descriptor_pack=dict(structure_descriptor_pack or {}),
        latent_feature_pack=dict(latent_feature_pack or {}),
    )


__all__ = ["build_inference_engine"]
