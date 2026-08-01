"""Shared runtime context builders for CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.core.feature_cache import prepare_feature_cache
from plasma_surrogate.core.input_modes import (
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    build_input_mode_effective_metadata,
    normalize_input_mode_cfg,
    validate_input_mode_cfg,
)
from plasma_surrogate.core.run_bundle import RunBundle, RunBundleLoader, ensure_preprocess_contract, require_artifacts
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.features.geometry_feature_store import GeometryFeatureStore


@dataclass
class RuntimeContext:
    cfg: dict[str, Any]
    run_dir: Path
    dataset: Any
    axis_mode: str
    feature_store: GeometryFeatureStore
    feature_meta: dict[str, Any]
    bundle: RunBundle | None = None


def _load_cfg(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_norm = normalize_input_mode_cfg(cfg)
    validate_input_mode_cfg(cfg_norm)
    return cfg_norm


def _resolve_axis_mode(cfg: dict[str, Any], *, fallback: str = "steady") -> str:
    pre_cfg = cfg.get("preprocessing", {})
    axis_schema_cfg = pre_cfg.get("axis_schema", {})
    if "mode" in axis_schema_cfg:
        return str(axis_schema_cfg["mode"])
    ds_cfg = cfg.get("dataset", {})
    return str(ds_cfg.get("axis_mode", fallback))


def build_preprocess_context(config_path: str | Path) -> RuntimeContext:
    cfg = _load_cfg(config_path)
    run_dir = Path(cfg.get("run_dir", "runs/mainline"))
    dataset = load_dataset(cfg, run_dir)
    axis_mode = _resolve_axis_mode(cfg, fallback="steady")
    input_mode_meta = build_input_mode_effective_metadata(cfg)
    provider_mode = str(input_mode_meta.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed"))
    geometry_provider = build_geometry_provider(dataset.geometry_root, provider_mode=provider_mode)
    feature_store, feature_meta = prepare_feature_cache(
        run_root=run_dir,
        geometry_provider=geometry_provider,
        features_cfg=cfg.get("features", {}),
        axis_mode=axis_mode,
        axis_value=0.0,
        geom_ref={"geom_id": "default"},
    )
    return RuntimeContext(
        cfg=cfg,
        run_dir=run_dir,
        dataset=dataset,
        axis_mode=axis_mode,
        feature_store=feature_store,
        feature_meta=feature_meta,
    )


def build_train_context(
    config_path: str | Path,
    *,
    ensure_preprocess: bool = True,
    on_missing_preprocess: Any | None = None,
) -> RuntimeContext:
    ctx = build_preprocess_context(config_path)
    if ensure_preprocess:
        ensure_preprocess_contract(ctx.run_dir, on_missing=on_missing_preprocess)
    bundle = RunBundleLoader.load(ctx.run_dir)
    return RuntimeContext(
        cfg=ctx.cfg,
        run_dir=ctx.run_dir,
        dataset=ctx.dataset,
        axis_mode=bundle.axis_schema_obj().mode,
        feature_store=ctx.feature_store,
        feature_meta=ctx.feature_meta,
        bundle=bundle,
    )


def build_infer_context(
    config_path: str | Path,
    *,
    ensure_preprocess: bool = True,
    ensure_checkpoint: bool = True,
    on_missing_preprocess: Any | None = None,
    on_missing_checkpoint: Any | None = None,
) -> RuntimeContext:
    ctx = build_train_context(
        config_path,
        ensure_preprocess=ensure_preprocess,
        on_missing_preprocess=on_missing_preprocess,
    )
    if ensure_checkpoint:
        try:
            require_artifacts(ctx.run_dir, ["checkpoints/meta.json"])
        except FileNotFoundError:
            if on_missing_checkpoint is None:
                raise
            on_missing_checkpoint()
            require_artifacts(ctx.run_dir, ["checkpoints/meta.json"])
    bundle = RunBundleLoader.load(ctx.run_dir)
    return RuntimeContext(
        cfg=ctx.cfg,
        run_dir=ctx.run_dir,
        dataset=ctx.dataset,
        axis_mode=bundle.axis_schema_obj().mode,
        feature_store=ctx.feature_store,
        feature_meta=ctx.feature_meta,
        bundle=bundle,
    )
