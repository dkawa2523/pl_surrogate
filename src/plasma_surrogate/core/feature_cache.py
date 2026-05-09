"""Shared helpers for deterministic featurization cache preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.data.geometry_provider import GeometryProviderLike
from plasma_surrogate.features.geometry_feature_store import GeometryFeatureStore


def prepare_feature_cache(
    *,
    run_root: str | Path,
    geometry_provider: GeometryProviderLike,
    features_cfg: dict[str, Any] | None = None,
    axis_mode: str = "steady",
    axis_value: float = 0.0,
    geom_ref: dict[str, Any] | None = None,
) -> tuple[GeometryFeatureStore, dict[str, Any]]:
    """Prepare feature cache and write geometry cache index for reproducibility."""
    root = Path(run_root)
    feat_cfg = dict(features_cfg or {})
    cache_cfg = dict(feat_cfg.get("cache", {}))
    store = GeometryFeatureStore(
        root / "featurization",
        delta_bulk=float(feat_cfg.get("delta_bulk", 3.0)),
        strict_hash_check=bool(cache_cfg.get("strict_hash_check", True)),
    )
    ref = dict(geom_ref or {"geom_id": "default"})
    meta = store.prepare(
        geom_ref=ref,
        axis_value=float(axis_value),
        axis_mode=str(axis_mode),
        geometry_provider=geometry_provider,
    )
    ArtifactStore(root / "featurization").save_json(
        "geometry_cache_index.json",
        {"default": {"geom_ref": ref, "feature_hash": meta["feature_hash"], "hashes": meta["hashes"]}},
    )
    return store, meta
