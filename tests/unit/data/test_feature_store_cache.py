from __future__ import annotations

from pathlib import Path

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.features.geometry_feature_store import GeometryFeatureStore


def test_geometry_feature_store_caches_and_reloads(geometry_root: Path, tmp_path: Path):
    provider = FixedGeometryProvider(geometry_root)
    store = GeometryFeatureStore(tmp_path / "featurization", delta_bulk=2.0, strict_hash_check=True)
    meta0 = store.prepare(
        geom_ref={"geom_id": "default"},
        axis_value=0.0,
        axis_mode="steady",
        geometry_provider=provider,
    )
    meta1 = store.prepare(
        geom_ref={"geom_id": "default"},
        axis_value=0.0,
        axis_mode="steady",
        geometry_provider=provider,
    )
    assert meta0["feature_hash"] == meta1["feature_hash"]

    ctx = store.get_context(
        geom_ref={"geom_id": "default"},
        axis_value=0.0,
        axis_mode="steady",
        geometry_provider=provider,
    )
    assert "mask_bulk" in ctx.regions
    assert "bc_basis" in ctx.regions
    assert "phi_bc_ext" in ctx.regions
