from __future__ import annotations

import numpy as np

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider


def test_fixed_geometry_provider_loads_context(geometry_root):
    provider = FixedGeometryProvider(geometry_root)
    ctx = provider.get()
    assert ctx.mask_plasma.shape == (8, 8)
    assert ctx.distance_any.shape == (8, 8)
    assert "wafer_mask" in ctx.regions


def test_fixed_geometry_provider_builds_coord_from_rz(tmp_path):
    root = tmp_path / "csv_ds"
    g = root / "geometry"
    g.mkdir(parents=True)
    np.save(g / "mask_plasma.npy", np.ones((4, 3), dtype=np.float32))
    np.save(g / "eps.npy", np.ones((4, 3), dtype=np.float32))
    np.save(g / "r_coords.npy", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.save(g / "z_coords.npy", np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32))
    provider = FixedGeometryProvider(root, coord_grid_source="coord_grid")
    ctx = provider.get()
    assert ctx.coord_grid.shape == (2, 4, 3)
    assert ctx.coord_source == "rz_linear"
    assert np.allclose(ctx.coord_grid[0, 0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(ctx.coord_grid[1, :, 0], np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32))


def test_fixed_geometry_provider_builds_signed_distance_when_missing(tmp_path):
    root = tmp_path / "csv_ds_signed"
    g = root / "geometry"
    g.mkdir(parents=True)
    mask = np.zeros((5, 5), dtype=np.float32)
    mask[1:4, 1:4] = 1.0
    np.save(g / "mask_plasma.npy", mask)
    np.save(g / "eps.npy", np.ones((5, 5), dtype=np.float32))
    provider = FixedGeometryProvider(root, coord_grid_source="normalized")
    ctx = provider.get()
    assert ctx.distance_signed is not None
    signed = np.asarray(ctx.distance_signed, dtype=np.float32)
    outside = mask <= 0.5
    inside = mask > 0.5
    assert np.any(signed[outside] < 0.0)
    assert np.any(signed[inside] >= 0.0)
