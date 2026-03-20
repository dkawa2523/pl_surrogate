from __future__ import annotations

import numpy as np

from plasma_surrogate.preprocessing.sampling import build_point_pools


def test_build_point_pools_has_required_keys():
    mask = np.ones((6, 6), dtype=np.float32)
    dist = np.arange(36, dtype=np.float32).reshape(6, 6)
    pools = build_point_pools(mask, dist, delta_edge=3.0, delta_bulk=20.0)
    assert "idx_plasma" in pools
    assert "idx_boundary_band" in pools
    assert "idx_bulk" in pools
    assert len(pools["idx_plasma"]) == 36
