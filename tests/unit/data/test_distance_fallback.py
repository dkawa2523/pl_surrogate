from __future__ import annotations

import numpy as np

from plasma_surrogate.data.geometry_context import build_distance_fields


def test_build_distance_fields_sdf_fallback():
    mask = np.ones((5, 5), dtype=np.float32)
    mask[0, :] = 0
    d_any, d0 = build_distance_fields(mask)
    assert d_any.shape == (5, 5)
    assert d0.shape == (5, 5)
    assert d_any[1, 2] == 0.0
    assert d_any[2, 2] > d_any[1, 1]