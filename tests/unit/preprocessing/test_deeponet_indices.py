from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.preprocessing.sampling import build_deeponet_indices


def test_build_deeponet_indices_is_deterministic():
    a = build_deeponet_indices(n_points=64, n_sensors=8, n_queries=16, seed=7)
    b = build_deeponet_indices(n_points=64, n_sensors=8, n_queries=16, seed=7)
    assert np.array_equal(a["sensor_indices"], b["sensor_indices"])
    assert np.array_equal(a["query_indices"], b["query_indices"])
    assert a["sensor_indices"].dtype == np.int64
    assert a["query_indices"].dtype == np.int64


def test_build_deeponet_indices_validates_sizes():
    with pytest.raises(ValueError):
        build_deeponet_indices(n_points=0, n_sensors=8, n_queries=16, seed=0)
    with pytest.raises(ValueError):
        build_deeponet_indices(n_points=64, n_sensors=0, n_queries=16, seed=0)
    with pytest.raises(ValueError):
        build_deeponet_indices(n_points=64, n_sensors=8, n_queries=0, seed=0)