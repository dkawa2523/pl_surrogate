import math

import numpy as np

from plasma_surrogate.eval.metrics import finite_pair_stats, r2_masked, rmse_masked, uniformity


def test_masked_rmse_uses_float64_for_large_linear_density() -> None:
    y_true = np.full((2, 2), 1.0e18, dtype=np.float32)
    y_pred = y_true + np.full((2, 2), 1.0e9, dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.float32)

    value = rmse_masked(y_true, y_pred, mask)

    assert math.isfinite(value)


def test_uniformity_uses_float64_for_large_linear_density() -> None:
    values = np.array([1.0e18, 1.1e18, 0.9e18], dtype=np.float32)

    value = uniformity(values)

    assert math.isfinite(value)
    assert value > 0.0


def test_masked_metrics_return_nan_when_active_prediction_is_nonfinite() -> None:
    y_true = np.ones((2, 2), dtype=np.float32)
    y_pred = np.ones((2, 2), dtype=np.float32)
    y_pred[0, 0] = np.inf
    mask = np.ones((2, 2), dtype=np.float32)

    assert math.isnan(rmse_masked(y_true, y_pred, mask))
    assert math.isnan(r2_masked(y_true, y_pred, mask))


def test_finite_pair_stats_counts_masked_nonfinite_values() -> None:
    y_true = np.ones((2, 2), dtype=np.float32)
    y_pred = np.ones((2, 2), dtype=np.float32)
    y_pred[0, 0] = np.nan
    mask = np.array([[1, 0], [1, 1]], dtype=np.float32)

    stats = finite_pair_stats(y_true, y_pred, mask)

    assert stats["n_active"] == 3.0
    assert stats["n_finite"] == 2.0
    assert stats["n_nonfinite"] == 1.0
    assert stats["finite_ratio"] == 2.0 / 3.0
