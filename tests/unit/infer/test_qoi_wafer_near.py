from __future__ import annotations

import numpy as np

from plasma_surrogate.infer.qoi import axisymmetric_weighted_cv, uniformity_values_for_region


def test_wafer_near_region_selects_only_adjacent_plasma_layers() -> None:
    target = np.arange(25, dtype=np.float32).reshape(5, 5)
    plasma = np.ones((5, 5), dtype=np.float32)
    wafer = np.zeros((5, 5), dtype=np.float32)
    wafer[4, :] = 1.0
    plasma[wafer > 0.5] = 0.0

    values, meta = uniformity_values_for_region(
        target,
        mask_plasma=plasma,
        wafer_mask=wafer,
        region="wafer_near",
        mid_height_band_px=1,
    )

    np.testing.assert_array_equal(values, target[3, :])
    assert meta["uniformity_region"] == "wafer_near"
    assert meta["uniformity_wafer_near_layers"] == 1.0


def test_axisymmetric_weighted_cv_uses_annular_radius_weights() -> None:
    mean, cv = axisymmetric_weighted_cv(
        np.array([1.0, 3.0], dtype=np.float32),
        np.array([1.0, 3.0], dtype=np.float32),
    )
    assert mean == 2.5
    assert np.isclose(cv, np.sqrt(0.75) / 2.5)
