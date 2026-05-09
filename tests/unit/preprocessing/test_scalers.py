from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.preprocessing.scalers import TransformBundle, fit_scalers_train_only


def test_fit_scalers_train_only_uses_train_indices():
    cond = np.array([[0.0], [1.0], [100.0]], dtype=np.float32)
    y = {
        "ne": np.array([[1.0], [10.0], [1000.0]], dtype=np.float32),
        "ni": np.array([[1.5], [11.0], [999.0]], dtype=np.float32),
        "Te": np.array([[0.0], [10.0], [1000.0]], dtype=np.float32),
        "phi": np.array([[0.0], [10.0], [1000.0]], dtype=np.float32),
    }
    train_idx = np.array([0, 1], dtype=np.int64)
    bundle = fit_scalers_train_only(cond, y, train_idx)
    transformed = bundle.cond_scaler.transform(np.array([[0.0], [1.0]], dtype=np.float32))
    assert abs(float(np.mean(transformed))) < 1e-6
    assert "ne" in bundle.y_scalers


def test_fit_scalers_train_only_fits_field_distribution():
    cond = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    # Train cases have identical case-means (=50), so mean-only fitting would collapse std to ~0.
    y = {
        "ne": np.array([[1.0, 100.0], [1.0, 100.0], [9.0, 9.0]], dtype=np.float32),
        "ni": np.array([[2.0, 100.0], [2.0, 100.0], [9.0, 9.0]], dtype=np.float32),
        "Te": np.array([[0.0, 100.0], [0.0, 100.0], [9.0, 9.0]], dtype=np.float32),
        "phi": np.array([[0.0, 100.0], [0.0, 100.0], [9.0, 9.0]], dtype=np.float32),
    }
    train_idx = np.array([0, 1], dtype=np.int64)
    bundle = fit_scalers_train_only(cond, y, train_idx)
    stats = bundle.y_scalers["ne"].to_dict()
    assert stats["type"] == "zscore"
    assert float(stats["std"][0]) > 10.0


def test_fit_scalers_train_only_plasma_only_ignores_fill_region():
    cond = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = {
        "ne": np.array(
            [
                [[10.0, 0.0], [10.0, 0.0]],
                [[12.0, 0.0], [12.0, 0.0]],
                [[50.0, 0.0], [50.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "ni": np.array(
            [
                [[9.0, 0.0], [9.0, 0.0]],
                [[11.0, 0.0], [11.0, 0.0]],
                [[40.0, 0.0], [40.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "Te": np.array(
            [
                [[3.0, 0.0], [3.0, 0.0]],
                [[4.0, 0.0], [4.0, 0.0]],
                [[20.0, 0.0], [20.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "phi": np.array(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0]],
                [[9.0, 0.0], [9.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    }
    train_idx = np.array([0, 1], dtype=np.int64)
    mask = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    bundle = fit_scalers_train_only(cond, y, train_idx, mask_plasma=mask, scaler_fit_policy="plasma_only")
    stats = bundle.y_scalers["ne"].to_dict()
    assert bundle.fit_policy == "plasma_only"
    assert bundle.mask_applied is True
    # mean should be based on [10,10,12,12] and not include fill zeros.
    assert np.isclose(float(stats["mean"][0]), 11.0)


def test_transform_bundle_cond_dim_roundtrip_and_fail_fast():
    cond = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    y = {
        "ne": np.array([[1.0], [10.0], [20.0]], dtype=np.float32),
        "ni": np.array([[1.0], [10.0], [20.0]], dtype=np.float32),
        "Te": np.array([[0.0], [10.0], [20.0]], dtype=np.float32),
        "phi": np.array([[0.0], [10.0], [20.0]], dtype=np.float32),
    }
    bundle = fit_scalers_train_only(cond, y, np.array([0, 1], dtype=np.int64))
    payload = bundle.to_dict()
    assert payload["cond_dim"] == 3

    restored = TransformBundle.from_dict(
        cond_scaler=payload["cond_scaler"],
        y_scalers=payload["y_scalers"],
        y_order=payload["y_order"],
        cond_dim=payload["cond_dim"],
    )
    ok = restored.transform_cond(np.zeros((2, 3), dtype=np.float32))
    assert ok.shape == (2, 3)

    with pytest.raises(ValueError, match="expected 3, got 2"):
        restored.transform_cond(np.zeros((2, 2), dtype=np.float32))


def test_transform_bundle_infers_legacy_cond_dim_from_scaler_stats():
    legacy_cond_scaler = {
        "type": "zscore",
        "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    legacy_y_scalers = {
        "ne": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "ni": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "Te": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "phi": {"type": "zscore", "mean": [0.0], "std": [1.0]},
    }
    bundle = TransformBundle.from_dict(cond_scaler=legacy_cond_scaler, y_scalers=legacy_y_scalers)
    assert bundle.cond_dim == 5
    out = bundle.transform_cond(np.zeros((1, 5), dtype=np.float32))
    assert out.shape == (1, 5)


def test_fit_scalers_train_only_target_transforms_quantile_clip_tracks_stats():
    cond = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = {
        "ne": np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
        "ni": np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
        "Te": np.array([[3.0], [3.2], [200.0]], dtype=np.float32),
        "phi": np.array([[5.0], [6.0], [-400.0]], dtype=np.float32),
    }
    train_idx = np.array([0, 1, 2], dtype=np.int64)

    standard = fit_scalers_train_only(cond, y, train_idx)
    clipped = fit_scalers_train_only(
        cond,
        y,
        train_idx,
        target_transforms={
            "Te": {"clip": {"mode": "quantile", "q_low": 0.01, "q_high": 0.99}},
            "phi": {"clip": {"mode": "quantile", "q_low": 0.01, "q_high": 0.99}},
        },
    )

    std_te = float(standard.y_scalers["Te"].to_dict()["std"][0])
    clipped_te = float(clipped.y_scalers["Te"].to_dict()["std"][0])
    std_phi = float(standard.y_scalers["phi"].to_dict()["std"][0])
    clipped_phi = float(clipped.y_scalers["phi"].to_dict()["std"][0])

    assert "clip_low" in clipped.target_transforms["Te"]["clip"]
    assert "clip_high" in clipped.target_transforms["Te"]["clip"]
    assert clipped_te < std_te
    assert clipped_phi < std_phi