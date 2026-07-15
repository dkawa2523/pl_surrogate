from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.preprocessing.scalers import ScalerFactory, TransformBundle, fit_scalers_train_only


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


def test_fit_scalers_train_only_uses_each_train_cases_plasma_mask_without_holdout_leakage():
    cond = np.arange(4, dtype=np.float32).reshape(-1, 1)
    fields = np.array(
        [
            [[1.0, 101.0], [102.0, 103.0]],
            [[201.0, 2.0], [202.0, 203.0]],
            [[1.0e6, 1.0e6], [1.0e6, 1.0e6]],
            [[-1.0e6, -1.0e6], [-1.0e6, -1.0e6]],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((4, 2, 2), dtype=np.float32)
    masks[0, 0, 0] = 1.0
    masks[1, 0, 1] = 1.0
    masks[2, 1, 0] = 1.0
    masks[3, 1, 1] = 1.0
    train_idx = np.array([0, 1], dtype=np.int64)

    bundle = fit_scalers_train_only(
        cond,
        {"ne": fields},
        train_idx,
        mask_plasma=masks,
        target_transforms={"ne": {"fit_scope": "plasma_only"}},
    )
    stats = bundle.y_scalers["ne"].to_dict()
    assert float(stats["mean"][0]) == pytest.approx(1.5)
    assert float(stats["std"][0]) == pytest.approx(0.5)

    changed_holdout = fields.copy()
    changed_holdout[2:] *= 1000.0
    repeated = fit_scalers_train_only(
        cond,
        {"ne": changed_holdout},
        train_idx,
        mask_plasma=masks,
        target_transforms={"ne": {"fit_scope": "plasma_only"}},
    )
    assert repeated.y_scalers["ne"].to_dict() == stats


@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 2)])
def test_fit_scalers_train_only_accepts_static_plasma_mask_forms(shape):
    cond = np.arange(2, dtype=np.float32).reshape(-1, 1)
    fields = np.array([[[1.0, 100.0], [3.0, 100.0]], [[5.0, 100.0], [7.0, 100.0]]])
    mask = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32).reshape(shape)
    bundle = fit_scalers_train_only(
        cond,
        {"ne": fields},
        np.array([0, 1], dtype=np.int64),
        mask_plasma=mask,
        target_transforms={"ne": {"fit_scope": "plasma_only"}},
    )
    assert float(bundle.y_scalers["ne"].to_dict()["mean"][0]) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "mask,match",
    [
        (np.zeros((3, 2, 2), dtype=np.float32), "case axis"),
        (np.zeros((2, 2, 2), dtype=np.float32), "no active cells"),
        (np.array([[[1.0, np.nan], [0.0, 0.0]]], dtype=np.float32), "non-finite"),
        (np.array([[[1.0, 0.25], [0.0, 0.0]]], dtype=np.float32), "must be binary"),
    ],
)
def test_fit_scalers_train_only_rejects_invalid_plasma_masks(mask, match):
    with pytest.raises(ValueError, match=match):
        fit_scalers_train_only(
            np.arange(2, dtype=np.float32).reshape(-1, 1),
            {"ne": np.ones((2, 2, 2), dtype=np.float32)},
            np.array([0], dtype=np.int64),
            mask_plasma=mask,
            target_transforms={"ne": {"fit_scope": "plasma_only"}},
        )


def test_fit_scalers_train_only_rejects_plasma_mask_spatial_mismatch():
    with pytest.raises(ValueError, match="spatial shape does not match"):
        fit_scalers_train_only(
            np.arange(2, dtype=np.float32).reshape(-1, 1),
            {"ne": np.ones((2, 3, 2), dtype=np.float32)},
            np.array([0], dtype=np.int64),
            mask_plasma=np.ones((2, 2), dtype=np.float32),
            target_transforms={"ne": {"fit_scope": "plasma_only"}},
        )


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


def test_robust_scaler_serializes_and_roundtrips():
    scaler = ScalerFactory.create("robust").fit(
        np.array([[0.0, 10.0], [2.0, 12.0], [100.0, 14.0]], dtype=np.float32)
    )
    transformed = scaler.transform(np.array([[2.0, 12.0]], dtype=np.float32))
    assert np.allclose(transformed, np.zeros((1, 2), dtype=np.float64))

    restored = ScalerFactory.from_dict(scaler.to_dict())
    roundtrip = restored.inverse_transform(transformed)
    assert np.allclose(roundtrip, np.array([[2.0, 12.0]], dtype=np.float64))

    constant = ScalerFactory.create("robust").fit(np.full((3, 1), 5.0, dtype=np.float32))
    constant_payload = constant.to_dict()
    assert constant_payload["iqr"] == [1.0]
    assert np.allclose(constant.transform(np.array([[5.0]], dtype=np.float32)), np.zeros((1, 1), dtype=np.float64))

    bundle = TransformBundle.from_dict(
        cond_scaler=scaler.to_dict(),
        y_scalers={"ne": {"type": "none"}},
        y_order=["ne"],
        cond_dim=2,
    )
    assert bundle.cond_dim == 2


def test_target_value_transforms_floor_and_roundtrip():
    cond = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = {
        "ne": np.array([[0.0], [0.01], [10.0]], dtype=np.float32),
        "ni": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        "Te": np.array([[3.0], [4.0], [5.0]], dtype=np.float32),
        "phi": np.array([[-2.0], [0.0], [2.0]], dtype=np.float32),
    }
    bundle = fit_scalers_train_only(
        cond,
        y,
        np.array([0, 1, 2], dtype=np.int64),
        y_scaler_type="none",
        target_transforms={
            "ne": {"value_transform": "log10_floor", "floor": 1.0e-3, "scaler": "none"},
            "ni": {"value_transform": "log1p", "floor": 0.0, "scaler": "none"},
            "phi": {"value_transform": "signed_log1p", "scaler": "none"},
        },
    )
    assert bundle.target_transforms["Te"]["floor"] == 0.0
    assert bundle.target_transforms["phi"]["floor"] == 0.0

    fields = {
        "ne": np.array([[0.0, 0.01]], dtype=np.float32),
        "ni": np.array([[0.0, 2.0]], dtype=np.float32),
        "Te": np.array([[3.0, 4.0]], dtype=np.float32),
        "phi": np.array([[-2.0, 2.0]], dtype=np.float32),
    }
    transformed = bundle.transform_field_dict(fields)
    assert transformed["ne"][0, 0] == pytest.approx(np.log10(1.0e-3))
    assert transformed["ni"][0, 1] == pytest.approx(np.log1p(2.0))
    assert transformed["phi"][0, 0] == pytest.approx(-np.log1p(2.0))

    inverse = bundle.inverse_field_dict(transformed)
    assert inverse["ne"][0, 0] == pytest.approx(1.0e-3)
    assert np.allclose(inverse["ni"], fields["ni"], atol=1e-6)
    assert np.allclose(inverse["phi"], fields["phi"], atol=1e-6)

    floor_default_bundle = fit_scalers_train_only(
        cond,
        y,
        np.array([0, 1, 2], dtype=np.int64),
        y_scaler_type="none",
        target_transforms={"ne": {"value_transform": "log10_floor", "scaler": "none"}},
    )
    assert floor_default_bundle.target_transforms["ne"]["floor"] == 0.0
    inverse_tiny = floor_default_bundle.inverse_field_dict({"ne": np.array([[-1000.0]], dtype=np.float32)})
    assert inverse_tiny["ne"][0, 0] == pytest.approx(1.0e-30)


def test_log10_transform_keeps_strict_positive_requirement():
    cond = np.array([[0.0], [1.0]], dtype=np.float32)
    y = {
        "ne": np.array([[0.0], [1.0]], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="requires strictly positive"):
        fit_scalers_train_only(
            cond,
            y,
            np.array([0, 1], dtype=np.int64),
            target_transforms={"ne": {"value_transform": "log10"}},
        )


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
            "Te": {
                "value_transform": "log1p",
                "clip": {"mode": "quantile", "q_low": 0.01, "q_high": 0.99},
            },
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

    extreme_scaled = {
        "Te": np.array([[-1.0e6, 1.0e6]], dtype=np.float32),
        "phi": np.array([[0.0, 0.0]], dtype=np.float32),
    }
    inverse = clipped.inverse_field_dict(extreme_scaled)
    te_clip = clipped.target_transforms["Te"]["clip"]
    assert np.all(np.isfinite(inverse["Te"]))
    assert inverse["Te"].min() == pytest.approx(np.expm1(float(te_clip["clip_low"])), rel=1e-5)
    assert inverse["Te"].max() == pytest.approx(np.expm1(float(te_clip["clip_high"])), rel=1e-5)


def test_inverse_quantile_clip_rejects_unfitted_artifact_bounds():
    cond = np.array([[0.0], [1.0]], dtype=np.float32)
    y = {"ne": np.array([[1.0], [10.0]], dtype=np.float32)}
    bundle = fit_scalers_train_only(cond, y, np.array([0, 1], dtype=np.int64))
    bundle.target_transforms["ne"]["clip"] = {"mode": "quantile", "q_low": 0.1, "q_high": 0.9}
    with pytest.raises(ValueError, match="missing fitted clip_low/clip_high"):
        bundle.inverse_field_dict({"ne": np.array([[0.0]], dtype=np.float32)})


def test_physical_bounds_are_stored_in_physical_and_transformed_spaces():
    cond = np.array([[0.0], [1.0]], dtype=np.float32)
    y = {"ne": np.array([[1.0e7], [1.0e20]], dtype=np.float32)}
    bundle = fit_scalers_train_only(
        cond,
        y,
        np.array([0, 1], dtype=np.int64),
        target_transforms={
            "ne": {
                "value_transform": "log10_floor",
                "floor": 1.0e-30,
                "scaler": "none",
                "clip": {"mode": "physical_bounds", "min": 1.0e8, "max": 1.0e19},
            }
        },
    )
    clip = bundle.target_transforms["ne"]["clip"]
    assert clip["min"] == pytest.approx(1.0e8)
    assert clip["max"] == pytest.approx(1.0e19)
    assert clip["clip_low"] == pytest.approx(8.0)
    assert clip["clip_high"] == pytest.approx(19.0)
    transformed = bundle.transform_field_dict(y)["ne"]
    np.testing.assert_allclose(transformed.reshape(-1), np.array([8.0, 19.0], dtype=np.float32))
    inverse = bundle.inverse_field_dict({"ne": np.array([[-1.0e6, 1.0e6]], dtype=np.float32)})["ne"]
    np.testing.assert_allclose(inverse.reshape(-1), np.array([1.0e8, 1.0e19], dtype=np.float32), rtol=1e-5)
