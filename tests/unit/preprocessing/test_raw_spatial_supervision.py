from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.preprocessing.spatial_features import (
    CaseSpatialFeatureSource,
    materialize_raw_spatial_batch,
)


def _source() -> CaseSpatialFeatureSource:
    h, w = 2, 3
    mask_plasma = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    distance_signed = np.asarray([[1.0, 2.0, -1.0], [3.0, -2.0, -3.0]], dtype=np.float32)
    distance_any = np.abs(distance_signed).astype(np.float32)
    static_data = np.stack([mask_plasma, distance_signed, distance_any], axis=0)
    case_data = np.stack(
        [np.full((1, h, w), float(case_id + 1), dtype=np.float32) for case_id in range(3)],
        axis=0,
    )
    distance_cfg = {
        "mode": "bounded",
        "signed_tanh_tau": 2.0,
        "proximity_tau": 4.0,
        "replace_distance_any": True,
    }
    scaler = {
        "contract_version": 2,
        "enabled": True,
        "mode": "zscore",
        "input_space": "post_distance_transform",
        "distance_transform_effective": distance_cfg,
        "channels": {
            "distance_signed": {"type": "zscore", "mean": [0.0], "std": [2.0]},
        },
    }
    return CaseSpatialFeatureSource(
        channels=("distance_signed", "distance_any", "mask_coil"),
        h=h,
        w=w,
        n_cases=3,
        source="test",
        static_data=static_data,
        static_channels=("mask_plasma", "distance_signed", "distance_any"),
        case_data=case_data,
        case_channels=("mask_coil",),
        distance_transform_cfg=distance_cfg,
        coord_feature_scaler_artifact=scaler,
    )


def test_raw_batch_preserves_requested_order_and_pretransform_values() -> None:
    source = _source()

    raw = source.raw_batch(
        np.asarray([2, 0], dtype=np.int64),
        ("mask_coil", "distance_any", "mask_plasma"),
    )

    assert raw.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(raw[0, ..., 0], np.full((2, 3), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[1, ..., 0], np.full((2, 3), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[:, ..., 1], np.broadcast_to(np.abs(source.static_data[1]), (2, 2, 3)))
    np.testing.assert_array_equal(raw[:, ..., 2], np.broadcast_to(source.static_data[0], (2, 2, 3)))

    transformed = source.batch(np.asarray([2], dtype=np.int64))[0]
    expected_signed = np.tanh(source.static_data[1] / 2.0) / 2.0
    expected_any = np.exp(-source.static_data[2] / 4.0)
    np.testing.assert_allclose(transformed[..., 0], expected_signed, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(transformed[..., 1], expected_any, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_array_equal(transformed[..., 2], np.full((2, 3), 3.0, dtype=np.float32))


def test_raw_batch_resolves_nested_subset_indices_to_original_cases() -> None:
    source = _source().subset(np.asarray([2, 0], dtype=np.int64)).subset(np.asarray([1], dtype=np.int64))

    raw = source.raw_batch(np.asarray([0], dtype=np.int64), ("mask_coil", "distance_signed"))

    assert source.n_cases == 1
    np.testing.assert_array_equal(raw[0, ..., 0], np.full((2, 3), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[0, ..., 1], source.static_data[1])


def test_raw_batch_rejects_unknown_channel_and_unsafe_indices() -> None:
    source = _source()

    with pytest.raises(ValueError, match="unknown_geometry"):
        source.raw_batch(np.asarray([0], dtype=np.int64), ("unknown_geometry",))
    with pytest.raises(IndexError, match="must be >= 0"):
        source.raw_batch(np.asarray([-1], dtype=np.int64), ("mask_plasma",))
    with pytest.raises(IndexError, match="out of range"):
        source.raw_batch(np.asarray([3], dtype=np.int64), ("mask_plasma",))


def test_raw_materializer_uses_explicit_raw_api_before_fallback() -> None:
    source = _source().subset(np.asarray([2, 0], dtype=np.int64))
    fallback = {"mask_coil": np.full((2, 3), -100.0, dtype=np.float32)}

    raw = materialize_raw_spatial_batch(
        source,
        np.asarray([1, 0], dtype=np.int64),
        channels=("mask_coil",),
        fallback=fallback,
    )

    assert raw is not None
    np.testing.assert_array_equal(raw[0, ..., 0], np.full((2, 3), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[1, ..., 0], np.full((2, 3), 3.0, dtype=np.float32))


def test_raw_materializer_safely_broadcasts_and_slices_mapping_fallback() -> None:
    class BatchOnlySource:
        def batch(self, _indices: np.ndarray) -> np.ndarray:
            raise AssertionError("transformed batch() must not be used for raw supervision")

    static_mask = np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)
    case_distance = np.stack(
        [np.full((2, 2), float(case_id), dtype=np.float32) for case_id in range(4)],
        axis=0,
    )

    raw = materialize_raw_spatial_batch(
        BatchOnlySource(),
        np.asarray([3, 1], dtype=np.int64),
        channels=("distance_any", "mask_plasma"),
        fallback={"mask_plasma": static_mask, "distance_any": case_distance},
    )

    assert raw is not None
    assert raw.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(raw[0, ..., 0], np.full((2, 2), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[1, ..., 0], np.full((2, 2), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(raw[0, ..., 1], static_mask)
    np.testing.assert_array_equal(raw[1, ..., 1], static_mask)


def test_raw_materializer_broadcasts_singleton_case_fallback_before_shuffle() -> None:
    singleton_mask = np.ones((1, 2, 2), dtype=np.float32)

    raw = materialize_raw_spatial_batch(
        None,
        np.asarray([3, 1, 2], dtype=np.int64),
        channels=("mask_plasma",),
        fallback={"mask_plasma": singleton_mask},
    )

    assert raw is not None
    assert raw.shape == (3, 2, 2, 1)
    np.testing.assert_array_equal(
        raw[..., 0],
        np.broadcast_to(singleton_mask, (3, 2, 2)),
    )


def test_raw_materializer_requires_complete_shape_aligned_fallback() -> None:
    indices = np.asarray([1], dtype=np.int64)

    with pytest.raises(ValueError, match="missing channels"):
        materialize_raw_spatial_batch(
            None,
            indices,
            channels=("mask_plasma", "distance_any"),
            fallback={"mask_plasma": np.ones((2, 2), dtype=np.float32)},
        )
    with pytest.raises(IndexError, match="cannot select case indices"):
        materialize_raw_spatial_batch(
            None,
            np.asarray([2], dtype=np.int64),
            channels=("distance_any",),
            fallback={"distance_any": np.ones((2, 2, 2), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="one spatial shape"):
        materialize_raw_spatial_batch(
            None,
            indices,
            channels=("mask_plasma", "distance_any"),
            fallback={
                "mask_plasma": np.ones((2, 2), dtype=np.float32),
                "distance_any": np.ones((2, 3), dtype=np.float32),
            },
        )


def test_raw_materializer_without_raw_source_or_fallback_returns_none() -> None:
    assert (
        materialize_raw_spatial_batch(
            np.zeros((2, 2, 2, 1), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            channels=("mask_plasma",),
        )
        is None
    )
