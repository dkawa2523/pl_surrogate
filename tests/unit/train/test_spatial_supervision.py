from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.preprocessing.spatial_features import CaseSpatialFeatureSource
from plasma_surrogate.train.spatial_supervision import (
    materialize_supervised_geometry,
    required_raw_supervision_channels,
    slice_case_map,
)


def test_materialize_supervised_geometry_prefers_unscaled_case_source() -> None:
    static = np.stack(
        [
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
            np.full((2, 3), -2.0, dtype=np.float32),
        ]
    )
    case = np.zeros((2, 1, 2, 3), dtype=np.float32)
    case[0, 0] = 3.0
    case[1, 0] = 7.0
    source = CaseSpatialFeatureSource(
        channels=("mask_plasma", "distance_any", "distance_signed"),
        h=2,
        w=3,
        n_cases=2,
        source="test",
        static_data=static,
        static_channels=("mask_plasma", "distance_signed", "unused"),
        case_data=case,
        case_channels=("distance_any",),
        distance_transform_cfg={"mode": "bounded", "signed_tanh_tau": 1.0, "proximity_tau": 1.0},
    )

    actual = materialize_supervised_geometry(
        source,
        np.asarray([1], dtype=np.int64),
        mask=np.zeros((2, 3), dtype=np.float32),
        distance_any=np.zeros((2, 3), dtype=np.float32),
        distance_signed=np.zeros((2, 3), dtype=np.float32),
    )

    np.testing.assert_array_equal(actual["mask"], np.ones((1, 2, 3), dtype=np.float32))
    np.testing.assert_array_equal(actual["distance_any"], np.full((1, 2, 3), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(actual["distance_signed"], np.full((1, 2, 3), 2.0, dtype=np.float32))


def test_materialize_supervised_geometry_slices_explicit_case_fallback() -> None:
    masks = np.stack(
        [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)],
        axis=0,
    )
    actual = materialize_supervised_geometry(
        None,
        np.asarray([1], dtype=np.int64),
        mask=masks,
        distance_any=np.full((2, 2), 4.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(actual["mask"], np.ones((1, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(actual["distance_any"], np.full((1, 2, 2), 4.0, dtype=np.float32))


def test_slice_case_map_broadcasts_singleton_case_axis() -> None:
    values = np.ones((1, 2, 3), dtype=np.float32)
    actual = slice_case_map(values, np.asarray([3, 1], dtype=np.int64), key="wafer_mask")
    assert actual is not None
    assert actual.shape == (2, 2, 3)
    np.testing.assert_array_equal(actual, 1.0)


def test_raw_source_and_static_fallback_are_composed_per_channel() -> None:
    source = CaseSpatialFeatureSource(
        channels=("mask_plasma",),
        h=2,
        w=2,
        n_cases=1,
        source="test",
        static_data=np.ones((1, 2, 2), dtype=np.float32),
        static_channels=("mask_plasma",),
        case_data=np.zeros((1, 0, 2, 2), dtype=np.float32),
        case_channels=(),
    )
    actual = materialize_supervised_geometry(
        source,
        np.asarray([0], dtype=np.int64),
        mask=np.zeros((2, 2), dtype=np.float32),
        distance_any=np.full((2, 2), 5.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(actual["mask"], np.ones((1, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(actual["distance_any"], np.full((1, 2, 2), 5.0, dtype=np.float32))


def test_raw_source_must_declare_channel_ownership() -> None:
    class AmbiguousRawSource:
        def raw_batch(self, indices, channels):
            return np.zeros((len(indices), 2, 2, len(channels)), dtype=np.float32)

    with pytest.raises(ValueError, match="must declare static_channels"):
        materialize_supervised_geometry(
            AmbiguousRawSource(),
            np.asarray([0], dtype=np.int64),
            mask=np.ones((2, 2), dtype=np.float32),
            distance_any=np.zeros((2, 2), dtype=np.float32),
        )


def test_materialize_supervised_geometry_composes_nearest_configured_boundary() -> None:
    source = CaseSpatialFeatureSource(
        channels=("mask_plasma", "distance_any", "part_sdf_nearest"),
        h=2,
        w=2,
        n_cases=2,
        source="test",
        static_data=np.stack(
            [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 5.0, dtype=np.float32)]
        ),
        static_channels=("mask_plasma", "distance_any"),
        case_data=np.asarray(
            [
                [[[9.0, 2.0], [-1.0, 8.0]]],
                [[[4.0, -3.0], [7.0, 0.5]]],
            ],
            dtype=np.float32,
        ),
        case_channels=("part_sdf_nearest",),
    )

    actual = materialize_supervised_geometry(
        source,
        np.asarray([1, 0], dtype=np.int64),
        mask=None,
        distance_any=None,
        boundary_distance_channels=["distance_any", "part_sdf_nearest"],
    )

    np.testing.assert_array_equal(
        actual["distance_any"],
        np.asarray(
            [
                [[4.0, 3.0], [5.0, 0.5]],
                [[5.0, 2.0], [1.0, 5.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_required_raw_supervision_channels_uses_loss_boundary_definition() -> None:
    required = required_raw_supervision_channels(
        loss_cfg={
            "supervised": {
                "mask": "plasma_only",
                "spatial": {
                    "boundary_weight": 0.25,
                    "boundary_distance_channels": ["distance_any", "part_sdf_nearest"],
                },
            }
        },
        selection_cfg={"mode": "last"},
    )

    assert required == ["mask_plasma", "distance_any", "part_sdf_nearest"]


def test_spatial_selection_uses_the_same_loss_boundary_definition() -> None:
    required = required_raw_supervision_channels(
        loss_cfg={
            "supervised": {
                "mask": "plasma_only",
                "spatial": {
                    "boundary_weight": 0.0,
                    "boundary_distance_channels": ["distance_any", "part_sdf_nearest"],
                },
            }
        },
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "spatial": {"boundary_weight": 0.25},
        },
    )

    assert required == ["mask_plasma", "distance_any", "part_sdf_nearest"]
