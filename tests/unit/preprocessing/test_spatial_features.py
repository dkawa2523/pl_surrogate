from __future__ import annotations

import numpy as np

from plasma_surrogate.preprocessing.spatial_features import (
    part_sdf_maps_from_stack,
    part_sdf_summary_maps_from_stack,
    part_source_maps_from_stack,
    sdf_from_part_mask,
)


def _two_active_parts() -> np.ndarray:
    stack = np.zeros((2, 9, 13), dtype=np.float32)
    stack[0, 0:2, 0:2] = 1.0
    stack[1, 0:2, 3:5] = 1.0
    return stack


def test_part_summary_ignores_fixed_inactive_slots() -> None:
    active = _two_active_parts()
    fixed_slots = np.zeros((6, *active.shape[1:]), dtype=np.float32)
    fixed_slots[:2] = active

    expected = part_sdf_summary_maps_from_stack(active)
    actual = part_sdf_summary_maps_from_stack(fixed_slots)

    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert float(np.max(actual["part_sdf_second"])) > float(max(active.shape[1:]))


def test_part_summary_uses_safe_fill_when_fewer_than_two_parts_are_active() -> None:
    stack = np.zeros((6, 9, 13), dtype=np.float32)
    stack[0, 0:2, 0:2] = 1.0
    fill = float(9 + 13 + 1)

    summary = part_sdf_summary_maps_from_stack(stack)

    np.testing.assert_array_equal(
        summary["part_sdf_nearest"],
        np.abs(sdf_from_part_mask(stack[0])),
    )
    np.testing.assert_array_equal(
        summary["part_sdf_second"],
        np.full((9, 13), fill, dtype=np.float32),
    )


def test_empty_part_slots_use_grid_diameter_upper_bound_fill() -> None:
    empty = np.zeros((9, 13), dtype=np.float32)
    fill = float(9 + 13 + 1)

    np.testing.assert_array_equal(
        sdf_from_part_mask(empty),
        np.full((9, 13), fill, dtype=np.float32),
    )
    maps = part_sdf_maps_from_stack(np.zeros((0, 9, 13), dtype=np.float32), slot_count=2)
    assert set(maps) == {"sdf_coil_01", "sdf_coil_02"}
    for value in maps.values():
        np.testing.assert_array_equal(value, np.full((9, 13), fill, dtype=np.float32))


def test_part_source_maps_are_order_invariant_and_additive() -> None:
    stack = _two_active_parts()
    expected = part_source_maps_from_stack(stack, source_tau=2.0)
    actual = part_source_maps_from_stack(stack[::-1], source_tau=2.0)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name])
    assert float(expected["part_source_sum"][0, 0]) > 1.0
    assert float(expected["part_source_sum"][0, 3]) > 1.0


def test_empty_part_source_maps_are_finite() -> None:
    maps = part_source_maps_from_stack(np.zeros((6, 9, 13), dtype=np.float32))
    assert np.isfinite(maps["part_sdf_union"]).all()
    np.testing.assert_array_equal(maps["part_source_sum"], np.zeros((9, 13), dtype=np.float32))
