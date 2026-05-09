from __future__ import annotations

from plasma_surrogate.preprocessing.sampling import build_phase_wrap_pairs, build_time_adjacent_pairs


def test_build_phase_wrap_pairs_uses_axis_endpoints():
    sample_ids = [10, 11, 12, 13]
    axis_values = [0.40, 0.98, 0.02, 0.70]
    pairs = build_phase_wrap_pairs(sample_ids, axis_values=axis_values)
    assert pairs == [(12, 11)]


def test_build_time_adjacent_pairs_sorted_by_axis():
    sample_ids = [20, 21, 22, 23]
    axis_values = [0.50, 0.10, 0.30, 0.20]
    pairs = build_time_adjacent_pairs(sample_ids, axis_values=axis_values)
    assert pairs == [(21, 23), (23, 22), (22, 20)]
