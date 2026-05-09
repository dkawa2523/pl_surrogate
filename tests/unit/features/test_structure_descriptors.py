from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.data.geometry_context import GeometryContext, build_coord_grid, build_signed_distance_fields
from plasma_surrogate.features.structure_descriptors import (
    STRUCT_DESC_V1,
    STRUCT_DESC_V2,
    build_struct_desc_v1,
    build_struct_desc_v2,
    build_structure_descriptor,
)


def _geom_ctx_with_parts() -> GeometryContext:
    h, w = 8, 8
    mask_plasma = np.ones((h, w), dtype=np.float32)
    mask_plasma[0, :] = 0.0
    mask_plasma[:, 0] = 0.0
    distance_signed = build_signed_distance_fields(mask_plasma).astype(np.float32)
    distance_any = np.abs(distance_signed).astype(np.float32)
    part_stack = np.zeros((2, h, w), dtype=np.float32)
    part_stack[0, 2:4, 2:4] = 1.0
    part_stack[1, 4:6, 5:7] = 1.0
    return GeometryContext(
        mask_plasma=mask_plasma,
        distance_any=distance_any,
        dist0=distance_any.copy(),
        eps=np.ones((h, w), dtype=np.float32),
        coord_grid=build_coord_grid((h, w)).astype(np.float32),
        distance_signed=distance_signed,
        regions={"part_mask_stack": part_stack},
    )


def test_struct_desc_v1_is_deterministic() -> None:
    geom = _geom_ctx_with_parts()
    first = build_struct_desc_v1(geom)
    second = build_structure_descriptor(STRUCT_DESC_V1, geom)
    assert first.profile == STRUCT_DESC_V1
    assert second.profile == STRUCT_DESC_V1
    assert first.feature_names == second.feature_names
    assert np.allclose(first.vector, second.vector)


def test_struct_desc_v1_has_fixed_feature_order_and_dim() -> None:
    geom = _geom_ctx_with_parts()
    pack = build_struct_desc_v1(geom)
    expected_dim = 6 + 2 * 9
    assert int(pack.vector.shape[0]) == expected_dim
    assert len(pack.feature_names) == expected_dim
    assert np.all(np.isfinite(pack.vector))


def test_struct_desc_v2_keeps_order_but_normalizes_pixel_scale_features() -> None:
    geom = _geom_ctx_with_parts()
    raw = build_struct_desc_v1(geom)
    normalized = build_struct_desc_v2(geom)
    via_registry = build_structure_descriptor(STRUCT_DESC_V2, geom)
    assert normalized.profile == STRUCT_DESC_V2
    assert normalized.feature_names == raw.feature_names
    assert np.allclose(normalized.vector, via_registry.vector)
    assert np.max(np.abs(normalized.vector)) < np.max(np.abs(raw.vector))


def test_struct_desc_v1_requires_part_mask_stack() -> None:
    geom = _geom_ctx_with_parts()
    geom.regions = {}
    with pytest.raises(ValueError, match="part_mask_stack"):
        build_struct_desc_v1(geom)
