from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.train.loss_composer import compose_supervised_numpy
from plasma_surrogate.train.loss_protocols import (
    LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
    loss_protocol_metadata,
    resolve_loss_protocol,
)


def _schema() -> dict:
    return {
        "targets": [
            {"id": "ne", "field_family": "density"},
            {"id": "ni", "field_family": "density"},
            {"id": "Te", "field_family": "temperature"},
            {"id": "phi", "field_family": "electrostatic"},
        ]
    }


def test_v3_resolves_explicit_case_balanced_spatial_defaults() -> None:
    resolved = resolve_loss_protocol(
        {"protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3},
        target_role_schema=_schema(),
    )

    assert resolved == {
        "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
        "protocol_effective": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
        "supervised": {
            "type": "huber",
            "huber_delta": 1.0,
            "mask": "plasma_only",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "weighted",
            "spatial": {
                "gradient_weight": 0.10,
                "gradient_spacing": [1.0, 1.0],
                "multiscale_weight": 0.05,
                "multiscale_scales": [2, 4],
                "boundary_weight": 0.25,
                "boundary_band_px": 2.0,
                "boundary_distance_channels": ["distance_any"],
            },
        },
        "group_weighting": {"mode": "uniform_by_group"},
        "target_role_schema": _schema(),
    }


def test_v3_metadata_hashes_the_complete_effective_formula() -> None:
    resolved = resolve_loss_protocol(
        {"protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3},
        target_role_schema=_schema(),
    )
    metadata = loss_protocol_metadata(resolved)
    changed = resolve_loss_protocol(
        {
            "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
            "supervised": {"spatial": {"boundary_weight": 0.5}},
        },
        target_role_schema=_schema(),
    )

    assert metadata["loss_protocol_effective"] == LOSS_PROTOCOL_PLASMA_SURROGATE_V3
    assert metadata["loss_protocol_version"] == 3
    assert len(metadata["loss_protocol_definition_hash"]) == 64
    assert "boundary_weight" in metadata["loss_protocol_effective_config"]
    assert metadata["loss_protocol_definition_hash"] != loss_protocol_metadata(changed)[
        "loss_protocol_definition_hash"
    ]


def test_v3_resolved_config_is_reentrant_and_directly_composable() -> None:
    schema = _schema()
    resolved = resolve_loss_protocol(
        {"protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3},
        target_role_schema=schema,
    )

    assert resolve_loss_protocol(resolved) == resolved

    field = np.zeros((1, 3, 3), dtype=np.float32)
    loss, _grads, parts = compose_supervised_numpy(
        {name: field.copy() for name in ("ne", "ni", "Te", "phi")},
        {name: field.copy() for name in ("ne", "ni", "Te", "phi")},
        y_order=["ne", "ni", "Te", "phi"],
        mask=np.ones_like(field),
        distance_any=np.zeros_like(field),
        loss_cfg=resolved,
    )

    assert loss == pytest.approx(0.0)
    assert parts["loss_supervised_group_density"] == pytest.approx(0.0)
    assert parts["loss_supervised_group_temperature"] == pytest.approx(0.0)
    assert parts["loss_supervised_group_electrostatic"] == pytest.approx(0.0)


def test_v3_deep_merges_spatial_overrides_without_hiding_defaults() -> None:
    resolved = resolve_loss_protocol(
        {
            "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
            "supervised": {
                "spatial": {
                    "boundary_weight": 0.25,
                    "boundary_band_px": 3.0,
                    "gradient_spacing": [0.5, 2.0],
                }
            },
        },
        target_role_schema=_schema(),
    )

    spatial = resolved["supervised"]["spatial"]
    assert spatial["boundary_weight"] == pytest.approx(0.25)
    assert spatial["boundary_band_px"] == pytest.approx(3.0)
    assert spatial["gradient_spacing"] == [0.5, 2.0]
    assert spatial["gradient_weight"] == pytest.approx(0.10)
    assert spatial["multiscale_weight"] == pytest.approx(0.05)
    assert spatial["multiscale_scales"] == [2, 4]


def test_v3_canonicalizes_boundary_distance_channel_names() -> None:
    resolved = resolve_loss_protocol(
        {
            "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
            "supervised": {
                "spatial": {
                    "boundary_distance_channels": [" distance_any ", "part_sdf_nearest"]
                }
            },
        },
        target_role_schema=_schema(),
    )

    assert resolved["supervised"]["spatial"]["boundary_distance_channels"] == [
        "distance_any",
        "part_sdf_nearest",
    ]


@pytest.mark.parametrize(
    "spatial,match",
    [
        ({"gradient_spacing": [1.0]}, "gradient_spacing"),
        ({"gradient_spacing": [1.0, 0.0]}, "gradient_spacing"),
        ({"multiscale_scales": [2.5]}, "multiscale_scales"),
        ({"multiscale_scales": [2, 2]}, "multiscale_scales"),
        ({"boundary_band_px": 0.0}, "boundary_band_px"),
        ({"boundary_distance_channels": []}, "boundary_distance_channels"),
        ({"boundary_distance_channels": ["distance_any", "distance_any"]}, "boundary_distance_channels"),
        ({"boundary_distance_channels": "distance_any"}, "boundary_distance_channels"),
        ({"unknown": 1.0}, "unsupported supervised.spatial"),
    ],
)
def test_v3_rejects_invalid_or_unknown_spatial_contract(spatial: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_loss_protocol(
            {
                "protocol": LOSS_PROTOCOL_PLASMA_SURROGATE_V3,
                "supervised": {"spatial": spatial},
            },
            target_role_schema=_schema(),
        )


def test_v2_defaults_remain_spatially_inert() -> None:
    resolved = resolve_loss_protocol({"protocol": "plasma_surrogate_v2"}, target_role_schema=_schema())

    assert resolved["supervised"] == {"type": "mse", "mask": "plasma_only"}
    assert resolved["group_weighting"] == {"mode": "none"}
    assert "spatial" not in resolved["supervised"]


def test_v3_rejects_unknown_supervised_mask() -> None:
    with pytest.raises(ValueError, match="supervised.mask"):
        resolve_loss_protocol(
            {"protocol": "plasma_surrogate_v3", "supervised": {"mask": "plamsa_only"}},
            target_role_schema=_schema(),
        )


def test_v3_boundary_supervision_requires_plasma_mask_mode() -> None:
    with pytest.raises(ValueError, match="requires supervised.mask=plasma_only"):
        resolve_loss_protocol(
            {"protocol": "plasma_surrogate_v3", "supervised": {"mask": "none"}},
            target_role_schema=_schema(),
        )

    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v3",
            "supervised": {"mask": "none", "spatial": {"boundary_weight": 0.0}},
        },
        target_role_schema=_schema(),
    )
    assert resolved["supervised"]["mask"] == "none"
