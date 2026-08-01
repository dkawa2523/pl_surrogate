from __future__ import annotations

import pytest

from plasma_surrogate.models.heads.role_grouped import configure_output_head_metadata


def _schema() -> dict:
    return {
        "targets": [
            {"id": "electron_density", "field_family": "density", "role": "density_electron"},
            {"id": "plasma_potential", "field_family": "electrostatic", "role": "potential"},
        ]
    }


def _configure(output_heads: dict):
    model = type("DummyModel", (), {})()
    configure_output_head_metadata(
        model,
        output_heads=output_heads,
        output_keys=["electron_density", "plasma_potential"],
        target_role_schema=_schema(),
        cfg_prefix="train.fno",
    )
    return model


def test_group_options_default_is_metadata_only() -> None:
    model = _configure(
        {
            "mode": "role_grouped",
            "group_options": {"electrostatic": {"head": "default"}},
        }
    )

    assert model.output_heads_mode == "role_grouped"
    assert model.output_head_group_options == {"electrostatic": {"head": "default"}}
    assert model.output_heads_effective["group_options"] == {"electrostatic": {"head": "default"}}


def test_group_options_accepts_spatial_refine() -> None:
    model = _configure(
        {
            "mode": "role_grouped",
            "group_options": {"density": {"head": "spatial_refine"}},
        }
    )

    assert model.output_head_group_options == {"density": {"head": "spatial_refine"}}
    assert model.output_heads_effective["group_options"] == {"density": {"head": "spatial_refine"}}


def test_group_options_reject_poisson_hybrid_until_implemented() -> None:
    with pytest.raises(ValueError, match="not implemented yet"):
        _configure(
            {
                "mode": "role_grouped",
                "group_options": {"electrostatic": {"head": "poisson_hybrid"}},
            }
        )


def test_group_options_reject_unknown_group() -> None:
    with pytest.raises(ValueError, match="does not match a resolved target group"):
        _configure(
            {
                "mode": "role_grouped",
                "group_options": {"not_a_group": {"head": "default"}},
            }
        )


def test_group_options_reject_unsupported_keys() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        _configure(
            {
                "mode": "role_grouped",
                "group_options": {"electrostatic": {"head": "default", "alpha": 0.5}},
            }
        )


def test_shared_output_head_rejects_group_options() -> None:
    with pytest.raises(ValueError, match="group_options requires"):
        _configure(
            {
                "mode": "shared",
                "group_options": {"electrostatic": {"head": "default"}},
            }
        )
