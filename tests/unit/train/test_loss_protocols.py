import pytest

from plasma_surrogate.train.loss_protocols import resolve_loss_protocol


def _role_schema():
    return {
        "targets": [
            {
                "id": "electron_density",
                "role": "density_electron",
                "positive": True,
                "field_family": "density",
                "default_region": "plasma_only",
            },
            {
                "id": "electron_temperature",
                "role": "temperature_electron",
                "positive": True,
                "field_family": "temperature",
            },
            {
                "id": "plasma_potential",
                "role": "potential",
                "positive": False,
                "field_family": "electrostatic",
            },
        ],
    }


def test_resolve_loss_protocol_without_protocol_preserves_user_config():
    raw = {"supervised": {"type": "mse", "mask": "none"}}

    resolved = resolve_loss_protocol(raw, target_role_schema=_role_schema())

    assert resolved["protocol_effective"] == "none"
    assert resolved["supervised"] == raw["supervised"]
    assert "region_balance" not in resolved["supervised"]
    assert raw == {"supervised": {"type": "mse", "mask": "none"}}


def test_resolve_loss_protocol_v2_expands_product_defaults_from_role_schema():
    resolved = resolve_loss_protocol({"protocol": "plasma_surrogate_v2"}, target_role_schema=_role_schema())
    sup = resolved["supervised"]

    assert resolved["protocol_effective"] == "plasma_surrogate_v2"
    assert sup["type"] == "huber"
    assert sup["mask"] == "plasma_only"
    assert sup["region_balance"] == {
        "enabled": True,
        "mode": "replace",
        "weight_boundary_in": 0.6,
        "weight_plasma_mid": 0.0,
        "weight_deep_plasma": 0.4,
    }
    assert sup["spatial_consistency"]["enabled"] is True
    assert sup["spatial_consistency"]["apply_region"] == "target_region"
    assert sup["spatial_consistency"]["multiscale"]["enabled"] is True
    assert sup["positive_penalty"] == {
        "enabled": True,
        "vars": ["electron_density", "electron_temperature"],
        "floor": 0.0,
        "lambda": 0.01,
    }
    assert sup["relative_weighting"] == {"enabled": False}
    assert sup["target_region_by_var"] == {
        "electron_density": "plasma_only",
        "electron_temperature": "plasma_only",
        "plasma_potential": "all_domain",
    }


def test_resolve_loss_protocol_v2_user_override_wins():
    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v2",
            "supervised": {
                "type": "mse",
                "spatial_consistency": {"enabled": False},
                "region_balance": {"weight_boundary_in": 0.75},
                "positive_penalty": {"enabled": False},
            },
        },
        target_role_schema=_role_schema(),
    )
    sup = resolved["supervised"]

    assert sup["type"] == "mse"
    assert sup["spatial_consistency"]["enabled"] is False
    assert sup["region_balance"]["weight_boundary_in"] == 0.75
    assert sup["region_balance"]["weight_deep_plasma"] == 0.4
    assert sup["positive_penalty"]["enabled"] is False
    assert sup["positive_penalty"]["vars"] == ["electron_density", "electron_temperature"]


def test_resolve_loss_protocol_v2_without_role_schema_uses_global_mask_only():
    resolved = resolve_loss_protocol({"protocol": "plasma_surrogate_v2"}, target_role_schema={})
    sup = resolved["supervised"]

    assert sup["mask"] == "plasma_only"
    assert sup["positive_penalty"] == {"enabled": False, "vars": [], "floor": 0.0, "lambda": 0.01}
    assert "target_region_by_var" not in sup


@pytest.mark.parametrize(
    "deprecated_key",
    ["region_weighting", "density_positivity_penalty", "density_relative_weighting"],
)
def test_resolve_loss_protocol_v2_rejects_deprecated_supervised_keys(deprecated_key):
    with pytest.raises(ValueError, match=deprecated_key):
        resolve_loss_protocol(
            {"protocol": "plasma_surrogate_v2", "supervised": {deprecated_key: {"enabled": True}}},
            target_role_schema=_role_schema(),
        )
