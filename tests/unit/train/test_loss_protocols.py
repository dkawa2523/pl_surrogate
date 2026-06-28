import pytest

from plasma_surrogate.train.loss_contract import REMOVED_SUPERVISED_KEYS
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
    assert sup == {"type": "mse", "mask": "plasma_only"}


def test_resolve_loss_protocol_v2_user_override_wins():
    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v2",
            "supervised": {
                "type": "mse",
                "mask": "none",
            },
        },
        target_role_schema=_role_schema(),
    )
    sup = resolved["supervised"]

    assert sup == {"type": "mse", "mask": "none"}


def test_resolve_loss_protocol_v2_without_role_schema_uses_global_mask_only():
    resolved = resolve_loss_protocol({"protocol": "plasma_surrogate_v2"}, target_role_schema={})
    sup = resolved["supervised"]

    assert sup == {"type": "mse", "mask": "plasma_only"}


@pytest.mark.parametrize(
    "deprecated_key",
    REMOVED_SUPERVISED_KEYS,
)
def test_resolve_loss_protocol_v2_rejects_research_or_deprecated_supervised_keys(deprecated_key):
    with pytest.raises(ValueError, match=deprecated_key):
        resolve_loss_protocol(
            {"protocol": "plasma_surrogate_v2", "supervised": {deprecated_key: {"enabled": True}}},
            target_role_schema=_role_schema(),
        )
