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
    assert resolved["group_weighting"] == {"mode": "none"}


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
    assert resolved["group_weighting"] == {"mode": "none"}


def test_resolve_loss_protocol_v2_accepts_huber_as_opt_in():
    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v2",
            "supervised": {
                "type": "huber",
                "huber_delta": 1.0,
                "mask": "plasma_only",
            },
        },
        target_role_schema=_role_schema(),
    )

    assert resolved["supervised"] == {
        "type": "huber",
        "huber_delta": 1.0,
        "mask": "plasma_only",
    }
    assert resolved["group_weighting"] == {"mode": "none"}


def test_resolve_loss_protocol_v2_without_role_schema_uses_global_mask_only():
    resolved = resolve_loss_protocol({"protocol": "plasma_surrogate_v2"}, target_role_schema={})
    sup = resolved["supervised"]

    assert sup == {"type": "mse", "mask": "plasma_only"}
    assert resolved["group_weighting"] == {"mode": "none"}


def test_v2_rejects_unknown_supervised_mask_instead_of_falling_back_to_all_pixels() -> None:
    with pytest.raises(ValueError, match="supervised.mask"):
        resolve_loss_protocol(
            {"protocol": "plasma_surrogate_v2", "supervised": {"mask": "plamsa_only"}},
            target_role_schema=_role_schema(),
        )


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


def test_resolve_loss_protocol_v2_accepts_group_weighting_override():
    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v2",
            "group_weighting": {"mode": "uniform_by_group"},
        },
        target_role_schema=_role_schema(),
    )

    assert resolved["group_weighting"] == {"mode": "uniform_by_group"}


def test_resolve_loss_protocol_v2_accepts_weighted_group_weighting():
    resolved = resolve_loss_protocol(
        {
            "protocol": "plasma_surrogate_v2",
            "group_weighting": {
                "mode": "weighted_by_group",
                "weights": {"density": 1.5, "temperature": 1.0, "electrostatic": 1.0},
            },
        },
        target_role_schema=_role_schema(),
    )

    assert resolved["group_weighting"] == {
        "mode": "weighted_by_group",
        "weights": {"density": 1.5, "temperature": 1.0, "electrostatic": 1.0},
    }


def test_resolve_loss_protocol_v2_rejects_unknown_group_weighting_mode():
    with pytest.raises(ValueError, match="group_weighting.mode"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "group_weighting": {"mode": "positive_penalty"},
            },
            target_role_schema=_role_schema(),
        )


def test_resolve_loss_protocol_v2_rejects_group_weights_without_weighted_mode():
    with pytest.raises(ValueError, match="weighted_by_group"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "group_weighting": {"mode": "uniform_by_group", "weights": {"density": 1.0}},
            },
            target_role_schema=_role_schema(),
        )


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, float("inf"), float("nan")])
def test_resolve_loss_protocol_v2_rejects_bad_group_weight(bad_weight):
    with pytest.raises(ValueError, match="group_weighting.weights"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "group_weighting": {"mode": "weighted_by_group", "weights": {"density": bad_weight}},
            },
            target_role_schema=_role_schema(),
        )


@pytest.mark.parametrize("bad_delta", [0.0, -1.0, float("inf"), float("nan")])
def test_resolve_loss_protocol_v2_rejects_bad_huber_delta(bad_delta):
    with pytest.raises(ValueError, match="huber_delta"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "supervised": {"type": "huber", "huber_delta": bad_delta},
            },
            target_role_schema=_role_schema(),
        )


def test_resolve_loss_protocol_v2_rejects_unknown_supervised_type():
    with pytest.raises(ValueError, match="supervised.type"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "supervised": {"type": "mae"},
            },
            target_role_schema=_role_schema(),
        )


@pytest.mark.parametrize("alias_cfg", [{"base": "mse"}, {"type": "huber", "delta": 1.0}])
def test_resolve_loss_protocol_v2_rejects_removed_supervised_aliases(alias_cfg):
    with pytest.raises(ValueError, match="removed"):
        resolve_loss_protocol(
            {
                "protocol": "plasma_surrogate_v2",
                "supervised": alias_cfg,
            },
            target_role_schema=_role_schema(),
        )


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"extra": True},
        {"supervised": {"normalization": "sample_mean"}},
        {"supervised": {"delta_by_var": {"electron_density": 1.0}}},
        {"group_weighting": {"mode": "none", "target_weights": {"density": 1.0}}},
    ],
)
def test_resolve_loss_protocol_v2_rejects_unsupported_contract_keys(bad_cfg):
    raw = {"protocol": "plasma_surrogate_v2", **bad_cfg}
    with pytest.raises(ValueError, match="plasma_surrogate_v2|group_weighting"):
        resolve_loss_protocol(raw, target_role_schema=_role_schema())
