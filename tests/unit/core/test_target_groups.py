from __future__ import annotations

import pytest

from plasma_surrogate.core.target_groups import resolve_target_groups


def _target_role_schema() -> dict:
    return {
        "version": 1,
        "vars": [
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "floating_signal",
            "ion_flux",
        ],
        "targets": [
            {
                "id": "ion_density",
                "role": "density_ion",
                "field_family": "density",
            },
            {
                "id": "electron_temperature",
                "role": "temperature_electron",
                "field_family": "temperature",
            },
            {
                "id": "plasma_potential",
                "role": "potential",
                "field_family": "electrostatic",
            },
            {
                "id": "floating_signal",
                "role": "diagnostic",
            },
            {
                "id": "ion_flux",
                "role": "flux_ion",
                "field_family": "flux",
            },
        ],
        "role_to_targets": {
            "density_ion": ["ion_density"],
            "temperature_electron": ["electron_temperature"],
            "potential": ["plasma_potential"],
            "diagnostic": ["floating_signal"],
            "flux_ion": ["ion_flux"],
        },
        "field_family_to_targets": {
            "density": ["ion_density"],
            "temperature": ["electron_temperature"],
            "electrostatic": ["plasma_potential"],
            "flux": ["ion_flux"],
        },
        "positive_targets": ["ion_density", "electron_temperature"],
    }


def test_resolve_target_groups_from_field_family() -> None:
    groups = resolve_target_groups(
        output_vars=[
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "floating_signal",
            "ion_flux",
        ],
        target_role_schema=_target_role_schema(),
    )

    assert list(groups) == ["density", "temperature", "electrostatic", "default", "flux"]
    assert groups["density"].targets == ("ion_density",)
    assert groups["density"].field_family == "density"
    assert groups["density"].source == "field_family"
    assert groups["density"].roles == ("density_ion",)
    assert groups["temperature"].targets == ("electron_temperature",)
    assert groups["electrostatic"].targets == ("plasma_potential",)
    assert groups["flux"].targets == ("ion_flux",)


def test_resolve_target_groups_uses_default_for_missing_field_family() -> None:
    groups = resolve_target_groups(
        output_vars=[
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "floating_signal",
            "ion_flux",
        ],
        target_role_schema=_target_role_schema(),
    )

    assert groups["default"].targets == ("floating_signal",)
    assert groups["default"].field_family is None
    assert groups["default"].roles == ("diagnostic",)


def test_resolve_target_groups_applies_custom_groups_first() -> None:
    schema = _target_role_schema()
    schema["vars"] = [
        "ion_density",
        "electron_temperature",
        "plasma_potential",
        "floating_signal",
    ]
    schema["targets"] = schema["targets"][:4]

    groups = resolve_target_groups(
        output_vars=[
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "floating_signal",
        ],
        target_role_schema=schema,
        custom_groups={"charged": ["plasma_potential", "ion_density"]},
    )

    assert list(groups) == ["charged", "temperature", "default"]
    assert groups["charged"].targets == ("ion_density", "plasma_potential")
    assert groups["charged"].source == "custom"
    assert groups["charged"].field_family is None
    assert "density" not in groups
    assert "electrostatic" not in groups


def test_resolve_target_groups_rejects_unknown_custom_target() -> None:
    with pytest.raises(ValueError, match="unknown targets"):
        resolve_target_groups(
            output_vars=[
                "ion_density",
                "electron_temperature",
                "plasma_potential",
                "floating_signal",
                "ion_flux",
            ],
            target_role_schema=_target_role_schema(),
            custom_groups={"bad": ["does_not_exist"]},
        )


def test_resolve_target_groups_rejects_duplicate_custom_target() -> None:
    with pytest.raises(ValueError, match="already assigned"):
        resolve_target_groups(
            output_vars=[
                "ion_density",
                "electron_temperature",
                "plasma_potential",
                "floating_signal",
                "ion_flux",
            ],
            target_role_schema=_target_role_schema(),
            custom_groups={
                "a": ["ion_density"],
                "b": ["ion_density"],
            },
        )


def test_resolve_target_groups_preserves_output_vars_order() -> None:
    schema = {
        "version": 1,
        "vars": ["b", "a", "c"],
        "targets": [
            {"id": "a", "field_family": "density"},
            {"id": "c", "field_family": "density"},
            {"id": "b", "field_family": "density"},
        ],
    }

    groups = resolve_target_groups(output_vars=["b", "a", "c"], target_role_schema=schema)

    assert groups["density"].targets == ("b", "a", "c")


def test_resolve_target_groups_omits_empty_groups() -> None:
    groups = resolve_target_groups(
        output_vars=[
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "floating_signal",
            "ion_flux",
        ],
        target_role_schema=_target_role_schema(),
        custom_groups={"empty": []},
    )

    assert "empty" not in groups
    assert "electric_field" not in groups


def test_resolve_target_groups_rejects_empty_custom_group_when_configured() -> None:
    with pytest.raises(ValueError, match="must contain at least one target"):
        resolve_target_groups(
            output_vars=["ion_density", "electron_temperature"],
            target_role_schema={
                "targets": [
                    {"id": "ion_density", "field_family": "density"},
                    {"id": "electron_temperature", "field_family": "temperature"},
                ]
            },
            custom_groups={"empty": []},
            allow_empty_custom_groups=False,
        )


def test_resolve_target_groups_can_require_custom_groups_to_cover_all_outputs() -> None:
    with pytest.raises(ValueError, match="missing output_vars"):
        resolve_target_groups(
            output_vars=["ion_density", "electron_temperature"],
            target_role_schema={
                "targets": [
                    {"id": "ion_density", "field_family": "density"},
                    {"id": "electron_temperature", "field_family": "temperature"},
                ]
            },
            custom_groups={"density": ["ion_density"]},
            custom_groups_missing="error",
            allow_empty_custom_groups=False,
        )


def test_resolve_target_groups_can_put_unspecified_custom_targets_in_default_group() -> None:
    groups = resolve_target_groups(
        output_vars=["ion_density", "electron_temperature", "plasma_potential"],
        target_role_schema={
            "targets": [
                {"id": "ion_density", "field_family": "density"},
                {"id": "electron_temperature", "field_family": "temperature"},
                {"id": "plasma_potential", "field_family": "electrostatic"},
            ]
        },
        custom_groups={"density": ["ion_density"]},
        custom_groups_missing="default",
        allow_empty_custom_groups=False,
    )

    assert list(groups) == ["density", "default"]
    assert groups["density"].targets == ("ion_density",)
    assert groups["default"].targets == ("electron_temperature", "plasma_potential")
    assert groups["default"].field_family is None
