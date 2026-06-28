from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.infer.derived_fields import (
    compute_configured_derived_fields,
    compute_default_derived_fields,
)


def _linear_potential() -> np.ndarray:
    yy, xx = np.mgrid[0:4, 0:5]
    return (xx + 2.0 * yy)[None, ...].astype(np.float32)


def test_default_derived_fields_resolve_potential_from_target_roles() -> None:
    fields = {"plasma_potential": _linear_potential()}
    schema = {
        "targets": [
            {
                "id": "plasma_potential",
                "role": "potential",
                "field_family": "electrostatic",
            }
        ]
    }

    derived = compute_default_derived_fields(fields, target_role_schema=schema)

    assert set(derived) == {"E_mag"}
    assert derived["E_mag"].shape == (1, 4, 5)
    assert np.allclose(derived["E_mag"], np.sqrt(5.0), rtol=1e-6, atol=1e-6)


def test_default_derived_fields_skip_when_potential_is_unresolved() -> None:
    fields = {"unlabeled_field": _linear_potential()}

    assert compute_default_derived_fields(fields) == {}
    with pytest.raises(ValueError, match="symbols unresolved"):
        compute_default_derived_fields(fields, strict=True)


def test_configured_negative_gradient_and_vector_magnitude_are_sequential() -> None:
    fields = {"plasma_potential": _linear_potential()}
    configs = [
        {
            "id": "electric_field",
            "operator": "negative_gradient",
            "source": "plasma_potential",
        },
        {
            "id": "electric_field_magnitude",
            "operator": "vector_magnitude",
            "sources": ["electric_field_x", "electric_field_y"],
        },
    ]

    derived = compute_configured_derived_fields(fields, configs)

    assert set(derived) == {"electric_field_x", "electric_field_y", "electric_field_magnitude"}
    assert np.allclose(derived["electric_field_x"], -1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose(derived["electric_field_y"], -2.0, rtol=1e-6, atol=1e-6)
    assert np.allclose(derived["electric_field_magnitude"], np.sqrt(5.0), rtol=1e-6, atol=1e-6)


def test_configured_electric_field_magnitude_can_resolve_potential_from_symbols() -> None:
    fields = {"logical_potential": _linear_potential()}
    configs = [{"operator": "electric_field_magnitude"}]

    derived = compute_configured_derived_fields(fields, configs, symbols={"potential": "logical_potential"})

    assert set(derived) == {"E_mag"}
    assert np.allclose(derived["E_mag"], np.sqrt(5.0), rtol=1e-6, atol=1e-6)


def test_configured_missing_source_can_skip_or_fail_fast() -> None:
    configs = [{"id": "electric_field", "operator": "negative_gradient", "source": "missing"}]

    assert compute_configured_derived_fields({}, configs, strict=False) == {}
    with pytest.raises(ValueError, match="source not found"):
        compute_configured_derived_fields({}, configs, strict=True)


def test_configured_unknown_operator_rejects_config() -> None:
    with pytest.raises(ValueError, match="unsupported derived field operator"):
        compute_configured_derived_fields({}, [{"operator": "curl"}])
