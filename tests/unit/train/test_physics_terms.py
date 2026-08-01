from __future__ import annotations

import pytest

from plasma_surrogate.train.physics_terms import resolve_numpy_terms, resolve_torch_terms


def _by_name(rows):
    return {row.name: row for row in rows}


def test_resolve_physics_terms_returns_registered_specs_with_required_fields():
    terms = _by_name(
        resolve_numpy_terms(
            {
                "terms": {
                    "poisson": {"enabled": True, "weight": 0.2},
                    "boundary": {"weight": 0.0},
                    "boundary_operator": {"enabled": False, "weight": 0.7},
                }
            }
        )
    )

    assert set(terms) == {"poisson", "boundary", "boundary_operator", "rho"}
    assert terms["poisson"].enabled is True
    assert terms["poisson"].weight == pytest.approx(0.2)
    assert terms["poisson"].source_key == "terms"
    assert terms["boundary"].enabled is False
    assert terms["boundary"].weight == pytest.approx(0.0)
    assert terms["boundary_operator"].enabled is False
    assert terms["boundary_operator"].weight == pytest.approx(0.7)
    assert terms["rho"].enabled is False
    assert terms["rho"].weight == pytest.approx(0.0)


@pytest.mark.parametrize("bad_weight", [-1.0, float("inf"), float("nan")])
def test_resolve_physics_terms_rejects_invalid_weight(bad_weight):
    with pytest.raises(ValueError, match="weight"):
        resolve_numpy_terms({"terms": {"poisson": {"enabled": True, "weight": bad_weight}}})


def test_resolve_physics_terms_rejects_removed_keys_even_with_resolved_terms():
    with pytest.raises(ValueError, match="removed physics keys"):
        resolve_numpy_terms(
            {
                "lambda_poisson": 0.1,
                "resolved_terms": [
                    {"name": "poisson", "enabled": True, "weight": 0.1, "source_key": "resolved_terms"}
                ],
            }
        )


def test_resolve_physics_terms_rejects_unknown_term_name():
    with pytest.raises(ValueError, match="unsupported physics term"):
        resolve_numpy_terms({"terms": {"continuity": {"enabled": True, "weight": 0.1}}})


def test_resolve_physics_terms_rejects_term_local_symbols():
    with pytest.raises(ValueError, match="top-level physics.symbols"):
        resolve_numpy_terms(
            {
                "terms": {
                    "poisson": {
                        "enabled": True,
                        "weight": 0.1,
                        "symbols": {"potential": "plasma_potential"},
                    }
                }
            }
        )


def test_resolve_physics_terms_list_form_requires_name_enabled_weight():
    with pytest.raises(ValueError, match="missing required keys"):
        resolve_numpy_terms({"terms": [{"name": "poisson", "weight": 0.1}]})


def test_resolve_torch_terms_uses_same_validation():
    with pytest.raises(ValueError, match="unsupported physics term"):
        resolve_torch_terms({"terms": [{"name": "electric_field_consistency", "enabled": True, "weight": 0.1}]})
