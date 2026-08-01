from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.physics_contract import build_physics_cfg, normalize_physics_terms, resolve_epoch_scaled_physics
from plasma_surrogate.data.geometry_context import GeometryContext


def _geom_ctx() -> GeometryContext:
    h, w = 6, 6
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    dist = np.ones((h, w), dtype=np.float32)
    eps = np.ones((h, w), dtype=np.float32)
    coord = np.zeros((2, h, w), dtype=np.float32)
    return GeometryContext(
        mask_plasma=mask,
        distance_any=dist,
        dist0=dist.copy(),
        eps=eps,
        coord_grid=coord,
        regions={},
        bc_dir_mask=None,
        bc_dir_value=None,
    )


def _term_weight(cfg: dict, name: str) -> float:
    for row in cfg.get("resolved_terms", []):
        if row.get("name") == name:
            return float(row.get("weight", 0.0))
    raise AssertionError(f"missing term: {name}")


def test_build_physics_cfg_disabled_default():
    cfg = build_physics_cfg({}, _geom_ctx(), default_enabled=False)
    assert cfg["enabled"] is False
    assert cfg["resolved_terms"] == []


def test_build_physics_cfg_enabled_with_boundary_operator():
    raw = {
        "enabled": True,
        "terms": {
            "poisson": {"weight": 0.12},
            "boundary": {"weight": 0.34},
            "boundary_operator": {"weight": 0.5},
        },
        "boundary_operator": {
            "enabled": True,
            "mode": "operator_prior",
            "primary_qoi_key": "Gamma_i",
        },
    }
    out = build_physics_cfg(raw, _geom_ctx(), default_primary_qoi_key="Gamma_i")
    assert out["enabled"] is True
    assert np.isclose(_term_weight(out, "poisson"), 0.12)
    assert np.isclose(_term_weight(out, "boundary"), 0.34)
    assert out["boundary_operator"]["enabled"] is True
    assert out["boundary_operator"]["mode"] == "operator_prior"
    assert out["boundary_operator"]["primary_qoi_key"] == "Gamma_i"
    assert out["bc_mask"].shape == (6, 6)
    assert out["bc_value"].shape == (6, 6)


def test_normalize_physics_terms_uses_terms_weight():
    raw = {
        "enabled": True,
        "terms": {
            "poisson": {"weight": 0.9},
            "boundary": {"weight": 0.8},
            "boundary_operator": {"weight": 0.7},
        },
    }
    terms = normalize_physics_terms(raw)
    assert np.isclose(float(terms["poisson"]["weight"]), 0.9)
    assert np.isclose(float(terms["boundary"]["weight"]), 0.8)
    assert np.isclose(float(terms["boundary_operator"]["weight"]), 0.7)


def test_build_physics_cfg_uses_normalized_weights():
    raw = {
        "enabled": True,
        "terms": {
            "poisson": {"weight": 0.4},
            "boundary": {"weight": 0.3},
            "boundary_operator": {"weight": 0.5},
        },
        "boundary_operator": {"enabled": True},
    }
    out = build_physics_cfg(raw, _geom_ctx(), default_primary_qoi_key="Gamma_i")
    assert np.isclose(_term_weight(out, "poisson"), 0.4)
    assert np.isclose(_term_weight(out, "boundary"), 0.3)
    assert np.isclose(float(out["boundary_operator"]["weight"]), 0.5)
    assert "resolved_terms" in out
    assert any(row["name"] == "poisson" for row in out["resolved_terms"])


def test_normalize_physics_terms_supports_list_form_and_priority():
    raw = {
        "enabled": True,
        "terms": [
            {"name": "poisson", "weight": 0.51, "enabled": True},
            {"name": "boundary", "weight": 0.52, "enabled": True},
            {"name": "boundary_operator", "weight": 0.53, "enabled": True},
        ],
    }
    terms = normalize_physics_terms(raw)
    assert np.isclose(float(terms["poisson"]["weight"]), 0.51)
    assert np.isclose(float(terms["boundary"]["weight"]), 0.52)
    assert np.isclose(float(terms["boundary_operator"]["weight"]), 0.53)
    assert terms["poisson"]["source_key"] == "terms"


def test_resolve_epoch_scaled_physics_monotonic_ramp():
    base = {
        "enabled": True,
        "boundary_operator": {"enabled": True},
        "resolved_terms": [
            {"name": "poisson", "weight": 0.2, "enabled": True, "source_key": "x"},
            {"name": "boundary", "weight": 0.1, "enabled": True, "source_key": "x"},
            {"name": "boundary_operator", "weight": 0.3, "enabled": True, "source_key": "x"},
            {"name": "rho", "weight": 0.0, "enabled": False, "source_key": "x"},
        ],
    }
    cur = {"physics_lambda_ramp": {"start_epoch": 2, "end_epoch": 6, "scale_from": 0.0, "scale_to": 1.0}}
    s0 = resolve_epoch_scaled_physics(base, epoch=0, curriculum_cfg=cur)
    s4 = resolve_epoch_scaled_physics(base, epoch=4, curriculum_cfg=cur)
    s8 = resolve_epoch_scaled_physics(base, epoch=8, curriculum_cfg=cur)
    assert np.isclose(_term_weight(s0, "poisson"), 0.0)
    assert _term_weight(s0, "poisson") <= _term_weight(s4, "poisson") <= _term_weight(s8, "poisson")
    assert np.isclose(_term_weight(s8, "poisson"), 0.2)


def test_removed_flat_physics_keys_fail_fast():
    raw = {"enabled": True, "lambda_poisson": 0.1}
    with pytest.raises(ValueError, match="removed physics keys"):
        build_physics_cfg(raw, _geom_ctx())


@pytest.mark.parametrize("bad_weight", [-1.0, float("inf"), float("nan")])
def test_normalize_physics_terms_rejects_invalid_weight(bad_weight):
    with pytest.raises(ValueError, match="weight"):
        normalize_physics_terms({"terms": {"poisson": {"enabled": True, "weight": bad_weight}}})


def test_normalize_physics_terms_rejects_unknown_term():
    with pytest.raises(ValueError, match="unsupported physics term"):
        normalize_physics_terms({"terms": {"continuity": {"enabled": True, "weight": 0.1}}})


def test_normalize_physics_terms_rejects_list_item_missing_required_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        normalize_physics_terms({"terms": [{"name": "poisson", "weight": 0.1}]})


def test_normalize_physics_terms_rejects_term_local_symbols():
    with pytest.raises(ValueError, match="top-level physics.symbols"):
        normalize_physics_terms(
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
