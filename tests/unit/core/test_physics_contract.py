from __future__ import annotations

import numpy as np

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


def test_build_physics_cfg_disabled_default():
    cfg = build_physics_cfg({}, _geom_ctx(), default_enabled=False)
    assert cfg["enabled"] is False
    assert cfg["resolved_terms"] == []


def test_build_physics_cfg_enabled_with_boundary_operator():
    raw = {
        "enabled": True,
        "lambda_poisson": 0.12,
        "lambda_bc": 0.34,
        "boundary_operator": {
            "enabled": True,
            "lambda": 0.5,
            "mode": "operator_prior",
            "primary_qoi_key": "Gamma_i",
        },
    }
    out = build_physics_cfg(raw, _geom_ctx(), default_primary_qoi_key="Gamma_i")
    assert out["enabled"] is True
    assert np.isclose(out["lambda_poisson"], 0.12)
    assert np.isclose(out["lambda_bc"], 0.34)
    assert out["boundary_operator"]["enabled"] is True
    assert out["boundary_operator"]["mode"] == "operator_prior"
    assert out["boundary_operator"]["primary_qoi_key"] == "Gamma_i"
    assert out["bc_mask"].shape == (6, 6)
    assert out["bc_value"].shape == (6, 6)


def test_normalize_physics_terms_prefers_terms_weight():
    raw = {
        "enabled": True,
        "lambda_poisson": 0.11,
        "lambda_bc": 0.22,
        "boundary_operator": {"lambda": 0.33},
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
        "lambda_poisson": 0.2,
        "lambda_bc": 0.3,
        "terms": {"poisson": {"weight": 0.4}},
        "boundary_operator": {"enabled": True, "lambda": 0.5},
    }
    out = build_physics_cfg(raw, _geom_ctx(), default_primary_qoi_key="Gamma_i")
    assert np.isclose(float(out["lambda_poisson"]), 0.4)
    assert np.isclose(float(out["lambda_bc"]), 0.3)
    assert np.isclose(float(out["boundary_operator"]["lambda"]), 0.5)
    assert "resolved_terms" in out
    assert any(row["name"] == "poisson" for row in out["resolved_terms"])


def test_normalize_physics_terms_supports_list_form_and_priority():
    raw = {
        "enabled": True,
        "lambda_poisson": 0.11,
        "lambda_bc": 0.22,
        "boundary_operator": {"lambda": 0.33},
        "terms": [
            {"name": "poisson", "weight": 0.51, "enabled": True},
            {"name": "boundary", "weight": 0.52},
            {"name": "boundary_operator", "weight": 0.53},
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
        "lambda_poisson": 0.2,
        "lambda_bc": 0.1,
        "boundary_operator": {"enabled": True, "lambda": 0.3},
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
    assert np.isclose(float(s0["lambda_poisson"]), 0.0)
    assert float(s0["lambda_poisson"]) <= float(s4["lambda_poisson"]) <= float(s8["lambda_poisson"])
    assert np.isclose(float(s8["lambda_poisson"]), 0.2)