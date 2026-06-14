from __future__ import annotations

import numpy as np

from plasma_surrogate.models.deeponet.boundary_operator_stub import BoundaryOperatorStub
from plasma_surrogate.train.losses import (
    build_signed_distance,
    boundary_operator_grad,
    boundary_operator_loss,
    boundary_grad,
    boundary_loss,
    laplacian2d,
    physics_loss_and_grad,
    poisson_residual_loss,
    masked_huber_loss,
    masked_region_huber_loss,
    sdf_continuous_weight_map,
)


def test_laplacian_and_poisson_constant_is_zero():
    phi = np.ones((2, 8, 8), dtype=np.float32) * 3.0
    lap = laplacian2d(phi)
    assert np.allclose(lap, 0.0)
    assert np.isclose(poisson_residual_loss(phi), 0.0)


def test_boundary_loss_and_grad_positive_when_mismatch():
    phi = np.zeros((1, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, :] = 1.0
    value = np.ones((4, 4), dtype=np.float32)
    loss = boundary_loss(phi, bc_mask=mask, bc_value=value)
    grad = boundary_grad(phi, bc_mask=mask, bc_value=value)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert float(np.max(np.abs(grad))) > 0.0


def test_physics_loss_and_grad_combines_terms():
    phi = np.random.default_rng(0).normal(size=(2, 6, 6)).astype(np.float32)
    cfg = {
        "enabled": True,
        "poisson_weight": 0.1,
        "boundary_weight": 0.2,
        "bc_mask": np.ones((6, 6), dtype=np.float32),
        "bc_value": np.zeros((6, 6), dtype=np.float32),
    }
    loss, grad, terms = physics_loss_and_grad(phi, cfg=cfg)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert terms["poisson"] > 0.0
    assert terms["boundary"] > 0.0


def test_boundary_operator_loss_and_grad_positive():
    b, h, w = 2, 6, 6
    phi = np.zeros((b, h, w), dtype=np.float32)
    log_ne = np.ones((b, h, w), dtype=np.float32) * 0.5
    te = np.ones((b, h, w), dtype=np.float32) * 2.0
    mask = np.zeros((h, w), dtype=np.float32)
    mask[0:2, :] = 1.0
    loss = boundary_operator_loss(phi=phi, log_ne=log_ne, te=te, mask_band=mask)
    grad = boundary_operator_grad(phi=phi, log_ne=log_ne, te=te, mask_band=mask)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert float(np.max(np.abs(grad))) > 0.0


def test_physics_loss_includes_boundary_operator_term():
    rng = np.random.default_rng(0)
    b, h, w = 2, 6, 6
    phi = rng.normal(size=(b, h, w)).astype(np.float32)
    log_ne = rng.normal(size=(b, h, w)).astype(np.float32)
    te = np.abs(rng.normal(size=(b, h, w)).astype(np.float32))
    mask_band = np.zeros((h, w), dtype=np.float32)
    mask_band[0:2, :] = 1.0
    cfg = {
        "enabled": True,
        "poisson_weight": 0.0,
        "boundary_weight": 0.0,
        "boundary_operator": {
            "enabled": True,
            "weight": 0.5,
            "mask_band": mask_band,
            "target_coeffs": {"log_ne": 0.1, "Te": 0.1, "bias": 0.0},
        },
    }
    loss, grad, terms = physics_loss_and_grad(phi, cfg=cfg, log_ne=log_ne, te=te)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert terms["boundary_operator"] > 0.0


def test_boundary_operator_mode_operator_prior_runs():
    rng = np.random.default_rng(1)
    b, h, w = 2, 6, 6
    phi = rng.normal(size=(b, h, w)).astype(np.float32)
    log_ne = rng.normal(size=(b, h, w)).astype(np.float32)
    te = np.abs(rng.normal(size=(b, h, w)).astype(np.float32))
    mask_band = np.zeros((h, w), dtype=np.float32)
    mask_band[0:2, :] = 1.0
    cfg = {
        "enabled": True,
        "poisson_weight": 0.0,
        "boundary_weight": 0.0,
        "boundary_operator": {
            "enabled": True,
            "weight": 0.5,
            "mode": "operator_prior",
            "mask_band": mask_band,
            "prior_coeffs": {"log_ne": 0.08, "Te": 0.06, "E_n": 0.04, "bias": 0.0},
        },
    }
    loss, grad, terms = physics_loss_and_grad(phi, cfg=cfg, log_ne=log_ne, te=te)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert terms["boundary_operator"] > 0.0


def test_boundary_operator_mode_operator_prior_with_handle_runs():
    rng = np.random.default_rng(11)
    b, h, w = 2, 6, 6
    phi = rng.normal(size=(b, h, w)).astype(np.float32)
    log_ne = rng.normal(size=(b, h, w)).astype(np.float32)
    te = np.abs(rng.normal(size=(b, h, w)).astype(np.float32))
    mask_band = np.zeros((h, w), dtype=np.float32)
    mask_band[0:2, :] = 1.0
    op = BoundaryOperatorStub(w_log_ne=0.09, w_te=0.05, w_en=0.02, bias=0.0)
    cfg = {
        "enabled": True,
        "poisson_weight": 0.0,
        "boundary_weight": 0.0,
        "boundary_operator": {
            "enabled": True,
            "weight": 0.5,
            "mode": "operator_prior",
            "mask_band": mask_band,
            "operator_handle": op,
        },
    }
    loss, grad, terms = physics_loss_and_grad(phi, cfg=cfg, log_ne=log_ne, te=te)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert terms["boundary_operator"] > 0.0


def test_boundary_operator_mode_external_operator_runs():
    rng = np.random.default_rng(2)
    b, h, w = 2, 6, 6
    phi = rng.normal(size=(b, h, w)).astype(np.float32)
    log_ne = rng.normal(size=(b, h, w)).astype(np.float32)
    te = np.abs(rng.normal(size=(b, h, w)).astype(np.float32))
    mask_band = np.zeros((h, w), dtype=np.float32)
    mask_band[0:2, :] = 1.0
    op = BoundaryOperatorStub(w_log_ne=0.07, w_te=0.05, w_en=0.03, bias=0.0)
    cfg = {
        "enabled": True,
        "poisson_weight": 0.0,
        "boundary_weight": 0.0,
        "boundary_operator": {
            "enabled": True,
            "weight": 0.5,
            "mode": "external_operator",
            "mask_band": mask_band,
            "external_operator_handle": op,
        },
    }
    loss, grad, terms = physics_loss_and_grad(phi, cfg=cfg, log_ne=log_ne, te=te)
    assert loss > 0.0
    assert grad.shape == phi.shape
    assert terms["boundary_operator"] > 0.0


def test_masked_huber_loss_respects_mask():
    pred = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    tgt = np.zeros_like(pred)
    full = masked_huber_loss(pred, tgt, mask=None, delta=1.0)
    mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    masked = masked_huber_loss(pred, tgt, mask=mask, delta=1.0)
    assert full > 0.0
    assert masked > 0.0
    assert masked < full


def test_masked_region_huber_loss_boundary_weighting():
    pred = np.array([[[4.0, 4.0], [1.0, 1.0]]], dtype=np.float32)
    tgt = np.zeros_like(pred)
    mask = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    dist = np.array([[0.5, 0.5], [5.0, 5.0]], dtype=np.float32)
    l_bulk = masked_region_huber_loss(
        pred,
        tgt,
        mask,
        dist,
        delta=1.0,
        boundary_delta=2.0,
        w_bulk=1.0,
        w_boundary=1.0,
    )
    l_edge = masked_region_huber_loss(
        pred,
        tgt,
        mask,
        dist,
        delta=1.0,
        boundary_delta=2.0,
        w_bulk=1.0,
        w_boundary=4.0,
    )
    assert l_edge > l_bulk


def test_build_signed_distance_sign_convention():
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    dist = np.array([[2.0, 0.3], [0.7, 3.0]], dtype=np.float32)
    signed = build_signed_distance(mask, dist)
    assert np.all(signed[mask > 0.5] >= 0.0)
    assert np.all(signed[mask <= 0.5] <= 0.0)


def test_sdf_continuous_weight_map_orders_boundary_inside_chamber():
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    signed = np.array([[0.0, 4.0], [-0.2, -5.0]], dtype=np.float32)
    w = sdf_continuous_weight_map(
        mask,
        signed,
        {
            "tau_in": 2.0,
            "tau_out": 1.5,
            "w_boundary": 3.0,
            "w_inner": 1.0,
            "w_chamber_floor": 0.08,
        },
    )
    boundary_w = float(w[0, 0])
    inner_w = float(w[0, 1])
    chamber_near_w = float(w[1, 0])
    chamber_far_w = float(w[1, 1])
    assert boundary_w > inner_w
    assert boundary_w > chamber_near_w
    assert chamber_near_w >= chamber_far_w
