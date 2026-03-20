from __future__ import annotations

import pytest

from plasma_surrogate.core.torch_backend import require_torch, torch_runtime_available
from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch
from plasma_surrogate.train.loss_composer import compose_torch
from plasma_surrogate.train.torch_losses import (
    masked_huber_loss_torch,
    masked_region_huber_loss_torch,
    physics_terms_torch,
    poisson_residual_fd_torch,
    sdf_continuous_weight_map_torch,
)


def test_poisson_residual_fd_torch_shape():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    phi = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    res = poisson_residual_fd_torch(phi)
    assert tuple(res.shape) == (2, 1, 8, 8)


def test_poisson_residual_fd_torch_accepts_2d_eps_mask():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    phi = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    eps = torch.ones((8, 8), dtype=torch.float32)
    mask = torch.ones((8, 8), dtype=torch.float32)
    res = poisson_residual_fd_torch(phi, eps=eps, mask=mask)
    assert tuple(res.shape) == (2, 1, 8, 8)


def test_physics_terms_torch_with_boundary_operator():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    pred = {
        "log_ne": torch.ones((2, 1, 8, 8), dtype=torch.float32),
        "Te": torch.ones((2, 1, 8, 8), dtype=torch.float32),
        "phi": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
    }
    bo = BoundaryOperatorTorch(primary_qoi_key="Gamma_i", freeze=False)
    total, terms = physics_terms_torch(
        pred_fields=pred,
        cond_vec=torch.zeros((2, 4), dtype=torch.float32),
        geom_ctx=None,
        cfg={
            "enabled": True,
            "lambda_poisson": 0.1,
            "boundary_operator": {
                "enabled": True,
                "lambda": 0.2,
                "mode": "operator_prior",
                "primary_qoi_key": "Gamma_i",
            },
        },
        boundary_operator_model=bo,
    )
    assert float(total.detach().cpu().item()) >= 0.0
    assert "poisson" in terms
    assert "boundary_operator" in terms


def test_physics_terms_torch_uses_pred_rho_eff_when_present():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    phi = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    pred_zero = {
        "log_ne": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "Te": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "phi": phi,
        "rho_eff": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    }
    pred_nonzero = {
        "log_ne": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "Te": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "phi": phi,
        "rho_eff": torch.ones((1, 1, 8, 8), dtype=torch.float32),
    }
    cfg = {"enabled": True, "lambda_poisson": 1.0, "scale_rho": 1.0}
    total0, terms0 = physics_terms_torch(
        pred_fields=pred_zero,
        cond_vec=torch.zeros((1, 2), dtype=torch.float32),
        geom_ctx=None,
        cfg=cfg,
        boundary_operator_model=None,
    )
    total1, terms1 = physics_terms_torch(
        pred_fields=pred_nonzero,
        cond_vec=torch.zeros((1, 2), dtype=torch.float32),
        geom_ctx=None,
        cfg=cfg,
        boundary_operator_model=None,
    )
    assert float(total1.detach().cpu().item()) > float(total0.detach().cpu().item())
    assert float(terms1["poisson"]) > float(terms0["poisson"])


def test_compose_torch_returns_common_component_keys():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    pred = {
        "log_ne": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "Te": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "phi": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    }
    total, components = compose_torch(
        pred_fields=pred,
        cond_vec=torch.zeros((1, 2), dtype=torch.float32),
        geom_ctx=None,
        physics_cfg={"enabled": True, "lambda_poisson": 0.1},
    )
    assert float(total.detach().cpu().item()) >= 0.0
    assert set(components.keys()) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}


def test_compose_torch_disabled_accepts_noncanonical_field_set():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    pred = {
        "custom_target": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    }
    total, components = compose_torch(
        pred_fields=pred,
        cond_vec=torch.zeros((1, 2), dtype=torch.float32),
        geom_ctx=None,
        physics_cfg={"enabled": False},
    )
    assert float(total.detach().cpu().item()) == 0.0
    assert set(components.keys()) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}


def test_masked_huber_loss_torch_respects_mask():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    tgt = torch.zeros_like(pred)
    full = masked_huber_loss_torch(pred, tgt, mask=None, delta=1.0)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    part = masked_huber_loss_torch(pred, tgt, mask=mask, delta=1.0)
    assert float(full.detach().cpu().item()) > 0.0
    assert float(part.detach().cpu().item()) > 0.0
    assert float(part.detach().cpu().item()) < float(full.detach().cpu().item())


def test_masked_region_huber_loss_torch_boundary_weighting():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    pred = torch.tensor([[[[4.0, 4.0], [1.0, 1.0]]]], dtype=torch.float32)
    tgt = torch.zeros_like(pred)
    mask = torch.ones((2, 2), dtype=torch.float32)
    dist = torch.tensor([[0.5, 0.5], [5.0, 5.0]], dtype=torch.float32)
    l_bulk = masked_region_huber_loss_torch(
        pred, tgt, mask, dist, delta=1.0, boundary_delta=2.0, w_bulk=1.0, w_boundary=1.0
    )
    l_edge = masked_region_huber_loss_torch(
        pred, tgt, mask, dist, delta=1.0, boundary_delta=2.0, w_bulk=1.0, w_boundary=4.0
    )
    assert float(l_edge.detach().cpu().item()) > float(l_bulk.detach().cpu().item())


def test_sdf_continuous_weight_map_torch_orders_weights():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    torch = require_torch()
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    signed = torch.tensor([[0.0, 4.0], [-0.2, -5.0]], dtype=torch.float32)
    w = sdf_continuous_weight_map_torch(
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
    assert float(w[0, 0, 0, 0].item()) > float(w[0, 0, 0, 1].item())
    assert float(w[0, 0, 0, 0].item()) > float(w[0, 0, 1, 0].item())
