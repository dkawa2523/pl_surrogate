from __future__ import annotations

import pytest
from plasma_surrogate.core.torch_backend import require_torch
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch

pytestmark = pytest.mark.torch_runtime


def test_boundary_operator_torch_predict_target_shape():
    require_torch_runtime()
    torch = require_torch()
    op = BoundaryOperatorTorch(primary_qoi_key="Gamma_i")
    log_ne = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    te = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    phi = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    out = op.predict_target(log_ne=log_ne, te=te, phi=phi, cond=None, geom_ctx=None)
    assert "Gamma_i" in out
    assert tuple(out["Gamma_i"].shape) == (2, 1, 8, 8)


def test_boundary_operator_torch_predict_target_with_sample_idx():
    require_torch_runtime()
    torch = require_torch()
    op = BoundaryOperatorTorch(primary_qoi_key="Gamma_i")
    log_ne = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    te = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    phi = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    sample_idx = torch.tensor([0, 3, 10, 20, 30], dtype=torch.int64)
    out = op.predict_target(
        log_ne=log_ne,
        te=te,
        phi=phi,
        cond=None,
        geom_ctx=None,
        sample_idx=sample_idx,
    )
    assert "Gamma_i" in out
    assert tuple(out["Gamma_i"].shape) == (2, 5, 1)
