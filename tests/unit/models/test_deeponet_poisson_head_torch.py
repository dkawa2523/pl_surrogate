from __future__ import annotations

import pytest
from types import SimpleNamespace

import numpy as np

from plasma_surrogate.core.torch_backend import require_torch
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.deeponet.poisson_head_torch import DeepONetPoissonHeadTorch

pytestmark = pytest.mark.torch_runtime


def test_deeponet_poisson_head_predict_phi_shape():
    require_torch_runtime()
    torch = require_torch()
    base = DeepONetPlasmaOperatorTorch(cond_dim=4, grid_shape=(8, 8), output_keys=["phi"])
    head = DeepONetPoissonHeadTorch.from_cache(
        deeponet_poisson=base,
        sensor_idx=np.arange(8, dtype=np.int64),
        query_idx=np.arange(64, dtype=np.int64),
        flatten_order="C",
        grid_shape=(8, 8),
        freeze=True,
    )
    yy = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    geom = SimpleNamespace(
        coord_grid=np.stack([xv, yv], axis=0).astype(np.float32),
        mask_plasma=np.ones((8, 8), dtype=np.float32),
        bc_dir_mask=np.zeros((8, 8), dtype=np.float32),
        bc_dir_value=np.zeros((8, 8), dtype=np.float32),
    )
    rho = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    cond = torch.zeros((2, 4), dtype=torch.float32)
    phi = head.predict_phi(rho_eff=rho, cond_vec=cond, geom_ctx=geom, refine_iters=1)
    assert tuple(phi.shape) == (2, 1, 8, 8)
