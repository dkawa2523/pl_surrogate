from __future__ import annotations

import pytest
from types import SimpleNamespace

import numpy as np

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.models.checkpoint import load_checkpoint, save_checkpoint
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.deeponet.poisson_head_torch import DeepONetPoissonHeadTorch

pytestmark = pytest.mark.torch_runtime


def _poisson_operator() -> DeepONetPlasmaOperatorTorch:
    return DeepONetPlasmaOperatorTorch(
        cond_dim=4,
        grid_shape=(8, 8),
        output_keys=["phi"],
        sensor_feature_names=["rho_eff"],
        query_feature_names=[],
    )


def test_deeponet_poisson_head_predict_phi_shape():
    require_torch_runtime()
    torch = require_torch()
    base = _poisson_operator()
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


def test_deeponet_poisson_head_rejects_generic_spatial_operator_contract():
    require_torch_runtime()
    generic = DeepONetPlasmaOperatorTorch(
        cond_dim=4,
        grid_shape=(8, 8),
        output_keys=["phi"],
    )

    with pytest.raises(ValueError, match="exactly one sensor feature named 'rho_eff'"):
        DeepONetPoissonHeadTorch.from_cache(
            deeponet_poisson=generic,
            sensor_idx=np.arange(8, dtype=np.int64),
            query_idx=np.arange(64, dtype=np.int64),
            flatten_order="C",
            grid_shape=(8, 8),
        )


def test_deeponet_poisson_head_metadata_preserves_operator_contract():
    require_torch_runtime()
    head = DeepONetPoissonHeadTorch.from_cache(
        deeponet_poisson=_poisson_operator(),
        sensor_idx=np.arange(8, dtype=np.int64),
        query_idx=np.arange(64, dtype=np.int64),
        flatten_order="C",
        grid_shape=(8, 8),
    )

    meta = head.to_meta()
    assert meta["sensor_feature_names"] == ["rho_eff"]
    assert meta["sensor_feature_dim"] == 1
    assert meta["query_feature_names"] == []
    assert meta["output_keys"] == ["phi"]


def test_deeponet_poisson_head_checkpoint_preserves_operator_contract(tmp_path):
    require_torch_runtime()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=4,
        grid_shape=(8, 8),
        output_keys=["ne", "rho_eff"],
    )
    model.attach_poisson_head(
        DeepONetPoissonHeadTorch.from_cache(
            deeponet_poisson=_poisson_operator(),
            sensor_idx=np.arange(8, dtype=np.int64),
            query_idx=np.arange(64, dtype=np.int64),
            flatten_order="C",
            grid_shape=(8, 8),
        )
    )

    checkpoint_dir = tmp_path / "deeponet_poisson_contract"
    save_checkpoint(model, checkpoint_dir)
    loaded = load_checkpoint(checkpoint_dir)

    assert loaded.poisson_head.net.sensor_feature_names == ["rho_eff"]
    assert loaded.poisson_head.net.sensor_feature_dim == 1
    assert loaded.poisson_head.net.query_feature_names == []
