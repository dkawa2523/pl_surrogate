from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


class _DummyDeepONetHead:
    def __init__(self):
        self.used_predict_phi = False

    def predict_phi(self, rho_eff, cond_vec, geom_ctx, refine_iters=0):
        del rho_eff, cond_vec, refine_iters
        self.used_predict_phi = True
        h, w = geom_ctx.mask_plasma.shape
        return np.zeros((1, 1, h, w), dtype=np.float32)

    def predict_fields(self, cond_vec):
        raise AssertionError("predict_fields should not be used when predict_phi is available")


def test_plasma_head_prefers_predict_phi_for_deeponet_mode(tmp_path: Path, geometry_root: Path):
    head = _DummyDeepONetHead()
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        phi_mode="deeponet_poisson",
        deeponet_head=head,
    )
    res = engine.single_run(
        cond={"c0": 0.1, "c1": 0.2, "c2": 0.3},
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
    )
    assert head.used_predict_phi is True
    assert res.fields_phys["phi"].shape == (1, 8, 8)

