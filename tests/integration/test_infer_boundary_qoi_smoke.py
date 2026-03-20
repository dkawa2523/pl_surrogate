from __future__ import annotations

from pathlib import Path

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_infer_boundary_qoi_and_diagnostics_smoke(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={
            "poisson_residual_limit": 1e3,
            "boundary_operator": {
                "delta_edge": 1.5,
                "wafer_only": False,
                "mode": "operator_prior",
                "prior_coeffs": {"log_ne": 0.08, "Te": 0.06, "E_n": 0.04, "bias": 0.0},
                "loss_limit": 1e9,
            },
        },
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})
    assert "boundary_gamma_uniformity" in res.qoi
    assert "boundary_operator_proxy_loss" in res.diagnostics
    assert res.diagnostics["boundary_band_n"] > 0
