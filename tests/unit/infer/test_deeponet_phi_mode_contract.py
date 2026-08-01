from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_inference_engine_deeponet_poisson_not_implemented(tmp_path: Path, geometry_root: Path):
    target_role_schema = {
        "targets": [
            {"id": "electron_density", "role": "density_electron", "field_family": "density"},
            {"id": "electron_temperature", "role": "temperature_electron", "field_family": "temperature"},
            {"id": "plasma_potential", "role": "potential", "field_family": "electrostatic"},
        ],
    }
    engine = InferenceEngine(
        model=GlobalMLP(
            input_dim=3,
            grid_shape=(8, 8),
            output_keys=["electron_density", "electron_temperature", "plasma_potential"],
            seed=0,
        ),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        phi_mode="deeponet_poisson",
        target_role_schema=target_role_schema,
    )
    with pytest.raises(NotImplementedError, match="deeponet_poisson head requires deeponet_head"):
        engine.single_run(
            cond={"c0": 0.1, "c1": 0.2, "c2": 0.3},
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )
