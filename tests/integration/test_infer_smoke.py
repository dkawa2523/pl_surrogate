from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_infer_single_and_batch_smoke(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})
    assert "uniformity" in res.qoi

    rows = engine.batch_run([cond, {"c0": 0.1, "c1": 0.2, "c2": 0.3}], geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})
    assert len(rows) == 2

    assert (tmp_path / "infer" / "batch" / "summary.csv").exists()
    single_dirs = [p for p in (tmp_path / "infer" / "single").iterdir() if p.is_dir()]
    assert len(single_dirs) >= 1
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()
    assert (single_dirs[0] / "ood_report.json").exists()
