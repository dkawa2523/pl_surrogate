from __future__ import annotations

from pathlib import Path

import json

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema
from tests._runtime_requirements import require_optuna


def test_optimize_runner_optuna_smoke(tmp_path: Path, geometry_root: Path):
    require_optuna()
    model = GlobalMLP(input_dim=2, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=2)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer_optuna",
    )

    result = engine.optimize_run(
        space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        geom_space=None,
        n_trials=4,
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
        seed=11,
        backend="optuna",
        backend_cfg={"sampler": "tpe", "n_startup_trials": 2},
    )
    assert "objective_value" in result
    assert "best_objective_value" not in result
    assert "objective_key" not in result
    assert (tmp_path / "infer_optuna" / "optimize" / "best.json").exists()
    with (tmp_path / "infer_optuna" / "optimize" / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["backend"] == "optuna"
    assert summary["objective_mode"] == "weighted_sum"
    assert "objective_value" in summary
    assert "search_value" in summary
    assert "feasible" in summary
    assert "best_objective_value" not in summary
    assert "best_search_value" not in summary
    assert "objective_key" not in summary
