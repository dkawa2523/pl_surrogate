from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_resolve_axis_samples_time_window():
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="time"),
        geometry_provider=FixedGeometryProvider("tests/fixtures/does_not_matter"),
        output_dir=Path("tests/fixtures/does_not_matter"),
    )
    samples = engine._resolve_axis_samples(
        {"mode": "time", "aggregation": "window_mean", "window": [0.2, 0.6], "n_points": 3}
    )
    assert len(samples) == 3
    assert samples[0]["value"] == pytest.approx(0.2)
    assert samples[-1]["value"] == pytest.approx(0.6)


def test_resolve_axis_samples_mode_mismatch_raises(tmp_path: Path, geometry_root: Path):
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
    )
    with pytest.raises(ValueError, match="axis mode mismatch"):
        engine._resolve_axis_samples({"mode": "time", "aggregation": "single", "value": 0.0})


def test_single_run_aggregated_window_mean(tmp_path: Path, geometry_root: Path):
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=1),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="time"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )
    out = engine.single_run_aggregated(
        cond={"c0": 0.2, "c1": 0.4},
        geom={"geom_id": "default"},
        axis={"mode": "time", "aggregation": "window_mean", "window": [0.0, 0.4], "n_points": 3},
    )
    assert "uniformity" in out.qoi
    single_dirs = [p for p in (tmp_path / "infer" / "single").iterdir() if p.is_dir()]
    assert len(single_dirs) == 3
    assert all((d / "diagnostics.json").exists() for d in single_dirs)
