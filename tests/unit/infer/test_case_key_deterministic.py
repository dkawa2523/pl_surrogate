from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_case_key_is_deterministic_for_same_payload(tmp_path: Path, geometry_root: Path):
    engine = InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )

    cond_a = {"c1": 0.2, "c0": 0.1, "c2": 0.3}
    cond_b = {"c2": 0.3, "c0": 0.1, "c1": 0.2}
    axis = {"mode": "steady", "value": 0.0}
    geom = {"geom_id": "default"}

    engine.single_run(cond=cond_a, geom=geom, axis=axis)
    engine.single_run(cond=cond_b, geom=geom, axis=axis)

    single_root = tmp_path / "infer" / "single"
    keys = sorted([p.name for p in single_root.iterdir() if p.is_dir()])
    assert len(keys) == 1
    assert len(keys[0]) == 40
    assert (single_root / keys[0] / "fields_model.npz").exists()
    assert (single_root / keys[0] / "fields_phys.npz").exists()


def test_mainline_rejects_geom_param_for_fixed_provider(tmp_path: Path, geometry_root: Path):
    engine = InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )

    with pytest.raises(ValueError, match="provider_mode=parametric_parts"):
        engine.single_run(
            cond={"c0": 0.1, "c1": 0.2, "c2": 0.3},
            geom={"geom_id": "alt_geometry", "geom_param": {"radius": 0.2, "gap": 1.1}},
            axis={"mode": "steady", "value": 0.0},
        )

    with pytest.raises(ValueError, match="provider_mode=parametric_parts"):
        engine.single_run(
            cond={"c0": 0.1, "c1": 0.2, "c2": 0.3},
            geom={"geom_param": "invalid"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_inference_result_uses_diagnostics_without_warning_buckets(tmp_path: Path, geometry_root: Path):
    engine = InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
        cond_stats={
            "c0": {"min": 0.0, "max": 1.0},
            "c1": {"min": 0.0, "max": 1.0},
            "c2": {"min": 0.0, "max": 1.0},
        },
    )
    result = engine.single_run(
        cond={"c0": 1.2, "c1": 0.2, "c2": 0.3},
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
    )
    assert result.diagnostics["physics_diagnostics_available"] is False
    assert "finite_ratio_ne" in result.diagnostics
    assert not hasattr(result, "warnings")
