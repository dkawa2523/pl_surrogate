from __future__ import annotations

from pathlib import Path

from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_infer_boundary_qoi_and_diagnostics_smoke(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        ood_cfg={
            "poisson_residual_limit": 1e3,
            "physics": {"symbols": {"density": "ne", "temperature": "Te", "potential": "phi"}},
            "qoi": {
                "uniformity": {
                    "target": "Te",
                    "region": "plasma_mid_height",
                    "mid_height_band_px": 1,
                }
            },
            "postprocess": {"positive_vars": ["Te"], "positive_floor": 0.0},
            "boundary_operator": {
                "enabled": True,
                "delta_edge": 1.5,
                "wafer_only": False,
                "mode": "operator_prior",
                "prior_coeffs": {"density": 0.08, "temperature": 0.06, "E_n": 0.04, "bias": 0.0},
                "loss_limit": 1e9,
            },
        },
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})
    assert "boundary_gamma_uniformity" in res.qoi
    assert res.qoi["uniformity_target"] == "Te"
    assert res.qoi["uniformity_region"] == "plasma_mid_height"
    assert "uniformity_max_density" in res.qoi
    assert res.qoi["uniformity_max_density"] >= res.qoi["uniformity_mean_density"]
    assert float(res.fields_phys["Te"].min()) >= 0.0
    assert "boundary_operator_proxy_loss" in res.diagnostics
    assert res.diagnostics["boundary_band_n"] > 0


def test_infer_uniformity_preferred_targets_and_positive_string_smoke(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer_preferred",
        ood_cfg={
            "uniformity_region": "plasma_mid_height",
            "mid_height_band_px": 1,
            "qoi": {
                "uniformity": {
                    "target": None,
                    "preferred_targets": ["missing", "Te"],
                    "region": None,
                    "mid_height_band_px": None,
                }
            },
            "postprocess": {"positive_vars": "Te", "positive_floor": None, "floor": 0.0},
        },
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})

    assert res.qoi["uniformity_target"] == "Te"
    assert res.qoi["uniformity_region"] == "plasma_mid_height"
    assert float(res.fields_phys["Te"].min()) >= 0.0


def test_infer_null_qoi_and_postprocess_configs_use_defaults(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer_null_cfg",
        ood_cfg={"qoi": None, "postprocess": None},
        target_role_schema={"positive_targets": ["ne"]},
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})

    assert "uniformity" in res.qoi
    assert "uniformity_target" in res.qoi


def test_infer_null_nested_ood_configs_use_defaults(tmp_path: Path, geometry_root: Path):
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), output_keys=["ne", "Te", "phi"], seed=0)
    engine = InferenceEngine(
        model=model,
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer_nested_null_cfg",
        ood_cfg={
            "qoi": {"uniformity": {"score_mode": None}},
            "uniformity_score_mode": None,
            "postprocess": None,
            "physics": None,
            "boundary_operator": None,
        },
        target_role_schema={"positive_targets": ["ne"]},
    )

    cond = {"c0": 0.2, "c1": 0.4, "c2": 0.6}
    res = engine.single_run(cond=cond, geom={"geom_id": "default"}, axis={"mode": "steady", "value": 0.0})

    assert res.qoi["uniformity_score_mode"] == "relative"
    assert "boundary_gamma_uniformity" not in res.qoi
