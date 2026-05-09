from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.cases import parse_batch_cases
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def _write_base_geometry(dataset_root: Path, *, h: int = 8, w: int = 8) -> None:
    g = dataset_root / "geometry"
    g.mkdir(parents=True, exist_ok=True)
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    np.save(g / "mask_plasma.npy", mask)
    np.save(g / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(g / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))


def _write_parametric_parts(dataset_root: Path) -> None:
    _write_base_geometry(dataset_root)
    g = dataset_root / "geometry"
    manifest = {
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.5, "max": 0.5},
        }
    }
    (g / "parts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mask_stack = np.zeros((1, 8, 8), dtype=np.float32)
    mask_stack[0, 3:5, 3:5] = 1.0
    np.savez(g / "parts_pack.npz", part_ids=np.array(["p0"], dtype=object), mask_stack=mask_stack.astype(np.float32))


def _build_engine(tmp_path: Path, provider_mode: str, *, input_mode: str) -> InferenceEngine:
    dataset_root = tmp_path / f"dataset_{provider_mode}_{input_mode}"
    if provider_mode == "parametric_parts":
        _write_parametric_parts(dataset_root)
    else:
        _write_base_geometry(dataset_root)
    provider = build_geometry_provider(dataset_root, provider_mode=provider_mode)
    return InferenceEngine(
        model=GlobalMLP(input_dim=2, grid_shape=(8, 8), seed=3),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=provider,
        output_dir=tmp_path / f"infer_{provider_mode}_{input_mode}",
        input_mode=input_mode,
    )


def test_table_only_rejects_geom_space(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "fixed", input_mode="table_only")
    with pytest.raises(ValueError, match="geom_space"):
        engine.optimize_run(
            space={"c0": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=3,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_table_plus_structure_fixed_provider_rejects_geom_space(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "fixed", input_mode="table_plus_structure")
    with pytest.raises(ValueError, match="provider_mode=parametric_parts"):
        engine.optimize_run(
            space={"c0": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=3,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_table_plus_structure_parametric_provider_joint_optimize_smoke(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "parametric_parts", input_mode="table_plus_structure")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = engine.optimize_run(
            space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=4,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
            seed=5,
            backend="random",
        )
    assert "best_geom_param" in out
    assert "part.p0.tx" in out["best_geom_param"]
    assert out["invalid_trial_count"] >= 0
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    summary = json.loads((engine.store.root / "optimize" / "summary.json").read_text(encoding="utf-8"))
    assert summary["geom_space_enabled_effective"] is True
    assert summary["geom_param_keys_effective"] == ["part.p0.tx"]
    assert "invalid_trial_count" in summary


def test_parametric_batch_cases_run_through_engine(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path, "parametric_parts", input_mode="table_plus_structure")
    cases = parse_batch_cases(
        {
            "cases": [
                {
                    "case_id": "base",
                    "cond": {"c0": 0.1, "c1": 0.2},
                    "geom": {"geom_id": "default"},
                },
                {
                    "case_id": "shifted",
                    "cond": {"c0": 0.3, "c1": 0.4},
                    "geom": {"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}},
                },
            ]
        },
        cond_schema=engine.cond_schema,
        axis_schema=engine.axis_schema,
    )

    for case in cases:
        result = engine.single_run_aggregated(cond=case.cond, geom=case.geom, axis=case.axis)
        assert "uniformity" in result.qoi
        assert np.isfinite(result.fields_model["phi"]).all()

    single_dirs = [p for p in (engine.store.root / "single").iterdir() if p.is_dir()]
    assert len(single_dirs) == 2
