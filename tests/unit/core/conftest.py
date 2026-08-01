from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def run_dir(tmp_path: Path, task_spec_dict: dict[str, Any]) -> Path:
    run_root = tmp_path / "run_bundle_case"
    (run_root / "preprocessing" / "scalers").mkdir(parents=True, exist_ok=True)
    (run_root / "preprocessing" / "schema").mkdir(parents=True, exist_ok=True)

    cond_scaler = {"type": "zscore", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
    y_scalers = {
        "ne": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "ni": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "Te": {"type": "zscore", "mean": [0.0], "std": [1.0]},
        "phi": {"type": "zscore", "mean": [0.0], "std": [1.0]},
    }
    cond_schema = {"order": ["c0", "c1", "c2"]}
    axis_schema = {"mode": "steady", "harmonics": 1}
    output_layout = {"order": "C", "shape": [4, 8, 8], "vars": ["ne", "ni", "Te", "phi"]}

    (run_root / "preprocessing" / "scalers" / "cond_scaler.json").write_text(
        json.dumps(cond_scaler, indent=2), encoding="utf-8"
    )
    (run_root / "preprocessing" / "scalers" / "y_scalers.json").write_text(
        json.dumps(y_scalers, indent=2), encoding="utf-8"
    )
    (run_root / "preprocessing" / "schema" / "cond_schema.json").write_text(
        json.dumps(cond_schema, indent=2), encoding="utf-8"
    )
    (run_root / "preprocessing" / "schema" / "axis_schema.json").write_text(
        json.dumps(axis_schema, indent=2), encoding="utf-8"
    )
    (run_root / "preprocessing" / "schema" / "output_layout.json").write_text(
        json.dumps(output_layout, indent=2), encoding="utf-8"
    )
    (run_root / "task_spec.yaml").write_text(json.dumps(task_spec_dict, indent=2), encoding="utf-8")
    return run_root


@pytest.fixture
def task_spec_dict() -> dict[str, Any]:
    return {
        "outputs": [
            {"name": "ne", "units": "m^-3", "transform": "zscore"},
            {"name": "ni", "units": "m^-3", "transform": "zscore"},
            {"name": "Te", "units": "eV", "transform": "zscore"},
            {"name": "phi", "units": "V", "transform": "zscore"},
        ],
        "transforms": {"ne": "zscore", "ni": "zscore", "Te": "zscore", "phi": "zscore"},
        "units": {"ne": "m^-3", "ni": "m^-3", "Te": "eV", "phi": "V"},
        "grid_spec": {
            "axes_order": ["y", "x"],
            "coord_components": ["x", "y"],
            "shape": [8, 8],
            "coord_system": "cartesian",
        },
        "metadata": {"source": "test_fixture"},
    }
