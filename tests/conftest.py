from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture()
def task_spec_dict() -> dict:
    return {
        "outputs": [
            {"name": "log_ne", "units": "log10(m^-3)", "transform": "zscore"},
            {"name": "Te", "units": "eV", "transform": "zscore"},
            {"name": "phi", "units": "V", "transform": "zscore"},
        ],
        "transforms": {"log_ne": "zscore", "Te": "zscore", "phi": "zscore"},
        "units": {"log_ne": "log10(m^-3)", "Te": "eV", "phi": "V"},
        "grid_spec": {
            "axes_order": ["y", "x"],
            "coord_components": ["x", "y"],
            "shape": [8, 8],
            "coord_system": "cartesian",
        },
    }


@pytest.fixture()
def run_dir(tmp_path: Path, task_spec_dict: dict) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "preprocessing" / "scalers").mkdir(parents=True)
    (run_dir / "preprocessing" / "schema").mkdir(parents=True)
    (run_dir / "preprocessing" / "stats").mkdir(parents=True)

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"task": {"spec_path": "task_spec.yaml"}}, f)

    with (run_dir / "task_spec.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(task_spec_dict, f)

    for path, payload in [
        (run_dir / "preprocessing" / "scalers" / "cond_scaler.json", {"type": "zscore", "mean": [0.0], "std": [1.0]}),
        (run_dir / "preprocessing" / "scalers" / "y_scalers.json", {"log_ne": {"type": "zscore", "mean": [0.0], "std": [1.0]}}),
        (run_dir / "preprocessing" / "schema" / "cond_schema.json", {"order": ["c0"]}),
        (run_dir / "preprocessing" / "schema" / "axis_schema.json", {"mode": "steady", "harmonics": 1}),
        (run_dir / "preprocessing" / "schema" / "channel_map.json", {"channels": []}),
        (run_dir / "preprocessing" / "stats" / "cond_stats.json", {"c0": {"min": 0.0, "max": 1.0}}),
    ]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    return run_dir


@pytest.fixture()
def geometry_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    g = root / "geometry"
    g.mkdir(parents=True)
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, :] = 0
    eps = np.ones((8, 8), dtype=np.float32)
    wafer = np.zeros((8, 8), dtype=np.float32)
    wafer[-2:, :] = 1

    np.save(g / "mask_plasma.npy", mask)
    np.save(g / "eps.npy", eps)
    np.save(g / "wafer_mask.npy", wafer)
    return root
