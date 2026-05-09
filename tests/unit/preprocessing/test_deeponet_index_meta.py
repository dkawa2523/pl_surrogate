from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.core.synthetic_data import build_synthetic_dataset
from plasma_surrogate.preprocessing.runner import PreprocessRunner
from tests._config_presets import (
    default_target_transforms_ne_ni_te_phi,
    runtime_table_plus_structure,
)


def test_preprocess_writes_deeponet_index_meta(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset = build_synthetic_dataset(
        {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 3},
        run_dir,
    )
    pre = PreprocessRunner(
        {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_ne_ni_te_phi()},
            "sampling": {"deeponet": {"enabled": True, "n_sensors": 6, "n_queries": 10, "seed": 5}},
        },
        run_dir / "preprocessing",
        runtime_cfg=runtime_table_plus_structure(),
    )
    pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)

    droot = run_dir / "preprocessing" / "sampling" / "deeponet"
    with (droot / "index_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["flatten_order"] == "C"
    assert meta["grid_shape"] == [8, 8]
    assert meta["n_points"] == 64
    assert meta["coord_system"] == "cartesian"

    sensor_coords = np.load(droot / "sensor_coords.npy")
    query_coords = np.load(droot / "query_coords.npy")
    assert sensor_coords.shape == (6, 2)
    assert query_coords.shape == (10, 2)
