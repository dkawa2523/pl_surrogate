from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.core.synthetic_data import build_synthetic_dataset
from plasma_surrogate.preprocessing.runner import PreprocessRunner


def test_preprocess_saves_output_layout_and_train_only_cond_stats(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset = build_synthetic_dataset(
        {"n_cases": 9, "height": 8, "width": 8, "cond_dim": 3, "seed": 11},
        run_dir,
    )
    pre = PreprocessRunner(
        {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        run_dir / "preprocessing",
    )
    output = pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)

    with (run_dir / "preprocessing" / "schema" / "output_layout.json").open("r", encoding="utf-8") as f:
        layout = json.load(f)
    assert layout == {"order": "C", "shape": [4, 8, 8], "vars": ["ne", "ni", "Te", "phi"]}
    assert (run_dir / "preprocessing" / "split" / "split_pressure_extrap_v1.json").exists()
    assert (run_dir / "preprocessing" / "scalers" / "xgrid_channel_scalers.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "pairs" / "phase_wrap_pairs.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "pairs" / "time_adj_pairs.json").exists()
    assert (run_dir / "preprocessing" / "stats" / "y_stats.json").exists()
    assert (run_dir / "preprocessing" / "validation" / "repro_hashes.json").exists()

    with (run_dir / "preprocessing" / "stats" / "cond_stats.json").open("r", encoding="utf-8") as f:
        cond_stats = json.load(f)

    train_ids = set(output.split["train"])
    train_rows = np.array(
        [[case["cond"]["c0"], case["cond"]["c1"], case["cond"]["c2"]] for case in dataset.cases if case["case_id"] in train_ids],
        dtype=np.float32,
    )
    assert np.isclose(cond_stats["c0"]["mean"], float(np.mean(train_rows[:, 0])))
    assert np.isclose(cond_stats["c1"]["mean"], float(np.mean(train_rows[:, 1])))
    assert np.isclose(cond_stats["c2"]["mean"], float(np.mean(train_rows[:, 2])))
