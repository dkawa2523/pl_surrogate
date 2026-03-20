from __future__ import annotations

import json
from pathlib import Path

from plasma_surrogate.core.synthetic_data import build_synthetic_dataset
from plasma_surrogate.preprocessing.runner import PreprocessRunner


def test_preprocess_writes_deeponet_task_caches(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset = build_synthetic_dataset(
        {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 17},
        run_dir,
    )
    pre = PreprocessRunner(
        {
            "split": {"seed": 2, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
            "sampling": {
                "deeponet": {
                    "tasks": {
                        "poisson_head": {"n_sensors": 8, "n_queries": 64, "seed": 5},
                        "boundary_operator": {
                            "n_sensors": 12,
                            "n_queries": 20,
                            "seed": 7,
                            "primary_qoi_key": "Gamma_i",
                        },
                    }
                }
            },
        },
        run_dir / "preprocessing",
    )
    pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)

    base = run_dir / "preprocessing" / "sampling" / "deeponet"
    assert (base / "poisson_head" / "sensor_query_index.json").exists()
    assert (base / "poisson_head" / "index_meta.json").exists()
    assert (base / "boundary_operator" / "sensor_query_index.json").exists()
    assert (base / "boundary_operator" / "index_meta.json").exists()
    assert (base / "task_hashes.json").exists()

    with (base / "boundary_operator" / "sensor_query_index.json").open("r", encoding="utf-8") as f:
        bo_idx = json.load(f)
    assert bo_idx["primary_qoi_key"] == "Gamma_i"

    with (run_dir / "preprocessing" / "validation" / "repro_hashes.json").open("r", encoding="utf-8") as f:
        repro = json.load(f)
    assert "deeponet_task_hashes" in repro
    assert "poisson_head" in repro["deeponet_task_hashes"]
    assert "boundary_operator" in repro["deeponet_task_hashes"]

    with (base / "boundary_operator" / "index_meta.json").open("r", encoding="utf-8") as f:
        bo_meta = json.load(f)
    assert "boundary_sampling_spec" in bo_meta
    assert bo_meta["boundary_sampling_spec"]["primary_qoi_key"] == "Gamma_i"


def test_preprocess_boundary_task_hash_changes_with_primary_qoi_key(tmp_path: Path):
    dataset = build_synthetic_dataset(
        {"n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 17},
        tmp_path / "dataset_root",
    )
    hashes = {}
    for key in ["Gamma_i", "Vs"]:
        run_dir = tmp_path / f"run_{key}"
        pre = PreprocessRunner(
            {
                "split": {"seed": 2, "ratios": [0.6, 0.2, 0.2]},
                "scalers": {
                    "target_transforms": {
                        "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                        "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                    }
                },
                "sampling": {
                    "deeponet": {
                        "tasks": {
                            "poisson_head": {"n_sensors": 8, "n_queries": 64, "seed": 5},
                            "boundary_operator": {
                                "n_sensors": 12,
                                "n_queries": 20,
                                "seed": 7,
                                "primary_qoi_key": key,
                            },
                        }
                    }
                },
            },
            run_dir / "preprocessing",
        )
        pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)
        with (run_dir / "preprocessing" / "validation" / "repro_hashes.json").open("r", encoding="utf-8") as f:
            repro = json.load(f)
        hashes[key] = repro["deeponet_task_hashes"]["boundary_operator"]

    assert hashes["Gamma_i"] != hashes["Vs"]
