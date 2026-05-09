from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import json

from plasma_surrogate.cli.main import main
from tests._config_presets import csv_npz_targets_with_ne_te_phi, runtime_table_only


def test_cli_pipeline_smoke(tmp_path: Path):
    run_dir = tmp_path / "cycle1_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {"type": "synthetic", "n_cases": 12, "height": 8, "width": 8, "cond_dim": 3, "seed": 7},
        "preprocessing": {
            "split": {"seed": 0, "ratios": [0.7, 0.15, 0.15]},
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
            "sampling": {
                "deeponet": {"enabled": True, "n_sensors": 8, "n_queries": 12, "seed": 11},
            },
        },
        "model": {"name": "global_mlp"},
        "train": {"epochs": 5, "lr": 0.01},
        "inference": {
            "single": {"enabled": True, "cond": {"c0": 0.2, "c1": 0.5, "c2": 0.8}},
            "batch": {
                "enabled": True,
                "conds": [
                    {"c0": 0.1, "c1": 0.2, "c2": 0.3},
                    {"c0": 0.7, "c1": 0.6, "c2": 0.5},
                ],
            },
            "optimize": {
                "enabled": True,
                "n_trials": 4,
                "sampler": "random",
                "space": {"c0": [0.0, 1.0], "c1": [0.0, 1.0], "c2": [0.0, 1.0]},
            },
        },
    }

    cfg_path = tmp_path / "pipeline.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0
    assert main(["viz", "--config", str(cfg_path)]) == 0

    assert (run_dir / "preprocessing" / "split" / "split_random_v1.json").exists()
    assert (run_dir / "preprocessing" / "split" / "split_interp_v1.json").exists()
    assert (run_dir / "preprocessing" / "split" / "split_interp_status_v1.json").exists()
    assert (run_dir / "preprocessing" / "split" / "split_extrap_v1.json").exists()
    assert (run_dir / "data_cleaning" / "report.json").exists()
    assert (run_dir / "preprocessing" / "split" / "split_pressure_extrap_v1.json").exists()
    assert (run_dir / "preprocessing" / "scalers" / "xgrid_channel_scalers.json").exists()
    assert (run_dir / "preprocessing" / "scalers" / "fit_policy.json").exists()
    assert (run_dir / "preprocessing" / "schema" / "output_layout.json").exists()
    assert (run_dir / "preprocessing" / "scalers" / "coord_scaler.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "deeponet" / "sensor_query_index.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "deeponet" / "index_meta.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "deeponet" / "sensor_coords.npy").exists()
    assert (run_dir / "preprocessing" / "sampling" / "deeponet" / "query_coords.npy").exists()
    assert (run_dir / "preprocessing" / "sampling" / "geometry" / "distance_signed.npy").exists()
    assert (run_dir / "preprocessing" / "scalers" / "distance_transform_stats.json").exists()
    assert (run_dir / "checkpoints" / "meta.json").exists()
    assert (run_dir / "inference" / "single").exists()
    assert (run_dir / "viz" / "plots" / "loss_curve.png").exists()
    assert (run_dir / "viz" / "tables" / "physics_diagnostics.csv").exists()
    assert (run_dir / "viz" / "tables" / "region_metrics.csv").exists()

    single_dirs = [p for p in (run_dir / "inference" / "single").iterdir() if p.is_dir()]
    assert single_dirs
    fields_model = np.load(single_dirs[0] / "fields_model.npz")["phi"]
    fields_phys = np.load(single_dirs[0] / "fields_phys.npz")["phi"]
    assert fields_model.shape == fields_phys.shape
    assert not np.allclose(fields_model, fields_phys)

    with (run_dir / "data_cleaning" / "report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    assert "duplicate_case_keys" in report
    assert "cond_range_summary" in report

    cases_summary = json.loads((run_dir / "inference" / "cases_summary.json").read_text(encoding="utf-8"))
    assert len(cases_summary["cases"]) == 3
    assert (run_dir / "inference" / "cases_summary.csv").exists()
    assert (run_dir / "inference" / "batch" / "summary.csv").exists()

    cfg_cases = json.loads(json.dumps(cfg))
    cfg_cases["inference"] = {
        "single": {"enabled": False},
        "batch": {
            "enabled": True,
            "cases": [
                {
                    "case_id": "case_a",
                    "cond": {"c0": 0.15, "c1": 0.25, "c2": 0.35},
                    "geom": {"geom_id": "default"},
                    "axis": {"mode": "steady", "value": 0.0},
                },
                {
                    "case_id": "case_b",
                    "cond": {"c0": 0.45, "c1": 0.55, "c2": 0.65},
                    "geom": {"geom_id": "default"},
                },
            ],
        },
        "optimize": {"enabled": False},
    }
    cfg_cases_path = tmp_path / "pipeline_cases.yaml"
    with cfg_cases_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_cases, f)

    assert main(["infer", "--config", str(cfg_cases_path)]) == 0
    cases_summary = json.loads((run_dir / "inference" / "cases_summary.json").read_text(encoding="utf-8"))
    assert [row["case_id"] for row in cases_summary["cases"]] == ["case_a", "case_b"]


def test_cli_pipeline_csv_npz_smoke(tmp_path: Path):
    dataset_root = tmp_path / "csv_dataset"
    (dataset_root / "geometry").mkdir(parents=True)
    np.save(dataset_root / "geometry" / "mask_plasma.npy", np.ones((8, 8), dtype=np.float32))
    np.save(dataset_root / "geometry" / "eps.npy", np.ones((8, 8), dtype=np.float32))
    np.save(dataset_root / "geometry" / "wafer_mask.npy", np.ones((8, 8), dtype=np.float32))

    for i in range(6):
        np.savez_compressed(
            dataset_root / f"case_{i:03d}.npz",
            log_ne=np.full((8, 8), 0.1 + i * 0.01, dtype=np.float32),
            Te=np.full((8, 8), 0.2 + i * 0.01, dtype=np.float32),
            phi=np.full((8, 8), 0.3 + i * 0.01, dtype=np.float32),
        )
    rows = ["case_id,axis,c0,c1,c2,fields_npz"]
    for i in range(6):
        rows.append(f"case_{i:03d},{i/10.0:.1f},0.{i},0.{i+1},0.{i+2},case_{i:03d}.npz")
    (dataset_root / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    run_dir = tmp_path / "csv_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {
            "type": "csv_npz",
            "root": str(dataset_root),
            "index_csv": "index.csv",
            "cond_columns": ["c0", "c1", "c2"],
            "targets": csv_npz_targets_with_ne_te_phi(),
            "axis_column": "axis",
            "fields_npz_column": "fields_npz",
            "case_id_column": "case_id",
            "geometry_root": "geometry",
        },
        "preprocessing": {
            "split": {"seed": 0, "ratios": [0.7, 0.15, 0.15]},
            "axis_schema": {"mode": "time", "harmonics": 1},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "global_mlp", "phi_mode": "direct"},
        "train": {"epochs": 2, "lr": 0.01},
        "inference": {
            "single": {
                "enabled": True,
                "cond": {"c0": 0.2, "c1": 0.3, "c2": 0.4},
                "axis": {"mode": "time", "value": 0.2},
            },
            "batch": {"enabled": False},
            "optimize": {"enabled": False},
        },
    }

    cfg_path = tmp_path / "pipeline_csv_npz.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0
    assert (run_dir / "data_cleaning" / "report.json").exists()
    with (run_dir / "preprocessing" / "validation" / "report.json").open("r", encoding="utf-8") as f:
        pre_report = json.load(f)
    assert "coord_grid_source" in pre_report
    assert "coord_grid_source_requested" in pre_report
    assert "coord_grid_source_applied" in pre_report
    assert "distance_signed_negative_ratio" in pre_report
    assert "distance_contract_status" in pre_report
    assert "coord_value_range_raw" in pre_report
    assert "coord_value_range_scaled" in pre_report
    assert "coord_scaler_status" in pre_report
    assert "distance_transform_stats_path" in pre_report
    assert "distance_transform_tau_auto" in pre_report
    assert (run_dir / "preprocessing" / "scalers" / "coord_scaler.json").exists()
    assert (run_dir / "inference" / "single").exists()


def test_cli_preprocess_csv_npz_group_split_no_leak(tmp_path: Path):
    dataset_root = tmp_path / "csv_group_dataset"
    (dataset_root / "geometry").mkdir(parents=True)
    np.save(dataset_root / "geometry" / "mask_plasma.npy", np.ones((8, 8), dtype=np.float32))
    np.save(dataset_root / "geometry" / "eps.npy", np.ones((8, 8), dtype=np.float32))
    np.save(dataset_root / "geometry" / "wafer_mask.npy", np.ones((8, 8), dtype=np.float32))

    rows = ["case_id,base_case_id,split_group,axis,c0,c1,c2,fields_npz"]
    groups = ["a", "b", "c", "d", "e", "f"]
    for gi, g in enumerate(groups):
        for ti in [0, 1]:
            cid = f"{g}_t{ti}"
            np.savez_compressed(
                dataset_root / f"{cid}.npz",
                log_ne=np.full((8, 8), 0.1 + gi * 0.01 + ti * 0.001, dtype=np.float32),
                Te=np.full((8, 8), 0.2 + gi * 0.01 + ti * 0.001, dtype=np.float32),
                phi=np.full((8, 8), 0.3 + gi * 0.01 + ti * 0.001, dtype=np.float32),
            )
            rows.append(
                f"{cid},{g},{g},{ti/10.0:.1f},0.{gi},0.{gi+1},0.{gi+2},{cid}.npz"
            )
    (dataset_root / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    run_dir = tmp_path / "csv_group_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {
            "type": "csv_npz",
            "root": str(dataset_root),
            "index_csv": "index.csv",
            "cond_columns": ["c0", "c1", "c2"],
            "targets": csv_npz_targets_with_ne_te_phi(),
            "axis_column": "axis",
            "fields_npz_column": "fields_npz",
            "case_id_column": "case_id",
            "base_case_id_column": "base_case_id",
            "split_group_column": "split_group",
            "geometry_root": "geometry",
        },
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "axis_schema": {"mode": "time", "harmonics": 1},
            "scalers": {
                "target_transforms": {
                    "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
                    "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "all", "clip": {"mode": "none"}},
                }
            },
        },
        "model": {"name": "global_mlp", "phi_mode": "direct"},
    }
    cfg_path = tmp_path / "pipeline_csv_group.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    with (run_dir / "preprocessing" / "split" / "split_random_v1.json").open("r", encoding="utf-8") as f:
        split = json.load(f)
    group_membership: dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        for cid in split[split_name]:
            group = cid.split("_")[0]
            if group in group_membership:
                assert group_membership[group] == split_name
            group_membership[group] = split_name
