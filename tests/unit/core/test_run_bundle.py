from __future__ import annotations

import json

import pytest

from plasma_surrogate.core.data_cleaning_audit import run_data_audit
from plasma_surrogate.core.run_bundle import (
    RunBundleLoader,
    ensure_preprocess_contract,
    require_artifacts,
    required_preprocess_artifacts,
)


def test_run_bundle_loader_loads_assets(run_dir):
    bundle = RunBundleLoader.load(run_dir)
    assert bundle.task_spec.grid_spec.shape == (8, 8)
    assert "cond_schema" in bundle.schemas
    assert bundle.transforms["cond_scaler"]["type"] == "zscore"


def test_run_bundle_checkpoint_scaler_uses_recorded_protocol(monkeypatch, run_dir):
    bundle = RunBundleLoader.load(run_dir)
    bundle.transforms["protocol_transforms"] = {
        "interp": {"cond_scaler": {"type": "sentinel"}, "y_scalers": {"phi": {"type": "sentinel"}}}
    }
    captured = {}

    def _fake_transform(split_name=None, *, require_protocol=False):
        captured["split_name"] = split_name
        captured["require_protocol"] = require_protocol
        return object()

    monkeypatch.setattr(bundle, "transform_bundle", _fake_transform)
    bundle.transform_bundle_for_checkpoint({"scaler_fit_split": "interp"})

    assert captured == {"split_name": "interp", "require_protocol": True}


def test_run_bundle_checkpoint_scaler_rejects_missing_split_provenance(run_dir):
    bundle = RunBundleLoader.load(run_dir)
    bundle.transforms["protocol_transforms"] = {
        "extrap": {"cond_scaler": {"type": "sentinel"}, "y_scalers": {"phi": {"type": "sentinel"}}}
    }

    with pytest.raises(ValueError, match="missing scaler_fit_split"):
        bundle.transform_bundle_for_checkpoint({})


def test_run_bundle_checkpoint_spatial_scaler_uses_recorded_protocol(run_dir):
    bundle = RunBundleLoader.load(run_dir)
    bundle.transforms["protocol_transforms"] = {
        "extrap": {
            "cond_scaler": {"type": "sentinel"},
            "y_scalers": {"phi": {"type": "sentinel"}},
            "coord_feature_scaler": {"contract_version": 3, "enabled": False},
            "distance_transform_stats": {"enabled": True, "signed_tanh_tau_auto": 2.0},
        }
    }

    selected = bundle.spatial_transform_artifacts_for_checkpoint({"scaler_fit_split": "extrap"})

    assert selected["coord_feature_scaler"]["contract_version"] == 3
    assert selected["distance_transform_stats"]["signed_tanh_tau_auto"] == 2.0


def test_run_bundle_loader_loads_optional_deeponet_schema(run_dir):
    dpath = run_dir / "preprocessing" / "sampling" / "deeponet" / "default"
    dpath.mkdir(parents=True)
    payload = {
        "sensor_indices": [0, 1, 2],
        "query_indices": [3, 4, 5],
        "seed": 7,
        "strategy": "uniform_fixed",
    }
    with (dpath / "sensor_query_index.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    with (dpath / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"flatten_order": "C", "grid_shape": [8, 8], "n_points": 64, "coord_system": "cartesian"}, f)

    bundle = RunBundleLoader.load(run_dir)
    assert bundle.schemas["deeponet_index"]["seed"] == 7
    assert bundle.schemas["deeponet_index"]["strategy"] == "uniform_fixed"
    assert bundle.schemas["deeponet_index_meta"]["grid_shape"] == [8, 8]


def test_run_bundle_loader_requires_task_spec_yaml(tmp_path):
    run_dir = tmp_path / "benchmark_like"
    (run_dir / "preprocessing" / "scalers").mkdir(parents=True)
    (run_dir / "preprocessing" / "schema").mkdir(parents=True)
    with (run_dir / "preprocessing" / "scalers" / "cond_scaler.json").open("w", encoding="utf-8") as f:
        json.dump({"type": "zscore", "mean": [0.0, 0.0], "std": [1.0, 1.0], "cond_dim": 2}, f)
    with (run_dir / "preprocessing" / "scalers" / "y_scalers.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ne": {"type": "zscore", "mean": [0.0], "std": [1.0]},
                "Te": {"type": "zscore", "mean": [0.0], "std": [1.0]},
                "phi": {"type": "zscore", "mean": [0.0], "std": [1.0]},
            },
            f,
        )
    with (run_dir / "preprocessing" / "schema" / "cond_schema.json").open("w", encoding="utf-8") as f:
        json.dump({"order": ["p", "q"]}, f)
    with (run_dir / "preprocessing" / "schema" / "axis_schema.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": "steady", "harmonics": 1}, f)
    with (run_dir / "preprocessing" / "schema" / "output_layout.json").open("w", encoding="utf-8") as f:
        json.dump({"order": "C", "shape": [3, 8, 8], "vars": ["ne", "Te", "phi"]}, f)


    with pytest.raises(FileNotFoundError, match="task_spec.yaml"):
        RunBundleLoader.load(run_dir)


def test_require_artifacts_raises_with_missing_paths(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "preprocessing" / "schema").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing required artifacts"):
        require_artifacts(run_dir, ["preprocessing/schema/cond_schema.json"])


def test_ensure_preprocess_contract_uses_default_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "preprocessing" / "schema").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ensure_preprocess_contract(run_dir)


def test_required_preprocess_artifacts_contains_split_and_scalers():
    required = required_preprocess_artifacts()
    assert "preprocessing/split/split_random_v1.json" in required
    assert "preprocessing/scalers/cond_scaler.json" in required


def test_run_data_audit_counts_and_hist():
    cases = [
        {"case_id": "a", "cond": {"c0": 0.1, "c1": 0.2}, "axis": 0.0, "y": {"ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]}},
        {"case_id": "b", "cond": {"c0": 1.2, "c1": 0.5}, "axis": 1.2, "y": {"ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]}},
        {"case_id": "c", "cond": {"c0": 0.3}, "y": {"ne": [[0.0]], "Te": [[0.0]]}},
    ]
    out = run_data_audit(cases, cond_order=["c0", "c1"], axis_mode="time")
    assert out["n_cases"] == 3
    assert out["missing_counts"]["cond"] >= 1
    assert out["missing_counts"]["axis"] >= 1
    assert out["missing_counts"]["y_phi"] >= 1
    assert out["range_violations"]["cond_out_of_unit_interval"] >= 1
    assert out["range_violations"]["axis_out_of_range"] >= 1
    assert out["axis_hist"]["mode"] == "time"
    assert len(out["axis_hist"]["counts"]) == 10
