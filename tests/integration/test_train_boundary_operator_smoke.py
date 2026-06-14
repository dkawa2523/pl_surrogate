from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from plasma_surrogate.cli.main import main
from tests._config_presets import default_target_transforms_four_field_example, runtime_table_only


def test_train_with_boundary_operator_smoke(tmp_path: Path):
    run_dir = tmp_path / "boundary_op_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 6},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        "model": {"name": "global_mlp", "phi_mode": "direct"},
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {
                "enabled": True,
                "terms": {
                    "poisson": {"weight": 0.02},
                    "boundary": {"weight": 0.01},
                    "boundary_operator": {"weight": 0.05},
                },
                "boundary_operator": {
                    "enabled": True,
                    "delta_edge": 1.5,
                    "target_coeffs": {"log_ne": 0.1, "Te": 0.05, "bias": 0.0},
                },
            },
        },
    }
    cfg_path = tmp_path / "boundary_op.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0

    metrics_path = run_dir / "train" / "scalars" / "metrics.csv"
    physics_terms_path = run_dir / "train" / "scalars" / "physics_terms.csv"
    resolved_physics_path = run_dir / "train" / "resolved_physics.json"
    assert metrics_path.exists()
    assert physics_terms_path.exists()
    assert resolved_physics_path.exists()
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "train_boundary_operator_loss" in rows[0]
    with physics_terms_path.open("r", encoding="utf-8") as f:
        phys_rows = list(csv.DictReader(f))
    assert phys_rows
    assert "train_boundary_operator_loss" in phys_rows[0]
    with resolved_physics_path.open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["enabled"] is True
    assert any(row["name"] == "boundary_operator" for row in resolved["resolved_terms"])


def test_train_with_boundary_operator_external_stub_smoke(tmp_path: Path):
    run_dir = tmp_path / "boundary_op_external_run"
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 5},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        "model": {"name": "global_mlp", "phi_mode": "direct"},
        "train": {
            "epochs": 2,
            "lr": 0.01,
            "physics": {
                "enabled": True,
                "terms": {
                    "poisson": {"weight": 0.01},
                    "boundary": {"weight": 0.01},
                    "boundary_operator": {"weight": 0.05},
                },
                "boundary_operator": {
                    "enabled": True,
                    "delta_edge": 1.5,
                    "mode": "external_operator",
                    "external_operator": {
                        "w_log_ne": 0.08,
                        "w_te": 0.06,
                        "w_en": 0.04,
                        "bias": 0.0,
                    },
                },
            },
        },
    }
    cfg_path = tmp_path / "boundary_op_external.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    try:
        main(["train", "--config", str(cfg_path)])
    except ValueError as exc:
        assert "external_operator is removed from mainline" in str(exc)
    else:
        raise AssertionError("external_operator mode should fail-fast on mainline")
