from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import summarize_gec_ccp_nn_operator_comparison_v1 as summary


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _resolved(seed: int, model: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "dataset": {
            "index_csv": summary.DATASET_INDEX,
            "cond_columns": ["PP0", "Td", "gamma", "PA"],
            "targets": [
                {"id": target, "value_transform": "identity"}
                for target in summary.TARGETS
            ],
        },
        "split": {"seed": 7, "ratios": [0.7, 0.15, 0.15]},
        "eval": {"target_vars_for_score": list(summary.TARGETS)},
        "eval_protocol": {
            "mode": "primary_axis",
            "primary_split": "interp",
            "interp_mode": "marginal",
        },
        "preprocessing": {
            "scalers": {
                "y_fit_policy": "plasma_only",
                "target_transforms": {
                    target: {
                        "value_transform": "identity",
                        "scaler": "zscore",
                        "fit_scope": "plasma_only",
                    }
                    for target in summary.TARGETS
                },
            }
        },
        "train": {
            "loss": {
                "protocol": summary.LOSS_PROTOCOL,
                "supervised": {
                    "type": "huber",
                    "mask": "plasma_only",
                    "normalization": "sample_mean",
                    "spatial": {
                        "gradient_weight": 0.1,
                        "multiscale_weight": 0.05,
                        "multiscale_scales": [2, 4],
                        "boundary_weight": 0.25,
                        "boundary_band_px": 2.0,
                        "boundary_distance_channels": list(summary.BOUNDARY_CHANNELS),
                    },
                },
                "group_weighting": {"mode": "uniform_by_group"},
            },
            model: {
                "selection": {
                    "mode": summary.SELECTION_MODE,
                    "weights": {target: 0.25 for target in summary.TARGETS},
                }
            },
        },
    }


def _write_seed(
    run_root: Path,
    *,
    seed: int,
    model: str,
    validation: float,
    quality: float,
) -> None:
    model_root = run_root / f"seed_{seed}" / "n78" / model
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_resolved(seed, model), sort_keys=False),
        encoding="utf-8",
    )

    leaderboard: dict[str, Any] = {
        "model_id": model,
        "target_metrics_valid": True,
        "primary_metric_reliable": True,
        "validation_selection_reliable": True,
        "scaler_train_only": True,
        "evaluation_geometry_case_aligned": True,
        "selection_split": "interp",
        "scaler_fit_split": "interp",
        "validation_selection_mode": "min",
        "validation_selection_metric": "selected_epoch_score",
        "validation_selection_objective_version": summary.SELECTION_OBJECTIVE_VERSION,
        "validation_selection_value": validation,
        "surrogate_quality_score": quality,
        "quality_score_protocol": summary.QUALITY_PROTOCOL,
        "quality_score_definition_hash": "quality-hash",
        "evaluation_boundary_distance_channels": str(list(summary.BOUNDARY_CHANNELS)),
        "protocol_variant": f"comparison_seed{seed}",
    }
    for target_index, target in enumerate(summary.TARGETS):
        leaderboard[f"test_rmse_{target}_plasma"] = quality + target_index
        leaderboard[f"test_r2_{target}_plasma"] = 0.5 + 0.01 * target_index
    _write_csv(model_root / "leaderboard.csv", leaderboard)

    diagnostics: dict[str, Any] = {"model_id": model}
    for field in summary.METRIC_FIELDS:
        if field not in {"surrogate_quality_score", "test_r2_plasma_mean"} and field not in leaderboard:
            diagnostics[field] = quality
    _write_csv(model_root / "diagnostics" / "diagnostics.csv", diagnostics)

    manifest = {
        "training": {
            "loss_protocol_effective": summary.LOSS_PROTOCOL,
            "loss_protocol_definition_hash": "loss-hash",
        },
        "artifacts": {"artifact_hashes": {"split_hash": "split-hash"}},
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = {
        "scaler_fit_split": "interp",
        "scaler_train_only": True,
        "loss_protocol_definition_hash": "loss-hash",
        "target_schema_hash": "target-hash",
        "feature_schema_hash": f"feature-{model}",
    }
    checkpoint_path = (
        model_root
        / "models"
        / model
        / "eval_protocol"
        / "interp"
        / "checkpoints"
        / "meta.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")


def _write_status(run_root: Path, *, seed: int, models: list[str]) -> None:
    path = run_root / f"seed_{seed}" / "run_status.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_size", "model_id", "status", "seconds"],
        )
        writer.writeheader()
        for model in models:
            writer.writerow(
                {
                    "dataset_size": 78,
                    "model_id": model,
                    "status": "passed",
                    "seconds": 10.0 + seed,
                }
            )


def test_comparison_summary_selects_seed_only_by_minimum_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "comparison"
    models = ["global_mlp", "fno"]
    seeds = [11, 12, 13]
    settings = {
        11: (0.30, 0.01),  # best test quality, not selected
        12: (0.10, 0.80),  # minimum validation, selected
        13: (0.20, 0.40),
    }
    for seed in seeds:
        validation, quality = settings[seed]
        for model in models:
            _write_seed(
                run_root,
                seed=seed,
                model=model,
                validation=validation,
                quality=quality,
            )
        _write_status(run_root, seed=seed, models=models)

    paths = summary.summarize_comparison(
        run_root=run_root,
        out_dir=run_root / "evaluation" / "seed_matrix",
        size=78,
        seeds=seeds,
        models=models,
    )

    assert all(path.exists() for path in paths)
    with paths[1].open("r", encoding="utf-8", newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    assert {row["validation_selected_seed"] for row in aggregate} == {"12"}
    assert float(aggregate[0]["surrogate_quality_score_mean"]) == pytest.approx(
        (0.01 + 0.80 + 0.40) / 3.0
    )
    quality_std = statistics.stdev([0.01, 0.80, 0.40])
    assert float(aggregate[0]["surrogate_quality_score_std"]) == pytest.approx(quality_std)
    assert quality_std > 0.0
    with paths[2].open("r", encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    assert {row["seed"] for row in selected} == {"12"}
    assert {row["selection_split"] for row in selected} == {"interp"}
    assert {row["test_metrics_used_for_selection"] for row in selected} == {"False"}
    report = paths[3].read_text(encoding="utf-8")
    assert "Representative plotting seeds use only" in report
    assert "Physical diagnostics (three-seed means)" in report
    assert "does not penalize sign violations" in report


def test_comparison_seed_audit_rejects_non_interp_protocol(tmp_path: Path) -> None:
    run_root = tmp_path / "comparison"
    _write_seed(run_root, seed=11, model="fno", validation=0.1, quality=0.2)
    _write_status(run_root, seed=11, models=["fno"])
    cfg_path = run_root / "seed_11" / "n78" / "fno" / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["eval_protocol"]["primary_split"] = "extrap"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="eval_protocol.primary_split mismatch"):
        summary._seed_row(run_root=run_root, seed=11, size=78, model="fno")


def test_comparison_seed_audit_rejects_nonfinite_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "comparison"
    _write_seed(run_root, seed=11, model="fno", validation=float("nan"), quality=0.2)
    _write_status(run_root, seed=11, models=["fno"])

    with pytest.raises(ValueError, match="validation_selection_value must be finite"):
        summary._seed_row(run_root=run_root, seed=11, size=78, model="fno")
