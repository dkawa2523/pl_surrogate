import math

import numpy as np

from plasma_surrogate.benchmark.runner import (
    BenchmarkModelResult,
    BenchmarkProbe,
    BenchmarkRunner,
    _attach_primary_metric_status,
)
from plasma_surrogate.benchmark.planning import resolve_primary_metric_config


def test_r2_plasma_mean_requires_all_requested_targets_to_be_finite() -> None:
    row = {
        "test_r2_electron_density_plasma": float("-inf"),
        "test_r2_ion_density_plasma": 0.1,
        "test_r2_electron_temperature_plasma": 0.2,
        "test_r2_potential_plasma": 0.3,
    }

    value, valid, invalid = BenchmarkRunner._r2_plasma_mean_status(
        row,
        target_vars=["electron_density", "ion_density", "electron_temperature", "potential"],
    )

    assert math.isnan(value)
    assert valid is False
    assert invalid == ["electron_density"]


def test_r2_plasma_mean_uses_all_requested_targets_when_valid() -> None:
    row = {
        "test_r2_electron_density_plasma": 0.1,
        "test_r2_ion_density_plasma": 0.3,
        "test_r2_electron_temperature_plasma": 0.5,
        "test_r2_potential_plasma": 0.7,
    }

    value, valid, invalid = BenchmarkRunner._r2_plasma_mean_status(
        row,
        target_vars=["electron_density", "ion_density", "electron_temperature", "potential"],
    )

    assert value == 0.4
    assert valid is True
    assert invalid == []


def test_primary_metric_reliability_uses_target_metric_validity() -> None:
    row = {
        "surrogate_quality_score": 0.4,
        "test_rmse_electron_density_plasma": 1.0,
        "test_r2_electron_density_plasma": 0.1,
        "test_rmse_ion_density_plasma": 1.0,
        "test_r2_ion_density_plasma": 0.2,
    }

    _attach_primary_metric_status(
        row,
        primary_metric="surrogate_quality_score",
        model_name="ffno",
        target_vars=["electron_density", "ion_density"],
    )

    assert row["primary_metric_value"] == 0.4
    assert row["target_metrics_valid"] is True
    assert row["primary_metric_reliable"] is True


def test_primary_metric_default_is_surrogate_quality_score_min() -> None:
    primary_metric, objective_mode = resolve_primary_metric_config({})

    assert primary_metric == "surrogate_quality_score"
    assert objective_mode == "min"


def test_primary_metric_accepts_explicit_product_metric() -> None:
    primary_metric, objective_mode = resolve_primary_metric_config(
        {"primary_metric": "surrogate_quality_score", "objective_mode": "min"}
    )

    assert primary_metric == "surrogate_quality_score"
    assert objective_mode == "min"


def test_dual_axis_row_combines_interp_and_extrap_metrics() -> None:
    split_rows = {
        "interp": {
            "model_id": "global_mlp",
            "surrogate_quality_score": 1.0,
            "test_r2_ne_plasma": 0.8,
            "test_r2_Te_plasma": 0.6,
        },
        "extrap": {
            "model_id": "global_mlp",
            "surrogate_quality_score": 3.0,
            "test_r2_ne_plasma": 0.4,
            "test_r2_Te_plasma": 0.2,
        },
    }

    row = BenchmarkRunner._combine_dual_axis_rows(
        split_rows=split_rows,
        primary_split="interp",
        interp_weight=0.25,
        extrap_weight=0.75,
        target_vars=["ne", "Te"],
    )

    assert math.isclose(row["surrogate_quality_score"], 2.5)
    assert row["surrogate_quality_score_interp"] == 1.0
    assert row["surrogate_quality_score_extrap"] == 3.0
    assert math.isclose(row["test_r2_plasma_mean_interp"], 0.7)
    assert math.isclose(row["test_r2_plasma_mean_extrap"], 0.3)
    assert math.isclose(row["test_r2_plasma_mean_dual"], 0.4)


def test_benchmark_model_result_reports_skip_without_fake_objective() -> None:
    summary = BenchmarkModelResult(
        status="skipped",
        skip_reason="case_varying_structure_inputs_not_supported_by_benchmark_inference",
    ).as_summary()

    assert summary["status"] == "skipped"
    assert summary["skip_reason"] == "case_varying_structure_inputs_not_supported_by_benchmark_inference"
    assert summary["objective_value"] is None
    assert summary["search_value"] is None
    assert summary["feasible"] is False
    assert "best_objective_value" not in summary
    assert "best_search_value" not in summary
    assert "objective_key" not in summary


def test_benchmark_probe_is_explicit_opt_in(tmp_path) -> None:
    assert BenchmarkProbe({}).enabled() is False
    assert BenchmarkProbe({"inference": {"benchmark_probe": {"enabled": True}}}).enabled() is True

    probe = BenchmarkProbe({}).run(
        model=object(),
        model_name="global_mlp",
        model_idx=0,
        model_dir=tmp_path,
        context=object(),
        profile_lock={},
        train_cfg={},
        effective_input_mode_meta={},
        true_eval={"phi": np.zeros((1, 1, 2, 2), dtype=np.float32)},
        pred_eval={"phi": np.zeros((1, 1, 2, 2), dtype=np.float32)},
        metric_mask=None,
        te_idx=np.asarray([0], dtype=np.int64),
        viz=object(),
    )

    assert probe.optimize.status == "skipped"
    assert probe.optimize.skip_reason == "benchmark_probe_disabled"
    assert probe.batch_rows == []
