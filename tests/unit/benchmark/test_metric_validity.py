import math

from plasma_surrogate.benchmark.runner import BenchmarkRunner, _attach_primary_metric_status, _resolve_primary_metric_config


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
    primary_metric, objective_mode = _resolve_primary_metric_config({})

    assert primary_metric == "surrogate_quality_score"
    assert objective_mode == "min"


def test_primary_metric_accepts_explicit_product_metric() -> None:
    primary_metric, objective_mode = _resolve_primary_metric_config(
        {"primary_metric": "surrogate_quality_score", "objective_mode": "min"}
    )

    assert primary_metric == "surrogate_quality_score"
    assert objective_mode == "min"
