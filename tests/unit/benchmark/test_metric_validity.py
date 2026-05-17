import math

from plasma_surrogate.benchmark.runner import BenchmarkRunner, _attach_primary_metric_status


def test_r2_plasma_mean_requires_all_requested_targets_to_be_finite() -> None:
    row = {
        "test_r2_ne_plasma": float("-inf"),
        "test_r2_ni_plasma": 0.1,
        "test_r2_Te_plasma": 0.2,
        "test_r2_phi_plasma": 0.3,
    }

    value, valid, invalid = BenchmarkRunner._r2_plasma_mean_status(
        row,
        target_vars=["ne", "ni", "Te", "phi"],
    )

    assert math.isnan(value)
    assert valid is False
    assert invalid == ["ne"]


def test_r2_plasma_mean_uses_all_requested_targets_when_valid() -> None:
    row = {
        "test_r2_ne_plasma": 0.1,
        "test_r2_ni_plasma": 0.3,
        "test_r2_Te_plasma": 0.5,
        "test_r2_phi_plasma": 0.7,
    }

    value, valid, invalid = BenchmarkRunner._r2_plasma_mean_status(
        row,
        target_vars=["ne", "ni", "Te", "phi"],
    )

    assert value == 0.4
    assert valid is True
    assert invalid == []


def test_primary_metric_reliability_respects_protocol_flag() -> None:
    row = {
        "test_r2_plasma_mean_dual": 0.4,
        "test_rmse_ne_plasma": 1.0,
        "test_r2_ne_plasma": 0.1,
        "test_rmse_ni_plasma": 1.0,
        "test_r2_ni_plasma": 0.2,
        "test_rmse_Te_plasma": 1.0,
        "test_r2_Te_plasma": 0.3,
        "test_rmse_phi_plasma": 1.0,
        "test_r2_phi_plasma": 0.4,
        "primary_metric_protocol_reliable": False,
    }

    _attach_primary_metric_status(
        row,
        primary_metric="test_r2_plasma_mean_dual",
        model_name="ffno",
        target_vars=["ne", "ni", "Te", "phi"],
    )

    assert row["primary_metric_value"] == 0.4
    assert row["target_metrics_valid"] is True
    assert row["primary_metric_reliable"] is False
