from __future__ import annotations

import math

import pytest

from plasma_surrogate.benchmark.tuning_selection import (
    require_validation_objective,
    validation_selection_from_history,
)


def test_validation_selection_uses_selected_balance_epoch_and_max_direction() -> None:
    out = validation_selection_from_history(
        [
            {
                "epoch": 0,
                "val_loss": 0.4,
                "selected_epoch_flag": 0,
                "selected_epoch_score": 0.1,
                "selection_mode_effective": "best_val_allvars_balance",
                "selection_valid_flag": 1,
            },
            {
                "epoch": 1,
                "val_loss": 0.5,
                "selected_epoch_flag": 1,
                "selected_epoch_score": 0.8,
                "selection_mode_effective": "best_val_allvars_balance",
                "selection_valid_flag": 1,
            },
        ]
    )

    assert out["validation_selection_value"] == 0.8
    assert out["validation_selection_mode"] == "max"
    assert out["validation_selected_epoch"] == 1
    assert out["validation_selection_reliable"] is True


def test_validation_selection_uses_val_loss_for_last_mode() -> None:
    out = validation_selection_from_history([{"epoch": 3, "val_loss": 0.25}])

    assert out["validation_selection_value"] == 0.25
    assert out["validation_selection_mode"] == "min"
    assert out["validation_selection_reliable"] is True


def test_validation_selection_uses_spatial_score_with_min_direction() -> None:
    out = validation_selection_from_history(
        [
            {
                "epoch": 2,
                "val_loss": 0.01,
                "selected_epoch_flag": 1,
                "selected_epoch_score": 0.42,
                "selection_mode_effective": "best_val_spatial_objective",
                "selection_objective_version": "case_macro_spatial_rmse_v1",
                "selection_valid_flag": 1,
            }
        ]
    )

    assert out["validation_selection_metric"] == "selected_epoch_score"
    assert out["validation_selection_value"] == 0.42
    assert out["validation_selection_mode"] == "min"
    assert out["validation_selection_objective_version"] == "case_macro_spatial_rmse_v1"
    assert out["validation_selection_reliable"] is True


def test_require_validation_objective_rejects_legacy_test_only_row() -> None:
    with pytest.raises(ValueError, match="must be retrained"):
        require_validation_objective({"surrogate_quality_score": 0.01})


def test_empty_history_is_not_a_finite_outer_objective() -> None:
    out = validation_selection_from_history([])
    assert math.isnan(out["validation_selection_value"])
    assert out["validation_selection_reliable"] is False
