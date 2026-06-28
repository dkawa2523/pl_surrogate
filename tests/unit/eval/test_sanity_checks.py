from __future__ import annotations

import numpy as np

from plasma_surrogate.eval.sanity_checks import EvalResult, build_metric_validity_flags


def test_metric_validity_flags_mark_empty_mask_and_nonfinite_metrics() -> None:
    flags = build_metric_validity_flags(
        target_vars=["ne"],
        metrics_plasma={"ne": float("nan")},
        r2_plasma={"ne": float("nan")},
        finite_stats_plasma={"ne": {"n_active": 0.0, "n_nonfinite": 0.0, "finite_ratio": 1.0}},
        mask_plasma=np.zeros((2, 2), dtype=np.float32),
        quality_components={"surrogate_quality_score": 1.0},
    )

    assert flags["target_metrics_valid"] is False
    assert flags["invalid_target_vars"] == ["ne"]
    assert flags["invalid_reasons"]["ne"] == ["empty_plasma_mask", "nonfinite_rmse", "nonfinite_r2"]
    assert flags["min_active_count_plasma"] == 0.0


def test_eval_result_keeps_core_metrics_and_diagnostics_explicit() -> None:
    result = EvalResult(
        core_metrics={"model_id": "ffno", "test_rmse_ne": 0.1, "surrogate_quality_score": 0.2},
        diagnostics={"score_physics_component": 0.3},
        spatial_tables={"distribution": [{"var": "ne"}]},
    )

    assert result.core_metrics["surrogate_quality_score"] == 0.2
    assert result.diagnostics == {"score_physics_component": 0.3}
    assert result.spatial_tables["distribution"] == [{"var": "ne"}]
