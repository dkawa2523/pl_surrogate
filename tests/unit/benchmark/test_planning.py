from __future__ import annotations

import pytest

from plasma_surrogate.benchmark.planning import BenchmarkPlanBuilder, build_leaderboard_header


def test_benchmark_plan_builder_resolves_eval_protocol_and_diagnostics_cfg():
    builder = BenchmarkPlanBuilder(
        {
            "eval_protocol": {"mode": "dual_axis", "scope": "common", "primary_split": "interp"},
            "eval": {
                "target_vars_for_score": ["ne", "phi"],
                "region_bands": {"mode": "fixed_px", "boundary_in_px": 3.0},
            },
            "inference": {"diagnostics": {"maps": {"enabled": True}}},
        }
    )

    plan = builder.eval_protocol_plan(y_vars=["ne", "ni", "phi"], scopes=("common",))

    assert plan.mode == "dual_axis"
    assert plan.target_vars_for_score == ["ne", "phi"]
    assert plan.region_bands["boundary_in_px"] == 3.0
    assert builder.inference_ood_cfg()["diagnostics"] == {"maps": {"enabled": True}}


def test_benchmark_plan_builder_rejects_unknown_eval_target():
    with pytest.raises(ValueError, match="target_vars_for_score"):
        BenchmarkPlanBuilder({"eval": {"target_vars_for_score": ["missing"]}}).eval_protocol_plan(
            y_vars=["ne"],
            scopes=("common",),
        )


def test_build_leaderboard_header_keeps_extra_columns_after_core_contract():
    header = build_leaderboard_header(
        target_vars=["ne"],
        leaderboard=[{"model_id": "m", "custom": 1.0}],
    )

    assert header[:5] == ["model_id", "test_rmse_ne", "test_rmse_ne_plasma", "test_r2_ne", "test_r2_ne_plasma"]
    assert header[-1] == "custom"
