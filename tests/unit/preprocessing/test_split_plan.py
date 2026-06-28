from __future__ import annotations

import pytest

from plasma_surrogate.preprocessing.split_plan import SplitPlanBuilder


def _cases():
    return [
        {"case_id": f"c{i}", "split_group": f"g{i}", "cond": {"p": float(i), "q": float(i % 2)}}
        for i in range(10)
    ]


def test_split_plan_builder_builds_all_product_splits():
    plan = SplitPlanBuilder({"seed": 0, "ratios": [0.6, 0.2, 0.2]}).build(
        cases=_cases(),
        cond_order=["p", "q"],
    )

    assert set(plan.random) == {"train", "val", "test"}
    assert set(plan.interp_marginal) == {"train", "val", "test"}
    assert set(plan.interp_overlap) == {"train", "val", "test"}
    assert set(plan.interp) == {"train", "val", "test"}
    assert set(plan.extrap) == {"train", "val", "test"}
    assert plan.structure_holdout == plan.random


def test_split_plan_builder_rejects_removed_pressure_extrap_keys():
    with pytest.raises(ValueError, match="pressure_extrap"):
        SplitPlanBuilder({"pressure_extrap_key": "p"}).build(cases=_cases(), cond_order=["p"])
