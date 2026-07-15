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
    assert plan.structure_holdout_meta["is_real_structure_holdout"] is False


def test_split_plan_builder_builds_real_structure_holdout_from_base_name():
    cases = [
        {
            "case_id": f"{base}_{i}",
            "split_group": f"{base}_{i}",
            "base_case_id": f"operating_case_{base}_{i}",
            "base_name": base,
            "cond": {"p": float(i), "q": float(i % 2)},
        }
        for base in ("base2", "base3", "base4")
        for i in range(4)
    ]
    plan = SplitPlanBuilder(
        {
            "seed": 0,
            "ratios": [0.5, 0.25, 0.25],
            "structure_holdout": {"required": True},
        }
    ).build(cases=cases, cond_order=["p", "q"])
    membership = {
        cid: split_name
        for split_name in ("train", "val", "test")
        for cid in plan.structure_holdout[split_name]
    }
    assert plan.structure_holdout_meta["is_real_structure_holdout"] is True
    assert plan.structure_holdout_meta["group_keys"] == ["base_name"]
    for base in ("base2", "base3", "base4"):
        assert len({membership[c["case_id"]] for c in cases if c["base_name"] == base}) == 1


def test_split_plan_builder_fails_fast_when_structure_holdout_is_required_but_missing():
    with pytest.raises(ValueError, match="Real structure holdout requested but unavailable"):
        SplitPlanBuilder({"structure_holdout": {"required": True}}).build(
            cases=_cases(),
            cond_order=["p", "q"],
        )


def test_split_plan_builder_rejects_removed_pressure_extrap_keys():
    with pytest.raises(ValueError, match="pressure_extrap"):
        SplitPlanBuilder({"pressure_extrap_key": "p"}).build(cases=_cases(), cond_order=["p"])


def test_split_plan_builder_rejects_disabling_condition_tie_grouping():
    with pytest.raises(ValueError, match="condition_group_ties=false"):
        SplitPlanBuilder({"condition_group_ties": False}).build(cases=_cases(), cond_order=["p", "q"])
