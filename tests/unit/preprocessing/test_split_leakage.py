from __future__ import annotations

from plasma_surrogate.preprocessing.split import (
    build_casewise_splits,
    build_condition_grouped_splits,
    build_extrapolation_split,
    build_group_kfold_splits,
    build_interpolation_overlap_split,
    build_interpolation_overlap_split_with_status,
    build_interpolation_split,
    build_structure_holdout_split,
)


def test_casewise_split_no_overlap():
    ids = [f"case_{i}" for i in range(20)]
    split = build_casewise_splits(ids, seed=42)
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(ids)


def test_casewise_split_enforces_minimum_non_empty_splits():
    ids = [f"case_{i}" for i in range(6)]
    split = build_casewise_splits(ids, seed=0, ratios=(0.8, 0.1, 0.1))
    assert len(split["train"]) >= 1
    assert len(split["val"]) >= 1
    assert len(split["test"]) >= 1


def test_casewise_split_respects_group_boundaries():
    ids = [
        "a_t0",
        "a_t1",
        "b_t0",
        "b_t1",
        "c_t0",
        "c_t1",
        "d_t0",
        "d_t1",
        "e_t0",
        "e_t1",
        "f_t0",
        "f_t1",
    ]
    groups = [cid.split("_")[0] for cid in ids]
    split = build_casewise_splits(ids, split_groups=groups, seed=3, ratios=(0.6, 0.2, 0.2))
    membership_by_group = {}
    for key in ["train", "val", "test"]:
        for cid in split[key]:
            membership_by_group[cid.split("_")[0]] = key
    for g in set(groups):
        rows = [cid for cid in ids if cid.startswith(g + "_")]
        row_splits = {membership_by_group[cid.split("_")[0]] for cid in rows}
        assert len(row_splits) == 1


def test_group_kfold_respects_group_boundaries():
    ids = [f"{g}_t{i}" for g in ["a", "b", "c", "d", "e", "f"] for i in range(2)]
    groups = [cid.split("_")[0] for cid in ids]
    folds = build_group_kfold_splits(ids, groups, n_folds=3, seed=9)
    assert len(folds) == 3
    for split in folds:
        membership = {}
        for k in ["train", "val", "test"]:
            for cid in split[k]:
                membership.setdefault(cid.split("_")[0], set()).add(k)
        assert all(len(v) == 1 for v in membership.values())
        assert len(split["train"]) > 0 and len(split["val"]) > 0 and len(split["test"]) > 0


def test_build_interpolation_split_keeps_train_value_coverage():
    ids = [f"case_{i}" for i in range(9)]
    cond_values = {
        "case_0": {"c0": 0.0, "c1": 0.0},
        "case_1": {"c0": 0.0, "c1": 1.0},
        "case_2": {"c0": 0.0, "c1": 2.0},
        "case_3": {"c0": 1.0, "c1": 0.0},
        "case_4": {"c0": 1.0, "c1": 1.0},
        "case_5": {"c0": 1.0, "c1": 2.0},
        "case_6": {"c0": 2.0, "c1": 0.0},
        "case_7": {"c0": 2.0, "c1": 1.0},
        "case_8": {"c0": 2.0, "c1": 2.0},
    }
    split = build_interpolation_split(ids, cond_values=cond_values, keys=["c0", "c1"], seed=2, ratios=(0.6, 0.2, 0.2))
    train = split["train"]
    c0_vals = {cond_values[c]["c0"] for c in train}
    c1_vals = {cond_values[c]["c1"] for c in train}
    assert c0_vals == {0.0, 1.0, 2.0}
    assert c1_vals == {0.0, 1.0, 2.0}
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_build_interpolation_overlap_split_keeps_duplicate_tuples_in_one_partition():
    ids = [f"case_{i}" for i in range(12)]
    # Duplicate tuples: 6 unique tuples x 2 repeats.
    cond_values = {}
    for i, cid in enumerate(ids):
        base = i % 6
        cond_values[cid] = {"c0": float(base // 3), "c1": float(base % 3)}
    split = build_interpolation_overlap_split(
        ids,
        cond_values=cond_values,
        keys=["c0", "c1"],
        seed=4,
        ratios=(0.6, 0.2, 0.2),
    )
    tr_tuples = {(cond_values[c]["c0"], cond_values[c]["c1"]) for c in split["train"]}
    te_tuples = {(cond_values[c]["c0"], cond_values[c]["c1"]) for c in split["test"]}
    va_tuples = {(cond_values[c]["c0"], cond_values[c]["c1"]) for c in split["val"]}
    assert tr_tuples.isdisjoint(va_tuples)
    assert tr_tuples.isdisjoint(te_tuples)
    assert va_tuples.isdisjoint(te_tuples)
    membership = {
        cid: split_name
        for split_name in ("train", "val", "test")
        for cid in split[split_name]
    }
    for i in range(6):
        assert membership[f"case_{i}"] == membership[f"case_{i + 6}"]
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_build_interpolation_overlap_split_with_status_allows_unique_tuples_with_marginal_coverage():
    ids = [f"case_{i}" for i in range(9)]
    cond_values = {
        cid: {"c0": float(i // 3), "c1": float(i % 3)}
        for i, cid in enumerate(ids)
    }
    out = build_interpolation_overlap_split_with_status(
        ids,
        cond_values=cond_values,
        keys=["c0", "c1"],
        seed=2,
        ratios=(0.6, 0.2, 0.2),
    )
    assert bool(out["feasible"]) is True
    assert str(out["reason"]) == ""
    split = out["split"]
    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_build_extrapolation_split_holds_out_high_end():
    ids = [f"case_{i}" for i in range(10)]
    cond_values = {cid: {"p": float(i)} for i, cid in enumerate(ids)}
    split = build_extrapolation_split(ids, cond_values=cond_values, key="p", holdout_ratio=0.2, val_ratio_within_remain=0.25)
    test_vals = [cond_values[c]["p"] for c in split["test"]]
    train_vals = [cond_values[c]["p"] for c in split["train"]]
    assert min(test_vals) >= max(train_vals)


def test_condition_grouped_split_never_crosses_complete_condition_ties():
    ids = [f"case_{i}" for i in range(12)]
    cond_values = {
        cid: {"p": float(i % 4), "q": float((i % 4) // 2)}
        for i, cid in enumerate(ids)
    }
    split = build_condition_grouped_splits(
        ids,
        cond_values=cond_values,
        keys=["p", "q"],
        seed=7,
        ratios=(0.6, 0.2, 0.2),
    )
    membership = {
        cid: split_name
        for split_name in ("train", "val", "test")
        for cid in split[split_name]
    }
    by_tuple: dict[tuple[float, float], set[str]] = {}
    for cid in ids:
        key = (cond_values[cid]["p"], cond_values[cid]["q"])
        by_tuple.setdefault(key, set()).add(membership[cid])
    assert all(len(partitions) == 1 for partitions in by_tuple.values())


def test_extrapolation_keeps_equal_levels_together():
    ids = [f"case_{i}" for i in range(12)]
    cond_values = {cid: {"p": float(i % 4)} for i, cid in enumerate(ids)}
    split = build_extrapolation_split(
        ids,
        cond_values=cond_values,
        key="p",
        holdout_ratio=0.25,
        val_ratio_within_remain=0.33,
    )
    memberships = {
        split_name: {cond_values[cid]["p"] for cid in split[split_name]}
        for split_name in ("train", "val", "test")
    }
    assert memberships["train"].isdisjoint(memberships["val"])
    assert memberships["train"].isdisjoint(memberships["test"])
    assert memberships["val"].isdisjoint(memberships["test"])


def test_structure_holdout_never_crosses_structure_groups():
    ids = [f"{group}_{idx}" for group in "abcdef" for idx in range(2)]
    groups = [cid.split("_")[0] for cid in ids]
    split = build_structure_holdout_split(
        ids,
        structure_groups=groups,
        seed=3,
        ratios=(0.6, 0.2, 0.2),
    )
    membership = {
        cid: split_name
        for split_name in ("train", "val", "test")
        for cid in split[split_name]
    }
    for group in set(groups):
        assert len({membership[cid] for cid in ids if cid.startswith(group + "_")}) == 1
