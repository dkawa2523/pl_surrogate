"""Split utilities with casewise leakage prevention."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Sequence

import numpy as np


def _combined_group_labels(
    case_ids: Sequence[str],
    *group_constraints: Sequence[Hashable] | None,
) -> list[str]:
    """Return connected-component labels satisfying every supplied grouping constraint."""

    ids = [str(v) for v in case_ids]
    n = len(ids)
    parent = list(range(n))

    def _find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for constraint in group_constraints:
        if constraint is None:
            continue
        labels = list(constraint)
        if len(labels) != n:
            raise ValueError("group constraint must have the same length as case_ids")
        first_by_label: dict[Hashable, int] = {}
        for idx, label in enumerate(labels):
            if label in first_by_label:
                _union(first_by_label[label], idx)
            else:
                first_by_label[label] = idx

    roots = [_find(idx) for idx in range(n)]
    canonical: dict[int, str] = {}
    for idx, root in enumerate(roots):
        canonical.setdefault(root, ids[idx])
    return [canonical[root] for root in roots]


def _condition_tuple_labels(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
) -> list[tuple[float, ...]]:
    keys_list = [str(key) for key in keys]
    if not keys_list:
        return [tuple() for _ in case_ids]
    out: list[tuple[float, ...]] = []
    for raw_cid in case_ids:
        cid = str(raw_cid)
        if cid not in cond_values:
            raise KeyError(f"Missing condition values for case_id={cid!r}")
        missing = [key for key in keys_list if key not in cond_values[cid]]
        if missing:
            raise KeyError(f"Missing condition keys for case_id={cid!r}: {missing}")
        values = tuple(float(cond_values[cid][key]) for key in keys_list)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Condition tuple contains non-finite values for case_id={cid!r}: {values}")
        out.append(values)
    return out


def build_condition_grouped_splits(
    case_ids: Sequence[str],
    *,
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    split_groups: Sequence[str] | None = None,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Split without crossing either user groups or complete condition-tuple ties."""

    tuple_labels = _condition_tuple_labels(case_ids, cond_values, keys)
    combined = _combined_group_labels(case_ids, split_groups, tuple_labels)
    return build_casewise_splits(case_ids, split_groups=combined, seed=seed, ratios=ratios)


def build_casewise_splits(
    case_ids: Sequence[str],
    split_groups: Sequence[str] | None = None,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    if len(case_ids) == 0:
        raise ValueError("No case IDs provided")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.0")

    ids = np.array(list(case_ids), dtype=object)
    groups = np.array(list(split_groups), dtype=object) if split_groups is not None else ids.copy()
    if groups.shape[0] != ids.shape[0]:
        raise ValueError("split_groups must have the same length as case_ids")

    unique_groups = np.unique(groups)
    n = len(unique_groups)
    if n < 3:
        raise ValueError("At least 3 case IDs are required to build non-empty train/val/test splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    counts = [n_train, n_val, n_test]

    # Ensure minimum 1 sample per split while preserving total.
    for idx in [1, 2, 0]:
        while counts[idx] < 1:
            donors = [j for j in range(3) if j != idx and counts[j] > 1]
            if not donors:
                raise ValueError("Unable to allocate at least one case for each split")
            donor = max(donors, key=lambda j: counts[j])
            counts[donor] -= 1
            counts[idx] += 1

    n_train, n_val, n_test = counts

    train_groups = set(unique_groups[:n_train].tolist())
    val_groups = set(unique_groups[n_train : n_train + n_val].tolist())
    test_groups = set(unique_groups[n_train + n_val : n_train + n_val + n_test].tolist())

    train = [str(case_id) for case_id, group_id in zip(ids.tolist(), groups.tolist()) if group_id in train_groups]
    val = [str(case_id) for case_id, group_id in zip(ids.tolist(), groups.tolist()) if group_id in val_groups]
    test = [str(case_id) for case_id, group_id in zip(ids.tolist(), groups.tolist()) if group_id in test_groups]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def _build_edge_extrapolation_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    key: str,
    holdout_ratio: float = 0.2,
    val_ratio_within_remain: float = 0.2,
    direction: str = "high",
) -> dict[str, list[str]]:
    """Build deterministic extrapolation split by holding out one edge for a condition key."""

    ids = list(case_ids)
    if len(ids) < 3:
        raise ValueError("At least 3 case IDs are required")
    if key == "":
        raise ValueError("split key must not be empty")
    direction_norm = str(direction).strip().lower()
    if direction_norm not in {"high", "low"}:
        raise ValueError("extrapolation direction must be one of: high, low")
    if holdout_ratio <= 0.0 or holdout_ratio >= 1.0:
        raise ValueError("holdout_ratio must be in (0,1)")
    if val_ratio_within_remain <= 0.0 or val_ratio_within_remain >= 1.0:
        raise ValueError("val_ratio_within_remain must be in (0,1)")

    level_by_case: dict[str, float] = {}
    cases_by_level: dict[float, list[str]] = {}
    for cid in ids:
        if cid not in cond_values or key not in cond_values[cid]:
            raise KeyError(f"Missing extrapolation condition key={key!r} for case_id={cid!r}")
        level = float(cond_values[cid][key])
        if not np.isfinite(level):
            raise ValueError(f"Extrapolation condition is non-finite for case_id={cid!r}, key={key!r}")
        level_by_case[cid] = level
        cases_by_level.setdefault(level, []).append(cid)

    levels = sorted(cases_by_level)
    if len(levels) < 3:
        raise ValueError(
            "Extrapolation split requires at least 3 distinct condition levels so train/val/test "
            f"can remain level-disjoint; key={key!r}, levels={levels}"
        )

    def _edge_count(ordered_levels: list[float], *, target_count: int, leave_levels: int) -> int:
        max_take = len(ordered_levels) - int(leave_levels)
        candidates: list[tuple[int, int]] = []
        running = 0
        for take in range(1, max_take + 1):
            running += len(cases_by_level[ordered_levels[take - 1]])
            candidates.append((abs(running - target_count), take))
        if not candidates:
            raise ValueError("Unable to allocate non-empty level-disjoint extrapolation split")
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    edge_order = levels if direction_norm == "low" else list(reversed(levels))
    target_test_count = max(1, int(round(len(ids) * holdout_ratio)))
    n_test_levels = _edge_count(edge_order, target_count=target_test_count, leave_levels=2)
    test_levels = set(edge_order[:n_test_levels])
    remain_levels = [level for level in levels if level not in test_levels]

    remain_edge_order = remain_levels if direction_norm == "low" else list(reversed(remain_levels))
    remain_count = sum(len(cases_by_level[level]) for level in remain_levels)
    target_val_count = max(1, int(round(remain_count * val_ratio_within_remain)))
    n_val_levels = _edge_count(remain_edge_order, target_count=target_val_count, leave_levels=1)
    val_levels = set(remain_edge_order[:n_val_levels])
    train_levels = set(remain_levels) - val_levels

    train = [cid for cid in ids if level_by_case[cid] in train_levels]
    val = [cid for cid in ids if level_by_case[cid] in val_levels]
    test = [cid for cid in ids if level_by_case[cid] in test_levels]
    if not train or not val or not test:
        raise RuntimeError("Level-disjoint extrapolation split unexpectedly produced an empty partition")
    return {"train": train, "val": val, "test": test}


def build_extrapolation_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    key: str,
    direction: str = "high",
    holdout_ratio: float = 0.2,
    val_ratio_within_remain: float = 0.2,
) -> dict[str, list[str]]:
    """Build deterministic extrapolation split by holding out high or low condition values."""

    return _build_edge_extrapolation_split(
        case_ids=case_ids,
        cond_values=cond_values,
        key=key,
        direction=direction,
        holdout_ratio=holdout_ratio,
        val_ratio_within_remain=val_ratio_within_remain,
    )


def build_interpolation_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    mode: str = "marginal",
    split_groups: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """
    Build interpolation-oriented split with train coverage across marginal condition values.

    This keeps case-level separation while ensuring each selected condition axis value appears in train.
    """

    mode_norm = str(mode).strip().lower()
    if mode_norm == "overlap":
        return build_interpolation_overlap_split(
            case_ids=case_ids,
            cond_values=cond_values,
            keys=keys,
            seed=seed,
            ratios=ratios,
            split_groups=split_groups,
        )
    if mode_norm != "marginal":
        raise ValueError("mode must be one of: marginal, overlap")

    keys_list = [str(k) for k in keys]
    if len(keys_list) == 0:
        return build_casewise_splits(
            case_ids=case_ids,
            split_groups=split_groups,
            seed=seed,
            ratios=ratios,
        )

    tuple_labels = _condition_tuple_labels(case_ids, cond_values, keys_list)
    combined_groups = _combined_group_labels(case_ids, split_groups, tuple_labels)
    split = build_casewise_splits(
        case_ids=case_ids,
        split_groups=combined_groups,
        seed=seed,
        ratios=ratios,
    )

    train = list(split["train"])
    val = list(split["val"])
    test = list(split["test"])
    all_ids = list(case_ids)
    all_vals = {
        k: sorted({float(cond_values[cid][k]) for cid in all_ids if cid in cond_values})
        for k in keys_list
    }

    group_by_case = {str(cid): str(group) for cid, group in zip(case_ids, combined_groups)}

    def _move_group_to_train(group_id: str) -> None:
        members = [str(cid) for cid in case_ids if group_by_case[str(cid)] == group_id]
        for cid in members:
            if cid in val:
                val.remove(cid)
            if cid in test:
                test.remove(cid)
            if cid not in train:
                train.append(cid)

    group_ids_by_level: dict[str, dict[float, set[str]]] = {
        key: {
            value: {
                group_by_case[cid]
                for cid in all_ids
                if np.isclose(float(cond_values[cid][key]), value)
            }
            for value in all_vals[key]
        }
        for key in keys_list
    }

    # Ensure train contains each marginal value that can be held out independently.  A value
    # represented by only one indivisible group cannot simultaneously occur in train and a
    # held-out partition, so keeping that group held out is preferable to emptying val/test.
    for k in keys_list:
        for v in all_vals[k]:
            if any(np.isclose(float(cond_values[c][k]), v) for c in train):
                continue
            if len(group_ids_by_level[k][v]) < 2:
                continue
            cand = [c for c in (val + test) if np.isclose(float(cond_values[c][k]), v)]
            if not cand:
                continue
            pick = cand[0]
            _move_group_to_train(group_by_case[pick])

    # Keep val/test non-empty without sacrificing train marginal coverage.
    rng = np.random.default_rng(seed + 101)
    for bucket in (val, test):
        if len(bucket) != 0:
            continue
        train_groups = sorted({group_by_case[cid] for cid in train})
        rng.shuffle(train_groups)
        moved = False
        for group_id in train_groups:
            members = [cid for cid in train if group_by_case[cid] == group_id]
            remaining = [cid for cid in train if group_by_case[cid] != group_id]
            if not remaining:
                continue
            if any(
                value not in {float(cond_values[cid][key]) for cid in remaining}
                for key in keys_list
                for value, supporting_groups in group_ids_by_level[key].items()
                if len(supporting_groups) >= 2
            ):
                continue
            for cid in members:
                train.remove(cid)
                bucket.append(cid)
            moved = True
            break
        if not moved:
            raise ValueError(
                "Interpolation split cannot keep train marginal coverage and non-empty val/test "
                "without crossing a condition-tuple group"
            )

    return {"train": train, "val": val, "test": test}


def build_interpolation_overlap_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_groups: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """
    Build interpolation split with train marginal-level overlap and exact-tuple isolation.

    Complete condition tuples are treated as indivisible groups.  "Overlap" therefore means
    that every held-out marginal condition level is represented in train, not that identical
    input tuples leak across train and test.
    """

    return build_interpolation_overlap_split_with_status(
        case_ids=case_ids,
        cond_values=cond_values,
        keys=keys,
        seed=seed,
        ratios=ratios,
        split_groups=split_groups,
    )["split"]


def build_interpolation_overlap_split_with_status(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_groups: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Build tuple-isolated interpolation split with marginal overlap status.

    Returns:
      {
        "split": {"train": [...], "val": [...], "test": [...]},
        "feasible": bool,
        "reason": str,
      }
    """

    keys_list = [str(k) for k in keys]
    if len(keys_list) == 0:
        split = build_casewise_splits(
            case_ids=case_ids,
            split_groups=split_groups,
            seed=seed,
            ratios=ratios,
        )
        return {"split": split, "feasible": True, "reason": ""}

    try:
        out = build_interpolation_split(
            case_ids=case_ids,
            cond_values=cond_values,
            keys=keys_list,
            seed=seed,
            ratios=ratios,
            mode="marginal",
            split_groups=split_groups,
        )
    except ValueError as exc:
        fallback = build_condition_grouped_splits(
            case_ids,
            cond_values=cond_values,
            keys=keys_list,
            split_groups=split_groups,
            seed=seed,
            ratios=ratios,
        )
        return {"split": fallback, "feasible": False, "reason": str(exc)}

    tuple_by_case = {
        str(cid): tuple(float(cond_values[str(cid)][key]) for key in keys_list)
        for cid in case_ids
    }
    train_tuples = {tuple_by_case[cid] for cid in out["train"]}
    val_tuples = {tuple_by_case[cid] for cid in out["val"]}
    test_tuples = {tuple_by_case[cid] for cid in out["test"]}
    if train_tuples & val_tuples or train_tuples & test_tuples or val_tuples & test_tuples:
        return {"split": out, "feasible": False, "reason": "condition_tuple_group_crossed"}

    train_levels = {
        key: {float(cond_values[cid][key]) for cid in out["train"]}
        for key in keys_list
    }
    missing: list[str] = []
    for split_name in ("val", "test"):
        for key in keys_list:
            held_levels = {float(cond_values[cid][key]) for cid in out[split_name]}
            absent = sorted(held_levels - train_levels[key])
            if absent:
                missing.append(f"{split_name}.{key}={absent}")
    if missing:
        return {"split": out, "feasible": False, "reason": "missing_train_marginal_levels:" + ";".join(missing)}
    return {"split": out, "feasible": True, "reason": ""}


def build_structure_holdout_split(
    case_ids: Sequence[str],
    *,
    structure_groups: Sequence[Hashable],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Build a real structure holdout with each structure group confined to one partition."""

    groups = list(structure_groups)
    if len(groups) != len(case_ids):
        raise ValueError("structure_groups must have the same length as case_ids")
    if any(str(group).strip() == "" for group in groups):
        raise ValueError("structure_groups must not contain empty values")
    if len(set(groups)) < 3:
        raise ValueError("At least 3 distinct structure groups are required for structure holdout")
    return build_casewise_splits(
        case_ids,
        split_groups=[str(group) for group in groups],
        seed=seed,
        ratios=ratios,
    )


def build_group_kfold_splits(
    case_ids: Sequence[str],
    split_groups: Sequence[str],
    n_folds: int = 5,
    seed: int = 0,
    val_ratio_within_remain: float = 0.2,
) -> list[dict[str, list[str]]]:
    """Build grouped K-fold splits where test fold is held-out group-wise."""

    ids = np.array(list(case_ids), dtype=object)
    groups = np.array(list(split_groups), dtype=object)
    if ids.shape[0] == 0:
        raise ValueError("No case IDs provided")
    if groups.shape[0] != ids.shape[0]:
        raise ValueError("split_groups must have the same length as case_ids")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if val_ratio_within_remain <= 0.0 or val_ratio_within_remain >= 1.0:
        raise ValueError("val_ratio_within_remain must be in (0,1)")

    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 3:
        raise ValueError("At least 3 unique groups are required for grouped k-fold split")
    k = int(min(n_folds, unique_groups.shape[0]))
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    folds = [np.asarray(x, dtype=object) for x in np.array_split(shuffled, k)]
    out: list[dict[str, list[str]]] = []

    for fold_idx in range(k):
        test_groups = set(folds[fold_idx].tolist())
        remain_groups = [g for i, f in enumerate(folds) if i != fold_idx for g in f.tolist()]
        if len(remain_groups) < 2:
            raise ValueError("Not enough remaining groups to form train/val split")
        n_val_groups = max(1, int(round(len(remain_groups) * float(val_ratio_within_remain))))
        n_val_groups = min(n_val_groups, len(remain_groups) - 1)
        val_groups = set(remain_groups[:n_val_groups])
        train_groups = set(remain_groups[n_val_groups:])

        train = [str(cid) for cid, gid in zip(ids.tolist(), groups.tolist()) if gid in train_groups]
        val = [str(cid) for cid, gid in zip(ids.tolist(), groups.tolist()) if gid in val_groups]
        test = [str(cid) for cid, gid in zip(ids.tolist(), groups.tolist()) if gid in test_groups]
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError("Grouped k-fold produced an empty split; adjust n_folds or val_ratio_within_remain")
        out.append({"train": train, "val": val, "test": test})
    return out
