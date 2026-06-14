"""Split utilities with casewise leakage prevention."""

from __future__ import annotations

from typing import Sequence

import numpy as np


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


def build_pressure_extrap_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    key: str,
    holdout_ratio: float = 0.2,
    val_ratio_within_remain: float = 0.2,
) -> dict[str, list[str]]:
    """Build deterministic extrapolation split by holding out high-end cases for a condition key."""

    ids = list(case_ids)
    if len(ids) < 3:
        raise ValueError("At least 3 case IDs are required")
    if key == "":
        raise ValueError("split key must not be empty")
    if holdout_ratio <= 0.0 or holdout_ratio >= 1.0:
        raise ValueError("holdout_ratio must be in (0,1)")
    if val_ratio_within_remain <= 0.0 or val_ratio_within_remain >= 1.0:
        raise ValueError("val_ratio_within_remain must be in (0,1)")

    ranked = sorted(ids, key=lambda cid: float(cond_values[cid][key]))
    n_test = max(1, int(round(len(ids) * holdout_ratio)))
    test = ranked[-n_test:]
    remain = ranked[:-n_test]
    n_val = max(1, int(round(len(remain) * val_ratio_within_remain)))
    val = remain[-n_val:]
    train = remain[:-n_val]
    if len(train) == 0:
        train = remain[:1]
        val = remain[1:]
    return {"train": train, "val": val, "test": test}


def build_extrapolation_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    key: str,
    holdout_ratio: float = 0.2,
    val_ratio_within_remain: float = 0.2,
) -> dict[str, list[str]]:
    """Unified extrapolation split entrypoint (wrapper over pressure-style holdout)."""

    return build_pressure_extrap_split(
        case_ids=case_ids,
        cond_values=cond_values,
        key=key,
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
        )
    if mode_norm != "marginal":
        raise ValueError("mode must be one of: marginal, overlap")

    split = build_casewise_splits(case_ids=case_ids, seed=seed, ratios=ratios)
    keys_list = [str(k) for k in keys]
    if len(keys_list) == 0:
        return split

    train = list(split["train"])
    val = list(split["val"])
    test = list(split["test"])
    all_ids = list(case_ids)
    all_vals = {
        k: sorted({float(cond_values[cid][k]) for cid in all_ids if cid in cond_values})
        for k in keys_list
    }

    def _move_to_train(src: list[str], cid: str) -> None:
        if cid in src:
            src.remove(cid)
        if cid not in train:
            train.append(cid)

    # Ensure train contains at least one sample for each marginal value.
    for k in keys_list:
        for v in all_vals[k]:
            if any(np.isclose(float(cond_values[c][k]), v) for c in train):
                continue
            cand = [c for c in (val + test) if np.isclose(float(cond_values[c][k]), v)]
            if not cand:
                continue
            pick = cand[0]
            if pick in val:
                _move_to_train(val, pick)
            else:
                _move_to_train(test, pick)

    # Keep val/test non-empty for downstream contracts.
    rng = np.random.default_rng(seed + 101)
    for bucket in (val, test):
        if len(bucket) == 0 and len(train) > 1:
            idx = int(rng.integers(0, len(train)))
            bucket.append(train.pop(idx))

    return {"train": train, "val": val, "test": test}


def build_interpolation_overlap_split(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    """
    Build interpolation split with strict tuple overlap where possible.

    If no duplicate condition tuples exist, this falls back to the marginal interpolation
    split because strict overlap is not feasible with unique tuples.
    """

    return build_interpolation_overlap_split_with_status(
        case_ids=case_ids,
        cond_values=cond_values,
        keys=keys,
        seed=seed,
        ratios=ratios,
    )["split"]


def build_interpolation_overlap_split_with_status(
    case_ids: Sequence[str],
    cond_values: dict[str, dict[str, float]],
    keys: Sequence[str],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, object]:
    """
    Build interpolation split with strict tuple overlap and return feasibility status.

    Returns:
      {
        "split": {"train": [...], "val": [...], "test": [...]},
        "feasible": bool,
        "reason": str,
      }
    """

    split = build_casewise_splits(case_ids=case_ids, seed=seed, ratios=ratios)
    keys_list = [str(k) for k in keys]
    if len(keys_list) == 0:
        return {"split": split, "feasible": True, "reason": ""}

    train = list(split["train"])
    val = list(split["val"])
    test = list(split["test"])
    cond_by_id = {str(cid): cond_values[str(cid)] for cid in case_ids}
    tuple_to_cases: dict[tuple[float, ...], list[str]] = {}
    for cid in case_ids:
        key = tuple(float(cond_by_id[str(cid)][k]) for k in keys_list)
        tuple_to_cases.setdefault(key, []).append(str(cid))
    candidate_keys = [k for k, rows in tuple_to_cases.items() if len(rows) >= 2]
    if len(candidate_keys) == 0:
        return {"split": split, "feasible": False, "reason": "no_duplicate_condition_tuples"}

    def _move(src: list[str], dst: list[str], cid: str) -> None:
        if cid in src:
            src.remove(cid)
        if cid not in dst:
            dst.append(cid)

    # Ensure at least one tuple appears in both train and test.
    seeded = np.random.default_rng(seed + 121)
    seeded.shuffle(candidate_keys)
    for key in candidate_keys:
        rows = tuple_to_cases[key]
        train_rows = [cid for cid in rows if cid in train]
        test_rows = [cid for cid in rows if cid in test]
        val_rows = [cid for cid in rows if cid in val]

        if train_rows and test_rows:
            break

        if not train_rows:
            if val_rows:
                _move(val, train, val_rows[0])
                train_rows = [val_rows[0]]
                val_rows = [x for x in val_rows if x != train_rows[0]]
            elif test_rows and len(test) > 1:
                _move(test, train, test_rows[0])
                train_rows = [test_rows[0]]
                test_rows = test_rows[1:]

        if not test_rows:
            if val_rows:
                _move(val, test, val_rows[0])
                test_rows = [val_rows[0]]
            elif train_rows and len(train) > 1:
                _move(train, test, train_rows[0])
                test_rows = [train_rows[0]]

        if train_rows and test_rows:
            break

    # Keep val/test non-empty for downstream contracts.
    for bucket in (val, test):
        if len(bucket) == 0 and len(train) > 1:
            idx = int(seeded.integers(0, len(train)))
            bucket.append(train.pop(idx))

    out = {"train": train, "val": val, "test": test}
    tr_tuples = {tuple(float(cond_values[c][k]) for k in keys_list) for c in out["train"]}
    te_tuples = {tuple(float(cond_values[c][k]) for k in keys_list) for c in out["test"]}
    if len(tr_tuples & te_tuples) == 0:
        return {"split": out, "feasible": False, "reason": "tuple_overlap_not_achievable_with_split_constraints"}
    return {"split": out, "feasible": True, "reason": ""}


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
