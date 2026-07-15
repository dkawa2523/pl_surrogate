"""Reliability checks for ICP Stage4 evaluation protocols.

The checks in this module deliberately use the resolved/configured evaluation split,
not the spelling of the primary metric.  A metric such as
``test_r2_group_default_plasma`` does not identify the split that produced it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


_SPLIT_FILES = {
    "random": "split_random_v1.json",
    "interp": "split_interp_v1.json",
    "extrap": "split_extrap_v1.json",
    "structure_holdout": "split_structure_holdout_v1.json",
}


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    benchmark = payload.get("benchmark")
    return dict(benchmark) if isinstance(benchmark, dict) else dict(payload)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return dict(payload)


def _split_count(output_dir: Path, split_name: str) -> int | None:
    filename = _SPLIT_FILES[split_name]
    path = output_dir / "preprocessing" / "split" / filename
    if not path.exists():
        return None
    try:
        split = _load_json(path)
    except Exception:
        return None
    test_ids = split.get("test")
    return len(test_ids) if isinstance(test_ids, list) else None


def _resolve_dataset_index(dataset_cfg: dict[str, Any], *, config_path: Path | None) -> Path | None:
    root_raw = str(dataset_cfg.get("root", "")).strip()
    index_raw = str(dataset_cfg.get("index_csv", "")).strip()
    if not root_raw or not index_raw:
        return None
    root = Path(root_raw)
    candidates: list[Path] = []
    if root.is_absolute():
        candidates.append(root / index_raw)
    else:
        candidates.append(Path.cwd() / root / index_raw)
        if config_path is not None:
            candidates.append(config_path.resolve().parent / root / index_raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_case_groups(
    benchmark_cfg: dict[str, Any],
    *,
    config_path: Path | None,
) -> tuple[dict[str, str], str]:
    dataset_cfg = dict(benchmark_cfg.get("dataset", {}) or {})
    case_column = str(dataset_cfg.get("case_id_column", "case_id"))
    group_column = str(dataset_cfg.get("split_group_column", "")).strip()
    if not group_column:
        return {}, "split_group_column_missing"
    index_path = _resolve_dataset_index(dataset_cfg, config_path=config_path)
    if index_path is None:
        return {}, "dataset_index_unavailable"
    try:
        with index_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}, "dataset_index_unreadable"
    groups: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get(case_column, "")).strip()
        group_id = str(row.get(group_column, "")).strip()
        if case_id and group_id:
            groups[case_id] = group_id
    return groups, "" if groups else "split_group_values_missing"


def _group_overlap_issues(
    split: dict[str, Any],
    *,
    groups_by_case: dict[str, str],
) -> list[str]:
    buckets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        ids = split.get(name, [])
        buckets[name] = {
            groups_by_case[str(case_id)]
            for case_id in ids
            if str(case_id) in groups_by_case
        }
    issues: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = buckets[left] & buckets[right]
        if overlap:
            issues.append(f"structure_group_overlap_{left}_{right}:{len(overlap)}")
    return issues


def assess_protocol_reliability(
    output_dir: str | Path,
    *,
    leaderboard_row: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    default_min_test_cases: int = 3,
) -> dict[str, str]:
    """Assess the protocol that actually produced a primary metric.

    For legacy outputs the resolved config is authoritative.  For new outputs an
    explicit ``primary_split_effective`` leaderboard field takes precedence, while
    disagreement with the resolved config is reported as an error.
    """

    output = Path(output_dir)
    row = dict(leaderboard_row or {})
    explicit_config_path = Path(config_path) if config_path is not None else None
    resolved_path = output / "resolved_config.yaml"
    resolved_cfg = _load_config(resolved_path)
    source_cfg = resolved_cfg or (_load_config(explicit_config_path) if explicit_config_path else {})
    source_path = resolved_path if resolved_cfg else explicit_config_path

    eval_protocol = dict(source_cfg.get("eval_protocol", {}) or {})
    mode = str(
        row.get("eval_protocol_mode_effective")
        or eval_protocol.get("mode")
        or ""
    ).strip().lower()
    configured_split = str(eval_protocol.get("primary_split", "")).strip().lower()
    row_split = str(row.get("primary_split_effective", "")).strip().lower()
    if mode == "single":
        effective_split = "random"
    else:
        effective_split = row_split or configured_split

    issues: list[str] = []
    if not source_cfg:
        issues.append("resolved_or_source_config_missing")
    if mode not in {"single", "primary_axis", "dual_axis"}:
        issues.append(f"eval_protocol_mode_unknown:{mode or 'missing'}")
    if row_split and configured_split and mode != "single" and row_split != configured_split:
        issues.append(f"primary_split_mismatch:row={row_split},config={configured_split}")
    if effective_split not in _SPLIT_FILES:
        issues.append(f"primary_split_unknown:{effective_split or 'missing'}")

    counts = {name: _split_count(output, name) for name in _SPLIT_FILES}
    min_test_cases_raw = eval_protocol.get("min_primary_test_cases", default_min_test_cases)
    try:
        min_test_cases = max(int(min_test_cases_raw), 1)
    except (TypeError, ValueError):
        min_test_cases = max(int(default_min_test_cases), 1)
        issues.append(f"min_primary_test_cases_invalid:{min_test_cases_raw}")
    min_test_groups_raw = eval_protocol.get("min_primary_test_groups", 3)
    try:
        min_test_groups = max(int(min_test_groups_raw), 1)
    except (TypeError, ValueError):
        min_test_groups = 3
        issues.append(f"min_primary_test_groups_invalid:{min_test_groups_raw}")

    split: dict[str, Any] = {}
    if effective_split in _SPLIT_FILES:
        split_path = output / "preprocessing" / "split" / _SPLIT_FILES[effective_split]
        if not split_path.exists():
            issues.append(f"primary_split_artifact_missing:{effective_split}")
        else:
            try:
                split = _load_json(split_path)
            except Exception:
                issues.append(f"primary_split_artifact_invalid:{effective_split}")

    if split:
        buckets: dict[str, list[str]] = {}
        for name in ("train", "val", "test"):
            values = split.get(name)
            if not isinstance(values, list):
                issues.append(f"primary_split_bucket_invalid:{name}")
                buckets[name] = []
            else:
                buckets[name] = [str(value) for value in values]
        test_cases = len(buckets["test"])
        if test_cases < min_test_cases:
            issues.append(f"{effective_split}_test_too_small:{test_cases}<{min_test_cases}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(buckets[left]) & set(buckets[right])
            if overlap:
                issues.append(f"case_id_overlap_{left}_{right}:{len(overlap)}")

        primary_test_groups: int | None = None
        if effective_split == "structure_holdout":
            groups_by_case, group_error = _load_case_groups(source_cfg, config_path=source_path)
            if group_error:
                issues.append(group_error)
            else:
                missing_ids = set().union(*(set(values) for values in buckets.values())) - set(groups_by_case)
                if missing_ids:
                    issues.append(f"split_group_missing_for_cases:{len(missing_ids)}")
                issues.extend(_group_overlap_issues(split, groups_by_case=groups_by_case))
                primary_test_groups = len(
                    {groups_by_case[case_id] for case_id in buckets["test"] if case_id in groups_by_case}
                )
                if primary_test_groups < min_test_groups:
                    issues.append(
                        f"structure_holdout_test_groups_too_small:"
                        f"{primary_test_groups}<{min_test_groups}"
                    )
    else:
        primary_test_groups = None

    scaler_fit_split = str(
        dict(dict(source_cfg.get("preprocessing", {}) or {}).get("scalers", {}) or {}).get("fit_split", "")
    ).strip().lower()
    if scaler_fit_split and effective_split and scaler_fit_split != effective_split:
        issues.append(f"scaler_fit_split_mismatch:{scaler_fit_split}!={effective_split}")

    reliable = not issues
    return {
        "eval_protocol_reliable": "true" if reliable else "false",
        "eval_protocol_issue": "|".join(issues),
        "primary_metric_protocol_reliable": "true" if reliable else "false",
        "eval_protocol_mode_effective": mode,
        "primary_split_effective": effective_split,
        "min_primary_test_cases": str(min_test_cases),
        "min_primary_test_groups": str(min_test_groups),
        "primary_test_groups": "" if primary_test_groups is None else str(primary_test_groups),
        "interp_test_cases": "" if counts["interp"] is None else str(counts["interp"]),
        "extrap_test_cases": "" if counts["extrap"] is None else str(counts["extrap"]),
        "structure_holdout_test_cases": ""
        if counts["structure_holdout"] is None
        else str(counts["structure_holdout"]),
        "scaler_fit_split": scaler_fit_split,
    }


def row_is_reliably_selectable(row: dict[str, Any]) -> bool:
    """Return true only for explicit, positive reliability fields."""

    return all(
        str(row.get(key, "")).strip().lower() == "true"
        for key in ("primary_metric_reliable", "primary_metric_protocol_reliable")
    )


def read_validation_selection(
    output_dir: str | Path,
    *,
    model_id: str,
    primary_split: str,
) -> dict[str, str]:
    """Read the validation score used to select the saved checkpoint."""

    output = Path(output_dir)
    model_root = output / "models" / str(model_id)
    candidates = [
        model_root / "eval_protocol" / str(primary_split) / "train" / "scalars" / "metrics.csv",
        model_root / "train" / "scalars" / "metrics.csv",
    ]
    metrics_path = next((path for path in candidates if path.exists()), None)
    if metrics_path is None:
        return {
            "validation_selection_reliable": "false",
            "validation_selection_issue": "validation_metrics_missing",
            "validation_selection_score": "",
            "validation_selected_epoch": "",
            "validation_metrics": "",
        }
    try:
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        rows = []
    scored: list[tuple[float, dict[str, str]]] = []
    selected: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        try:
            score = float(row.get("val_balance_score", "nan"))
        except (TypeError, ValueError):
            continue
        if score != score or score in {float("inf"), float("-inf")}:
            continue
        item = (score, dict(row))
        scored.append(item)
        try:
            selected_flag = float(row.get("selected_epoch_flag", 0.0)) > 0.5
        except (TypeError, ValueError):
            selected_flag = False
        if selected_flag:
            selected.append(item)
    pool = selected or scored
    if not pool:
        return {
            "validation_selection_reliable": "false",
            "validation_selection_issue": "validation_score_missing_or_nonfinite",
            "validation_selection_score": "",
            "validation_selected_epoch": "",
            "validation_metrics": str(metrics_path),
        }
    score, row = max(pool, key=lambda item: item[0])
    try:
        selection_valid = float(row.get("selection_valid_flag", 1.0)) > 0.5
    except (TypeError, ValueError):
        selection_valid = False
    return {
        "validation_selection_reliable": "true" if selection_valid else "false",
        "validation_selection_issue": "" if selection_valid else "selection_valid_flag_false",
        "validation_selection_score": f"{score:.12g}",
        "validation_selected_epoch": str(row.get("epoch", "")),
        "validation_selection_score_ne": str(row.get("selection_score_ne", "")),
        "validation_selection_score_ni": str(row.get("selection_score_ni", "")),
        "validation_selection_score_Te": str(row.get("selection_score_Te", "")),
        "validation_selection_score_phi": str(row.get("selection_score_phi", "")),
        "validation_metrics": str(metrics_path),
    }


def validation_row_is_selectable(row: dict[str, Any]) -> bool:
    """Require a reliable protocol and an explicit validation checkpoint score."""

    return (
        str(row.get("primary_metric_protocol_reliable", "")).strip().lower() == "true"
        and str(row.get("validation_selection_reliable", "")).strip().lower() == "true"
    )


def enforce_structure_study_protocol(config: dict[str, Any]) -> None:
    """Fail before training if a structural study is not group-held-out."""

    benchmark = dict(config.get("benchmark", config) or {})
    runtime = dict(benchmark.get("runtime", {}) or {})
    if str(runtime.get("input_mode", "")).strip().lower() != "table_plus_structure":
        return
    eval_protocol = dict(benchmark.get("eval_protocol", {}) or {})
    primary_split = str(eval_protocol.get("primary_split", "")).strip().lower()
    mode = str(eval_protocol.get("mode", "")).strip().lower()
    if primary_split != "structure_holdout" or mode != "primary_axis":
        raise ValueError(
            "ICP structural study requires eval_protocol.mode=primary_axis and "
            "eval_protocol.primary_split=structure_holdout; use a separate process-interpolation "
            "study if train/test structure overlap is intentional"
        )
    scalers = dict(dict(benchmark.get("preprocessing", {}) or {}).get("scalers", {}) or {})
    fit_split = str(scalers.get("fit_split", "")).strip().lower()
    if fit_split != primary_split:
        raise ValueError(
            "ICP structural study requires preprocessing.scalers.fit_split to match "
            f"the primary split ({primary_split!r}), got {fit_split!r}"
        )


__all__ = [
    "assess_protocol_reliability",
    "enforce_structure_study_protocol",
    "read_validation_selection",
    "row_is_reliably_selectable",
    "validation_row_is_selectable",
]
