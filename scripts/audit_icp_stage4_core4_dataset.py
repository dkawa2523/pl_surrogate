from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


TARGETS = ("ne", "ni", "Te", "phi")
STRUCTURE_AUDIT_COLUMNS = ("llcoil", "rrc", "nncoil", "rrce", "zzc")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _source_split_summary(source_root: Path) -> dict[str, Any]:
    manifest = source_root / "learning_manifest.csv"
    if not manifest.exists():
        return {}
    rows = _read_csv(manifest)
    nncoil_by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row.get("split", "")).strip()
        nncoil = str(int(float(row.get("nncoil", "0")))) if row.get("nncoil") else ""
        if split and nncoil:
            split_counts = nncoil_by_split.setdefault(split, {})
            split_counts[nncoil] = split_counts.get(nncoil, 0) + 1
    return {
        "source_manifest": str(manifest),
        "source_split_counts": _count_by(rows, "split"),
        "nncoil_by_source_split": {k: dict(sorted(v.items())) for k, v in sorted(nncoil_by_split.items())},
    }


def _field_stats(dataset_root: Path, rows: list[dict[str, str]], *, limit: int) -> dict[str, Any]:
    stats = {
        name: {"min": float("inf"), "max": float("-inf"), "finite": 0, "total": 0}
        for name in TARGETS
    }
    selected = rows if limit <= 0 else rows[:limit]
    for row in selected:
        rel = str(row.get("fields_npz", "")).strip()
        if not rel:
            continue
        with np.load(dataset_root / rel, allow_pickle=False) as data:
            for name in TARGETS:
                if name not in data.files:
                    continue
                arr = np.asarray(data[name], dtype=np.float32)
                finite = np.isfinite(arr)
                item = stats[name]
                item["finite"] += int(np.sum(finite))
                item["total"] += int(arr.size)
                if np.any(finite):
                    vals = arr[finite]
                    item["min"] = min(float(item["min"]), float(np.min(vals)))
                    item["max"] = max(float(item["max"]), float(np.max(vals)))
    out: dict[str, Any] = {}
    for name, item in stats.items():
        total = int(item["total"])
        finite = int(item["finite"])
        has_finite = finite > 0
        out[name] = {
            "min": float(item["min"]) if has_finite else None,
            "max": float(item["max"]) if has_finite else None,
            "finite_rate": None if total == 0 else float(finite / max(total, 1)),
            "sampled_cases": int(len(selected)),
        }
    return out


def _structure_summary(dataset_root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    has_structure_col = any(str(row.get("structure_npz", "")).strip() for row in rows)
    if not has_structure_col:
        return {"structure_npz_column": False}
    first_rel = next(str(row.get("structure_npz", "")).strip() for row in rows if str(row.get("structure_npz", "")).strip())
    with np.load(dataset_root / first_rel, allow_pickle=True) as data:
        keys = [str(v) for v in data.files]
        shapes = {key: [int(v) for v in np.asarray(data[key]).shape] for key in keys}
    part_stack_count = 0
    for row in rows:
        rel = str(row.get("structure_npz", "")).strip()
        if not rel:
            continue
        with np.load(dataset_root / rel, allow_pickle=True) as data:
            if "part_mask_stack" in data.files:
                part_stack_count += 1
    return {
        "structure_npz_column": True,
        "first_structure_keys": keys,
        "first_structure_shapes": shapes,
        "part_mask_stack_cases": int(part_stack_count),
        "part_lite_v1_ready": bool(part_stack_count == len(rows)),
        "parts_manifest": str(dataset_root / "geometry" / "parts_manifest.json")
        if (dataset_root / "geometry" / "parts_manifest.json").exists()
        else "",
    }


def audit(dataset_root: Path, *, source_root: Path | None, index_csv: str, field_limit: int) -> dict[str, Any]:
    rows = _read_csv(dataset_root / index_csv)
    groups = sorted({str(row.get("split_group", row.get("base_case_id", ""))).strip() for row in rows})
    rows_per_group = _count_by(rows, "split_group")
    numeric_stats = {
        key: _finite_stats([float(row[key]) for row in rows if row.get(key) not in (None, "")])
        for key in ("pp", "pp0", *STRUCTURE_AUDIT_COLUMNS)
        if any(key in row for row in rows)
    }
    cond_candidates = [key for key in ("pp", "pp0", *STRUCTURE_AUDIT_COLUMNS) if any(key in row for row in rows)]
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "index_csv": index_csv,
        "rows": int(len(rows)),
        "structure_groups": int(len(groups)),
        "rows_per_group": {
            "min": int(min(rows_per_group.values())) if rows_per_group else 0,
            "max": int(max(rows_per_group.values())) if rows_per_group else 0,
        },
        "condition_columns_present": cond_candidates,
        "recommended_scalar_condition_columns": ["pp", "pp0"],
        "structure_audit_columns_present": [key for key in STRUCTURE_AUDIT_COLUMNS if key in cond_candidates],
        "numeric_stats": numeric_stats,
        "field_stats": _field_stats(dataset_root, rows, limit=field_limit),
        "structure": _structure_summary(dataset_root, rows),
    }
    if source_root is not None:
        summary.update(_source_split_summary(source_root))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an ICP Stage4 Core4 csv_npz dataset.")
    parser.add_argument(
        "--dataset-root",
        default="data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1",
    )
    parser.add_argument("--source-root", default="data/outputs_icp_stage4_enriched_360")
    parser.add_argument("--index-csv", default="index.csv")
    parser.add_argument("--field-limit", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    source_root = Path(args.source_root) if str(args.source_root).strip() else None
    summary = audit(
        Path(args.dataset_root),
        source_root=source_root,
        index_csv=str(args.index_csv),
        field_limit=int(args.field_limit),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if str(args.out).strip():
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
