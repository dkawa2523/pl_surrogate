#!/usr/bin/env python
"""Read-only verification for the portable ICP UNO v47 protocol inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INVENTORY = HERE / "SOURCE_INVENTORY.json"
PORTABILITY = HERE / "PORTABILITY_MANIFEST.json"
REQUIRED_DOCS = (
    "README.md",
    "README_JA.md",
    "RUN_CHECKLIST.md",
    "docs/01_MODEL_COMPARISON_CONTRACT.md",
    "docs/02_ENVIRONMENT_AND_TRANSFER.md",
    "docs/03_TRAINING_RUNBOOK.md",
    "docs/04_EVALUATION_AND_OPTIMIZATION.md",
    "docs/05_CONFERENCE_FIGURES.md",
    "docs/06_RECOVERY_AUDIT_ACCEPTANCE.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["case_id"]) for row in csv.DictReader(handle)}


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-formal-references",
        action="store_true",
        help="fail if the adopted final v45 reference checkpoints were not transferred",
    )
    args = parser.parse_args()

    inventory = dict(_json(INVENTORY))
    protocol = dict(_json(PORTABILITY))
    failures: list[str] = []
    warnings: list[str] = []
    checked: list[dict[str, Any]] = []

    for relative in REQUIRED_DOCS:
        path = HERE / relative
        if not path.is_file():
            failures.append(f"missing protocol document: {relative}")

    for item in inventory["files"]:
        relative = str(item["path"])
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")
            continue
        actual_size = int(path.stat().st_size)
        actual_hash = _sha256(path)
        expected_size = int(item["bytes"])
        expected_hash = str(item["sha256"]).upper()
        if actual_size != expected_size:
            failures.append(
                f"size mismatch: {relative}: expected={expected_size}, actual={actual_size}"
            )
        if actual_hash != expected_hash:
            failures.append(
                f"sha256 mismatch: {relative}: expected={expected_hash}, actual={actual_hash}"
            )
        checked.append({"path": relative, "bytes": actual_size, "sha256": actual_hash})

    for item in inventory.get("formal_reference_checkpoints", []):
        relative = str(item["path"])
        path = ROOT / relative
        if not path.is_file():
            message = f"formal reference checkpoint not present: {relative}"
            (failures if args.require_formal_references else warnings).append(message)
            continue
        actual_hash = _sha256(path)
        expected_hash = str(item["sha256"]).upper()
        if actual_hash != expected_hash:
            failures.append(
                f"formal checkpoint changed: {relative}: expected={expected_hash}, actual={actual_hash}"
            )

    dataset_root = ROOT / str(protocol["dataset"]["root"])
    index_ids = _csv_ids(dataset_root / "index.csv")
    if len(index_ids) != int(protocol["dataset"]["case_count"]):
        failures.append(
            f"dataset case count mismatch: expected={protocol['dataset']['case_count']}, actual={len(index_ids)}"
        )
    membership = dict(_json(dataset_root / "split_membership.json"))
    expected_split = dict(protocol["dataset"]["split"])
    split_keys = {"train": "train", "validation": "val", "test": "test"}
    split_sets: dict[str, set[str]] = {}
    for label, key in split_keys.items():
        values = {str(value) for value in membership[key]}
        split_sets[label] = values
        if len(values) != int(expected_split[label]):
            failures.append(
                f"split count mismatch for {label}: expected={expected_split[label]}, actual={len(values)}"
            )
    if split_sets["train"] & split_sets["validation"]:
        failures.append("train and validation memberships overlap")
    if split_sets["train"] & split_sets["test"]:
        failures.append("train and test memberships overlap")
    if split_sets["validation"] & split_sets["test"]:
        failures.append("validation and test memberships overlap")
    if set().union(*split_sets.values()) != index_ids:
        failures.append("split membership does not exactly cover index.csv case ids")

    isolation = dict(protocol["isolation"])
    run_root = ROOT / str(isolation["run_root"])
    report_root = ROOT / str(isolation["report_root"])
    if not _inside(run_root, ROOT / "runs"):
        failures.append(f"run root escapes workspace runs directory: {run_root}")
    if not _inside(report_root, ROOT / "reports/icp_conference_materials"):
        failures.append(f"report root escapes conference report directory: {report_root}")
    for forbidden in isolation["forbidden_write_roots"]:
        forbidden_path = (ROOT / str(forbidden)).resolve()
        if run_root.resolve() == forbidden_path or report_root.resolve() == forbidden_path:
            failures.append(f"new output root equals forbidden root: {forbidden}")

    result = {
        "study_id": protocol["study_id"],
        "status": "pass" if not failures else "fail",
        "root": str(ROOT),
        "required_file_count_checked": len(checked),
        "dataset_case_count": len(index_ids),
        "split_counts": {name: len(values) for name, values in split_sets.items()},
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
