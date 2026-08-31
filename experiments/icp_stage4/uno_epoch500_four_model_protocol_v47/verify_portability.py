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
    "DATASET_ACQUISITION_JA.md",
    "REMOTE_AGENT_500EPOCH_PROMPT_JA.md",
    "REMOTE_TRAINING_START_HERE_JA.md",
    "RUN_CHECKLIST.md",
    "docs/01_MODEL_COMPARISON_CONTRACT.md",
    "docs/02_ENVIRONMENT_AND_TRANSFER.md",
    "docs/03_TRAINING_RUNBOOK.md",
    "docs/04_EVALUATION_AND_OPTIMIZATION.md",
    "docs/05_CONFERENCE_FIGURES.md",
    "docs/06_RECOVERY_AUDIT_ACCEPTANCE.md",
)


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}


def _payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        # Git is allowed to materialize text as LF or CRLF.  Inventory hashes
        # describe canonical LF content so a Windows checkout verifies exactly
        # like a Linux checkout without weakening binary/weight verification.
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def _sha256_payload(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest().upper()


def _sha256(path: Path) -> str:
    return _sha256_payload(_payload(path))


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
        payload = _payload(path)
        actual_size = len(payload)
        actual_hash = _sha256_payload(payload)
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
        meta_relative = str(item.get("meta_path", "")).strip()
        if meta_relative:
            meta_path = ROOT / meta_relative
            if not meta_path.is_file():
                message = f"formal reference metadata not present: {meta_relative}"
                (failures if args.require_formal_references else warnings).append(message)
            else:
                actual_meta_hash = _sha256(meta_path)
                expected_meta_hash = str(item["meta_sha256"]).upper()
                if actual_meta_hash != expected_meta_hash:
                    failures.append(
                        "formal checkpoint metadata changed: "
                        f"{meta_relative}: expected={expected_meta_hash}, actual={actual_meta_hash}"
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
