#!/usr/bin/env python
"""Build an immutable hard-link view for the ICP v45 comparison.

Legacy structures receive the same 3x3 cross-section vacuum-field quadrature
used by v43.  No COMSOL response field is added to the model inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
COMBINED = ROOT / "data/outputs_icp_stage4_plus_v43_csv_npz_causal_em_pabs_v1"
LEGACY_VACUUM_Q3 = ROOT / "data/outputs_icp_stage4_enriched_360_csv_npz_vacuum_field_q3_v45"
V43 = ROOT / "data/outputs_icp_v43_structure_600_csv_npz_causal_em_pabs_v1"
LEGACY_RAW = ROOT / "data/outputs_icp_stage4_enriched_360"
V43_RAW = ROOT / "data/outputs_icp_v43_structure_600"
DEFAULT_OUT = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
DIM_FIELDS = tuple(
    f"coil_{slot:02d}_{name}"
    for slot in range(1, 7)
    for name in ("active", "r_center", "z_center", "width", "height")
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_verify(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"existing target differs in size: {target}")
        return
    os.link(source, target)


def _link_tree(source: Path, target: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file():
            _link_or_verify(path, target / path.relative_to(source))
            count += 1
    return count


def _coil_vector(case_id: str, source_dataset: str) -> list[float]:
    raw_root = LEGACY_RAW if source_dataset == "legacy_stage4" else V43_RAW
    path = raw_root / "structure/coil_layout" / f"{case_id}__coil_layout.csv"
    rows = _read_csv(path)
    active = [row for row in rows if int(round(float(row["active"]))) == 1]
    active.sort(key=lambda row: (int(float(row.get("order", 0))), int(float(row["coil_index"]))))
    if not 1 <= len(active) <= 6:
        raise ValueError(f"expected 1..6 active coils for {case_id}, got {len(active)}")
    values: list[float] = []
    for slot in range(6):
        if slot < len(active):
            row = active[slot]
            values.extend(
                [
                    1.0,
                    float(row["r_center"]),
                    float(row["z_center"]),
                    float(row["width"]),
                    float(row["height"]),
                ]
            )
        else:
            values.extend([0.0] * 5)
    return values


def _normalize_split(value: str) -> str:
    name = str(value).strip().lower()
    if name == "validation":
        name = "val"
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unsupported source split: {value!r}")
    return name


def _prepare_legacy_q3() -> None:
    if (LEGACY_VACUUM_Q3 / "conversion_summary.json").is_file():
        summary = json.loads((LEGACY_VACUUM_Q3 / "conversion_summary.json").read_text(encoding="utf-8"))
        if int(summary.get("vacuum_field_quadrature_order", 0)) == 3:
            return
        raise RuntimeError(f"existing legacy vacuum dataset is not quadrature order 3: {LEGACY_VACUUM_Q3}")
    from scripts.prepare_icp_stage4_vacuum_field_dataset import prepare_vacuum_field_dataset

    prepare_vacuum_field_dataset(
        src_root=ROOT / "data/outputs_icp_stage4_enriched_360_csv_npz_causal_em_pabs_v1",
        dst_root=LEGACY_VACUUM_Q3,
        coil_layout_root=LEGACY_RAW / "structure/coil_layout",
        length_unit_to_m=1.0e-2,
        quadrature_order=3,
    )


def prepare(out_root: Path) -> dict[str, Any]:
    _prepare_legacy_q3()
    rows = _read_csv(COMBINED / "index.csv")
    out_root.mkdir(parents=True, exist_ok=True)
    # Both source datasets share the same fixed chamber grid.  The combined
    # hard-link view intentionally omitted geometry, so take it from the
    # canonical legacy compact dataset used by the conference runs.
    linked = _link_tree(
        ROOT / "data/outputs_icp_stage4_enriched_360_csv_npz_causal_em_pabs_v1/geometry",
        out_root / "geometry",
    )
    explicit_rows: list[dict[str, Any]] = []
    membership: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    geometry_by_case: dict[str, tuple[float, ...]] = {}

    for index, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        source_dataset = str(row["source_dataset"])
        source_structure_root = LEGACY_VACUUM_Q3 if source_dataset == "legacy_stage4" else V43
        field_source = COMBINED / row["fields_npz"]
        structure_source = source_structure_root / row["structure_npz"]
        _link_or_verify(field_source, out_root / row["fields_npz"])
        _link_or_verify(structure_source, out_root / row["structure_npz"])
        linked += 2

        split = _normalize_split(row["source_split"])
        membership[split].append(case_id)
        vector = _coil_vector(case_id, source_dataset)
        geometry_by_case[case_id] = tuple(vector)
        enriched: dict[str, Any] = dict(row)
        enriched.update(dict(zip(DIM_FIELDS, vector, strict=True)))
        explicit_rows.append(enriched)
        if index % 120 == 0 or index == len(rows):
            print(f"prepared {index}/{len(rows)} combined cases", flush=True)

    original_fields = list(rows[0].keys())
    _write_csv(out_root / "index.csv", rows, original_fields)
    _write_csv(out_root / "index_explicit_dimension.csv", explicit_rows, [*original_fields, *DIM_FIELDS])
    (out_root / "split_membership.json").write_text(
        json.dumps(membership, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(V43 / "design_metadata.csv", out_root / "design_metadata_v43.csv")

    dim_columns = ("llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0")
    collisions: dict[tuple[float, ...], list[str]] = defaultdict(list)
    row_by_case = {str(row["case_id"]): row for row in rows}
    for row in rows:
        key = tuple(float(row[name]) for name in dim_columns)
        collisions[key].append(str(row["case_id"]))
    collision_rows: list[dict[str, Any]] = []
    colliding_cases: set[str] = set()
    for group_index, (key, case_ids) in enumerate(collisions.items(), start=1):
        distinct_geometry = {geometry_by_case[case_id] for case_id in case_ids}
        if len(distinct_geometry) <= 1:
            continue
        colliding_cases.update(case_ids)
        for case_id in case_ids:
            source = row_by_case[case_id]
            collision_rows.append(
                {
                    "collision_group": group_index,
                    "case_id": case_id,
                    "source_split": _normalize_split(source["source_split"]),
                    "source_dataset": source["source_dataset"],
                    "dimension_tuple": "|".join(f"{value:.10g}" for value in key),
                    "distinct_geometries_in_group": len(distinct_geometry),
                }
            )
    if collision_rows:
        _write_csv(out_root / "formal_dimension_collision_cases.csv", collision_rows, list(collision_rows[0]))

    split_group_by_case = {str(row["case_id"]): str(row["split_group"]) for row in rows}
    group_memberships: dict[str, set[str]] = defaultdict(set)
    for split_name, case_ids in membership.items():
        for case_id in case_ids:
            group_memberships[split_group_by_case[case_id]].add(split_name)
    leakage = {key: sorted(value) for key, value in group_memberships.items() if len(value) > 1}
    if leakage:
        raise RuntimeError(f"split-group leakage detected: {leakage}")

    v43_design = _read_csv(V43 / "design_metadata.csv")
    test_kind_counts = Counter(
        row["layout_kind"] for row in v43_design if _normalize_split(row["split"]) == "test"
    )
    summary = {
        "schema": "icp-conference-continuity-v45-dataset-view",
        "case_count": len(rows),
        "split_counts": {key: len(value) for key, value in membership.items()},
        "hard_links_verified_or_created": linked,
        "legacy_vacuum_quadrature_order": 3,
        "v43_vacuum_quadrature_order": 3,
        "formal_dimension_collision_case_count": len(colliding_cases),
        "formal_dimension_collision_row_count": len(collision_rows),
        "test_layout_kind_counts": dict(sorted(test_kind_counts.items())),
        "index_sha256": _sha256(out_root / "index.csv"),
        "explicit_dimension_index_sha256": _sha256(out_root / "index_explicit_dimension.csv"),
        "split_membership_sha256": _sha256(out_root / "split_membership.json"),
    }
    (out_root / "dataset_view_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
