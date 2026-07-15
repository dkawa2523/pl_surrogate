#!/usr/bin/env python3
"""Prepare an order-invariant ICP part-lite dataset from an existing Core4 dataset.

This avoids rereading the large raw COMSOL field exports. Field NPZ files are
hard-linked when possible; structure NPZ files are rewritten with a fixed-slot
``part_mask_stack`` whose inactive slots are zero masks. Order-invariant
``part_lite_v1`` summaries ignore slot ordering, while descriptor models retain
a stable feature dimension.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"dataset index has no rows: {path}")
    required = {"case_id", "structure_npz", "nncoil"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"dataset index is missing columns {missing}: {path}")
    return rows


def _active_part_masks(
    layout_path: Path,
    *,
    r_coords: np.ndarray,
    z_coords: np.ndarray,
    expected_count: int,
    slot_count: int = 6,
) -> tuple[np.ndarray, list[str]]:
    with layout_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"coil_index", "active", "r_min", "r_max", "z_min", "z_max"}
    if not rows:
        raise ValueError(f"coil layout has no rows: {layout_path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"coil layout is missing columns {missing}: {layout_path}")

    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        raw_index = float(row["coil_index"])
        if not np.isfinite(raw_index) or not raw_index.is_integer():
            raise ValueError(f"coil_index must be a finite integer, got {row['coil_index']!r}: {layout_path}")
        idx = int(raw_index)
        if idx <= 0:
            raise ValueError(f"coil_index must be positive, got {idx}: {layout_path}")
        if idx > int(slot_count):
            raise ValueError(
                f"coil_index={idx} exceeds fixed descriptor slot_count={slot_count}: {layout_path}"
            )
        if idx in by_index:
            raise ValueError(f"duplicate coil_index={idx}: {layout_path}")
        by_index[idx] = row

    r = np.asarray(r_coords, dtype=np.float32).reshape(1, -1)
    z = np.asarray(z_coords, dtype=np.float32).reshape(-1, 1)
    masks = [np.zeros((z.shape[0], r.shape[1]), dtype=np.uint8) for _ in range(int(slot_count))]
    active_part_ids: list[str] = []
    for idx, row in sorted(by_index.items()):
        raw_active = float(row.get("active", 0) or 0)
        if not np.isfinite(raw_active) or not raw_active.is_integer() or int(raw_active) not in {0, 1}:
            raise ValueError(f"active must be 0 or 1, got {row.get('active')!r}: {layout_path}")
        if int(raw_active) != 1:
            continue
        r_min = float(row["r_min"])
        r_max = float(row["r_max"])
        z_min = float(row["z_min"])
        z_max = float(row["z_max"])
        if not all(np.isfinite(value) for value in (r_min, r_max, z_min, z_max)):
            raise ValueError(f"coil bounds must be finite for coil_index={idx}: {layout_path}")
        if not (r_min <= r_max and z_min <= z_max):
            raise ValueError(f"invalid coil bounds for coil_index={idx}: {layout_path}")
        mask = ((r >= r_min) & (r <= r_max) & (z >= z_min) & (z <= z_max)).astype(np.uint8)
        if not np.any(mask):
            raise ValueError(f"active coil_index={idx} maps to an empty grid mask: {layout_path}")
        if any(np.any((existing > 0) & (mask > 0)) for existing in masks):
            raise ValueError(f"active coil masks overlap at coil_index={idx}: {layout_path}")
        masks[idx - 1] = mask
        active_part_ids.append(f"coil_{idx:02d}")

    if len(active_part_ids) != int(expected_count):
        raise ValueError(
            f"active coil count mismatch: layout={len(active_part_ids)} "
            f"index.nncoil={expected_count}: {layout_path}"
        )
    if not active_part_ids:
        raise ValueError(f"coil layout has no active parts: {layout_path}")
    return np.stack(masks, axis=0).astype(np.uint8), active_part_ids


def _clone_non_structure_files(src_root: Path, dst_root: Path) -> tuple[int, int]:
    linked = 0
    copied = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        if rel.parts and rel.parts[0] == "structure_features":
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts and rel.parts[0] == "fields":
            try:
                os.link(src, dst)
                linked += 1
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        copied += 1
    return linked, copied


def prepare_part_lite_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    coil_layout_root: Path,
) -> dict[str, Any]:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    coil_layout_root = coil_layout_root.resolve()
    if src_root == dst_root:
        raise ValueError("source and destination dataset roots must differ")
    if not (src_root / "index.csv").exists():
        raise FileNotFoundError(src_root / "index.csv")
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(f"destination must be absent or empty: {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    linked, copied = _clone_non_structure_files(src_root, dst_root)
    rows = _read_index(src_root / "index.csv")
    out_structure = dst_root / "structure_features"
    out_structure.mkdir(parents=True, exist_ok=True)
    part_counts: dict[int, int] = {}

    for row_idx, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        rel = Path(str(row["structure_npz"]))
        src_npz = src_root / rel
        if not src_npz.exists():
            raise FileNotFoundError(src_npz)
        with np.load(src_npz, allow_pickle=False) as data:
            payload = {str(key): np.asarray(data[key]) for key in data.files}
        if "r_coords" not in payload or "z_coords" not in payload:
            raise ValueError(f"structure NPZ lacks r_coords/z_coords: {src_npz}")
        expected_count_raw = float(row["nncoil"])
        if not np.isfinite(expected_count_raw) or not expected_count_raw.is_integer():
            raise ValueError(f"index.nncoil must be a finite integer: case={case_id}, value={row['nncoil']!r}")
        expected_count = int(expected_count_raw)
        mask_stack, active_part_ids = _active_part_masks(
            coil_layout_root / f"{case_id}__coil_layout.csv",
            r_coords=payload["r_coords"],
            z_coords=payload["z_coords"],
            expected_count=expected_count,
        )
        payload["part_mask_stack"] = mask_stack
        payload["part_ids"] = np.asarray([f"coil_{idx:02d}" for idx in range(1, 7)], dtype=np.str_)
        payload["active_part_ids"] = np.asarray(active_part_ids, dtype=np.str_)
        dst_npz = dst_root / rel
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst_npz, **payload)
        part_counts[expected_count] = part_counts.get(expected_count, 0) + 1
        if row_idx % 60 == 0 or row_idx == len(rows):
            print(f"prepared {row_idx}/{len(rows)} structure packs", flush=True)

    summary_path = dst_root / "conversion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "dataset_root": str(dst_root),
            "part_mask_stack": True,
            "part_mask_stack_cases": int(len(rows)),
            "part_mask_stack_mode": "fixed_6_slots_with_inactive_zero_masks",
            "part_lite_v1_ready": True,
            "part_count_distribution": {str(key): int(value) for key, value in sorted(part_counts.items())},
            "prepared_from_dataset": str(src_root),
            "field_files_hardlinked": int(linked),
            "nonfield_files_copied": int(copied),
            "preparation_seconds": float(time.perf_counter() - started),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root",
        default="data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1",
    )
    parser.add_argument(
        "--dst-root",
        default="data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2",
    )
    parser.add_argument(
        "--coil-layout-root",
        default="data/outputs_icp_stage4_enriched_360/structure/coil_layout",
    )
    args = parser.parse_args()
    summary = prepare_part_lite_dataset(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        coil_layout_root=Path(args.coil_layout_root),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
