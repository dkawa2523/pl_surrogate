#!/usr/bin/env python3
"""One-off converter: ICP stage4 enriched exports -> csv_npz core4 dataset.

This stays outside the runtime package because the source layout is a local
COMSOL export bundle, not a stable project input contract.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_COND_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0"]
PROCESS_COND_COLUMNS = ["pp", "pp0"]
STRUCTURE_AUDIT_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc"]
TARGET_SOURCES = {
    "ne": ("path_data_electron_density", "density_linear"),
    "ni": ("path_data_ion_density", "density_linear"),
    "Te": ("path_data_electron_temperature", "identity"),
    "phi": ("path_data_electric_potential", "identity"),
}
PART_IDS = [f"coil_{i:02d}" for i in range(1, 7)]
PART_SDF_KEYS = [f"sdf_coil_{i:02d}" for i in range(1, 7)]


def _fieldnames(*, structure_spatial_v1: bool) -> list[str]:
    if structure_spatial_v1:
        return [
            "case_id",
            "base_case_id",
            "split_group",
            "axis",
            *PROCESS_COND_COLUMNS,
            *STRUCTURE_AUDIT_COLUMNS,
            "fields_npz",
            "structure_npz",
        ]
    return ["case_id", "base_case_id", "split_group", "axis", *LEGACY_COND_COLUMNS, "fields_npz"]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"learning manifest has no rows: {path}")
    required = {
        "case_id",
        "case_group_id",
        "operation_id",
        "split",
        "path_masks_domain_masks",
        *LEGACY_COND_COLUMNS,
        *[source for source, _ in TARGET_SOURCES.values()],
    }
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"learning manifest missing columns: {missing}")
    return rows


def _load_domain_mask(path: Path) -> dict[str, Any]:
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8-sig")
    if raw.ndim == 0:
        raise ValueError(f"domain mask has no rows: {path}")
    names = set(raw.dtype.names or ())
    required = {"r", "z", "plasma_mask"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"domain mask missing columns {missing}: {path}")

    r = np.asarray(raw["r"], dtype=np.float64).reshape(-1)
    z = np.asarray(raw["z"], dtype=np.float64).reshape(-1)
    plasma_vec = np.asarray(raw["plasma_mask"], dtype=np.float32).reshape(-1)
    r_unique64 = np.unique(r)
    z_unique64 = np.unique(z)
    r_unique = r_unique64.astype(np.float32)
    z_unique = z_unique64.astype(np.float32)
    h = int(z_unique.shape[0])
    w = int(r_unique.shape[0])
    if h * w != int(r.shape[0]):
        raise ValueError(f"incomplete domain grid in {path}: rows={r.shape[0]} expected={h*w}")

    rid = np.searchsorted(r_unique64, r).astype(np.int64)
    zid = np.searchsorted(z_unique64, z).astype(np.int64)
    def _mask_grid(name: str, default: float = 0.0) -> np.ndarray:
        vec = (
            np.asarray(raw[name], dtype=np.float32).reshape(-1)
            if name in names
            else np.full_like(plasma_vec, float(default), dtype=np.float32)
        )
        grid = np.zeros((h, w), dtype=np.float32)
        grid[zid, rid] = (vec > 0.5).astype(np.float32)
        return grid

    mask = _mask_grid("plasma_mask")
    coil_mask = _mask_grid("coil_mask")
    outside_mask = _mask_grid("outside_mask")
    valid_field_mask = _mask_grid("valid_field_mask", default=1.0)
    return {
        "r": r.astype(np.float32),
        "z": z.astype(np.float32),
        "rid": rid,
        "zid": zid,
        "r_coords": r_unique,
        "z_coords": z_unique,
        "plasma_vec": (plasma_vec > 0.5),
        "mask_plasma": mask,
        "mask_coil": coil_mask,
        "outside_mask": outside_mask,
        "valid_field_mask": valid_field_mask,
        "shape": (h, w),
    }


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask > 0.5, dtype=bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (m & (~up | ~down | ~left | ~right)).astype(np.uint8)


def _outside_boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask > 0.5, dtype=bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (~m & (up | down | left | right)).astype(np.uint8)


def _distance_from_seeds(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask > 0, dtype=bool)
    seeds = np.asarray(seed_mask > 0, dtype=bool) & active
    h, w = active.shape
    if not np.any(seeds):
        return np.zeros((h, w), dtype=np.float32)
    inf = np.float32(h + w + 1)
    dist = np.where(seeds, np.float32(0.0), inf).astype(np.float32)
    for j in range(1, w):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j - 1] + np.float32(1.0))
    for j in range(w - 2, -1, -1):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j + 1] + np.float32(1.0))
    for i in range(1, h):
        dist[i, :] = np.minimum(dist[i, :], dist[i - 1, :] + np.float32(1.0))
    for i in range(h - 2, -1, -1):
        dist[i, :] = np.minimum(dist[i, :], dist[i + 1, :] + np.float32(1.0))
    dist[~active] = 0.0
    dist[dist >= inf] = 0.0
    return dist.astype(np.float32)


def _signed_distance_for_part(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    if int(np.sum(m)) <= 0:
        return np.full(m.shape, float(max(m.shape)), dtype=np.float32)
    inside = _distance_from_seeds(m, _boundary_mask(m))
    outside = _distance_from_seeds((1 - m).astype(np.uint8), _outside_boundary_mask(m))
    return np.where(m > 0, -inside, outside).astype(np.float32)


def _read_coil_layout(path: Path, *, r_coords: np.ndarray, z_coords: np.ndarray) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"coil layout not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_index = {int(float(row["coil_index"])): row for row in rows}
    h = int(np.asarray(z_coords).shape[0])
    w = int(np.asarray(r_coords).shape[0])
    r = np.asarray(r_coords, dtype=np.float32).reshape(1, w)
    z = np.asarray(z_coords, dtype=np.float32).reshape(h, 1)
    masks: list[np.ndarray] = []
    sdfs: dict[str, np.ndarray] = {}
    active_count = 0
    for idx, part_id in enumerate(PART_IDS, start=1):
        del part_id
        row = by_index.get(idx, {})
        active = int(float(row.get("active", 0) or 0)) == 1
        mask = np.zeros((h, w), dtype=np.float32)
        if active:
            active_count += 1
            r_min = float(row["r_min"])
            r_max = float(row["r_max"])
            z_min = float(row["z_min"])
            z_max = float(row["z_max"])
            mask = ((r >= r_min) & (r <= r_max) & (z >= z_min) & (z <= z_max)).astype(np.float32)
        masks.append(mask.astype(np.float32))
        sdfs[f"sdf_coil_{idx:02d}"] = _signed_distance_for_part(mask).astype(np.float32)
    return {
        "part_ids": list(PART_IDS),
        "mask_stack": np.stack(masks, axis=0).astype(np.float32),
        "sdf_maps": sdfs,
        "active_count": int(active_count),
    }


def _load_scalar_field(path: Path) -> np.ndarray:
    vals = np.loadtxt(path, delimiter=",", comments="%", usecols=(2,), dtype=np.float64)
    return np.asarray(vals, dtype=np.float32).reshape(-1)


def _field_to_grid(
    vals: np.ndarray,
    *,
    rid: np.ndarray,
    zid: np.ndarray,
    plasma_vec: np.ndarray,
    shape: tuple[int, int],
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    if int(vals.shape[0]) != int(rid.shape[0]):
        raise ValueError(f"field row count mismatch: got={vals.shape[0]} expected={rid.shape[0]}")
    valid_vec = np.isfinite(vals) & np.asarray(plasma_vec, dtype=bool)
    grid = np.full(shape, float(fill_value), dtype=np.float32)
    grid[zid[valid_vec], rid[valid_vec]] = vals[valid_vec].astype(np.float32)
    valid_grid = np.zeros(shape, dtype=bool)
    valid_grid[zid[valid_vec], rid[valid_vec]] = True
    return grid, valid_grid


def _write_index(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _select_smoke_rows(rows: list[dict[str, Any]], n_groups: int) -> list[dict[str, Any]]:
    if n_groups <= 0:
        return []
    groups = []
    seen: set[str] = set()
    for row in rows:
        group = str(row["split_group"])
        if group not in seen:
            groups.append(group)
            seen.add(group)
        if len(groups) >= n_groups:
            break
    selected = set(groups)
    return [row for row in rows if str(row["split_group"]) in selected]


def convert(
    *,
    src_root: Path,
    dst_root: Path,
    density_floor: float,
    fill_value: float,
    smoke_groups: int,
    validate_masks: bool,
    structure_spatial_v1: bool,
    part_sdf_lite_v1: bool = False,
) -> dict[str, Any]:
    structure_spatial_v1 = bool(structure_spatial_v1 or part_sdf_lite_v1)
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(f"destination already exists and is not empty: {dst_root}")
    fields_dir = dst_root / "fields"
    geom_dir = dst_root / "geometry"
    structure_dir = dst_root / "structure_features"
    fields_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)
    if structure_spatial_v1:
        structure_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = src_root / "learning_manifest.csv"
    manifest_rows = _read_manifest(manifest_path)
    first_mask = _load_domain_mask(src_root / manifest_rows[0]["path_masks_domain_masks"])
    shape = tuple(int(v) for v in first_mask["shape"])
    mask_plasma = np.asarray(first_mask["mask_plasma"], dtype=np.float32)

    if validate_masks:
        for i, row in enumerate(manifest_rows[1:], start=2):
            mask_now = _load_domain_mask(src_root / row["path_masks_domain_masks"])
            same_coords = (
                np.array_equal(mask_now["r_coords"], first_mask["r_coords"])
                and np.array_equal(mask_now["z_coords"], first_mask["z_coords"])
            )
            same_mask = np.array_equal(mask_now["mask_plasma"], mask_plasma)
            if not same_coords or ((not structure_spatial_v1) and not same_mask):
                raise ValueError(
                    "fixed-provider conversion requires identical r/z coordinates"
                    f"{' and plasma_mask' if not structure_spatial_v1 else ''}; "
                    f"first case differs from row={i} case_id={row['case_id']}"
                )

    np.save(geom_dir / "mask_plasma.npy", mask_plasma.astype(np.float32))
    np.save(geom_dir / "eps.npy", np.ones_like(mask_plasma, dtype=np.float32))
    np.save(geom_dir / "wafer_mask.npy", mask_plasma.astype(np.float32))
    np.save(geom_dir / "r_coords.npy", np.asarray(first_mask["r_coords"], dtype=np.float32))
    np.save(geom_dir / "z_coords.npy", np.asarray(first_mask["z_coords"], dtype=np.float32))

    index_rows: list[dict[str, Any]] = []
    target_stats: dict[str, dict[str, float]] = {
        key: {"min": float("inf"), "max": float("-inf"), "invalid_plasma_cells": 0.0}
        for key in TARGET_SOURCES
    }
    source_split_counts: dict[str, int] = {}
    groups: set[str] = set()
    reference_parts: dict[str, Any] | None = None

    for i, row in enumerate(manifest_rows, start=1):
        case_id = str(row["case_id"])
        mask_info = first_mask
        if structure_spatial_v1:
            mask_info = _load_domain_mask(src_root / row["path_masks_domain_masks"])
            if not (
                np.array_equal(mask_info["r_coords"], first_mask["r_coords"])
                and np.array_equal(mask_info["z_coords"], first_mask["z_coords"])
            ):
                raise ValueError(f"case-specific structure requires fixed r/z grid: case={case_id}")
        case_rid = np.asarray(mask_info["rid"], dtype=np.int64)
        case_zid = np.asarray(mask_info["zid"], dtype=np.int64)
        case_plasma_vec = np.asarray(mask_info["plasma_vec"], dtype=bool)
        case_mask_plasma = np.asarray(mask_info["mask_plasma"], dtype=np.float32)
        payload: dict[str, np.ndarray] = {}
        for target_key, (path_col, transform) in TARGET_SOURCES.items():
            vals = _load_scalar_field(src_root / row[path_col])
            grid, valid_grid = _field_to_grid(
                vals,
                rid=case_rid,
                zid=case_zid,
                plasma_vec=case_plasma_vec,
                shape=shape,
                fill_value=fill_value,
            )
            invalid_plasma = (case_mask_plasma > 0.5) & (~valid_grid)
            target_stats[target_key]["invalid_plasma_cells"] += float(np.sum(invalid_plasma))
            if transform == "density_linear":
                active = valid_grid & (case_mask_plasma > 0.5)
                out = np.full(shape, float(density_floor), dtype=np.float32)
                out[active] = np.maximum(grid[active], float(density_floor)).astype(np.float32)
            else:
                out = np.where(valid_grid & (case_mask_plasma > 0.5), grid, float(fill_value)).astype(np.float32)
            if not np.all(np.isfinite(out)):
                raise ValueError(f"converted field contains non-finite values: case={case_id} target={target_key}")
            active_vals = out[case_mask_plasma > 0.5]
            target_stats[target_key]["min"] = min(target_stats[target_key]["min"], float(np.min(active_vals)))
            target_stats[target_key]["max"] = max(target_stats[target_key]["max"], float(np.max(active_vals)))
            payload[target_key] = out

        rel_npz = Path("fields") / f"{case_id}.npz"
        np.savez_compressed(fields_dir / f"{case_id}.npz", **payload)
        rel_structure_npz = ""
        if structure_spatial_v1:
            rel_structure = Path("structure_features") / f"{case_id}.npz"
            structure_payload: dict[str, Any] = {
                "mask_plasma": case_mask_plasma.astype(np.float32),
                "mask_coil": np.asarray(mask_info["mask_coil"], dtype=np.float32),
                "valid_field_mask": np.asarray(mask_info["valid_field_mask"], dtype=np.float32),
                "outside_mask": np.asarray(mask_info["outside_mask"], dtype=np.float32),
                "r_coords": np.asarray(mask_info["r_coords"], dtype=np.float32),
                "z_coords": np.asarray(mask_info["z_coords"], dtype=np.float32),
            }
            if part_sdf_lite_v1:
                coil_layout_path = src_root / "structure" / "coil_layout" / f"{case_id}__coil_layout.csv"
                parts = _read_coil_layout(
                    coil_layout_path,
                    r_coords=np.asarray(mask_info["r_coords"], dtype=np.float32),
                    z_coords=np.asarray(mask_info["z_coords"], dtype=np.float32),
                )
                structure_payload.update(parts["sdf_maps"])
                structure_payload["part_mask_stack"] = np.asarray(parts["mask_stack"], dtype=np.float32)
                if reference_parts is None or int(parts["active_count"]) > int(reference_parts["active_count"]):
                    reference_parts = {
                        "case_id": case_id,
                        "active_count": int(parts["active_count"]),
                        "mask_stack": np.asarray(parts["mask_stack"], dtype=np.float32),
                    }
            np.savez_compressed(structure_dir / f"{case_id}.npz", **structure_payload)
            rel_structure_npz = rel_structure.as_posix()

        group = str(row["case_group_id"])
        split = str(row["split"])
        source_split_counts[split] = source_split_counts.get(split, 0) + 1
        groups.add(group)
        index_row: dict[str, Any] = {
            "case_id": case_id,
            "base_case_id": group,
            "split_group": group,
            "axis": 0.0,
            "fields_npz": rel_npz.as_posix(),
        }
        for col in (PROCESS_COND_COLUMNS if structure_spatial_v1 else LEGACY_COND_COLUMNS):
            index_row[col] = float(row[col])
        if structure_spatial_v1:
            for col in STRUCTURE_AUDIT_COLUMNS:
                index_row[col] = float(row[col])
            index_row["structure_npz"] = rel_structure_npz
        index_rows.append(index_row)
        if i % 25 == 0 or i == len(manifest_rows):
            print(f"converted {i}/{len(manifest_rows)} cases")

    fieldnames = _fieldnames(structure_spatial_v1=structure_spatial_v1)
    _write_index(dst_root / "index.csv", index_rows, fieldnames=fieldnames)
    smoke_rows = _select_smoke_rows(index_rows, smoke_groups)
    if smoke_rows:
        _write_index(dst_root / "index_smoke.csv", smoke_rows, fieldnames=fieldnames)

    if part_sdf_lite_v1:
        if reference_parts is None:
            raise ValueError("part_sdf_lite_v1 requested but no reference parts were built")
        np.savez_compressed(
            geom_dir / "parts_pack.npz",
            part_ids=np.asarray(PART_IDS, dtype=object),
            mask_stack=np.asarray(reference_parts["mask_stack"], dtype=np.float32),
        )
        param_specs: dict[str, dict[str, float]] = {}
        for part_id in PART_IDS:
            param_specs[f"part.{part_id}.tx"] = {"default": 0.0, "min": -0.02, "max": 0.02}
            param_specs[f"part.{part_id}.ty"] = {"default": 0.0, "min": -0.02, "max": 0.02}
            param_specs[f"part.{part_id}.scale_x"] = {"default": 1.0, "min": 0.9, "max": 1.1}
            param_specs[f"part.{part_id}.scale_y"] = {"default": 1.0, "min": 0.9, "max": 1.1}
        with (geom_dir / "parts_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "part_ids": list(PART_IDS),
                    "plasma_mode": "preserve",
                    "reference_case_id": str(reference_parts["case_id"]),
                    "param_specs": param_specs,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    summary = {
        "source_root": str(src_root),
        "dataset_root": str(dst_root),
        "n_cases": int(len(index_rows)),
        "n_groups": int(len(groups)),
        "shape": [int(shape[0]), int(shape[1])],
        "cond_columns": list(PROCESS_COND_COLUMNS if structure_spatial_v1 else LEGACY_COND_COLUMNS),
        "structure_spatial_v1": bool(structure_spatial_v1),
        "part_sdf_lite_v1": bool(part_sdf_lite_v1),
        "structure_audit_columns": list(STRUCTURE_AUDIT_COLUMNS if structure_spatial_v1 else []),
        "structure_npz_column": "structure_npz" if structure_spatial_v1 else "",
        "part_sdf_channels": list(PART_SDF_KEYS if part_sdf_lite_v1 else []),
        "part_mask_stack": bool(part_sdf_lite_v1),
        "part_lite_v1_ready": bool(part_sdf_lite_v1),
        "parts_manifest": "geometry/parts_manifest.json" if part_sdf_lite_v1 else "",
        "parts_pack": "geometry/parts_pack.npz" if part_sdf_lite_v1 else "",
        "targets": {
            "ne": {"source_key": "ne", "source_field": "electron_density", "value_transform": "identity"},
            "ni": {"source_key": "ni", "source_field": "ion_density", "value_transform": "identity"},
            "Te": {"source_key": "Te", "source_field": "electron_temperature", "value_transform": "identity"},
            "phi": {"source_key": "phi", "source_field": "electric_potential", "value_transform": "identity"},
        },
        "source_split_counts": source_split_counts,
        "smoke_index": "index_smoke.csv" if smoke_rows else "",
        "smoke_cases": int(len(smoke_rows)),
        "smoke_groups": int(len({str(r["split_group"]) for r in smoke_rows})),
        "mask_active_cells": int(np.sum(mask_plasma > 0.5)),
        "mask_active_ratio": float(np.mean(mask_plasma > 0.5)),
        "density_floor": float(density_floor),
        "fill_value": float(fill_value),
        "target_stats_plasma_space": target_stats,
    }
    with (dst_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="data/outputs_icp_stage4_enriched_360")
    parser.add_argument("--dst-root", default="data/outputs_icp_stage4_enriched_360_csv_npz_core4_linear")
    parser.add_argument("--density-floor", type=float, default=1.0e8)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--smoke-groups", type=int, default=6)
    parser.add_argument("--skip-mask-validation", action="store_true")
    parser.add_argument(
        "--structure-spatial-v1",
        action="store_true",
        help="Write process-only cond columns plus per-case structure_features/*.npz.",
    )
    parser.add_argument(
        "--part-sdf-lite-v1",
        action="store_true",
        help="Write per-coil SDF channels and parametric parts artifacts for icp_part_sdf_lite_v1.",
    )
    args = parser.parse_args()

    summary = convert(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        density_floor=float(args.density_floor),
        fill_value=float(args.fill_value),
        smoke_groups=int(args.smoke_groups),
        validate_masks=not bool(args.skip_mask_validation),
        structure_spatial_v1=bool(args.structure_spatial_v1),
        part_sdf_lite_v1=bool(args.part_sdf_lite_v1),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
