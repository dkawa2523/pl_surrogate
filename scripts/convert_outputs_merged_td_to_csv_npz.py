#!/usr/bin/env python3
"""One-off converter: outputs_merged_td -> csv_npz dataset contract.

This script is intentionally kept outside the runtime package because this
conversion is not treated as a regular pipeline feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


_TIME_PATTERN = re.compile(r"@\s*t\s*=\s*([0-9eE+.\-]+)")


def _read_export_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    header_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("%"):
                header_line = line[1:].strip()
    cols = [c.strip() for c in header_line.split(",")] if header_line else []
    raw = np.loadtxt(path, delimiter=",", comments="%")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        raise ValueError(f"CSV must have at least 3 columns (R,Z,value): {path}")
    return raw[:, 0], raw[:, 1], raw[:, 2:], cols


def _grid_index(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_unique = np.unique(r)
    z_unique = np.unique(z)
    rid = np.searchsorted(r_unique, r)
    zid = np.searchsorted(z_unique, z)
    h = z_unique.shape[0]
    w = r_unique.shape[0]
    if h * w != r.shape[0]:
        raise ValueError(f"Incomplete mesh points: got={r.shape[0]} expected={h*w}")
    return rid.astype(np.int64), zid.astype(np.int64), r_unique.astype(np.float32), z_unique.astype(np.float32)


def _vector_to_grid(
    vals: np.ndarray,
    rid: np.ndarray,
    zid: np.ndarray,
    h: int,
    w: int,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, w), float(fill_value), dtype=np.float32)
    valid = np.isfinite(vals)
    grid[zid[valid], rid[valid]] = vals[valid].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[zid[valid], rid[valid]] = 1.0
    return grid, mask


def _parse_times(col_names: list[str], n_values: int) -> np.ndarray:
    names = col_names[2:] if len(col_names) >= 2 + n_values else []
    times: list[float] = []
    for name in names:
        match = _TIME_PATTERN.search(name)
        if match:
            times.append(float(match.group(1)))
    if len(times) != n_values:
        return np.linspace(0.0, 1.0, n_values, dtype=np.float64)
    arr = np.asarray(times, dtype=np.float64)
    tmin = float(np.min(arr))
    tmax = float(np.max(arr))
    if tmax <= tmin:
        return np.zeros((n_values,), dtype=np.float64)
    return (arr - tmin) / (tmax - tmin)


def convert_outputs_merged_td(
    src_root: Path,
    dst_root: Path,
    *,
    mode: str,
    cond_columns: list[str],
    ne_floor: float,
    fill_value: float,
) -> dict[str, object]:
    src = Path(src_root)
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    fields_dir = dst / "fields"
    geom_dir = dst / "geometry"
    fields_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    with (src / "conditions.csv").open("r", encoding="utf-8") as f:
        conditions = {str(r["case_id"]): r for r in csv.DictReader(f)}
    with (src / "export_file_index.csv").open("r", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))

    if mode == "periodic":
        group = "periodic"
        keys = {
            "Ne": "time_periodic:Ne",
            "Ni": "time_periodic:Ni",
            "Te": "time_periodic:Te",
            "phi": "time_periodic:Phi",
        }
    elif mode == "transient":
        group = "transient"
        keys = {
            "Ne": "tp_to_td:Ne 1",
            "Ni": "tp_to_td:Ni 1",
            "Te": "tp_to_td:Te 1",
            "phi": "tp_to_td:Phi 1",
        }
    else:
        raise ValueError("mode must be one of {'periodic','transient'}")

    files_by_case: dict[str, dict[str, Path]] = {}
    for row in index_rows:
        rel = str(row["relative_path"])
        rel_norm = rel.replace("\\", "/")
        if not rel_norm.startswith(f"{group}/"):
            continue
        out_key = str(row["output_key"])
        for var, expected in keys.items():
            if out_key == expected:
                cid = str(row["case_id"])
                files_by_case.setdefault(cid, {})[var] = src / Path(rel_norm)

    missing_cases = sorted([cid for cid in conditions.keys() if cid not in files_by_case])
    if missing_cases:
        raise ValueError(f"Missing export files for cases: {missing_cases[:5]}")

    sample_rows: list[dict[str, object]] = []
    mask_plasma: np.ndarray | None = None
    rid0: np.ndarray | None = None
    zid0: np.ndarray | None = None
    coord_r: np.ndarray | None = None
    coord_z: np.ndarray | None = None
    h, w = 0, 0

    for base_case_id in sorted(files_by_case.keys()):
        file_map = files_by_case[base_case_id]
        if set(file_map.keys()) != {"Ne", "Ni", "Te", "phi"}:
            raise ValueError(f"Case {base_case_id} does not have Ne/Ni/Te/phi quartet")

        r_phi, z_phi, phi_vals, phi_cols = _read_export_csv(file_map["phi"])
        r_ne, z_ne, ne_vals, _ = _read_export_csv(file_map["Ne"])
        r_ni, z_ni, ni_vals, _ = _read_export_csv(file_map["Ni"])
        r_te, z_te, te_vals, _ = _read_export_csv(file_map["Te"])
        if not (
            np.allclose(r_phi, r_ne)
            and np.allclose(r_phi, r_ni)
            and np.allclose(r_phi, r_te)
            and np.allclose(z_phi, z_ne)
            and np.allclose(z_phi, z_ni)
            and np.allclose(z_phi, z_te)
        ):
            raise ValueError(f"R/Z grid mismatch in case={base_case_id}")

        rid, zid, r_unique, z_unique = _grid_index(r_phi, z_phi)
        if rid0 is None:
            rid0, zid0 = rid, zid
            coord_r, coord_z = r_unique, z_unique
            h, w = int(z_unique.shape[0]), int(r_unique.shape[0])
        else:
            if h != z_unique.shape[0] or w != r_unique.shape[0]:
                raise ValueError(f"Grid shape mismatch for case={base_case_id}")
            if not (np.array_equal(rid0, rid) and np.array_equal(zid0, zid)):
                raise ValueError(f"Grid ordering mismatch for case={base_case_id}")

        if not (ne_vals.shape[1] == ni_vals.shape[1] == te_vals.shape[1] == phi_vals.shape[1]):
            raise ValueError(f"Time channel count mismatch for case={base_case_id}")
        n_steps = int(ne_vals.shape[1])
        if mode == "periodic" and n_steps != 1:
            raise ValueError(f"Periodic export must have 1 snapshot per case, got={n_steps}")
        axis_values = np.zeros((1,), dtype=np.float64) if mode == "periodic" else _parse_times(phi_cols, n_steps)

        for step_idx in range(n_steps):
            ne_grid, ne_mask = _vector_to_grid(ne_vals[:, step_idx], rid, zid, h, w, fill_value)
            ni_grid, ni_mask = _vector_to_grid(ni_vals[:, step_idx], rid, zid, h, w, fill_value)
            te_grid, te_mask = _vector_to_grid(te_vals[:, step_idx], rid, zid, h, w, fill_value)
            phi_grid, phi_mask = _vector_to_grid(phi_vals[:, step_idx], rid, zid, h, w, fill_value)

            # NaN in source CSV is chamber-part region (= non-plasma). Exclude it from mask.
            valid_mask = (ne_mask > 0.5) & (ni_mask > 0.5) & (te_mask > 0.5) & (phi_mask > 0.5)
            mask_now = valid_mask.astype(np.float32)
            mask_plasma = mask_now if mask_plasma is None else (mask_plasma * mask_now)

            ne_safe = np.maximum(ne_grid, float(ne_floor))
            ni_safe = np.maximum(ni_grid, float(ne_floor))
            log_ne = np.where(valid_mask, np.log10(ne_safe), float(fill_value)).astype(np.float32)
            log_ni = np.where(valid_mask, np.log10(ni_safe), float(fill_value)).astype(np.float32)
            te_out = np.where(valid_mask, te_grid, float(fill_value)).astype(np.float32)
            phi_out = np.where(valid_mask, phi_grid, float(fill_value)).astype(np.float32)

            sample_id = f"{base_case_id}__steady" if mode == "periodic" else f"{base_case_id}__t{step_idx:03d}"
            rel_npz = Path("fields") / f"{sample_id}.npz"
            np.savez_compressed(fields_dir / f"{sample_id}.npz", log_ne=log_ne, log_ni=log_ni, Te=te_out, phi=phi_out)

            cond_row = conditions[base_case_id]
            row: dict[str, object] = {
                "case_id": sample_id,
                "base_case_id": base_case_id,
                "split_group": base_case_id,
                "axis": float(axis_values[step_idx]),
                "fields_npz": rel_npz.as_posix(),
            }
            for col in cond_columns:
                row[col] = float(cond_row[col])
            sample_rows.append(row)

    if mask_plasma is None or coord_r is None or coord_z is None:
        raise ValueError("No samples were converted from outputs_merged_td")

    np.save(geom_dir / "mask_plasma.npy", mask_plasma.astype(np.float32))
    np.save(geom_dir / "eps.npy", np.ones_like(mask_plasma, dtype=np.float32))
    np.save(geom_dir / "wafer_mask.npy", mask_plasma.astype(np.float32))
    np.save(geom_dir / "r_coords.npy", coord_r.astype(np.float32))
    np.save(geom_dir / "z_coords.npy", coord_z.astype(np.float32))

    index_path = dst / "index.csv"
    fieldnames = ["case_id", "base_case_id", "split_group", "axis", *cond_columns, "fields_npz"]
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    return {
        "mode": mode,
        "n_base_cases": len(files_by_case),
        "n_samples": len(sample_rows),
        "shape": [int(h), int(w)],
        "cond_columns": cond_columns,
        "index_csv": str(index_path),
        "dataset_root": str(dst),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="data/outputs_merged_td", help="Path to outputs_merged_td directory")
    parser.add_argument("--dst-root", required=True, help="Destination dataset root")
    parser.add_argument("--mode", choices=["periodic", "transient"], required=True)
    parser.add_argument(
        "--cond-columns",
        default="PP0,Td,gamma",
        help="Comma-separated condition column names in conditions.csv",
    )
    parser.add_argument("--ne-floor", type=float, default=1.0e8, help="Floor applied before log10 for Ne")
    parser.add_argument("--fill-value", type=float, default=0.0, help="Fill value used outside plasma mask")
    args = parser.parse_args()

    cond_columns = [s.strip() for s in args.cond_columns.split(",") if s.strip()]
    if len(cond_columns) == 0:
        raise ValueError("cond-columns must contain at least one column name")

    summary = convert_outputs_merged_td(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        mode=args.mode,
        cond_columns=cond_columns,
        ne_floor=float(args.ne_floor),
        fill_value=float(args.fill_value),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
