#!/usr/bin/env python3
"""Convert ICP_stage4 enriched COMSOL exports to a multi-field structure dataset.

This is an external dataset preparation script for the ICP_stage4 conference
problem setting. It keeps coil geometry source parameters out of scalar model
conditions and writes structure feature inputs plus 2D field targets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from convert_outputs_icp_stage4_enriched_to_csv_npz_core4 import (
    PART_IDS,
    PART_SDF_KEYS,
    _load_domain_mask,
    _read_coil_layout,
    _read_manifest,
    _select_smoke_rows,
    _write_index,
)


PROCESS_COND_COLUMNS = ["pp", "pp0"]
GEOMETRY_SOURCE_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc"]
PLANNED_PROCESS_CONDITION_COLUMNS = [
    "coil_power",
    "bias_power",
    "pressure",
    "gas_flow_rate",
    "gas_composition",
    "rf_frequency",
    "substrate_temperature",
    "wall_temperature",
]

TARGET_SPECS: dict[str, dict[str, Any]] = {
    "ne": {
        "path_col": "path_data_electron_density",
        "usecol": 2,
        "scope": "plasma_only",
        "floor": 1.0e8,
        "source_field": "electron_density",
        "source_expr": "ne",
        "units": "m^-3",
    },
    "ni": {
        "path_col": "path_data_ion_density",
        "usecol": 2,
        "scope": "plasma_only",
        "floor": 1.0e8,
        "source_field": "ion_density",
        "source_expr": "ni",
        "units": "m^-3",
    },
    "phi": {
        "path_col": "path_data_electric_potential",
        "usecol": 2,
        "scope": "plasma_only",
        "floor": None,
        "source_field": "electric_potential",
        "source_expr": "V",
        "units": "V",
    },
    "Te": {
        "path_col": "path_data_electron_temperature",
        "usecol": 2,
        "scope": "plasma_only",
        "floor": 0.0,
        "source_field": "electron_temperature",
        "source_expr": "Te",
        "units": "eV",
    },
    "Br": {
        "path_col": "path_data_magnetic_field",
        "usecol": 2,
        "scope": "valid_field",
        "floor": None,
        "source_field": "magnetic_field",
        "source_expr": "mf.Br",
        "units": "T",
    },
    "Bz": {
        "path_col": "path_data_magnetic_field",
        "usecol": 4,
        "scope": "valid_field",
        "floor": None,
        "source_field": "magnetic_field",
        "source_expr": "mf.Bz",
        "units": "T",
    },
    "Jelr": {
        "path_col": "path_data_electron_current",
        "usecol": 2,
        "scope": "plasma_only",
        "floor": None,
        "source_field": "electron_current",
        "source_expr": "plas.Jelr",
        "units": "A/m^2",
    },
    "Jelz": {
        "path_col": "path_data_electron_current",
        "usecol": 4,
        "scope": "plasma_only",
        "floor": None,
        "source_field": "electron_current",
        "source_expr": "plas.Jelz",
        "units": "A/m^2",
    },
}


def _fieldnames() -> list[str]:
    return [
        "case_id",
        "base_case_id",
        "split_group",
        "source_split",
        "operation_id",
        "axis",
        *PROCESS_COND_COLUMNS,
        *GEOMETRY_SOURCE_COLUMNS,
        "fields_npz",
        "structure_npz",
    ]


def _required_manifest_columns() -> set[str]:
    return {
        "case_id",
        "case_group_id",
        "operation_id",
        "split",
        "path_masks_domain_masks",
        "path_structure_coil_layout",
        *PROCESS_COND_COLUMNS,
        *GEOMETRY_SOURCE_COLUMNS,
        *{str(spec["path_col"]) for spec in TARGET_SPECS.values()},
    }


def _complex_to_float(token: str, *, complex_mode: str) -> float:
    text = str(token).strip()
    if not text:
        return float("nan")
    lower = text.lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    if "i" not in lower and "j" not in lower:
        return float(text)
    try:
        value = complex(text.replace("i", "j").replace("I", "j"))
    except ValueError:
        return float("nan")
    if complex_mode == "magnitude":
        return float(abs(value))
    if complex_mode == "real":
        return float(value.real)
    if complex_mode == "imag":
        return float(value.imag)
    raise ValueError(f"unknown complex mode: {complex_mode}")


def _read_components(path: Path, usecols: list[int], *, complex_mode: str) -> dict[int, np.ndarray]:
    cols = sorted(set(int(c) for c in usecols))
    try:
        raw = np.loadtxt(path, delimiter=",", comments="%", usecols=tuple(cols), dtype=np.float64)
    except ValueError:
        raw_str = np.genfromtxt(
            path,
            delimiter=",",
            comments="%",
            usecols=tuple(cols),
            dtype=str,
            encoding="utf-8",
        )
        if raw_str.ndim == 1:
            raw_str = raw_str.reshape(-1, 1)
        raw = np.empty(raw_str.shape, dtype=np.float64)
        for j in range(raw_str.shape[1]):
            raw[:, j] = np.fromiter(
                (_complex_to_float(v, complex_mode=complex_mode) for v in raw_str[:, j]),
                dtype=np.float64,
                count=raw_str.shape[0],
            )
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return {col: np.asarray(raw[:, idx], dtype=np.float32).reshape(-1) for idx, col in enumerate(cols)}


def _active_vec(mask_info: dict[str, Any], scope: str) -> np.ndarray:
    rid = np.asarray(mask_info["rid"], dtype=np.int64)
    zid = np.asarray(mask_info["zid"], dtype=np.int64)
    if scope == "valid_field":
        grid = np.asarray(mask_info["valid_field_mask"], dtype=np.float32)
        return (grid[zid, rid] > 0.5).reshape(-1)
    if scope == "plasma_only":
        return np.asarray(mask_info["plasma_vec"], dtype=bool).reshape(-1)
    raise ValueError(f"unknown target scope: {scope}")


def _field_to_grid(
    vals: np.ndarray,
    *,
    mask_info: dict[str, Any],
    scope: str,
    fill_value: float,
    floor: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    rid = np.asarray(mask_info["rid"], dtype=np.int64)
    zid = np.asarray(mask_info["zid"], dtype=np.int64)
    shape = tuple(int(v) for v in mask_info["shape"])
    if int(vals.shape[0]) != int(rid.shape[0]):
        raise ValueError(f"field row count mismatch: got={vals.shape[0]} expected={rid.shape[0]}")
    active = _active_vec(mask_info, scope)
    valid_vec = np.isfinite(vals) & active
    grid = np.full(shape, float(fill_value), dtype=np.float32)
    values = np.asarray(vals[valid_vec], dtype=np.float32)
    if floor is not None:
        values = np.maximum(values, np.float32(float(floor))).astype(np.float32)
    grid[zid[valid_vec], rid[valid_vec]] = values
    valid_grid = np.zeros(shape, dtype=np.uint8)
    valid_grid[zid[valid_vec], rid[valid_vec]] = 1
    return grid.astype(np.float32), valid_grid


def _init_target_stats() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "min": float("inf"),
            "max": float("-inf"),
            "finite": 0,
            "active": 0,
            "scope": str(spec["scope"]),
        }
        for name, spec in TARGET_SPECS.items()
    }


def _update_stats(stats: dict[str, dict[str, Any]], name: str, arr: np.ndarray, valid: np.ndarray) -> None:
    item = stats[name]
    active = np.asarray(valid > 0, dtype=bool)
    vals = np.asarray(arr, dtype=np.float32)[active]
    finite = np.isfinite(vals)
    item["active"] = int(item["active"]) + int(active.sum())
    item["finite"] = int(item["finite"]) + int(finite.sum())
    if np.any(finite):
        fvals = vals[finite]
        item["min"] = min(float(item["min"]), float(np.min(fvals)))
        item["max"] = max(float(item["max"]), float(np.max(fvals)))


def _finalize_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, item in stats.items():
        active = int(item["active"])
        finite = int(item["finite"])
        out[name] = {
            "scope": str(item["scope"]),
            "min": None if finite <= 0 else float(item["min"]),
            "max": None if finite <= 0 else float(item["max"]),
            "finite_rate_in_scope": None if active <= 0 else float(finite / max(active, 1)),
            "active_cells_total": active,
        }
    return out


def _write_parts_manifest(path: Path, reference_case_id: str) -> None:
    param_specs: dict[str, dict[str, float]] = {}
    for part_id in PART_IDS:
        param_specs[f"part.{part_id}.tx"] = {"default": 0.0, "min": -0.02, "max": 0.02}
        param_specs[f"part.{part_id}.ty"] = {"default": 0.0, "min": -0.02, "max": 0.02}
        param_specs[f"part.{part_id}.scale_x"] = {"default": 1.0, "min": 0.9, "max": 1.1}
        param_specs[f"part.{part_id}.scale_y"] = {"default": 1.0, "min": 0.9, "max": 1.1}
    payload = {
        "part_ids": list(PART_IDS),
        "plasma_mode": "preserve",
        "reference_case_id": reference_case_id,
        "optimization_space": "coil_structure_features_v1",
        "param_specs": param_specs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_readme(summary: dict[str, Any]) -> str:
    targets = ", ".join(str(v) for v in summary["target_order"])
    process = ", ".join(str(v) for v in summary["scalar_condition_columns"])
    geometry = ", ".join(str(v) for v in summary["geometry_source_columns"])
    return f"""# ICP_stage4 Multi-Field Structure Dataset

This dataset is organized for the unified ICP coil-design problem setting:

```text
process scalar conditions + coil structure feature maps
  -> 2D fields: {targets}
```

## Inputs

- Scalar process conditions available in this source dataset: `{process}`
- Geometry source columns: `{geometry}`
- Geometry source columns are used to generate structure features only. They
  are not scalar model inputs.

Future datasets can add process columns such as coil power, bias power,
pressure, gas flow rate, gas composition, RF frequency, substrate temperature,
and wall temperature when those quantities are present in the source manifest or
new COMSOL runs.

## Structure Features

Each `structure_features/*.npz` contains:

- `mask_plasma`, `mask_coil`, `valid_field_mask`, `outside_mask`
- `part_mask_stack`
- `sdf_coil_01` ... `sdf_coil_06`
- `r_coords`, `z_coords`

This supports both fixed-slot SDF features and order-invariant part summary
features such as `part_lite_v1`.

## Targets

Each `fields/*.npz` contains target arrays and matching `valid_<target>` masks:

- `ne`, `ni`, `phi`, `Te`
- `Br`, `Bz`
- `Jelr`, `Jelz`

Complex COMSOL phasor values are converted with
`complex_field_mode={summary["complex_field_mode"]}`.

## Files

- `index.csv`: all {summary["n_cases"]} cases
- `index_smoke.csv`: {summary["smoke_cases"]} smoke cases
- `conversion_summary.json`: conversion and target stats
- `dataset_spec.json`: problem-level input/output schema
"""


def convert(
    *,
    src_root: Path,
    dst_root: Path,
    fill_value: float,
    smoke_groups: int,
    limit_cases: int,
    complex_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        elif any(dst_root.iterdir()):
            raise FileExistsError(f"destination already exists and is not empty: {dst_root}")
    fields_dir = dst_root / "fields"
    structure_dir = dst_root / "structure_features"
    geom_dir = dst_root / "geometry"
    fields_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = src_root / "learning_manifest.csv"
    rows = _read_manifest(manifest_path)
    missing = sorted(_required_manifest_columns() - set(rows[0].keys()))
    if missing:
        raise ValueError(f"learning manifest missing columns: {missing}")
    if limit_cases > 0:
        rows = rows[: int(limit_cases)]

    first_mask = _load_domain_mask(src_root / rows[0]["path_masks_domain_masks"])
    shape = tuple(int(v) for v in first_mask["shape"])
    np.save(geom_dir / "r_coords.npy", np.asarray(first_mask["r_coords"], dtype=np.float32))
    np.save(geom_dir / "z_coords.npy", np.asarray(first_mask["z_coords"], dtype=np.float32))
    np.save(geom_dir / "mask_plasma.npy", np.asarray(first_mask["mask_plasma"], dtype=np.float32))
    np.save(geom_dir / "valid_field_mask.npy", np.asarray(first_mask["valid_field_mask"], dtype=np.float32))
    np.save(geom_dir / "wafer_mask.npy", np.asarray(first_mask["mask_plasma"], dtype=np.float32))
    np.save(geom_dir / "eps.npy", np.ones(shape, dtype=np.float32))

    index_rows: list[dict[str, Any]] = []
    stats = _init_target_stats()
    groups: set[str] = set()
    split_counts: dict[str, int] = {}
    nncoil_counts: dict[str, int] = {}
    reference_parts: dict[str, Any] | None = None

    specs_by_path: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, spec in TARGET_SPECS.items():
        specs_by_path.setdefault(str(spec["path_col"]), []).append((name, spec))

    for idx, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        group = str(row["case_group_id"])
        split = str(row["split"])
        mask_info = _load_domain_mask(src_root / row["path_masks_domain_masks"])
        if not (
            np.array_equal(mask_info["r_coords"], first_mask["r_coords"])
            and np.array_equal(mask_info["z_coords"], first_mask["z_coords"])
        ):
            raise ValueError(f"case-specific grid differs from first case: {case_id}")

        field_payload: dict[str, Any] = {}
        for path_col, specs in specs_by_path.items():
            usecols = [int(spec["usecol"]) for _, spec in specs]
            components = _read_components(src_root / row[path_col], usecols, complex_mode=complex_mode)
            for name, spec in specs:
                arr, valid = _field_to_grid(
                    components[int(spec["usecol"])],
                    mask_info=mask_info,
                    scope=str(spec["scope"]),
                    fill_value=float(fill_value),
                    floor=spec["floor"],
                )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"non-finite converted field: case={case_id} target={name}")
                field_payload[name] = arr
                field_payload[f"valid_{name}"] = valid.astype(np.uint8)
                _update_stats(stats, name, arr, valid)

        fields_rel = Path("fields") / f"{case_id}.npz"
        np.savez_compressed(fields_dir / f"{case_id}.npz", **field_payload)

        parts = _read_coil_layout(
            src_root / row["path_structure_coil_layout"],
            r_coords=np.asarray(mask_info["r_coords"], dtype=np.float32),
            z_coords=np.asarray(mask_info["z_coords"], dtype=np.float32),
        )
        structure_payload: dict[str, Any] = {
            "mask_plasma": np.asarray(mask_info["mask_plasma"], dtype=np.float32),
            "mask_coil": np.asarray(mask_info["mask_coil"], dtype=np.float32),
            "valid_field_mask": np.asarray(mask_info["valid_field_mask"], dtype=np.float32),
            "outside_mask": np.asarray(mask_info["outside_mask"], dtype=np.float32),
            "r_coords": np.asarray(mask_info["r_coords"], dtype=np.float32),
            "z_coords": np.asarray(mask_info["z_coords"], dtype=np.float32),
            "part_mask_stack": np.asarray(parts["mask_stack"], dtype=np.float32),
            **{key: np.asarray(value, dtype=np.float32) for key, value in parts["sdf_maps"].items()},
        }
        structure_rel = Path("structure_features") / f"{case_id}.npz"
        np.savez_compressed(structure_dir / f"{case_id}.npz", **structure_payload)
        if reference_parts is None or int(parts["active_count"]) > int(reference_parts["active_count"]):
            reference_parts = {
                "case_id": case_id,
                "active_count": int(parts["active_count"]),
                "mask_stack": np.asarray(parts["mask_stack"], dtype=np.float32),
            }

        groups.add(group)
        split_counts[split] = split_counts.get(split, 0) + 1
        nncoil_key = str(int(float(row["nncoil"])))
        nncoil_counts[nncoil_key] = nncoil_counts.get(nncoil_key, 0) + 1
        index_row: dict[str, Any] = {
            "case_id": case_id,
            "base_case_id": group,
            "split_group": group,
            "source_split": split,
            "operation_id": str(row["operation_id"]),
            "axis": 0.0,
            "fields_npz": fields_rel.as_posix(),
            "structure_npz": structure_rel.as_posix(),
        }
        for col in PROCESS_COND_COLUMNS:
            index_row[col] = float(row[col])
        for col in GEOMETRY_SOURCE_COLUMNS:
            index_row[col] = float(row[col])
        index_rows.append(index_row)
        if idx % 20 == 0 or idx == len(rows):
            print(f"converted {idx}/{len(rows)} cases")

    if reference_parts is None:
        raise ValueError("no coil parts were built")
    np.savez_compressed(
        geom_dir / "parts_pack.npz",
        part_ids=np.asarray(PART_IDS, dtype=object),
        mask_stack=np.asarray(reference_parts["mask_stack"], dtype=np.float32),
    )
    _write_parts_manifest(geom_dir / "parts_manifest.json", str(reference_parts["case_id"]))

    fieldnames = _fieldnames()
    _write_index(dst_root / "index.csv", index_rows, fieldnames=fieldnames)
    smoke_rows = _select_smoke_rows(index_rows, int(smoke_groups))
    if smoke_rows:
        _write_index(dst_root / "index_smoke.csv", smoke_rows, fieldnames=fieldnames)

    target_summary = {
        name: {
            "source_field": str(spec["source_field"]),
            "source_expr": str(spec["source_expr"]),
            "units": str(spec["units"]),
            "scope": str(spec["scope"]),
            "floor": spec["floor"],
            "value_transform": "identity",
        }
        for name, spec in TARGET_SPECS.items()
    }
    summary = {
        "dataset_kind": "icp_stage4_multifield_structure_v1",
        "source_root": str(src_root),
        "dataset_root": str(dst_root),
        "source_manifest": str(manifest_path),
        "n_cases": int(len(index_rows)),
        "n_groups": int(len(groups)),
        "shape": [int(shape[0]), int(shape[1])],
        "targets": target_summary,
        "target_order": list(TARGET_SPECS.keys()),
        "target_stats": _finalize_stats(stats),
        "scalar_condition_columns": list(PROCESS_COND_COLUMNS),
        "planned_process_condition_columns": list(PLANNED_PROCESS_CONDITION_COLUMNS),
        "geometry_source_columns": list(GEOMETRY_SOURCE_COLUMNS),
        "structure_feature_policy": {
            "coil_geometry_sources_are_not_scalar_conditions": True,
            "structure_npz_column": "structure_npz",
            "part_mask_stack": True,
            "part_sdf_channels": list(PART_SDF_KEYS),
            "recommended_profiles": [
                "icp_part_sdf_lite_v1",
                "part_lite_v1",
                "struct_spatial_v1",
            ],
        },
        "source_split_counts": dict(sorted(split_counts.items())),
        "nncoil_counts": dict(sorted(nncoil_counts.items())),
        "smoke_index": "index_smoke.csv" if smoke_rows else "",
        "smoke_cases": int(len(smoke_rows)),
        "smoke_groups": int(len({str(r["split_group"]) for r in smoke_rows})),
        "fill_value": float(fill_value),
        "complex_field_mode": str(complex_mode),
        "parts_manifest": "geometry/parts_manifest.json",
        "parts_pack": "geometry/parts_pack.npz",
    }
    (dst_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dst_root / "dataset_spec.json").write_text(
        json.dumps(
            {
                "problem": "process scalar + coil structure features -> 2D multi-physics fields",
                "inputs": {
                    "available_scalar_process_conditions": list(PROCESS_COND_COLUMNS),
                    "planned_process_conditions_for_future_datasets": list(
                        PLANNED_PROCESS_CONDITION_COLUMNS
                    ),
                    "geometry_sources_for_feature_generation": list(GEOMETRY_SOURCE_COLUMNS),
                    "structure_features": {
                        "masks": ["mask_plasma", "mask_coil", "valid_field_mask", "outside_mask"],
                        "part_masks": ["part_mask_stack"],
                        "part_sdfs": list(PART_SDF_KEYS),
                    },
                },
                "outputs": list(TARGET_SPECS.keys()),
                "notes": [
                    "Coil count, placement, and dimensions are not scalar model inputs.",
                    "QoI values are derived from 2D predicted fields, not learned as primary targets.",
                    "Additional process conditions require source data columns or new COMSOL runs.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (dst_root / "README.md").write_text(_dataset_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="data/outputs_icp_stage4_enriched_360")
    parser.add_argument(
        "--dst-root",
        default="data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1",
    )
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument(
        "--complex-mode",
        choices=("magnitude", "real", "imag"),
        default="magnitude",
        help="How to convert COMSOL complex phasor CSV values to real-valued training fields.",
    )
    parser.add_argument("--smoke-groups", type=int, default=6)
    parser.add_argument("--limit-cases", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = convert(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        fill_value=float(args.fill_value),
        smoke_groups=int(args.smoke_groups),
        limit_cases=int(args.limit_cases),
        complex_mode=str(args.complex_mode),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
