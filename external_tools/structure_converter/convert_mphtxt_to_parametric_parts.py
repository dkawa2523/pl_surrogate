#!/usr/bin/env python3
"""Build parametric-parts geometry artifacts from mphtxt structure files.

This tool intentionally lives outside `src/plasma_surrogate` because it is a
dataset conversion utility, not a runtime feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_MESH_VERTEX_COUNT_RE = re.compile(r"^\s*(\d+)\s*#\s*number of mesh vertices\s*$", re.IGNORECASE)
_FLOAT_PAIR_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class MphtxtStats:
    n_vertices: int
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    r_mean: float
    z_mean: float


def _read_mphtxt_vertex_stats(path: Path) -> MphtxtStats:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n_vertices = 0
    for line in text:
        m = _MESH_VERTEX_COUNT_RE.match(line)
        if m is not None:
            n_vertices = int(m.group(1))
            break
    if n_vertices <= 0:
        raise ValueError(f"failed to parse mesh vertex count from mphtxt: {path}")

    start = None
    for idx, line in enumerate(text):
        if line.strip().lower() == "# mesh vertex coordinates":
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"failed to locate mesh vertex coordinate block: {path}")

    coords: list[tuple[float, float]] = []
    for line in text[start:]:
        if len(coords) >= n_vertices:
            break
        m = _FLOAT_PAIR_RE.match(line)
        if m is None:
            continue
        coords.append((float(m.group(1)), float(m.group(2))))
    if len(coords) != n_vertices:
        raise ValueError(
            f"mphtxt vertex block size mismatch: expected={n_vertices}, parsed={len(coords)}, file={path}"
        )

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"invalid coordinate matrix parsed from {path}")
    return MphtxtStats(
        n_vertices=int(arr.shape[0]),
        r_min=float(np.min(arr[:, 0])),
        r_max=float(np.max(arr[:, 0])),
        z_min=float(np.min(arr[:, 1])),
        z_max=float(np.max(arr[:, 1])),
        r_mean=float(np.mean(arr[:, 0])),
        z_mean=float(np.mean(arr[:, 1])),
    )


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    out = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    for _ in range(max(0, int(steps))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = np.maximum.reduce(
            [
                p[:-2, :-2],
                p[:-2, 1:-1],
                p[:-2, 2:],
                p[1:-1, :-2],
                p[1:-1, 1:-1],
                p[1:-1, 2:],
                p[2:, :-2],
                p[2:, 1:-1],
                p[2:, 2:],
            ]
        ).astype(np.float32)
    return out


def _warp_template(
    template: np.ndarray,
    *,
    tx: float,
    ty: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    src = (np.asarray(template, dtype=np.float32) > 0.5).astype(np.float32)
    h, w = src.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    sx = max(float(sx), 1.0e-3)
    sy = max(float(sy), 1.0e-3)
    x_src = (xv - np.float32(tx)) / np.float32(sx)
    y_src = (yv - np.float32(ty)) / np.float32(sy)
    ix = np.clip(np.rint(x_src * np.float32(max(w - 1, 1))).astype(np.int64), 0, max(w - 1, 0))
    iy = np.clip(np.rint(y_src * np.float32(max(h - 1, 1))).astype(np.int64), 0, max(h - 1, 0))
    return src[iy, ix].astype(np.float32)


def _read_structure_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"empty structure file index: {path}")
    return rows


def convert_mphtxt_to_parametric_parts(
    *,
    source_root: Path,
    structure_index_csv: Path,
    conditions_with_structure_csv: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    geometry_root = dataset_root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)

    mask_plasma_path = geometry_root / "mask_plasma.npy"
    if not mask_plasma_path.exists():
        raise FileNotFoundError(f"missing base geometry mask: {mask_plasma_path}")
    mask_plasma = np.asarray(np.load(mask_plasma_path), dtype=np.float32)
    if mask_plasma.ndim != 2:
        raise ValueError(f"geometry/mask_plasma.npy must be 2-D; got shape={mask_plasma.shape}")

    solid_template = (1.0 - (mask_plasma > 0.5).astype(np.float32)).astype(np.float32)
    if float(np.sum(solid_template)) <= 0.0:
        raise ValueError("mask_plasma yields empty solid template; cannot build part masks")
    solid_template = _dilate(solid_template, steps=1)

    structure_rows = _read_structure_rows(structure_index_csv)
    with conditions_with_structure_csv.open("r", encoding="utf-8") as f:
        cond_rows = list(csv.DictReader(f))
    if len(cond_rows) == 0:
        raise ValueError(f"empty conditions_with_structure: {conditions_with_structure_csv}")
    base_names = sorted({str(r.get("base_name", "")).strip() for r in cond_rows if str(r.get("base_name", "")).strip()})
    if len(base_names) == 0:
        raise ValueError("conditions_with_structure.csv must include non-empty base_name")

    by_base: dict[str, Path] = {}
    for row in structure_rows:
        name = str(row.get("base_name", "")).strip()
        rel = str(row.get("model_mphtxt_relative_path", "")).strip()
        if name and rel:
            by_base[name] = source_root / rel.replace("\\", "/")

    missing_base = [name for name in base_names if name not in by_base]
    if missing_base:
        raise ValueError(f"structure index is missing base_name entries referenced by conditions: {missing_base}")

    stats_by_base: dict[str, MphtxtStats] = {}
    for name in base_names:
        mphtxt = by_base[name]
        if not mphtxt.exists():
            raise FileNotFoundError(f"mphtxt not found for base={name!r}: {mphtxt}")
        stats_by_base[name] = _read_mphtxt_vertex_stats(mphtxt)

    r_means = np.asarray([stats_by_base[name].r_mean for name in base_names], dtype=np.float64)
    z_means = np.asarray([stats_by_base[name].z_mean for name in base_names], dtype=np.float64)
    r_span = float(np.max(r_means) - np.min(r_means))
    z_span = float(np.max(z_means) - np.min(z_means))
    if r_span <= 1.0e-12:
        r_span = 1.0
    if z_span <= 1.0e-12:
        z_span = 1.0

    mask_stack: list[np.ndarray] = []
    for idx, name in enumerate(base_names):
        st = stats_by_base[name]
        tx = 0.08 * (float(st.r_mean - np.mean(r_means)) / r_span)
        ty = 0.08 * (float(st.z_mean - np.mean(z_means)) / z_span)
        width = max(float(st.r_max - st.r_min), 1.0e-9)
        height = max(float(st.z_max - st.z_min), 1.0e-9)
        sx = 0.95 + 0.10 * (width / max(np.max([max(float(stats_by_base[n].r_max - stats_by_base[n].r_min), 1.0e-9) for n in base_names]), 1.0e-9))
        sy = 0.95 + 0.10 * (height / max(np.max([max(float(stats_by_base[n].z_max - stats_by_base[n].z_min), 1.0e-9) for n in base_names]), 1.0e-9))
        warped = _warp_template(solid_template, tx=tx, ty=ty, sx=sx, sy=sy)
        if float(np.sum(warped)) <= 0.0:
            # deterministic fallback: shift by index to keep non-empty, unique masks.
            shift_y = int(idx % 3) - 1
            shift_x = int((idx // 3) % 3) - 1
            warped = np.roll(np.roll(solid_template, shift_y, axis=0), shift_x, axis=1).astype(np.float32)
        mask_stack.append((warped > 0.5).astype(np.float32))

    stack = np.stack(mask_stack, axis=0).astype(np.float32)
    part_ids = np.asarray(base_names, dtype=object)

    part_param_specs: dict[str, dict[str, float]] = {}
    for name in base_names:
        part_param_specs[f"part.{name}.tx"] = {"min": -0.10, "max": 0.10, "default": 0.0}
        part_param_specs[f"part.{name}.ty"] = {"min": -0.10, "max": 0.10, "default": 0.0}
        part_param_specs[f"part.{name}.scale_x"] = {"min": 0.80, "max": 1.20, "default": 1.0}
        part_param_specs[f"part.{name}.scale_y"] = {"min": 0.80, "max": 1.20, "default": 1.0}
        part_param_specs[f"part.{name}.rotation_deg"] = {"min": -15.0, "max": 15.0, "default": 0.0}
        part_param_specs[f"part.{name}.fillet"] = {"min": -0.02, "max": 0.02, "default": 0.0}
    part_param_specs["gap.delta"] = {"min": -0.04, "max": 0.04, "default": 0.0}
    part_param_specs["offset.x"] = {"min": -0.08, "max": 0.08, "default": 0.0}
    part_param_specs["offset.y"] = {"min": -0.08, "max": 0.08, "default": 0.0}

    parts_pack_path = geometry_root / "parts_pack.npz"
    np.savez_compressed(parts_pack_path, mask_stack=stack, part_ids=part_ids)

    manifest_path = geometry_root / "parts_manifest.json"
    manifest = {
        "version": "v1",
        "generator": "external_tools/structure_converter/convert_mphtxt_to_parametric_parts.py",
        "part_ids": [str(v) for v in base_names],
        "param_specs": part_param_specs,
        "source": {
            "conditions_with_structure_csv": str(conditions_with_structure_csv),
            "structure_file_index_csv": str(structure_index_csv),
            "source_root": str(source_root),
            "mphtxt_files": {name: str(by_base[name]) for name in base_names},
            "mphtxt_vertex_stats": {
                name: {
                    "n_vertices": int(stats_by_base[name].n_vertices),
                    "r_min": float(stats_by_base[name].r_min),
                    "r_max": float(stats_by_base[name].r_max),
                    "z_min": float(stats_by_base[name].z_min),
                    "z_max": float(stats_by_base[name].z_max),
                    "r_mean": float(stats_by_base[name].r_mean),
                    "z_mean": float(stats_by_base[name].z_mean),
                }
                for name in base_names
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "dataset_root": str(dataset_root),
        "geometry_root": str(geometry_root),
        "parts_manifest": str(manifest_path),
        "parts_pack": str(parts_pack_path),
        "n_parts": int(stack.shape[0]),
        "shape": [int(v) for v in stack.shape],
        "part_ids": [str(v) for v in base_names],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory that contains conditions_with_structure.csv and structure/*.mphtxt",
    )
    parser.add_argument(
        "--structure-index-csv",
        required=True,
        help="Path to structure_file_index.csv",
    )
    parser.add_argument(
        "--conditions-with-structure-csv",
        required=True,
        help="Path to conditions_with_structure.csv",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Converted csv_npz dataset root where geometry/ exists",
    )
    args = parser.parse_args()

    summary = convert_mphtxt_to_parametric_parts(
        source_root=Path(args.source_root),
        structure_index_csv=Path(args.structure_index_csv),
        conditions_with_structure_csv=Path(args.conditions_with_structure_csv),
        dataset_root=Path(args.dataset_root),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

