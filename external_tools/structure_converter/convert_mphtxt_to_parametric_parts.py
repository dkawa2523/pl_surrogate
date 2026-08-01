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


def _read_mphtxt_edge_entities(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read vertex coordinates, edg connectivity, and geometric entity ids."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n_vertices = 0
    for line in lines:
        match = _MESH_VERTEX_COUNT_RE.match(line)
        if match is not None:
            n_vertices = int(match.group(1))
            break
    if n_vertices <= 0:
        raise ValueError(f"failed to parse mesh vertex count from mphtxt: {path}")
    try:
        coord_marker = next(i for i, line in enumerate(lines) if line.strip().lower() == "# mesh vertex coordinates")
    except StopIteration as exc:
        raise ValueError(f"failed to locate mesh vertex coordinate block: {path}") from exc
    vertices: list[tuple[float, float]] = []
    for line in lines[coord_marker + 1 :]:
        match = _FLOAT_PAIR_RE.match(line)
        if match is None:
            if vertices:
                break
            continue
        vertices.append((float(match.group(1)), float(match.group(2))))
        if len(vertices) == n_vertices:
            break
    if len(vertices) != n_vertices:
        raise ValueError(
            f"mphtxt vertex block size mismatch: expected={n_vertices}, parsed={len(vertices)}, file={path}"
        )

    try:
        edge_type = next(
            i for i, line in enumerate(lines) if re.match(r"^\s*\d+\s+edg\s*#\s*type name\s*$", line, re.IGNORECASE)
        )
        element_marker = next(i for i in range(edge_type + 1, len(lines)) if lines[i].strip().lower() == "# elements")
    except StopIteration as exc:
        raise ValueError(f"failed to locate edg element block in mphtxt: {path}") from exc
    n_edges = int(lines[element_marker - 1].split()[0])
    edge_rows: list[tuple[int, int]] = []
    cursor = element_marker + 1
    while cursor < len(lines) and len(edge_rows) < n_edges:
        tokens = lines[cursor].split()
        cursor += 1
        if len(tokens) != 2:
            continue
        try:
            edge_rows.append((int(tokens[0]), int(tokens[1])))
        except ValueError:
            continue
    if len(edge_rows) != n_edges:
        raise ValueError(f"mphtxt edg block size mismatch: expected={n_edges}, parsed={len(edge_rows)}, file={path}")
    try:
        entity_marker = next(
            i for i in range(cursor, len(lines)) if lines[i].strip().lower() == "# geometric entity indices"
        )
    except StopIteration as exc:
        raise ValueError(f"failed to locate edg geometric entity indices in mphtxt: {path}") from exc
    entity_ids: list[int] = []
    for line in lines[entity_marker + 1 :]:
        tokens = line.split()
        if len(tokens) != 1:
            if entity_ids:
                break
            continue
        try:
            entity_ids.append(int(tokens[0]))
        except ValueError:
            if entity_ids:
                break
        if len(entity_ids) == n_edges:
            break
    if len(entity_ids) != n_edges:
        raise ValueError(
            f"mphtxt edg entity count mismatch: expected={n_edges}, parsed={len(entity_ids)}, file={path}"
        )
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    edges_arr = np.asarray(edge_rows, dtype=np.int64)
    entity_arr = np.asarray(entity_ids, dtype=np.int64)
    if np.any(edges_arr < 0) or np.any(edges_arr >= n_vertices):
        raise ValueError(f"mphtxt edg connectivity references an invalid vertex: {path}")
    return vertices_arr, edges_arr, entity_arr


def _rasterize_edge_entities(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    entity_ids: np.ndarray,
    entity_slots: list[int],
    r_coords: np.ndarray,
    z_coords: np.ndarray,
) -> np.ndarray:
    r = np.asarray(r_coords, dtype=np.float64).reshape(-1)
    z = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    if len(r) < 2 or len(z) < 2 or np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("r_coords/z_coords must be strictly increasing with at least two points")
    slot_by_entity = {int(entity_id): idx for idx, entity_id in enumerate(entity_slots)}
    stack = np.zeros((len(entity_slots), len(z), len(r)), dtype=np.float32)
    r_index = np.arange(len(r), dtype=np.float64)
    z_index = np.arange(len(z), dtype=np.float64)
    for (v0, v1), entity_id_raw in zip(edges.tolist(), entity_ids.tolist()):
        entity_id = int(entity_id_raw)
        if entity_id not in slot_by_entity:
            raise ValueError(f"edge references undeclared geometric entity id={entity_id}")
        p0 = vertices[int(v0)]
        p1 = vertices[int(v1)]
        x0 = float(np.interp(p0[0], r, r_index))
        x1 = float(np.interp(p1[0], r, r_index))
        y0 = float(np.interp(p0[1], z, z_index))
        y1 = float(np.interp(p1[1], z, z_index))
        steps = max(1, int(np.ceil(2.0 * max(abs(x1 - x0), abs(y1 - y0)))))
        t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
        xx = np.clip(np.rint(x0 + (x1 - x0) * t).astype(np.int64), 0, len(r) - 1)
        yy = np.clip(np.rint(y0 + (y1 - y0) * t).astype(np.int64), 0, len(z) - 1)
        stack[slot_by_entity[entity_id], yy, xx] = 1.0
    return stack


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
    structure_feature_root = dataset_root / "structure_features"
    geometry_root.mkdir(parents=True, exist_ok=True)
    structure_feature_root.mkdir(parents=True, exist_ok=True)

    mask_plasma_path = geometry_root / "mask_plasma.npy"
    if not mask_plasma_path.exists():
        raise FileNotFoundError(f"missing base geometry mask: {mask_plasma_path}")
    mask_plasma = np.asarray(np.load(mask_plasma_path), dtype=np.float32)
    if mask_plasma.ndim != 2:
        raise ValueError(f"geometry/mask_plasma.npy must be 2-D; got shape={mask_plasma.shape}")

    r_coords_path = geometry_root / "r_coords.npy"
    z_coords_path = geometry_root / "z_coords.npy"
    if not r_coords_path.exists() or not z_coords_path.exists():
        raise FileNotFoundError(
            "exact mphtxt edge rasterization requires geometry/r_coords.npy and geometry/z_coords.npy"
        )
    r_coords = np.asarray(np.load(r_coords_path), dtype=np.float32)
    z_coords = np.asarray(np.load(z_coords_path), dtype=np.float32)
    if mask_plasma.shape != (len(z_coords), len(r_coords)):
        raise ValueError(
            "base geometry mask shape does not match r/z coordinates: "
            f"mask={mask_plasma.shape}, coords={(len(z_coords), len(r_coords))}"
        )

    structure_rows = _read_structure_rows(structure_index_csv)
    with conditions_with_structure_csv.open("r", encoding="utf-8") as f:
        cond_rows = list(csv.DictReader(f))
    if len(cond_rows) == 0:
        raise ValueError(f"empty conditions_with_structure: {conditions_with_structure_csv}")
    base_names = sorted({str(r.get("base_name", "")).strip() for r in cond_rows if str(r.get("base_name", "")).strip()})
    if len(base_names) == 0:
        raise ValueError("conditions_with_structure.csv must include non-empty base_name")

    by_base: dict[str, Path] = {}
    td_by_base: dict[str, float] = {}
    for row in structure_rows:
        name = str(row.get("base_name", "")).strip()
        rel = str(row.get("model_mphtxt_relative_path", "")).strip()
        if name and rel:
            by_base[name] = source_root / rel.replace("\\", "/")
            raw_td = str(row.get("td_value", "")).strip()
            if raw_td:
                td_by_base[name] = float(raw_td)

    missing_base = [name for name in base_names if name not in by_base]
    if missing_base:
        raise ValueError(f"structure index is missing base_name entries referenced by conditions: {missing_base}")

    stats_by_base: dict[str, MphtxtStats] = {}
    edge_mesh_by_base: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in base_names:
        mphtxt = by_base[name]
        if not mphtxt.exists():
            raise FileNotFoundError(f"mphtxt not found for base={name!r}: {mphtxt}")
        stats_by_base[name] = _read_mphtxt_vertex_stats(mphtxt)
        edge_mesh_by_base[name] = _read_mphtxt_edge_entities(mphtxt)

    entity_slots = sorted(
        {
            int(entity_id)
            for _, _, entity_ids in edge_mesh_by_base.values()
            for entity_id in entity_ids.tolist()
        }
    )
    if not entity_slots:
        raise ValueError("mphtxt edge meshes contain no geometric entity ids")
    entity_set = set(entity_slots)
    for name, (_, _, entity_ids) in edge_mesh_by_base.items():
        if set(int(v) for v in entity_ids.tolist()) != entity_set:
            raise ValueError(
                "all alternative structures must expose the same edg entity slots: "
                f"base={name!r}, expected={entity_slots}, got={sorted(set(entity_ids.tolist()))}"
            )
    boundary_stacks: dict[str, np.ndarray] = {}
    mask_stack: list[np.ndarray] = []
    for name in base_names:
        vertices, edges, entity_ids = edge_mesh_by_base[name]
        boundary_stack = _rasterize_edge_entities(
            vertices=vertices,
            edges=edges,
            entity_ids=entity_ids,
            entity_slots=entity_slots,
            r_coords=r_coords,
            z_coords=z_coords,
        )
        if any(float(np.sum(boundary_stack[idx])) <= 0.0 for idx in range(len(entity_slots))):
            raise ValueError(f"rasterized mphtxt contains an empty edg entity slot: base={name!r}")
        boundary_stacks[name] = boundary_stack
        mask_stack.append(np.maximum.reduce(boundary_stack, axis=0).astype(np.float32))

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

    # base2/base3/base4 are mutually exclusive chamber structures.  Enrich the
    # converter-produced, field-aligned domain pack with exact COMSOL edg
    # connectivity rasterized into stable geometric-entity slots.
    structure_npz_by_alternative: dict[str, str] = {}
    slot_ids = [f"boundary_{entity_id:02d}" for entity_id in entity_slots]
    for name in base_names:
        rel_structure_npz = Path("structure_features") / f"{name}.npz"
        structure_path = dataset_root / rel_structure_npz
        if not structure_path.exists():
            raise FileNotFoundError(
                "field-aligned structure pack is missing; run "
                "scripts/convert_outputs_merged_td_to_csv_npz.py first: "
                f"{structure_path}"
            )
        with np.load(structure_path, allow_pickle=True) as existing:
            structure_payload = {key: np.asarray(existing[key]) for key in existing.files}
        for required_key in ("mask_plasma", "valid_field_mask", "outside_mask"):
            if required_key not in structure_payload:
                raise ValueError(f"structure pack missing key={required_key!r}: {structure_path}")
            if tuple(structure_payload[required_key].shape) != tuple(mask_plasma.shape):
                raise ValueError(
                    f"structure pack shape mismatch for base={name!r} key={required_key}: "
                    f"expected={mask_plasma.shape}, got={structure_payload[required_key].shape}"
                )
        boundary_stack = boundary_stacks[name].astype(np.float32)
        structure_payload["mask_coil"] = np.maximum.reduce(boundary_stack, axis=0).astype(np.float32)
        structure_payload["part_mask_stack"] = boundary_stack
        structure_payload["part_ids"] = np.asarray(slot_ids, dtype=object)
        structure_payload["geometric_entity_ids"] = np.asarray(entity_slots, dtype=np.int64)
        structure_payload["structure_source"] = np.asarray("simulation_export_mask+mphtxt_edg_connectivity")
        np.savez_compressed(dataset_root / rel_structure_npz, **structure_payload)
        structure_npz_by_alternative[name] = rel_structure_npz.as_posix()

    manifest_path = geometry_root / "parts_manifest.json"
    manifest = {
        "version": "v1",
        "generator": "external_tools/structure_converter/convert_mphtxt_to_parametric_parts.py",
        "part_semantics": "alternatives",
        "selection_key": "base_name",
        "default_alternative_id": str(base_names[0]),
        "plasma_mode": "preserve",
        "geometry_fidelity": "mphtxt_edg_connectivity",
        "approximate": False,
        "part_ids": [str(v) for v in base_names],
        "entity_slot_ids": slot_ids,
        "geometric_entity_ids": entity_slots,
        "param_specs": part_param_specs,
        "structure_npz_by_alternative": structure_npz_by_alternative,
        "alternative_conditions": {
            name: ({"Td": float(td_by_base[name])} if name in td_by_base else {})
            for name in base_names
        },
        "source": {
            "conditions_with_structure_csv": str(conditions_with_structure_csv),
            "structure_file_index_csv": str(structure_index_csv),
            "source_root": str(source_root),
            "mphtxt_files": {name: str(by_base[name]) for name in base_names},
            "mphtxt_edge_entity_counts": {
                name: int(len(edge_mesh_by_base[name][1])) for name in base_names
            },
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
        "part_semantics": "alternatives",
        "geometry_fidelity": "mphtxt_edg_connectivity",
        "approximate": False,
        "structure_feature_root": str(structure_feature_root),
        "structure_npz_by_alternative": structure_npz_by_alternative,
        "n_parts": int(stack.shape[0]),
        "shape": [int(v) for v in stack.shape],
        "part_ids": [str(v) for v in base_names],
        "entity_slot_ids": slot_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="data/outputs_merged_td_all_success_pa_ext0520",
        help="Root directory that contains conditions_with_structure.csv and structure/*.mphtxt",
    )
    parser.add_argument(
        "--structure-index-csv",
        default="data/outputs_merged_td_all_success_pa_ext0520/structure_file_index.csv",
        help="Path to structure_file_index.csv",
    )
    parser.add_argument(
        "--conditions-with-structure-csv",
        default="data/outputs_merged_td_all_success_pa_ext0520/conditions_with_structure.csv",
        help="Path to conditions_with_structure.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/outputs_merged_td_csv_periodic_ext0520_v2",
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
