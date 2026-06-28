"""Dataset loading entrypoint for synthetic and csv_npz inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.synthetic_data import SyntheticDataset, build_synthetic_dataset


def _resolve_path(raw: str | Path, *, run_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate_run = run_dir / path
    if candidate_run.exists():
        return candidate_run
    return path.resolve()


def _as_hw(arr: np.ndarray, *, key: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"{key} must be [H,W] (or [1,H,W]), got {out.shape}")
    return out.astype(np.float32)


def load_csv_npz_dataset(ds_cfg: dict[str, Any], run_dir: str | Path) -> SyntheticDataset:
    """Load a minimal 2D dataset from index CSV + per-case fields npz."""

    root_raw = ds_cfg.get("root")
    if root_raw is None:
        raise ValueError("dataset.type=csv_npz requires dataset.root")
    root = _resolve_path(root_raw, run_dir=Path(run_dir))
    index_rel = ds_cfg.get("index_csv", "index.csv")
    index_csv = root / str(index_rel)
    if not index_csv.exists():
        raise FileNotFoundError(f"csv_npz index not found: {index_csv}")

    cond_columns = list(ds_cfg.get("cond_columns", []))
    if len(cond_columns) == 0:
        raise ValueError("dataset.type=csv_npz requires non-empty cond_columns")
    legacy_keys = ["output_vars", "output_key_map", "output_value_transform"]
    for key in legacy_keys:
        if key in ds_cfg:
            raise ValueError(
                f"dataset.{key} is removed from mainline. "
                "Use dataset.targets=[{id, source_key, units, dtype, value_transform, role, positive, field_family}]"
            )
    targets_raw = ds_cfg.get("targets")
    if not isinstance(targets_raw, list) or len(targets_raw) == 0:
        raise ValueError(
            "dataset.type=csv_npz requires non-empty dataset.targets "
            "with entries: {id, source_key?, units?, dtype?, value_transform?, role?, positive?, field_family?}"
        )
    targets: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    for i, entry in enumerate(targets_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset.targets[{i}] must be an object")
        target_id = str(entry.get("id", "")).strip()
        if target_id == "":
            raise ValueError(f"dataset.targets[{i}].id is required")
        if target_id in seen_target_ids:
            raise ValueError(f"dataset.targets has duplicated id: {target_id}")
        seen_target_ids.add(target_id)
        source_key = str(entry.get("source_key", target_id)).strip()
        if source_key == "":
            raise ValueError(f"dataset.targets[{i}].source_key must not be empty")
        value_transform = str(entry.get("value_transform", "identity")).strip().lower()
        if value_transform != "identity":
            raise ValueError(
                f"dataset.targets[{i}].value_transform={value_transform!r} is removed from mainline. "
                "Use linear physical fields with value_transform=identity."
            )
        target_meta: dict[str, Any] = {
            "id": target_id,
            "source_key": source_key,
            "value_transform": value_transform,
        }
        for key in ("units", "dtype", "role", "field_family", "default_region"):
            if key in entry and entry.get(key) is not None:
                value = str(entry.get(key)).strip()
                if value:
                    target_meta[key] = value
        if "positive" in entry:
            raw_positive = entry.get("positive")
            if isinstance(raw_positive, bool):
                target_meta["positive"] = raw_positive
            elif isinstance(raw_positive, str) and raw_positive.strip().lower() in {"true", "1", "yes", "on"}:
                target_meta["positive"] = True
            elif isinstance(raw_positive, str) and raw_positive.strip().lower() in {"false", "0", "no", "off"}:
                target_meta["positive"] = False
            else:
                raise ValueError(f"dataset.targets[{i}].positive must be boolean-compatible")
        targets.append(target_meta)
    axis_column = str(ds_cfg.get("axis_column", "axis"))
    fields_col = str(ds_cfg.get("fields_npz_column", "fields_npz"))
    structure_col_raw = ds_cfg.get("structure_npz_column")
    structure_col = str(structure_col_raw) if structure_col_raw is not None else None
    case_id_col = str(ds_cfg.get("case_id_column", "case_id"))
    base_case_id_col_raw = ds_cfg.get("base_case_id_column")
    split_group_col_raw = ds_cfg.get("split_group_column")
    base_case_id_col = str(base_case_id_col_raw) if base_case_id_col_raw is not None else None
    split_group_col = str(split_group_col_raw) if split_group_col_raw is not None else base_case_id_col

    geometry_rel = ds_cfg.get("geometry_root", "geometry")
    geometry_path = Path(str(geometry_rel))
    if not geometry_path.is_absolute():
        geometry_path = root / geometry_path
    # FixedGeometryProvider expects dataset_root with a "geometry/" child.
    if (geometry_path / "mask_plasma.npy").exists():
        geometry_root = geometry_path.parent
    elif (geometry_path / "geometry" / "mask_plasma.npy").exists():
        geometry_root = geometry_path
    else:
        # Keep default contract for downstream fail-fast checks.
        geometry_root = root

    cases: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    seen_case_ids: set[str] = set()

    with index_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {case_id_col, axis_column, fields_col, *cond_columns}
        if structure_col is not None:
            required.add(structure_col)
        if base_case_id_col is not None:
            required.add(base_case_id_col)
        if split_group_col is not None:
            required.add(split_group_col)
        missing_header = sorted([k for k in required if k not in (reader.fieldnames or [])])
        if missing_header:
            raise ValueError(f"csv_npz index missing required columns: {missing_header}")

        for i, row in enumerate(reader):
            case_id = str(row.get(case_id_col, f"case_{i:05d}"))
            if case_id in seen_case_ids:
                raise ValueError(f"csv_npz index has duplicated case_id: {case_id}")
            seen_case_ids.add(case_id)
            axis = float(row[axis_column])
            cond = {key: float(row[key]) for key in cond_columns}

            npz_path = Path(str(row[fields_col]))
            if not npz_path.is_absolute():
                npz_path = root / npz_path
            if not npz_path.exists():
                raise FileNotFoundError(f"fields npz not found for case={case_id}: {npz_path}")

            with np.load(npz_path) as data:
                target_ids = [t["id"] for t in targets]
                resolved_sources = {t["id"]: t["source_key"] for t in targets}
                transforms = {t["id"]: t["value_transform"] for t in targets}
                missing_keys = [resolved_sources[k] for k in target_ids if resolved_sources[k] not in data.files]
                if missing_keys:
                    raise ValueError(f"fields npz missing keys for case={case_id}: {missing_keys}")
                y: dict[str, np.ndarray] = {}
                for name in target_ids:
                    source_key = resolved_sources[name]
                    arr = _as_hw(data[source_key], key=source_key)
                    transform = str(transforms.get(name, "identity")).strip().lower()
                    if transform != "identity":
                        raise ValueError(
                            "dataset.targets[].value_transform supports only identity in mainline; "
                            f"got {transform!r} for target id={name}"
                        )
                    y[name] = arr

            first_key = target_ids[0]
            shape = tuple(int(v) for v in y[first_key].shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    f"csv_npz requires fixed 2D grid shape across cases: expected={expected_shape}, got={shape}"
                )

            case_payload: dict[str, Any] = {
                "case_id": case_id,
                "cond": cond,
                "axis": axis,
                "y": y,
            }
            if structure_col is not None:
                structure_path = Path(str(row[structure_col]))
                if not structure_path.is_absolute():
                    structure_path = root / structure_path
                if not structure_path.exists():
                    raise FileNotFoundError(f"structure npz not found for case={case_id}: {structure_path}")
                case_payload["structure_npz"] = str(structure_path)
            if base_case_id_col is not None:
                case_payload["base_case_id"] = str(row[base_case_id_col])
            if split_group_col is not None:
                case_payload["split_group"] = str(row[split_group_col])
            cases.append(case_payload)

    if expected_shape is None:
        raise ValueError(f"csv_npz index has no rows: {index_csv}")

    return SyntheticDataset(
        cases=cases,
        cond_order=[str(k) for k in cond_columns],
        geometry_root=geometry_root,
        shape=expected_shape,
        structure_root=root,
        target_metadata=targets,
    )


def load_dataset(cfg: dict[str, Any], run_dir: str | Path) -> SyntheticDataset:
    """Load dataset from config while preserving existing SyntheticDataset contract."""

    ds_cfg = dict(cfg.get("dataset", {}))
    ds_type = str(ds_cfg.get("type", "synthetic"))
    if ds_type == "synthetic":
        return build_synthetic_dataset(ds_cfg, run_dir)
    if ds_type == "csv_npz":
        return load_csv_npz_dataset(ds_cfg, run_dir=run_dir)
    raise ValueError(f"Unsupported dataset.type: {ds_type}")


__all__ = ["load_dataset", "load_csv_npz_dataset"]
