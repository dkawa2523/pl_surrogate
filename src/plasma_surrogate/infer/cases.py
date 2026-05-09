"""Lightweight inference-case parsing for CLI inference runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


@dataclass(frozen=True)
class InferenceCase:
    case_id: str
    cond: dict[str, Any]
    geom: dict[str, Any]
    axis: dict[str, Any]


def default_axis(axis_schema: AxisSchema) -> dict[str, Any]:
    return {"mode": str(axis_schema.mode), "value": 0.0}


def parse_single_case(
    single_cfg: dict[str, Any] | None,
    *,
    cond_schema: CondSchema,
    axis_schema: AxisSchema,
) -> InferenceCase:
    cfg = dict(single_cfg or {})
    cond = cfg.get("cond", {k: 0.5 for k in cond_schema.order})
    return InferenceCase(
        case_id=str(cfg.get("case_id", "single")),
        cond=_validated_cond(cond, cond_schema=cond_schema, label="inference.single.cond"),
        geom=_validated_geom(cfg.get("geom", {"geom_id": "default"}), label="inference.single.geom"),
        axis=_validated_axis(cfg.get("axis"), axis_schema=axis_schema),
    )


def parse_batch_cases(
    batch_cfg: dict[str, Any] | None,
    *,
    cond_schema: CondSchema,
    axis_schema: AxisSchema,
    config_dir: str | Path | None = None,
) -> list[InferenceCase]:
    cfg = dict(batch_cfg or {})
    cases: list[InferenceCase] = []

    if "cases" in cfg:
        raw_cases = cfg.get("cases") or []
        if not isinstance(raw_cases, list):
            raise ValueError("inference.batch.cases must be a list")
        for idx, raw in enumerate(raw_cases):
            if not isinstance(raw, dict):
                raise ValueError(f"inference.batch.cases[{idx}] must be a mapping")
            cases.append(_case_from_mapping(raw, idx=idx, cond_schema=cond_schema, axis_schema=axis_schema))

    if "csv" in cfg and cfg.get("csv"):
        cases.extend(
            _cases_from_csv(
                dict(cfg.get("csv") or {}),
                cond_schema=cond_schema,
                axis_schema=axis_schema,
                config_dir=config_dir,
                offset=len(cases),
            )
        )

    if cases:
        return cases

    conds = cfg.get("conds", [{k: 0.2 for k in cond_schema.order}, {k: 0.8 for k in cond_schema.order}])
    if not isinstance(conds, list):
        raise ValueError("inference.batch.conds must be a list")
    axis = _validated_axis(cfg.get("axis"), axis_schema=axis_schema)
    geom = _validated_geom(cfg.get("geom", {"geom_id": "default"}), label="inference.batch.geom")
    out = []
    for idx, cond in enumerate(conds):
        out.append(
            InferenceCase(
                case_id=f"batch_{idx:06d}",
                cond=_validated_cond(cond, cond_schema=cond_schema, label=f"inference.batch.conds[{idx}]"),
                geom=dict(geom),
                axis=dict(axis),
            )
        )
    return out


def _case_from_mapping(
    raw: dict[str, Any],
    *,
    idx: int,
    cond_schema: CondSchema,
    axis_schema: AxisSchema,
) -> InferenceCase:
    return InferenceCase(
        case_id=str(raw.get("case_id", f"case_{idx:06d}")),
        cond=_validated_cond(raw.get("cond", {}), cond_schema=cond_schema, label=f"inference.batch.cases[{idx}].cond"),
        geom=_validated_geom(raw.get("geom", {"geom_id": "default"}), label=f"inference.batch.cases[{idx}].geom"),
        axis=_validated_axis(raw.get("axis"), axis_schema=axis_schema),
    )


def _cases_from_csv(
    csv_cfg: dict[str, Any],
    *,
    cond_schema: CondSchema,
    axis_schema: AxisSchema,
    config_dir: str | Path | None,
    offset: int,
) -> list[InferenceCase]:
    path_raw = csv_cfg.get("path")
    if not path_raw:
        raise ValueError("inference.batch.csv.path is required")
    path = Path(str(path_raw))
    if not path.is_absolute() and config_dir is not None:
        path = Path(config_dir) / path
    cond_columns = [str(v) for v in list(csv_cfg.get("cond_columns") or []) if str(v).strip()]
    if not cond_columns:
        raise ValueError("inference.batch.csv.cond_columns must be non-empty")
    case_id_column = str(csv_cfg.get("case_id_column", "")).strip() or None
    geom_id_column = str(csv_cfg.get("geom_id_column", "")).strip() or None

    cases: list[InferenceCase] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"inference.batch.csv requires a header row: {path}")
        missing = [col for col in cond_columns if col not in reader.fieldnames]
        for optional_col in [case_id_column, geom_id_column]:
            if optional_col and optional_col not in reader.fieldnames:
                missing.append(optional_col)
        if missing:
            raise ValueError(f"inference.batch.csv missing columns {sorted(set(missing))}: {path}")
        for row_idx, row in enumerate(reader):
            cond = {col: _parse_csv_value(row.get(col, "")) for col in cond_columns}
            geom = {"geom_id": str(row.get(geom_id_column, "default") or "default")} if geom_id_column else {"geom_id": "default"}
            case_id = str(row.get(case_id_column, "")).strip() if case_id_column else ""
            cases.append(
                InferenceCase(
                    case_id=case_id or f"csv_{offset + row_idx:06d}",
                    cond=_validated_cond(cond, cond_schema=cond_schema, label=f"inference.batch.csv row {row_idx}"),
                    geom=geom,
                    axis=default_axis(axis_schema),
                )
            )
    return cases


def _validated_cond(raw: Any, *, cond_schema: CondSchema, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    cond = dict(raw)
    for key in cond_schema.order:
        try:
            cond_schema._lookup(cond, key)
        except Exception as exc:
            raise ValueError(f"{label} missing or invalid required condition {key!r}") from exc
    return cond


def _validated_geom(raw: Any, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {"geom_id": "default"}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    geom = dict(raw)
    geom.setdefault("geom_id", "default")
    return geom


def _validated_axis(raw: Any, *, axis_schema: AxisSchema) -> dict[str, Any]:
    if raw is None:
        return default_axis(axis_schema)
    if not isinstance(raw, dict):
        raise ValueError("inference axis must be a mapping")
    axis = dict(raw)
    axis.setdefault("mode", axis_schema.mode)
    axis.setdefault("value", 0.0)
    return axis


def _parse_csv_value(value: Any) -> Any:
    text = "" if value is None else str(value).strip()
    if text == "":
        return text
    try:
        return float(text)
    except ValueError:
        return text


__all__ = ["InferenceCase", "default_axis", "parse_batch_cases", "parse_single_case"]
