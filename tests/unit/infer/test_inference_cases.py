from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.infer.cases import parse_batch_cases, parse_single_case
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def test_parse_inline_batch_cases() -> None:
    cases = parse_batch_cases(
        {
            "cases": [
                {
                    "case_id": "c001",
                    "cond": {"c0": 0.1, "c1": 0.2},
                    "geom": {"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}},
                    "axis": {"mode": "steady", "value": 0.0},
                }
            ]
        },
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
    )

    assert len(cases) == 1
    assert cases[0].case_id == "c001"
    assert cases[0].cond == {"c0": 0.1, "c1": 0.2}
    assert cases[0].geom == {"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}}
    assert cases[0].axis == {"mode": "steady", "value": 0.0}


def test_parse_csv_batch_cases(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "case_id,geom_id,c0,c1\n"
        "a,default,0.1,0.2\n"
        "b,alt,0.3,0.4\n",
        encoding="utf-8",
    )

    cases = parse_batch_cases(
        {
            "csv": {
                "path": "candidates.csv",
                "cond_columns": ["c0", "c1"],
                "case_id_column": "case_id",
                "geom_id_column": "geom_id",
            }
        },
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="time"),
        config_dir=tmp_path,
    )

    assert [case.case_id for case in cases] == ["a", "b"]
    assert cases[0].cond == {"c0": 0.1, "c1": 0.2}
    assert cases[1].geom == {"geom_id": "alt"}
    assert cases[0].axis == {"mode": "time", "value": 0.0}


def test_parse_single_case_defaults_axis() -> None:
    case = parse_single_case(
        {"cond": {"c0": 0.5}},
        cond_schema=CondSchema(order=["c0"]),
        axis_schema=AxisSchema(mode="time"),
    )

    assert case.case_id == "single"
    assert case.axis == {"mode": "time", "value": 0.0}


def test_parse_batch_case_rejects_missing_required_cond() -> None:
    with pytest.raises(ValueError, match="missing or invalid required condition 'c1'"):
        parse_batch_cases(
            {"cases": [{"cond": {"c0": 0.1}}]},
            cond_schema=CondSchema(order=["c0", "c1"]),
            axis_schema=AxisSchema(mode="steady"),
        )
