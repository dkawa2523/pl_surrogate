from __future__ import annotations

import json

from plasma_surrogate.core.data_cleaning_audit import run_data_audit, run_synthetic_data_audit


def test_run_synthetic_data_audit_adds_duplicate_and_cond_ranges():
    cases = [
        {
            "case_id": "c0",
            "cond": {"c0": 0.1, "c1": 0.8},
            "axis": 0.0,
            "y": {"log_ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]},
        },
        {
            "case_id": "c1",
            "cond": {"c0": 0.1, "c1": 0.8},
            "axis": 0.0,
            "y": {"log_ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]},
        },
        {
            "case_id": "c2",
            "cond": {"c0": 0.9, "c1": 0.2},
            "axis": 0.0,
            "y": {"log_ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]},
        },
    ]
    audit = run_synthetic_data_audit(cases=cases, cond_order=["c0", "c1"], axis_mode="steady")
    assert audit["n_cases"] == 3
    assert "missing_counts" in audit
    assert "range_violations" in audit
    assert "axis_hist" in audit
    assert "duplicate_case_keys" in audit
    assert "cond_range_summary" in audit

    assert len(audit["duplicate_case_keys"]) == 1
    duplicate_payload = json.loads(audit["duplicate_case_keys"][0])
    assert duplicate_payload["axis"] == 0.0
    assert duplicate_payload["cond"]["c0"] == 0.1
    assert duplicate_payload["cond"]["c1"] == 0.8

    summary = audit["cond_range_summary"]
    assert summary["c0"]["min"] == 0.1
    assert summary["c0"]["max"] == 0.9
    assert summary["c1"]["min"] == 0.2
    assert summary["c1"]["max"] == 0.8


def test_run_data_audit_has_same_contract_as_synthetic_wrapper():
    cases = [
        {"case_id": "a", "cond": {"c0": 0.1}, "axis": 0.0, "y": {"log_ne": [[0.0]], "Te": [[0.0]], "phi": [[0.0]]}}
    ]
    out = run_data_audit(cases=cases, cond_order=["c0"], axis_mode="steady")
    out_wrap = run_synthetic_data_audit(cases=cases, cond_order=["c0"], axis_mode="steady")
    assert out["n_cases"] == 1
    assert out.keys() == out_wrap.keys()