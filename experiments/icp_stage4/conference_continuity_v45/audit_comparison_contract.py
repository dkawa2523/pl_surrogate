#!/usr/bin/env python
"""Machine-check the frozen data and equal-training comparison contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
CONFIGS = ROOT / "configs/experimental/icp_stage4/conference_continuity_v45/final"
REPORT = ROOT / "reports/icp_conference_materials/conference_continuity_v45"
MODELS = (
    "conference_dimension",
    "explicit_dimension",
    "conference_sdf",
    "sdf_vacuum",
    "sdf_vacuum_structure",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _training_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    uno = {
        key: value for key, value in cfg["train"]["u_no"].items()
        if key not in {"input_features", "initial_weights_path"}
    }
    return {
        "seed": cfg["seed"],
        "split": cfg["split"],
        "eval_protocol": cfg["eval_protocol"],
        "target_columns": cfg["dataset"].get("target_columns", cfg["dataset"].get("targets")),
        "uno_except_input_and_warmstart": uno,
    }


def main() -> int:
    rows = _rows(DATASET / "index.csv")
    explicit = _rows(DATASET / "index_explicit_dimension.csv")
    membership = json.loads((DATASET / "split_membership.json").read_text(encoding="utf-8"))
    sets = {name: set(map(str, membership[name])) for name in ("train", "val", "test")}
    disjoint = not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    all_members = sets["train"] | sets["val"] | sets["test"]
    case_ids = {row["case_id"] for row in rows}
    metadata = {row["case_id"]: row for row in _rows(DATASET / "design_metadata_v43.csv")}
    family_counts = {
        family: sum(
            case_id in sets["test"] and metadata.get(case_id, {}).get("layout_kind") == family
            for case_id in metadata
        )
        for family in ("regular_anchor", "unknown_gap_topology", "unseen_rank_height_size", "coupled_transform")
    }

    cfgs = {
        model: yaml.safe_load((CONFIGS / model / "seed_1237.yaml").read_text(encoding="utf-8"))["benchmark"]
        for model in MODELS
    }
    contracts = {model: _training_contract(cfg) for model, cfg in cfgs.items()}
    reference = contracts[MODELS[0]]
    training_contract_equal = {model: contract == reference for model, contract in contracts.items()}
    input_channels = {
        model: list(cfg["train"]["u_no"]["input_features"]["features"])
        for model, cfg in cfgs.items()
    }
    cond_columns = {model: list(cfg["dataset"]["cond_columns"]) for model, cfg in cfgs.items()}
    forbidden_response_channels = {
        model: [name for name in channels if name.lower().startswith("comsol_") or name.lower() in {"br", "bz", "bmag"}]
        for model, channels in input_channels.items()
    }
    vacuum_cases = [row for row in rows if row["source_dataset"] == "v43_structure"][:20]
    vacuum_checks = []
    for row in vacuum_cases:
        with np.load(DATASET / row["structure_npz"], allow_pickle=False) as pack:
            names = ("vacuum_aphi_unit", "vacuum_br_unit", "vacuum_bz_unit", "vacuum_bmag_unit")
            vacuum_checks.append(all(name in pack and np.isfinite(pack[name]).all() for name in names))
    v43_quality = json.loads(
        (ROOT / "data/outputs_icp_v43_structure_600/qa/quality_summary.json").read_text(
            encoding="utf-8"
        )
    )

    result = {
        "status": "pass",
        "case_count": len(rows),
        "unique_case_count": len(case_ids),
        "explicit_index_case_count": len(explicit),
        "split_counts": {name: len(values) for name, values in sets.items()},
        "split_disjoint": disjoint,
        "membership_exactly_matches_index": all_members == case_ids,
        "test_family_counts": family_counts,
        "training_contract_equal_except_input_and_warmstart": training_contract_equal,
        "input_channels": input_channels,
        "condition_columns": cond_columns,
        "forbidden_response_channels": forbidden_response_channels,
        "sampled_v43_vacuum_fields_finite": all(vacuum_checks),
        "sampled_v43_vacuum_case_count": len(vacuum_checks),
        "v43_source_quality": {
            "trust_status": v43_quality.get("trust_status"),
            "requested_case_count": v43_quality.get("requested_case_count"),
            "accepted_case_count": v43_quality.get("accepted_case_count"),
            "critical_failure_count": v43_quality.get("critical_failure_count"),
            "branch_audit_failure_case_ids": v43_quality.get("branch_audit_failure_case_ids"),
            "family_split_leaks": v43_quality.get("family_split_leaks"),
        },
        "hashes": {
            "index.csv": _sha256(DATASET / "index.csv"),
            "index_explicit_dimension.csv": _sha256(DATASET / "index_explicit_dimension.csv"),
            "split_membership.json": _sha256(DATASET / "split_membership.json"),
        },
        "allowed_differences": [
            "dataset.index_csv",
            "dataset.cond_columns",
            "train.u_no.input_features.features",
            "train.u_no.initial_weights_path",
            "runtime.structure.feature_profile",
            "runtime.structure.provider_mode",
            "output paths and descriptive metadata",
        ],
    }
    checks = (
        len(rows) == 957,
        len(case_ids) == 957,
        len(explicit) == 957,
        disjoint,
        all_members == case_ids,
        family_counts == {"regular_anchor": 25, "unknown_gap_topology": 25, "unseen_rank_height_size": 25, "coupled_transform": 25},
        all(training_contract_equal.values()),
        not any(forbidden_response_channels.values()),
        all(vacuum_checks),
        v43_quality.get("trust_status") == "accepted",
        int(v43_quality.get("critical_failure_count", -1)) == 0,
    )
    if not all(checks):
        result["status"] = "fail"
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "comparison_contract_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
