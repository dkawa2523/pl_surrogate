#!/usr/bin/env python
"""Validate the completed isolated v46 evaluation before reporting it."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46"
RUN_ROOT = ROOT / "runs/icp_uno_structure_em_v46/final"
MODELS = (
    "formal_dimension", "formal_sdf", "A_multires", "AB_separate_fusion",
    "ABC_adaptive_mix", "ABD_shared_coils", "ABE_em_shape_amplitude",
    "ABF_structure_delta", "ABG_sdf_em_aux", "ABH_target_decoders", "ALL_combined",
)
CANDIDATES = MODELS[2:]
FAMILIES = ("unknown_gap_topology", "unseen_rank_height_size", "coupled_transform")
METRICS = (
    "ne_rel_l2_pct", "ni_rel_l2_pct", "Te_rel_l2_pct", "phi_rel_l2_pct",
    "bohm_profile_rel_l2_pct", "bohm_uniformity_truth_pct",
    "bohm_uniformity_prediction_pct", "bohm_uniformity_abs_error_pp",
)
REPRESENTATIVE_CASES = (
    "v43_te_n4_variant_1__center", "v43_te_n4_variant_2__center",
    "v43_te_n4_variant_3__center",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    issues: list[str] = []
    case_rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    response_rows = _read_csv(REPORT / "structural_response_metrics_unknown75.csv")
    model_summary = _read_csv(REPORT / "model_summary.csv")
    paired = _read_csv(REPORT / "paired_effects_vs_formal_sdf.csv")

    pairs = [(row["model"], row["case_id"]) for row in case_rows]
    if len(pairs) != len(set(pairs)):
        issues.append("duplicate model/case rows in case_metrics_unknown75.csv")
    response_pairs = [(row["model"], row["case_id"]) for row in response_rows]
    if len(response_pairs) != len(set(response_pairs)):
        issues.append("duplicate model/case rows in structural_response_metrics_unknown75.csv")

    coverage: dict[str, dict[str, int]] = {}
    for model in MODELS:
        selected = [row for row in case_rows if row["model"] == model]
        selected_response = [row for row in response_rows if row["model"] == model]
        family_counts = {family: sum(row["layout_kind"] == family for row in selected) for family in FAMILIES}
        coverage[model] = {"unknown_cases": len(selected), "response_rows": len(selected_response), **family_counts}
        if len(selected) != 75 or len(selected_response) != 75:
            issues.append(f"{model}: expected 75 case and response rows")
        if any(count != 25 for count in family_counts.values()):
            issues.append(f"{model}: expected 25 cases in every unknown family")
        for row in selected:
            for metric in METRICS:
                if not np.isfinite(float(row[metric])):
                    issues.append(f"{model}/{row['case_id']}: non-finite {metric}")

    if {row["model"] for row in model_summary} != set(MODELS):
        issues.append("model_summary.csv does not contain exactly the 11 declared models")
    if len(paired) != len(CANDIDATES) * 2:
        issues.append("paired_effects_vs_formal_sdf.csv should contain two metrics per candidate")
    if any(int(row["n_paired_cases"]) != 75 for row in paired):
        issues.append("paired bootstrap rows do not all use 75 matched cases")
    for candidate in CANDIDATES:
        if not (RUN_ROOT / candidate / "seed_1237/leaderboard.csv").is_file():
            issues.append(f"missing completed leaderboard for {candidate}")

    pack_path = REPORT / "representative_predictions.npz"
    with np.load(pack_path, allow_pickle=False) as pack:
        missing = []
        for model in MODELS:
            for case_id in REPRESENTATIVE_CASES:
                for suffix in ("ni_prediction", "bohm_profile_prediction"):
                    key = f"{model}__{case_id}__{suffix}"
                    if key not in pack:
                        missing.append(key)
        if missing:
            issues.append(f"representative prediction pack missing {len(missing)} required keys")

    formal_audit_path = REPORT / "formal_preservation_audit.json"
    if not formal_audit_path.is_file():
        issues.append("formal preservation audit has not been run")
        formal_unchanged = False
    else:
        formal_unchanged = bool(json.loads(formal_audit_path.read_text(encoding="utf-8"))["formal_models_unchanged"])
        if not formal_unchanged:
            issues.append("formal checkpoint hash mismatch")

    result = {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "declared_model_count": len(MODELS),
        "expected_rows": len(MODELS) * 75,
        "actual_case_rows": len(case_rows),
        "actual_response_rows": len(response_rows),
        "coverage": coverage,
        "formal_models_unchanged": formal_unchanged,
        "checks": [
            "unique model/case keys", "75 matched cases per model", "25 cases per family",
            "finite physical metrics", "complete paired bootstrap table",
            "complete representative field pack", "candidate leaderboards present",
            "formal checkpoint hashes unchanged",
        ],
    }
    (REPORT / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
