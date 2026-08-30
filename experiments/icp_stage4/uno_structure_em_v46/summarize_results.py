#!/usr/bin/env python
"""Build auditable comparison tables for the isolated v46 UNO study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46"
RUN_ROOT = ROOT / "runs/icp_uno_structure_em_v46/final"
WEIGHT_META = ROOT / "experiments/icp_stage4/uno_structure_em_v46/configs/initial_weights"
V45_COST = ROOT / "reports/icp_conference_materials/conference_continuity_v45/TRAINING_COST.csv"
SEED = 1237
CANDIDATES = (
    "A_multires",
    "AB_separate_fusion",
    "ABC_adaptive_mix",
    "ABD_shared_coils",
    "ABE_em_shape_amplitude",
    "ABF_structure_delta",
    "ABG_sdf_em_aux",
    "ABH_target_decoders",
    "ALL_combined",
)
MODELS = ("formal_dimension", "formal_sdf", *CANDIDATES)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {
        (row["source"], row["model"], row["layout_kind"], row["metric"]): row for row in rows
    }


def _metric(
    index: dict[tuple[str, str, str, str], dict[str, str]],
    source: str,
    model: str,
    metric: str,
    field: str = "median",
) -> float:
    return float(index[(source, model, "all_unknown", metric)][field])


def _formal_training_cost() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_csv(V45_COST):
        if row["model"] == "Formal Dimension (7 scalars)":
            key = "formal_dimension"
        elif row["model"] == "Formal SDF":
            key = "formal_sdf"
        else:
            continue
        result[key] = {
            "selected_epoch": int(row["selected_epoch"]),
            "training_elapsed_min": float(row["training_elapsed_min"]),
            "standard_test_primary_metric": float(row["standard_test_primary_metric"]),
            "parameter_count": "",
            "warm_start_fraction": "",
        }
    return result


def _candidate_training_cost(model: str) -> dict[str, Any]:
    run = RUN_ROOT / model / f"seed_{SEED}"
    progress = json.loads(
        (
            run
            / "models/u_no/eval_protocol/interp/train/scalars/progress_latest.json"
        ).read_text(encoding="utf-8")
    )
    leaderboard = _read_csv(run / "leaderboard.csv")[0]
    warm = json.loads((WEIGHT_META / f"{model}_seed{SEED}.json").read_text(encoding="utf-8"))
    return {
        "selected_epoch": int(float(leaderboard["validation_selected_epoch"])),
        "training_elapsed_min": float(progress["elapsed_seconds"]) / 60.0,
        "standard_test_primary_metric": float(leaderboard["primary_metric_value"]),
        "parameter_count": int(warm["target_parameter_elements"]),
        "warm_start_fraction": float(warm["copied_fraction_lower_bound"]),
    }


def _paired_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(int(repeats), values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _paired_effects(case_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics = ("ni_rel_l2_pct", "bohm_profile_rel_l2_pct")
    formal = {
        (row["case_id"], metric): float(row[metric])
        for row in case_rows
        if row["model"] == "formal_sdf"
        for metric in metrics
    }
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for model in CANDIDATES:
        selected = [row for row in case_rows if row["model"] == model]
        for metric in metrics:
            differences = np.asarray(
                [float(row[metric]) - formal[(row["case_id"], metric)] for row in selected],
                dtype=np.float64,
            )
            low, high = _paired_ci(differences, rng)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_paired_cases": int(differences.size),
                    "median_error_difference_pp_vs_formal_sdf": float(np.median(differences)),
                    "bootstrap_ci_low_pp": low,
                    "bootstrap_ci_high_pp": high,
                    "fraction_cases_better_than_formal_sdf": float(np.mean(differences < 0.0)),
                    "direction": "candidate_better" if high < 0.0 else "candidate_worse" if low > 0.0 else "inconclusive",
                }
            )
    return rows


def main() -> int:
    summary_rows = _read_csv(REPORT / "summary_metrics.csv")
    case_rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    index = _summary_index(summary_rows)
    training = _formal_training_cost()
    training.update({model: _candidate_training_cost(model) for model in CANDIDATES})
    formal_ni = _metric(index, "case", "formal_sdf", "ni_rel_l2_pct")
    formal_bohm = _metric(index, "case", "formal_sdf", "bohm_profile_rel_l2_pct")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        ni = _metric(index, "case", model, "ni_rel_l2_pct")
        bohm = _metric(index, "case", model, "bohm_profile_rel_l2_pct")
        rows.append(
            {
                "model": model,
                "role": "formal_reference" if model.startswith("formal_") else "isolated_candidate",
                "unknown_ni_rel_l2_median_pct": ni,
                "unknown_ni_rel_l2_p90_pct": _metric(index, "case", model, "ni_rel_l2_pct", "p90"),
                "unknown_bohm_rel_l2_median_pct": bohm,
                "unknown_bohm_rel_l2_p90_pct": _metric(
                    index, "case", model, "bohm_profile_rel_l2_pct", "p90"
                ),
                "structural_delta_ni_rel_l2_median_pct": _metric(
                    index, "response", model, "ni_delta_rel_l2_pct"
                ),
                "structural_delta_bohm_rel_l2_median_pct": _metric(
                    index, "response", model, "bohm_delta_rel_l2_pct"
                ),
                "ni_improvement_vs_formal_sdf_pct": 100.0 * (formal_ni - ni) / formal_ni,
                "bohm_improvement_vs_formal_sdf_pct": 100.0 * (formal_bohm - bohm) / formal_bohm,
                **training[model],
            }
        )
    _write_csv(REPORT / "model_summary.csv", rows)
    paired = _paired_effects(case_rows)
    _write_csv(REPORT / "paired_effects_vs_formal_sdf.csv", paired)

    candidate_rows = [row for row in rows if row["role"] == "isolated_candidate"]
    best_ni = min(candidate_rows, key=lambda row: float(row["unknown_ni_rel_l2_median_pct"]))
    best_bohm = min(candidate_rows, key=lambda row: float(row["unknown_bohm_rel_l2_median_pct"]))
    joint_winners = [
        row["model"]
        for row in candidate_rows
        if float(row["unknown_ni_rel_l2_median_pct"]) < formal_ni
        and float(row["unknown_bohm_rel_l2_median_pct"]) < formal_bohm
    ]
    decision = {
        "formal_model_status": "preserved; no automatic promotion from this experimental study",
        "fixed_evaluation_contract": "75 G4 unknown structures, 25 cases per family, seed 1237",
        "best_candidate_unknown_ni": best_ni["model"],
        "best_candidate_unknown_bohm": best_bohm["model"],
        "candidates_beating_formal_sdf_on_both_medians": joint_winners,
        "promotion_rule": (
            "A candidate is only a promotion candidate if it improves both unknown ni and Bohm medians, "
            "has no family-level regression larger than the declared tolerance, and paired bootstrap results "
            "do not indicate a clear regression. This study does not overwrite the formal model."
        ),
    }
    (REPORT / "decision_summary.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
