#!/usr/bin/env python
"""Build the canonical portable-report artifact for ICP v45 results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/conference_continuity_v45"
RUN_ROOT = ROOT / "runs/icp_conference_continuity_v45/final"
MODELS = (
    "conference_dimension",
    "explicit_dimension",
    "conference_sdf",
    "sdf_vacuum",
    "sdf_vacuum_structure",
)
LABELS = {
    "conference_dimension": "Formal Dimension (7 scalars)",
    "explicit_dimension": "Explicit Dimension (all coils)",
    "conference_sdf": "Formal SDF",
    "sdf_vacuum": "SDF + vacuum magnetic field",
    "sdf_vacuum_structure": "SDF + magnetic + structural maps",
}
FAMILY_LABELS = {
    "unknown_gap_topology": "Unknown spacing",
    "unseen_rank_height_size": "Unknown height + size",
    "coupled_transform": "Coupled spacing/height/size",
    "all_unknown": "All unknown combinations",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric(summary: list[dict[str, str]], model: str, family: str, metric: str, field: str = "median") -> float:
    matches = [
        row for row in summary
        if row["model"] == model and row["layout_kind"] == family and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"metric lookup expected one row: {model}, {family}, {metric}")
    return float(matches[0][field])


def _training_row(model: str) -> dict[str, Any]:
    run = RUN_ROOT / model / "seed_1237"
    leaderboard = _read_csv(run / "leaderboard.csv")[0]
    progress_path = run / "models/u_no/eval_protocol/interp/train/scalars/progress_latest.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    pack = run / "preprocessing/features/case_structure_feature_pack.npz"
    return {
        "model": LABELS[model],
        "selected_epoch": int(float(leaderboard.get("validation_selected_epoch", 0) or 0)),
        "training_elapsed_min": round(float(progress.get("elapsed_seconds", 0.0)) / 60.0, 2),
        "standard_test_primary_metric": round(float(leaderboard.get("primary_metric_value", "nan")), 6),
        "input_mode": leaderboard.get("input_mode_effective", ""),
        "case_varying_structure": leaderboard.get("has_case_varying_structure_inputs_effective", ""),
        "feature_profile": leaderboard.get("structure_feature_profile_effective", ""),
        "feature_pack_gb": round(pack.stat().st_size / 1.0e9, 3) if pack.is_file() else 0.0,
    }


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(REPORT / "summary_metrics.csv")
    case_rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    response_rows = _read_csv(REPORT / "structural_response_metrics_unknown75.csv")
    paired_rows = _read_csv(REPORT / "paired_effects_unknown75.csv")

    model_summary: list[dict[str, Any]] = []
    for order, model in enumerate(MODELS, start=1):
        model_summary.append(
            {
                "order": order,
                "model_id": model,
                "model": LABELS[model],
                "ni_median_rel_l2_pct": round(_metric(summary, model, "all_unknown", "ni_rel_l2_pct"), 4),
                "ni_ci_low_pct": round(_metric(summary, model, "all_unknown", "ni_rel_l2_pct", "bootstrap_median_ci_low"), 4),
                "ni_ci_high_pct": round(_metric(summary, model, "all_unknown", "ni_rel_l2_pct", "bootstrap_median_ci_high"), 4),
                "bohm_median_rel_l2_pct": round(_metric(summary, model, "all_unknown", "bohm_profile_rel_l2_pct"), 4),
                "ni_response_cosine": round(_metric(summary, model, "all_unknown", "ni_delta_cosine"), 4),
                "ni_response_gain": round(_metric(summary, model, "all_unknown", "ni_delta_gain"), 4),
                "ni_response_error_pct": round(_metric(summary, model, "all_unknown", "ni_delta_rel_l2_pct"), 4),
            }
        )
    _write_csv(REPORT / "MODEL_SUMMARY.csv", model_summary)

    family_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for family in FAMILY_LABELS:
            family_rows.append(
                {
                    "model_id": model,
                    "model": LABELS[model],
                    "family_id": family,
                    "family": FAMILY_LABELS[family],
                    "ni_median_rel_l2_pct": round(_metric(summary, model, family, "ni_rel_l2_pct"), 4),
                    "ni_ci_low_pct": round(_metric(summary, model, family, "ni_rel_l2_pct", "bootstrap_median_ci_low"), 4),
                    "ni_ci_high_pct": round(_metric(summary, model, family, "ni_rel_l2_pct", "bootstrap_median_ci_high"), 4),
                    "bohm_median_rel_l2_pct": round(_metric(summary, model, family, "bohm_profile_rel_l2_pct"), 4),
                    "response_cosine": round(_metric(summary, model, family, "ni_delta_cosine"), 4),
                    "response_gain": round(_metric(summary, model, family, "ni_delta_gain"), 4),
                }
            )
    _write_csv(REPORT / "FAMILY_SUMMARY.csv", family_rows)

    sdf_models = MODELS[2:]
    incremental_rows: list[dict[str, Any]] = []
    case_lookup = {
        (row["model"], row["case_id"]): row for row in case_rows
    }
    for order, model in enumerate(sdf_models, start=1):
        if order == 1:
            ni_win = ""
            bohm_win = ""
        else:
            previous = sdf_models[order - 2]
            ids = sorted({case for candidate, case in case_lookup if candidate == model})
            ni_win = round(
                100.0 * sum(float(case_lookup[(model, case)]["ni_rel_l2_pct"]) < float(case_lookup[(previous, case)]["ni_rel_l2_pct"]) for case in ids) / len(ids),
                2,
            )
            bohm_win = round(
                100.0 * sum(float(case_lookup[(model, case)]["bohm_profile_rel_l2_pct"]) < float(case_lookup[(previous, case)]["bohm_profile_rel_l2_pct"]) for case in ids) / len(ids),
                2,
            )
        incremental_rows.append(
            {
                "order": order,
                "model_id": model,
                "model": LABELS[model],
                "ni_median_rel_l2_pct": round(_metric(summary, model, "all_unknown", "ni_rel_l2_pct"), 4),
                "bohm_median_rel_l2_pct": round(_metric(summary, model, "all_unknown", "bohm_profile_rel_l2_pct"), 4),
                "ni_paired_win_rate_vs_previous_pct": ni_win,
                "bohm_paired_win_rate_vs_previous_pct": bohm_win,
            }
        )
    _write_csv(REPORT / "INCREMENTAL_EFFECT.csv", incremental_rows)

    training_rows = [_training_row(model) for model in MODELS]
    _write_csv(REPORT / "TRAINING_COST.csv", training_rows)

    paired_overall = [row for row in paired_rows if row["layout_kind"] == "all_unknown"]

    contracts = [
        {"order": 1, "model": LABELS[MODELS[0]], "geometry_input": "llcoil, rrc, nncoil, rrce, zzc", "only_change": "Conference continuity baseline", "scientific_role": "Original seven-scalar Dimension baseline"},
        {"order": 2, "model": LABELS[MODELS[1]], "geometry_input": "6 x (active, r, z, width, height)", "only_change": "Full coil rectangles replace five geometry scalars", "scientific_role": "Strong dimension fairness control"},
        {"order": 3, "model": LABELS[MODELS[2]], "geometry_input": "union SDF + count-neutral source mean", "only_change": "Conference SDF representation", "scientific_role": "Conference continuity SDF baseline"},
        {"order": 4, "model": LABELS[MODELS[3]], "geometry_input": "formal SDF + unit-current Aphi/Br/Bz/Bmag", "only_change": "Deterministic vacuum electromagnetic carrier", "scientific_role": "Magnetic-input ablation"},
        {"order": 5, "model": LABELS[MODELS[4]], "geometry_input": "SDF + B + bounded second-nearest/competition/solid proximity", "only_change": "Order-invariant bounded multi-coil summaries", "scientific_role": "Structural-input ablation"},
    ]
    _write_csv(REPORT / "MODEL_CONTRACT.csv", contracts)

    best_ni = min(model_summary, key=lambda row: row["ni_median_rel_l2_pct"])
    best_bohm = min(model_summary, key=lambda row: row["bohm_median_rel_l2_pct"])
    formal_dim = next(row for row in model_summary if row["model_id"] == "conference_dimension")
    explicit_dim = next(row for row in model_summary if row["model_id"] == "explicit_dimension")
    best_sdf = min((row for row in model_summary if row["model_id"] in sdf_models), key=lambda row: row["ni_median_rel_l2_pct"])
    sdf_vs_formal = 100.0 * (formal_dim["ni_median_rel_l2_pct"] - best_sdf["ni_median_rel_l2_pct"]) / formal_dim["ni_median_rel_l2_pct"]
    sdf_vs_explicit = 100.0 * (explicit_dim["ni_median_rel_l2_pct"] - best_sdf["ni_median_rel_l2_pct"]) / explicit_dim["ni_median_rel_l2_pct"]

    def paired(reference: str, candidate: str, metric: str) -> dict[str, str]:
        matches = [
            row for row in paired_overall
            if row["reference_model"] == reference
            and row["candidate_model"] == candidate
            and row["metric"] == metric
        ]
        if len(matches) != 1:
            raise ValueError(f"paired lookup expected one row: {reference}, {candidate}, {metric}")
        return matches[0]

    sdf_vs_explicit_paired = paired("explicit_dimension", "conference_sdf", "ni_rel_l2_pct")
    vacuum_vs_sdf_paired = paired("conference_sdf", "sdf_vacuum", "ni_rel_l2_pct")
    structure_vs_vacuum_paired = paired("sdf_vacuum", "sdf_vacuum_structure", "ni_rel_l2_pct")

    enhanced_candidates = []
    for candidate in ("sdf_vacuum", "sdf_vacuum_structure"):
        candidate_paired = paired("conference_sdf", candidate, "ni_rel_l2_pct")
        candidate_summary = next(row for row in model_summary if row["model_id"] == candidate)
        if (
            float(candidate_paired["paired_median_change_ci_high_pp"]) < 0.0
            and candidate_summary["bohm_median_rel_l2_pct"]
            <= next(row for row in model_summary if row["model_id"] == "conference_sdf")["bohm_median_rel_l2_pct"]
        ):
            enhanced_candidates.append(candidate)
    if enhanced_candidates:
        recommended_model_id = min(
            enhanced_candidates,
            key=lambda candidate: next(row for row in model_summary if row["model_id"] == candidate)["ni_median_rel_l2_pct"],
        )
        recommendation = f"Adopt {LABELS[recommended_model_id]} as the next conference model; it has a paired ion-density improvement whose interval excludes zero without degrading the aggregate Bohm median."
    else:
        recommended_model_id = "conference_sdf"
        recommendation = "Keep Formal SDF as the conference axis.  Vacuum magnetic and structural inputs are useful ablations, but their aggregate Bohm error is not lower and their paired ion-density advantage over Formal SDF is not decisive enough to justify the added data volume."
    if best_sdf["ni_median_rel_l2_pct"] < explicit_dim["ni_median_rel_l2_pct"]:
        claim = "The best SDF variant outperformed both the formal and full-information Dimension controls on median ion-density error in this frozen test."
        boundary = "This supports an accuracy advantage on the tested G4 pool, not unrestricted extrapolation outside the generated coil-design envelope."
    elif best_sdf["ni_median_rel_l2_pct"] < formal_dim["ni_median_rel_l2_pct"]:
        claim = "The best SDF variant outperformed the formal seven-scalar Dimension baseline, but not the full-information Dimension control."
        boundary = "The result supports better handling of geometry omitted by the formal Dimension vector; it does not establish that SDF is intrinsically more accurate than a complete dimension encoding."
    else:
        claim = "No SDF variant outperformed the formal Dimension baseline on median ion-density error in this frozen test."
        boundary = "The present training does not support an accuracy-superiority claim; SDF remains a flexible input interface whose optimization value requires separate evidence."

    headline = [{
        "unknown_cases": 75,
        "best_ni_model": best_ni["model"],
        "best_ni_error_pct": best_ni["ni_median_rel_l2_pct"],
        "best_bohm_model": best_bohm["model"],
        "best_bohm_error_pct": best_bohm["bohm_median_rel_l2_pct"],
        "best_sdf_vs_formal_dim_pct": round(sdf_vs_formal, 2),
        "best_sdf_vs_explicit_dim_pct": round(sdf_vs_explicit, 2),
    }]
    _write_csv(REPORT / "HEADLINE.csv", headline)

    jst = timezone(timedelta(hours=9))
    generated_at = datetime.now(jst).isoformat(timespec="seconds")
    sources = [
        {"id": "headline", "label": "Frozen G4 headline results", "path": "HEADLINE.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Headline metrics computed from audited model and family summaries.", "sql": "SELECT * FROM read_csv_auto('HEADLINE.csv', header=true)", "tables_used": ["HEADLINE.csv"]}},
        {"id": "models", "label": "Model-level unknown-combination metrics", "path": "MODEL_SUMMARY.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Median physical-space errors and structural-response metrics over 75 frozen G4 cases.", "sql": "SELECT * FROM read_csv_auto('MODEL_SUMMARY.csv', header=true)", "tables_used": ["MODEL_SUMMARY.csv"]}},
        {"id": "families", "label": "Family-resolved metrics", "path": "FAMILY_SUMMARY.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Bootstrap family summaries from case_metrics_unknown75.csv and structural_response_metrics_unknown75.csv.", "sql": "SELECT * FROM read_csv_auto('FAMILY_SUMMARY.csv', header=true)", "tables_used": ["FAMILY_SUMMARY.csv"]}},
        {"id": "incremental", "label": "Controlled SDF additions", "path": "INCREMENTAL_EFFECT.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Paired case effects for formal SDF, vacuum magnetic carrier, and bounded structural maps.", "sql": "SELECT * FROM read_csv_auto('INCREMENTAL_EFFECT.csv', header=true)", "tables_used": ["INCREMENTAL_EFFECT.csv"]}},
        {"id": "contract", "label": "Controlled model contract", "path": "MODEL_CONTRACT.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Pre-declared model ladder and the single controlled change at each rung.", "sql": "SELECT * FROM read_csv_auto('MODEL_CONTRACT.csv', header=true)", "tables_used": ["MODEL_CONTRACT.csv"]}},
        {"id": "cost", "label": "Observed training and preprocessing cost", "path": "TRAINING_COST.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Training progress, selected epoch, feature profile and case-pack size from completed run artifacts.", "sql": "SELECT * FROM read_csv_auto('TRAINING_COST.csv', header=true)", "tables_used": ["TRAINING_COST.csv"]}},
        {"id": "paired", "label": "Paired effects on the same unknown cases", "path": "paired_effects_unknown75.csv", "query": {"engine": "DuckDB", "language": "sql", "description": "Case-paired win rates and bootstrap intervals for model differences on the frozen unknown pool.", "sql": "SELECT * FROM read_csv_auto('paired_effects_unknown75.csv', header=true)", "tables_used": ["paired_effects_unknown75.csv"]}},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ICP unknown-coil-combination retraining and evaluation",
            "description": "Controlled Dimension/SDF comparison rooted in the conference UNO, with magnetic and structural ablations on 75 frozen COMSOL G4 cases.",
            "generatedAt": generated_at,
            "cards": [
                {"id": "cases", "description": "Frozen non-regular G4 COMSOL cases.", "dataset": "headline", "sourceId": "headline", "metrics": [{"label": "Unknown cases", "field": "unknown_cases", "format": "number"}]},
                {"id": "best_ni", "description": "Lowest model median plasma ion-density relative L2.", "dataset": "headline", "sourceId": "headline", "metrics": [{"label": "Best nᵢ error", "field": "best_ni_error_pct", "format": "number", "unit": "%"}]},
                {"id": "best_bohm", "description": "Lowest model median wafer Bohm-profile relative L2.", "dataset": "headline", "sourceId": "headline", "metrics": [{"label": "Best Bohm error", "field": "best_bohm_error_pct", "format": "number", "unit": "%"}]},
                {"id": "sdf_vs_dim", "description": "Positive means best SDF has lower nᵢ error than explicit all-coil Dimension.", "dataset": "headline", "sourceId": "headline", "metrics": [{"label": "Best SDF vs explicit Dim.", "field": "best_sdf_vs_explicit_dim_pct", "format": "number", "unit": "%"}]},
            ],
            "charts": [
                {"id": "overall", "title": "Unknown-combination median ion-density error", "subtitle": "75 frozen G4 cases; lower is better.", "type": "bar", "dataset": "model_summary", "sourceId": "models", "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"field": "ni_median_rel_l2_pct", "type": "quantitative", "label": "Median relative L2 (%)"}}, "yAxisTitle": "Median relative L2 (%)", "valueFormat": "number", "layout": "full"},
                {"id": "family", "title": "Ion-density error by unknown-structure family", "subtitle": "Family medians; each family contains 25 matched cases.", "type": "bar", "dataset": "family_nonoverall", "sourceId": "families", "encodings": {"x": {"field": "family", "type": "nominal", "label": "Unknown structure"}, "y": {"field": "ni_median_rel_l2_pct", "type": "quantitative", "label": "Median relative L2 (%)"}, "color": {"field": "model", "type": "nominal", "label": "Model"}}, "yAxisTitle": "Median relative L2 (%)", "valueFormat": "number", "layout": "full"},
                {"id": "incremental", "title": "Controlled SDF input additions", "subtitle": "Same architecture, split, seed, optimizer and 50-epoch budget.", "type": "bar", "dataset": "incremental_effect", "sourceId": "incremental", "encodings": {"x": {"field": "model", "type": "nominal", "label": "SDF rung"}, "y": {"field": "ni_median_rel_l2_pct", "type": "quantitative", "label": "Median nᵢ relative L2 (%)"}}, "yAxisTitle": "Median nᵢ relative L2 (%)", "valueFormat": "number", "layout": "full"},
                {"id": "response", "title": "Structural response fidelity", "subtitle": "Unknown layout minus same-count/same-operation regular anchor. Ideal point is (1,1).", "type": "scatter", "dataset": "family_nonoverall", "sourceId": "families", "encodings": {"x": {"field": "response_cosine", "type": "quantitative", "label": "Response direction cosine"}, "y": {"field": "response_gain", "type": "quantitative", "label": "Response gain"}, "color": {"field": "model", "type": "nominal", "label": "Model"}}, "xAxisTitle": "Direction cosine (ideal 1)", "yAxisTitle": "Gain (ideal 1)", "layout": "full"},
            ],
            "tables": [
                {"id": "contract_table", "title": "Controlled model ladder", "dataset": "model_contract", "sourceId": "contract", "defaultSort": {"field": "order", "direction": "asc"}, "columns": [{"field": "order", "label": "#", "format": "number"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "geometry_input", "label": "Geometry input", "type": "text"}, {"field": "only_change", "label": "Controlled change", "type": "text"}, {"field": "scientific_role", "label": "Role", "type": "text"}]},
                {"id": "model_table", "title": "Overall frozen-test results", "dataset": "model_summary", "sourceId": "models", "defaultSort": {"field": "ni_median_rel_l2_pct", "direction": "asc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "ni_median_rel_l2_pct", "label": "nᵢ L2 (%)", "format": "number"}, {"field": "ni_ci_low_pct", "label": "95% CI low", "format": "number"}, {"field": "ni_ci_high_pct", "label": "95% CI high", "format": "number"}, {"field": "bohm_median_rel_l2_pct", "label": "Bohm L2 (%)", "format": "number"}, {"field": "ni_response_cosine", "label": "Response cosine", "format": "number"}, {"field": "ni_response_gain", "label": "Response gain", "format": "number"}]},
                {"id": "incremental_table", "title": "Magnetic and structural ablation", "dataset": "incremental_effect", "sourceId": "incremental", "defaultSort": {"field": "order", "direction": "asc"}, "columns": [{"field": "order", "label": "#", "format": "number"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "ni_median_rel_l2_pct", "label": "nᵢ L2 (%)", "format": "number"}, {"field": "bohm_median_rel_l2_pct", "label": "Bohm L2 (%)", "format": "number"}, {"field": "ni_paired_win_rate_vs_previous_pct", "label": "nᵢ win rate (%)", "format": "number"}, {"field": "bohm_paired_win_rate_vs_previous_pct", "label": "Bohm win rate (%)", "format": "number"}]},
                {"id": "paired_table", "title": "Case-paired model effects", "dataset": "paired_overall", "sourceId": "paired", "columns": [{"field": "comparison", "label": "Comparison", "type": "text"}, {"field": "metric", "label": "Metric", "type": "text"}, {"field": "paired_win_rate_pct", "label": "Candidate wins (%)", "format": "number"}, {"field": "paired_median_change_pp", "label": "Median change (pp)", "format": "number"}, {"field": "paired_median_change_ci_low_pp", "label": "95% CI low", "format": "number"}, {"field": "paired_median_change_ci_high_pp", "label": "95% CI high", "format": "number"}, {"field": "paired_median_error_ratio", "label": "Median error ratio", "format": "number"}]},
                {"id": "cost_table", "title": "Observed implementation cost", "dataset": "training_cost", "sourceId": "cost", "defaultSort": {"field": "training_elapsed_min", "direction": "asc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "selected_epoch", "label": "Selected epoch", "format": "number"}, {"field": "standard_test_primary_metric", "label": "Standard test primary", "format": "number"}, {"field": "training_elapsed_min", "label": "Train min", "format": "number"}, {"field": "feature_pack_gb", "label": "Case pack GB", "format": "number"}, {"field": "feature_profile", "label": "Feature profile", "type": "text"}]},
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# ICP unknown-coil-combination retraining and evaluation"},
                {"id": "summary", "type": "markdown", "sourceId": "models", "body": f"## Technical summary\n\n{claim} {boundary} The best ion-density model was **{best_ni['model']}** at **{best_ni['ni_median_rel_l2_pct']:.2f}%** median relative L2; the best Bohm-profile model was **{best_bohm['model']}** at **{best_bohm['bohm_median_rel_l2_pct']:.2f}%**."},
                {"id": "headline_strip", "type": "metric-strip", "cardIds": ["cases", "best_ni", "best_bohm", "sdf_vs_dim"]},
                {"id": "scope", "type": "markdown", "sourceId": "contract", "body": "## Scope and controlled comparison\n\nAll five models use the frozen 957-case dataset, the same 685/136/136 membership, UNO trunk, four targets (ne, ni, Te, phi), optimizer, seed 1237, 50-epoch budget, and validation selection. The formal conference checkpoints initialize common weights; all layers remain trainable. The only intended differences are the geometry inputs listed below."},
                {"id": "contract", "type": "table", "tableId": "contract_table", "layout": "full"},
                {"id": "main_intro", "type": "markdown", "sourceId": "models", "body": "## Primary result: physical-space error on frozen unknown combinations\n\nThe primary scalar is plasma-masked physical relative L2 for ion density. It is reported per case and summarized by the median with a 2,000-resample case-bootstrap interval. Static publication figures `00` and `01` in the same folder show the actual COMSOL and prediction distributions and wafer profiles for three representative geometries."},
                {"id": "overall", "type": "chart", "chartId": "overall", "layout": "full"},
                {"id": "model_results", "type": "table", "tableId": "model_table", "layout": "full"},
                {"id": "paired_intro", "type": "markdown", "sourceId": "paired", "body": f"## Paired evidence on the same cases\n\nFormal SDF has lower ion-density error than explicit all-coil Dimension in **{float(sdf_vs_explicit_paired['paired_win_rate_pct']):.1f}%** of the 75 cases.  The paired median change is **{float(sdf_vs_explicit_paired['paired_median_change_pp']):.2f} percentage points**, with a 95% bootstrap interval of **[{float(sdf_vs_explicit_paired['paired_median_change_ci_low_pp']):.2f}, {float(sdf_vs_explicit_paired['paired_median_change_ci_high_pp']):.2f}]**.  Negative means the candidate is better."},
                {"id": "paired_results", "type": "table", "tableId": "paired_table", "layout": "full"},
                {"id": "family_intro", "type": "markdown", "sourceId": "families", "body": "## The three unknown-structure families are not pooled away\n\nUnknown spacing, unseen height-size ranking, and coupled transformations each contribute 25 cases. A model must improve across families rather than win through one easy subset."},
                {"id": "family", "type": "chart", "chartId": "family", "layout": "full"},
                {"id": "ablation_intro", "type": "markdown", "sourceId": "incremental", "body": "## Magnetic and structural effects are paired additions\n\nThe vacuum-field carrier is deterministic from geometry and unit current; no COMSOL response field is used as an input. The structural maps are bounded, order-invariant summaries. Paired win rates show whether a lower median reflects broad case-level improvement."},
                {"id": "incremental", "type": "chart", "chartId": "incremental", "layout": "full"},
                {"id": "incremental_results", "type": "table", "tableId": "incremental_table", "layout": "full"},
                {"id": "response_intro", "type": "markdown", "sourceId": "families", "body": "## Absolute accuracy and geometry sensitivity are different tests\n\nFor every unknown case, the matching regular layout with the same coil count and operating condition is subtracted. Direction cosine measures whether the predicted change points the same way as COMSOL; gain measures whether its magnitude is correct. Both ideally equal 1."},
                {"id": "response", "type": "chart", "chartId": "response", "layout": "full"},
                {"id": "fairness", "type": "markdown", "sourceId": "models", "body": f"## Interpretation and fairness boundary\n\nThe formal seven-scalar Dimension vector has 160 representation-collision cases in the 957-case dataset because distinct individual coil rectangles can share the same aggregate variables. Explicit Dimension removes this omission by supplying all six coil rectangles. Best-SDF ion-density improvement is **{sdf_vs_formal:+.2f}%** versus formal Dimension and **{sdf_vs_explicit:+.2f}%** versus explicit Dimension (positive means SDF is lower error). {boundary}"},
                {"id": "ablation_interpretation", "type": "markdown", "sourceId": "paired", "body": f"## Interpretation of the magnetic and structural additions\n\n{recommendation}  The paired ion-density change for vacuum B versus Formal SDF is {float(vacuum_vs_sdf_paired['paired_median_change_pp']):.2f} points with interval [{float(vacuum_vs_sdf_paired['paired_median_change_ci_low_pp']):.2f}, {float(vacuum_vs_sdf_paired['paired_median_change_ci_high_pp']):.2f}].  Structural maps versus vacuum B change it by {float(structure_vs_vacuum_paired['paired_median_change_pp']):.2f} points with interval [{float(structure_vs_vacuum_paired['paired_median_change_ci_low_pp']):.2f}, {float(structure_vs_vacuum_paired['paired_median_change_ci_high_pp']):.2f}]."},
                {"id": "cost_intro", "type": "markdown", "sourceId": "cost", "body": "## Accuracy is judged together with implementation cost\n\nCase-varying magnetic and structural grids increase preprocessing storage and loading cost.  The table reports the completed feature-pack size, observed 50-epoch training time, selected epoch, and standard-test primary metric for every model."},
                {"id": "cost", "type": "table", "tableId": "cost_table", "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": "## Limitations and robustness\n\nThis is one training seed, so bootstrap intervals quantify case variation, not training-seed variation. G4 is frozen and COMSOL-backed but remains inside the generated v43 design envelope; it does not certify arbitrary coil topology. The magnetic carrier is a vacuum unit-current field and does not encode plasma-current feedback. The evaluation contains no new COMSOL cases beyond the frozen dataset. Formal adoption should repeat the strongest and explicit-Dimension models at additional seeds if their paired intervals overlap."},
                {"id": "next", "type": "markdown", "body": f"## Recommended next steps\n\n1. Use figure 00 as the primary visual and figure 02 as the quantitative support.\n2. State the claim at the explicit-Dimension boundary, not only against the seven-scalar baseline.\n3. {recommendation}\n4. Repeat Formal SDF and the strongest enhanced SDF across two additional seeds before promoting a small ablation effect.\n5. Run equal-budget offline optimization on the existing G4 pool before any continuous optimization claim."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Does the winning representation retain its ordering across training seeds?\n- Is Bohm-profile improvement driven by ion density, electron temperature, or both?\n- Does better structural response reduce offline optimization regret under the same candidate budget?"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "model_summary": model_summary,
                "family_summary": family_rows,
                "family_nonoverall": [row for row in family_rows if row["family_id"] != "all_unknown"],
                "incremental_effect": incremental_rows,
                "model_contract": contracts,
                "training_cost": training_rows,
                "paired_overall": paired_overall,
            },
        },
    }
    (REPORT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = {
        "claim": claim,
        "boundary": boundary,
        "best_ni": best_ni,
        "best_bohm": best_bohm,
        "best_sdf_vs_formal_dimension_pct": sdf_vs_formal,
        "best_sdf_vs_explicit_dimension_pct": sdf_vs_explicit,
        "case_count": len({row["case_id"] for row in case_rows}),
        "response_row_count": len(response_rows),
        "recommended_model_id": recommended_model_id,
        "recommendation": recommendation,
        "formal_sdf_vs_explicit_dimension_paired": sdf_vs_explicit_paired,
        "vacuum_vs_formal_sdf_paired": vacuum_vs_sdf_paired,
        "structure_vs_vacuum_paired": structure_vs_vacuum_paired,
    }
    (REPORT / "decision_summary.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
