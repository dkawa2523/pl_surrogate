#!/usr/bin/env python
"""Summarize paired model effects on the frozen ICP unknown-structure cases."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/conference_continuity_v45"
FAMILIES = ("unknown_gap_topology", "unseen_rank_height_size", "coupled_transform")
METRICS = ("ni_rel_l2_pct", "bohm_profile_rel_l2_pct")
COMPARISONS = (
    ("conference_dimension", "conference_sdf", "Formal SDF vs formal Dimension"),
    ("explicit_dimension", "conference_sdf", "Formal SDF vs explicit Dimension"),
    ("conference_sdf", "sdf_vacuum", "Vacuum B addition vs formal SDF"),
    ("sdf_vacuum", "sdf_vacuum_structure", "Structural addition vs vacuum B"),
    ("conference_sdf", "sdf_vacuum_structure", "B + structure vs formal SDF"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_median_ci(values: np.ndarray, *, seed: int, repeats: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def main() -> int:
    rows = _read_csv(REPORT / "case_metrics_unknown75.csv")
    by_model_case = {(row["model"], row["case_id"]): row for row in rows}
    output: list[dict[str, object]] = []
    row_seed = 1237
    for reference, candidate, label in COMPARISONS:
        reference_ids = {case_id for model, case_id in by_model_case if model == reference}
        candidate_ids = {case_id for model, case_id in by_model_case if model == candidate}
        if reference_ids != candidate_ids:
            raise ValueError(f"case mismatch: {reference} vs {candidate}")
        for family in (*FAMILIES, "all_unknown"):
            case_ids = sorted(
                case_id
                for case_id in reference_ids
                if family == "all_unknown"
                or by_model_case[(candidate, case_id)]["layout_kind"] == family
            )
            for metric in METRICS:
                reference_values = np.asarray(
                    [float(by_model_case[(reference, case_id)][metric]) for case_id in case_ids],
                    dtype=np.float64,
                )
                candidate_values = np.asarray(
                    [float(by_model_case[(candidate, case_id)][metric]) for case_id in case_ids],
                    dtype=np.float64,
                )
                difference = candidate_values - reference_values
                ratio = candidate_values / np.maximum(reference_values, np.finfo(np.float64).tiny)
                diff_low, diff_high = _bootstrap_median_ci(difference, seed=row_seed)
                ratio_low, ratio_high = _bootstrap_median_ci(ratio, seed=row_seed + 10000)
                row_seed += 1
                output.append(
                    {
                        "comparison": label,
                        "reference_model": reference,
                        "candidate_model": candidate,
                        "layout_kind": family,
                        "metric": metric,
                        "n": len(case_ids),
                        "reference_median_pct": float(np.median(reference_values)),
                        "candidate_median_pct": float(np.median(candidate_values)),
                        "paired_win_rate_pct": float(100.0 * np.mean(candidate_values < reference_values)),
                        "paired_median_change_pp": float(np.median(difference)),
                        "paired_median_change_ci_low_pp": diff_low,
                        "paired_median_change_ci_high_pp": diff_high,
                        "paired_median_error_ratio": float(np.median(ratio)),
                        "paired_median_error_ratio_ci_low": ratio_low,
                        "paired_median_error_ratio_ci_high": ratio_high,
                    }
                )
    _write_csv(REPORT / "paired_effects_unknown75.csv", output)
    print(f"wrote {len(output)} paired summaries to {REPORT / 'paired_effects_unknown75.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
