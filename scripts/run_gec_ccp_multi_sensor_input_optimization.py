"""Reuse one CPU FNO for three independent noisy GEC-CCP measurements.

Each pseudo measurement is generated once from an interpolation-test COMSOL
case.  Pointwise noise is frozen for every optimizer trial, so the objective is
deterministic.  The script delegates each case to the single-measurement runner
and then aggregates accuracy, budget, and measured-cost evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = Path(__file__).with_name("run_gec_ccp_sensor_input_optimization.py")
DEFAULT_RUN_ROOT = Path("runs/gec_ccp_spatial_v3/seed_413/n78/fno")
DEFAULT_OUT_ROOT = Path("runs/gec_ccp_multi_sensor_input_optimization_cpu")
DEFAULT_CASES = (
    "case_td003_pa050_pp0_3_gamma_010__steady",
    "case_td016_pp0_1_gamma_007__steady",
    "case_td030_pa200_pp0_5_gamma_004__steady",
)
METHODS = ("cmaes", "tpe", "random")
COLORS = {"cmaes": "#0072B2", "tpe": "#D55E00", "random": "#009E73"}
PARAMETERS = ("PP0", "PA", "gamma")
PARAM_LABELS = {"PP0": "PP0 [W]", "PA": "PA [Torr]", "gamma": r"$\gamma$"}
BUDGETS = (10, 25, 50, 100)
CONDITIONS_CSV = Path("data/outputs_merged_td_all_success_pa_ext0520/conditions.csv")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _best_trial(rows: list[dict[str, str]]) -> int:
    return int(min(rows, key=lambda row: float(row["loss"]))["trial"])


def _best_at_budget(rows: list[dict[str, str]], budget: int) -> float:
    eligible = [row for row in rows if int(row["trial"]) <= budget]
    return min(float(row["loss"]) for row in eligible)


def _run_measurement(
    *,
    case_id: str,
    case_root: Path,
    run_root: Path,
    trials: int,
    optimizer_seed: int,
    noise_seed: int,
    point_noise_rel: float,
    calibration_noise_rel: float,
    cpu_threads: int,
) -> None:
    command = [
        sys.executable,
        str(SINGLE_RUNNER),
        "--run-root",
        str(run_root),
        "--case-id",
        case_id,
        "--out-root",
        str(case_root),
        "--trials",
        str(trials),
        "--optimizer-seed",
        str(optimizer_seed),
        "--noise-seed",
        str(noise_seed),
        "--point-noise-rel",
        str(point_noise_rel),
        "--calibration-noise-rel",
        str(calibration_noise_rel),
        "--cpu-threads",
        str(cpu_threads),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _aggregate_outputs(
    out_root: Path,
    cases: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    case_meta_rows: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(cases, start=1):
        measurement_id = f"M{case_index}"
        case_root = out_root / measurement_id
        meta = _read_json(case_root / "run_metadata.json")
        local_summary = _read_csv(case_root / "summary.csv")
        local_trials = _read_csv(case_root / "trials.csv")
        local_observations = _read_csv(case_root / "observations.csv")
        truth = dict(meta["truth_cond"])
        initial = dict(meta["initial_metrics"])
        truth_input = dict(meta["truth_input_metrics"])
        case_meta_rows.append(
            {
                "measurement_id": measurement_id,
                "case_id": case_id,
                "Td": truth["Td"],
                "truth_PP0": truth["PP0"],
                "truth_PA": truth["PA"],
                "truth_gamma": truth["gamma"],
                "realized_sensor_noise_rel": meta["realized_sensor_noise_rel"],
                "initial_sensor_loss": initial["sensor_loss"],
                "initial_profile_truth_rel_l2": initial["profile_truth_rel_l2"],
                "initial_fullfield_truth_rel_l2": initial["fullfield_truth_rel_l2"],
                "truth_input_sensor_loss": truth_input["sensor_loss"],
                "truth_input_profile_truth_rel_l2": truth_input["profile_truth_rel_l2"],
                "truth_input_fullfield_truth_rel_l2": truth_input["fullfield_truth_rel_l2"],
                "case_total_wall_s": meta["total_wall_s"],
                "engine_load_s": meta["engine_load_s"],
            }
        )
        for row in local_observations:
            observation_rows.append({"measurement_id": measurement_id, "case_id": case_id, **row})
        for method in METHODS:
            method_trials = [row for row in local_trials if row["method"] == method]
            best_trial = _best_trial(method_trials)
            summary = next(row for row in local_summary if row["method"] == method)
            initial_sensor_loss = float(initial["sensor_loss"])
            profile_truth = float(summary["profile_truth_rel_l2"])
            fullfield_truth = float(summary["fullfield_truth_rel_l2"])
            summary_rows.append(
                {
                    "measurement_id": measurement_id,
                    "case_id": case_id,
                    "Td": truth["Td"],
                    "truth_PP0": truth["PP0"],
                    "truth_PA": truth["PA"],
                    "truth_gamma": truth["gamma"],
                    "realized_sensor_noise_rel": meta["realized_sensor_noise_rel"],
                    "initial_sensor_loss": initial_sensor_loss,
                    "initial_profile_truth_rel_l2": initial["profile_truth_rel_l2"],
                    "initial_fullfield_truth_rel_l2": initial["fullfield_truth_rel_l2"],
                    "truth_input_sensor_loss": truth_input["sensor_loss"],
                    "truth_input_profile_truth_rel_l2": truth_input["profile_truth_rel_l2"],
                    "truth_input_fullfield_truth_rel_l2": truth_input["fullfield_truth_rel_l2"],
                    **summary,
                    "best_trial": best_trial,
                    "sensor_loss_reduction_fraction": 1.0 - float(summary["loss"]) / initial_sensor_loss,
                    "profile_truth_improved": profile_truth < float(initial["profile_truth_rel_l2"]),
                    "fullfield_truth_improved": fullfield_truth < float(initial["fullfield_truth_rel_l2"]),
                }
            )
            for budget in BUDGETS:
                if budget <= len(method_trials):
                    budget_rows.append(
                        {
                            "measurement_id": measurement_id,
                            "case_id": case_id,
                            "method": method,
                            "budget": budget,
                            "best_loss": _best_at_budget(method_trials, budget),
                        }
                    )
        for row in local_trials:
            trial_rows.append({"measurement_id": measurement_id, "case_id": case_id, **row})
    return summary_rows, trial_rows, observation_rows, budget_rows, case_meta_rows


def _first_hit(rows: list[dict[str, Any]], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["trial"])
    return None


def _quality_tables(
    summary_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    case_meta_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for meta in case_meta_rows:
        measurement_id = str(meta["measurement_id"])
        case_id = str(meta["case_id"])
        true_input_loss = float(meta["truth_input_sensor_loss"])
        case_trials = [row for row in trial_rows if row["case_id"] == case_id]
        global_best = min(float(row["loss"]) for row in case_trials)
        for method in METHODS:
            rows = sorted(
                [row for row in case_trials if row["method"] == method],
                key=lambda row: int(row["trial"]),
            )
            final_loss = float(rows[-1]["best_loss"])
            practical_hit = _first_hit(
                rows, lambda row: float(row["best_loss"]) / true_input_loss <= 1.10
            )
            strict_hit = _first_hit(
                rows, lambda row: float(row["best_loss"]) / true_input_loss <= 1.00
            )
            regret_hits = {
                tolerance: _first_hit(
                    rows,
                    lambda row, tol=tolerance: float(row["best_loss"]) / global_best - 1.0 <= tol,
                )
                for tolerance in (0.02, 0.05, 0.10)
            }
            summary = next(
                row
                for row in summary_rows
                if row["case_id"] == case_id and row["method"] == method
            )
            quality_rows.append(
                {
                    "measurement_id": measurement_id,
                    "case_id": case_id,
                    "method": method,
                    "true_input_sensor_loss": true_input_loss,
                    "case_global_best_loss": global_best,
                    "final_loss": final_loss,
                    "final_ratio_to_true_input": final_loss / true_input_loss,
                    "final_relative_regret": final_loss / global_best - 1.0,
                    "hit_true_input_1p10": practical_hit is not None,
                    "first_hit_true_input_1p10": practical_hit if practical_hit is not None else "",
                    "hit_true_input_1p00": strict_hit is not None,
                    "first_hit_true_input_1p00": strict_hit if strict_hit is not None else "",
                    "hit_regret_2pct": regret_hits[0.02] is not None,
                    "first_hit_regret_2pct": regret_hits[0.02] if regret_hits[0.02] is not None else "",
                    "hit_regret_5pct": regret_hits[0.05] is not None,
                    "first_hit_regret_5pct": regret_hits[0.05] if regret_hits[0.05] is not None else "",
                    "hit_regret_10pct": regret_hits[0.10] is not None,
                    "first_hit_regret_10pct": regret_hits[0.10] if regret_hits[0.10] is not None else "",
                    "last_improvement_trial": int(summary["best_trial"]),
                }
            )
            for row in rows:
                best_loss = float(row["best_loss"])
                history_rows.append(
                    {
                        "measurement_id": measurement_id,
                        "case_id": case_id,
                        "method": method,
                        "trial": int(row["trial"]),
                        "best_loss": best_loss,
                        "ratio_to_true_input": best_loss / true_input_loss,
                        "relative_regret": best_loss / global_best - 1.0,
                    }
                )
    return quality_rows, history_rows


def _method_aggregate(
    summary_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in summary_rows if row["method"] == method]
        quality = [row for row in quality_rows if row["method"] == method]
        practical_hits = [
            int(row["first_hit_true_input_1p10"])
            for row in quality
            if str(row["first_hit_true_input_1p10"]).strip()
        ]
        strict_hits = [
            int(row["first_hit_true_input_1p00"])
            for row in quality
            if str(row["first_hit_true_input_1p00"]).strip()
        ]
        regret_hits = [
            int(row["first_hit_regret_10pct"])
            for row in quality
            if str(row["first_hit_regret_10pct"]).strip()
        ]
        rows.append(
            {
                "method": method,
                "n_measurements": len(selected),
                "mean_sensor_loss": float(np.mean([float(row["loss"]) for row in selected])),
                "max_sensor_loss": max(float(row["loss"]) for row in selected),
                "mean_profile_truth_rel_l2": float(
                    np.mean([float(row["profile_truth_rel_l2"]) for row in selected])
                ),
                "max_profile_truth_rel_l2": max(
                    float(row["profile_truth_rel_l2"]) for row in selected
                ),
                "mean_fullfield_truth_rel_l2": float(
                    np.mean([float(row["fullfield_truth_rel_l2"]) for row in selected])
                ),
                "max_fullfield_truth_rel_l2": max(
                    float(row["fullfield_truth_rel_l2"]) for row in selected
                ),
                "mean_condition_normalized_rmse": float(
                    np.mean([float(row["condition_normalized_rmse"]) for row in selected])
                ),
                "mean_last_improvement_trial": float(
                    np.mean([int(row["last_improvement_trial"]) for row in quality])
                ),
                "mean_final_relative_regret": float(
                    np.mean([float(row["final_relative_regret"]) for row in quality])
                ),
                "true_input_1p10_success_cases": len(practical_hits),
                "true_input_1p10_median_hit_trial": (
                    float(np.median(practical_hits)) if practical_hits else ""
                ),
                "true_input_1p00_success_cases": len(strict_hits),
                "true_input_1p00_median_hit_trial": float(np.median(strict_hits)) if strict_hits else "",
                "regret_10pct_success_cases": len(regret_hits),
                "regret_10pct_median_hit_trial": float(np.median(regret_hits)) if regret_hits else "",
                "profile_truth_improved_cases": sum(
                    str(row["profile_truth_improved"]).lower() == "true" for row in selected
                ),
                "fullfield_truth_improved_cases": sum(
                    str(row["fullfield_truth_improved"]).lower() == "true" for row in selected
                ),
                "optimizer_wall_s": sum(float(row["wall_s"]) for row in selected),
                "invalid_trials": sum(int(row["invalid_trials"]) for row in selected),
            }
        )
    return rows


def _cost_tables(
    *,
    run_root: Path,
    summary_rows: list[dict[str, Any]],
    case_meta_rows: list[dict[str, Any]],
    trials: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_rows = _read_csv(REPO_ROOT / CONDITIONS_CSV)
    elapsed_by_id = {row["case_id"]: float(row["elapsed_s"]) for row in condition_rows}
    td_groups: dict[float, list[float]] = {}
    for row in condition_rows:
        td_groups.setdefault(float(row["Td"]), []).append(float(row["elapsed_s"]))
    td_mean = {key: float(np.mean(values)) for key, values in td_groups.items()}
    td_median = {key: float(np.median(values)) for key, values in td_groups.items()}
    split = _read_json(run_root / "preprocessing" / "split" / "split_interp_v1.json")
    train_val_ids = [str(case).removesuffix("__steady") for case in split["train"] + split["val"]]
    train_val_data_s = sum(elapsed_by_id[case_id] for case_id in train_val_ids)
    full_data_s = sum(float(row["elapsed_s"]) for row in condition_rows)
    progress = _read_json(
        run_root
        / "models"
        / "fno"
        / "eval_protocol"
        / "interp"
        / "train"
        / "scalars"
        / "progress_latest.json"
    )
    training_s = float(progress["elapsed_seconds"])
    direct_fixed_mean_s = sum(trials * td_mean[float(row["Td"])] for row in case_meta_rows)
    direct_fixed_median_s = sum(trials * td_median[float(row["Td"])] for row in case_meta_rows)
    verification_mean_s = sum(td_mean[float(row["Td"])] for row in case_meta_rows)
    cost_rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in summary_rows if row["method"] == method]
        search_s = sum(float(row["wall_s"]) for row in selected)
        direct_best_mean_s = sum(
            int(row["best_trial"]) * td_mean[float(row["Td"])] for row in selected
        )
        direct_best_median_s = sum(
            int(row["best_trial"]) * td_median[float(row["Td"])] for row in selected
        )
        cost_rows.append(
            {
                "method": method,
                "n_measurements": len(selected),
                "trials_per_measurement": trials,
                "best_trials_total": sum(int(row["best_trial"]) for row in selected),
                "surrogate_search_s": search_s,
                "direct_fixed_budget_mean_est_s": direct_fixed_mean_s,
                "direct_fixed_budget_median_est_s": direct_fixed_median_s,
                "direct_oracle_best_mean_est_s": direct_best_mean_s,
                "direct_oracle_best_median_est_s": direct_best_median_s,
                "surrogate_65_build_s": train_val_data_s + training_s + search_s,
                "surrogate_78_benchmark_s": full_data_s + training_s + search_s,
                "surrogate_65_build_plus_3_validation_s": (
                    train_val_data_s + training_s + search_s + verification_mean_s
                ),
                "fixed_budget_speedup_65_build": direct_fixed_mean_s
                / (train_val_data_s + training_s + search_s),
                "fixed_budget_speedup_existing_model": direct_fixed_mean_s / search_s,
            }
        )
    assumptions = {
        "train_val_cases": len(train_val_ids),
        "train_val_data_s": train_val_data_s,
        "full_benchmark_cases": len(condition_rows),
        "full_data_s": full_data_s,
        "fno_interp_training_s": training_s,
        "td_mean_comsol_s": {str(key): value for key, value in td_mean.items()},
        "td_median_comsol_s": {str(key): value for key, value in td_median.items()},
        "three_candidate_validation_mean_est_s": verification_mean_s,
        "direct_cost_is_estimate": True,
        "oracle_best_trial_is_retrospective_lower_bound": True,
    }
    return cost_rows, assumptions


def _plot_loss_history(
    out_root: Path,
    cases: tuple[str, ...],
    trial_rows: list[dict[str, Any]],
    case_meta_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.6, 8.6), sharex=True)
    meta_by_case = {row["case_id"]: row for row in case_meta_rows}
    for index, (ax, case_id) in enumerate(zip(axes, cases, strict=True), start=1):
        for method in METHODS:
            rows = [
                row for row in trial_rows if row["case_id"] == case_id and row["method"] == method
            ]
            xs = [int(row["trial"]) for row in rows]
            raw = 100.0 * np.asarray([float(row["loss"]) for row in rows])
            best = 100.0 * np.asarray([float(row["best_loss"]) for row in rows])
            ax.plot(xs, raw, color=COLORS[method], alpha=0.14, linewidth=0.7)
            ax.plot(xs, best, color=COLORS[method], linewidth=1.8, label=method)
        true_input = 100.0 * float(meta_by_case[case_id]["truth_input_sensor_loss"])
        ax.axhline(true_input, color="black", linestyle=":", linewidth=1.1, label="true-input FNO")
        ax.set_yscale("log")
        ax.set_ylabel("loss [%]")
        ax.set_title(f"M{index}: {case_id}", fontsize=9)
        ax.grid(which="both", alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("CPU FNO evaluations")
    fig.suptitle("Fixed-noise sensor loss: raw and best-so-far", y=0.995)
    fig.tight_layout()
    fig.savefig(out_root / "loss_history_by_case.png", dpi=180)
    plt.close(fig)


def _plot_common_quality(
    out_root: Path,
    history_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for method in METHODS:
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        trials = sorted(
            {int(row["trial"]) for row in history_rows if row["method"] == method}
        )
        for trial in trials:
            values = np.asarray(
                [
                    float(row["ratio_to_true_input"])
                    for row in history_rows
                    if row["method"] == method and int(row["trial"]) == trial
                ],
                dtype=float,
            )
            medians.append(float(np.median(values)))
            lows.append(float(np.min(values)))
            highs.append(float(np.max(values)))
        axes[0].plot(trials, medians, color=COLORS[method], linewidth=2.0, label=method)
        axes[0].fill_between(trials, lows, highs, color=COLORS[method], alpha=0.10)
    axes[0].axhline(1.10, color="black", linestyle="--", linewidth=1.1, label="common target 1.10")
    axes[0].axhline(1.00, color="black", linestyle=":", linewidth=1.1, label="strict target 1.00")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CPU FNO evaluations")
    axes[0].set_ylabel("best loss / true-input FNO loss")
    axes[0].set_title("Common-quality convergence (median and range)")
    axes[0].grid(which="both", alpha=0.22)
    axes[0].legend(fontsize=8)

    x = np.arange(3, dtype=float)
    offsets = {"cmaes": -0.18, "tpe": 0.0, "random": 0.18}
    for method in METHODS:
        selected = sorted(
            [row for row in quality_rows if row["method"] == method],
            key=lambda row: str(row["measurement_id"]),
        )
        ys = [
            float(row["first_hit_true_input_1p10"])
            if str(row["first_hit_true_input_1p10"]).strip()
            else 104.0
            for row in selected
        ]
        reached = [bool(str(row["first_hit_true_input_1p10"]).strip()) for row in selected]
        for index, (value, hit) in enumerate(zip(ys, reached, strict=True)):
            axes[1].scatter(
                x[index] + offsets[method],
                value,
                color=COLORS[method] if hit else "white",
                edgecolor=COLORS[method],
                marker="o" if hit else "v",
                s=52,
                label=method if index == 0 else None,
                zorder=3,
            )
            if not hit:
                axes[1].text(x[index] + offsets[method], value - 3.0, "NR", ha="center", va="top", fontsize=7)
    axes[1].axhline(100.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xticks(x, ("M1", "M2", "M3"))
    axes[1].set_ylim(0.0, 108.0)
    axes[1].set_ylabel("first trial reaching ratio <= 1.10")
    axes[1].set_title("Time to the same quality (NR = not reached)")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "common_quality_convergence.png", dpi=180)
    plt.close(fig)


def _plot_budget(out_root: Path, budget_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in METHODS:
        means = []
        for budget in BUDGETS:
            values = [
                100.0 * float(row["best_loss"])
                for row in budget_rows
                if row["method"] == method and int(row["budget"]) == budget
            ]
            means.append(float(np.mean(values)))
        ax.plot(BUDGETS, means, marker="o", color=COLORS[method], label=method)
    ax.set_yscale("log")
    ax.set_xlabel("trial budget per measurement")
    ax.set_ylabel("mean best sensor loss over 3 measurements [%]")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "budget_comparison.png", dpi=180)
    plt.close(fig)


def _plot_errors(out_root: Path, summary_rows: list[dict[str, Any]], cases: tuple[str, ...]) -> None:
    labels = [f"M{index}" for index in range(1, len(cases) + 1)]
    entries = ("initial",) + METHODS
    colors = {"initial": "#999999", **COLORS}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    specs = (
        ("profile_truth_rel_l2", "initial_profile_truth_rel_l2", "truth profile error [%]"),
        ("fullfield_truth_rel_l2", "initial_fullfield_truth_rel_l2", "full-field truth error [%]"),
    )
    x = np.arange(len(cases), dtype=float)
    width = 0.19
    for ax, (method_key, initial_key, ylabel) in zip(axes, specs, strict=True):
        for offset_index, entry in enumerate(entries):
            values = []
            for case_id in cases:
                row = next(item for item in summary_rows if item["case_id"] == case_id)
                if entry == "initial":
                    values.append(100.0 * float(row[initial_key]))
                else:
                    method_row = next(
                        item
                        for item in summary_rows
                        if item["case_id"] == case_id and item["method"] == entry
                    )
                    values.append(100.0 * float(method_row[method_key]))
            offset = (offset_index - 1.5) * width
            ax.bar(x + offset, values, width=width, color=colors[entry], label=entry)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "truth_error_by_case.png", dpi=180)
    plt.close(fig)


def _plot_parameter_recovery(
    out_root: Path,
    summary_rows: list[dict[str, Any]],
    cases: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    x = np.arange(len(cases), dtype=float)
    offsets = {"cmaes": -0.16, "tpe": 0.0, "random": 0.16}
    for ax, key in zip(axes, PARAMETERS, strict=True):
        truth_values = [
            float(next(row for row in summary_rows if row["case_id"] == case_id)[f"truth_{key}"])
            for case_id in cases
        ]
        ax.plot(x, truth_values, color="black", marker="x", linewidth=1.1, label="truth")
        for method in METHODS:
            values = [
                float(
                    next(
                        row
                        for row in summary_rows
                        if row["case_id"] == case_id and row["method"] == method
                    )[key]
                )
                for case_id in cases
            ]
            ax.scatter(x + offsets[method], values, color=COLORS[method], label=method, zorder=3)
        ax.set_xticks(x, [f"M{index}" for index in range(1, len(cases) + 1)])
        ax.set_ylabel(PARAM_LABELS[key])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "parameter_recovery.png", dpi=180)
    plt.close(fig)


def _plot_costs(
    out_root: Path,
    cost_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> None:
    cma = next(row for row in cost_rows if row["method"] == "cmaes")
    build_s = float(assumptions["train_val_data_s"]) + float(assumptions["fno_interp_training_s"])
    search_s = float(cma["surrogate_search_s"])
    fixed_direct_s = float(cma["direct_fixed_budget_mean_est_s"])
    oracle_direct_s = float(cma["direct_oracle_best_mean_est_s"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    labels = ["Direct\n100/case", "Direct\noracle-best", "Surrogate\nfrom scratch", "Existing FNO\nsearch only"]
    values_h = [fixed_direct_s / 3600.0, oracle_direct_s / 3600.0, (build_s + search_s) / 3600.0, search_s / 3600.0]
    axes[0].bar(labels, values_h, color=["#CC6677", "#EE99AA", "#4477AA", "#66CCEE"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("wall-time estimate [h] (log scale)")
    axes[0].set_title("Three-measurement CMA-ES campaign")
    axes[0].grid(axis="y", which="both", alpha=0.22)
    for index, value in enumerate(values_h):
        axes[0].text(index, value * 1.08, f"{value:.3g}", ha="center", va="bottom", fontsize=8)

    measurements = np.arange(1, 11, dtype=float)
    direct_fixed_per = fixed_direct_s / 3.0
    direct_oracle_per = oracle_direct_s / 3.0
    fno_per = search_s / 3.0
    axes[1].plot(measurements, measurements * direct_fixed_per / 3600.0, label="direct: 100/case", color="#CC6677")
    axes[1].plot(
        measurements,
        measurements * direct_oracle_per / 3600.0,
        label="direct: oracle-best",
        color="#EE99AA",
        linestyle="--",
    )
    axes[1].plot(
        measurements,
        (build_s + measurements * fno_per) / 3600.0,
        label="surrogate: 65-case build",
        color="#4477AA",
    )
    axes[1].plot(measurements, measurements * fno_per / 3600.0, label="existing FNO", color="#66CCEE")
    axes[1].axvline(3.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("number of independent measurements")
    axes[1].set_ylabel("cumulative time [h]")
    axes[1].set_title("Amortization (CMA-ES)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "cost_and_amortization.png", dpi=180)
    plt.close(fig)


def _write_index(
    *,
    out_root: Path,
    cases: tuple[str, ...],
    point_noise_rel: float,
    calibration_noise_rel: float,
    trials: int,
    case_meta_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
    measurement_wall_s: float,
) -> None:
    cma_cost = next(row for row in cost_rows if row["method"] == "cmaes")
    best_truth_method = min(method_rows, key=lambda row: float(row["mean_profile_truth_rel_l2"]))
    random_row = next(row for row in method_rows if row["method"] == "random")
    mean_noise = float(np.mean([float(row["realized_sensor_noise_rel"]) for row in case_meta_rows]))
    mean_initial_profile = float(
        np.mean([float(row["initial_profile_truth_rel_l2"]) for row in case_meta_rows])
    )
    mean_initial_field = float(
        np.mean([float(row["initial_fullfield_truth_rel_l2"]) for row in case_meta_rows])
    )
    mean_true_input_sensor = float(
        np.mean([float(row["truth_input_sensor_loss"]) for row in case_meta_rows])
    )
    mean_true_input_profile = float(
        np.mean([float(row["truth_input_profile_truth_rel_l2"]) for row in case_meta_rows])
    )
    mean_true_input_field = float(
        np.mean([float(row["truth_input_fullfield_truth_rel_l2"]) for row in case_meta_rows])
    )
    lines = [
        "# GEC-CCP three-measurement input optimization (CPU FNO)",
        "",
        "- 同じ学習済FNOを、条件と構造が異なるheld-out 3ケースへ再利用しました。",
        f"- 各センサー点に標準偏差{100.0 * point_noise_rel:.3g}%の固定Gaussianノイズを付加しました。",
        f"- 校正ノイズは{100.0 * calibration_noise_rel:.3g}%で、候補評価ごとの再抽選はしていません。",
        "- 目的関数は線形電子密度profileのrelative L2です。loss履歴の表示だけ対数軸です。",
        "- γは探索へ含めていますが、電子密度に対する低感度のため同定成功とは解釈しません。",
        f"- budget: {trials} CPU FNO evaluations / optimizer / measurement",
        "",
        "## Measurements",
        "",
        "| ID | held-out case | Td | PP0 [W] | PA [Torr] | gamma | realized noise [%] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, case_id in enumerate(cases, start=1):
        meta = next(row for row in case_meta_rows if row["case_id"] == case_id)
        lines.append(
            f"| M{index} | [`{case_id}`](M{index}/index.md) | {float(meta['Td']):.3g} | "
            f"{float(meta['truth_PP0']):.3g} | {float(meta['truth_PA']):.3g} | "
            f"{float(meta['truth_gamma']):.3g} | "
            f"{100.0 * float(meta['realized_sensor_noise_rel']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            f"- mean realized sensor noise: {100.0 * mean_noise:.4f}%",
            f"- common initial condition: truth-profile {100.0 * mean_initial_profile:.4f}%, full-field {100.0 * mean_initial_field:.4f}%",
            f"- true-input FNO: noisy-sensor {100.0 * mean_true_input_sensor:.4f}%, truth-profile {100.0 * mean_true_input_profile:.4f}%, full-field {100.0 * mean_true_input_field:.4f}%",
            "",
            "## Aggregate accuracy",
            "",
            "| method | mean sensor loss [%] | mean truth-profile [%] | mean full-field [%] | final regret [%] | R<=1.10 hits | median hit trial | R<=1.00 hits | CPU [s] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {100.0 * float(row['mean_sensor_loss']):.4f} | "
            f"{100.0 * float(row['mean_profile_truth_rel_l2']):.4f} | "
            f"{100.0 * float(row['mean_fullfield_truth_rel_l2']):.4f} | "
            f"{100.0 * float(row['mean_final_relative_regret']):.3f} | "
            f"{row['true_input_1p10_success_cases']}/3 | "
            f"{row['true_input_1p10_median_hit_trial'] or 'NR'} | "
            f"{row['true_input_1p00_success_cases']}/3 | "
            f"{float(row['optimizer_wall_s']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Cost interpretation",
            "",
            f"- COMSOL train+validation 65-case acquisition: {float(assumptions['train_val_data_s']) / 3600.0:.3f} h",
            f"- FNO interpolation training: {float(assumptions['fno_interp_training_s']) / 60.0:.2f} min",
            f"- Direct COMSOL estimate, 3 measurements x {trials} trials: {float(cma_cost['direct_fixed_budget_mean_est_s']) / 3600.0:.3f} h",
            f"- Direct COMSOL estimate using Td-wise medians: {float(cma_cost['direct_fixed_budget_median_est_s']) / 3600.0:.3f} h",
            f"- Surrogate from-scratch cost (65 cases + training + CMA search): {float(cma_cost['surrogate_65_build_s']) / 3600.0:.3f} h",
            f"- Full 78-case benchmark cost + training + CMA search: {float(cma_cost['surrogate_78_benchmark_s']) / 3600.0:.3f} h",
            f"- 65-case build + estimated one COMSOL validation per measurement: {float(cma_cost['surrogate_65_build_plus_3_validation_s']) / 3600.0:.3f} h",
            f"- Existing-FNO marginal CMA search: {float(cma_cost['surrogate_search_s']):.2f} s",
            f"- Fixed-budget from-scratch speedup: {float(cma_cost['fixed_budget_speedup_65_build']):.2f}x",
            f"- Td-wise median-time speedup: {float(cma_cost['direct_fixed_budget_median_est_s']) / float(cma_cost['surrogate_65_build_s']):.2f}x",
            f"- Direct COMSOL oracle-best estimate: {float(cma_cost['direct_oracle_best_mean_est_s']) / 3600.0:.3f} h",
            "- oracle-bestは最終最良trialを事後に知る下限で、実運用の停止時間ではありません。",
            "- 直接COMSOL時間は既存78ケースのTd別実測平均による推定で、最適候補の新規COMSOL検証は未実施です。",
            "",
            "## Interpretation",
            "",
            f"- clean truth-profile誤差が最小だったのは `{best_truth_method['method']}` の平均 {100.0 * float(best_truth_method['mean_profile_truth_rel_l2']):.4f}%です。",
            f"- Randomも3/3ケースを改善しましたが、平均truth-profile誤差は {100.0 * float(random_row['mean_profile_truth_rel_l2']):.4f}%でした。",
            f"- 共通品質 R<=1.10 には`{best_truth_method['method']}` {best_truth_method['true_input_1p10_success_cases']}/3、Random {random_row['true_input_1p10_success_cases']}/3ケースが到達しました。",
            "- 従来のbest_trialは各手法自身の異なる最終値の最終更新時刻であり、収束速度の比較から除外しました。数値列はlast_improvement_trialとしてquality_attainment.csvにのみ残しています。",
            f"- `{best_truth_method['method']}` のnoisy-sensor lossは {100.0 * float(best_truth_method['mean_sensor_loss']):.4f}%でtrue-input FNOより低い一方、truth/full-field誤差はtrue-inputより高く、ノイズまたはサロゲート誤差の入力補償が含まれます。",
            "- 3ケース・1 seedなので最適化手法の一般順位は主張しません。サロゲートの有効性は、全ケースの真値誤差改善と反復評価時間で判定します。",
            "",
            "## Graphs",
            "",
            "- [common_quality_convergence.png](common_quality_convergence.png)",
            "- [loss_history_by_case.png](loss_history_by_case.png)",
            "- [budget_comparison.png](budget_comparison.png)",
            "- [truth_error_by_case.png](truth_error_by_case.png)",
            "- [parameter_recovery.png](parameter_recovery.png)",
            "- [cost_and_amortization.png](cost_and_amortization.png)",
            "",
            "## Tables",
            "",
            "- [case_method_summary.csv](case_method_summary.csv)",
            "- [method_aggregate.csv](method_aggregate.csv)",
            "- [budget_summary.csv](budget_summary.csv)",
            "- [quality_attainment.csv](quality_attainment.csv)",
            "- [common_quality_history.csv](common_quality_history.csv)",
            "- [all_trials.csv](all_trials.csv)",
            "- [all_observations.csv](all_observations.csv)",
            "- [runtime_comparison.csv](runtime_comparison.csv)",
            "- [case_metadata.csv](case_metadata.csv)",
            "- [run_metadata.json](run_metadata.json)",
            "",
            f"Total three-measurement execution wall time: {measurement_wall_s:.2f} s",
        ]
    )
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--case-ids", nargs=3, default=DEFAULT_CASES)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--optimizer-seed", type=int, default=20260715)
    parser.add_argument("--noise-seed", type=int, default=20260715)
    parser.add_argument("--point-noise-rel", type=float, default=0.05)
    parser.add_argument("--calibration-noise-rel", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 20))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.trials < max(BUDGETS):
        raise ValueError(f"--trials must be at least {max(BUDGETS)}")
    cases = tuple(str(case_id) for case_id in args.case_ids)
    if len(set(cases)) != 3:
        raise ValueError("three distinct case ids are required")
    started = time.perf_counter()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    for case_index, case_id in enumerate(cases, start=1):
        case_root = out_root / f"M{case_index}"
        complete = all((case_root / name).exists() for name in ("summary.csv", "trials.csv", "run_metadata.json"))
        if not (args.reuse_existing and complete):
            _run_measurement(
                case_id=case_id,
                case_root=case_root,
                run_root=args.run_root,
                trials=int(args.trials),
                optimizer_seed=int(args.optimizer_seed),
                noise_seed=int(args.noise_seed) + case_index - 1,
                point_noise_rel=float(args.point_noise_rel),
                calibration_noise_rel=float(args.calibration_noise_rel),
                cpu_threads=int(args.cpu_threads),
            )
    summary_rows, trial_rows, observation_rows, budget_rows, case_meta_rows = _aggregate_outputs(
        out_root, cases
    )
    quality_rows, quality_history_rows = _quality_tables(
        summary_rows, trial_rows, case_meta_rows
    )
    method_rows = _method_aggregate(summary_rows, quality_rows)
    cost_rows, assumptions = _cost_tables(
        run_root=args.run_root,
        summary_rows=summary_rows,
        case_meta_rows=case_meta_rows,
        trials=int(args.trials),
    )
    _write_csv(out_root / "case_method_summary.csv", summary_rows)
    _write_csv(out_root / "method_aggregate.csv", method_rows)
    _write_csv(out_root / "all_trials.csv", trial_rows)
    _write_csv(out_root / "all_observations.csv", observation_rows)
    _write_csv(out_root / "budget_summary.csv", budget_rows)
    _write_csv(out_root / "quality_attainment.csv", quality_rows)
    _write_csv(out_root / "common_quality_history.csv", quality_history_rows)
    _write_csv(out_root / "runtime_comparison.csv", cost_rows)
    _write_csv(out_root / "case_metadata.csv", case_meta_rows)
    _plot_loss_history(out_root, cases, trial_rows, case_meta_rows)
    _plot_common_quality(out_root, quality_history_rows, quality_rows)
    _plot_budget(out_root, budget_rows)
    _plot_errors(out_root, summary_rows, cases)
    _plot_parameter_recovery(out_root, summary_rows, cases)
    _plot_costs(out_root, cost_rows, assumptions)
    aggregation_wall_s = time.perf_counter() - started
    measurement_wall_s = sum(float(row["case_total_wall_s"]) for row in case_meta_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "case_ids": list(cases),
        "case_selection": "three held-out diagonal cases spanning Td, PP0, PA, gamma, and structures",
        "trials_per_method_per_measurement": int(args.trials),
        "methods": list(METHODS),
        "optimizer_seed": int(args.optimizer_seed),
        "noise_seeds": [int(args.noise_seed) + index for index in range(3)],
        "point_noise_rel": float(args.point_noise_rel),
        "calibration_noise_rel": float(args.calibration_noise_rel),
        "noise_is_frozen_per_measurement": True,
        "objective": "linear ne sensor-profile relative L2; no log transform",
        "primary_convergence_target": "best loss / true-input FNO sensor loss <= 1.10",
        "strict_convergence_target": "best loss / true-input FNO sensor loss <= 1.00",
        "best_trial_interpretation": "last improvement only; not used for cross-method speed ranking",
        "cost_assumptions": assumptions,
        "three_measurement_execution_wall_s": measurement_wall_s,
        "latest_orchestration_or_aggregation_wall_s": aggregation_wall_s,
    }
    _write_json(out_root / "run_metadata.json", metadata)
    _write_index(
        out_root=out_root,
        cases=cases,
        point_noise_rel=float(args.point_noise_rel),
        calibration_noise_rel=float(args.calibration_noise_rel),
        trials=int(args.trials),
        case_meta_rows=case_meta_rows,
        method_rows=method_rows,
        cost_rows=cost_rows,
        assumptions=assumptions,
        measurement_wall_s=measurement_wall_s,
    )
    print(out_root / "index.md")


if __name__ == "__main__":
    main()
