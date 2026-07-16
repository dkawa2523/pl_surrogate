"""Evaluate optimizer convergence over fixed measurements and five seeds.

The synthetic observations and their noise draws are identical for every
optimizer seed.  Cross-method speed is measured by first attainment of a
shared physical target, not by the last time each method improved its own
different final value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
MULTI_RUNNER = Path(__file__).with_name("run_gec_ccp_multi_sensor_input_optimization.py")
BASELINE_ROOT = Path("runs/gec_ccp_multi_sensor_input_optimization_cpu")
DEFAULT_OUT_ROOT = Path("runs/gec_ccp_multi_sensor_input_optimization_cpu_multiseed")
SEEDS = (20260715, 20260716, 20260717, 20260718, 20260719)
METHODS = ("cmaes", "tpe", "random")
COLORS = {"cmaes": "#0072B2", "tpe": "#D55E00", "random": "#009E73"}
BUDGETS = (25, 50, 100)
PRACTICAL_RATIO = 1.10
STRICT_RATIO = 1.00


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_root(out_root: Path, seed: int, baseline_root: Path) -> Path:
    return baseline_root if seed == SEEDS[0] else out_root / f"seed_{seed}"


def _run_seed(
    *,
    root: Path,
    optimizer_seed: int,
    noise_seed: int,
    trials: int,
    cpu_threads: int,
    reuse_existing: bool,
) -> None:
    complete = all(
        (root / filename).exists()
        for filename in ("case_method_summary.csv", "all_trials.csv", "case_metadata.csv", "run_metadata.json")
    ) and all((root / f"M{index}" / "observations.csv").exists() for index in range(1, 4))
    if reuse_existing and complete:
        return
    command = [
        sys.executable,
        str(MULTI_RUNNER),
        "--out-root",
        str(root),
        "--optimizer-seed",
        str(optimizer_seed),
        "--noise-seed",
        str(noise_seed),
        "--point-noise-rel",
        "0.05",
        "--calibration-noise-rel",
        "0.0",
        "--trials",
        str(trials),
        "--cpu-threads",
        str(cpu_threads),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    if reuse_existing:
        command.append("--reuse-existing")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _validate_seed_root(
    root: Path,
    *,
    optimizer_seed: int,
    noise_seed: int,
    trials: int,
    cases: list[str] | None,
    observation_hashes: dict[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    meta = _read_json(root / "run_metadata.json")
    if int(meta["optimizer_seed"]) != optimizer_seed:
        raise ValueError(f"optimizer seed mismatch in {root}")
    expected_noise = [noise_seed + index for index in range(3)]
    if [int(value) for value in meta["noise_seeds"]] != expected_noise:
        raise ValueError(f"noise seeds mismatch in {root}")
    if float(meta["point_noise_rel"]) != 0.05 or float(meta["calibration_noise_rel"]) != 0.0:
        raise ValueError(f"noise configuration mismatch in {root}")
    if int(meta["trials_per_method_per_measurement"]) != trials:
        raise ValueError(f"trial budget mismatch in {root}")
    if tuple(meta["methods"]) != METHODS:
        raise ValueError(f"optimizer methods mismatch in {root}")
    if meta.get("device") != "cpu" or not bool(meta.get("noise_is_frozen_per_measurement")):
        raise ValueError(f"execution protocol mismatch in {root}")
    current_cases = [str(case_id) for case_id in meta["case_ids"]]
    if cases is not None and current_cases != cases:
        raise ValueError(f"case ids mismatch in {root}")
    current_hashes = {
        f"M{index}": _sha256(root / f"M{index}" / "observations.csv") for index in range(1, 4)
    }
    if observation_hashes is not None and current_hashes != observation_hashes:
        raise ValueError(f"synthetic observations differ across optimizer seeds: {root}")
    trial_rows = _read_csv(root / "all_trials.csv")
    expected_pairs = {(case_id, method) for case_id in current_cases for method in METHODS}
    actual_pairs = {(row["case_id"], row["method"]) for row in trial_rows}
    if actual_pairs != expected_pairs:
        raise ValueError(f"case/method coverage mismatch in {root}")
    for case_id, method in expected_pairs:
        trial_ids = sorted(
            int(row["trial"])
            for row in trial_rows
            if row["case_id"] == case_id and row["method"] == method
        )
        if trial_ids != list(range(1, trials + 1)):
            raise ValueError(f"incomplete trial sequence for {case_id}/{method} in {root}")
    return current_cases, current_hashes


def _first_hit(rows: list[dict[str, Any]], target_ratio: float) -> int | None:
    for row in rows:
        if float(row["ratio_to_true_input"]) <= target_ratio:
            return int(row["trial"])
    return None


def _aggregate(
    roots: dict[int, Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    all_summary: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    attainment: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    execution_wall_s = 0.0
    for optimizer_seed, root in roots.items():
        summary = _read_csv(root / "case_method_summary.csv")
        trials = _read_csv(root / "all_trials.csv")
        case_meta = _read_csv(root / "case_metadata.csv")
        run_meta = _read_json(root / "run_metadata.json")
        execution_wall_s += float(run_meta["three_measurement_execution_wall_s"])
        truth_input_by_case = {
            row["case_id"]: float(row["truth_input_sensor_loss"]) for row in case_meta
        }
        for row in summary:
            all_summary.append({"optimizer_seed": optimizer_seed, **row})
        for row in trials:
            enriched = {"optimizer_seed": optimizer_seed, **row}
            all_trials.append(enriched)
        for case_id, true_input_loss in truth_input_by_case.items():
            measurement_id = next(
                row["measurement_id"] for row in case_meta if row["case_id"] == case_id
            )
            for method in METHODS:
                rows = sorted(
                    [
                        row
                        for row in trials
                        if row["case_id"] == case_id and row["method"] == method
                    ],
                    key=lambda row: int(row["trial"]),
                )
                history: list[dict[str, Any]] = []
                for row in rows:
                    ratio = float(row["best_loss"]) / true_input_loss
                    record = {
                        "optimizer_seed": optimizer_seed,
                        "measurement_id": measurement_id,
                        "case_id": case_id,
                        "method": method,
                        "trial": int(row["trial"]),
                        "best_loss": float(row["best_loss"]),
                        "true_input_sensor_loss": true_input_loss,
                        "ratio_to_true_input": ratio,
                    }
                    history.append(record)
                    all_history.append(record)
                practical_hit = _first_hit(history, PRACTICAL_RATIO)
                strict_hit = _first_hit(history, STRICT_RATIO)
                summary_row = next(
                    row for row in summary if row["case_id"] == case_id and row["method"] == method
                )
                attainment.append(
                    {
                        "optimizer_seed": optimizer_seed,
                        "measurement_id": measurement_id,
                        "case_id": case_id,
                        "method": method,
                        "final_ratio_to_true_input": float(history[-1]["ratio_to_true_input"]),
                        "hit_ratio_1p10": practical_hit is not None,
                        "first_hit_ratio_1p10": practical_hit if practical_hit is not None else "",
                        "hit_ratio_1p00": strict_hit is not None,
                        "first_hit_ratio_1p00": strict_hit if strict_hit is not None else "",
                        "last_improvement_trial": int(summary_row["best_trial"]),
                        "profile_truth_rel_l2": float(summary_row["profile_truth_rel_l2"]),
                        "fullfield_truth_rel_l2": float(summary_row["fullfield_truth_rel_l2"]),
                    }
                )
    return all_summary, all_trials, all_history, attainment, execution_wall_s


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _seed_cluster_bootstrap_interval(seed_rates: list[float]) -> tuple[float, float]:
    """Conditional interval over optimizer seeds; the three cases stay clustered."""
    if not seed_rates:
        return 0.0, 0.0
    values = np.asarray(seed_rates, dtype=float)
    rng = np.random.default_rng(20260715)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _method_aggregate(attainment: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in attainment if row["method"] == method]
        hits = [int(row["first_hit_ratio_1p10"]) for row in rows if str(row["first_hit_ratio_1p10"]).strip()]
        strict_hits = [int(row["first_hit_ratio_1p00"]) for row in rows if str(row["first_hit_ratio_1p00"]).strip()]
        final_ratios = [float(row["final_ratio_to_true_input"]) for row in rows]
        profile_errors = [float(row["profile_truth_rel_l2"]) for row in rows]
        field_errors = [float(row["fullfield_truth_rel_l2"]) for row in rows]
        capped_cost = sum(hit if hit is not None else max(BUDGETS) for hit in [
            int(row["first_hit_ratio_1p10"]) if str(row["first_hit_ratio_1p10"]).strip() else None
            for row in rows
        ])
        record: dict[str, Any] = {
            "method": method,
            "n_case_seed_runs": len(rows),
            "ratio_1p10_successes": len(hits),
            "ratio_1p10_success_rate": len(hits) / len(rows),
            "ratio_1p10_hit_trial_median": float(np.median(hits)) if hits else "",
            "ratio_1p10_hit_trial_q25": _percentile(hits, 25.0) if hits else "",
            "ratio_1p10_hit_trial_q75": _percentile(hits, 75.0) if hits else "",
            "ratio_1p10_ert_trials": capped_cost / len(hits) if hits else "",
            "ratio_1p00_successes": len(strict_hits),
            "ratio_1p00_success_rate": len(strict_hits) / len(rows),
            "final_ratio_median": float(np.median(final_ratios)),
            "final_ratio_q25": _percentile(final_ratios, 25.0),
            "final_ratio_q75": _percentile(final_ratios, 75.0),
            "truth_profile_error_median": float(np.median(profile_errors)),
            "fullfield_truth_error_median": float(np.median(field_errors)),
            "mean_last_improvement_trial_not_speed_metric": float(
                np.mean([int(row["last_improvement_trial"]) for row in rows])
            ),
        }
        for budget in BUDGETS:
            trajectories = sorted({
                (int(row["optimizer_seed"]), str(row["case_id"]))
                for row in history
                if row["method"] == method
            })
            success = 0
            successes_by_seed: dict[int, list[bool]] = {}
            for optimizer_seed, case_id in trajectories:
                at_budget = next(
                    row
                    for row in history
                    if row["method"] == method
                    and int(row["optimizer_seed"]) == optimizer_seed
                    and row["case_id"] == case_id
                    and int(row["trial"]) == budget
                )
                attained = float(at_budget["ratio_to_true_input"]) <= PRACTICAL_RATIO
                success += attained
                successes_by_seed.setdefault(optimizer_seed, []).append(attained)
            seed_rates = [
                float(np.mean(successes_by_seed[seed])) for seed in sorted(successes_by_seed)
            ]
            low, high = _seed_cluster_bootstrap_interval(seed_rates)
            record[f"successes_at_{budget}"] = success
            record[f"success_rate_at_{budget}"] = success / len(trajectories)
            record[f"success_rate_at_{budget}_seed_bootstrap_low"] = low
            record[f"success_rate_at_{budget}_seed_bootstrap_high"] = high
        output.append(record)
    return output


def _plot_quality_history(out_root: Path, history: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method in METHODS:
        medians: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for trial in range(1, 101):
            values = [
                float(row["ratio_to_true_input"])
                for row in history
                if row["method"] == method and int(row["trial"]) == trial
            ]
            medians.append(float(np.median(values)))
            q25.append(_percentile(values, 25.0))
            q75.append(_percentile(values, 75.0))
        xs = np.arange(1, 101)
        ax.plot(xs, medians, color=COLORS[method], linewidth=2.0, label=method)
        ax.fill_between(xs, q25, q75, color=COLORS[method], alpha=0.16)
    ax.axhline(PRACTICAL_RATIO, color="black", linestyle="--", label="target 1.10")
    ax.axhline(STRICT_RATIO, color="black", linestyle=":", label="strict 1.00")
    ax.set_yscale("log")
    ax.set_xlabel("CPU FNO evaluations")
    ax.set_ylabel("best loss / true-input FNO loss")
    ax.set_title("Five optimizer seeds x three fixed measurements")
    ax.grid(which="both", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "multiseed_common_quality.png", dpi=180)
    plt.close(fig)


def _plot_hit_rates(out_root: Path, method_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(BUDGETS), dtype=float)
    width = 0.24
    for method_index, method in enumerate(METHODS):
        row = next(item for item in method_rows if item["method"] == method)
        rates = [float(row[f"success_rate_at_{budget}"]) for budget in BUDGETS]
        lows = [
            float(row[f"success_rate_at_{budget}_seed_bootstrap_low"]) for budget in BUDGETS
        ]
        highs = [
            float(row[f"success_rate_at_{budget}_seed_bootstrap_high"]) for budget in BUDGETS
        ]
        errors = np.asarray([[rate - low for rate, low in zip(rates, lows, strict=True)], [high - rate for rate, high in zip(rates, highs, strict=True)]])
        ax.bar(
            x + (method_index - 1) * width,
            rates,
            width=width,
            yerr=errors,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(x, [str(budget) for budget in BUDGETS])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("trial budget per measurement")
    ax.set_ylabel("R <= 1.10 attainment rate (15 runs)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "multiseed_attainment_rate.png", dpi=180)
    plt.close(fig)


def _plot_final_quality(out_root: Path, attainment: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    specs = (
        ("final_ratio_to_true_input", "final loss / true-input FNO loss", (PRACTICAL_RATIO, STRICT_RATIO)),
        ("profile_truth_rel_l2", "truth-profile error [%]", ()),
        ("fullfield_truth_rel_l2", "full-field truth error [%]", ()),
    )
    for ax, (key, ylabel, reference_lines) in zip(axes, specs, strict=True):
        data = []
        for method in METHODS:
            values = [float(row[key]) for row in attainment if row["method"] == method]
            if key != "final_ratio_to_true_input":
                values = [100.0 * value for value in values]
            data.append(values)
        box = ax.boxplot(data, tick_labels=METHODS, patch_artist=True, showfliers=False)
        for patch, method in zip(box["boxes"], METHODS, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.45)
        rng = np.random.default_rng(20260715)
        for index, (method, values) in enumerate(zip(METHODS, data, strict=True), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(index + jitter, values, s=14, color=COLORS[method], alpha=0.75)
        for line_index, value in enumerate(reference_lines):
            ax.axhline(value, color="black", linestyle="--" if line_index == 0 else ":")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_root / "multiseed_final_quality.png", dpi=180)
    plt.close(fig)


def _write_index(
    out_root: Path,
    method_rows: list[dict[str, Any]],
    execution_wall_s: float,
) -> None:
    def optional_number(value: Any, digits: int = 1) -> str:
        return f"{float(value):.{digits}f}" if str(value).strip() else "--"

    lines = [
        "# GEC-CCP optimizer seed study on three fixed measurements",
        "",
        "- 仮想計測3件と5%点ノイズdrawは全optimizer seedで完全に同一です。",
        "- optimizer seedのみ5本に変え、各手法15 run（3ケース x 5 seed）を評価しました。",
        "- 主指標Rはbest sensor loss / true-input FNO sensor lossです。共通targetはR<=1.10、厳密targetはR<=1.00です。",
        "- 各手法自身の異なる最終bestの更新時刻はlast improvementであり、速度比較には使用しません。",
        "- optimizer設定はtestケースを見た後に変更せず固定しました。",
        "- 成功率の誤差棒は、同じoptimizer seedの3ケースを一つのclusterとして再標本化した95% bootstrap区間です。",
        "",
        "## Common-quality results",
        "",
        "| method | R<=1.10 success | ERT [trial] | hit median, successes only [trial] | R<=1.00 success | final R median [IQR] | truth-profile median [%] | full-field median [%] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {row['ratio_1p10_successes']}/{row['n_case_seed_runs']} "
            f"({100.0 * float(row['ratio_1p10_success_rate']):.1f}%) | "
            f"{optional_number(row['ratio_1p10_ert_trials'])} | "
            f"{optional_number(row['ratio_1p10_hit_trial_median'])} | "
            f"{row['ratio_1p00_successes']}/{row['n_case_seed_runs']} | "
            f"{float(row['final_ratio_median']):.4f} "
            f"[{float(row['final_ratio_q25']):.4f}, {float(row['final_ratio_q75']):.4f}] | "
            f"{100.0 * float(row['truth_profile_error_median']):.4f} | "
            f"{100.0 * float(row['fullfield_truth_error_median']):.4f} |"
        )
    lines.extend(
        [
            "",
            "ERT includes the 100-trial censoring cost for runs that did not attain R<=1.10. "
            "The hit median is conditional on success and must not be ranked without the success rate.",
        ]
    )
    lines.extend(
        [
            "",
            "## Attainment by budget",
            "",
            "| method | trial 25 | trial 50 | trial 100 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in method_rows:
        total = row["n_case_seed_runs"]
        lines.append(
            f"| `{row['method']}` | {row['successes_at_25']}/{total} | "
            f"{row['successes_at_50']}/{total} | {row['successes_at_100']}/{total} |"
        )
    lines.extend(
        [
            "",
            "## Graphs",
            "",
            "- [multiseed_common_quality.png](multiseed_common_quality.png)",
            "- [multiseed_attainment_rate.png](multiseed_attainment_rate.png)",
            "- [multiseed_final_quality.png](multiseed_final_quality.png)",
            "",
            "## Tables",
            "",
            "- [method_aggregate.csv](method_aggregate.csv)",
            "- [quality_attainment.csv](quality_attainment.csv)",
            "- [multiseed_case_method_summary.csv](multiseed_case_method_summary.csv)",
            "- [multiseed_trials.csv](multiseed_trials.csv)",
            "- [multiseed_quality_history.csv](multiseed_quality_history.csv)",
            "- [run_metadata.json](run_metadata.json)",
            "",
            f"Total five-seed three-measurement execution wall time: {execution_wall_s:.2f} s",
        ]
    )
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--noise-seed", type=int, default=20260715)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 20))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.trials != 100:
        raise ValueError("this fixed-budget study requires --trials=100")
    started = time.perf_counter()
    args.out_root.mkdir(parents=True, exist_ok=True)
    roots = {seed: _seed_root(args.out_root, seed, args.baseline_root) for seed in SEEDS}
    cases: list[str] | None = None
    hashes: dict[str, str] | None = None
    for seed in SEEDS:
        _run_seed(
            root=roots[seed],
            optimizer_seed=seed,
            noise_seed=int(args.noise_seed),
            trials=int(args.trials),
            cpu_threads=int(args.cpu_threads),
            reuse_existing=bool(args.reuse_existing),
        )
        cases, hashes = _validate_seed_root(
            roots[seed],
            optimizer_seed=seed,
            noise_seed=int(args.noise_seed),
            trials=int(args.trials),
            cases=cases,
            observation_hashes=hashes,
        )
    summary, trials, history, attainment, execution_wall_s = _aggregate(roots)
    method_rows = _method_aggregate(attainment, history)
    _write_csv(args.out_root / "multiseed_case_method_summary.csv", summary)
    _write_csv(args.out_root / "multiseed_trials.csv", trials)
    _write_csv(args.out_root / "multiseed_quality_history.csv", history)
    _write_csv(args.out_root / "quality_attainment.csv", attainment)
    _write_csv(args.out_root / "method_aggregate.csv", method_rows)
    _plot_quality_history(args.out_root, history)
    _plot_hit_rates(args.out_root, method_rows)
    _plot_final_quality(args.out_root, attainment)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_seeds": list(SEEDS),
        "noise_seed_base": int(args.noise_seed),
        "point_noise_rel": 0.05,
        "calibration_noise_rel": 0.0,
        "measurements_are_identical_across_optimizer_seeds": True,
        "case_ids": cases,
        "trials_per_method_per_measurement": int(args.trials),
        "primary_target_ratio": PRACTICAL_RATIO,
        "strict_target_ratio": STRICT_RATIO,
        "total_fno_optimizer_evaluations": len(SEEDS) * 3 * 3 * int(args.trials),
        "total_measurement_execution_wall_s": execution_wall_s,
        "latest_orchestration_wall_s": time.perf_counter() - started,
        "seed_roots": {str(seed): str(root) for seed, root in roots.items()},
    }
    _write_json(args.out_root / "run_metadata.json", metadata)
    _write_index(args.out_root, method_rows, execution_wall_s)
    print(args.out_root / "index.md")


if __name__ == "__main__":
    main()
