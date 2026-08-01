"""Summarize staged ICP spatial-learning runs against model-matched baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


TARGETS = ("ne", "ni", "Te", "phi")
CANDIDATES = {
    "transform_case_balance": ("transform_case_balance", "unet", "U-Net compact"),
    "smooth_structure": ("smooth_structure", "unet", "U-Net compact"),
    "smooth_spatial_loss": ("smooth_spatial_loss", "unet", "U-Net compact"),
    "relative_spatial_objective": ("relative_spatial_objective", "unet", "U-Net compact"),
    "uno_relative_spatial_objective": (
        "uno_relative_spatial_objective",
        "u_no",
        "UNO m10",
    ),
}
VARIANTS = tuple(CANDIDATES)


def _baseline_label(model_name: str) -> str:
    return "baseline_" + str(model_name).lower().replace("-", "").replace(" ", "_")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _stats(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(finite)), float(np.percentile(finite, 90.0)), float(np.max(finite))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(*, run_root: Path, baseline_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_baseline_rows = _read_csv(baseline_csv)
    summary: list[dict[str, Any]] = []
    baseline_names = list(dict.fromkeys(spec[2] for spec in CANDIDATES.values()))
    for baseline_name in baseline_names:
        baseline_rows = [
            row for row in all_baseline_rows if str(row.get("model", "")) == baseline_name
        ]
        if not baseline_rows:
            continue
        for target in TARGETS:
            selected = [row for row in baseline_rows if str(row.get("target", "")) == target]
            field_stats = _stats([float(row["relative_l2"]) for row in selected])
            gradient_stats = _stats([float(row["gradient_relative_l2"]) for row in selected])
            summary.append(
                {
                    "variant": _baseline_label(baseline_name),
                    "baseline_variant": "",
                    "model_id": "",
                    "target": target,
                    "physical_rel_l2_median": field_stats[0],
                    "physical_rel_l2_p90": field_stats[1],
                    "physical_rel_l2_worst": field_stats[2],
                    "gradient_rel_l2_median": gradient_stats[0],
                    "gradient_rel_l2_p90": gradient_stats[1],
                    "gradient_rel_l2_worst": gradient_stats[2],
                    "physical_bound_fraction_worst": float("nan"),
                    "transformed_huber_worst": float("nan"),
                }
            )
    for variant, (run_dir, model_id, baseline_name) in CANDIDATES.items():
        path = (
            run_root
            / run_dir
            / f"models/{model_id}/eval_protocol/structure_holdout/eval/spatial_distribution_by_case.csv"
        )
        rows = _read_csv(path)
        for target in TARGETS:
            selected = [row for row in rows if str(row.get("var", "")) == target]
            if not selected:
                continue
            field_stats = _stats([float(row["physical_rel_l2"]) for row in selected])
            gradient_stats = _stats([float(row["gradient_rel_l2"]) for row in selected])
            bound_stats = _stats([float(row["physical_bound_fraction"]) for row in selected])
            transformed_stats = _stats(
                [float(row["transformed_huber"]) for row in selected if str(row.get("transformed_huber", ""))]
            )
            summary.append(
                {
                    "variant": variant,
                    "baseline_variant": _baseline_label(baseline_name),
                    "model_id": model_id,
                    "target": target,
                    "physical_rel_l2_median": field_stats[0],
                    "physical_rel_l2_p90": field_stats[1],
                    "physical_rel_l2_worst": field_stats[2],
                    "gradient_rel_l2_median": gradient_stats[0],
                    "gradient_rel_l2_p90": gradient_stats[1],
                    "gradient_rel_l2_worst": gradient_stats[2],
                    "physical_bound_fraction_worst": bound_stats[2],
                    "transformed_huber_worst": transformed_stats[2],
                }
            )
    acceptance: list[dict[str, Any]] = []
    for variant in VARIANTS:
        current = {row["target"]: row for row in summary if row["variant"] == variant}
        if set(current) != set(TARGETS):
            acceptance.append({"variant": variant, "complete": False, "accepted": False, "score": float("inf")})
            continue
        baseline_variant = str(next(iter(current.values()))["baseline_variant"])
        baseline = {
            row["target"]: row for row in summary if row["variant"] == baseline_variant
        }
        if set(baseline) != set(TARGETS):
            acceptance.append({"variant": variant, "complete": False, "accepted": False, "score": float("inf")})
            continue
        density_median = all(
            float(current[target]["gradient_rel_l2_median"])
            <= 0.75 * float(baseline[target]["gradient_rel_l2_median"])
            for target in ("ne", "ni")
        )
        density_p90 = all(
            float(current[target]["gradient_rel_l2_p90"])
            <= 0.80 * float(baseline[target]["gradient_rel_l2_p90"])
            for target in ("ne", "ni")
        )
        field_gate = all(
            float(current[target]["physical_rel_l2_median"])
            <= 1.05 * float(baseline[target]["physical_rel_l2_median"])
            for target in TARGETS
        )
        thermal_gate = all(
            float(current[target]["physical_rel_l2_median"])
            <= 1.10 * float(baseline[target]["physical_rel_l2_median"])
            for target in ("Te", "phi")
        )
        bound_gate = all(
            float(current[target]["physical_bound_fraction_worst"]) < 1.0e-3
            for target in TARGETS
        )
        score = float(
            np.mean(
                [
                    float(current[target]["physical_rel_l2_median"])
                    / max(float(baseline[target]["physical_rel_l2_median"]), 1.0e-12)
                    + float(current[target]["gradient_rel_l2_median"])
                    / max(float(baseline[target]["gradient_rel_l2_median"]), 1.0e-12)
                    for target in TARGETS
                ]
            )
        )
        acceptance.append(
            {
                "variant": variant,
                "complete": True,
                "density_gradient_median_gate": density_median,
                "density_gradient_p90_gate": density_p90,
                "field_regression_gate": field_gate,
                "thermal_potential_gate": thermal_gate,
                "bound_saturation_gate": bound_gate,
                "accepted": bool(density_median and density_p90 and field_gate and thermal_gate and bound_gate),
                "score": score,
            }
        )
    return summary, acceptance


def _plot(summary: list[dict[str, Any]], *, out_dir: Path, run_root: Path) -> None:
    import matplotlib.pyplot as plt

    variants = [
        *[_baseline_label(name) for name in dict.fromkeys(spec[2] for spec in CANDIDATES.values())],
        *VARIANTS,
    ]
    available = [variant for variant in variants if any(row["variant"] == variant for row in summary)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(TARGETS), dtype=np.float64)
    width = 0.8 / max(len(available), 1)
    for idx, variant in enumerate(available):
        by_target = {row["target"]: row for row in summary if row["variant"] == variant}
        axes[0].bar(x + idx * width, [by_target[t]["physical_rel_l2_median"] for t in TARGETS], width, label=variant)
        axes[1].bar(x + idx * width, [by_target[t]["gradient_rel_l2_median"] for t in TARGETS], width, label=variant)
    for axis, title in zip(axes, ("Physical relative L2 (median)", "Gradient relative L2 (median)"), strict=True):
        axis.set_title(title)
        axis.set_xticks(x + width * (len(available) - 1) / 2, TARGETS)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(out_dir / "spatial_metric_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for variant in VARIANTS:
        run_dir, model_id, _baseline_name = CANDIDATES[variant]
        metrics_path = run_root / run_dir / f"models/{model_id}/eval_protocol/structure_holdout/train/scalars/metrics.csv"
        rows = _read_csv(metrics_path)
        if rows:
            axis.plot([float(row["epoch"]) for row in rows], [float(row["val_loss"]) for row in rows], label=variant)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation loss")
    axis.set_title("Case-balanced validation loss")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(out_dir / "loss_history_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/icp_stage4_spatial_learning_v1"))
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("runs/icp_stage4_selected_models_spatial_truth_pred_error/per_case_target_metrics.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_root / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, acceptance = build_summary(run_root=args.run_root, baseline_csv=args.baseline_csv)
    _write_csv(out_dir / "metric_summary.csv", summary)
    _write_csv(out_dir / "acceptance.csv", acceptance)
    _plot(summary, out_dir=out_dir, run_root=args.run_root)
    accepted = [row for row in acceptance if bool(row.get("accepted", False))]
    complete = [row for row in acceptance if bool(row.get("complete", False))]
    candidates = accepted or complete
    winner = min(candidates, key=lambda row: float(row["score"]))["variant"] if candidates else "pending"
    quality_line = (
        "- Numerical quality gate: passed; visual field review remains required before downstream optimization."
        if accepted
        else "- Numerical quality gate: not yet passed; downstream shape optimization remains blocked."
    )
    lines = [
        "# ICP Stage4 spatial-learning validation",
        "",
        f"- Recommended completed variant: `{winner}`",
        f"- Completed variants: {len(complete)}/{len(VARIANTS)}",
        quality_line,
        "",
        "![Spatial metrics](spatial_metric_comparison.png)",
        "",
        "![Loss history](loss_history_comparison.png)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "index.md")


if __name__ == "__main__":
    main()
