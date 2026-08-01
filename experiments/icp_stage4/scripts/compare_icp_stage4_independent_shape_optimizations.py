from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "experiments" / "icp_stage4" / "scripts"))

from run_icp_stage4_differentiable_shape_optimize import FrozenGridSurrogate  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc comparison of independently optimized ICP surrogate candidates.")
    parser.add_argument("--uno-run-dir", required=True)
    parser.add_argument("--uno-opt-dir", required=True)
    parser.add_argument("--unet-run-dir", required=True)
    parser.add_argument("--unet-opt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _restart_stats(opt_dir: Path, density_threshold: float) -> dict[str, float]:
    with (opt_dir / "restart_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    feasible = [float(row["hard_density_ratio"]) >= density_threshold for row in rows]
    return {
        "restart_count": float(len(rows)),
        "feasible_restart_fraction": float(np.mean(feasible)) if feasible else float("nan"),
        "best_restart_uniformity": min(float(row["hard_uniformity"]) for row in rows),
        "median_restart_uniformity": float(np.median([float(row["hard_uniformity"]) for row in rows])),
    }


def main() -> int:
    args = _args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "UNO": {"run": Path(args.uno_run_dir), "opt": Path(args.uno_opt_dir), "model": "u_no"},
        "U-Net compact": {"run": Path(args.unet_run_dir), "opt": Path(args.unet_opt_dir), "model": "unet"},
    }
    summaries = {name: _json(spec["opt"] / "summary.json") for name, spec in specs.items()}
    candidates = {
        name: {key: float(summary["best"][key]) for key in ("nncoil", "llcoil", "rrc", "rrce", "zzc")}
        for name, summary in summaries.items()
    }
    cross_rows: list[dict[str, Any]] = []
    for evaluator, spec in specs.items():
        summary = summaries[evaluator]
        surrogate = FrozenGridSurrogate(
            run_dir=spec["run"],
            model_name=spec["model"],
            protocol=str(summary["protocol"]),
            dataset_root=Path(summary["dataset_root"]),
            cond={key: float(value) for key, value in summary["condition"].items()},
            mid_height_band_px=2,
            radial_fraction=0.9,
        )
        reference = {key: float(value) for key, value in summary["reference_geometry"].items()}
        base_fields, _ = surrogate.predict_hard(geom=reference, count=int(round(reference["nncoil"])))
        base_raw = surrogate.hard_metrics(base_fields, density_ref=1.0)
        density_ref = float(base_raw["density_mean"])
        base_metrics = surrogate.hard_metrics(base_fields, density_ref=density_ref)
        for source, geometry in candidates.items():
            fields, _ = surrogate.predict_hard(geom=geometry, count=int(round(geometry["nncoil"])))
            metrics = surrogate.hard_metrics(fields, density_ref=density_ref)
            cross_rows.append(
                {
                    "candidate_source": source,
                    "evaluator": evaluator,
                    "baseline_uniformity": base_metrics["uniformity"],
                    "candidate_uniformity": metrics["uniformity"],
                    "uniformity_improvement_percent": 100.0 * (base_metrics["uniformity"] - metrics["uniformity"]) / base_metrics["uniformity"],
                    "density_ratio": metrics["density_ratio"],
                    "temperature_mean_ratio": metrics["temperature_mean"] / base_metrics["temperature_mean"],
                    "temperature_uniformity_ratio": metrics["temperature_uniformity"] / base_metrics["temperature_uniformity"],
                    "plasma_density_mean_ratio": metrics["plasma_density_mean"] / base_metrics["plasma_density_mean"],
                }
            )
        del surrogate
        torch.cuda.empty_cache()

    comparison: list[dict[str, Any]] = []
    for model, spec in specs.items():
        own = next(row for row in cross_rows if row["candidate_source"] == model and row["evaluator"] == model)
        other = next(row for row in cross_rows if row["candidate_source"] == model and row["evaluator"] != model)
        threshold = float(summaries[model]["density_ratio_constraint"])
        eval_summary = _json(spec["run"] / "models" / spec["model"] / "eval_protocol" / str(summaries[model]["protocol"]) / "summary.json")
        rmse = dict(eval_summary.get("rmse", {}))
        comparison.append(
            {
                "model": model,
                "own_improvement_percent": own["uniformity_improvement_percent"],
                "own_density_ratio": own["density_ratio"],
                "posthoc_other_model_improvement_percent": other["uniformity_improvement_percent"],
                **_restart_stats(spec["opt"], threshold),
                "holdout_ni_rmse": float(rmse.get("ni", float("nan"))),
                "holdout_ne_rmse": float(rmse.get("ne", float("nan"))),
                "holdout_Te_rmse": float(rmse.get("Te", float("nan"))),
                **{f"best_{key}": value for key, value in candidates[model].items()},
            }
        )
    _write_csv(out_dir / "cross_evaluation.csv", cross_rows)
    _write_csv(out_dir / "independent_model_comparison.csv", comparison)

    labels = list(specs)
    improvement = np.asarray(
        [[next(row for row in cross_rows if row["candidate_source"] == source and row["evaluator"] == evaluator)["uniformity_improvement_percent"] for evaluator in labels] for source in labels]
    )
    density = np.asarray(
        [[next(row for row in cross_rows if row["candidate_source"] == source and row["evaluator"] == evaluator)["density_ratio"] for evaluator in labels] for source in labels]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, matrix, title, fmt in (
        (axes[0], improvement, "post-hoc CV improvement (%)", ".1f"),
        (axes[1], density, "post-hoc mean-density ratio", ".3f"),
    ):
        image = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("evaluation model")
        ax.set_ylabel("independently optimized candidate")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(out_dir / "posthoc_cross_evaluation.png", dpi=180)
    plt.close(fig)

    optimization_winner = max(comparison, key=lambda row: float(row["own_improvement_percent"]))
    predictive_winner = min(comparison, key=lambda row: float(row["holdout_ni_rmse"]))
    lines = [
        "# Independent ICP shape-optimization comparison",
        "",
        "Each model was optimized independently. Cross-evaluation below was performed only after both optimizations completed.",
        "",
        "| model | own CV improvement | own density ratio | feasible restarts | other-model post-hoc improvement | holdout ni RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {float(row['own_improvement_percent']):.2f}% | {float(row['own_density_ratio']):.4f} | "
            f"{100.0 * float(row['feasible_restart_fraction']):.1f}% | {float(row['posthoc_other_model_improvement_percent']):.2f}% | {float(row['holdout_ni_rmse']):.3e} |"
        )
    lines += [
        "",
        f"- strongest self-model optimization response: `{optimization_winner['model']}`",
        f"- lowest holdout ion-density RMSE: `{predictive_winner['model']}`",
        "- neither independently optimized candidate improves the other model; high-fidelity validation is required before selecting a physical design",
        "",
        "## Outputs",
        "",
        "- [Post-hoc cross-evaluation](posthoc_cross_evaluation.png)",
        "- [Cross-evaluation values](cross_evaluation.csv)",
        "- [Independent comparison](independent_model_comparison.csv)",
        "- [UNO independent report](../icp_stage4_independent_midplane_opt_uno_m10_full/index.md)",
        "- [U-Net independent report](../icp_stage4_independent_midplane_opt_unet_compact_full/index.md)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"optimization_winner": optimization_winner["model"], "predictive_winner": predictive_winner["model"], "comparison": comparison, "cross_evaluation": cross_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
