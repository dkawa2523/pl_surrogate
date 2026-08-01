"""Train ICP grid surrogates with a coil-generation-ready input contract.

The recipe intentionally changes only the representation contract and UNO's
FFT boundary treatment.  Target transforms, spatial loss, split, seed, and
evaluation remain identical to the validated spatial-learning-v2 runs.

The optional ``--linear-targets`` recipe is a deliberately minimal ablation:
all four physical fields stay linear, receive only train-split z-score scaling,
and use one case-balanced pointwise MSE loss.  It exists to test the target
transform itself without introducing a new architecture or auxiliary loss.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SOURCE_CONFIGS = {
    "unet": Path(
        "configs/experimental/icp_stage4/generated_spatial_learning_v2/"
        "benchmark_icp_stage4_relative_spatial_objective.yaml"
    ),
    "u_no": Path(
        "configs/experimental/icp_stage4/generated_spatial_learning_v2/"
        "benchmark_icp_stage4_uno_relative_spatial_objective.yaml"
    ),
}
MODEL_RUN_NAMES = {"unet": "unet", "u_no": "uno"}
CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_coil_structure_v1")
RUN_ROOT = Path("runs/icp_stage4_coil_structure_v1")
LINEAR_CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_linear_targets_v1")
LINEAR_RUN_ROOT = Path("runs/icp_stage4_linear_targets_v1")
PHYSICAL_CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_axisymmetric_energy_weighted_v1")
PHYSICAL_RUN_ROOT = Path("runs/icp_stage4_axisymmetric_energy_weighted_v1")
BOHM_OPT_CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_simple_bohm_wafer_opt_v1")
# Companion inference configs consume the checkpoints trained by the physical
# weighting recipe; they must not create a second copy of the same model.
BOHM_OPT_RUN_ROOT = PHYSICAL_RUN_ROOT
FEATURE_PROFILE = "part_source_v1"
FEATURE_CHANNELS = ["x", "y", "distance_signed", "part_sdf_union", "part_source_sum"]
PROCESS_CONDITIONS = ["pp", "pp0"]
TARGETS = ["ne", "ni", "Te", "phi"]


def _use_linear_targets(bench: dict[str, Any], *, model_name: str) -> None:
    """Keep targets in physical space and reduce training to one plain loss."""

    scalers = bench["preprocessing"]["scalers"]
    scalers["enforce_target_transforms"] = True
    scalers["target_transforms"] = {
        name: {
            "value_transform": "identity",
            "scaler": "zscore",
            "fit_scope": "plasma_only",
            "clip": {"mode": "none"},
        }
        for name in TARGETS
    }

    equal_weights = {name: 1.0 for name in TARGETS}
    loss = bench["train"].setdefault("loss", {})
    loss["multitask"] = {
        "weighting": "fixed",
        "fixed_weights_by_var": dict(equal_weights),
    }
    loss["supervised"] = {
        "type": "mse",
        "mask": "plasma_only",
        "nan_region_policy": "mask_only",
        "normalization": "sample_mean",
        "target_weights": dict(equal_weights),
    }
    bench["train"][model_name]["selection"] = {"mode": "best_val_loss"}


def _use_axisymmetric_energy_weighting(bench: dict[str, Any], *, model_name: str) -> None:
    """Apply the minimal axisymmetric plasma weighting on top of linear targets."""

    _use_linear_targets(bench, model_name=model_name)
    bench["train"]["loss"]["supervised"]["physical_weighting"] = {
        "axisymmetric_volume": True,
        "density_source": "ne",
        "density_weighted_targets": ["Te"],
    }
    bench["metadata"].update(
        {
            "study": "icp_axisymmetric_energy_weighted_v1",
            "target_space": "physical_linear_zscore",
            "training_objective": "axisymmetric_volume_and_electron_energy_weighted_mse",
        }
    )


def _use_simple_bohm_optimization(bench: dict[str, Any], *, model_name: str) -> None:
    """Configure one transparent joint process/coil optimization problem."""

    _use_axisymmetric_energy_weighting(bench, model_name=model_name)
    inference = bench.setdefault("inference", {})
    inference["qoi"] = {
        "uniformity": {
            "target": "ni",
            "preferred_targets": ["ni", "ne", "Te"],
            "region": "wafer_near",
            "mid_height_band_px": 2,
        },
        "bohm_flux": {
            "density_target": "ni",
            "temperature_target": "Te",
        },
    }
    inference["single"] = {
        "enabled": True,
        "case_id": "case_g002_op01_reference",
        "cond": {"pp": 2823.804, "pp0": 0.08349},
        "geom": {"geom_id": "default"},
        "axis": {"mode": "steady", "value": 0.0},
    }
    inference["optimize"] = {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 192,
        "seed": 411,
        "space": {
            "pp": [586.66255, 2902.9775],
            "pp0": [0.00385545, 0.0867409],
        },
        "geom": {"geom_id": "default"},
        "geom_space": {
            f"part.coil_{index:02d}.tx": [-0.015, 0.015]
            for index in range(1, 7)
        },
        "backend_cfg": {
            "sampler": "tpe",
            "n_startup_trials": 32,
            "multivariate": False,
        },
        "objective": {
            "mode": "weighted_sum",
            "terms": [
                {"key": "bohm_flux_area_cv", "direction": "min", "weight": 1.0}
            ],
        },
        "constraints": [
            {"key": "bohm_flux_area_mean", "lower": 6.0249874e16}
        ],
        "output": {"save_fields": "top_k", "top_k": 5},
    }
    bench["metadata"].update(
        {
            "study": "icp_simple_bohm_wafer_opt_v1",
            "optimization_variables": "pp_pp0_and_six_coil_radial_shifts",
            "optimization_objective": "wafer_near_axisymmetric_bohm_flux_cv",
            "reference_case_id": "case_g002_op01",
            "reference_bohm_flux_area_mean": 7.5312343e16,
            "minimum_bohm_flux_area_mean_ratio": 0.80,
        }
    )


def build_model_config(
    source: dict[str, Any],
    *,
    model_name: str,
    run_root: Path,
    epochs: int,
    linear_targets: bool = False,
    physical_weighting: bool = False,
    simple_bohm_optimize: bool = False,
) -> dict[str, Any]:
    if model_name not in SOURCE_CONFIGS:
        raise ValueError(f"unsupported model={model_name!r}")
    cfg = deepcopy(source)
    bench = cfg["benchmark"]
    run_name = MODEL_RUN_NAMES[model_name]
    bench["output_dir"] = str(run_root / run_name).replace("\\", "/")
    bench["dataset"]["cond_columns"] = list(PROCESS_CONDITIONS)
    bench.setdefault("metadata", {}).update(
        {
            "study": "icp_coil_structure_v1",
            "representation": FEATURE_PROFILE,
            "structure_condition_columns": list(PROCESS_CONDITIONS),
            "geometry_dimensions_as_conditions": False,
        }
    )

    bench["runtime"]["input_mode"] = "table_plus_structure"
    bench["runtime"]["structure"]["feature_profile"] = FEATURE_PROFILE
    coord = bench["preprocessing"]["coord_features"]
    coord["channels_from_profile"] = FEATURE_PROFILE
    coord["require_case_variation"] = True

    model = bench["train"][model_name]
    model["epochs"] = int(epochs)
    model["input_features"]["features"] = list(FEATURE_CHANNELS)
    if model_name == "u_no":
        model["model_cfg"].setdefault("uno_cfg", {}).update(
            {"padding_fraction": 0.08, "padding_mode": "reflect"}
        )

    if simple_bohm_optimize:
        _use_simple_bohm_optimization(bench, model_name=model_name)
    elif physical_weighting:
        _use_axisymmetric_energy_weighting(bench, model_name=model_name)
    elif linear_targets:
        _use_linear_targets(bench, model_name=model_name)
        bench["metadata"].update(
            {
                "study": "icp_linear_targets_v1",
                "target_space": "physical_linear_zscore",
                "training_objective": "case_balanced_mse",
            }
        )

    protocol_name = (
        "icp_stage4_simple_bohm_wafer_opt_v1"
        if simple_bohm_optimize
        else "icp_stage4_axisymmetric_energy_weighted_v1"
        if physical_weighting
        else ("icp_stage4_linear_targets_v1" if linear_targets else "icp_stage4_coil_structure_v1")
    )
    bench["eval"]["protocol_variant"] = f"{protocol_name}_{run_name}"
    bench["eval"]["spatial_distribution_audit"] = {
        "enabled": True,
        "vars": "all",
        "top_fraction": 0.10,
        "plot_worst_cases": 3,
        "plot_representative_cases": True,
    }
    return cfg


def generate_configs(
    *,
    config_root: Path,
    run_root: Path,
    epochs: int,
    linear_targets: bool = False,
    physical_weighting: bool = False,
    simple_bohm_optimize: bool = False,
) -> dict[str, Path]:
    config_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model_name, source_path in SOURCE_CONFIGS.items():
        source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        cfg = build_model_config(
            source,
            model_name=model_name,
            run_root=run_root,
            epochs=epochs,
            linear_targets=linear_targets,
            physical_weighting=physical_weighting,
            simple_bohm_optimize=simple_bohm_optimize,
        )
        recipe = (
            "simple_bohm_wafer_opt_v1"
            if simple_bohm_optimize
            else "axisymmetric_energy_weighted_v1"
            if physical_weighting
            else ("linear_targets_v1" if linear_targets else "coil_structure_v1")
        )
        path = config_root / f"benchmark_icp_stage4_{recipe}_{MODEL_RUN_NAMES[model_name]}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        paths[model_name] = path
    return paths


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "status", "seconds", "config", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)


def run_configs(paths: dict[str, Path], *, run_root: Path, models: list[str]) -> None:
    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    from plasma_surrogate.benchmark.runner import BenchmarkRunner

    rows: list[dict[str, Any]] = []
    status_path = run_root / "status.csv"
    for model_name in models:
        output_dir = run_root / MODEL_RUN_NAMES[model_name]
        start = time.perf_counter()
        status = "passed"
        try:
            BenchmarkRunner.from_yaml(paths[model_name]).run()
        except BaseException:
            status = "failed"
            raise
        finally:
            rows.append(
                {
                    "model": model_name,
                    "status": status,
                    "seconds": time.perf_counter() - start,
                    "config": paths[model_name],
                    "output_dir": output_dir,
                }
            )
            _write_status(status_path, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(*, run_root: Path) -> Path:
    """Write one compact numerical/visual comparison for completed models."""

    import matplotlib.pyplot as plt
    import numpy as np

    specs = {
        "unet": run_root / "unet/models/unet/eval_protocol/structure_holdout",
        "uno": run_root / "uno/models/u_no/eval_protocol/structure_holdout",
    }
    out_dir = run_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for model_name, root in specs.items():
        for row in _read_csv(root / "eval/spatial_distribution_summary.csv"):
            metric_rows.append(
                {
                    "model": model_name,
                    "var": row["var"],
                    "physical_rel_l2_median": float(row["physical_rel_l2_median"]),
                    "physical_rel_l2_p90": float(row["physical_rel_l2_p90"]),
                    "gradient_rel_l2_median": float(row["gradient_rel_l2_median"]),
                    "gradient_rel_l2_p90": float(row["gradient_rel_l2_p90"]),
                    "physical_bound_fraction_max": float(row["physical_bound_fraction_max"]),
                }
            )
    _write_csv(out_dir / "metric_summary.csv", metric_rows)

    x = np.arange(len(TARGETS), dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for index, model_name in enumerate(specs):
        by_var = {row["var"]: row for row in metric_rows if row["model"] == model_name}
        offset = (index - 0.5) * 0.35
        axes[0].bar(
            x + offset,
            [by_var[var]["physical_rel_l2_median"] for var in TARGETS],
            0.35,
            label=model_name,
        )
        axes[1].bar(
            x + offset,
            [by_var[var]["gradient_rel_l2_median"] for var in TARGETS],
            0.35,
            label=model_name,
        )
    for axis, title in zip(
        axes,
        ("Physical relative L2 (case median)", "Gradient relative L2 (case median)"),
        strict=True,
    ):
        axis.set_xticks(x, TARGETS)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    fig.savefig(out_dir / "spatial_metric_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for model_name, root in specs.items():
        rows = _read_csv(root / "train/scalars/metrics.csv")
        axis.plot(
            [float(row["epoch"]) for row in rows],
            [float(row["val_loss"]) for row in rows],
            label=model_name,
        )
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation loss")
    axis.set_title("Case-balanced spatial validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(out_dir / "loss_history_comparison.png", dpi=160)
    plt.close(fig)

    lines = [
        "# ICP コイル構造サロゲート：U-Net / UNO 空間分布評価",
        "",
        "- 入力: `part_source_v1`（座標・装置境界SDF・コイルunion SDF・加算型source場）",
        "- 条件: `pp`, `pp0` のみ。構造寸法を条件ベクトルへ与えていない。",
        "- 結論: 両モデルとも収束したが、線状・波状残差が残るため形状最適化への採用は保留。",
        "",
        "## 学習履歴と空間指標",
        "",
        "![Loss history](loss_history_comparison.png)",
        "",
        "![Spatial metrics](spatial_metric_comparison.png)",
        "",
        "| model | var | physical rel-L2 median | p90 | gradient rel-L2 median | p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in specs:
        by_var = {row["var"]: row for row in metric_rows if row["model"] == model_name}
        for var in TARGETS:
            row = by_var[var]
            lines.append(
                f"| {model_name} | {var} | {row['physical_rel_l2_median']:.4f} | "
                f"{row['physical_rel_l2_p90']:.4f} | {row['gradient_rel_l2_median']:.4f} | "
                f"{row['gradient_rel_l2_p90']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 確認された課題",
            "",
            "- U-Net: 密度分布の絶対誤差に細い網状線が残る。relative L2はUNOより良いが、密度勾配誤差はUNOより大きい。",
            "- UNO: 反射paddingで強い境界不連続は弱まったが、内部に波状・横縞状の残差が残る。phiの誤差はU-Netより大きい。",
            "- 共通: 最悪低密度ケースでは真値ノルムが極小になりrelative指標が発散する。median/p90と絶対誤差図を併用する必要がある。",
            "- 入力: `part_source_sum` がプラズマ領域でほぼゼロになり、モデルがmedial-axisを持つunion SDFへ依存している。",
        ]
    )
    for model_name, root in specs.items():
        lines.extend(["", f"## {model_name.upper()}：真値・予測・絶対誤差", ""])
        plot_dir = root / "eval/plots"
        for level in ("median", "p90", "worst"):
            lines.extend([f"### {level}", ""])
            for var in TARGETS:
                matches = sorted(plot_dir.glob(f"spatial_distribution_{level}_{var}_*.png"))
                if not matches:
                    continue
                relative = matches[0].relative_to(run_root).as_posix()
                lines.extend([f"**{var}**", "", f"![{model_name} {level} {var}](../{relative})", ""])
    index_path = out_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--models", nargs="+", choices=tuple(SOURCE_CONFIGS), default=list(SOURCE_CONFIGS))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--linear-targets", action="store_true")
    parser.add_argument("--axisymmetric-energy-weighted", action="store_true")
    parser.add_argument("--simple-bohm-optimize", action="store_true")
    args = parser.parse_args()
    simple_bohm_optimize = bool(args.simple_bohm_optimize)
    physical_weighting = bool(args.axisymmetric_energy_weighted or simple_bohm_optimize)
    linear_targets = bool(args.linear_targets or physical_weighting)
    config_root = args.config_root or (
        BOHM_OPT_CONFIG_ROOT
        if simple_bohm_optimize
        else PHYSICAL_CONFIG_ROOT
        if physical_weighting
        else (LINEAR_CONFIG_ROOT if linear_targets else CONFIG_ROOT)
    )
    run_root = args.run_root or (
        BOHM_OPT_RUN_ROOT
        if simple_bohm_optimize
        else PHYSICAL_RUN_ROOT
        if physical_weighting
        else (LINEAR_RUN_ROOT if linear_targets else RUN_ROOT)
    )
    paths = generate_configs(
        config_root=config_root,
        run_root=run_root,
        epochs=args.epochs,
        linear_targets=linear_targets,
        physical_weighting=physical_weighting,
        simple_bohm_optimize=simple_bohm_optimize,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.run:
        run_configs(paths, run_root=run_root, models=list(args.models))
    if args.summarize:
        print(summarize_runs(run_root=run_root))


if __name__ == "__main__":
    main()
