"""Generate and optionally run the staged ICP spatial-learning validation plan."""

from __future__ import annotations

import argparse
import csv
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SOURCE_CONFIG = Path(
    "configs/experimental/icp_stage4/generated_major_fixes_part_lite_v2_e80/"
    "full/benchmark_icp_stage4_core4_full_unet.yaml"
)
UNO_SOURCE_CONFIG = Path(
    "configs/experimental/icp_stage4/generated_major_fixes_part_lite_v2_e80_uno3/uno_tuning/"
    "benchmark_icp_stage4_uno_tuning__u_no_m10_w48_l4_lr3e4__seed411.yaml"
)
FFNO_SOURCE_CONFIG = Path(
    "configs/experimental/icp_stage4/generated_major_fixes_part_lite_v2_e80_ffno_m12_w48_l4_lr4e4/"
    "full/benchmark_icp_stage4_core4_full_ffno.yaml"
)
CONFIG_ROOT = Path("configs/experimental/icp_stage4/generated_spatial_learning_v1")
RUN_ROOT = Path("runs/icp_stage4_spatial_learning_v1")
COND_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0"]
VARIANTS = (
    "transform_case_balance",
    "smooth_structure",
    "smooth_spatial_loss",
    "relative_spatial_objective",
)


def _physical_target_transforms() -> dict[str, dict[str, Any]]:
    return {
        "ne": {
            "value_transform": "log10_floor",
            "floor": 1.0e-30,
            "scaler": "zscore",
            "fit_scope": "plasma_only",
            "clip": {"mode": "physical_bounds", "min": 1.0e8, "max": 1.0e19},
        },
        "ni": {
            "value_transform": "log10_floor",
            "floor": 1.0e-30,
            "scaler": "zscore",
            "fit_scope": "plasma_only",
            "clip": {"mode": "physical_bounds", "min": 1.0e8, "max": 1.0e19},
        },
        "Te": {
            "value_transform": "log1p",
            "floor": 0.0,
            "scaler": "robust",
            "fit_scope": "plasma_only",
            "clip": {"mode": "physical_bounds", "min": 0.0, "max": 20.0},
        },
        "phi": {
            "value_transform": "signed_log1p",
            "floor": 0.0,
            "scaler": "robust",
            "fit_scope": "plasma_only",
            "clip": {"mode": "physical_bounds", "min": -50.0, "max": 50.0},
        },
    }


def build_variant_config(source: dict[str, Any], *, variant: str, run_root: Path) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant={variant!r}; expected one of {VARIANTS}")
    cfg = deepcopy(source)
    bench = cfg.setdefault("benchmark", {})
    bench["output_dir"] = str(run_root / variant).replace("\\", "/")
    bench["seed"] = 411
    bench.setdefault("metadata", {}).update(
        {
            "study": "generic_spatial_learning_v1",
            "variant": variant,
            "structure_condition_columns": list(COND_COLUMNS),
        }
    )

    dataset = bench.setdefault("dataset", {})
    dataset["cond_columns"] = list(COND_COLUMNS)

    preprocessing = bench.setdefault("preprocessing", {})
    scalers = preprocessing.setdefault("scalers", {})
    scalers["enforce_target_transforms"] = True
    scalers["fit_split"] = "structure_holdout"
    scalers["target_transforms"] = _physical_target_transforms()

    smooth = variant in {
        "smooth_structure",
        "smooth_spatial_loss",
        "relative_spatial_objective",
    }
    feature_profile = "smooth_structure_v1" if smooth else "part_lite_v1"
    channels = [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "boundary_band",
        "solid_proximity",
    ] if smooth else [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "normal_x",
        "normal_y",
        "curvature_proxy",
        "boundary_band",
        "part_sdf_nearest",
        "part_sdf_second",
        "part_gap_proxy",
        "solid_proximity",
    ]
    coord_features = preprocessing.setdefault("coord_features", {})
    coord_features["enabled"] = True
    coord_features["channels_from_profile"] = feature_profile

    runtime = bench.setdefault("runtime", {})
    runtime["input_mode"] = "table_plus_structure"
    runtime.setdefault("structure", {})["feature_profile"] = feature_profile
    runtime["structure"]["provider_mode"] = "parametric_parts"

    train = bench.setdefault("train", {})
    unet = train.setdefault("unet", {})
    unet["epochs"] = 70
    unet["lr"] = 3.0e-4
    unet.setdefault("optimizer", {}).update(
        {"type": "adamw", "lr": 3.0e-4, "weight_decay": 1.0e-4, "schedule": "cosine"}
    )
    unet.setdefault("model_cfg", {}).update(
        {"backend": "torch", "base_channels": 32, "upsample_mode": "deconv"}
    )
    unet["model_cfg"]["conv_cfg"] = {"depth": 2}
    unet["input_features"] = {
        "mode": "geom_feature_pack",
        "features": channels,
        "require_pack": "error",
        "distance_transform": {
            "mode": "bounded_auto",
            "signed_tanh_tau": 8.0,
            "proximity_tau": 6.0,
            "replace_distance_any": True,
        },
    }
    if variant == "relative_spatial_objective":
        unet["selection"] = {
            "mode": "best_val_spatial_objective",
            "eval_every_n_epochs": 1,
            "warmup_epochs": 5,
            "spatial": {
                "point_weight": 1.0,
                "gradient_weight": 0.10,
                "gradient_normalization": "target_rms",
                "gradient_epsilon": 0.05,
                "boundary_weight": 0.0,
            },
            "case_aggregation": {
                "median_weight": 1.0,
                "p90_weight": 0.25,
                "worst_weight": 0.05,
            },
        }
    else:
        unet["selection"] = {"mode": "best_val_loss"}

    supervised = train.setdefault("loss", {}).setdefault("supervised", {})
    supervised.update(
        {
            "type": "huber",
            "huber_delta": 1.0,
            "mask": "plasma_only",
            "nan_region_policy": "mask_only",
            "normalization": "sample_mean",
        }
    )
    supervised.pop("spatial", None)
    if variant == "smooth_spatial_loss":
        supervised["spatial"] = {
            "gradient_weight": 0.10,
            "multiscale_weight": 0.05,
            "multiscale_scales": [2, 4],
        }
    elif variant == "relative_spatial_objective":
        supervised["spatial"] = {
            "gradient_weight": 0.02,
            "gradient_normalization": "target_rms",
            "gradient_epsilon": 0.05,
            "multiscale_weight": 0.05,
            "multiscale_scales": [2, 4],
        }

    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["protocol_variant"] = f"icp_stage4_spatial_learning_v1_{variant}"
    eval_cfg["quality_score"] = {
        "mode": "spatial_huber",
        "target_aggregation": "uniform_by_target",
        "delta": 1.0,
        "boundary_alpha": 0.0,
        "avgpool_lambda": 0.05,
        "gradient_lambda": 0.10,
        "multiscale_scales": [2, 4],
        "p90_weight": 0.25,
        "worst_weight": 0.10,
    }
    eval_cfg["spatial_distribution_audit"] = {
        "enabled": True,
        "vars": "all",
        "top_fraction": 0.10,
        "plot_worst_cases": 3,
        "plot_representative_cases": True,
    }
    return cfg


def generate_configs(*, source_path: Path, config_root: Path, run_root: Path) -> dict[str, Path]:
    with source_path.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream) or {}
    config_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for variant in VARIANTS:
        path = config_root / f"benchmark_icp_stage4_{variant}.yaml"
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                build_variant_config(source, variant=variant, run_root=run_root),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
        paths[variant] = path
    return paths


def generate_uno_config(
    *,
    source_path: Path,
    config_root: Path,
    run_root: Path,
    variant: str,
) -> Path:
    with source_path.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream) or {}
    cfg = build_variant_config(source, variant=variant, run_root=run_root)
    bench = cfg["benchmark"]
    bench["output_dir"] = str(run_root / f"uno_{variant}").replace("\\", "/")
    bench.setdefault("metadata", {})["comparison_model"] = "u_no_m10_w48_l4_lr3e4"
    train = bench["train"]
    generated_unet = train.pop("unet")
    uno = train.setdefault("u_no", {})
    uno["epochs"] = 70
    uno["lr"] = 3.0e-4
    uno["optimizer"] = deepcopy(generated_unet["optimizer"])
    uno["input_features"] = deepcopy(generated_unet["input_features"])
    uno["selection"] = deepcopy(generated_unet["selection"])
    uno.setdefault("model_cfg", {}).update(
        {
            "backend": "torch",
            "n_modes": 10,
            "fno_n_modes": 10,
            "uno_cfg": {"width": 48, "n_layers": 4, "dropout": 0.0},
        }
    )
    config_root.mkdir(parents=True, exist_ok=True)
    path = config_root / f"benchmark_icp_stage4_uno_{variant}.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    return path


def generate_ffno_config(
    *,
    source_path: Path,
    config_root: Path,
    run_root: Path,
    variant: str,
) -> Path:
    """Apply the shared spatial-learning recipe to the selected FFNO baseline."""
    with source_path.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream) or {}
    cfg = build_variant_config(source, variant=variant, run_root=run_root)
    bench = cfg["benchmark"]
    bench["output_dir"] = str(run_root / f"ffno_{variant}").replace("\\", "/")
    bench.setdefault("metadata", {})["comparison_model"] = "ffno_m12_w48_l4_lr4e4"

    train = bench["train"]
    generated_unet = train.pop("unet")
    ffno = train.setdefault("ffno", {})
    ffno["epochs"] = 70
    ffno["lr"] = 4.0e-4
    ffno["optimizer"] = deepcopy(generated_unet["optimizer"])
    ffno["optimizer"]["lr"] = 4.0e-4
    ffno["input_features"] = deepcopy(generated_unet["input_features"])
    ffno["selection"] = deepcopy(generated_unet["selection"])
    ffno.setdefault("model_cfg", {}).update(
        {
            "backend": "torch",
            "n_modes": 12,
            "fno_n_modes": 12,
        }
    )
    spectral_cfg = ffno["model_cfg"].setdefault("spectral_cfg", {})
    spectral_cfg.update(
        {
            "width": 48,
            "n_layers": 4,
            "dropout": 0.0,
            "dealias_ratio": 0.85,
            "taper_alpha": 1.5,
        }
    )

    config_root.mkdir(parents=True, exist_ok=True)
    path = config_root / f"benchmark_icp_stage4_ffno_{variant}.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    return path


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "status", "seconds", "config", "output_dir"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_configs(paths: dict[str, Path], *, run_root: Path, variants: list[str]) -> None:
    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    from plasma_surrogate.benchmark.runner import BenchmarkRunner

    status_path = run_root / "status.csv"
    rows: list[dict[str, Any]] = []
    if status_path.exists():
        with status_path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    rows = [row for row in rows if str(row.get("variant", "")) not in set(variants)]
    for variant in variants:
        output_dir = run_root / variant
        leaderboard = output_dir / "leaderboard.csv"
        if leaderboard.exists():
            rows.append(
                {"variant": variant, "status": "skipped_existing", "seconds": 0.0, "config": paths[variant], "output_dir": output_dir}
            )
            _write_status(status_path, rows)
            continue
        start = time.perf_counter()
        status = "passed"
        try:
            BenchmarkRunner.from_yaml(paths[variant]).run()
        except BaseException:
            status = "failed"
            rows.append(
                {"variant": variant, "status": status, "seconds": time.perf_counter() - start, "config": paths[variant], "output_dir": output_dir}
            )
            _write_status(status_path, rows)
            raise
        rows.append(
            {"variant": variant, "status": status, "seconds": time.perf_counter() - start, "config": paths[variant], "output_dir": output_dir}
        )
        _write_status(status_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--uno-source", type=Path, default=UNO_SOURCE_CONFIG)
    parser.add_argument("--uno-variant", choices=VARIANTS, default=None)
    parser.add_argument("--run-uno", action="store_true")
    parser.add_argument("--ffno-source", type=Path, default=FFNO_SOURCE_CONFIG)
    parser.add_argument("--ffno-variant", choices=VARIANTS, default=None)
    parser.add_argument("--run-ffno", action="store_true")
    args = parser.parse_args()
    paths = generate_configs(source_path=args.source, config_root=args.config_root, run_root=args.run_root)
    for variant, path in paths.items():
        print(f"{variant}: {path}")
    if args.run:
        run_configs(paths, run_root=args.run_root, variants=list(args.variants))
    if args.uno_variant is not None:
        uno_path = generate_uno_config(
            source_path=args.uno_source,
            config_root=args.config_root,
            run_root=args.run_root,
            variant=str(args.uno_variant),
        )
        print(f"uno_{args.uno_variant}: {uno_path}")
        if args.run_uno:
            run_configs(
                {f"uno_{args.uno_variant}": uno_path},
                run_root=args.run_root,
                variants=[f"uno_{args.uno_variant}"],
            )
    if args.ffno_variant is not None:
        ffno_path = generate_ffno_config(
            source_path=args.ffno_source,
            config_root=args.config_root,
            run_root=args.run_root,
            variant=str(args.ffno_variant),
        )
        print(f"ffno_{args.ffno_variant}: {ffno_path}")
        if args.run_ffno:
            run_configs(
                {f"ffno_{args.ffno_variant}": ffno_path},
                run_root=args.run_root,
                variants=[f"ffno_{args.ffno_variant}"],
            )


if __name__ == "__main__":
    main()
