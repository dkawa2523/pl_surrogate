from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


CORE_MODELS = ("global_mlp", "unet", "ffno", "cno", "cno_operator_unet")
DEFAULT_MODELS = ("global_mlp", "unet", "ffno", "cno_operator_unet")
LEGACY_COND_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0"]
PROCESS_COND_COLUMNS = ["pp", "pp0"]
ICP_STRUCT_SPATIAL_CHANNELS = [
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "mask_coil",
    "distance_coil",
    "coil_proximity",
]
ICP_PART_SDF_LITE_CHANNELS = [
    *ICP_STRUCT_SPATIAL_CHANNELS,
    "sdf_coil_01",
    "sdf_coil_02",
    "sdf_coil_03",
    "sdf_coil_04",
    "sdf_coil_05",
    "sdf_coil_06",
]
DATASET_ROOT = "data/outputs_icp_stage4_enriched_360_csv_npz_core4_linear"
STRUCT_SPATIAL_DATASET_ROOT = "data/outputs_icp_stage4_enriched_360_csv_npz_core4_struct_spatial_v1"
PART_SDF_LITE_DATASET_ROOT = "data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1"
TEMPLATE_ROOT = Path("configs/benchmarkrun_ext0520/templates")
DEFAULT_OUT_ROOT = Path("configs/experimental/icp_stage4/generated")
DEFAULT_RUN_ROOT = Path("runs/icp_stage4_core4")


SIZE_SPECS = {
    "smoke": {
        "index_csv": "index_smoke.csv",
        "epochs": 1,
    },
    "full": {
        "index_csv": "index.csv",
        "epochs": None,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ICP Stage4 Core4 benchmark configs.")
    parser.add_argument("--sizes", nargs="+", choices=tuple(SIZE_SPECS), default=list(SIZE_SPECS))
    parser.add_argument("--models", nargs="+", choices=CORE_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--template-root", default=str(TEMPLATE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--dataset-root", default="", help="Override dataset root.")
    parser.add_argument(
        "--structure-spatial-v1",
        action="store_true",
        help="Generate process-only configs using per-case ICP structure spatial features.",
    )
    parser.add_argument(
        "--part-sdf-lite-v1",
        action="store_true",
        help="Generate process-only configs using icp_part_sdf_lite_v1 per-coil SDF features.",
    )
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=0,
        help="Override full-run epochs when >0; by default template epochs are preserved.",
    )
    parser.add_argument(
        "--primary-metric",
        default="",
        help="Optional benchmark.eval.primary_metric override; default preserves the template value.",
    )
    parser.add_argument(
        "--primary-mode",
        choices=("", "min", "max"),
        default="",
        help="Optional benchmark.eval.primary_mode override used with --primary-metric.",
    )
    parser.add_argument(
        "--dual-axis",
        action="store_true",
        help="Keep dual-axis interp+extrap training. Default runs only eval_protocol.primary_split.",
    )
    parser.add_argument(
        "--primary-split",
        choices=("", "interp", "extrap", "structure_holdout"),
        default="",
        help="Optional primary split override. For --structure-spatial-v1 the default is extrap.",
    )
    return parser.parse_args()


def _load_template(template_root: Path, model: str) -> dict[str, Any]:
    path = template_root / f"benchmark_ext0520_{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing template for {model}: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_dataset_targets_linear(cfg: dict[str, Any]) -> None:
    cfg["targets"] = [
        {
            "id": "ne",
            "source_key": "ne",
            "value_transform": "identity",
            "units": "m^-3",
            "dtype": "float32",
        },
        {
            "id": "ni",
            "source_key": "ni",
            "value_transform": "identity",
            "units": "m^-3",
            "dtype": "float32",
        },
        {
            "id": "Te",
            "source_key": "Te",
            "value_transform": "identity",
            "units": "eV",
            "dtype": "float32",
        },
        {
            "id": "phi",
            "source_key": "phi",
            "value_transform": "identity",
            "units": "V",
            "dtype": "float32",
        },
    ]


def _set_target_transforms(cfg: dict[str, Any]) -> None:
    scalers = cfg.setdefault("preprocessing", {}).setdefault("scalers", {})
    scalers["enforce_target_transforms"] = True
    scalers["y_fit_policy"] = "plasma_only"
    target_transforms = scalers.setdefault("target_transforms", {})
    for name in ("ne", "ni", "Te", "phi"):
        spec = dict(target_transforms.get(name, {}))
        spec.setdefault("scaler", "zscore")
        spec.setdefault("fit_scope", "plasma_only")
        spec.setdefault("clip", {"mode": "none"})
        spec["value_transform"] = "identity"
        target_transforms[name] = spec


def _set_epochs(train_cfg: dict[str, Any], model: str, epochs: int | None) -> None:
    if epochs is None:
        return
    model_cfg = train_cfg.setdefault(model, {})
    model_cfg["epochs"] = int(max(1, epochs))
    if "optimizer" in model_cfg and isinstance(model_cfg["optimizer"], dict):
        model_cfg["optimizer"]["warmup_epochs"] = 0
    if "selection" in model_cfg and isinstance(model_cfg["selection"], dict):
        model_cfg["selection"]["eval_every_n_epochs"] = 1
        model_cfg["selection"]["warmup_epochs"] = 0


def _make_smoke_lightweight(train_cfg: dict[str, Any], model: str) -> None:
    train_cfg.setdefault("unet_like", {})["batch_size_cases"] = 1
    loss_cfg = train_cfg.setdefault("loss", {}).setdefault("supervised", {})
    spatial = loss_cfg.get("spatial_consistency")
    if isinstance(spatial, dict):
        spatial["enabled"] = False

    model_cfg = train_cfg.setdefault(model, {})
    model_cfg["batch_size_cases"] = 1
    if model == "global_mlp":
        model_cfg.setdefault("model_cfg", {})["hidden"] = [16]
        refresh = model_cfg.setdefault("output_head_refresh", {})
        refresh["enabled"] = True
        refresh["every_n_epochs"] = 1
    elif model == "unet":
        net_cfg = model_cfg.setdefault("model_cfg", {})
        net_cfg["base_channels"] = 8
        net_cfg.setdefault("conv_cfg", {})["depth"] = 1
    elif model == "ffno":
        net_cfg = model_cfg.setdefault("model_cfg", {})
        net_cfg["n_modes"] = 4
        net_cfg["fno_n_modes"] = 4
        spectral = net_cfg.setdefault("spectral_cfg", {})
        spectral["width"] = 16
        spectral["n_layers"] = 1
    elif model == "cno":
        cno_cfg = model_cfg.setdefault("model_cfg", {}).setdefault("cno_cfg", {})
        cno_cfg["width"] = 16
        cno_cfg["n_layers"] = 1
    elif model == "cno_operator_unet":
        cno_cfg = model_cfg.setdefault("model_cfg", {}).setdefault("cno_operator_unet_cfg", {})
        cno_cfg["width"] = 16
        cno_cfg["depth"] = 2
        cno_cfg["blocks_per_level"] = 1


def _mutate_config(
    cfg: dict[str, Any],
    *,
    size: str,
    model: str,
    smoke_epochs: int,
    full_epochs: int,
    run_root: Path,
    primary_metric: str,
    primary_mode: str,
    primary_split_override: str,
    dataset_root: str,
    structure_spatial_v1: bool,
    part_sdf_lite_v1: bool,
    dual_axis: bool,
) -> dict[str, Any]:
    root = dict(cfg)
    bench = root.setdefault("benchmark", root)
    spec = SIZE_SPECS[size]

    bench["output_dir"] = str(run_root / size / model).replace("\\", "/")
    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["protocol_variant"] = f"icp_stage4_core4_{size}_{model}"
    eval_protocol = bench.setdefault("eval_protocol", {})
    primary_split_override = str(primary_split_override).strip().lower()
    if primary_metric:
        eval_cfg["primary_metric"] = str(primary_metric)
        if primary_mode:
            eval_cfg["primary_mode"] = str(primary_mode)
        if "_extrap" in str(primary_metric):
            eval_protocol["primary_split"] = "extrap"
        elif "_interp" in str(primary_metric):
            eval_protocol["primary_split"] = "interp"
        elif "_structure_holdout" in str(primary_metric):
            eval_protocol["primary_split"] = "structure_holdout"
    if primary_split_override:
        eval_protocol["primary_split"] = primary_split_override
    structure_inputs_enabled = bool(structure_spatial_v1 or part_sdf_lite_v1)
    feature_profile = "icp_part_sdf_lite_v1" if part_sdf_lite_v1 else "icp_struct_spatial_v1"
    feature_channels = list(ICP_PART_SDF_LITE_CHANNELS if part_sdf_lite_v1 else ICP_STRUCT_SPATIAL_CHANNELS)
    provider_mode = "parametric_parts" if part_sdf_lite_v1 else "fixed"
    if not structure_inputs_enabled:
        feature_profile = "geom_v1_mainline"
        feature_channels = list(ICP_STRUCT_SPATIAL_CHANNELS)
        provider_mode = "fixed"
    if structure_inputs_enabled and not primary_metric and not primary_split_override:
        eval_protocol["primary_split"] = "extrap"
    primary_split = str(eval_protocol.get("primary_split", "interp")).strip().lower()
    metric_effective = str(eval_cfg.get("primary_metric", "")).strip()
    if not dual_axis:
        eval_protocol["mode"] = "primary_axis"
        if metric_effective.endswith("_dual"):
            eval_cfg["primary_metric"] = metric_effective[: -len("_dual")] + f"_{primary_split}"

    dataset = bench.setdefault("dataset", {})
    dataset["type"] = "csv_npz"
    dataset["root"] = dataset_root
    dataset["index_csv"] = spec["index_csv"]
    _set_dataset_targets_linear(dataset)
    dataset["cond_columns"] = list(PROCESS_COND_COLUMNS if structure_inputs_enabled else LEGACY_COND_COLUMNS)
    dataset["axis_column"] = "axis"
    dataset["fields_npz_column"] = "fields_npz"
    if structure_inputs_enabled:
        dataset["structure_npz_column"] = "structure_npz"
    else:
        dataset.pop("structure_npz_column", None)
    dataset["case_id_column"] = "case_id"
    dataset["base_case_id_column"] = "base_case_id"
    dataset["split_group_column"] = "split_group"
    dataset["geometry_root"] = "geometry"

    bench["split"] = {"seed": 7, "ratios": [0.8, 0.1, 0.1]}
    _set_target_transforms(bench)
    bench.setdefault("preprocessing", {}).setdefault("scalers", {})["fit_split"] = primary_split
    coord_features = bench.setdefault("preprocessing", {}).setdefault("coord_features", {})
    if structure_inputs_enabled and model != "global_mlp":
        coord_features["channels_from_profile"] = feature_profile
        coord_features.pop("case_output", None)
        coord_features.setdefault("static_output", "features/static_spatial_feature_pack.npz")
        coord_features.setdefault("case_structure_output", "features/case_structure_feature_pack.npz")
        coord_features.setdefault("distance_transform_stats", {})["enabled"] = True
        coord_features.setdefault("scaling", {}).update(
            {
                "enabled": True,
                "mode": "zscore",
                "fit_scope": "train_split",
                "mask_scope": "plasma_plus_band",
            }
        )
    elif model == "global_mlp":
        coord_features.pop("channels_from_profile", None)

    eval_cfg.setdefault(
        "spatial_distribution_audit",
        {"enabled": True, "vars": "density", "top_fraction": 0.10, "plot_worst_cases": 2},
    )

    train_cfg = bench.setdefault("train", {})
    epochs = smoke_epochs if size == "smoke" else (full_epochs if full_epochs > 0 else None)
    _set_epochs(train_cfg, model, epochs)
    if size == "smoke":
        _make_smoke_lightweight(train_cfg, model)
    if structure_inputs_enabled and model != "global_mlp":
        model_train_cfg = train_cfg.setdefault(model, {})
        input_features = model_train_cfg.setdefault("input_features", {})
        input_features["mode"] = "geom_feature_pack"
        input_features["require_pack"] = "off" if part_sdf_lite_v1 else "error"
        input_features["features"] = list(feature_channels)
        if model == "unet":
            model_train_cfg.setdefault("model_cfg", {}).setdefault("output_heads", {})["mode"] = "split_density_field"
        if model == "ffno":
            net_cfg = model_train_cfg.setdefault("model_cfg", {})
            net_cfg["n_modes"] = 16
            net_cfg["fno_n_modes"] = 16
            spectral = net_cfg.setdefault("spectral_cfg", {})
            spectral["width"] = 64
            model_train_cfg["batch_size_cases"] = min(int(model_train_cfg.get("batch_size_cases", 4)), 2)
        if model == "cno_operator_unet":
            model_train_cfg["batch_size_cases"] = min(int(model_train_cfg.get("batch_size_cases", 6)), 4)

        loss_cfg = train_cfg.setdefault("loss", {})
        supervised = loss_cfg.setdefault("supervised", {})
        supervised.setdefault("mask", "plasma_only")
        supervised.setdefault("nan_region_policy", "sdf_continuous")
        supervised["spatial_consistency"] = {
            "enabled": True,
            "vars": ["ne", "ni"],
            "mode": "grad_huber",
            "lambda": 0.02,
            "delta": 1.0,
            "apply_region": "plasma_only",
            "normalize_by_var_scale": True,
            "multiscale": {"enabled": False, "scales": [1, 2, 4], "scale_weights": [1.0, 0.7, 0.5]},
        }
        supervised["region_balance"] = {
            "enabled": True,
            "vars": ["ne", "ni"],
            "mode": "additive",
            "additive_lambda": 0.15,
            "boundary_in_px": 2.0,
            "mid_plasma_px": 10.0,
            "deep_plasma_px": 10.0,
            "weight_boundary_in": 0.25,
            "weight_plasma_mid": 0.25,
            "weight_deep_plasma": 0.50,
        }
        supervised["density_relative_weighting"] = {
            "enabled": True,
            "vars": ["ne", "ni"],
            "lambda": 0.05,
            "eps_by_var": {"ne": 1.0e16, "ni": 1.0e16},
            "eps": 1.0e16,
            "eps_min": 1.0,
        }
        multitask = loss_cfg.setdefault("multitask", {})
        multitask["weighting"] = "fixed"
        multitask["fixed_weights_by_var"] = {"ne": 1.5, "ni": 1.5, "Te": 1.0, "phi": 1.0}

    if model == "global_mlp":
        runtime = {
            "input_mode": "table_only",
            "strict_input_mode": "error",
            "allow_mode_fallback": False,
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        }
    else:
        runtime = {
            "input_mode": "table_plus_structure",
            "strict_input_mode": "error",
            "allow_mode_fallback": False,
            "structure": {
                "feature_profile": feature_profile if structure_inputs_enabled else "geom_v1_mainline",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "auto",
                "provider_mode": provider_mode,
            },
        }
    bench["runtime"] = runtime
    root["runtime"] = runtime
    return root


def _validate_generated(
    path: Path,
    *,
    size: str,
    model: str,
    dataset_root: str,
    structure_spatial_v1: bool,
    part_sdf_lite_v1: bool,
    dual_axis: bool,
) -> None:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.get("benchmark", {})
    dataset = bench.get("dataset", {})
    if dataset.get("root") != dataset_root:
        raise ValueError(f"{path}: dataset.root mismatch")
    if dataset.get("index_csv") != SIZE_SPECS[size]["index_csv"]:
        raise ValueError(f"{path}: dataset.index_csv mismatch")
    structure_inputs_enabled = bool(structure_spatial_v1 or part_sdf_lite_v1)
    expected_cond_columns = PROCESS_COND_COLUMNS if structure_inputs_enabled else LEGACY_COND_COLUMNS
    if list(dataset.get("cond_columns", [])) != expected_cond_columns:
        raise ValueError(f"{path}: cond_columns mismatch")
    if structure_inputs_enabled and dataset.get("structure_npz_column") != "structure_npz":
        raise ValueError(f"{path}: structure_npz_column mismatch")
    target_by_id = {str(item.get("id")): dict(item) for item in list(dataset.get("targets", []))}
    for name in ("ne", "ni"):
        if target_by_id.get(name, {}).get("source_key") != name:
            raise ValueError(f"{path}: dataset target {name} must use linear source_key={name}")
        if target_by_id.get(name, {}).get("value_transform") != "identity":
            raise ValueError(f"{path}: dataset target {name} must use identity value_transform")
    if list(bench.get("split", {}).get("ratios", [])) != [0.8, 0.1, 0.1]:
        raise ValueError(f"{path}: split ratios mismatch")
    transforms = bench.get("preprocessing", {}).get("scalers", {}).get("target_transforms", {})
    if structure_inputs_enabled:
        if bench.get("preprocessing", {}).get("scalers", {}).get("fit_split") != bench.get("eval_protocol", {}).get("primary_split"):
            raise ValueError(f"{path}: preprocessing.scalers.fit_split must match primary_split")
    for name in ("ne", "ni"):
        if transforms.get(name, {}).get("value_transform") != "identity":
            raise ValueError(f"{path}: {name} target transform must be identity")
    output_dir = str(bench.get("output_dir", ""))
    expected_suffix = f"{size}/{model}".replace("\\", "/")
    if not output_dir.replace("\\", "/").endswith(expected_suffix):
        raise ValueError(f"{path}: output_dir mismatch")
    primary_metric = str(bench.get("eval", {}).get("primary_metric", ""))
    if "_extrap" in primary_metric and bench.get("eval_protocol", {}).get("primary_split") != "extrap":
        raise ValueError(f"{path}: eval_protocol.primary_split must be extrap")
    expected_mode = "dual_axis" if dual_axis else "primary_axis"
    if str(bench.get("eval_protocol", {}).get("mode", "")) != expected_mode:
        raise ValueError(f"{path}: eval_protocol.mode mismatch")
    runtime = dict(bench.get("runtime", {}))
    structure = dict(runtime.get("structure", {}))
    expected_profile = (
        "icp_part_sdf_lite_v1"
        if (part_sdf_lite_v1 and model != "global_mlp")
        else "icp_struct_spatial_v1" if (structure_spatial_v1 and model != "global_mlp") else (
        "none" if model == "global_mlp" else "geom_v1_mainline"
        )
    )
    if str(structure.get("feature_profile", "")) != expected_profile:
        raise ValueError(f"{path}: feature_profile mismatch")


def main() -> int:
    args = _parse_args()
    template_root = Path(args.template_root)
    out_root = Path(args.out_root)
    run_root = Path(args.run_root)
    dataset_root = str(args.dataset_root).strip()
    if not dataset_root:
        if bool(args.part_sdf_lite_v1):
            dataset_root = PART_SDF_LITE_DATASET_ROOT
        else:
            dataset_root = STRUCT_SPATIAL_DATASET_ROOT if bool(args.structure_spatial_v1) else DATASET_ROOT
    written: list[Path] = []
    for size in args.sizes:
        for model in args.models:
            cfg = _mutate_config(
                _load_template(template_root, model),
                size=size,
                model=model,
                smoke_epochs=int(args.smoke_epochs),
                full_epochs=int(args.full_epochs),
                run_root=run_root,
                primary_metric=str(args.primary_metric).strip(),
                primary_mode=str(args.primary_mode).strip(),
                primary_split_override=str(args.primary_split).strip(),
                dataset_root=dataset_root,
                structure_spatial_v1=bool(args.structure_spatial_v1),
                part_sdf_lite_v1=bool(args.part_sdf_lite_v1),
                dual_axis=bool(args.dual_axis),
            )
            out_dir = out_root / size
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"benchmark_icp_stage4_core4_{size}_{model}.yaml"
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            _validate_generated(
                out_path,
                size=size,
                model=model,
                dataset_root=dataset_root,
                structure_spatial_v1=bool(args.structure_spatial_v1),
                part_sdf_lite_v1=bool(args.part_sdf_lite_v1),
                dual_axis=bool(args.dual_axis),
            )
            written.append(out_path)

    for path in written:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
