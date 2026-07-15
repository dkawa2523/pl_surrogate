from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


CORE_MODELS = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "fno",
    "ffno",
    "cno",
    "cno_operator_unet",
)
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
PART_LITE_CHANNELS = [
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
    parser.add_argument(
        "--part-lite-v1",
        action="store_true",
        help=(
            "Generate process-only configs using part_lite_v1 order-invariant part summary features. "
            "The dataset root must contain per-case structure_npz part_mask_stack entries."
        ),
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
        help=(
            "Optional primary split override. Structural single-axis studies default to "
            "structure_holdout so validation/test coil groups are unseen during training."
        ),
    )
    parser.add_argument(
        "--linear-target-preprocessing",
        action="store_true",
        help="Keep the old identity/zscore target preprocessing instead of the ICP Stage4 structure defaults.",
    )
    parser.add_argument(
        "--enable-optimize",
        action="store_true",
        help="Write an enabled two_stage inference.optimize block into generated configs.",
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


def _set_target_transforms(cfg: dict[str, Any], *, structure_defaults: bool) -> None:
    scalers = cfg.setdefault("preprocessing", {}).setdefault("scalers", {})
    scalers["enforce_target_transforms"] = True
    scalers["y_fit_policy"] = "plasma_only"
    if structure_defaults:
        scalers["cond"] = "robust"
    target_transforms = scalers.setdefault("target_transforms", {})
    if structure_defaults:
        target_transforms.update(
            {
                "ne": {
                    "value_transform": "log10_floor",
                    "floor": 1.0e-30,
                    "scaler": "zscore",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "quantile", "q_low": 0.001, "q_high": 0.999},
                },
                "ni": {
                    "value_transform": "log10_floor",
                    "floor": 1.0e-30,
                    "scaler": "zscore",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "quantile", "q_low": 0.001, "q_high": 0.999},
                },
                "Te": {
                    "value_transform": "log1p",
                    "floor": 0.0,
                    "scaler": "robust",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "quantile", "q_low": 0.001, "q_high": 0.999},
                },
                "phi": {
                    "value_transform": "signed_log1p",
                    "scaler": "robust",
                    "fit_scope": "plasma_only",
                    "clip": {"mode": "quantile", "q_low": 0.001, "q_high": 0.999},
                },
            }
        )
        return
    for name in ("ne", "ni", "Te", "phi"):
        spec = dict(target_transforms.get(name, {}))
        spec.setdefault("scaler", "zscore")
        spec.setdefault("fit_scope", "plasma_only")
        spec.setdefault("clip", {"mode": "none"})
        spec["value_transform"] = "identity"
        target_transforms[name] = spec


def _set_inference_defaults(cfg: dict[str, Any], *, enable_optimize: bool, size: str) -> None:
    inf = cfg.setdefault("inference", {})
    inf["qoi"] = {
        "uniformity": {
            "target": "ne",
            "preferred_targets": ["ne", "ni", "Te"],
            "region": "plasma_mid_height",
            "mid_height_band_px": 2,
        }
    }
    inf["postprocess"] = {"positive_vars": ["ne", "ni", "Te"], "positive_floor": 1.0e-30}
    n_trials = 32 if size == "smoke" else 512
    inf["optimize"] = {
        "enabled": bool(enable_optimize),
        "backend": "two_stage",
        "n_trials": n_trials,
        "backend_cfg": {
            "n_initial": 16 if size == "smoke" else 256,
            "top_k": 4 if size == "smoke" else 16,
            "local_trials_per_seed": 4 if size == "smoke" else 16,
            "local_radius_frac": 0.25,
        },
        "objective": {
            "mode": "weighted_sum",
            "terms": [
                {"key": "uniformity", "direction": "min", "weight": 1.0},
                {
                    "key": "poisson_residual_norm",
                    "direction": "min",
                    "weight": 0.05,
                    "transform": "log1p_abs",
                    "scale": 1.0,
                },
                {"key": "boundary_gamma_uniformity", "direction": "min", "weight": 0.1},
            ],
        },
    }


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
    elif model in {"unetpp", "unetpp_attn"}:
        net_cfg = model_cfg.setdefault("model_cfg", {})
        conv_cfg = net_cfg.setdefault("conv_cfg", {})
        conv_cfg["base_channels"] = 8
        conv_cfg["depth"] = 2
        if model == "unetpp_attn":
            conv_cfg.setdefault("attention_cfg", {})["enabled"] = True
    elif model in {"fno", "ffno"}:
        net_cfg = model_cfg.setdefault("model_cfg", {})
        net_cfg["n_modes"] = 4
        net_cfg["fno_n_modes"] = 4
        spectral = net_cfg.setdefault("spectral_cfg", {})
        spectral["width"] = 16
        spectral["n_layers"] = 1
    elif model == "deeponet_pod":
        pod_cfg = model_cfg.setdefault("model_cfg", {})
        pod_cfg["hidden_dim"] = 32
        pod_cfg["latent_dim"] = 32
        pod_cfg["coeff_loss_weight"] = 0.1
        pod_cfg.setdefault("basis", {})["rank"] = 8
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
    part_lite_v1: bool,
    linear_target_preprocessing: bool,
    enable_optimize: bool,
    dual_axis: bool,
) -> dict[str, Any]:
    root = dict(cfg)
    bench = root.setdefault("benchmark", root)
    spec = SIZE_SPECS[size]

    bench["output_dir"] = str(run_root / size / model).replace("\\", "/")
    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["interp_mode"] = "marginal"
    eval_cfg["protocol_variant"] = f"icp_stage4_core4_{size}_{model}"
    eval_protocol = bench.setdefault("eval_protocol", {})
    eval_protocol["interp_mode"] = "marginal"
    primary_split_override = str(primary_split_override).strip().lower()
    if primary_metric:
        eval_cfg["primary_metric"] = str(primary_metric)
        if primary_mode:
            eval_cfg["primary_mode"] = str(primary_mode)
            eval_cfg["objective_mode"] = str(primary_mode)
        if "_extrap" in str(primary_metric):
            eval_protocol["primary_split"] = "extrap"
        elif "_interp" in str(primary_metric):
            eval_protocol["primary_split"] = "interp"
        elif "_structure_holdout" in str(primary_metric):
            eval_protocol["primary_split"] = "structure_holdout"
    if primary_split_override:
        eval_protocol["primary_split"] = primary_split_override
    structure_inputs_enabled = bool(structure_spatial_v1 or part_sdf_lite_v1 or part_lite_v1)
    if part_lite_v1:
        feature_profile = "part_lite_v1"
        feature_channels = list(PART_LITE_CHANNELS)
    elif part_sdf_lite_v1:
        feature_profile = "icp_part_sdf_lite_v1"
        feature_channels = list(ICP_PART_SDF_LITE_CHANNELS)
    else:
        feature_profile = "icp_struct_spatial_v1"
        feature_channels = list(ICP_STRUCT_SPATIAL_CHANNELS)
    provider_mode = "parametric_parts" if (part_sdf_lite_v1 or part_lite_v1) else "fixed"
    if not structure_inputs_enabled:
        feature_profile = "geom_v1_mainline"
        feature_channels = list(ICP_STRUCT_SPATIAL_CHANNELS)
        provider_mode = "fixed"
    metric_lower = str(primary_metric).strip().lower()
    metric_selects_split = any(
        token in metric_lower for token in ("_interp", "_extrap", "_structure_holdout")
    )
    if (
        structure_inputs_enabled
        and not dual_axis
        and not primary_split_override
        and not metric_selects_split
    ):
        eval_protocol["primary_split"] = "structure_holdout"
    primary_split = str(eval_protocol.get("primary_split", "interp")).strip().lower()
    eval_protocol["min_primary_test_cases"] = max(
        int(eval_protocol.get("min_primary_test_cases", 3)),
        3,
    )
    eval_protocol["min_primary_test_groups"] = max(
        int(eval_protocol.get("min_primary_test_groups", 3)),
        3,
    )
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
    if structure_inputs_enabled:
        bench["split"]["structure_holdout"] = {
            "enabled": True,
            "required": True,
            "group_key": "base_case_id",
        }
    _set_target_transforms(
        bench,
        structure_defaults=bool(structure_inputs_enabled and not linear_target_preprocessing),
    )
    preprocessing = bench.setdefault("preprocessing", {})
    preprocessing.setdefault("scalers", {})["fit_split"] = primary_split
    if structure_inputs_enabled:
        # The converted ICP datasets provide explicit physical r/z coordinate vectors.
        # Request that source directly and reject any fallback to normalized coordinates.
        preprocessing["coord_grid_source"] = "rz_linear"
        preprocessing.setdefault("coord_grid_contract", {})["require_requested_source"] = "error"
    coord_features = preprocessing.setdefault("coord_features", {})
    grid_pack_model = bool(structure_inputs_enabled and model not in {"global_mlp", "deeponet_pod"})
    descriptor_model = bool(structure_inputs_enabled and model == "deeponet_pod")
    if grid_pack_model:
        coord_features["channels_from_profile"] = feature_profile
        coord_features.pop("channels", None)
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
    elif descriptor_model:
        coord_features["channels_from_profile"] = feature_profile
        coord_features.pop("channels", None)
        coord_features.pop("case_output", None)
        coord_features.setdefault("static_output", "features/static_spatial_feature_pack.npz")
        coord_features.setdefault("case_structure_output", "features/case_structure_feature_pack.npz")
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
    if grid_pack_model:
        model_train_cfg = train_cfg.setdefault(model, {})
        input_features = model_train_cfg.setdefault("input_features", {})
        input_features["mode"] = "geom_feature_pack"
        input_features["require_pack"] = "error"
        input_features["features"] = list(feature_channels)
        if model == "unet":
            model_train_cfg.setdefault("model_cfg", {}).setdefault("output_heads", {})["mode"] = "shared"
        if model == "fno":
            net_cfg = model_train_cfg.setdefault("model_cfg", {})
            net_cfg["n_modes"] = 12
            net_cfg["fno_n_modes"] = 12
            spectral = net_cfg.setdefault("spectral_cfg", {})
            spectral["width"] = 48
            model_train_cfg["batch_size_cases"] = min(int(model_train_cfg.get("batch_size_cases", 4)), 2)
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
        supervised["type"] = "huber"
        supervised["huber_delta"] = 1.0
        supervised.setdefault("mask", "plasma_only")
        supervised["nan_region_policy"] = "mask_only"
        supervised["target_weights"] = {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
        for old_key in (
            "boundary_weight",
            "target_region_by_var",
            "spatial_consistency",
            "region_balance",
            "density_relative_weighting",
        ):
            supervised.pop(old_key, None)
        multitask = loss_cfg.setdefault("multitask", {})
        multitask["weighting"] = "fixed"
        multitask["fixed_weights_by_var"] = {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
        _set_inference_defaults(bench, enable_optimize=enable_optimize, size=size)
    if descriptor_model:
        loss_cfg = train_cfg.setdefault("loss", {})
        supervised = loss_cfg.setdefault("supervised", {})
        supervised["type"] = "huber"
        supervised["huber_delta"] = 1.0
        supervised.setdefault("mask", "plasma_only")
        supervised["nan_region_policy"] = "mask_only"
        supervised["target_weights"] = {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
        model_train_cfg = train_cfg.setdefault(model, {})
        model_train_cfg["batch_size_cases"] = min(int(model_train_cfg.get("batch_size_cases", 4)), 4)
        _set_inference_defaults(bench, enable_optimize=enable_optimize, size=size)

    if model == "global_mlp":
        runtime = {
            "input_mode": "table_only",
            "strict_input_mode": "error",
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        }
    elif model == "deeponet_pod":
        runtime = {
            "input_mode": "table_plus_structure",
            "strict_input_mode": "error",
            "structure": {
                "feature_profile": feature_profile if structure_inputs_enabled else "geom_v1_mainline",
                "descriptor_profile": "struct_desc_v2",
                "latent_profile": "none",
                "adapter_mode": "descriptor_branch",
                "provider_mode": "parametric_parts",
            },
        }
    else:
        runtime = {
            "input_mode": "table_plus_structure",
            "strict_input_mode": "error",
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
    part_lite_v1: bool,
    linear_target_preprocessing: bool,
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
    structure_inputs_enabled = bool(structure_spatial_v1 or part_sdf_lite_v1 or part_lite_v1)
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
        holdout_cfg = dict(bench.get("split", {}).get("structure_holdout", {}) or {})
        if not bool(holdout_cfg.get("required", False)) or holdout_cfg.get("group_key") != "base_case_id":
            raise ValueError(f"{path}: structural configs must require base_case_id structure holdout")
        if bench.get("preprocessing", {}).get("coord_grid_source") != "rz_linear":
            raise ValueError(f"{path}: structural configs must request coord_grid_source=rz_linear")
        coord_contract = bench.get("preprocessing", {}).get("coord_grid_contract", {})
        if coord_contract.get("require_requested_source") != "error":
            raise ValueError(f"{path}: structural configs must enforce coord-grid source")
    if structure_inputs_enabled and not linear_target_preprocessing:
        expected_transforms = {"ne": "log10_floor", "ni": "log10_floor", "Te": "log1p", "phi": "signed_log1p"}
    else:
        expected_transforms = {"ne": "identity", "ni": "identity", "Te": "identity", "phi": "identity"}
    for name, expected_transform in expected_transforms.items():
        if transforms.get(name, {}).get("value_transform") != expected_transform:
            raise ValueError(f"{path}: {name} target transform must be {expected_transform}")
        if structure_inputs_enabled and not linear_target_preprocessing:
            if transforms.get(name, {}).get("clip", {}).get("mode") != "quantile":
                raise ValueError(f"{path}: {name} must use fitted quantile bounds for stable inverse evaluation")
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
        "part_lite_v1"
        if (part_lite_v1 and model != "global_mlp")
        else "icp_part_sdf_lite_v1"
        if (part_sdf_lite_v1 and model != "global_mlp")
        else "icp_struct_spatial_v1" if (structure_spatial_v1 and model != "global_mlp") else (
        "none" if model == "global_mlp" else "geom_v1_mainline"
        )
    )
    if str(structure.get("feature_profile", "")) != expected_profile:
        raise ValueError(f"{path}: feature_profile mismatch")
    if structure_inputs_enabled and model not in {"global_mlp", "deeponet_pod"}:
        input_features = dict(dict(bench.get("train", {})).get(model, {}).get("input_features", {}))
        if input_features.get("require_pack") != "error":
            raise ValueError(f"{path}: structural grid models must require the case feature pack")


def main() -> int:
    args = _parse_args()
    if sum(bool(v) for v in (args.structure_spatial_v1, args.part_sdf_lite_v1, args.part_lite_v1)) > 1:
        raise ValueError("Choose at most one of --structure-spatial-v1, --part-sdf-lite-v1, or --part-lite-v1")
    template_root = Path(args.template_root)
    out_root = Path(args.out_root)
    run_root = Path(args.run_root)
    dataset_root = str(args.dataset_root).strip()
    if not dataset_root:
        if bool(args.part_sdf_lite_v1 or args.part_lite_v1):
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
                part_lite_v1=bool(args.part_lite_v1),
                linear_target_preprocessing=bool(args.linear_target_preprocessing),
                enable_optimize=bool(args.enable_optimize),
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
                part_lite_v1=bool(args.part_lite_v1),
                linear_target_preprocessing=bool(args.linear_target_preprocessing),
                dual_axis=bool(args.dual_axis),
            )
            written.append(out_path)

    for path in written:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
