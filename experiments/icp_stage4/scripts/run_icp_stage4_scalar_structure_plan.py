from __future__ import annotations

import argparse
import csv
import runpy
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_icp_stage4_struct_suite_plan import (  # noqa: E402
    _leaderboard_for_config,
    _limit_threads,
    _matching_processes,
    _run_config,
    _set_below_normal_priority,
    _wait_until_idle,
)
from run_icp_stage4_uno_tuning_plan import UNO_RECIPES, _make_uno_cfg  # noqa: E402


MODEL_ORDER = ("u_no", "unet", "ffno", "deeponet_pod", "unetpp", "unetpp_attn", "fno", "cno")
GRID_MODELS = ("u_no", "unet", "unetpp", "unetpp_attn", "fno", "ffno", "cno")
SCALAR_STRUCTURE_COND_COLUMNS = ("llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0")
SCALAR_SPATIAL_PROFILE = "part_lite_static_v1"
SCALAR_SPATIAL_FEATURES = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "normal_x",
    "normal_y",
    "curvature_proxy",
    "boundary_band",
)

SOURCE_CONFIG_ROOT_DEFAULT = Path("configs/experimental/icp_stage4/generated_major_fixes_part_lite_v2_e80")
SCALAR_CONFIG_ROOT_DEFAULT = Path(
    "configs/experimental/icp_stage4/generated_scalar_dimension_structure_holdout_v2"
)
RUN_ROOT_DEFAULT = Path("runs/icp_stage4_scalar_dimension_structure_holdout_v2")
DIRECT_UNO_STATUS_DEFAULT = Path(
    "runs/icp_stage4_major_fixes_part_lite_v2_e80_uno3/uno_tuning_status.csv"
)
DATASET_ROOT_DEFAULT = Path("data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2")
U_NO_RECIPE_DEFAULT = "u_no_m10_w48_l4_lr3e4"
U_NO_EPOCHS_DEFAULT = 70
UNET_BEST_RECIPE = "unet_compact_ch32_d2_lr3e4"
FFNO_BEST_RECIPE = "ffno_m12_w48_l4_lr4e4"
SCALAR_U_NO_RECIPES = tuple(dict(recipe) for recipe in UNO_RECIPES if recipe.get("group") == "allvars")


def _adamw_cosine_optimizer(*, lr: float) -> dict[str, Any]:
    return {
        "type": "adamw",
        "lr": float(lr),
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "schedule": "cosine",
        "warmup_epochs": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and run ICP Stage4 scalar-structure configs. "
            "The configs use structure dimensions/coordinate scalars as conditions and avoid case-varying structure maps."
        )
    )
    parser.add_argument("--source-config-root", default=str(SOURCE_CONFIG_ROOT_DEFAULT))
    parser.add_argument("--scalar-config-root", default=str(SCALAR_CONFIG_ROOT_DEFAULT))
    parser.add_argument("--run-root", default=str(RUN_ROOT_DEFAULT))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument(
        "--u-no-recipe",
        choices=tuple(str(recipe["recipe"]) for recipe in SCALAR_U_NO_RECIPES),
        default=U_NO_RECIPE_DEFAULT,
    )
    parser.add_argument("--u-no-epochs", type=int, default=U_NO_EPOCHS_DEFAULT)
    parser.add_argument("--status-csv", default=None)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-wait-uno", action="store_true")
    parser.add_argument("--uno-status-csv", default=str(DIRECT_UNO_STATUS_DEFAULT))
    parser.add_argument("--uno-process-pattern", default="run_icp_stage4_uno_tuning_plan.py")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--max-gpu-mem-mib", type=float, default=2200.0)
    parser.add_argument("--wait-command-absent", action="append", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _norm_model_list(raw_models: list[str]) -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    allowed = set(MODEL_ORDER)
    for raw in raw_models:
        model = str(raw).strip()
        if not model or model in seen:
            continue
        if model not in allowed:
            raise ValueError(f"Unsupported scalar-structure model={model!r}; allowed={sorted(allowed)}")
        seen.add(model)
        models.append(model)
    return models


def _set_runtime_scalar_structure(cfg: dict[str, Any], *, model: str) -> None:
    if model == "deeponet_pod":
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
    else:
        runtime = {
            "input_mode": "table_plus_structure",
            "strict_input_mode": "error",
            "structure": {
                "feature_profile": SCALAR_SPATIAL_PROFILE,
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "auto",
                "provider_mode": "fixed",
            },
        }
    cfg.setdefault("benchmark", {})["runtime"] = runtime
    cfg["runtime"] = runtime


def _set_coord_features_scalar_structure(bench: dict[str, Any]) -> None:
    preprocessing = bench.setdefault("preprocessing", {})
    preprocessing.setdefault("coord_grid_contract", {})["require_requested_source"] = "error"
    preprocessing["coord_grid_source"] = "rz_linear"
    coord_features = preprocessing.setdefault("coord_features", {})
    coord_features["enabled"] = True
    coord_features["output"] = "features/coord_feature_pack.npz"
    coord_features["channels_from_profile"] = SCALAR_SPATIAL_PROFILE
    coord_features.pop("channels", None)
    coord_features.pop("static_output", None)
    coord_features.pop("case_output", None)
    coord_features.pop("case_structure_output", None)
    coord_features.pop("require_case_variation", None)
    coord_features.pop("coil_proximity_tau_fixed", None)
    coord_features.setdefault("distance_transform_stats", {})["enabled"] = True
    coord_features.setdefault("scaling", {}).update(
        {
            "enabled": True,
            "mode": "zscore",
            "fit_scope": "train_split",
            "mask_scope": "plasma_plus_band",
        }
    )


def _set_grid_input_features_scalar_structure(bench: dict[str, Any], *, model: str) -> None:
    if model not in GRID_MODELS:
        return
    train_cfg = bench.setdefault("train", {})
    model_cfg = train_cfg.setdefault(model, {})
    input_features = model_cfg.setdefault("input_features", {})
    input_features["mode"] = "geom_feature_pack"
    input_features["require_pack"] = "error"
    input_features["features"] = list(SCALAR_SPATIAL_FEATURES)


def _apply_latest_best_model_recipe(bench: dict[str, Any], *, model: str) -> None:
    """Pin the selected structural-model recipe, including the effective shared batch key."""

    if model not in {"unet", "ffno"}:
        return
    train_cfg = bench.setdefault("train", {})
    shared_cfg = train_cfg.setdefault("unet_like", {})
    shared_cfg["shuffle_cases"] = True
    metadata = bench.setdefault("metadata", {})

    if model == "unet":
        shared_cfg["batch_size_cases"] = 4
        family_cfg = train_cfg.setdefault("unet", {})
        family_cfg["epochs"] = 70
        family_cfg["lr"] = 3.0e-4
        family_cfg["optimizer"] = _adamw_cosine_optimizer(lr=3.0e-4)
        model_cfg = family_cfg.setdefault("model_cfg", {})
        model_cfg["backend"] = "torch"
        model_cfg["base_channels"] = 32
        model_cfg["upsample_mode"] = "deconv"
        model_cfg.setdefault("conv_cfg", {})["depth"] = 2
        model_cfg.setdefault("output_heads", {})["mode"] = "shared"
        metadata["model_recipe"] = UNET_BEST_RECIPE
        return

    # grid_training reads train.unet_like.batch_size_cases.  Keep the family
    # value in sync as an auditable declaration, but do not rely on it alone.
    shared_cfg["batch_size_cases"] = 2
    family_cfg = train_cfg.setdefault("ffno", {})
    family_cfg["batch_size_cases"] = 2
    family_cfg["epochs"] = 70
    family_cfg["lr"] = 4.0e-4
    family_cfg["optimizer"] = _adamw_cosine_optimizer(lr=4.0e-4)
    model_cfg = family_cfg.setdefault("model_cfg", {})
    model_cfg["n_modes"] = 12
    model_cfg["fno_n_modes"] = 12
    spectral_cfg = model_cfg.setdefault("spectral_cfg", {})
    spectral_cfg.update(
        {
            "width": 48,
            "n_layers": 4,
            "dropout": 0.0,
            "dealias_ratio": 0.85,
            "taper_alpha": 1.5,
            "skip_filter": "none",
            "factorized_cfg": {
                "enabled": True,
                "mode": "separable_1d",
                "share_weights": False,
            },
            "local_skip_cfg": {"enabled": True, "init_scale": 0.0},
        }
    )
    metadata["model_recipe"] = FFNO_BEST_RECIPE


def _set_structure_holdout_protocol(bench: dict[str, Any]) -> None:
    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["primary_metric"] = "test_r2_group_default_plasma"
    eval_cfg["primary_mode"] = "max"
    eval_cfg["objective_mode"] = "max"

    eval_protocol = bench.setdefault("eval_protocol", {})
    eval_protocol["mode"] = "primary_axis"
    eval_protocol["primary_split"] = "structure_holdout"
    eval_protocol["min_primary_test_cases"] = max(int(eval_protocol.get("min_primary_test_cases", 3)), 3)
    eval_protocol["min_primary_test_groups"] = max(int(eval_protocol.get("min_primary_test_groups", 3)), 3)

    split = bench.setdefault("split", {})
    split["seed"] = 7
    split["ratios"] = [0.8, 0.1, 0.1]
    split["structure_holdout"] = {
        "enabled": True,
        "required": True,
        "group_key": "base_case_id",
    }

    scalers = bench.setdefault("preprocessing", {}).setdefault("scalers", {})
    scalers["cond"] = "robust"
    scalers["fit_split"] = "structure_holdout"


def _make_scalar_structure_config(
    base_cfg: dict[str, Any],
    *,
    model: str,
    run_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    bench = cfg.setdefault("benchmark", {})
    bench["output_dir"] = str(run_root / "full" / model).replace("\\", "/")
    bench.setdefault("metadata", {})["input_family"] = "scalar_structure"
    bench["metadata"]["structure_condition_columns"] = list(SCALAR_STRUCTURE_COND_COLUMNS)
    bench["metadata"]["case_varying_structure_maps"] = False

    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["protocol_variant"] = f"icp_stage4_scalar_structure_v2_{model}"
    eval_cfg["input_family"] = "scalar_structure"
    _set_structure_holdout_protocol(bench)

    dataset = bench.setdefault("dataset", {})
    dataset["root"] = str(dataset_root).replace("\\", "/")
    dataset["index_csv"] = "index.csv"
    dataset["cond_columns"] = list(SCALAR_STRUCTURE_COND_COLUMNS)
    dataset.pop("structure_npz_column", None)
    dataset.setdefault("geometry_root", "geometry")

    _set_coord_features_scalar_structure(bench)
    if model == "deeponet_pod":
        bench.setdefault("preprocessing", {})["coord_features"] = {"enabled": False}
    _set_grid_input_features_scalar_structure(bench, model=model)
    _apply_latest_best_model_recipe(bench, model=model)
    _set_runtime_scalar_structure(cfg, model=model)
    return cfg


def _resolve_u_no_recipe(name: str) -> dict[str, Any]:
    selected = [dict(recipe) for recipe in SCALAR_U_NO_RECIPES if str(recipe.get("recipe", "")) == str(name)]
    if len(selected) != 1:
        raise ValueError(f"Unknown or ambiguous U-NO recipe={name!r}")
    return selected[0]


def _load_source_config(args: argparse.Namespace, *, model: str) -> dict[str, Any]:
    source_root = Path(args.source_config_root)
    source_model = "fno" if model == "u_no" else model
    src = source_root / "full" / f"benchmark_icp_stage4_core4_full_{source_model}.yaml"
    if not src.exists():
        raise FileNotFoundError(f"Missing source config for {model}: {src}")
    with src.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    if model != "u_no":
        return base_cfg
    seed = int(dict(base_cfg.get("benchmark", {})).get("seed", 411))
    recipe = _resolve_u_no_recipe(str(args.u_no_recipe))
    cfg = _make_uno_cfg(
        base_cfg,
        recipe=recipe,
        run_root=Path(args.run_root),
        epochs=int(args.u_no_epochs),
        seed=seed,
    )
    metadata = cfg.setdefault("benchmark", {}).setdefault("metadata", {})
    metadata["model_recipe"] = str(recipe["recipe"])
    metadata["model_recipe_settings"] = {
        "epochs": int(args.u_no_epochs),
        "lr": float(recipe["lr"]),
        "n_modes": int(recipe["n_modes"]),
        "width": int(recipe["width"]),
        "n_layers": int(recipe["n_layers"]),
        "batch_size_cases": 2,
    }
    return cfg


def _validate_optimizer(
    family_cfg: dict[str, Any],
    *,
    model: str,
    epochs: int,
    lr: float,
) -> None:
    if int(family_cfg.get("epochs", -1)) != int(epochs):
        raise ValueError(f"scalar {model} recipe epochs must be {epochs}")
    if float(family_cfg.get("lr", -1.0)) != float(lr):
        raise ValueError(f"scalar {model} recipe lr must be {lr}")
    optimizer = dict(family_cfg.get("optimizer", {}) or {})
    expected = _adamw_cosine_optimizer(lr=lr)
    for key, value in expected.items():
        if optimizer.get(key) != value:
            raise ValueError(f"scalar {model} optimizer.{key} must be {value!r}")


def _validate_selected_model_recipe(bench: dict[str, Any], *, model: str) -> None:
    train_cfg = dict(bench.get("train", {}) or {})
    shared_cfg = dict(train_cfg.get("unet_like", {}) or {})
    metadata = dict(bench.get("metadata", {}) or {})

    if model == "unet":
        if metadata.get("model_recipe") != UNET_BEST_RECIPE:
            raise ValueError(f"scalar unet must use recipe={UNET_BEST_RECIPE}")
        if int(shared_cfg.get("batch_size_cases", -1)) != 4:
            raise ValueError("scalar unet effective batch_size_cases must be 4")
        family_cfg = dict(train_cfg.get("unet", {}) or {})
        _validate_optimizer(family_cfg, model=model, epochs=70, lr=3.0e-4)
        model_cfg = dict(family_cfg.get("model_cfg", {}) or {})
        conv_cfg = dict(model_cfg.get("conv_cfg", {}) or {})
        heads_cfg = dict(model_cfg.get("output_heads", {}) or {})
        if (
            model_cfg.get("backend") != "torch"
            or int(model_cfg.get("base_channels", -1)) != 32
            or model_cfg.get("upsample_mode") != "deconv"
            or int(conv_cfg.get("depth", -1)) != 2
            or heads_cfg.get("mode") != "shared"
        ):
            raise ValueError("scalar unet compact architecture contract mismatch")
        return

    if model == "ffno":
        if metadata.get("model_recipe") != FFNO_BEST_RECIPE:
            raise ValueError(f"scalar ffno must use recipe={FFNO_BEST_RECIPE}")
        if int(shared_cfg.get("batch_size_cases", -1)) != 2:
            raise ValueError("scalar ffno effective batch_size_cases must be 2")
        family_cfg = dict(train_cfg.get("ffno", {}) or {})
        if int(family_cfg.get("batch_size_cases", -1)) != 2:
            raise ValueError("scalar ffno declared batch_size_cases must be 2")
        _validate_optimizer(family_cfg, model=model, epochs=70, lr=4.0e-4)
        model_cfg = dict(family_cfg.get("model_cfg", {}) or {})
        spectral_cfg = dict(model_cfg.get("spectral_cfg", {}) or {})
        expected_spectral = {
            "width": 48,
            "n_layers": 4,
            "dropout": 0.0,
            "dealias_ratio": 0.85,
            "taper_alpha": 1.5,
            "skip_filter": "none",
            "factorized_cfg": {
                "enabled": True,
                "mode": "separable_1d",
                "share_weights": False,
            },
            "local_skip_cfg": {"enabled": True, "init_scale": 0.0},
        }
        if int(model_cfg.get("n_modes", -1)) != 12 or int(model_cfg.get("fno_n_modes", -1)) != 12:
            raise ValueError("scalar ffno Fourier modes must be 12")
        for key, value in expected_spectral.items():
            if spectral_cfg.get(key) != value:
                raise ValueError(f"scalar ffno spectral_cfg.{key} must be {value!r}")
        return

    if model != "u_no":
        return
    if bench.get("profile") != "m7_u_no_experimental":
        raise ValueError("scalar u_no must use the isolated U-NO profile")
    if dict(bench.get("eval_protocol", {}) or {}).get("scope") != "u_no_isolated":
        raise ValueError("scalar u_no must retain eval_protocol.scope=u_no_isolated")
    if "fno" in train_cfg or "u_no" not in train_cfg:
        raise ValueError("scalar u_no must train benchmark.train.u_no, not benchmark.train.fno")
    if int(shared_cfg.get("batch_size_cases", -1)) != 2:
        raise ValueError("scalar u_no effective batch_size_cases must be 2")
    recipe_name = str(metadata.get("model_recipe", "")).strip()
    if not recipe_name:
        raise ValueError("scalar u_no must record its selected model_recipe")
    recipe = _resolve_u_no_recipe(recipe_name)
    settings = dict(metadata.get("model_recipe_settings", {}) or {})
    expected_settings = {
        "lr": float(recipe["lr"]),
        "n_modes": int(recipe["n_modes"]),
        "width": int(recipe["width"]),
        "n_layers": int(recipe["n_layers"]),
        "batch_size_cases": 2,
    }
    if int(settings.get("epochs", 0)) <= 0:
        raise ValueError("scalar u_no recipe epochs must be positive")
    for key, value in expected_settings.items():
        if settings.get(key) != value:
            raise ValueError(f"scalar u_no model_recipe_settings.{key} must be {value!r}")
    family_cfg = dict(train_cfg.get("u_no", {}) or {})
    _validate_optimizer(
        family_cfg,
        model=model,
        epochs=int(settings.get("epochs", -1)),
        lr=float(settings.get("lr", -1.0)),
    )
    model_cfg = dict(family_cfg.get("model_cfg", {}) or {})
    uno_cfg = dict(model_cfg.get("uno_cfg", {}) or {})
    if (
        int(model_cfg.get("n_modes", -1)) != int(settings.get("n_modes", -2))
        or int(model_cfg.get("fno_n_modes", -1)) != int(settings.get("n_modes", -2))
        or int(uno_cfg.get("width", -1)) != int(settings.get("width", -2))
        or int(uno_cfg.get("n_layers", -1)) != int(settings.get("n_layers", -2))
        or int(family_cfg.get("batch_size_cases", -1)) != int(settings.get("batch_size_cases", -2))
    ):
        raise ValueError("scalar u_no selected recipe contract mismatch")


def _validate_scalar_structure_config(cfg: dict[str, Any], *, model: str) -> None:
    bench = dict(cfg.get("benchmark", {}) or {})
    eval_cfg = dict(bench.get("eval", {}) or {})
    if eval_cfg.get("primary_metric") != "test_r2_group_default_plasma":
        raise ValueError("scalar structure config must use test_r2_group_default_plasma")
    if eval_cfg.get("objective_mode") != "max" or eval_cfg.get("primary_mode") != "max":
        raise ValueError("scalar structure R2 objective must use max mode")
    eval_protocol = dict(bench.get("eval_protocol", {}) or {})
    if eval_protocol.get("mode") != "primary_axis" or eval_protocol.get("primary_split") != "structure_holdout":
        raise ValueError("scalar structure config must use primary_axis structure_holdout evaluation")

    dataset = dict(bench.get("dataset", {}) or {})
    if list(dataset.get("cond_columns", [])) != list(SCALAR_STRUCTURE_COND_COLUMNS):
        raise ValueError("scalar structure cond_columns mismatch")
    if "structure_npz_column" in dataset:
        raise ValueError("scalar structure config must not consume case-varying structure_npz")
    holdout = dict(dict(bench.get("split", {}) or {}).get("structure_holdout", {}) or {})
    if not bool(holdout.get("required", False)) or holdout.get("group_key") != "base_case_id":
        raise ValueError("scalar structure config must require base_case_id structure holdout")

    preprocessing = dict(bench.get("preprocessing", {}) or {})
    if preprocessing.get("coord_grid_source") != "rz_linear":
        raise ValueError("scalar structure config must use the physical rz_linear coordinate grid")
    if dict(preprocessing.get("coord_grid_contract", {}) or {}).get("require_requested_source") != "error":
        raise ValueError("scalar structure config must fail on coordinate-grid fallback")
    if dict(preprocessing.get("scalers", {}) or {}).get("fit_split") != "structure_holdout":
        raise ValueError("scalar structure scalers must fit the structure_holdout train split")

    runtime = dict(bench.get("runtime", {}) or {})
    structure = dict(runtime.get("structure", {}) or {})
    if structure.get("provider_mode") != "fixed":
        raise ValueError("scalar structure config must use a fixed geometry provider")
    if model == "deeponet_pod":
        if runtime.get("input_mode") != "table_only":
            raise ValueError("scalar DeepONet-POD must use table_only input")
        return
    if runtime.get("input_mode") != "table_plus_structure":
        raise ValueError(f"scalar grid model={model} must use table_plus_structure")
    if structure.get("feature_profile") != SCALAR_SPATIAL_PROFILE:
        raise ValueError(f"scalar grid model={model} must use {SCALAR_SPATIAL_PROFILE}")
    input_features = dict(dict(bench.get("train", {}) or {}).get(model, {}).get("input_features", {}) or {})
    if input_features.get("require_pack") != "error":
        raise ValueError(f"scalar grid model={model} must require its spatial feature pack")
    if list(input_features.get("features", [])) != list(SCALAR_SPATIAL_FEATURES):
        raise ValueError(f"scalar grid model={model} static feature channels mismatch")
    _validate_selected_model_recipe(bench, model=model)


def _generate_configs(args: argparse.Namespace, *, models: list[str]) -> list[tuple[str, Path]]:
    out_root = Path(args.scalar_config_root) / "full"
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path]] = []
    for model in models:
        base_cfg = _load_source_config(args, model=model)
        cfg = _make_scalar_structure_config(
            base_cfg,
            model=model,
            run_root=Path(args.run_root),
            dataset_root=Path(args.dataset_root),
        )
        _validate_scalar_structure_config(cfg, model=model)
        out = out_root / f"benchmark_icp_stage4_core4_full_{model}.yaml"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
        generated.append((model, out))
        print(f"[SCALAR CONFIG] model={model} path={out}", flush=True)
    return generated


def _read_status_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _leaderboard_exists(row: dict[str, str]) -> bool:
    raw = str(row.get("leaderboard") or "").strip()
    return bool(raw) and Path(raw).exists()


def _uno_status_text(status_csv: Path, *, process_pattern: str) -> tuple[bool, str]:
    rows = _read_status_rows(status_csv)
    active = _matching_processes([process_pattern])
    running = [
        row
        for row in rows
        if str(row.get("stage", "")).strip() == "uno_tuning"
        and str(row.get("status", "")).strip().lower() == "running"
        and not _leaderboard_exists(row)
    ]
    pending = [
        row
        for row in rows
        if str(row.get("stage", "")).strip() == "uno_tuning"
        and str(row.get("status", "")).strip().lower() in {"", "pending"}
    ]
    done = (not active) and (not running) and (not pending)
    active_text = ",".join(str(row.get("pid", "")) for row in active[:4]) if active else "-"
    running_text = ",".join(str(row.get("recipe", "")) for row in running[:4]) if running else "-"
    return done, f"uno_process_pids={active_text} running_recipes={running_text} rows={len(rows)}"


def _wait_for_uno_direct(args: argparse.Namespace) -> None:
    if args.skip_wait_uno:
        print("[WAIT UNO] skipped by --skip-wait-uno", flush=True)
        return
    status_csv = Path(args.uno_status_csv)
    poll = max(int(args.poll_seconds), 10)
    while True:
        done, text = _uno_status_text(status_csv, process_pattern=str(args.uno_process_pattern))
        print(f"[WAIT UNO] done={done} {text}", flush=True)
        if done:
            return
        time.sleep(poll)


def _run_summary(args: argparse.Namespace, *, models: list[str]) -> Path:
    summary_dir = Path(args.summary_dir or (Path(args.run_root) / "summary"))
    argv = [
        "experiments/icp_stage4/scripts/summarize_icp_stage4_core4_benchmarks.py",
        "--sizes",
        "full",
        "--models",
        *models,
        "--config-root",
        str(Path(args.scalar_config_root)),
        "--run-root",
        str(Path(args.run_root)),
        "--out-dir",
        str(summary_dir),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv
    return summary_dir / "comparison_summary.csv"


def _all_leaderboards_complete(items: list[tuple[str, Path]]) -> bool:
    return all(_leaderboard_for_config(cfg_path).exists() for _, cfg_path in items)


def _completed_models(items: list[tuple[str, Path]]) -> list[str]:
    return [model for model, cfg_path in items if _leaderboard_for_config(cfg_path).exists()]


def main() -> int:
    args = _parse_args()
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    _set_below_normal_priority()
    _limit_threads()

    models = _norm_model_list(list(args.models))
    generated = _generate_configs(args, models=models)
    if args.generate_only:
        print("[SCALAR PLAN] generate-only complete", flush=True)
        return 0

    _wait_for_uno_direct(args)

    run_root = Path(args.run_root)
    logs = run_root / "logs"
    status_csv = Path(args.status_csv or (run_root / "scalar_structure_status.csv"))
    wait_absent = list(args.wait_command_absent or [])
    if str(args.uno_process_pattern) not in wait_absent:
        wait_absent.append(str(args.uno_process_pattern))
    idle_args = argparse.Namespace(
        poll_seconds=int(args.poll_seconds),
        idle_checks=int(args.idle_checks),
        max_gpu_util=float(args.max_gpu_util),
        max_gpu_mem_mib=float(args.max_gpu_mem_mib),
        wait_command_absent=wait_absent,
    )

    for model, cfg_path in generated:
        if not _leaderboard_for_config(cfg_path).exists():
            _wait_until_idle(idle_args, label=f"scalar_structure:{model}")
        ok = _run_config(
            cfg_path,
            stage="scalar_structure_full",
            model=model,
            recipe="",
            status_csv=status_csv,
            log_dir=logs,
            stop_on_failure=bool(args.stop_on_failure),
        )
        if ok and _leaderboard_for_config(cfg_path).exists():
            done_models = _completed_models(generated)
            if done_models:
                _run_summary(args, models=done_models)
        else:
            print(f"[SCALAR PLAN] summary skipped after failed model={model}", flush=True)

    complete = _all_leaderboards_complete(generated)
    summary_csv = _run_summary(args, models=models) if complete else Path(args.summary_dir or (Path(args.run_root) / "summary")) / "comparison_summary.csv"
    print(f"[SCALAR PLAN] complete={complete} summary={summary_csv}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
