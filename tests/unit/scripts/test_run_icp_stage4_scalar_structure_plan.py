from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from plasma_surrogate.features.structure_feature_registry import (
    resolve_spatial_channels_for_feature_profile,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "experiments" / "icp_stage4" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_icp_stage4_scalar_structure_plan import (  # noqa: E402
    DATASET_ROOT_DEFAULT,
    GRID_MODELS,
    MODEL_ORDER,
    RUN_ROOT_DEFAULT,
    SCALAR_CONFIG_ROOT_DEFAULT,
    SCALAR_SPATIAL_FEATURES,
    SCALAR_SPATIAL_PROFILE,
    SCALAR_STRUCTURE_COND_COLUMNS,
    SOURCE_CONFIG_ROOT_DEFAULT,
    U_NO_EPOCHS_DEFAULT,
    U_NO_RECIPE_DEFAULT,
    _generate_configs,
    _make_scalar_structure_config,
    _parse_args,
    _validate_scalar_structure_config,
)
from summarize_icp_stage4_core4_benchmarks import CORE_MODELS  # noqa: E402


def _source_config(model: str) -> dict[str, Any]:
    model_cfg: dict[str, Any]
    if model == "unet":
        model_cfg = {
            "backend": "torch",
            "base_channels": 64,
            "conv_cfg": {"depth": 4},
            "output_heads": {"mode": "role_grouped"},
        }
    elif model in {"fno", "ffno"}:
        model_cfg = {
            "n_modes": 16,
            "fno_n_modes": 16,
            "spectral_cfg": {"width": 64, "n_layers": 5},
        }
    else:
        model_cfg = {}

    family_cfg: dict[str, Any] = {
        "epochs": 80,
        "lr": 8.0e-4,
        "optimizer": {
            "type": "adamw",
            "lr": 8.0e-4,
            "weight_decay": 1.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "schedule": "cosine",
            "warmup_epochs": 0,
        },
        "target_family": "allvars",
        "target_vars": ["ne", "ni", "Te", "phi"],
        "selection": {
            "mode": "best_val_allvars_balance",
            "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
        },
        "model_cfg": model_cfg,
        "batch_size_cases": 7,
    }
    if model != "deeponet_pod":
        family_cfg["input_features"] = {
            "mode": "geom_feature_pack",
            "require_pack": "off",
            "features": ["x", "part_sdf_nearest"],
        }

    return {
        "benchmark": {
            "profile": f"m7_{model}_isolated",
            "seed": 411,
            "eval": {
                "primary_metric": "surrogate_quality_score",
                "primary_mode": "min",
                "objective_mode": "min",
            },
            "eval_protocol": {
                "mode": "dual_axis",
                "primary_split": "interp",
                "min_primary_test_cases": 1,
                "min_primary_test_groups": 1,
            },
            "dataset": {
                "type": "csv_npz",
                "root": "data/legacy",
                "index_csv": "index_old.csv",
                "cond_columns": ["pp", "pp0"],
                "fields_npz_column": "fields_npz",
                "case_id_column": "case_id",
                "base_case_id_column": "base_case_id",
                "split_group_column": "split_group",
                "geometry_root": "geometry",
                "structure_npz_column": "structure_npz",
            },
            "split": {
                "seed": 99,
                "ratios": [0.6, 0.2, 0.2],
                "structure_holdout": {
                    "enabled": False,
                    "required": False,
                    "group_key": "split_group",
                },
            },
            "preprocessing": {
                "scalers": {"cond": "zscore", "fit_split": "interp"},
                "coord_grid_source": "coord_grid",
                "coord_grid_contract": {"require_requested_source": "off"},
                "coord_features": {
                    "enabled": True,
                    "channels": ["x", "part_sdf_nearest"],
                    "channels_from_profile": "part_lite_v1",
                    "case_output": "features/legacy_case_pack.npz",
                    "case_structure_output": "features/legacy_structure_pack.npz",
                    "require_case_variation": True,
                    "coil_proximity_tau_fixed": 3.0,
                },
            },
            "train": {
                "unet_like": {"batch_size_cases": 6, "shuffle_cases": True},
                model: family_cfg,
                "loss": {
                    "supervised": {
                        "type": "huber",
                        "mask": "plasma_only",
                        "huber_delta": 1.0,
                    }
                },
            },
            "runtime": {
                "input_mode": "table_plus_structure",
                "strict_input_mode": "off",
                "structure": {
                    "feature_profile": "part_lite_v1",
                    "descriptor_profile": "struct_desc_v2",
                    "latent_profile": "none",
                    "adapter_mode": "descriptor_branch",
                    "provider_mode": "parametric_parts",
                },
            },
        }
    }


def _write_source(root: Path, model: str) -> Path:
    path = root / "full" / f"benchmark_icp_stage4_core4_full_{model}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_source_config(model), sort_keys=False), encoding="utf-8")
    return path


def _generation_args(tmp_path: Path, source_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source_config_root=str(source_root),
        scalar_config_root=str(tmp_path / "generated"),
        run_root=str(tmp_path / "runs"),
        dataset_root=str(tmp_path / "dataset_part_lite_v2"),
        u_no_recipe=U_NO_RECIPE_DEFAULT,
        u_no_epochs=U_NO_EPOCHS_DEFAULT,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def test_defaults_use_latest_isolated_roots_and_include_u_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_DIR / "run_icp_stage4_scalar_structure_plan.py")])
    args = _parse_args()

    assert Path(args.source_config_root) == SOURCE_CONFIG_ROOT_DEFAULT == Path(
        "configs/experimental/icp_stage4/generated_major_fixes_part_lite_v2_e80"
    )
    assert Path(args.dataset_root) == DATASET_ROOT_DEFAULT == Path(
        "data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2"
    )
    assert Path(args.scalar_config_root) == SCALAR_CONFIG_ROOT_DEFAULT == Path(
        "configs/experimental/icp_stage4/generated_scalar_dimension_structure_holdout_v2"
    )
    assert Path(args.run_root) == RUN_ROOT_DEFAULT == Path(
        "runs/icp_stage4_scalar_dimension_structure_holdout_v2"
    )
    assert list(args.models) == list(MODEL_ORDER)
    assert list(args.models[:4]) == ["u_no", "unet", "ffno", "deeponet_pod"]
    assert "u_no" in GRID_MODELS
    assert args.u_no_recipe == U_NO_RECIPE_DEFAULT == "u_no_m10_w48_l4_lr3e4"
    assert args.u_no_epochs == U_NO_EPOCHS_DEFAULT == 70


def test_scalar_config_overrides_legacy_protocol_and_coordinate_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "part_lite_v2"
    cfg = _make_scalar_structure_config(
        _source_config("unet"),
        model="unet",
        run_root=tmp_path / "runs",
        dataset_root=dataset_root,
    )
    benchmark = cfg["benchmark"]

    assert benchmark["eval"]["primary_metric"] == "test_r2_group_default_plasma"
    assert benchmark["eval"]["primary_mode"] == "max"
    assert benchmark["eval"]["objective_mode"] == "max"
    assert benchmark["eval_protocol"]["mode"] == "primary_axis"
    assert benchmark["eval_protocol"]["primary_split"] == "structure_holdout"
    assert benchmark["eval_protocol"]["min_primary_test_cases"] == 3
    assert benchmark["eval_protocol"]["min_primary_test_groups"] == 3
    assert benchmark["split"] == {
        "seed": 7,
        "ratios": [0.8, 0.1, 0.1],
        "structure_holdout": {
            "enabled": True,
            "required": True,
            "group_key": "base_case_id",
        },
    }

    preprocessing = benchmark["preprocessing"]
    assert preprocessing["scalers"]["cond"] == "robust"
    assert preprocessing["scalers"]["fit_split"] == "structure_holdout"
    assert preprocessing["coord_grid_source"] == "rz_linear"
    assert preprocessing["coord_grid_contract"]["require_requested_source"] == "error"

    dataset = benchmark["dataset"]
    assert Path(dataset["root"]) == dataset_root
    assert dataset["index_csv"] == "index.csv"
    assert dataset["cond_columns"] == list(SCALAR_STRUCTURE_COND_COLUMNS)
    assert dataset["fields_npz_column"] == "fields_npz"
    assert dataset["base_case_id_column"] == "base_case_id"
    assert dataset["split_group_column"] == "split_group"
    assert "structure_npz_column" not in dataset
    _validate_scalar_structure_config(cfg, model="unet")


@pytest.mark.parametrize("model", ["u_no", "unet", "ffno"])
def test_grid_scalar_config_uses_exact_static9_fixed_geometry(
    tmp_path: Path,
    model: str,
) -> None:
    cfg = _make_scalar_structure_config(
        _source_config(model),
        model=model,
        run_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset",
    )
    benchmark = cfg["benchmark"]
    runtime = benchmark["runtime"]
    coord_features = benchmark["preprocessing"]["coord_features"]
    input_features = benchmark["train"][model]["input_features"]

    assert runtime == cfg["runtime"]
    assert runtime["input_mode"] == "table_plus_structure"
    assert runtime["strict_input_mode"] == "error"
    assert runtime["structure"] == {
        "feature_profile": SCALAR_SPATIAL_PROFILE,
        "descriptor_profile": "none",
        "latent_profile": "none",
        "adapter_mode": "auto",
        "provider_mode": "fixed",
    }
    assert tuple(SCALAR_SPATIAL_FEATURES) == (
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
    assert resolve_spatial_channels_for_feature_profile(SCALAR_SPATIAL_PROFILE) == tuple(
        SCALAR_SPATIAL_FEATURES
    )
    assert coord_features["channels_from_profile"] == SCALAR_SPATIAL_PROFILE
    for forbidden in (
        "channels",
        "static_output",
        "case_output",
        "case_structure_output",
        "require_case_variation",
        "coil_proximity_tau_fixed",
    ):
        assert forbidden not in coord_features
    assert input_features["mode"] == "geom_feature_pack"
    assert input_features["require_pack"] == "error"
    assert input_features["features"] == list(SCALAR_SPATIAL_FEATURES)
    assert not any(name.startswith("part_") for name in input_features["features"])
    assert "structure_npz_column" not in benchmark["dataset"]
    assert benchmark["metadata"]["case_varying_structure_maps"] is False
    if model != "u_no":
        _validate_scalar_structure_config(cfg, model=model)


def test_deeponet_pod_scalar_config_is_table_only(tmp_path: Path) -> None:
    cfg = _make_scalar_structure_config(
        _source_config("deeponet_pod"),
        model="deeponet_pod",
        run_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset",
    )
    benchmark = cfg["benchmark"]
    runtime = benchmark["runtime"]

    assert runtime == cfg["runtime"]
    assert runtime["input_mode"] == "table_only"
    assert runtime["strict_input_mode"] == "error"
    assert runtime["structure"] == {
        "feature_profile": "none",
        "descriptor_profile": "none",
        "latent_profile": "none",
        "adapter_mode": "none",
        "provider_mode": "fixed",
    }
    assert benchmark["preprocessing"]["coord_features"] == {"enabled": False}
    assert benchmark["dataset"]["cond_columns"] == list(SCALAR_STRUCTURE_COND_COLUMNS)
    assert "structure_npz_column" not in benchmark["dataset"]
    assert benchmark["eval_protocol"]["primary_split"] == "structure_holdout"
    assert benchmark["preprocessing"]["scalers"]["fit_split"] == "structure_holdout"
    _validate_scalar_structure_config(cfg, model="deeponet_pod")


def test_generate_u_no_from_temp_fno_source_uses_adopted_recipe(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, "fno")
    args = _generation_args(tmp_path, source_root)

    generated = _generate_configs(args, models=["u_no"])

    assert [model for model, _ in generated] == ["u_no"]
    cfg_path = generated[0][1]
    assert cfg_path.name == "benchmark_icp_stage4_core4_full_u_no.yaml"
    benchmark = _load_yaml(cfg_path)["benchmark"]
    family = benchmark["train"]["u_no"]
    model_cfg = family["model_cfg"]
    assert family["epochs"] == 70
    assert family["lr"] == pytest.approx(3.0e-4)
    assert family["optimizer"]["lr"] == pytest.approx(3.0e-4)
    assert family["batch_size_cases"] == 2
    assert model_cfg["n_modes"] == 10
    assert model_cfg["fno_n_modes"] == 10
    assert model_cfg["uno_cfg"]["width"] == 48
    assert model_cfg["uno_cfg"]["n_layers"] == 4
    assert benchmark["train"]["unet_like"]["batch_size_cases"] == 2
    assert "fno" not in benchmark["train"]


def test_generated_unet_and_ffno_keep_adopted_best_settings(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, "unet")
    _write_source(source_root, "ffno")
    args = _generation_args(tmp_path, source_root)

    generated = dict(_generate_configs(args, models=["unet", "ffno"]))
    unet = _load_yaml(generated["unet"])["benchmark"]["train"]
    unet_family = unet["unet"]
    assert unet_family["epochs"] == 70
    assert unet_family["lr"] == pytest.approx(3.0e-4)
    assert unet_family["optimizer"]["lr"] == pytest.approx(3.0e-4)
    assert unet["unet_like"]["batch_size_cases"] == 4
    assert unet_family["model_cfg"]["output_heads"]["mode"] == "shared"

    ffno = _load_yaml(generated["ffno"])["benchmark"]["train"]
    ffno_family = ffno["ffno"]
    ffno_model_cfg = ffno_family["model_cfg"]
    assert ffno_family["epochs"] == 70
    assert ffno_family["lr"] == pytest.approx(4.0e-4)
    assert ffno_family["optimizer"]["lr"] == pytest.approx(4.0e-4)
    assert ffno["unet_like"]["batch_size_cases"] == 2
    assert ffno_family["batch_size_cases"] == 2
    assert ffno_model_cfg["n_modes"] == 12
    assert ffno_model_cfg["fno_n_modes"] == 12
    assert ffno_model_cfg["spectral_cfg"]["width"] == 48
    assert ffno_model_cfg["spectral_cfg"]["n_layers"] == 4


@pytest.mark.parametrize(("model", "invalid_batch"), [("unet", 6), ("ffno", 6)])
def test_validator_rejects_stale_effective_shared_batch(
    tmp_path: Path,
    model: str,
    invalid_batch: int,
) -> None:
    cfg = _make_scalar_structure_config(
        _source_config(model),
        model=model,
        run_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset",
    )
    cfg["benchmark"]["train"]["unet_like"]["batch_size_cases"] = invalid_batch

    with pytest.raises(ValueError, match="effective batch_size_cases"):
        _validate_scalar_structure_config(cfg, model=model)


def test_icp_stage4_summarizer_accepts_u_no() -> None:
    assert "u_no" in CORE_MODELS
