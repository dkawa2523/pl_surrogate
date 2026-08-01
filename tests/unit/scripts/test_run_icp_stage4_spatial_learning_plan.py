from __future__ import annotations

import runpy
from pathlib import Path

import yaml


SCRIPT = Path("experiments/icp_stage4/scripts/run_icp_stage4_spatial_learning_plan.py")


def _module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT), run_name="icp_stage4_spatial_learning_plan_test")


def test_generate_staged_spatial_learning_configs(tmp_path: Path) -> None:
    module = _module()
    source = {
        "benchmark": {
            "dataset": {"cond_columns": ["pp", "pp0"]},
            "preprocessing": {"scalers": {}, "coord_features": {}},
            "runtime": {"input_mode": "table_plus_structure", "structure": {}},
            "train": {
                "unet": {"model_cfg": {}, "optimizer": {}, "selection": {}},
                "loss": {"supervised": {"type": "huber"}},
            },
            "eval": {},
        }
    }
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    paths = module["generate_configs"](
        source_path=source_path,
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
    )
    assert set(paths) == set(module["VARIANTS"])

    smooth = yaml.safe_load(paths["smooth_structure"].read_text(encoding="utf-8"))["benchmark"]
    assert smooth["runtime"]["structure"]["feature_profile"] == "smooth_structure_v1"
    assert smooth["dataset"]["cond_columns"] == module["COND_COLUMNS"]
    assert smooth["train"]["unet"]["selection"]["mode"] == "best_val_loss"
    assert "spatial" not in smooth["train"]["loss"]["supervised"]

    spatial = yaml.safe_load(paths["smooth_spatial_loss"].read_text(encoding="utf-8"))["benchmark"]
    assert spatial["train"]["loss"]["supervised"]["spatial"] == {
        "gradient_weight": 0.10,
        "multiscale_weight": 0.05,
        "multiscale_scales": [2, 4],
    }
    for spec in spatial["preprocessing"]["scalers"]["target_transforms"].values():
        assert spec["clip"]["mode"] == "physical_bounds"

    relative = yaml.safe_load(
        paths["relative_spatial_objective"].read_text(encoding="utf-8")
    )["benchmark"]
    relative_loss = relative["train"]["loss"]["supervised"]["spatial"]
    relative_selection = relative["train"]["unet"]["selection"]
    assert relative_loss["gradient_normalization"] == "target_rms"
    assert relative_loss["gradient_epsilon"] == 0.05
    assert relative_selection["mode"] == "best_val_spatial_objective"
    assert relative_selection["spatial"]["gradient_normalization"] == "target_rms"
    assert relative_selection["case_aggregation"] == {
        "median_weight": 1.0,
        "p90_weight": 0.25,
        "worst_weight": 0.05,
    }
    assert relative["eval"]["quality_score"]["target_aggregation"] == "uniform_by_target"


def test_generate_uno_config_applies_winning_recipe_without_unet_lane(tmp_path: Path) -> None:
    module = _module()
    source = {
        "benchmark": {
            "dataset": {"cond_columns": ["pp", "pp0"]},
            "preprocessing": {"scalers": {}, "coord_features": {}},
            "runtime": {"input_mode": "table_plus_structure", "structure": {}},
            "train": {
                "u_no": {"model_cfg": {}, "optimizer": {}, "selection": {}},
                "loss": {"supervised": {"type": "huber"}},
            },
            "eval": {},
        }
    }
    source_path = tmp_path / "uno_source.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    path = module["generate_uno_config"](
        source_path=source_path,
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        variant="smooth_spatial_loss",
    )
    bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
    assert "unet" not in bench["train"]
    assert bench["train"]["u_no"]["model_cfg"]["uno_cfg"] == {
        "width": 48,
        "n_layers": 4,
        "dropout": 0.0,
    }
    assert bench["train"]["u_no"]["selection"]["mode"] == "best_val_loss"


def test_generate_uno_config_reuses_relative_spatial_selection(tmp_path: Path) -> None:
    module = _module()
    source = {
        "benchmark": {
            "dataset": {"cond_columns": ["pp", "pp0"]},
            "preprocessing": {"scalers": {}, "coord_features": {}},
            "runtime": {"input_mode": "table_plus_structure", "structure": {}},
            "train": {
                "u_no": {"model_cfg": {}, "optimizer": {}, "selection": {}},
                "loss": {"supervised": {"type": "huber"}},
            },
            "eval": {},
        }
    }
    source_path = tmp_path / "uno_source.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    path = module["generate_uno_config"](
        source_path=source_path,
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        variant="relative_spatial_objective",
    )
    bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
    selection = bench["train"]["u_no"]["selection"]
    assert selection["mode"] == "best_val_spatial_objective"
    assert selection["spatial"]["gradient_normalization"] == "target_rms"
