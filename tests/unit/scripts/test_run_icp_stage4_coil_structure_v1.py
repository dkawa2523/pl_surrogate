from __future__ import annotations

import runpy
from pathlib import Path

import yaml


SCRIPT = Path("experiments/icp_stage4/scripts/run_icp_stage4_coil_structure_v1.py")


def test_generated_configs_share_representation_and_only_process_conditions(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="coil_structure_v1_test")
    paths = module["generate_configs"](
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        epochs=3,
    )
    assert set(paths) == {"unet", "u_no"}
    for model_name, path in paths.items():
        bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
        assert bench["dataset"]["cond_columns"] == ["pp", "pp0"]
        assert bench["runtime"]["structure"]["feature_profile"] == "part_source_v1"
        assert bench["train"][model_name]["input_features"]["features"] == [
            "x",
            "y",
            "distance_signed",
            "part_sdf_union",
            "part_source_sum",
        ]
        assert bench["train"][model_name]["epochs"] == 3
    uno = yaml.safe_load(paths["u_no"].read_text(encoding="utf-8"))["benchmark"]
    assert uno["train"]["u_no"]["model_cfg"]["uno_cfg"]["padding_fraction"] == 0.08
    assert uno["train"]["u_no"]["model_cfg"]["uno_cfg"]["padding_mode"] == "reflect"


def test_linear_target_recipe_has_no_nonlinear_transform_or_spatial_loss(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="linear_targets_v1_test")
    paths = module["generate_configs"](
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        epochs=3,
        linear_targets=True,
    )
    for model_name, path in paths.items():
        bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
        transforms = bench["preprocessing"]["scalers"]["target_transforms"]
        assert set(transforms) == {"ne", "ni", "Te", "phi"}
        assert all(spec["value_transform"] == "identity" for spec in transforms.values())
        assert all(spec["scaler"] == "zscore" for spec in transforms.values())
        assert all(spec["clip"]["mode"] == "none" for spec in transforms.values())
        assert bench["train"]["loss"]["supervised"]["type"] == "mse"
        assert "huber_delta" not in bench["train"]["loss"]["supervised"]
        assert "spatial" not in bench["train"]["loss"]["supervised"]
        assert bench["train"][model_name]["selection"] == {"mode": "best_val_loss"}


def test_axisymmetric_energy_weighted_recipe_is_shared_by_unet_and_uno(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="axisymmetric_energy_weighted_v1_test")
    paths = module["generate_configs"](
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        epochs=3,
        linear_targets=True,
        physical_weighting=True,
    )
    for model_name, path in paths.items():
        bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
        assert bench["metadata"]["study"] == "icp_axisymmetric_energy_weighted_v1"
        physical = bench["train"]["loss"]["supervised"]["physical_weighting"]
        assert physical == {
            "axisymmetric_volume": True,
            "density_source": "ne",
            "density_weighted_targets": ["Te"],
        }
        assert bench["train"][model_name]["selection"] == {"mode": "best_val_loss"}


def test_simple_bohm_optimizer_has_one_objective_and_joint_search_space(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT), run_name="simple_bohm_wafer_opt_v1_test")
    paths = module["generate_configs"](
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        epochs=3,
        linear_targets=True,
        physical_weighting=True,
        simple_bohm_optimize=True,
    )
    for path in paths.values():
        bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
        assert bench["metadata"]["study"] == "icp_simple_bohm_wafer_opt_v1"
        qoi = bench["inference"]["qoi"]
        assert qoi["uniformity"]["region"] == "wafer_near"
        assert qoi["bohm_flux"] == {"density_target": "ni", "temperature_target": "Te"}
        optimize = bench["inference"]["optimize"]
        assert set(optimize["space"]) == {"pp", "pp0"}
        assert set(optimize["geom_space"]) == {
            f"part.coil_{index:02d}.tx" for index in range(1, 7)
        }
        assert optimize["objective"]["terms"] == [
            {"key": "bohm_flux_area_cv", "direction": "min", "weight": 1.0}
        ]
        assert optimize["constraints"] == [
            {"key": "bohm_flux_area_mean", "lower": 6.0249874e16}
        ]
