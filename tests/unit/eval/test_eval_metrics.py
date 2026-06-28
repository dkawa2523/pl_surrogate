from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from plasma_surrogate.eval.core_metrics import build_benchmark_eval_row, build_region_metrics
from plasma_surrogate.eval.payloads import build_eval_metrics_payload, build_viz_tables_payload
from plasma_surrogate.eval.physics_metrics import build_single_case_physics_metrics
from plasma_surrogate.eval.spatial_metrics import build_spatial_distribution_by_case_rows


def test_build_single_case_physics_metrics_reads_scalar_diagnostics(tmp_path: Path):
    case_dir = tmp_path / "abc123"
    case_dir.mkdir(parents=True)
    with (case_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "physics_diagnostics_available": True,
                "poisson_residual_norm": 1.5,
                "bc_potential_mae": 0.25,
                "poisson_residual_map_l2": 0.75,
                "boundary_operator_residual_map_l2": 1.25,
            },
            f,
        )
    with (case_dir / "qoi.json").open("w", encoding="utf-8") as f:
        json.dump({"uniformity": 0.12}, f)
    row = build_single_case_physics_metrics(case_dir)
    assert row["case_key"] == "abc123"
    assert row["physics_diagnostics_available"] is True
    assert float(row["poisson_residual_norm"]) == 1.5
    assert float(row["bc_potential_mae"]) == 0.25
    assert float(row["poisson_residual_map_l2"]) == 0.75
    assert float(row["boundary_residual_map_l2"]) == 1.25


def test_build_region_metrics_and_benchmark_row():
    density = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    masks = {
        "plasma": np.array([[1, 1], [1, 1]], dtype=bool),
        "bulk": np.array([[1, 0], [0, 1]], dtype=bool),
        "boundary": np.array([[0, 1], [1, 0]], dtype=bool),
    }
    region = build_region_metrics(
        case_key="k1",
        density=density,
        mask_plasma=masks["plasma"],
        mask_bulk=masks["bulk"],
        mask_boundary=masks["boundary"],
    )
    assert region["case_key"] == "k1"
    assert float(region["bulk_mean_density"]) == 2.5

    pred = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4) / 32.0
    row = build_benchmark_eval_row(
        model_id="global_mlp",
        metrics={
            "electron_density": 0.1,
            "ion_density": 0.11,
            "electron_temperature": 0.2,
            "potential": 0.3,
        },
        r2_scores={
            "electron_density": 0.91,
            "ion_density": 0.89,
            "electron_temperature": 0.82,
            "potential": 0.73,
        },
        pred_eval={
            "potential": pred,
            "electron_density": pred + 1.0,
            "ion_density": pred + 1.25,
            "electron_temperature": pred + 2.0,
        },
        true_eval={
            "potential": pred + 0.5,
            "electron_density": pred + 1.5,
            "ion_density": pred + 1.75,
            "electron_temperature": pred + 2.5,
        },
        mask_plasma=np.ones((4, 4), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.6, "boundary_operator_proxy_loss": 0.7},
        target_vars_for_score=["electron_density", "ion_density"],
    )
    assert row["model_id"] == "global_mlp"
    assert float(row["test_r2_potential"]) == 0.73
    assert "test_rmse_potential_plasma" in row
    assert "test_r2_potential_plasma" in row
    assert "score_total" not in row
    diagnostics = row["_diagnostics"]
    assert "test_rmse_electron_density_boundary_in" in diagnostics
    assert "test_rmse_ion_density_boundary_in" in diagnostics
    assert "test_r2_electron_density_plasma_deep" in diagnostics
    assert "test_r2_ion_density_plasma_deep" in diagnostics
    assert "test_neg_ratio_electron_density_plasma" in diagnostics
    assert "test_neg_ratio_ion_density_plasma" in diagnostics
    assert "continuity_grad_ratio_all_plasma" in diagnostics
    assert "continuity_lap_ratio_all_plasma" in diagnostics
    assert "surrogate_quality_score" in row
    assert np.isfinite(float(row["surrogate_quality_score"]))
    assert json.loads(str(diagnostics["quality_components"]))["surrogate_quality_score"] == float(
        row["surrogate_quality_score"]
    )
    assert json.loads(str(diagnostics["validity_flags"]))["target_metrics_valid"] is True
    assert "score_nrmse_component" in diagnostics
    assert "test_r2_potential" in row


def test_spatial_distribution_uses_eval_region_by_var_not_loss_contract():
    true = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    pred = true + 1.0
    mask = np.array([[1, 0], [0, 0]], dtype=np.float32)

    plasma_rows = build_spatial_distribution_by_case_rows(
        pred_eval={"density": pred},
        true_eval={"density": true},
        mask_plasma=mask,
        vars_for_summary=["density"],
    )
    all_domain_rows = build_spatial_distribution_by_case_rows(
        pred_eval={"density": pred},
        true_eval={"density": true},
        mask_plasma=mask,
        vars_for_summary=["density"],
        region_by_var={"density": "all_domain"},
    )

    assert plasma_rows[0]["target_region"] == "plasma_only"
    assert plasma_rows[0]["n_points"] == 1.0
    assert all_domain_rows[0]["target_region"] == "all_domain"
    assert all_domain_rows[0]["n_points"] == 4.0


def test_quality_score_uses_unitless_normalized_rmse():
    mask = np.ones((2, 2), dtype=np.float32)
    true_small = np.array([[[[0.0, 2.0], [0.0, 2.0]]]], dtype=np.float32)
    pred_small = true_small + 1.0
    true_large = true_small * 1000.0
    pred_large = true_large + 1000.0
    row = build_benchmark_eval_row(
        model_id="scale_test",
        metrics={"small": 1.0, "large": 1000.0},
        r2_scores={"small": 0.0, "large": 0.0},
        pred_eval={"small": pred_small, "large": pred_large},
        true_eval={"small": true_small, "large": true_large},
        mask_plasma=mask,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["small", "large"],
    )

    diagnostics = row["_diagnostics"]
    assert np.isclose(float(diagnostics["mean_nrmse_plasma_by_target"]), 1.0)
    assert np.isclose(float(diagnostics["score_nrmse_component"]), 1.0)


def test_quality_score_ignores_map_diagnostics_on_default_path():
    mask = np.ones((2, 2), dtype=np.float32)
    true = np.ones((1, 1, 2, 2), dtype=np.float32)
    pred = true.copy()
    base = {
        "model_id": "map_diag",
        "metrics": {"density": 0.0},
        "r2_scores": {"density": 1.0},
        "pred_eval": {"density": pred},
        "true_eval": {"density": true},
        "mask_plasma": mask,
        "target_vars_for_score": ["density"],
    }
    scalar = build_benchmark_eval_row(
        **base,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
    )
    with_maps = build_benchmark_eval_row(
        **base,
        single_diagnostics={
            "poisson_residual_norm": 0.0,
            "boundary_operator_proxy_loss": 0.0,
            "poisson_residual_map_l2": 999.0,
            "boundary_operator_residual_map_l2": 999.0,
        },
    )

    assert float(with_maps["surrogate_quality_score"]) == float(scalar["surrogate_quality_score"])


def test_build_benchmark_eval_row_supports_dynamic_target_names():
    pred = np.zeros((1, 1, 4, 4), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="custom_model",
        metrics={"density_main": 0.2, "temp_main": 0.3},
        r2_scores={"density_main": 0.7, "temp_main": 0.6},
        pred_eval={"density_main": pred + 1.0, "temp_main": pred + 2.0},
        true_eval={"density_main": pred + 1.5, "temp_main": pred + 2.5},
        mask_plasma=np.ones((4, 4), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density_main", "temp_main"],
    )
    assert "test_rmse_density_main" in row
    assert "test_r2_density_main_plasma" in row
    assert "test_rmse_temp_main_plasma_deep" in row["_diagnostics"]


def test_build_benchmark_eval_row_adds_group_metrics_from_target_role_schema():
    base = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    true_eval = {
        "electron_density": base,
        "ion_density": base,
        "electron_temperature": base,
        "plasma_potential": base,
    }
    pred_eval = {
        "electron_density": base + 1.0,
        "ion_density": base + 3.0,
        "electron_temperature": base + 2.0,
        "plasma_potential": base + 4.0,
    }

    row = build_benchmark_eval_row(
        model_id="grouped",
        metrics={
            "electron_density": 1.0,
            "ion_density": 3.0,
            "electron_temperature": 2.0,
            "plasma_potential": 4.0,
        },
        r2_scores={
            "electron_density": 0.9,
            "ion_density": 0.7,
            "electron_temperature": 0.5,
            "plasma_potential": 0.1,
        },
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["electron_density", "ion_density"],
        output_vars=[
            "electron_density",
            "ion_density",
            "electron_temperature",
            "plasma_potential",
            "ion_flux",
        ],
        target_role_schema={
            "targets": [
                {"id": "electron_density", "field_family": "density"},
                {"id": "ion_density", "field_family": "density"},
                {"id": "electron_temperature", "field_family": "temperature"},
                {"id": "plasma_potential", "field_family": "electrostatic"},
                {"id": "ion_flux", "field_family": "flux"},
            ],
        },
    )

    assert np.isclose(float(row["test_rmse_group_density"]), 2.0)
    assert np.isclose(float(row["test_r2_group_density"]), 0.8)
    assert np.isclose(
        float(row["test_rmse_group_density_plasma"]),
        np.mean(
            [
                float(row["test_rmse_electron_density_plasma"]),
                float(row["test_rmse_ion_density_plasma"]),
            ]
        ),
    )
    assert float(row["test_rmse_group_temperature"]) == float(row["test_rmse_electron_temperature"])
    assert float(row["test_rmse_group_temperature_plasma"]) == float(row["test_rmse_electron_temperature_plasma"])
    assert np.isclose(float(row["test_r2_group_electrostatic"]), 0.1)
    assert "test_rmse_group_flux" not in row
    assert "test_r2_group_flux_plasma" not in row


def test_build_benchmark_eval_row_skips_group_metrics_without_target_role_schema():
    pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="no_schema",
        metrics={"density": 1.0},
        r2_scores={"density": 0.0},
        pred_eval={"density": pred + 1.0},
        true_eval={"density": pred},
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
        output_vars=["density"],
        target_role_schema={},
    )

    assert "test_rmse_density" in row
    assert not any(str(key).startswith("test_rmse_group_") for key in row)
    assert not any(str(key).startswith("test_r2_group_") for key in row)


def test_build_benchmark_eval_row_adds_region_and_group_region_metrics():
    distance_signed = np.array(
        [
            [0.0, 1.0, 3.0, 12.0],
            [0.0, 1.0, 3.0, 12.0],
            [-1.0, -1.0, -12.0, -12.0],
            [-1.0, -1.0, -12.0, -12.0],
        ],
        dtype=np.float32,
    )
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    true_temperature = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    pred_temperature = true_temperature.copy()
    boundary = np.logical_and(mask_plasma > 0.5, distance_signed <= 2.0)
    deep = np.logical_and(mask_plasma > 0.5, distance_signed > 10.0)
    outside = distance_signed < 0.0
    pred_temperature[:, :, boundary] += 2.0
    pred_temperature[:, :, deep] += 1.0
    pred_temperature[:, :, outside] += 3.0

    row = build_benchmark_eval_row(
        model_id="region_diag",
        metrics={"electron_temperature": 0.0},
        r2_scores={"electron_temperature": 1.0},
        pred_eval={"electron_temperature": pred_temperature},
        true_eval={"electron_temperature": true_temperature},
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["electron_temperature"],
        output_vars=["electron_temperature"],
        target_role_schema={
            "targets": [{"id": "electron_temperature", "field_family": "temperature"}],
        },
        extended_diagnostics_enabled=True,
    )

    assert np.isclose(float(row["test_rmse_electron_temperature_boundary_band"]), 2.0)
    assert np.isclose(float(row["test_rmse_electron_temperature_deep_plasma"]), 1.0)
    assert np.isclose(float(row["test_rmse_electron_temperature_outside"]), 3.0)
    assert np.isclose(float(row["test_rmse_group_temperature_boundary_band"]), 2.0)
    assert np.isclose(float(row["test_rmse_group_temperature_deep_plasma"]), 1.0)
    assert np.isclose(float(row["test_rmse_group_temperature_outside"]), 3.0)
    assert np.isclose(float(row["_diagnostics"]["test_rmse_electron_temperature_boundary_in"]), 2.0)
    assert np.isclose(float(row["_diagnostics"]["test_rmse_electron_temperature_plasma_deep"]), 1.0)


def test_build_benchmark_eval_row_builds_region_metrics_from_mask_and_distance_any():
    mask_plasma = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    distance_any = np.array(
        [
            [0.0, 1.0, 3.0, 12.0],
            [0.0, 1.0, 3.0, 12.0],
            [1.0, 1.0, 12.0, 12.0],
            [1.0, 1.0, 12.0, 12.0],
        ],
        dtype=np.float32,
    )
    true_density = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    pred_density = true_density.copy()
    boundary = np.logical_and(mask_plasma > 0.5, distance_any <= 2.0)
    deep = np.logical_and(mask_plasma > 0.5, distance_any > 10.0)
    outside = mask_plasma <= 0.5
    pred_density[:, :, boundary] += 2.0
    pred_density[:, :, deep] += 1.0
    pred_density[:, :, outside] += 3.0

    row = build_benchmark_eval_row(
        model_id="region_diag_distance_any",
        metrics={"density": 0.0},
        r2_scores={"density": 1.0},
        pred_eval={"density": pred_density},
        true_eval={"density": true_density},
        mask_plasma=None,
        region_mask_plasma=mask_plasma,
        distance_any=distance_any,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
        extended_diagnostics_enabled=True,
    )

    assert np.isclose(float(row["test_rmse_density_boundary_band"]), 2.0)
    assert np.isclose(float(row["test_rmse_density_deep_plasma"]), 1.0)
    assert np.isclose(float(row["test_rmse_density_outside"]), 3.0)


def test_build_benchmark_eval_row_skips_region_columns_without_region_masks():
    pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="no_region",
        metrics={"density": 1.0},
        r2_scores={"density": 0.0},
        pred_eval={"density": pred + 1.0},
        true_eval={"density": pred},
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
    )

    assert "test_rmse_density" in row
    assert "test_rmse_density_boundary_band" not in row
    assert "test_rmse_density_deep_plasma" not in row
    assert "test_rmse_density_outside" not in row


def test_build_benchmark_eval_row_keeps_extended_region_columns_out_of_default_row():
    distance_signed = np.array([[0.0, 12.0], [-1.0, -12.0]], dtype=np.float32)
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    true_density = np.zeros((1, 1, 2, 2), dtype=np.float32)
    pred_density = np.ones((1, 1, 2, 2), dtype=np.float32)

    row = build_benchmark_eval_row(
        model_id="default_region_diag",
        metrics={"density": 1.0},
        r2_scores={"density": 0.0},
        pred_eval={"density": pred_density},
        true_eval={"density": true_density},
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
        output_vars=["density"],
        target_role_schema={"targets": [{"id": "density", "field_family": "density"}]},
    )

    assert "test_rmse_density" in row
    assert "test_rmse_density_boundary_band" not in row
    assert "test_rmse_density_deep_plasma" not in row
    assert "test_rmse_density_outside" not in row
    assert "test_rmse_group_density_boundary_band" not in row
    assert "test_rmse_density_boundary_in" in row["_diagnostics"]


def test_build_benchmark_eval_row_adds_positive_diagnostics_from_schema():
    true = np.zeros((1, 1, 2, 2), dtype=np.float32)
    density_pred = np.array([[[[-1.0, 0.5], [2.0, 3.0]]]], dtype=np.float32)
    temperature_pred = np.ones((1, 1, 2, 2), dtype=np.float32)

    row = build_benchmark_eval_row(
        model_id="positive_diag",
        metrics={"density_main": 0.0, "temperature_main": 0.0},
        r2_scores={"density_main": 1.0, "temperature_main": 1.0},
        pred_eval={"density_main": density_pred, "temperature_main": temperature_pred},
        true_eval={"density_main": true, "temperature_main": true},
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density_main"],
        output_vars=["density_main", "temperature_main"],
        target_role_schema={
            "positive_targets": ["density_main"],
            "targets": [
                {"id": "density_main", "field_family": "density", "positive": True},
                {"id": "temperature_main", "field_family": "temperature", "positive": False},
            ],
        },
        extended_diagnostics_enabled=True,
    )

    assert np.isclose(float(row["positive_violation_rate_density_main"]), 0.25)
    assert np.isclose(float(row["negative_min_density_main"]), -1.0)
    assert np.isclose(float(row["positive_violation_rate_group_density"]), 0.25)
    assert "positive_violation_rate_temperature_main" not in row
    assert "negative_min_temperature_main" not in row


def test_build_benchmark_eval_row_keeps_positive_diagnostics_out_of_default_row():
    pred = np.array([[[[-1.0, 0.5], [2.0, 3.0]]]], dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="default_positive_diag",
        metrics={"density": 0.0},
        r2_scores={"density": 1.0},
        pred_eval={"density": pred},
        true_eval={"density": np.zeros_like(pred)},
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
        output_vars=["density"],
        target_role_schema={
            "positive_targets": ["density"],
            "targets": [{"id": "density", "field_family": "density", "positive": True}],
        },
    )

    assert "positive_violation_rate_density" not in row
    assert "negative_min_density" not in row
    assert "positive_violation_rate_group_density" not in row
    assert np.isclose(float(row["_diagnostics"]["positive_violation_rate_density"]), 0.25)


def test_build_benchmark_eval_row_skips_positive_diagnostics_without_positive_targets():
    pred = np.full((1, 1, 2, 2), -1.0, dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="no_positive",
        metrics={"density": 1.0},
        r2_scores={"density": 0.0},
        pred_eval={"density": pred},
        true_eval={"density": np.zeros_like(pred)},
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
        output_vars=["density"],
        target_role_schema={"targets": [{"id": "density", "field_family": "density"}]},
    )

    assert "test_rmse_density" in row
    assert not any(str(key).startswith("positive_violation_rate_") for key in row)
    assert not any(str(key).startswith("negative_min_") for key in row)


def test_benchmark_eval_row_marks_empty_plasma_mask_invalid() -> None:
    pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="empty_mask",
        metrics={"density": 0.0},
        r2_scores={"density": 1.0},
        pred_eval={"density": pred},
        true_eval={"density": pred},
        mask_plasma=np.zeros((2, 2), dtype=np.float32),
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density"],
    )

    flags = json.loads(str(row["_diagnostics"]["validity_flags"]))
    assert math.isnan(float(row["test_rmse_density_plasma"]))
    assert math.isinf(float(row["surrogate_quality_score"]))
    assert flags["target_metrics_valid"] is False
    assert flags["quality_score_valid"] is False
    assert flags["invalid_reasons"]["density"] == ["empty_plasma_mask", "nonfinite_rmse", "nonfinite_r2"]


def test_surrogate_quality_score_increases_for_worse_prediction():
    mask = np.ones((4, 4), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32), indexing="ij")
    true = (xx * xx + yy * yy + 1.0).reshape(1, 1, 4, 4)
    pred_good = true + 0.1
    pred_bad = true + 1.0

    good = build_benchmark_eval_row(
        model_id="good",
        metrics={"electron_density": 0.0},
        r2_scores={"electron_density": 1.0},
        pred_eval={"electron_density": pred_good},
        true_eval={"electron_density": true},
        mask_plasma=mask,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["electron_density"],
    )
    bad = build_benchmark_eval_row(
        model_id="bad",
        metrics={"electron_density": 0.0},
        r2_scores={"electron_density": 0.0},
        pred_eval={"electron_density": pred_bad},
        true_eval={"electron_density": true},
        mask_plasma=mask,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["electron_density"],
    )

    assert np.isfinite(float(good["surrogate_quality_score"]))
    assert np.isfinite(float(bad["surrogate_quality_score"]))
    assert float(bad["surrogate_quality_score"]) > float(good["surrogate_quality_score"])


def test_surrogate_quality_score_uses_positive_role_sign_penalty():
    mask = np.ones((4, 4), dtype=np.float32)
    true = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) + 1.0
    pred = true.copy()
    pred[:, :, :2, :] = -1.0
    base_kwargs = {
        "model_id": "sign",
        "metrics": {"electron_density": 0.0},
        "r2_scores": {"electron_density": 0.0},
        "pred_eval": {"electron_density": pred},
        "true_eval": {"electron_density": true},
        "mask_plasma": mask,
        "single_diagnostics": {"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        "target_vars_for_score": ["electron_density"],
    }

    no_schema = build_benchmark_eval_row(**base_kwargs)
    with_schema = build_benchmark_eval_row(
        **base_kwargs,
        target_role_schema={
            "positive_targets": ["electron_density"],
            "targets": [{"id": "electron_density", "role": "density_electron", "positive": True}],
        },
    )

    no_schema_diag = no_schema["_diagnostics"]
    with_schema_diag = with_schema["_diagnostics"]
    assert float(no_schema_diag["score_sign_component"]) == 0.0
    assert float(no_schema_diag["positive_target_negative_ratio_penalty"]) == 0.0
    assert float(with_schema_diag["score_sign_component"]) > 0.0
    assert float(with_schema["surrogate_quality_score"]) > float(no_schema["surrogate_quality_score"])


def test_build_benchmark_eval_row_adds_sdf_boundary_deep_contrast():
    distance_signed = np.array(
        [[0.0, 1.0, 3.0, 12.0], [0.5, 1.5, 6.0, 14.0], [-1.0, -3.0, 5.0, 20.0], [0.2, 4.0, 16.0, -5.0]],
        dtype=np.float32,
    )
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    true_temperature = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    pred_temperature = true_temperature.copy()
    boundary = np.logical_and(mask_plasma > 0.5, distance_signed <= 2.0)
    deep = np.logical_and(mask_plasma > 0.5, distance_signed > 10.0)
    pred_temperature[:, :, boundary] += 2.0
    pred_temperature[:, :, deep] += 1.0

    row = build_benchmark_eval_row(
        model_id="fno",
        metrics={"temperature": 0.0},
        r2_scores={"temperature": 1.0},
        pred_eval={"temperature": pred_temperature},
        true_eval={"temperature": true_temperature},
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["temperature"],
    )

    diagnostics = row["_diagnostics"]
    assert np.isclose(float(diagnostics["test_rmse_temperature_boundary_to_deep_ratio"]), 2.0)
    assert np.isclose(float(diagnostics["sdf_boundary_to_deep_rmse_ratio_mean"]), 2.0)
    assert "test_r2_temperature_boundary_minus_deep" in diagnostics
    assert "sdf_boundary_minus_deep_r2_mean" in diagnostics


def test_build_eval_and_viz_payload_helpers():
    true_eval = {
        "density": np.zeros((2, 1, 3, 3), dtype=np.float32),
        "temperature": np.zeros((2, 1, 3, 3), dtype=np.float32),
        "potential": np.zeros((2, 1, 3, 3), dtype=np.float32),
    }
    pred_eval = {
        "density": np.ones((2, 1, 3, 3), dtype=np.float32),
        "temperature": np.ones((2, 1, 3, 3), dtype=np.float32),
        "potential": np.ones((2, 1, 3, 3), dtype=np.float32),
    }
    payload = build_eval_metrics_payload(true_eval=true_eval, pred_eval=pred_eval, eps=None)
    assert "rmse" in payload
    assert "r2" in payload
    payload_masked = build_eval_metrics_payload(
        true_eval=true_eval,
        pred_eval=pred_eval,
        eps=None,
        mask_plasma=np.ones((3, 3), dtype=np.float32),
    )
    assert "rmse_plasma" in payload_masked
    assert "r2_plasma" in payload_masked

    tables = build_viz_tables_payload(
        diag_rows=[{"case_key": "a", "poisson_residual_norm": 1.0}],
        region_rows=[{"case_key": "a", "plasma_mean_density": 1.0}],
    )
    assert "diag_header" in tables and "diag_rows" in tables
    assert "region_header" in tables and "region_rows" in tables
