from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.eval.metrics_builder import (
    build_benchmark_eval_row,
    build_eval_metrics_payload,
    build_region_metrics,
    build_single_case_physics_metrics,
    build_viz_tables_payload,
)


def test_build_single_case_physics_metrics_reads_maps(tmp_path: Path):
    case_dir = tmp_path / "abc123"
    case_dir.mkdir(parents=True)
    with (case_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump({"poisson_residual_norm": 1.5, "bc_phi_mae": 0.25}, f)
    with (case_dir / "qoi.json").open("w", encoding="utf-8") as f:
        json.dump({"uniformity": 0.12}, f)
    np.savez(
        case_dir / "diagnostics_maps.npz",
        poisson_residual_map=np.ones((4, 4), dtype=np.float32),
        boundary_operator_residual_map=np.full((4, 4), 2.0, dtype=np.float32),
    )
    row = build_single_case_physics_metrics(case_dir)
    assert row["case_key"] == "abc123"
    assert float(row["poisson_residual_norm"]) == 1.5
    assert float(row["poisson_residual_map_l2"]) == 1.0
    assert float(row["boundary_residual_map_l2"]) == 2.0


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

    pred = np.zeros((2, 1, 4, 4), dtype=np.float32)
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
        single_qoi={"uniformity": 0.4, "boundary_gamma_uniformity": 0.5},
        single_diagnostics={"poisson_residual_norm": 0.6, "boundary_operator_proxy_loss": 0.7},
        target_vars_for_score=["electron_density", "ion_density"],
    )
    assert row["model_id"] == "global_mlp"
    assert float(row["test_r2_potential"]) == 0.73
    assert "test_rmse_potential_plasma" in row
    assert "test_r2_potential_plasma" in row
    assert "score_total" not in row
    assert "test_rmse_electron_density_boundary_in" in row
    assert "test_rmse_ion_density_boundary_in" in row
    assert "test_r2_electron_density_plasma_deep" in row
    assert "test_r2_ion_density_plasma_deep" in row
    assert "test_neg_ratio_electron_density_plasma" in row
    assert "test_neg_ratio_ion_density_plasma" in row
    assert "continuity_grad_ratio_all_plasma" in row
    assert "continuity_lap_ratio_all_plasma" in row
    assert "surrogate_quality_score" in row
    assert np.isfinite(float(row["surrogate_quality_score"]))
    assert "score_nrmse_component" in row
    assert "test_r2_potential" in row


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
        single_qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["small", "large"],
    )

    assert np.isclose(float(row["mean_nrmse_plasma_by_target"]), 1.0)
    assert np.isclose(float(row["score_nrmse_component"]), 1.0)


def test_build_benchmark_eval_row_supports_dynamic_target_names():
    pred = np.zeros((1, 1, 4, 4), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="custom_model",
        metrics={"density_main": 0.2, "temp_main": 0.3},
        r2_scores={"density_main": 0.7, "temp_main": 0.6},
        pred_eval={"density_main": pred + 1.0, "temp_main": pred + 2.0},
        true_eval={"density_main": pred + 1.5, "temp_main": pred + 2.5},
        mask_plasma=np.ones((4, 4), dtype=np.float32),
        single_qoi={"uniformity": 0.1, "boundary_gamma_uniformity": 0.0},
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["density_main", "temp_main"],
    )
    assert "test_rmse_density_main" in row
    assert "test_r2_density_main_plasma" in row
    assert "test_rmse_temp_main_plasma_deep" in row


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
        single_qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
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
        single_qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
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
        "single_qoi": {"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
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

    assert float(no_schema["score_sign_component"]) == 0.0
    assert float(no_schema["positive_target_negative_ratio_penalty"]) == 0.0
    assert float(with_schema["score_sign_component"]) > 0.0
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
        single_qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        target_vars_for_score=["temperature"],
    )

    assert np.isclose(float(row["test_rmse_temperature_boundary_to_deep_ratio"]), 2.0)
    assert np.isclose(float(row["sdf_boundary_to_deep_rmse_ratio_mean"]), 2.0)
    assert "test_r2_temperature_boundary_minus_deep" in row
    assert "sdf_boundary_minus_deep_r2_mean" in row


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
