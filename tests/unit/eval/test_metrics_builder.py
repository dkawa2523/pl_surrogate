from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.eval.metrics_builder import (
    build_benchmark_eval_row,
    build_eval_metrics_payload,
    build_region_metrics,
    build_single_case_physics_metrics,
    build_spatial_error_by_case_rows,
    build_spatial_error_summary_rows,
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
    ne = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    masks = {
        "plasma": np.array([[1, 1], [1, 1]], dtype=bool),
        "bulk": np.array([[1, 0], [0, 1]], dtype=bool),
        "boundary": np.array([[0, 1], [1, 0]], dtype=bool),
    }
    region = build_region_metrics(
        case_key="k1",
        density=ne,
        mask_plasma=masks["plasma"],
        mask_bulk=masks["bulk"],
        mask_boundary=masks["boundary"],
    )
    assert region["case_key"] == "k1"
    assert float(region["bulk_mean_density"]) == 2.5

    pred_phi = np.zeros((2, 1, 4, 4), dtype=np.float32)
    row = build_benchmark_eval_row(
        model_id="global_mlp",
        metrics={"ne": 0.1, "ni": 0.11, "Te": 0.2, "phi": 0.3},
        r2_scores={"ne": 0.91, "ni": 0.89, "Te": 0.82, "phi": 0.73},
        pred_eval={"phi": pred_phi, "ne": pred_phi + 1.0, "ni": pred_phi + 1.25, "Te": pred_phi + 2.0},
        true_eval={"phi": pred_phi + 0.5, "ne": pred_phi + 1.5, "ni": pred_phi + 1.75, "Te": pred_phi + 2.5},
        mask_plasma=np.ones((4, 4), dtype=np.float32),
        single_qoi={"uniformity": 0.4, "boundary_gamma_uniformity": 0.5},
        single_diagnostics={"poisson_residual_norm": 0.6, "boundary_operator_proxy_loss": 0.7},
        opt_best_uniformity=0.8,
        aggregate_cfg={
            "enabled": True,
            "rmse_weight": 0.5,
            "r2_weight": 0.4,
            "boundary_penalty_weight": 0.1,
            "use_plasma_metrics": True,
        },
        target_vars_for_score=["ne", "ni"],
    )
    assert row["model_id"] == "global_mlp"
    assert float(row["qoi_uniformity"]) == 0.4
    assert float(row["opt_best_uniformity"]) == 0.8
    assert float(row["test_r2_phi"]) == 0.73
    assert "test_rmse_phi_plasma" in row
    assert "test_r2_phi_plasma" in row
    assert "score_total" in row
    assert float(row["score_total"]) != 0.0
    assert "test_r2_logpair_plasma_mean" not in row
    assert "score_total_logpair" not in row
    assert "test_rmse_ne_boundary_in" in row
    assert "test_rmse_ni_boundary_in" in row
    assert "test_r2_ne_plasma_deep" in row
    assert "test_r2_ni_plasma_deep" in row
    assert "test_neg_ratio_ne_plasma" in row
    assert "test_neg_ratio_ni_plasma" in row
    assert "continuity_grad_ratio_all_plasma" in row
    assert "continuity_lap_ratio_all_plasma" in row


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
        opt_best_uniformity=0.0,
        target_vars_for_score=["density_main", "temp_main"],
        aggregate_cfg={"enabled": True, "use_plasma_metrics": True},
    )
    assert "test_rmse_density_main" in row
    assert "test_r2_density_main_plasma" in row
    assert "test_rmse_temp_main_plasma_deep" in row


def test_build_benchmark_eval_row_adds_sdf_boundary_deep_contrast():
    distance_signed = np.array(
        [[0.0, 1.0, 3.0, 12.0], [0.5, 1.5, 6.0, 14.0], [-1.0, -3.0, 5.0, 20.0], [0.2, 4.0, 16.0, -5.0]],
        dtype=np.float32,
    )
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    true_te = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    pred_te = true_te.copy()
    boundary = np.logical_and(mask_plasma > 0.5, distance_signed <= 2.0)
    deep = np.logical_and(mask_plasma > 0.5, distance_signed > 10.0)
    pred_te[:, :, boundary] += 2.0
    pred_te[:, :, deep] += 1.0

    row = build_benchmark_eval_row(
        model_id="fno",
        metrics={"Te": 0.0},
        r2_scores={"Te": 1.0},
        pred_eval={"Te": pred_te},
        true_eval={"Te": true_te},
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        single_qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
        single_diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
        opt_best_uniformity=0.0,
        target_vars_for_score=["Te"],
    )

    assert np.isclose(float(row["test_rmse_Te_boundary_to_deep_ratio"]), 2.0)
    assert np.isclose(float(row["sdf_boundary_to_deep_rmse_ratio_mean"]), 2.0)
    assert "test_r2_Te_boundary_minus_deep" in row
    assert "sdf_boundary_minus_deep_r2_mean" in row


def test_build_eval_and_viz_payload_helpers():
    true_eval = {
        "ne": np.zeros((2, 1, 3, 3), dtype=np.float32),
        "Te": np.zeros((2, 1, 3, 3), dtype=np.float32),
        "phi": np.zeros((2, 1, 3, 3), dtype=np.float32),
    }
    pred_eval = {
        "ne": np.ones((2, 1, 3, 3), dtype=np.float32),
        "Te": np.ones((2, 1, 3, 3), dtype=np.float32),
        "phi": np.ones((2, 1, 3, 3), dtype=np.float32),
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
    assert "phi_poisson_residual" in payload["rmse"]
    assert "phi_poisson_residual_norm" in payload["rmse"]

    tables = build_viz_tables_payload(
        diag_rows=[{"case_key": "a", "poisson_residual_norm": 1.0}],
        region_rows=[{"case_key": "a", "plasma_mean_density": 1.0}],
    )
    assert "diag_header" in tables and "diag_rows" in tables
    assert "region_header" in tables and "region_rows" in tables


def test_build_spatial_error_summary_rows_has_regions():
    true_eval = {
        "Te": np.ones((2, 1, 4, 4), dtype=np.float32),
        "phi": np.ones((2, 1, 4, 4), dtype=np.float32) * 2.0,
    }
    pred_eval = {
        "Te": np.zeros((2, 1, 4, 4), dtype=np.float32),
        "phi": np.zeros((2, 1, 4, 4), dtype=np.float32),
    }
    distance_signed = np.array(
        [[-3.0, -1.0, 0.0, 1.0], [2.0, 5.0, 11.0, 20.0], [-12.0, -1.5, 0.5, 9.0], [3.0, 12.0, 25.0, -20.0]],
        dtype=np.float32,
    )
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    rows = build_spatial_error_summary_rows(
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        vars_for_summary=["Te", "phi"],
    )
    assert len(rows) > 0
    regions = {(str(r["var"]), str(r["region"])) for r in rows}
    assert ("Te", "boundary_in") in regions
    assert ("phi", "plasma_deep") in regions


def test_build_spatial_error_summary_rows_signed_quantile_mode_runs():
    true_eval = {"Te": np.ones((1, 1, 4, 4), dtype=np.float32)}
    pred_eval = {"Te": np.zeros((1, 1, 4, 4), dtype=np.float32)}
    distance_signed = np.array(
        [[0.0, 0.2, 0.4, 0.6], [0.8, 1.0, 1.2, 1.4], [1.6, 1.8, 2.0, 2.2], [2.4, 2.6, 2.8, 3.0]],
        dtype=np.float32,
    )
    mask_plasma = np.ones((4, 4), dtype=np.float32)
    rows = build_spatial_error_summary_rows(
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        vars_for_summary=["Te"],
        region_band_cfg={"mode": "signed_quantile", "boundary_q": 0.2, "deep_q": 0.7},
    )
    assert any(str(r["region"]) == "boundary_in" for r in rows)
    assert any(str(r["region"]) == "plasma_deep" for r in rows)


def test_build_spatial_error_by_case_rows_has_case_dimension():
    true_eval = {
        "Te": np.ones((2, 1, 4, 4), dtype=np.float32),
        "phi": np.ones((2, 1, 4, 4), dtype=np.float32) * 2.0,
    }
    pred_eval = {
        "Te": np.zeros((2, 1, 4, 4), dtype=np.float32),
        "phi": np.zeros((2, 1, 4, 4), dtype=np.float32),
    }
    distance_signed = np.array(
        [[-3.0, -1.0, 0.0, 1.0], [2.0, 5.0, 11.0, 20.0], [-12.0, -1.5, 0.5, 9.0], [3.0, 12.0, 25.0, -20.0]],
        dtype=np.float32,
    )
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    rows = build_spatial_error_by_case_rows(
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        vars_for_summary=["Te", "phi"],
        case_ids=["c0", "c1"],
    )
    assert len(rows) > 0
    keys = {(str(r["case_id"]), str(r["var"]), str(r["region"])) for r in rows}
    assert ("c0", "Te", "boundary_in") in keys
    assert ("c1", "phi", "plasma_deep") in keys


def test_build_spatial_error_rows_boundary_type_breakdown_has_bc_dir():
    true_eval = {"Te": np.ones((1, 1, 3, 3), dtype=np.float32)}
    pred_eval = {"Te": np.zeros((1, 1, 3, 3), dtype=np.float32)}
    distance_signed = np.array([[0.0, 0.5, 3.0], [1.0, 8.0, 12.0], [-1.0, -3.0, 20.0]], dtype=np.float32)
    distance_any = np.abs(distance_signed)
    mask_plasma = (distance_signed >= 0.0).astype(np.float32)
    bc_dir_mask = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    rows = build_spatial_error_summary_rows(
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        distance_any=distance_any,
        bc_dir_mask=bc_dir_mask,
        boundary_type_breakdown=True,
        vars_for_summary=["Te"],
    )
    boundary_types = {(str(r["region"]), str(r.get("boundary_type", "na"))) for r in rows}
    assert ("boundary_in", "bc_dir") in boundary_types
    assert ("boundary_in", "interface") in boundary_types


def test_build_spatial_error_summary_rows_uses_nan_when_region_has_no_points():
    true_eval = {"Te": np.ones((1, 1, 2, 2), dtype=np.float32)}
    pred_eval = {"Te": np.zeros((1, 1, 2, 2), dtype=np.float32)}
    # No deep-plasma pixels in this tiny map.
    distance_signed = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    mask_plasma = np.ones((2, 2), dtype=np.float32)
    rows = build_spatial_error_summary_rows(
        pred_eval=pred_eval,
        true_eval=true_eval,
        mask_plasma=mask_plasma,
        distance_signed=distance_signed,
        vars_for_summary=["Te"],
    )
    deep_rows = [r for r in rows if str(r["region"]) == "plasma_deep" and str(r["var"]) == "Te"]
    assert deep_rows
    assert np.isnan(float(deep_rows[0]["rmse"]))
    assert np.isnan(float(deep_rows[0]["r2"]))
