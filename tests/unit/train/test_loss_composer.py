from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.spatial_regions import build_region_masks
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.train.loss_composer import compose_numpy, compose_supervised_numpy, compose_supervised_torch


def _pred_fields(batch: int = 2, h: int = 6, w: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "log_ne": rng.normal(size=(batch, h, w)).astype(np.float32),
        "Te": np.abs(rng.normal(size=(batch, h, w)).astype(np.float32)),
        "phi": rng.normal(size=(batch, h, w)).astype(np.float32),
    }


def test_compose_numpy_disabled_returns_zero_components():
    pred = _pred_fields()
    total, grad, components = compose_numpy(pred, physics_cfg={"enabled": False})
    assert np.isclose(total, 0.0)
    assert grad.shape == pred["phi"].shape
    assert set(components.keys()) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}
    assert all(np.isclose(v, 0.0) for v in components.values())


def test_compose_numpy_returns_common_component_keys():
    pred = _pred_fields()
    cfg = {
        "enabled": True,
        "lambda_poisson": 0.1,
        "lambda_bc": 0.2,
        "bc_mask": np.ones((6, 6), dtype=np.float32),
        "bc_value": np.zeros((6, 6), dtype=np.float32),
    }
    total, grad, components = compose_numpy(pred, physics_cfg=cfg)
    assert total > 0.0
    assert grad.shape == pred["phi"].shape
    assert set(components.keys()) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}
    assert components["physics"] > 0.0
    assert components["poisson"] > 0.0
    assert components["boundary"] > 0.0


def test_compose_numpy_resolves_physics_symbols_for_noncanonical_targets():
    pred = {
        "density_main": np.ones((1, 6, 6), dtype=np.float32),
        "temp_main": np.ones((1, 6, 6), dtype=np.float32),
        "potential_main": np.zeros((1, 6, 6), dtype=np.float32),
    }
    cfg = {
        "enabled": True,
        "lambda_poisson": 0.1,
        "symbols": {"density": "density_main", "temperature": "temp_main", "potential": "potential_main"},
    }
    total, grad, components = compose_numpy(pred, physics_cfg=cfg)
    assert total >= 0.0
    assert grad.shape == pred["potential_main"].shape
    assert components["physics"] >= 0.0


def test_compose_numpy_disabled_accepts_non_phi_only_targets():
    pred = {"custom_var": np.zeros((1, 4, 4), dtype=np.float32)}
    total, grad, components = compose_numpy(pred, physics_cfg={"enabled": False})
    assert np.isclose(total, 0.0)
    assert grad.shape == pred["custom_var"].shape
    assert set(components.keys()) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}


def test_compose_numpy_resolved_terms_can_disable_poisson():
    pred = _pred_fields()
    cfg = {
        "enabled": True,
        "lambda_poisson": 1.0,
        "lambda_bc": 0.0,
        "bc_mask": np.ones((6, 6), dtype=np.float32),
        "bc_value": np.zeros((6, 6), dtype=np.float32),
    }
    total_on, _, components_on = compose_numpy(pred, physics_cfg=cfg)
    total_off, _, components_off = compose_numpy(
        pred,
        physics_cfg=cfg,
        resolved_terms=[
            {"name": "poisson", "enabled": False, "weight": 0.0, "source_key": "test"},
            {"name": "boundary", "enabled": False, "weight": 0.0, "source_key": "test"},
            {"name": "boundary_operator", "enabled": False, "weight": 0.0, "source_key": "test"},
            {"name": "rho", "enabled": False, "weight": 0.0, "source_key": "test"},
        ],
    )
    assert total_on > total_off
    assert components_on["poisson"] > 0.0
    assert np.isclose(components_off["poisson"], 0.0)


def test_compose_supervised_numpy_with_mask_and_uncertainty():
    rng = np.random.default_rng(3)
    pred = {
        "log_ne": rng.normal(size=(2, 6, 6)).astype(np.float32),
        "log_ni": rng.normal(size=(2, 6, 6)).astype(np.float32),
        "Te": rng.normal(size=(2, 6, 6)).astype(np.float32),
        "phi": rng.normal(size=(2, 6, 6)).astype(np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.zeros((6, 6), dtype=np.float32)
    mask[0:3, :] = 1.0
    dist = np.ones((6, 6), dtype=np.float32) * 5.0
    dist[0:2, :] = 0.5
    with pytest.warns(DeprecationWarning, match="region_weighting"):
        loss, grads, per_var = compose_supervised_numpy(
            pred,
            tgt,
            y_order=["log_ne", "log_ni", "Te", "phi"],
            loss_cfg={
                "supervised": {
                    "type": "huber",
                    "delta": 1.0,
                    "region_weighting": {"enabled": True, "boundary_delta": 2.0, "w_bulk": 1.0, "w_boundary": 3.0},
                },
                "multitask": {"weighting": "uncertainty", "sigma_init": {"phi": -0.5}},
            },
            mask=mask,
            distance_any=dist,
        )
    assert loss > 0.0
    assert set(grads.keys()) == {"log_ne", "log_ni", "Te", "phi"}
    assert set(per_var.keys()) == {"log_ne", "log_ni", "Te", "phi"}
    for name, g in grads.items():
        assert g.shape == pred[name].shape


def test_compose_supervised_numpy_robust_weighting_reduces_outlier_impact():
    pred = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.zeros((1, 2, 2), dtype=np.float32),
        "phi": np.zeros((1, 2, 2), dtype=np.float32),
    }
    tgt = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.array([[[3.0, 3.0], [3.0, 100.0]]], dtype=np.float32),
        "phi": np.array([[[10.0, 10.0], [10.0, -300.0]]], dtype=np.float32),
    }
    mask = np.ones((2, 2), dtype=np.float32)

    base_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "huber", "delta": 1.0}},
        mask=mask,
    )
    robust_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "huber",
                "delta": 1.0,
                "robust_weighting": {"enabled": True, "vars": ["Te", "phi"], "mad_scale": 2.5, "min_weight": 0.1},
            }
        },
        mask=mask,
    )
    assert robust_loss < base_loss


def test_compose_supervised_numpy_sdf_continuous_policy_runs():
    rng = np.random.default_rng(10)
    pred = {
        "log_ne": rng.normal(size=(1, 4, 4)).astype(np.float32),
        "log_ni": rng.normal(size=(1, 4, 4)).astype(np.float32),
        "Te": rng.normal(size=(1, 4, 4)).astype(np.float32),
        "phi": rng.normal(size=(1, 4, 4)).astype(np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2, :] = 1.0
    dist = np.ones((4, 4), dtype=np.float32)
    loss, grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "huber",
                "delta": 1.0,
                "nan_region_policy": "sdf_continuous",
                "sdf_weighting": {
                    "tau_in": 2.0,
                    "tau_out": 1.5,
                    "w_boundary": 3.0,
                    "w_inner": 1.0,
                    "w_chamber_floor": 0.05,
                },
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert np.isfinite(loss)
    assert all(np.isfinite(v) for v in per_var.values())
    assert set(grads.keys()) == {"log_ne", "log_ni", "Te", "phi"}


def test_compose_supervised_numpy_sdf_contract_error_on_non_negative_outside():
    pred = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.zeros((1, 2, 2), dtype=np.float32),
        "phi": np.zeros((1, 2, 2), dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    # Invalid signed build input for chamber side: outside distance is zero only.
    distance_any = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    try:
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["log_ne", "log_ni", "Te", "phi"],
            loss_cfg={
                "supervised": {
                    "type": "huber",
                    "delta": 1.0,
                    "nan_region_policy": "sdf_continuous",
                    "sdf_distance_contract": "error",
                }
            },
            mask=mask,
            distance_any=distance_any,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "signed distance has no negative chamber-side values" in str(exc)


def test_compose_supervised_numpy_normalization_modes_change_loss_scale():
    pred = {
        "log_ne": np.array([[[2.0, 2.0], [2.0, 2.0]], [[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        "log_ni": np.array([[[2.0, 2.0], [2.0, 2.0]], [[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        "Te": np.array([[[2.0, 2.0], [2.0, 2.0]], [[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        "phi": np.array([[[2.0, 2.0], [2.0, 2.0]], [[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    loss_pixel, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "pixel_mean"}},
        mask=mask,
    )
    loss_sample, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean"}},
        mask=mask,
    )
    loss_none, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "none"}},
        mask=mask,
    )
    assert not np.isclose(loss_sample, loss_pixel)
    assert loss_none > loss_pixel


def test_compose_supervised_numpy_sample_mean_with_group_ids_is_casewise():
    pred = {"phi": np.zeros((4, 1, 1), dtype=np.float32)}
    tgt = {"phi": np.array([[[2.0]], [[0.0]], [[0.0]], [[1.0]]], dtype=np.float32)}
    mask = np.ones((4, 1, 1), dtype=np.float32)
    # Three points from case 0, one point from case 1.
    group_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    loss_point_mean, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean"}},
        mask=mask,
    )
    loss_case_mean, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean"}},
        mask=mask,
        group_ids=group_ids,
    )
    assert not np.isclose(loss_point_mean, loss_case_mean)


def test_compose_supervised_numpy_sample_mean_epoch_group_mode_is_rejected():
    pred = {"phi": np.zeros((4, 1, 1), dtype=np.float32)}
    tgt = {"phi": np.array([[[2.0]], [[0.5]], [[0.2]], [[1.0]]], dtype=np.float32)}
    mask = np.ones((4, 1, 1), dtype=np.float32)
    group_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    cfg = {"supervised": {"type": "mse", "normalization": "sample_mean", "sample_mean_group_mode": "epoch"}}
    with pytest.raises(ValueError, match="sample_mean_group_mode"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg=cfg,
            mask=mask,
            group_ids=group_ids,
        )


def test_compose_supervised_numpy_sample_mean_count_denominator_preserves_boundary_weight_for_points():
    pred = {"phi": np.zeros((2, 1, 1), dtype=np.float32)}
    tgt = {"phi": np.ones((2, 1, 1), dtype=np.float32)}
    mask = np.ones((2, 1, 1), dtype=np.float32)
    dist = np.array([[[0.0]], [[10.0]]], dtype=np.float32)  # first is boundary, second is bulk
    cfg_weighted = {
        "supervised": {
            "type": "mse",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "weighted",
            "region_weighting": {"enabled": True, "boundary_delta": 2.0, "w_boundary": 3.0, "w_bulk": 1.0},
        }
    }
    cfg_count = {
        "supervised": {
            "type": "mse",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "count",
            "region_weighting": {"enabled": True, "boundary_delta": 2.0, "w_boundary": 3.0, "w_bulk": 1.0},
        }
    }
    with pytest.warns(DeprecationWarning, match="region_weighting"):
        loss_weighted, _, _ = compose_supervised_numpy(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg=cfg_weighted,
            mask=mask,
            distance_any=dist,
        )
    with pytest.warns(DeprecationWarning, match="region_weighting"):
        loss_count, _, _ = compose_supervised_numpy(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg=cfg_count,
            mask=mask,
            distance_any=dist,
        )
    assert loss_count > loss_weighted


def test_compose_supervised_numpy_coord_objective_is_rejected():
    pred = {"phi": np.array([[[1.0]], [[1.0]]], dtype=np.float32)}
    tgt = {"phi": np.zeros((2, 1, 1), dtype=np.float32)}
    mask = np.array([[[1.0]], [[0.0]]], dtype=np.float32)
    dist = np.ones((2, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="coord_objective is removed"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg={
                "supervised": {
                    "type": "mse",
                    "normalization": "sample_mean",
                    "coord_objective": {"mode": "split", "chamber_aux_weight": 0.5, "chamber_aux_vars": ["phi"]},
                }
            },
            mask=mask,
            distance_any=dist,
        )


def test_compose_supervised_numpy_chamber_weight_by_var_overrides_log_density():
    pred = {
        "log_ne": np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [5, 5, 5, 5], [5, 5, 5, 5]]], dtype=np.float32),
        "log_ni": np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [5, 5, 5, 5], [5, 5, 5, 5]]], dtype=np.float32),
        "Te": np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 3, 3], [3, 3, 3, 3]]], dtype=np.float32),
        "phi": np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 3, 3], [3, 3, 3, 3]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2, :] = 1.0
    dist = np.ones((4, 4), dtype=np.float32)
    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "nan_region_policy": "sdf_continuous",
                "sdf_weighting": {"w_chamber_floor": 0.2},
            }
        },
        mask=mask,
        distance_any=dist,
    )
    override_loss, _, override_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "nan_region_policy": "sdf_continuous",
                "sdf_weighting": {"w_chamber_floor": 0.2},
                "chamber_weight_by_var": {"log_ne": 0.0, "log_ni": 0.0, "Te": 0.2, "phi": 0.2},
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert override_loss < base_loss
    assert override_per_var["log_ne"] < base_per_var["log_ne"]
    assert override_per_var["log_ni"] < base_per_var["log_ni"]


def test_compose_supervised_numpy_global_target_region_plasma_only_ignores_chamber():
    pred = {
        "log_ne": np.ones((1, 4, 4), dtype=np.float32),
        "log_ni": np.ones((1, 4, 4), dtype=np.float32),
        "Te": np.ones((1, 4, 4), dtype=np.float32),
        "phi": np.ones((1, 4, 4), dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2, :] = 1.0
    dist = np.ones((4, 4), dtype=np.float32)
    loss, _, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "global_target_region": "plasma_only",
                "chamber_weight_by_var": {"phi": 0.8, "Te": 0.8, "log_ne": 0.8, "log_ni": 0.8},
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert np.isfinite(loss)
    assert all(v > 0 for v in per_var.values())


def test_compose_supervised_numpy_label_clip_from_scaler_reduces_outlier_target_effect():
    pred = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.zeros((1, 2, 2), dtype=np.float32),
        "phi": np.zeros((1, 2, 2), dtype=np.float32),
    }
    tgt = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.array([[[0.0, 0.0], [0.0, 500.0]]], dtype=np.float32),
        "phi": np.array([[[0.0, 0.0], [0.0, -500.0]]], dtype=np.float32),
    }
    mask = np.ones((2, 2), dtype=np.float32)
    base_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "huber", "delta": 1.0}},
        mask=mask,
    )
    clip_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "huber",
                "delta": 1.0,
                "label_clip_from_scaler": True,
                "robust_clip_stats": {
                    "Te": {"clip_q01": -5.0, "clip_q99": 5.0},
                    "phi": {"clip_q01": -10.0, "clip_q99": 10.0},
                },
            }
        },
        mask=mask,
    )
    assert clip_loss < base_loss


def test_compose_supervised_numpy_chamber_aux_band_only_adds_aux_signal():
    pred = {
        "log_ne": np.array([[[0.0, 0.0], [2.0, 2.0]]], dtype=np.float32),
    }
    tgt = {"log_ne": np.zeros((1, 2, 2), dtype=np.float32)}
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    dist = np.ones((2, 2), dtype=np.float32)

    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne"],
        loss_cfg={"supervised": {"type": "mse", "nan_region_policy": "mask_only"}},
        mask=mask,
        distance_any=dist,
    )
    aux_loss, _, aux_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "nan_region_policy": "mask_only",
                "chamber_aux": {
                    "enabled": True,
                    "scope": "band_only",
                    "band_px": 2.0,
                    "weight_by_var": {"log_ne": 0.1},
                    "grad_clip_abs": 3.0,
                },
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert aux_loss > base_loss
    assert "__chamber_aux__" in aux_per_var
    assert aux_per_var["__chamber_aux__"] > 0.0
    assert aux_per_var["log_ne"] > base_per_var["log_ne"]


def test_compose_supervised_numpy_region_balance_tracks_band_losses():
    pred = {
        "log_ne": np.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=np.float32),
        "log_ni": np.array([[[2.0], [4.0], [6.0], [8.0]]], dtype=np.float32),
        "phi": np.array([[[1.0], [1.0], [1.0], [1.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [12.0], [14.0]]], dtype=np.float32)

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "boundary_in_px": 2.0,
                    "deep_plasma_px": 10.0,
                    "vars": ["log_ne", "log_ni"],
                    "weight_boundary_in": 0.6,
                    "weight_deep_plasma": 0.4,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )
    assert np.isfinite(loss)
    assert np.isfinite(np.sum(grads["phi"]))
    assert "__region_boundary_in__" in per_var
    assert "__region_mid_plasma__" in per_var
    assert "__region_deep_plasma__" in per_var
    assert per_var["__region_balance_applied__"] == 1.0


def test_compose_supervised_numpy_boundary_type_weighting_applies_only_target_vars():
    pred = {
        "Te": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
        "phi": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
        "log_ne": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance = np.array([[[0.5], [0.5], [3.0], [3.0]]], dtype=np.float32)
    bc_dir = np.array([[[1.0], [0.0], [0.0], [0.0]]], dtype=np.float32)

    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi", "log_ne"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
        distance_any=distance,
    )
    w_loss, _, w_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi", "log_ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_type_weighting": {
                    "enabled": True,
                    "vars": ["Te", "phi"],
                    "band_px": 2.0,
                    "weights": {"interface": 1.0, "bc_dir": 2.0, "wafer": 1.0},
                },
            }
        },
        mask=mask,
        distance_any=distance,
        bc_dir_mask=bc_dir,
    )
    assert w_loss > base_loss
    assert w_per_var["Te"] > base_per_var["Te"]
    assert w_per_var["phi"] > base_per_var["phi"]
    assert np.isclose(w_per_var["log_ne"], base_per_var["log_ne"])


def test_compose_supervised_numpy_boundary_profile_weighting_applies_only_boundary_target_vars():
    pred = {
        "Te": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
        "phi": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
        "log_ne": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance_signed = np.array([[[0.1], [1.5], [3.5], [8.0]]], dtype=np.float32)
    distance_any = np.abs(distance_signed)

    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi", "log_ne"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean", "sample_mean_weight_denominator": "count"}},
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    prof_loss, _, prof_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi", "log_ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te", "phi"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "alpha": 0.35,
                    "tau_px": 0.8,
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    assert prof_loss > base_loss
    assert prof_per_var["Te"] > base_per_var["Te"]
    assert prof_per_var["phi"] > base_per_var["phi"]
    assert np.isclose(prof_per_var["log_ne"], base_per_var["log_ne"])


def test_compose_supervised_numpy_boundary_profile_weighting_parts_can_target_mid_and_deep():
    pred = {
        "Te": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
        "phi": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance_signed = np.array([[[3.0], [6.0], [12.0], [14.0]]], dtype=np.float32)
    distance_any = np.abs(distance_signed)

    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean", "sample_mean_weight_denominator": "count"}},
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    prof_loss, _, prof_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "parts": {
                        "plasma_mid": {"alpha": 0.35, "tau_px": 2.0},
                        "plasma_deep": {"alpha": 0.20, "tau_px": 4.0},
                    },
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    assert prof_loss > base_loss
    assert prof_per_var["Te"] > base_per_var["Te"]
    assert np.isclose(prof_per_var["phi"], base_per_var["phi"])


def test_compose_supervised_numpy_boundary_profile_weighting_type_multiplier_scales_boundary_when_present():
    pred = {"Te": np.array([[[2.0], [2.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 2, 1), dtype=np.float32)
    distance_signed = np.array([[[0.2], [0.2]]], dtype=np.float32)
    distance_any = np.abs(distance_signed)
    bc_dir = np.array([[[1.0], [0.0]]], dtype=np.float32)

    prof_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "alpha": 0.35,
                    "tau_px": 0.8,
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    prof_type_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "alpha": 0.35,
                    "tau_px": 0.8,
                    "type_multiplier": {"interface": 1.0, "bc_dir": 2.0, "wafer": 1.0},
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
        bc_dir_mask=bc_dir,
    )
    assert prof_type_loss > prof_loss


def test_compose_supervised_numpy_boundary_profile_weighting_normalize_false_keeps_legacy_behavior():
    pred = {"Te": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance_signed = np.array([[[0.1], [0.8], [1.2], [1.8]]], dtype=np.float32)
    distance_any = np.abs(distance_signed)
    cfg_base = {
        "supervised": {
            "type": "mse",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "count",
            "boundary_profile_weighting": {
                "enabled": True,
                "vars": ["Te"],
                "band_px": 2.0,
                "mode": "exp_decay",
                "alpha": 0.35,
                "tau_px": 0.8,
            },
        }
    }
    cfg_false = {
        "supervised": {
            "type": "mse",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "count",
            "boundary_profile_weighting": {
                "enabled": True,
                "vars": ["Te"],
                "band_px": 2.0,
                "mode": "exp_decay",
                "alpha": 0.35,
                "tau_px": 0.8,
                "normalize_plasma_mean_one": False,
            },
        }
    }
    loss_base, grads_base, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg=cfg_base,
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    loss_false, grads_false, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg=cfg_false,
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    assert np.isclose(loss_base, loss_false, atol=1e-8)
    assert np.allclose(grads_base["Te"], grads_false["Te"], atol=1e-8)


def test_compose_supervised_numpy_boundary_profile_weighting_normalize_true_preserves_constant_loss_scale():
    pred = {"Te": np.array([[[2.0], [2.0], [2.0], [2.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance_signed = np.array([[[0.1], [0.8], [1.2], [1.8]]], dtype=np.float32)
    distance_any = np.abs(distance_signed)

    loss_plain, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={"supervised": {"type": "mse", "normalization": "sample_mean", "sample_mean_weight_denominator": "count"}},
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    loss_profile, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "alpha": 0.35,
                    "tau_px": 0.8,
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    loss_norm, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "normalization": "sample_mean",
                "sample_mean_weight_denominator": "count",
                "boundary_profile_weighting": {
                    "enabled": True,
                    "vars": ["Te"],
                    "band_px": 2.0,
                    "mode": "exp_decay",
                    "alpha": 0.35,
                    "tau_px": 0.8,
                    "normalize_plasma_mean_one": True,
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
        distance_signed=distance_signed,
    )
    assert loss_profile > loss_plain
    assert np.isclose(loss_norm, loss_plain, atol=1e-6)


def test_compose_supervised_numpy_region_balance_mid_band_changes_loss_when_enabled():
    pred = {"Te": np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 5, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [3.0], [8.0], [14.0]]], dtype=np.float32)

    base_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "mode": "additive",
                    "additive_lambda": 0.25,
                    "vars": ["Te"],
                    "boundary_in_px": 2.0,
                    "deep_plasma_px": 10.0,
                    "weight_boundary_in": 0.7,
                    "weight_deep_plasma": 0.3,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )
    mid_loss, _, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "mode": "additive",
                    "additive_lambda": 0.25,
                    "vars": ["Te"],
                    "boundary_in_px": 2.0,
                    "mid_plasma_px": 10.0,
                    "deep_plasma_px": 10.0,
                    "weight_boundary_in": 0.5,
                    "weight_plasma_mid": 0.3,
                    "weight_deep_plasma": 0.2,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )
    assert mid_loss != base_loss
    assert per_var["__region_mid_plasma__"] > 0.0


def test_compose_supervised_numpy_region_balance_schedule_changes_weights_monotonic():
    pred = {"Te": np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 5, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [3.0], [8.0], [14.0]]], dtype=np.float32)
    cfg = {
        "supervised": {
            "type": "mse",
            "region_balance": {
                "enabled": True,
                "mode": "additive",
                "additive_lambda": 0.25,
                "vars": ["Te"],
                "boundary_in_px": 2.0,
                "mid_plasma_px": 10.0,
                "deep_plasma_px": 10.0,
                "weight_boundary_in": 0.95,
                "weight_plasma_mid": 0.0,
                "weight_deep_plasma": 0.05,
                "schedule": {
                    "enabled": True,
                    "warmup_epochs": 1,
                    "ramp_epochs": 2,
                    "boundary_start": 0.95,
                    "boundary_end": 0.70,
                    "mid_start": 0.00,
                    "mid_end": 0.20,
                    "deep_start": 0.05,
                    "deep_end": 0.10,
                },
            },
        }
    }
    loss_e0, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg=cfg,
        mask=mask,
        distance_any=distance,
        epoch_idx=0,
    )
    loss_e2, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg=cfg,
        mask=mask,
        distance_any=distance,
        epoch_idx=2,
    )
    loss_e4, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg=cfg,
        mask=mask,
        distance_any=distance,
        epoch_idx=4,
    )
    assert np.isfinite(loss_e0)
    assert np.isfinite(loss_e2)
    assert np.isfinite(loss_e4)
    assert loss_e0 != loss_e2
    assert loss_e2 != loss_e4


def test_compose_supervised_numpy_region_balance_vars_te_phi_do_not_change_log_losses():
    pred = {
        "log_ne": np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32),
        "log_ni": np.array([[[1.5], [2.5], [3.5], [4.5]]], dtype=np.float32),
        "Te": np.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=np.float32),
        "phi": np.array([[[2.0], [4.0], [6.0], [8.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [12.0], [14.0]]], dtype=np.float32)

    base_loss, _, base_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
        distance_any=distance,
    )
    rb_loss, _, rb_per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "boundary_in_px": 2.0,
                    "deep_plasma_px": 10.0,
                    "vars": ["Te", "phi"],
                    "weight_boundary_in": 0.7,
                    "weight_deep_plasma": 0.3,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )
    assert np.isclose(rb_per_var["log_ne"], base_per_var["log_ne"])
    assert np.isclose(rb_per_var["log_ni"], base_per_var["log_ni"])
    assert rb_loss != base_loss


def test_compose_supervised_numpy_region_balance_additive_keeps_base_and_adds_band_term():
    pred = {
        "Te": np.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=np.float32),
    }
    tgt = {"Te": np.zeros_like(pred["Te"])}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [12.0], [14.0]]], dtype=np.float32)
    base_loss, _, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
        distance_any=distance,
    )
    add_loss, _, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "mode": "additive",
                    "additive_lambda": 0.25,
                    "vars": ["Te"],
                    "boundary_in_px": 2.0,
                    "deep_plasma_px": 10.0,
                    "weight_boundary_in": 0.7,
                    "weight_deep_plasma": 0.3,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )
    assert add_loss > base_loss
    assert per_var["__region_balance_applied__"] == 1.0


def test_compose_supervised_numpy_fixed_weights_by_var_scales_loss_and_grad():
    pred = {
        "log_ne": np.array([[[1.0]]], dtype=np.float32),
        "Te": np.array([[[1.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 1), dtype=np.float32)
    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "Te"],
        loss_cfg={"supervised": {"type": "mse"}, "multitask": {"weighting": "fixed"}},
        mask=mask,
    )
    w_loss, w_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "Te"],
        loss_cfg={
            "supervised": {"type": "mse"},
            "multitask": {"weighting": "fixed", "fixed_weights_by_var": {"Te": 2.0}},
        },
        mask=mask,
    )
    assert w_loss > base_loss
    assert np.allclose(w_grads["log_ne"], base_grads["log_ne"], atol=1e-6)
    assert np.allclose(w_grads["Te"], base_grads["Te"] * 2.0, atol=1e-6)


def test_compose_supervised_numpy_fixed_weights_unknown_var_raises():
    pred = {"Te": np.array([[[1.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    with pytest.raises(ValueError, match="contains unknown vars"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["Te"],
            loss_cfg={
                "supervised": {"type": "mse"},
                "multitask": {"weighting": "fixed", "fixed_weights_by_var": {"phi": 1.2}},
            },
            mask=np.ones((1, 1), dtype=np.float32),
        )


def test_compose_supervised_numpy_clip_and_robust_for_te_phi_are_finite():
    pred = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.zeros((1, 2, 2), dtype=np.float32),
        "phi": np.zeros((1, 2, 2), dtype=np.float32),
    }
    tgt = {
        "log_ne": np.zeros((1, 2, 2), dtype=np.float32),
        "log_ni": np.zeros((1, 2, 2), dtype=np.float32),
        "Te": np.array([[[0.0, 0.0], [1000.0, -1000.0]]], dtype=np.float32),
        "phi": np.array([[[0.0, 0.0], [2000.0, -2000.0]]], dtype=np.float32),
    }
    mask = np.ones((2, 2), dtype=np.float32)
    loss, grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["log_ne", "log_ni", "Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "huber",
                "delta": 1.0,
                "label_clip_from_scaler": True,
                "robust_clip_stats": {
                    "Te": {"clip_q01": -5.0, "clip_q99": 5.0},
                    "phi": {"clip_q01": -10.0, "clip_q99": 10.0},
                },
                "robust_weighting": {"enabled": True, "vars": ["Te", "phi"], "mad_scale": 3.0, "min_weight": 0.2},
            }
        },
        mask=mask,
    )
    assert np.isfinite(loss)
    assert all(np.isfinite(v) for v in per_var.values())
    for g in grads.values():
        assert np.all(np.isfinite(g))


def test_compose_supervised_numpy_delta_by_var_changes_only_target_var_huber_grad():
    pred = {
        "Te": np.array([[[3.0]]], dtype=np.float32),
        "phi": np.array([[[3.0]]], dtype=np.float32),
    }
    tgt = {k: np.zeros_like(v) for k, v in pred.items()}
    mask = np.ones((1, 1), dtype=np.float32)

    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={"supervised": {"type": "huber", "delta": 1.0}},
        mask=mask,
    )
    delta_loss, delta_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={"supervised": {"type": "huber", "delta": 1.0, "delta_by_var": {"Te": 2.0}}},
        mask=mask,
    )
    assert delta_loss > base_loss
    assert np.allclose(delta_grads["phi"], base_grads["phi"], atol=1e-6)
    assert np.mean(np.abs(delta_grads["Te"])) > np.mean(np.abs(base_grads["Te"]))


def test_compose_supervised_numpy_delta_by_var_unknown_var_raises():
    pred = {"Te": np.array([[[1.0]]], dtype=np.float32)}
    tgt = {"Te": np.zeros_like(pred["Te"])}
    with pytest.raises(ValueError, match="delta_by_var contains unknown vars"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["Te"],
            loss_cfg={"supervised": {"type": "huber", "delta": 1.0, "delta_by_var": {"phi": 1.5}}},
            mask=np.ones((1, 1), dtype=np.float32),
        )


def test_compose_supervised_numpy_density_positivity_penalty_applies_only_target_vars():
    pred = {
        "ne": np.array([[[-1.0]]], dtype=np.float32),
        "Te": np.array([[[-1.0]]], dtype=np.float32),
    }
    tgt = {
        "ne": np.zeros((1, 1, 1), dtype=np.float32),
        "Te": np.zeros((1, 1, 1), dtype=np.float32),
    }
    mask = np.ones((1, 1), dtype=np.float32)
    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "Te"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
    )
    pos_loss, pos_grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "Te"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "density_positivity_penalty": {
                    "enabled": True,
                    "vars": ["ne"],
                    "floor": 0.0,
                    "lambda": 0.5,
                },
            }
        },
        mask=mask,
    )
    assert pos_loss > base_loss
    assert np.mean(np.abs(pos_grads["ne"])) > np.mean(np.abs(base_grads["ne"]))
    assert np.allclose(pos_grads["Te"], base_grads["Te"], atol=1e-6)
    assert per_var["__density_positivity_penalty__"] > 0.0


def test_compose_supervised_numpy_density_relative_weighting_applies_only_target_vars():
    pred = {
        "ne": np.array([[[2.0]]], dtype=np.float32),
        "phi": np.array([[[2.0]]], dtype=np.float32),
    }
    tgt = {
        "ne": np.array([[[1.0]]], dtype=np.float32),
        "phi": np.array([[[1.0]]], dtype=np.float32),
    }
    mask = np.ones((1, 1), dtype=np.float32)
    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "phi"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
    )
    rel_loss, rel_grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "density_relative_weighting": {
                    "enabled": True,
                    "vars": ["ne"],
                    "lambda": 0.5,
                    "eps": 1.0e-2,
                },
            }
        },
        mask=mask,
    )
    assert rel_loss > base_loss
    assert np.mean(np.abs(rel_grads["ne"])) > np.mean(np.abs(base_grads["ne"]))
    assert np.allclose(rel_grads["phi"], base_grads["phi"], atol=1e-6)
    assert per_var["__density_relative_weighting__"] > 0.0


def test_compose_supervised_numpy_density_relative_weighting_unknown_var_raises():
    pred = {"ne": np.array([[[1.0]]], dtype=np.float32)}
    tgt = {"ne": np.zeros((1, 1, 1), dtype=np.float32)}
    with pytest.raises(ValueError, match="density_relative_weighting.vars contains unknown vars"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["ne"],
            loss_cfg={
                "supervised": {
                    "type": "mse",
                    "density_relative_weighting": {
                        "enabled": True,
                        "vars": ["ni"],
                        "lambda": 0.2,
                        "eps": 1.0e-6,
                    },
                }
            },
            mask=np.ones((1, 1), dtype=np.float32),
        )


def test_compose_supervised_numpy_density_positivity_uses_physical_affine_when_provided():
    pred = {"ne": np.array([[[-0.2]]], dtype=np.float32)}
    tgt = {"ne": np.array([[[-0.2]]], dtype=np.float32)}
    mask = np.ones((1, 1), dtype=np.float32)
    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
    )
    pos_loss, pos_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "density_positivity_penalty": {
                    "enabled": True,
                    "vars": ["ne"],
                    "floor": 0.0,
                    "lambda": 1.0,
                    "affine_by_var": {"ne": {"mean": 10.0, "std": 2.0}},
                },
            }
        },
        mask=mask,
    )
    # scaled pred is negative, but physical pred is positive (9.6), so penalty must not be added.
    assert np.isclose(pos_loss, base_loss)
    assert np.allclose(pos_grads["ne"], base_grads["ne"], atol=1e-6)


def test_compose_supervised_numpy_density_relative_weighting_uses_physical_eps_domain():
    pred = {"ne": np.array([[[2.0]]], dtype=np.float32)}
    tgt = {"ne": np.array([[[1.0]]], dtype=np.float32)}
    mask = np.ones((1, 1), dtype=np.float32)
    rel_loss, rel_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "density_relative_weighting": {
                    "enabled": True,
                    "vars": ["ne"],
                    "lambda": 1.0,
                    "eps": 1.0e-3,
                    "affine_by_var": {"ne": {"mean": 100.0, "std": 10.0}},
                },
            }
        },
        mask=mask,
    )
    # relative term in physical space should remain finite and contribute positive loss/grad.
    assert rel_loss > 0.0
    assert float(np.mean(np.abs(rel_grads["ne"]))) > 0.0


def test_compose_supervised_numpy_density_positivity_large_std_is_stable():
    pred = {"ne": np.array([[[-1.0]]], dtype=np.float32)}
    tgt = {"ne": np.zeros((1, 1, 1), dtype=np.float32)}
    mask = np.ones((1, 1), dtype=np.float32)
    loss, grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "density_positivity_penalty": {
                    "enabled": True,
                    "vars": ["ne"],
                    "floor": 0.0,
                    "lambda": 1.0,
                    "affine_by_var": {"ne": {"mean": 0.0, "std": 1.0e15}},
                },
            }
        },
        mask=mask,
    )
    assert np.isfinite(loss)
    assert np.isfinite(per_var["__density_positivity_penalty__"])
    assert np.all(np.isfinite(grads["ne"]))
    assert float(np.mean(np.abs(grads["ne"]))) < 10.0


def test_compose_supervised_numpy_spatial_consistency_grad_huber_adds_loss_and_grad():
    pred = {
        "ne": np.array([[[0.0, 2.0], [0.0, 2.0]]], dtype=np.float32),
        "ni": np.array([[[0.0, 2.0], [0.0, 2.0]]], dtype=np.float32),
    }
    tgt = {
        "ne": np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
        "ni": np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
    }
    mask = np.ones((1, 2, 2), dtype=np.float32)
    dist = np.ones((1, 2, 2), dtype=np.float32)
    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "ni"],
        loss_cfg={"supervised": {"type": "mse", "mask": "plasma_only"}},
        mask=mask,
        distance_any=dist,
    )
    sc_loss, sc_grads, per_var = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "ni"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "mask": "plasma_only",
                "spatial_consistency": {
                    "enabled": True,
                    "vars": ["ne", "ni"],
                    "mode": "grad_huber",
                    "lambda": 0.1,
                    "delta": 1.0,
                    "apply_region": "plasma_only",
                },
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert sc_loss > base_loss
    assert "__spatial_consistency__" in per_var
    assert per_var["__spatial_consistency__"] > 0.0
    assert float(np.mean(np.abs(sc_grads["ne"] - base_grads["ne"]))) > 0.0


def test_compose_supervised_numpy_spatial_consistency_multiscale_runs_and_changes_grad():
    pred = {
        "ne": np.array([[[0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]]], dtype=np.float32),
        "ni": np.array([[[0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]]], dtype=np.float32),
    }
    tgt = {
        "ne": np.array([[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=np.float32),
        "ni": np.array([[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=np.float32),
    }
    mask = np.ones((1, 4, 4), dtype=np.float32)
    dist = np.ones((1, 4, 4), dtype=np.float32)
    loss_single, grads_single, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "ni"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "mask": "plasma_only",
                "spatial_consistency": {
                    "enabled": True,
                    "vars": ["ne", "ni"],
                    "mode": "grad_huber",
                    "lambda": 0.1,
                    "delta": 1.0,
                    "apply_region": "plasma_only",
                },
            }
        },
        mask=mask,
        distance_any=dist,
    )
    loss_multi, grads_multi, _ = compose_supervised_numpy(
        pred,
        tgt,
        y_order=["ne", "ni"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "mask": "plasma_only",
                "spatial_consistency": {
                    "enabled": True,
                    "vars": ["ne", "ni"],
                    "mode": "grad_huber",
                    "lambda": 0.1,
                    "delta": 1.0,
                    "apply_region": "plasma_only",
                    "multiscale": {
                        "enabled": True,
                        "scales": [1, 2],
                        "scale_weights": [1.0, 0.5],
                    },
                },
            }
        },
        mask=mask,
        distance_any=dist,
    )
    assert np.isfinite(loss_single)
    assert np.isfinite(loss_multi)
    assert loss_multi >= loss_single
    assert np.all(np.isfinite(grads_multi["ne"]))


def test_build_region_masks_signed_quantile_monotonic():
    mask = np.ones((1, 4, 4), dtype=np.float32)
    d_signed = np.array(
        [[[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]],
        dtype=np.float32,
    )
    regions = build_region_masks(
        mask_plasma=mask,
        distance_any=None,
        distance_signed=d_signed,
        mode="signed_quantile",
        boundary_q=0.2,
        deep_q=0.8,
    )
    assert np.sum(regions["boundary_in"]) > 0
    assert np.sum(regions["plasma_deep"]) > 0
    boundary_thr = float(np.asarray(regions["boundary_threshold"]).reshape(-1)[0])
    deep_thr = float(np.asarray(regions["deep_threshold"]).reshape(-1)[0])
    assert boundary_thr <= deep_thr


def test_compose_supervised_torch_region_balance_changes_loss():
    require_torch_runtime()

    pred = {
        "Te": np.array([[[2.0, 2.0], [0.5, 0.5]]], dtype=np.float32),
        "phi": np.array([[[1.0, 1.0], [0.2, 0.2]]], dtype=np.float32),
    }
    tgt = np.zeros((1, 2, 2, 2), dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=np.float32)
    distance_any = np.array([[[1.0, 1.0], [8.0, 8.0]]], dtype=np.float32)

    loss_base, _ = compose_supervised_torch(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
        distance_any=distance_any,
    )
    loss_rb, _ = compose_supervised_torch(
        pred,
        tgt,
        y_order=["Te", "phi"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "vars": ["Te", "phi"],
                    "mode": "additive",
                    "additive_lambda": 0.5,
                    "boundary_in_px": 2.0,
                    "mid_plasma_px": 4.0,
                    "deep_plasma_px": 6.0,
                    "weight_boundary_in": 0.1,
                    "weight_plasma_mid": 0.2,
                    "weight_deep_plasma": 0.7,
                },
            }
        },
        mask=mask,
        distance_any=distance_any,
    )
    assert float(loss_rb.detach().cpu().item()) != pytest.approx(float(loss_base.detach().cpu().item()))


def test_compose_supervised_torch_region_weighting_alias_warns():
    require_torch_runtime()

    pred = {"phi": np.ones((1, 2, 2), dtype=np.float32)}
    tgt = np.zeros((1, 1, 2, 2), dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=np.float32)
    distance_any = np.ones((1, 2, 2), dtype=np.float32)

    with pytest.warns(DeprecationWarning, match="region_weighting"):
        compose_supervised_torch(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg={
                "supervised": {
                    "type": "mse",
                    "region_weighting": {"enabled": True, "boundary_delta": 2.0, "w_bulk": 1.0, "w_boundary": 3.0},
                }
            },
            mask=mask,
            distance_any=distance_any,
        )


def test_compose_supervised_region_contract_conflict_rejects():
    pred = {"phi": np.ones((1, 2, 2), dtype=np.float32)}
    tgt = {"phi": np.zeros((1, 2, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="region_balance and supervised.region_weighting"):
        compose_supervised_numpy(
            pred,
            tgt,
            y_order=["phi"],
            loss_cfg={
                "supervised": {
                    "type": "mse",
                    "region_balance": {"enabled": True},
                    "region_weighting": {"enabled": True},
                }
            },
        )
