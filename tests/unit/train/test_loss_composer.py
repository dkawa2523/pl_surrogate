from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.train.loss_composer import compose_numpy, compose_supervised_numpy
from plasma_surrogate.train.loss_protocols import resolve_loss_protocol


def _field(value: float, shape: tuple[int, ...] = (1, 4, 4)) -> np.ndarray:
    return np.full(shape, float(value), dtype=np.float32)


def test_compose_numpy_disabled_returns_zero_components():
    pred = {"custom_var": _field(0.0)}
    total, grad, components = compose_numpy(pred, physics_cfg={"enabled": False})

    assert total == pytest.approx(0.0)
    assert grad.shape == pred["custom_var"].shape
    assert set(components) == {"data", "physics", "poisson", "boundary", "boundary_operator", "rho"}
    assert all(v == pytest.approx(0.0) for v in components.values())


def test_compose_numpy_resolves_explicit_physics_symbols_for_dynamic_targets():
    rng = np.random.default_rng(4)
    pred = {
        "density_main": np.ones((1, 6, 6), dtype=np.float32),
        "temp_main": np.ones((1, 6, 6), dtype=np.float32),
        "potential_main": rng.normal(size=(1, 6, 6)).astype(np.float32),
    }
    cfg = {
        "enabled": True,
        "terms": {"poisson": {"weight": 0.1}, "boundary": {"weight": 0.2}},
        "symbols": {"density": "density_main", "temperature": "temp_main", "potential": "potential_main"},
        "bc_mask": np.ones((6, 6), dtype=np.float32),
        "bc_value": np.zeros((6, 6), dtype=np.float32),
    }

    total, grad, components = compose_numpy(pred, physics_cfg=cfg)

    assert total > 0.0
    assert grad.shape == pred["potential_main"].shape
    assert components["physics"] > 0.0
    assert components["poisson"] >= 0.0
    assert components["boundary"] > 0.0


def test_compose_numpy_resolves_unique_physics_roles_for_dynamic_targets():
    pred = {
        "electron_density": _field(1.0),
        "electron_temperature": _field(1.0),
        "plasma_potential": _field(0.0),
    }
    cfg = {
        "enabled": True,
        "terms": {"poisson": {"weight": 0.1}},
        "target_role_schema": {
            "targets": [
                {"id": "electron_density", "role": "density_electron", "field_family": "density"},
                {"id": "electron_temperature", "role": "temperature_electron", "field_family": "temperature"},
                {"id": "plasma_potential", "role": "potential", "field_family": "electrostatic"},
            ]
        },
    }

    total, grad, components = compose_numpy(pred, physics_cfg=cfg)

    assert total >= 0.0
    assert grad.shape == pred["plasma_potential"].shape
    assert components["physics"] >= 0.0


def test_compose_numpy_ambiguous_density_role_requires_symbol():
    pred = {
        "electron_density": _field(1.0),
        "ion_density": _field(1.0),
        "electron_temperature": _field(1.0),
        "plasma_potential": _field(0.0),
    }
    cfg = {
        "enabled": True,
        "target_role_schema": {
            "targets": [
                {"id": "electron_density", "field_family": "density"},
                {"id": "ion_density", "field_family": "density"},
                {"id": "electron_temperature", "role": "temperature_electron"},
                {"id": "plasma_potential", "role": "potential"},
            ]
        },
    }

    with pytest.raises(ValueError, match="ambiguous"):
        compose_numpy(pred, physics_cfg=cfg)


def test_compose_numpy_resolved_terms_can_disable_physics_terms():
    rng = np.random.default_rng(7)
    pred = {
        "density_main": np.ones((1, 6, 6), dtype=np.float32),
        "temp_main": np.ones((1, 6, 6), dtype=np.float32),
        "potential_main": rng.normal(size=(1, 6, 6)).astype(np.float32),
    }
    cfg = {
        "enabled": True,
        "terms": {"poisson": {"weight": 1.0}, "boundary": {"weight": 0.2}},
        "symbols": {"density": "density_main", "temperature": "temp_main", "potential": "potential_main"},
        "bc_mask": np.ones((6, 6), dtype=np.float32),
        "bc_value": np.zeros((6, 6), dtype=np.float32),
    }

    total_on, _, components_on = compose_numpy(pred, physics_cfg=cfg)
    total_off, _, components_off = compose_numpy(
        pred,
        physics_cfg=cfg,
        resolved_terms=[
            {"name": "poisson", "enabled": False, "weight": 0.0},
            {"name": "boundary", "enabled": False, "weight": 0.0},
            {"name": "boundary_operator", "enabled": False, "weight": 0.0},
            {"name": "rho", "enabled": False, "weight": 0.0},
        ],
    )

    assert total_on > total_off
    assert components_on["physics"] > 0.0
    assert components_off["physics"] == pytest.approx(0.0)


def test_compose_supervised_numpy_masked_huber_returns_dynamic_target_grads():
    pred = {
        "electron_density": _field(2.0, (2, 3, 3)),
        "plasma_potential": _field(-1.0, (2, 3, 3)),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}
    mask = np.ones((3, 3), dtype=np.float32)

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "plasma_potential"],
        loss_cfg={"supervised": {"type": "huber", "delta": 1.0}},
        mask=mask,
    )

    assert loss > 0.0
    assert set(grads) == set(pred)
    assert set(per_var) == set(pred)
    assert all(grads[name].shape == pred[name].shape for name in pred)


def test_compose_supervised_numpy_region_balance_tracks_region_components():
    pred = {
        "electron_density": np.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=np.float32),
        "plasma_potential": np.array([[[1.0], [1.0], [1.0], [1.0]]], dtype=np.float32),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}
    mask = np.ones((1, 4, 1), dtype=np.float32)
    distance = np.array([[[0.5], [1.5], [12.0], [14.0]]], dtype=np.float32)

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "plasma_potential"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "region_balance": {
                    "enabled": True,
                    "boundary_in_px": 2.0,
                    "deep_plasma_px": 10.0,
                    "vars": ["electron_density"],
                    "weight_boundary_in": 0.6,
                    "weight_deep_plasma": 0.4,
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )

    assert np.isfinite(loss)
    assert np.isfinite(np.sum(grads["plasma_potential"]))
    assert per_var["__region_balance_applied__"] == pytest.approx(1.0)
    assert "__region_boundary_in__" in per_var
    assert "__region_deep_plasma__" in per_var


def test_compose_supervised_numpy_positive_penalty_accepts_arbitrary_target_name():
    pred = {
        "electron_density": np.array([[[-1.0]]], dtype=np.float32),
        "plasma_potential": np.array([[[-1.0]]], dtype=np.float32),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}

    base_loss, base_grads, _ = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "plasma_potential"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=np.ones((1, 1), dtype=np.float32),
    )
    pos_loss, pos_grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "plasma_potential"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "positive_penalty": {
                    "enabled": True,
                    "vars": ["electron_density"],
                    "floor": 0.0,
                    "lambda": 0.5,
                },
            }
        },
        mask=np.ones((1, 1), dtype=np.float32),
    )

    assert pos_loss > base_loss
    assert np.mean(np.abs(pos_grads["electron_density"])) > np.mean(np.abs(base_grads["electron_density"]))
    assert np.allclose(pos_grads["plasma_potential"], base_grads["plasma_potential"], atol=1e-6)
    assert per_var["__positive_penalty__"] > 0.0


def test_compose_supervised_numpy_spatial_consistency_multiscale_runs():
    pred = {
        "electron_density": np.array(
            [[[0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]]],
            dtype=np.float32,
        )
    }
    target = {
        "electron_density": np.array(
            [[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
    }
    mask = np.ones((1, 4, 4), dtype=np.float32)
    distance = np.ones((1, 4, 4), dtype=np.float32)

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density"],
        loss_cfg={
            "supervised": {
                "type": "mse",
                "mask": "plasma_only",
                "spatial_consistency": {
                    "enabled": True,
                    "vars": ["electron_density"],
                    "mode": "grad_huber",
                    "lambda": 0.1,
                    "delta": 1.0,
                    "apply_region": "plasma_only",
                    "multiscale": {"enabled": True, "scales": [1, 2], "scale_weights": [1.0, 0.5]},
                },
            }
        },
        mask=mask,
        distance_any=distance,
    )

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grads["electron_density"]))
    assert per_var["__spatial_consistency__"] > 0.0


def test_compose_supervised_numpy_protocol_v2_spatial_consistency_multiscale_runs():
    loss_cfg = resolve_loss_protocol(
        {"protocol": "plasma_surrogate_v2", "supervised": {"positive_penalty": {"enabled": False}}},
        target_role_schema={
            "targets": [
                {
                    "id": "electron_density",
                    "role": "density_electron",
                    "positive": True,
                    "field_family": "density",
                }
            ]
        },
    )
    pred = {
        "electron_density": np.array(
            [[[0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]]],
            dtype=np.float32,
        )
    }
    target = {
        "electron_density": np.array(
            [[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
    }

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density"],
        loss_cfg=loss_cfg,
        mask=np.ones((1, 4, 4), dtype=np.float32),
        distance_any=np.ones((1, 4, 4), dtype=np.float32),
    )

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grads["electron_density"]))
    assert per_var["__spatial_consistency__"] > 0.0


def test_compose_supervised_numpy_target_region_by_var_all_domain_includes_chamber():
    pred = {"magnetic_field": np.zeros((1, 2, 2), dtype=np.float32)}
    target = {"magnetic_field": np.array([[[0.0, 0.0], [2.0, 2.0]]], dtype=np.float32)}
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    plasma_loss, _, plasma_per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["magnetic_field"],
        loss_cfg={"supervised": {"type": "mse"}},
        mask=mask,
    )
    all_loss, _, all_per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["magnetic_field"],
        loss_cfg={"supervised": {"type": "mse", "target_region_by_var": {"magnetic_field": "all_domain"}}},
        mask=mask,
    )

    assert plasma_loss == pytest.approx(0.0)
    assert plasma_per_var["magnetic_field"] == pytest.approx(0.0)
    assert all_loss > 0.0
    assert all_per_var["magnetic_field"] > plasma_per_var["magnetic_field"]


def test_compose_supervised_numpy_target_region_by_var_rejects_unknown_or_bad_region():
    pred = {"magnetic_field": np.zeros((1, 1, 1), dtype=np.float32)}
    target = {"magnetic_field": np.zeros((1, 1, 1), dtype=np.float32)}

    with pytest.raises(ValueError, match="target_region_by_var contains unknown vars"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["magnetic_field"],
            loss_cfg={"supervised": {"target_region_by_var": {"other": "all_domain"}}},
        )
    with pytest.raises(ValueError, match="target_region_by_var values"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["magnetic_field"],
            loss_cfg={"supervised": {"target_region_by_var": {"magnetic_field": "coil_near"}}},
        )


def test_compose_supervised_removed_region_weighting_rejects():
    pred = {"plasma_potential": np.ones((1, 2, 2), dtype=np.float32)}
    target = {"plasma_potential": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="supervised.region_weighting is removed"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["plasma_potential"],
            loss_cfg={
                "supervised": {
                    "type": "mse",
                    "region_balance": {"enabled": True},
                    "region_weighting": {"enabled": True},
                }
            },
        )
