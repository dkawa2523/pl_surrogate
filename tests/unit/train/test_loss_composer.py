from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.loss_composer import compose_numpy, compose_supervised_numpy, compose_supervised_torch
from plasma_surrogate.train.loss_contract import REMOVED_SUPERVISED_KEYS
from plasma_surrogate.train.loss_protocols import resolve_loss_protocol
from tests._runtime_requirements import require_torch_runtime


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
        loss_cfg={"supervised": {"type": "huber", "huber_delta": 1.0}},
        mask=mask,
    )

    assert loss > 0.0
    assert set(grads) == set(pred)
    assert set(per_var) == set(pred)
    assert all(grads[name].shape == pred[name].shape for name in pred)


def test_compose_supervised_numpy_huber_delta_changes_loss_shape():
    pred = {"electron_density": _field(2.0, (1, 2, 2))}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density"],
        loss_cfg={"supervised": {"type": "huber", "huber_delta": 1.0}},
    )

    assert loss == pytest.approx(1.5)
    assert per_var["electron_density"] == pytest.approx(1.5)
    assert np.allclose(grads["electron_density"], np.full((1, 2, 2), 0.25, dtype=np.float32))


@pytest.mark.parametrize("bad_delta", [0.0, -1.0, float("inf"), float("nan")])
def test_compose_supervised_numpy_rejects_bad_huber_delta(bad_delta):
    pred = {"electron_density": _field(1.0, (1, 2, 2))}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="huber_delta"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["electron_density"],
            loss_cfg={"supervised": {"type": "huber", "huber_delta": bad_delta}},
        )


def test_compose_supervised_numpy_rejects_unknown_supervised_type():
    pred = {"electron_density": _field(1.0, (1, 2, 2))}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="supervised.type"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["electron_density"],
            loss_cfg={"supervised": {"type": "smooth_l1"}},
        )


@pytest.mark.parametrize("alias_cfg", [{"base": "mse"}, {"type": "huber", "delta": 1.0}])
def test_compose_supervised_numpy_rejects_removed_supervised_aliases(alias_cfg):
    pred = {"electron_density": _field(1.0, (1, 2, 2))}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="removed"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["electron_density"],
            loss_cfg={"supervised": alias_cfg},
        )


@pytest.mark.parametrize(
    "removed_key",
    REMOVED_SUPERVISED_KEYS,
)
def test_compose_supervised_numpy_rejects_research_supervised_keys(removed_key):
    pred = {"electron_density": np.ones((1, 2, 2), dtype=np.float32)}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match=removed_key):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["electron_density"],
            loss_cfg={"supervised": {"type": "mse", removed_key: {"enabled": True}}},
            mask=np.ones((2, 2), dtype=np.float32),
        )


def test_compose_supervised_numpy_protocol_v2_uses_standard_loss_only():
    loss_cfg = resolve_loss_protocol(
        {"protocol": "plasma_surrogate_v2"},
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
    assert "__spatial_consistency__" not in per_var


def test_compose_supervised_numpy_group_weighting_none_preserves_target_sum():
    pred = {
        "electron_density": _field(2.0, (1, 2, 2)),
        "electron_temperature": _field(1.0, (1, 2, 2)),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "electron_temperature"],
        loss_cfg={"supervised": {"type": "mse"}, "group_weighting": {"mode": "none"}},
    )

    assert loss == pytest.approx(2.5)
    assert per_var == pytest.approx({"electron_density": 2.0, "electron_temperature": 0.5})
    assert np.allclose(grads["electron_density"], np.full((1, 2, 2), 0.5, dtype=np.float32))


def test_compose_supervised_numpy_uniform_by_target_averages_targets():
    pred = {
        "electron_density": _field(2.0, (1, 2, 2)),
        "electron_temperature": _field(1.0, (1, 2, 2)),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=["electron_density", "electron_temperature"],
        loss_cfg={"supervised": {"type": "mse"}, "group_weighting": {"mode": "uniform_by_target"}},
    )

    assert loss == pytest.approx(1.25)
    assert per_var == pytest.approx({"electron_density": 1.0, "electron_temperature": 0.25})
    assert np.allclose(grads["electron_density"], np.full((1, 2, 2), 0.25, dtype=np.float32))


def test_compose_supervised_numpy_uniform_by_group_balances_field_families():
    y_order = ["electron_density", "ion_density", "electron_temperature"]
    pred = {
        "electron_density": _field(2.0, (1, 2, 2)),
        "ion_density": _field(1.0, (1, 2, 2)),
        "electron_temperature": _field(4.0, (1, 2, 2)),
    }
    target = {name: np.zeros_like(value) for name, value in pred.items()}
    schema = {
        "targets": [
            {"id": "electron_density", "field_family": "density"},
            {"id": "ion_density", "field_family": "density"},
            {"id": "electron_temperature", "field_family": "temperature"},
        ]
    }

    loss, grads, per_var = compose_supervised_numpy(
        pred,
        target,
        y_order=y_order,
        loss_cfg={
            "supervised": {"type": "mse"},
            "group_weighting": {"mode": "uniform_by_group"},
            "target_role_schema": schema,
        },
    )

    assert loss == pytest.approx(4.625)
    assert per_var["electron_density"] == pytest.approx(0.5)
    assert per_var["ion_density"] == pytest.approx(0.125)
    assert per_var["electron_temperature"] == pytest.approx(4.0)
    assert per_var["loss_supervised_group_density"] == pytest.approx(0.625)
    assert per_var["loss_supervised_group_temperature"] == pytest.approx(4.0)
    assert np.allclose(grads["electron_temperature"], np.full((1, 2, 2), 0.5, dtype=np.float32))


def test_compose_supervised_numpy_uniform_by_group_requires_target_role_schema():
    pred = {"electron_density": _field(1.0, (1, 2, 2))}
    target = {"electron_density": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="requires target_role_schema.targets"):
        compose_supervised_numpy(
            pred,
            target,
            y_order=["electron_density"],
            loss_cfg={"supervised": {"type": "mse"}, "group_weighting": {"mode": "uniform_by_group"}},
        )


@pytest.mark.torch_runtime
def test_compose_supervised_torch_uniform_by_group_reports_breakdown():
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    y_order = ["electron_density", "ion_density", "electron_temperature"]
    pred = {
        "electron_density": torch.full((1, 1, 2, 2), 2.0),
        "ion_density": torch.full((1, 1, 2, 2), 1.0),
        "electron_temperature": torch.full((1, 1, 2, 2), 4.0),
    }
    target = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    schema = {
        "targets": [
            {"id": "electron_density", "field_family": "density"},
            {"id": "ion_density", "field_family": "density"},
            {"id": "electron_temperature", "field_family": "temperature"},
        ]
    }

    loss, per_var = compose_supervised_torch(
        pred,
        target,
        y_order=y_order,
        loss_cfg={
            "supervised": {"type": "mse"},
            "group_weighting": {"mode": "uniform_by_group"},
            "target_role_schema": schema,
        },
    )

    assert float(loss.detach().cpu().item()) == pytest.approx(4.625)
    assert per_var["loss_supervised_group_density"] == pytest.approx(0.625)
    assert per_var["loss_supervised_group_temperature"] == pytest.approx(4.0)


@pytest.mark.torch_runtime
def test_compose_supervised_torch_huber_is_differentiable():
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    pred = {"electron_density": torch.full((1, 1, 2, 2), 2.0, requires_grad=True)}
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    loss, per_var = compose_supervised_torch(
        pred,
        target,
        y_order=["electron_density"],
        loss_cfg={"supervised": {"type": "huber", "huber_delta": 1.0}},
    )
    loss.backward()

    assert float(loss.detach().cpu().item()) == pytest.approx(1.5)
    assert per_var["electron_density"] == pytest.approx(1.5)
    assert pred["electron_density"].grad is not None
    assert np.allclose(
        pred["electron_density"].grad.detach().cpu().numpy(),
        np.full((1, 1, 2, 2), 0.25, dtype=np.float32),
    )


def test_compose_supervised_removed_region_weighting_rejects():
    pred = {"plasma_potential": np.ones((1, 2, 2), dtype=np.float32)}
    target = {"plasma_potential": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="region_weighting"):
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
