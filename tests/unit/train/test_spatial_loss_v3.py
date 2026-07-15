from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.loss_composer import (
    _masked_pool_numpy,
    _masked_pool_torch,
    compose_supervised_numpy,
    compose_supervised_torch,
)
from tests._runtime_requirements import require_torch_runtime


def _loss_cfg(
    *,
    boundary_weight: float = 0.0,
    boundary_band_px: float = 1.0,
    gradient_weight: float = 0.0,
    gradient_spacing: tuple[float, float] = (1.0, 1.0),
    gradient_normalization: str = "none",
    gradient_epsilon: float = 0.05,
    multiscale_weight: float = 0.0,
    multiscale_scales: tuple[int, ...] = (2,),
) -> dict:
    return {
        "supervised": {
            "type": "mse",
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "weighted",
            "spatial": {
                "boundary_weight": boundary_weight,
                "boundary_band_px": boundary_band_px,
                "gradient_weight": gradient_weight,
                "gradient_spacing": list(gradient_spacing),
                "gradient_normalization": gradient_normalization,
                "gradient_epsilon": gradient_epsilon,
                "multiscale_weight": multiscale_weight,
                "multiscale_scales": list(multiscale_scales),
            },
        }
    }


def test_target_rms_gradient_normalization_is_scale_invariant() -> None:
    target = np.array([[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]], dtype=np.float32)
    pred = target * 1.25
    mask = np.ones_like(target)
    cfg = _loss_cfg(
        gradient_weight=1.0,
        gradient_normalization="target_rms",
        gradient_epsilon=1.0e-6,
    )

    loss_a, _grads_a, parts_a = compose_supervised_numpy(
        {"field": pred},
        {"field": target},
        y_order=["field"],
        mask=mask,
        loss_cfg=cfg,
    )
    loss_b, _grads_b, parts_b = compose_supervised_numpy(
        {"field": pred * 10.0},
        {"field": target * 10.0},
        y_order=["field"],
        mask=mask,
        loss_cfg=cfg,
    )

    assert parts_a["loss_supervised_spatial_gradient_field"] == pytest.approx(
        parts_b["loss_supervised_spatial_gradient_field"]
    )
    # The point term remains scale-sensitive; only the spatial shape term is normalized.
    assert loss_b > loss_a


def test_numpy_ignores_nonfinite_values_outside_mask_for_all_spatial_terms() -> None:
    pred = np.zeros((1, 3, 3), dtype=np.float32)
    target = np.zeros_like(pred)
    mask = np.ones_like(pred)
    distance = np.zeros_like(pred)
    mask[:, -1, -1] = 0.0
    pred[:, -1, -1] = np.nan
    target[:, -1, -1] = np.inf
    distance[:, -1, -1] = np.nan

    loss, grads, parts = compose_supervised_numpy(
        {"field": pred},
        {"field": target},
        y_order=["field"],
        mask=mask,
        distance_any=distance,
        loss_cfg=_loss_cfg(
            boundary_weight=0.2,
            gradient_weight=0.1,
            multiscale_weight=0.05,
            multiscale_scales=(2, 4),
        ),
    )

    assert loss == pytest.approx(0.0)
    assert np.all(np.isfinite(grads["field"]))
    assert grads["field"][0, -1, -1] == pytest.approx(0.0)
    assert parts["loss_supervised_spatial_boundary_field"] == pytest.approx(0.0)
    assert parts["loss_supervised_spatial_gradient_field"] == pytest.approx(0.0)
    assert parts["loss_supervised_spatial_multiscale_field"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad_side", ["prediction", "target"])
def test_numpy_fails_fast_for_nonfinite_values_on_active_mask(bad_side: str) -> None:
    pred = np.zeros((1, 2, 2), dtype=np.float32)
    target = np.zeros_like(pred)
    if bad_side == "prediction":
        pred[0, 0, 0] = np.nan
    else:
        target[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite prediction/target on active mask"):
        compose_supervised_numpy(
            {"field": pred},
            {"field": target},
            y_order=["field"],
            mask=np.ones_like(pred),
            loss_cfg=_loss_cfg(),
        )


def test_masked_multiscale_padding_keeps_edge_cells_and_weights_valid_fraction() -> None:
    values = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.ones_like(values)
    _pooled, fractions, _cache = _masked_pool_numpy(values, mask, scale=2)

    np.testing.assert_allclose(
        fractions,
        np.array([[[1.0, 0.5], [0.5, 0.25]]], dtype=np.float32),
    )

    pred = values.copy()
    pred[0, -1, -1] = 2.0
    loss, _grads, parts = compose_supervised_numpy(
        {"field": pred},
        {"field": values},
        y_order=["field"],
        mask=mask,
        loss_cfg=_loss_cfg(multiscale_weight=1.0, multiscale_scales=(2,)),
    )

    assert parts["loss_supervised_point_field"] == pytest.approx(2.0 / 9.0)
    assert parts["loss_supervised_spatial_multiscale_field"] == pytest.approx(2.0 / 9.0)
    assert loss == pytest.approx(4.0 / 9.0)


def test_gradient_spacing_scales_gradient_component_in_physical_coordinates() -> None:
    pred = np.array([[[0.0, 0.0], [2.0, 2.0]]], dtype=np.float32)
    target = np.zeros_like(pred)
    mask = np.ones_like(pred)

    _loss_1, _grad_1, parts_1 = compose_supervised_numpy(
        {"field": pred},
        {"field": target},
        y_order=["field"],
        mask=mask,
        loss_cfg=_loss_cfg(gradient_weight=1.0, gradient_spacing=(1.0, 1.0)),
    )
    _loss_2, _grad_2, parts_2 = compose_supervised_numpy(
        {"field": pred},
        {"field": target},
        y_order=["field"],
        mask=mask,
        loss_cfg=_loss_cfg(gradient_weight=1.0, gradient_spacing=(2.0, 1.0)),
    )

    assert parts_1["loss_supervised_spatial_gradient_field"] == pytest.approx(1.0)
    assert parts_2["loss_supervised_spatial_gradient_field"] == pytest.approx(0.25)


def test_boundary_loss_is_separate_and_distance_driven() -> None:
    pred = np.array([[[2.0, 0.0, 2.0], [2.0, 0.0, 2.0]]], dtype=np.float32)
    target = np.zeros_like(pred)
    distance = np.array([[[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)

    loss, _grads, parts = compose_supervised_numpy(
        {"field": pred},
        {"field": target},
        y_order=["field"],
        mask=np.ones_like(pred),
        distance_any=distance,
        loss_cfg=_loss_cfg(boundary_weight=2.0, boundary_band_px=1.0),
    )

    assert parts["loss_supervised_point_field"] == pytest.approx(4.0 / 3.0)
    assert parts["loss_supervised_spatial_boundary_field"] == pytest.approx(4.0)
    assert loss == pytest.approx(16.0 / 3.0)


def test_boundary_loss_requires_distance_when_enabled() -> None:
    field = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="requires distance_any"):
        compose_supervised_numpy(
            {"field": field},
            {"field": field},
            y_order=["field"],
            mask=np.ones_like(field),
            loss_cfg=_loss_cfg(boundary_weight=0.2),
        )


def test_numpy_sample_mean_rejects_an_empty_case_mask() -> None:
    field = np.zeros((2, 2, 2), dtype=np.float32)
    mask = np.ones_like(field)
    mask[1] = 0.0

    with pytest.raises(ValueError, match=r"point loss has no active samples.*case_indices=\[1\]"):
        compose_supervised_numpy(
            {"field": field},
            {"field": field},
            y_order=["field"],
            mask=mask,
            loss_cfg=_loss_cfg(),
        )


def test_numpy_boundary_and_gradient_terms_reject_missing_spatial_support() -> None:
    field = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="boundary band has no active samples"):
        compose_supervised_numpy(
            {"field": field},
            {"field": field},
            y_order=["field"],
            mask=np.ones_like(field),
            distance_any=np.full_like(field, 9.0),
            loss_cfg=_loss_cfg(boundary_weight=0.2, boundary_band_px=1.0),
        )

    singleton_mask = np.zeros_like(field)
    singleton_mask[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="gradient has no active adjacent pixels"):
        compose_supervised_numpy(
            {"field": field},
            {"field": field},
            y_order=["field"],
            mask=singleton_mask,
            loss_cfg=_loss_cfg(gradient_weight=0.1),
        )


@pytest.mark.torch_runtime
def test_torch_ignores_nonfinite_values_outside_mask_and_fails_inside() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    pred = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    mask = torch.ones((1, 3, 3), dtype=torch.float32)
    mask[0, -1, -1] = 0.0
    with torch.no_grad():
        pred[0, 0, -1, -1] = float("nan")
        target[0, 0, -1, -1] = float("inf")

    loss, _parts = compose_supervised_torch(
        {"field": pred},
        target,
        y_order=["field"],
        mask=mask,
        loss_cfg=_loss_cfg(gradient_weight=0.1, multiscale_weight=0.05, multiscale_scales=(2, 4)),
    )
    loss.backward()
    assert float(loss.detach().cpu().item()) == pytest.approx(0.0)
    assert torch.all(torch.isfinite(pred.grad))

    active_bad = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    active_bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite prediction/target on active mask"):
        compose_supervised_torch(
            {"field": active_bad},
            torch.zeros_like(active_bad),
            y_order=["field"],
            mask=torch.ones((1, 2, 2), dtype=torch.float32),
            loss_cfg=_loss_cfg(),
        )


@pytest.mark.torch_runtime
def test_torch_spatial_terms_reject_empty_case_and_boundary_support() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    field = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    mask = torch.ones((2, 2, 2), dtype=torch.float32)
    mask[1] = 0.0

    with pytest.raises(ValueError, match=r"point loss has no active samples.*case_indices=\[1\]"):
        compose_supervised_torch(
            {"field": field},
            field,
            y_order=["field"],
            mask=mask,
            loss_cfg=_loss_cfg(),
        )

    with pytest.raises(ValueError, match="boundary band has no active samples"):
        compose_supervised_torch(
            {"field": field[:1]},
            field[:1],
            y_order=["field"],
            mask=torch.ones((1, 2, 2), dtype=torch.float32),
            distance_any=torch.full((1, 2, 2), 9.0, dtype=torch.float32),
            loss_cfg=_loss_cfg(boundary_weight=0.2, boundary_band_px=1.0),
        )


@pytest.mark.torch_runtime
def test_numpy_torch_spatial_loss_and_gradients_match() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    rng = np.random.default_rng(31)
    pred_np = rng.normal(size=(2, 3, 5)).astype(np.float32)
    target_np = rng.normal(size=(2, 3, 5)).astype(np.float32)
    mask = np.ones((2, 3, 5), dtype=np.float32)
    mask[0, 0, -1] = 0.0
    mask[1, -1, 0] = 0.0
    distance = np.abs(rng.normal(size=(2, 3, 5))).astype(np.float32)
    cfg = {
        "supervised": {
            "type": "huber",
            "huber_delta": 1.0,
            "normalization": "sample_mean",
            "sample_mean_weight_denominator": "weighted",
            "spatial": {
                "boundary_weight": 0.3,
                "boundary_band_px": 0.75,
                "gradient_weight": 0.2,
                "gradient_spacing": [0.5, 2.0],
                "multiscale_weight": 0.1,
                "multiscale_scales": [2, 4],
            },
        }
    }

    numpy_loss, numpy_grads, numpy_parts = compose_supervised_numpy(
        {"field": pred_np},
        {"field": target_np},
        y_order=["field"],
        mask=mask,
        distance_any=distance,
        loss_cfg=cfg,
    )

    pred_t = torch.tensor(pred_np[:, None], dtype=torch.float32, requires_grad=True)
    target_t = torch.tensor(target_np[:, None], dtype=torch.float32)
    torch_loss, torch_parts = compose_supervised_torch(
        {"field": pred_t},
        target_t,
        y_order=["field"],
        mask=torch.tensor(mask, dtype=torch.float32),
        distance_any=torch.tensor(distance, dtype=torch.float32),
        loss_cfg=cfg,
    )
    torch_loss.backward()

    assert float(torch_loss.detach().cpu().item()) == pytest.approx(numpy_loss, rel=2.0e-6, abs=2.0e-6)
    assert set(torch_parts) == set(numpy_parts)
    for key in numpy_parts:
        assert torch_parts[key] == pytest.approx(numpy_parts[key], rel=2.0e-6, abs=2.0e-6)
    np.testing.assert_allclose(
        pred_t.grad.detach().cpu().numpy()[:, 0],
        numpy_grads["field"],
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.torch_runtime
def test_numpy_torch_pool_valid_fractions_match() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    values = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.ones_like(values)
    numpy_pool, numpy_fraction, _cache = _masked_pool_numpy(values, mask, scale=2)
    torch_pool, torch_fraction = _masked_pool_torch(
        torch.tensor(values[:, None]),
        torch.tensor(mask[:, None]),
        scale=2,
    )

    np.testing.assert_allclose(torch_pool.detach().cpu().numpy()[:, 0], numpy_pool)
    np.testing.assert_allclose(torch_fraction.detach().cpu().numpy()[:, 0], numpy_fraction)
