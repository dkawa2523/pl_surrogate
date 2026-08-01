from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.models.deeponet.pod_deeponet_torch import fit_pod_basis_from_targets
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import CoordMLPPODResidual
from tests._runtime_requirements import require_torch_runtime


def _build_model() -> tuple[CoordMLPPODResidual, np.ndarray]:
    rng = np.random.default_rng(123)
    basis_targets = rng.normal(size=(6, 2, 4, 4)).astype(np.float32)
    basis_bundle = fit_pod_basis_from_targets(
        basis_targets,
        output_keys=["ne", "Te"],
        requested_rank=2,
        center=True,
        per_var=True,
    )
    model = CoordMLPPODResidual(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["ne", "Te"],
        input_feature_channels=["x", "y", "mask_plasma"],
        pod_basis_bundle=basis_bundle,
        model_cfg={
            "basis": {"rank": 2},
            "cond_hidden": [8],
            "latent_dim": 6,
            "residual_hidden": [8],
            "point_encoder": {"xy_fourier_frequencies": 2},
            "coeff_loss_weight": 0.1,
        },
        seed=7,
    )
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 4, dtype=np.float32),
        np.linspace(0.0, 1.0, 4, dtype=np.float32),
        indexing="ij",
    )
    spatial = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float32)
    model.set_static_spatial_features(spatial)
    return model, spatial


def test_evaluate_auxiliary_losses_is_cache_mode_and_grad_safe() -> None:
    require_torch_runtime()
    model, spatial = _build_model()
    cond = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target = np.random.default_rng(456).normal(size=(2, 2, 4, 4)).astype(np.float32)
    pred = model.forward_raw(cond, training=True, spatial_features=spatial)
    cached_diagnostics = model.auxiliary_loss_diagnostics(
        target,
        loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": 0.25}},
    )

    # Exercise exact mode restoration, including a deliberately mixed module tree.
    model.net.train()
    model.net.point_encoder.eval()
    mode_before = [(module, bool(module.training)) for module in model.net.modules()]
    cached_out = model._torch_last_out
    cached_coeff = model._torch_last_coeff_norm
    params = list(model.net.parameters())
    params[0].grad = model.torch.full_like(params[0], 0.125)
    grad_refs = [param.grad for param in params]
    grad_values = [None if grad is None else grad.detach().clone() for grad in grad_refs]

    diagnostics = model.evaluate_auxiliary_losses(
        cond,
        target,
        spatial_features=spatial,
        loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": 0.25}},
    )

    assert set(diagnostics) == {
        "coeff_loss",
        "loss_aux_coeff",
        "loss_aux_coeff_ne",
        "loss_aux_coeff_Te",
        "loss_aux_coeff_weighted",
        "loss_aux_total",
    }
    assert diagnostics == pytest.approx(cached_diagnostics)
    assert diagnostics["loss_aux_coeff_weighted"] == pytest.approx(
        0.25 * diagnostics["loss_aux_coeff"]
    )
    assert model._torch_last_out is cached_out
    assert model._torch_last_coeff_norm is cached_coeff
    assert [bool(module.training) for module, _ in mode_before] == [
        was_training for _, was_training in mode_before
    ]
    for param, grad_ref, grad_value in zip(params, grad_refs, grad_values, strict=True):
        assert param.grad is grad_ref
        if grad_value is not None:
            model.torch.testing.assert_close(param.grad, grad_value)

    # The validation call must leave the cached graph usable by the training path.
    backward_diagnostics = model.backward_raw(
        np.zeros_like(pred),
        lr=0.0,
        apply_step=False,
        target_raw=target,
        loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": 0.25}},
    )
    assert backward_diagnostics["loss_aux_total"] == pytest.approx(
        diagnostics["loss_aux_total"]
    )


def test_evaluate_auxiliary_losses_validates_shapes_and_weight() -> None:
    require_torch_runtime()
    model, spatial = _build_model()
    cond = np.ones((2, 3), dtype=np.float32)
    target = np.ones((2, 2, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=r"cond must be \[B,input_dim\]"):
        model.evaluate_auxiliary_losses(np.ones((1, 2, 3), dtype=np.float32), target)
    with pytest.raises(ValueError, match="cond feature mismatch"):
        model.evaluate_auxiliary_losses(np.ones((2, 4), dtype=np.float32), target)
    cond_nonfinite = cond.copy()
    cond_nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="cond must contain only finite values"):
        model.evaluate_auxiliary_losses(cond_nonfinite, target)
    with pytest.raises(ValueError, match=r"target_raw must be \[B,C,H,W\]"):
        model.evaluate_auxiliary_losses(cond, np.ones((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="target_raw batch mismatch"):
        model.evaluate_auxiliary_losses(cond, np.ones((1, 2, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="target_raw channel mismatch"):
        model.evaluate_auxiliary_losses(cond, np.ones((2, 3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="target_raw grid mismatch"):
        model.evaluate_auxiliary_losses(cond, np.ones((2, 2, 3, 4), dtype=np.float32))
    target_nonfinite = target.copy()
    target_nonfinite[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="target_raw must contain only finite values"):
        model.evaluate_auxiliary_losses(cond, target_nonfinite)
    with pytest.raises(ValueError, match="spatial features batch mismatch"):
        model.evaluate_auxiliary_losses(
            cond,
            target,
            spatial_features=np.repeat(spatial[None, ...], 3, axis=0),
        )
    with pytest.raises(ValueError, match="coeff_loss_weight must be finite and >= 0"):
        model.evaluate_auxiliary_losses(
            cond,
            target,
            loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": -0.1}},
        )


def test_evaluate_auxiliary_losses_does_not_require_training_cache() -> None:
    require_torch_runtime()
    model, _ = _build_model()
    model.net.eval()
    cond = np.ones((1, 3), dtype=np.float32)
    target = np.zeros((1, 2, 4, 4), dtype=np.float32)

    diagnostics = model.evaluate_auxiliary_losses(cond, target)

    assert np.isfinite(list(diagnostics.values())).all()
    assert model.net.training is False
    assert model._torch_last_out is None
    assert model._torch_last_coeff_norm is None
