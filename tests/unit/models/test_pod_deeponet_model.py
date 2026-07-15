from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from plasma_surrogate.infer.engine import InferenceEngine
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.pod_deeponet_torch import (
    PODDeepONetTorch,
    POD_DEEPONET_IMPL_VERSION,
    fit_pod_basis_from_targets,
    normalize_pod_deeponet_model_cfg,
)
from plasma_surrogate.models.checkpoint import load_checkpoint, save_checkpoint



def _build_targets(n: int = 6, h: int = 4, w: int = 4) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    out = np.zeros((n, 2, h, w), dtype=np.float32)
    for i in range(n):
        out[i, 0] = (i + 1) * xv + 0.1 * yv
        out[i, 1] = (i + 1) * yv - 0.05 * xv
    return out


def test_fit_pod_basis_helper_clamps_rank() -> None:
    y = _build_targets(n=3, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=32,
        center=True,
        per_var=True,
    )
    assert set(bundle.basis_by_var.keys()) == {"density", "temperature"}
    assert bundle.basis_by_var["density"].shape == (3, 4, 4)
    assert bundle.mean_by_var["temperature"].shape == (4, 4)
    assert bundle.rank_by_var == {"density": 3, "temperature": 3}
    assert bundle.coeff_std_by_var["density"].shape == (3,)
    assert np.all(bundle.coeff_std_by_var["density"] > 0.0)


def test_fit_pod_basis_ignores_inactive_fill_values_with_case_masks() -> None:
    y = _build_targets(n=3, h=3, w=3)[:, :1]
    mask = np.ones((3, 3, 3), dtype=np.float32)
    mask[0, 0, 0] = 0.0
    mask[1, 2, 2] = 0.0
    first = y.copy()
    second = y.copy()
    first[:, 0][mask == 0.0] = 1.0e20
    second[:, 0][mask == 0.0] = -1.0e20

    bundle_a = fit_pod_basis_from_targets(
        first,
        output_keys=["density"],
        requested_rank=3,
        center=True,
        per_var=True,
        active_mask=mask,
    )
    bundle_b = fit_pod_basis_from_targets(
        second,
        output_keys=["density"],
        requested_rank=3,
        center=True,
        per_var=True,
        active_mask=mask,
    )

    np.testing.assert_allclose(bundle_a.mean_by_var["density"], bundle_b.mean_by_var["density"])
    np.testing.assert_allclose(
        np.abs(bundle_a.basis_by_var["density"]),
        np.abs(bundle_b.basis_by_var["density"]),
        atol=1.0e-6,
    )


def test_fit_pod_basis_fails_closed_for_empty_or_invalid_active_support() -> None:
    y = _build_targets(n=2, h=3, w=3)[:, :1]
    empty = np.ones((2, 3, 3), dtype=np.float32)
    empty[1] = 0.0
    with pytest.raises(ValueError, match="empty active masks"):
        fit_pod_basis_from_targets(
            y,
            output_keys=["density"],
            requested_rank=2,
            center=True,
            per_var=True,
            active_mask=empty,
        )

    invalid = y.copy()
    invalid[0, 0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite active targets"):
        fit_pod_basis_from_targets(
            invalid,
            output_keys=["density"],
            requested_rank=2,
            center=True,
            per_var=True,
            active_mask=np.ones((2, 3, 3), dtype=np.float32),
        )


def test_normalize_pod_model_cfg_canonicalizes_hidden() -> None:
    cfg = normalize_pod_deeponet_model_cfg(
        {"hidden_dim": 16, "latent_dim": 12, "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True}},
        model_type="deeponet_pod",
    )
    assert cfg["hidden"] == [16, 12]
    assert "hidden_dim" not in cfg
    assert cfg["basis"]["fit_scope"] == "train_only"


def test_pod_deeponet_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()

    y = _build_targets(n=5, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=4,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden_dim": 16, "latent_dim": 12, "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    cond = np.random.default_rng(3).normal(size=(2, 3)).astype(np.float32)
    pred = model.forward(cond)
    assert pred.shape == (2, 2, 4, 4)
    assert np.all(np.isfinite(pred))

    ckpt = save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(ckpt)
    pred_loaded = loaded.forward(cond)
    assert pred_loaded.shape == pred.shape
    assert loaded.to_meta()["basis_rank_by_var"] == {"density": bundle.rank_by_var["density"], "temperature": bundle.rank_by_var["temperature"]}
    assert np.allclose(loaded.basis_bundle_numpy().basis_by_var["density"], bundle.basis_by_var["density"])
    assert loaded.to_meta()["impl_version"] == POD_DEEPONET_IMPL_VERSION
    assert "coeff_std_by_var" in loaded.to_meta()


def test_pod_deeponet_cached_aux_loss_diagnostics_are_side_effect_free() -> None:
    require_torch_runtime()

    y = _build_targets(n=5, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=4,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={
            "hidden": [16, 12],
            "coeff_loss_weight": 0.25,
            "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True, "center": True},
        },
    )
    cond = np.random.default_rng(4).normal(size=(2, 3)).astype(np.float32)
    target = y[:2]
    pred = model.forward_raw(cond, training=True)
    cached_out = model._torch_last_out
    cached_coeff = model._torch_last_coeff_norm
    training_state = bool(model.net.training)

    diagnostics = model.auxiliary_loss_diagnostics(target)
    diagnostics_repeat = model.auxiliary_loss_diagnostics(target)

    assert diagnostics == diagnostics_repeat
    assert diagnostics["coeff_loss"] == diagnostics["loss_aux_coeff"]
    assert diagnostics["loss_aux_coeff_weighted"] == pytest.approx(0.25 * diagnostics["loss_aux_coeff"])
    assert diagnostics["loss_aux_total"] == diagnostics["loss_aux_coeff_weighted"]
    assert model._torch_last_out is cached_out
    assert model._torch_last_coeff_norm is cached_coeff
    assert bool(model.net.training) is training_state

    backward_diagnostics = model.backward_raw(
        np.zeros_like(pred, dtype=np.float32),
        lr=1.0e-3,
        apply_step=False,
        target_raw=target,
    )
    for key in ("coeff_loss", "loss_aux_coeff", "loss_aux_coeff_weighted", "loss_aux_total"):
        assert backward_diagnostics[key] == pytest.approx(diagnostics[key])


def test_pod_deeponet_evaluates_auxiliary_loss_without_mutating_training_state() -> None:
    require_torch_runtime()

    y = _build_targets(n=5, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=4,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={
            "hidden": [16, 12],
            "coeff_loss_weight": 0.25,
            "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True, "center": True},
        },
    )
    cond = np.random.default_rng(8).normal(size=(2, 3)).astype(np.float32)
    target = y[:2]
    model.forward_raw(cond, training=True)
    expected = model.auxiliary_loss_diagnostics(target)
    cached_out = model._torch_last_out
    cached_coeff = model._torch_last_coeff_norm
    module_modes = [bool(module.training) for module in model.net.modules()]
    grad_refs = []
    for parameter in model.net.parameters():
        parameter.grad = model.torch.ones_like(parameter)
        grad_refs.append((parameter, parameter.grad, parameter.grad.detach().clone()))

    diagnostics = model.evaluate_auxiliary_losses(
        cond,
        target,
        spatial_features=np.zeros((2, 4, 4, 1), dtype=np.float32),
    )

    assert diagnostics == pytest.approx(expected)
    assert model._torch_last_out is cached_out
    assert model._torch_last_coeff_norm is cached_coeff
    assert [bool(module.training) for module in model.net.modules()] == module_modes
    for parameter, grad_ref, grad_value in grad_refs:
        assert parameter.grad is grad_ref
        assert model.torch.equal(parameter.grad, grad_value)


def test_pod_deeponet_auxiliary_evaluation_validates_inputs_and_weight() -> None:
    require_torch_runtime()

    target = _build_targets(n=4, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        target,
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={
            "hidden": [8, 6],
            "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True},
        },
    )
    cond = np.zeros((2, 3), dtype=np.float32)
    diagnostics = model.evaluate_auxiliary_losses(
        cond,
        target[:2],
        loss_cfg={"deeponet_pod": {"coeff_loss_weight": 0.5}},
    )
    assert diagnostics["loss_aux_total"] == pytest.approx(0.5 * diagnostics["loss_aux_coeff"])
    assert model._torch_last_out is None
    assert model._torch_last_coeff_norm is None

    with pytest.raises(ValueError, match="feature mismatch"):
        model.evaluate_auxiliary_losses(np.zeros((2, 2), dtype=np.float32), target[:2])
    with pytest.raises(ValueError, match="batch mismatch"):
        model.evaluate_auxiliary_losses(cond, target[:1])
    with pytest.raises(ValueError, match="finite and >= 0"):
        model.evaluate_auxiliary_losses(
            cond,
            target[:2],
            loss_cfg={"deeponet_pod": {"coeff_loss_weight": float("nan")}},
        )


def test_pod_deeponet_declares_cond_only_inference_capability() -> None:
    assert PODDeepONetTorch.requires_spatial_features is False


@pytest.mark.parametrize("model_type", ["deeponet_plasma_pod", "geom_deeponet_pod"])
def test_pod_deeponet_checkpoint_preserves_family_model_type(tmp_path: Path, model_type: str) -> None:
    require_torch_runtime()

    bundle = fit_pod_basis_from_targets(
        _build_targets(n=4, h=4, w=4),
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden": [16, 12], "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True}},
        model_type=model_type,
    )
    ckpt = save_checkpoint(model, tmp_path / f"ckpt_{model_type}")
    loaded = load_checkpoint(ckpt)
    assert loaded.to_meta()["model_type"] == model_type


def test_pod_deeponet_rejects_legacy_checkpoint_format(tmp_path: Path) -> None:
    require_torch_runtime()
    y = _build_targets(n=4, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden": [16, 12], "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    ckpt = save_checkpoint(model, tmp_path / "ckpt_legacy")
    meta_path = Path(ckpt) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.pop("impl_version", None)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with pytest.raises(ValueError, match="expected impl_version"):
        load_checkpoint(ckpt)


def test_pod_deeponet_inference_does_not_use_grid_spatial_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    require_torch_runtime()

    bundle = fit_pod_basis_from_targets(
        _build_targets(n=4, h=4, w=4),
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden": [16, 12], "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    engine = InferenceEngine(
        model=model,
        cond_schema=SimpleNamespace(),
        axis_schema=SimpleNamespace(mode="steady"),
        geometry_provider=SimpleNamespace(),
        output_dir=tmp_path / "infer",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("grid-spatial path should not be used for deeponet_pod")

    monkeypatch.setattr(engine.predictor, "_predict_grid_spatial_fields", _fail)
    out = engine.predictor.predict_fields(
        np.zeros((3,), dtype=np.float32),
        geom=SimpleNamespace(mask_plasma=np.ones((4, 4), dtype=np.float32)),
    )
    assert set(out.keys()) == {"density", "temperature"}
