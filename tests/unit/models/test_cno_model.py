from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np

from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet
from plasma_surrogate.models.cno.simple_cno import CNOBaseline
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint

pytestmark = pytest.mark.torch_runtime


def _cno_cfg() -> dict[str, object]:
    return {"width": 16, "n_layers": 2, "dropout": 0.0, "kernel_size": 3}


def _cno_operator_unet_cfg() -> dict[str, object]:
    return {
        "width": 12,
        "depth": 2,
        "blocks_per_level": 1,
        "max_width": 24,
        "dropout": 0.0,
        "kernel_size": 3,
        "downsample": "blur",
        "upsample": "bilinear",
        "use_film": True,
    }


def test_build_cno_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="cno",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={"backend": "torch", "cno_cfg": _cno_cfg()},
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, CNOBaseline)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]


def test_build_cno_operator_unet_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="cno_operator_unet",
        input_dim=3,
        grid_shape=(9, 8),
        model_cfg={"backend": "torch", "cno_operator_unet_cfg": _cno_operator_unet_cfg()},
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, CNOOperatorUNet)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]


def test_cno_forward_shape() -> None:
    require_torch_runtime()
    model = CNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        cno_cfg=_cno_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(9).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(10).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (2, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}
    assert np.isfinite(out).all()


def test_cno_operator_unet_forward_shape() -> None:
    require_torch_runtime()
    model = CNOOperatorUNet(
        input_dim=3,
        grid_shape=(9, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        cno_operator_unet_cfg=_cno_operator_unet_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(13).normal(size=(9, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(14).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (2, 2, 9, 8)
    assert set(pred.keys()) == {"density", "temperature"}
    assert np.isfinite(out).all()


def test_cno_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = CNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        cno_cfg=_cno_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(11).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(12).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_cno")
    loaded = load_checkpoint(tmp_path / "ckpt_cno")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CNOBaseline)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_cno_operator_unet_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = CNOOperatorUNet(
        input_dim=3,
        grid_shape=(9, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        cno_operator_unet_cfg=_cno_operator_unet_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(15).normal(size=(9, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(16).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_cno_operator_unet")
    loaded = load_checkpoint(tmp_path / "ckpt_cno_operator_unet")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CNOOperatorUNet)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_cno_rejects_even_kernel_size() -> None:
    require_torch_runtime()
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        CNOBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            cno_cfg={"width": 16, "n_layers": 2, "dropout": 0.0, "kernel_size": 4},
            backend="torch",
        )
