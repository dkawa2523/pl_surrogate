from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np

from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline

pytestmark = pytest.mark.torch_runtime


def test_build_unetpp_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unetpp",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={
            "backend": "torch",
            "conv_cfg": {"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
            "output_heads": {"mode": "shared"},
        },
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, UNetPPBaseline)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]


def test_unetpp_static_spatial_features_shape_validation() -> None:
    require_torch_runtime()
    model = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
        output_heads={"mode": "shared"},
    )
    with pytest.raises(ValueError, match="channels mismatch"):
        model.set_static_spatial_features(np.zeros((8, 8, 4), dtype=np.float32))


def test_unetpp_forward_shape_with_dynamic_output_keys() -> None:
    require_torch_runtime()
    model = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
        output_heads={"mode": "shared"},
    )
    spatial = np.zeros((8, 8, 5), dtype=np.float32)
    spatial[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :]
    spatial[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    spatial[..., 2] = 1.0
    spatial[..., 3] = 0.5
    spatial[..., 4] = 0.25
    model.set_static_spatial_features(spatial)
    cond = np.ones((3, 3), dtype=np.float32)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (3, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}


def test_unetpp_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
        output_heads={"mode": "shared"},
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, UNetPPBaseline)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_unetpp_forward_accepts_batched_spatial_features() -> None:
    require_torch_runtime()
    model = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
        output_heads={"mode": "shared"},
    )
    spatial = np.random.default_rng(3).normal(size=(2, 8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(4).normal(size=(2, 3)).astype(np.float32)
    out = model.forward(cond, spatial_features=spatial)
    assert out.shape == (2, 2, 8, 8)


@pytest.mark.parametrize("upsample_mode", ["deconv", "resize_conv"])
def test_unetpp_build_smoke_supports_upsample_variants(upsample_mode: str) -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unetpp",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={
            "backend": "torch",
            "conv_cfg": {"base_channels": 8, "depth": 2, "upsample_mode": upsample_mode, "nested_skip": True},
            "output_heads": {"mode": "shared"},
        },
        out_channels=2,
        output_keys=["density", "temperature"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, UNetPPBaseline)
