from __future__ import annotations

import os

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.models.unet.simple_unet import UNetBaseline


def test_unet_torch_forward_accepts_batched_spatial_features() -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = UNetBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 1, "upsample_mode": "deconv"},
        output_heads={"mode": "shared"},
    )
    spatial = np.random.default_rng(7).normal(size=(2, 8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(8).normal(size=(2, 3)).astype(np.float32)
    out = model.forward(cond, spatial_features=spatial)
    pred = model.predict_fields(cond, spatial_features=spatial)
    assert out.shape == (2, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}
