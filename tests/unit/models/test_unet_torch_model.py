from __future__ import annotations

import pytest
import numpy as np

from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def test_unet_torch_forward_accepts_batched_spatial_features() -> None:
    require_torch_runtime()
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
