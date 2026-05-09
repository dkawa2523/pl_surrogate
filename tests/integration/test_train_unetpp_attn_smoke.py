from __future__ import annotations

import pytest
import numpy as np

from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.train.trainer import Trainer

pytestmark = pytest.mark.torch_runtime


def test_train_unetpp_attn_smoke(tmp_path) -> None:
    require_torch_runtime()

    rng = np.random.default_rng(9)
    n, d, h, w = 10, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    y = np.zeros((n, 4, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.15 * xv
        y[i, 1] = cond[i, 0] + 0.10 * yv
        y[i, 2] = cond[i, 1] + 0.20 * (xv + yv)
        y[i, 3] = cond[i, 2] + 0.05 * (xv - yv)

    spatial = np.stack(
        [
            np.tile(xx, (h, 1)),
            np.tile(yy[:, None], (1, w)),
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.25, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    tr = np.arange(0, 7)
    va = np.arange(7, 10)
    model = UNetPPBaseline(
        input_dim=d,
        grid_shape=(h, w),
        out_channels=4,
        output_keys=["density_e", "density_i", "temperature", "potential"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={
            "base_channels": 8,
            "depth": 2,
            "upsample_mode": "bilinear",
            "nested_skip": True,
            "attention_cfg": {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"},
        },
        output_heads={"mode": "shared"},
    )
    model.set_static_spatial_features(spatial)
    trainer = Trainer(tmp_path / "train_unetpp_attn")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=2,
        lr=1e-2,
        physics_cfg={"enabled": False},
    )
    assert len(out.history) == 2
