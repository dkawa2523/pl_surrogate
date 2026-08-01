from __future__ import annotations

import numpy as np

from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.train.trainer import Trainer


def _require_torch_runtime_from_env() -> None:
    require_torch_runtime(enable_backend=False, refresh=False)


def test_train_unet_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n, d, h, w = 12, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")

    y = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.2 * xv
        y[i, 1] = cond[i, 1] + 0.3 * yv
        y[i, 2] = cond[i, 2] + 0.1 * (xv + yv)

    tr = np.arange(0, 8)
    va = np.arange(8, 12)

    spatial_features = np.stack([xv, yv], axis=-1).astype(np.float32)
    model = UNetBaseline(input_dim=d, grid_shape=(h, w), seed=1)
    model.set_static_spatial_features(spatial_features)
    trainer = Trainer(tmp_path / "train_unet")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=6,
        lr=5e-2,
        physics_cfg={"enabled": False},
    )
    assert out.history[-1]["train_loss"] <= out.history[0]["train_loss"]


def test_train_unet_torch_backend_smoke(tmp_path):
    _require_torch_runtime_from_env()

    rng = np.random.default_rng(0)
    n, d, h, w = 10, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    mask = np.ones((1, h, w), dtype=np.float32)
    dist = np.ones((1, h, w), dtype=np.float32)

    y = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.2 * xv
        y[i, 1] = cond[i, 1] + 0.3 * yv
        y[i, 2] = cond[i, 2] + 0.1 * (xv + yv)

    tr = np.arange(0, 7)
    va = np.arange(7, 10)
    model = UNetBaseline(
        input_dim=d,
        grid_shape=(h, w),
        seed=1,
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8},
    )
    model.set_static_spatial_features(np.stack([xv, yv], axis=-1).astype(np.float32))
    trainer = Trainer(tmp_path / "train_unet_torch")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=4,
        lr=1e-2,
        physics_cfg={"enabled": False},
        loss_cfg={"supervised": {"mask": "plasma_only", "normalization": "sample_mean"}},
        supervised_mask=mask,
        supervised_distance=dist,
    )
    assert len(out.history) > 0
    assert np.isfinite(float(out.history[-1].get("val_balance_score", 0.0)))
