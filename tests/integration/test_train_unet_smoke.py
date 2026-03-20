from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.train.trainer import Trainer


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

    model = UNetBaseline(input_dim=d, grid_shape=(h, w), seed=1)
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
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")

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
        selection_cfg={
            "mode": "best_val_field_boundary_balance",
            "eval_every_n_epochs": 1,
            "warmup_epochs": 0,
            "boundary_band_px": 1.0,
            "weights": {"Te": 0.5, "phi": 0.5},
        },
    )
    assert len(out.history) > 0
    assert np.isfinite(float(out.history[-1].get("val_balance_score", 0.0)))


def test_train_unet_torch_backend_depth2_smoke(tmp_path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")

    rng = np.random.default_rng(1)
    n, d, h, w = 8, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    y = np.zeros((n, 2, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.15 * xv
        y[i, 1] = cond[i, 1] + 0.25 * yv

    tr = np.arange(0, 6)
    va = np.arange(6, 8)
    model = UNetBaseline(
        input_dim=d,
        grid_shape=(h, w),
        seed=1,
        out_channels=2,
        output_keys=["Te", "phi"],
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8, "depth": 2},
    )
    trainer = Trainer(tmp_path / "train_unet_torch_depth2")
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


def test_unet_torch_backend_invalid_depth_raises():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    with pytest.raises(ValueError, match="conv_cfg.depth"):
        UNetBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            backend="torch",
            conv_cfg={"base_channels": 8, "depth": 3},
        )


def test_unet_torch_backend_invalid_upsample_mode_raises():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    with pytest.raises(ValueError, match="upsample_mode"):
        UNetBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            backend="torch",
            conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bad_mode"},
        )


def test_train_unet_torch_backend_resize_conv_depth2_smoke(tmp_path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")

    rng = np.random.default_rng(11)
    n, d, h, w = 8, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    y = np.zeros((n, 4, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.15 * xv
        y[i, 1] = cond[i, 0] + 0.12 * yv
        y[i, 2] = cond[i, 1] + 0.25 * yv
        y[i, 3] = cond[i, 2] + 0.1 * (xv + yv)

    tr = np.arange(0, 6)
    va = np.arange(6, 8)
    model = UNetBaseline(
        input_dim=d,
        grid_shape=(h, w),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "resize_conv"},
        output_heads={"mode": "shared"},
    )
    trainer = Trainer(tmp_path / "train_unet_torch_resize_conv_depth2")
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


def test_unet_torch_split_density_field_forward_smoke():
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    model = UNetBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=4,
        output_keys=["log_ne", "log_ni", "Te", "phi"],
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8, "depth": 1},
        output_heads={"mode": "split_density_field"},
    )
    cond = np.zeros((2, 3), dtype=np.float32)
    out = model.forward(cond, training=False)
    assert out.shape == (2, 4, 8, 8)


def test_train_unet_torch_allvars_boundary_selection_smoke(tmp_path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    rng = np.random.default_rng(2)
    n, d, h, w = 10, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    y = np.zeros((n, 4, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.2 * xv
        y[i, 1] = cond[i, 0] + 0.1 * yv
        y[i, 2] = cond[i, 1] + 0.3 * yv
        y[i, 3] = cond[i, 2] + 0.1 * (xv + yv)
    mask = np.ones((1, h, w), dtype=np.float32)
    dist = np.ones((1, h, w), dtype=np.float32)
    tr = np.arange(0, 7)
    va = np.arange(7, 10)
    model = UNetBaseline(
        input_dim=d,
        grid_shape=(h, w),
        out_channels=4,
        output_keys=["log_ne", "log_ni", "Te", "phi"],
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8, "depth": 1},
        output_heads={"mode": "split_density_field"},
    )
    trainer = Trainer(tmp_path / "train_unet_torch_allvars_selection")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=3,
        lr=1e-3,
        physics_cfg={"enabled": False},
        loss_cfg={"supervised": {"mask": "plasma_only", "normalization": "sample_mean"}},
        supervised_mask=mask,
        supervised_distance=dist,
        selection_cfg={
            "mode": "best_val_allvars_boundary_balance",
            "eval_every_n_epochs": 1,
            "warmup_epochs": 0,
            "boundary_band_px": 1.0,
            "weights": {"log_ne": 0.25, "log_ni": 0.25, "Te": 0.25, "phi": 0.25},
        },
        unet_optimizer_cfg={"type": "adamw", "schedule": "cosine", "warmup_epochs": 1},
    )
    assert len(out.history) == 3
    assert np.isfinite(float(out.history[-1].get("val_balance_score", 0.0)))


def test_train_unet_torch_allvars_boundary_selection_ignores_legacy_density_guard_cfg(tmp_path):
    if not torch_runtime_available():
        pytest.skip("torch backend disabled for this environment")
    rng = np.random.default_rng(3)
    n, d, h, w = 10, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    y = np.zeros((n, 4, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.2 * xv
        y[i, 1] = cond[i, 0] + 0.1 * yv
        y[i, 2] = cond[i, 1] + 0.3 * yv
        y[i, 3] = cond[i, 2] + 0.1 * (xv + yv)
    mask = np.ones((1, h, w), dtype=np.float32)
    dist = np.full((1, h, w), 20.0, dtype=np.float32)
    dist[:, :2, :] = 0.5
    tr = np.arange(0, 7)
    va = np.arange(7, 10)
    model = UNetBaseline(
        input_dim=d,
        grid_shape=(h, w),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        backend="torch",
        input_feature_channels=["x", "y"],
        conv_cfg={"base_channels": 8, "depth": 1},
        output_heads={"mode": "split_density_field"},
    )
    trainer = Trainer(tmp_path / "train_unet_torch_allvars_selection_guard")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=3,
        lr=1e-3,
        physics_cfg={"enabled": False},
        loss_cfg={"supervised": {"mask": "plasma_only", "normalization": "sample_mean"}},
        supervised_mask=mask,
        supervised_distance=dist,
        selection_cfg={
            "mode": "best_val_allvars_boundary_balance",
            "eval_every_n_epochs": 1,
            "warmup_epochs": 0,
            "boundary_band_px": 1.0,
            "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
            "density_guard": {
                "enabled": True,
                "min_r2_plasma": 0.99,
                "min_r2_deep": 0.99,
                "max_neg_ratio": 0.0,
                "require_both": True,
            },
        },
        unet_optimizer_cfg={"type": "adamw", "schedule": "cosine", "warmup_epochs": 1},
    )
    assert len(out.history) == 3
    assert np.isfinite(float(out.history[-1].get("val_balance_score", 0.0)))
