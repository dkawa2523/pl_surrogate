from __future__ import annotations

import os

import numpy as np

from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.train.trainer import Trainer


def test_train_fno_smoke(tmp_path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    rng = np.random.default_rng(1)
    n, d, h, w = 12, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")

    y = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.2 * np.sin(np.pi * xv)
        y[i, 1] = cond[i, 1] + 0.2 * np.cos(np.pi * yv)
        y[i, 2] = cond[i, 2] + 0.1 * np.sin(np.pi * (xv + yv))

    tr = np.arange(0, 8)
    va = np.arange(8, 12)
    model = FNOBaseline(
        input_dim=d,
        grid_shape=(h, w),
        seed=2,
        n_modes=2,
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        spectral_cfg={"width": 16, "n_layers": 2, "dealias_ratio": 0.67, "taper_alpha": 4.0},
    )
    trainer = Trainer(tmp_path / "train_fno")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=6,
        lr=5e-3,
        physics_cfg={"enabled": False},
    )
    assert out.history[-1]["train_loss"] <= out.history[0]["train_loss"]


def test_fno_spectral_filter_is_effective_on_retained_modes():
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    model = FNOBaseline(
        input_dim=3,
        grid_shape=(16, 16),
        seed=3,
        n_modes=6,
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        spectral_cfg={"width": 16, "n_layers": 1, "dealias_ratio": 0.67, "taper_alpha": 4.0},
    )
    spec = model.net.blocks[0].spec
    mask = spec._spectral_filter(
        h=16,
        w_half=(16 // 2) + 1,
        m1=min(spec.modes, 16),
        m2=min(spec.modes, (16 // 2) + 1),
        device=spec.weight_pos.device,
    )
    mask_np = mask.detach().cpu().numpy()
    assert mask_np.shape == (16, 9)
    assert float(mask_np[0, 0]) > 0.999
    assert float(mask_np.min()) < 0.99


def test_fno_skip_filter_match_spectral_reduces_high_frequency():
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    model = FNOBaseline(
        input_dim=3,
        grid_shape=(16, 16),
        seed=4,
        n_modes=6,
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        spectral_cfg={
            "width": 16,
            "n_layers": 1,
            "dealias_ratio": 0.67,
            "taper_alpha": 4.0,
            "skip_filter": "match_spectral",
        },
    )
    torch = model.torch
    block = model.net.blocks[0]
    x = torch.randn(2, 16, 16, 16)
    skip_raw = block.skip(x)
    skip_filt = block.spec.apply_filter_map(skip_raw)
    ft_raw = torch.fft.rfft2(skip_raw, norm="ortho").abs().detach()
    ft_filt = torch.fft.rfft2(skip_filt, norm="ortho").abs().detach()
    assert float(ft_filt.mean()) < float(ft_raw.mean())
