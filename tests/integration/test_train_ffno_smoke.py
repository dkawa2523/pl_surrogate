from __future__ import annotations

import os

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.train.trainer import Trainer


def _synthetic_grid_regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
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
    return cond, y, tr, va, h, w


def _spectral_cfg() -> dict[str, object]:
    return {
        "width": 16,
        "n_layers": 2,
        "dealias_ratio": 0.67,
        "taper_alpha": 4.0,
        "skip_filter": "match_spectral",
    }


@pytest.mark.parametrize("local_skip_enabled", [False, True])
def test_train_ffno_smoke(tmp_path, local_skip_enabled: bool):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    cond, y, tr, va, h, w = _synthetic_grid_regression_data()
    model = FFNOBaseline(
        input_dim=cond.shape[1],
        grid_shape=(h, w),
        seed=2,
        n_modes=2,
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        spectral_cfg={
            **_spectral_cfg(),
            "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
            "local_skip_cfg": {"enabled": bool(local_skip_enabled), "init_scale": 0.0},
        },
    )
    trainer = Trainer(tmp_path / "train_ffno")
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


def test_train_ffno_and_fno_share_minimal_training_contract(tmp_path):
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    cond, y, tr, va, h, w = _synthetic_grid_regression_data()
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    models = {
        "fno": FNOBaseline(
            input_dim=cond.shape[1],
            grid_shape=(h, w),
            seed=3,
            n_modes=2,
            input_feature_channels=channels,
            spectral_cfg=_spectral_cfg(),
        ),
        "ffno": FFNOBaseline(
            input_dim=cond.shape[1],
            grid_shape=(h, w),
            seed=4,
            n_modes=2,
            input_feature_channels=channels,
            spectral_cfg={
                **_spectral_cfg(),
                "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
            },
        ),
    }
    for model_name, model in models.items():
        trainer = Trainer(tmp_path / f"train_{model_name}")
        out = trainer.run_unet(
            model,
            cond[tr],
            y[tr],
            cond[va],
            y[va],
            epochs=2,
            lr=5e-3,
            physics_cfg={"enabled": False},
        )
        pred = model.forward(cond[va])
        assert out.history[-1]["train_loss"] <= out.history[0]["train_loss"]
        assert pred.shape == (len(va), 3, h, w)
        assert np.isfinite(pred).all()
