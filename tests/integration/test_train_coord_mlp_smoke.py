from __future__ import annotations

import pytest
import numpy as np


from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.train.trainer import Trainer

pytestmark = pytest.mark.torch_runtime


@pytest.mark.parametrize(
    ("model_name", "model_cfg"),
    [
        (
            "coord_mlp_fourier",
            {
                "cond_hidden": [16, 16],
                "latent_dim": 16,
                "decoder_hidden": [32, 32],
                "decoder_activation": "gelu",
                "embedding": {"type": "fourier", "n_frequencies": 4, "include_raw": True, "frequency_scale": 10.0},
            },
        ),
        (
            "coord_mlp_siren",
            {
                "cond_hidden": [16, 16],
                "latent_dim": 16,
                "decoder_hidden": [32, 32],
                "embedding": {"type": "none"},
                "siren": {
                    "enabled": True,
                    "fusion": "split_add",
                    "w0_initial": 10.0,
                    "w0_hidden": 1.0,
                    "fusion_cfg": {"cond_gain_init": 0.7, "point_gain_init": 1.3, "branch_norm": True},
                },
            },
        ),
    ],
)
def test_train_coord_mlp_smoke(tmp_path, model_name: str, model_cfg: dict[str, object]) -> None:
    require_torch_runtime()

    rng = np.random.default_rng(11)
    n, d, h, w = 12, 3, 8, 8
    cond = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    spatial = np.stack(
        [
            xv,
            yv,
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.25, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    y = np.zeros((n, 4, h, w), dtype=np.float32)
    for i in range(n):
        y[i, 0] = cond[i, 0] + 0.15 * xv
        y[i, 1] = cond[i, 1] + 0.10 * yv
        y[i, 2] = cond[i, 2] + 0.08 * np.sin(np.pi * (xv + yv))
        y[i, 3] = (cond[i, 0] - cond[i, 1]) + 0.05 * np.cos(np.pi * xv)

    tr = np.arange(0, 8)
    va = np.arange(8, 12)
    model = CoordMLPTorch(
        input_dim=d,
        grid_shape=(h, w),
        out_channels=4,
        output_keys=["density_e", "density_i", "temperature", "potential"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=model_cfg,
    )
    model.set_static_spatial_features(spatial)
    trainer = Trainer(tmp_path / f"train_{model_name}")
    out = trainer.run_unet(
        model,
        cond[tr],
        y[tr],
        cond[va],
        y[va],
        epochs=4,
        lr=1e-2,
        physics_cfg={"enabled": False},
    )
    losses = [float(item["train_loss"]) for item in out.history if "train_loss" in item]
    assert all(np.isfinite(v) for v in losses)
    assert min(losses) <= losses[0]
    pred = model.forward(cond[va], spatial_features=np.repeat(spatial[None, ...], len(va), axis=0))
    assert np.all(np.isfinite(pred))
