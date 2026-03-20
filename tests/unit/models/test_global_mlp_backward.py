from __future__ import annotations

import numpy as np

from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.train.trainer import train_one_epoch_global


def test_global_mlp_one_step_reduces_loss():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 3)).astype(np.float32)
    # simple deterministic target
    y_target = np.repeat((0.3 * x[:, :1] - 0.1 * x[:, 1:2] + 0.2 * x[:, 2:3]), 4, axis=1).astype(np.float32)

    model = GlobalMLP(
        input_dim=3,
        grid_shape=(2, 2),
        out_channels=1,
        seed=7,
        hidden=[16, 16],
        dropout=0.0,
    )

    loss0 = train_one_epoch_global(model, x, y_target, lr=5e-3)
    loss1 = train_one_epoch_global(model, x, y_target, lr=5e-3)
    assert loss1 < loss0


def test_global_mlp_backward_layer_multiplier_updates_output_more():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8, 3)).astype(np.float32)
    model = GlobalMLP(
        input_dim=3,
        grid_shape=(2, 2),
        out_channels=1,
        seed=3,
        hidden=[8, 8],
        dropout=0.0,
    )
    pred = model.forward(x, training=True)
    grad = np.ones_like(pred, dtype=np.float32) * 0.01
    diag = model.backward(
        grad,
        lr=1e-3,
        layer_lr_multipliers={"hidden": 1.0, "output": 10.0},
    )
    assert float(diag["step_rel_output"]) > float(diag["step_rel_hidden_mean"])


def test_global_mlp_fit_output_layer_ridge_reduces_loss():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    y_target = np.repeat((0.4 * x[:, :1] + 0.1 * x[:, 1:2] - 0.2 * x[:, 2:3]), 4, axis=1).astype(np.float32)
    model = GlobalMLP(
        input_dim=3,
        grid_shape=(2, 2),
        out_channels=1,
        seed=8,
        hidden=[16, 16],
        dropout=0.0,
    )
    pred0 = model.forward(x, training=False)
    loss0 = float(np.mean((pred0 - y_target) ** 2))
    hidden = model.extract_hidden(x)
    cond = model.fit_output_layer_ridge(hidden, y_target, ridge=1e-4)
    pred1 = model.forward(x, training=False)
    loss1 = float(np.mean((pred1 - y_target) ** 2))
    assert np.isfinite(cond)
    assert loss1 < loss0
