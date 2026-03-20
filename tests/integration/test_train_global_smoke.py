from __future__ import annotations

import csv
import numpy as np

from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.train.trainer import Trainer, train_one_epoch_global


def test_train_global_smoke():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    target = rng.normal(size=(32, 3 * 4 * 4)).astype(np.float32)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), seed=0)

    loss0 = train_one_epoch_global(model, x, target, lr=5e-2)
    loss1 = train_one_epoch_global(model, x, target, lr=5e-2)
    assert loss1 <= loss0


def test_run_global_saves_optimization_diagnostics(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 3)).astype(np.float32)
    y = rng.normal(size=(12, 3 * 4 * 4)).astype(np.float32)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, output_keys=["log_ne", "Te", "phi"], seed=2)
    trainer = Trainer(tmp_path / "train")
    out = trainer.run_global(
        model,
        x[:8],
        y[:8],
        x[8:10],
        y[8:10],
        epochs=2,
        lr=1e-2,
        batch_size_cases=3,
        shuffle_cases=True,
        seed=5,
        loss_cfg={"supervised": {"normalization": "sample_mean"}},
    )
    assert len(out.history) == 2
    diag_csv = tmp_path / "train" / "scalars" / "optimization_diagnostics.csv"
    assert diag_csv.exists()
    with diag_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "step_rel_hidden_mean" in rows[0]
    assert "step_rel_output" in rows[0]
    assert "grad_scale_applied" in rows[0]
    assert "clip_ratio" in rows[0]
    assert "step_rel_output_to_hidden" in rows[0]
