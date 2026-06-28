from __future__ import annotations

import csv
import numpy as np

from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.train import trainer as trainer_module
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
    model = GlobalMLP(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=3,
        output_keys=["density", "temperature", "potential"],
        seed=2,
    )
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


def test_run_global_allvars_selection_marks_best_epoch(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 3)).astype(np.float32)
    y = rng.normal(size=(12, 4 * 4 * 4)).astype(np.float32)
    model = GlobalMLP(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        hidden=[8],
        dropout=0.0,
        seed=7,
    )
    calls: list[int] = []

    def fake_balance_score(**_kwargs):
        calls.append(1)
        score = 10.0 if len(calls) == 1 else 1.0
        return score, {"r2_ne_plasma": score}

    monkeypatch.setattr(trainer_module, "allvars_plasma_balance_score", fake_balance_score)
    out = Trainer(tmp_path / "train_select").run_global(
        model,
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        epochs=3,
        lr=1e-2,
        loss_cfg={"supervised": {"type": "mse", "mask": "plasma_only"}},
        supervised_mask=np.ones((4, 4), dtype=np.float32),
        selection_cfg={
            "mode": "best_val_allvars_balance",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
        },
    )

    assert len(calls) == 3
    assert out.history[0]["selected_epoch_flag"] == 1.0
    assert out.history[-1]["selection_valid_flag"] == 1.0
    assert out.history[-1]["selection_mode_effective"] == "best_val_allvars_balance"
