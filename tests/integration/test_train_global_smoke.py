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


def test_run_global_spatial_selection_minimizes_and_restores_best_epoch(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(10, 2)).astype(np.float32)
    y = rng.normal(size=(10, 2 * 3 * 3)).astype(np.float32)
    model = GlobalMLP(
        input_dim=2,
        grid_shape=(3, 3),
        out_channels=2,
        output_keys=["density", "potential"],
        hidden=[4],
        dropout=0.0,
        seed=11,
    )
    scores = iter([3.0, 1.0, 2.0])
    seen_distance: list[np.ndarray] = []
    seen_group_weights: list[dict[str, float]] = []

    def fake_spatial_objective(**kwargs):
        seen_distance.append(np.asarray(kwargs["distance_any"], dtype=np.float32))
        seen_group_weights.append(dict(kwargs["group_weights"] or {}))
        score = next(scores)
        return score, {
            "spatial_density": score + 0.1,
            "spatial_potential": score + 0.2,
            "spatial_group_density": score + 0.1,
            "spatial_group_electrostatic": score + 0.2,
        }

    restored: list[dict[str, np.ndarray]] = []
    original_restore = trainer_module._restore_model_state_numpy

    def spy_restore(target_model, state):
        restored.append(state)
        original_restore(target_model, state)

    monkeypatch.setattr(trainer_module, "case_macro_spatial_objective", fake_spatial_objective)
    monkeypatch.setattr(trainer_module, "_restore_model_state_numpy", spy_restore)
    distance = np.ones((3, 3), dtype=np.float32)
    out = Trainer(tmp_path / "train_spatial_select").run_global(
        model,
        x[:7],
        y[:7],
        x[7:],
        y[7:],
        epochs=3,
        lr=1e-2,
        supervised_mask=np.ones((3, 3), dtype=np.float32),
        supervised_distance=distance,
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "target_role_schema": {
                "targets": [
                    {"id": "density", "field_family": "density"},
                    {"id": "potential", "field_family": "electrostatic"},
                ]
            },
        },
    )

    assert len(seen_distance) == 3
    assert all(item.shape == (3, *distance.shape) for item in seen_distance)
    assert all(np.array_equal(item, np.broadcast_to(distance, item.shape)) for item in seen_distance)
    assert seen_group_weights == [
        {"density": 0.5, "electrostatic": 0.5},
        {"density": 0.5, "electrostatic": 0.5},
        {"density": 0.5, "electrostatic": 0.5},
    ]
    assert out.history[1]["selected_epoch_flag"] == 1.0
    assert out.history[-1]["selected_epoch_score"] == 1.0
    assert out.history[-1]["selection_mode_effective"] == "best_val_spatial_objective"
    assert out.history[-1]["selection_objective_version"] == "case_macro_spatial_rmse_v2"
    assert out.history[1]["selection_score_density"] == 1.1
    assert out.history[1]["selection_score_group_electrostatic"] == 1.2
    assert len(restored) == 1


def test_run_global_group_selection_resolves_and_passes_complete_groups(tmp_path, monkeypatch):
    model = GlobalMLP(
        input_dim=2,
        grid_shape=(3, 3),
        out_channels=2,
        output_keys=["density", "potential"],
        hidden=[4],
        dropout=0.0,
        seed=13,
    )
    seen: list[tuple[set[str], dict[str, float]]] = []

    def fake_group_score(**kwargs):
        groups = kwargs["groups"]
        weights = dict(kwargs["group_weights"])
        seen.append((set(groups), weights))
        return 1.0, {
            "r2_density_plasma": 1.0,
            "r2_potential_plasma": 1.0,
            "r2_group_density_plasma": 1.0,
            "r2_group_electrostatic_plasma": 1.0,
        }

    monkeypatch.setattr(trainer_module, "group_plasma_balance_score", fake_group_score)
    cond = np.zeros((4, 2), dtype=np.float32)
    target = np.ones((4, 2 * 3 * 3), dtype=np.float32)
    out = Trainer(tmp_path / "global_group_select").run_global(
        model,
        cond[:2],
        target[:2],
        cond[2:],
        target[2:],
        epochs=1,
        lr=1.0e-2,
        selection_cfg={
            "mode": "best_val_group_balance",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "target_role_schema": {
                "targets": [
                    {"id": "density", "field_family": "density"},
                    {"id": "potential", "field_family": "electrostatic"},
                ]
            },
        },
    )

    assert seen == [
        (
            {"density", "electrostatic"},
            {"density": 0.5, "electrostatic": 0.5},
        )
    ]
    assert out.history[-1]["selection_mode_effective"] == "best_val_group_balance"
    assert out.history[-1]["selection_valid_flag"] == 1.0
