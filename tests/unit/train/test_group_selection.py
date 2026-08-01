from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.core.target_groups import TargetGroup
from plasma_surrogate.train import trainer as trainer_module
from plasma_surrogate.train.selection import (
    group_plasma_balance_score,
    resolve_selection_group_weights,
    resolve_selection_target_groups,
)
from plasma_surrogate.train.target_contracts import validate_mainline_selection_contract
from plasma_surrogate.train.trainer import Trainer


def _schema() -> dict:
    return {
        "targets": [
            {"id": "electron_density", "role": "density_electron", "field_family": "density"},
            {"id": "ion_density", "role": "density_ion", "field_family": "density"},
            {"id": "temperature", "role": "temperature", "field_family": "temperature"},
            {"id": "potential", "role": "potential", "field_family": "electrostatic"},
        ]
    }


def test_group_selection_score_uses_group_mean_then_group_weight() -> None:
    y_vars = ["electron_density", "ion_density", "temperature", "potential"]
    groups = resolve_selection_target_groups(output_vars=y_vars, target_role_schema=_schema())
    weights = resolve_selection_group_weights(None, groups=groups, cfg_prefix="train.fno")
    target = np.array(
        [
            [
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )
    pred = target.copy()
    pred[:, 1] = 0.0
    mask = np.ones((1, 1, 2), dtype=bool)

    score, parts = group_plasma_balance_score(
        pred=pred,
        target=target,
        y_vars=y_vars,
        plasma_mask=mask,
        groups=groups,
        group_weights=weights,
    )

    assert parts["r2_electron_density_plasma"] == 1.0
    assert parts["r2_ion_density_plasma"] == -1.0
    assert parts["r2_group_density_plasma"] == 0.0
    assert parts["r2_group_temperature_plasma"] == 1.0
    assert parts["r2_group_electrostatic_plasma"] == 1.0
    assert np.isclose(score, 2.0 / 3.0)


def test_group_selection_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="target_role_schema.targets"):
        resolve_selection_target_groups(output_vars=["a"], target_role_schema={})


def test_group_selection_filters_full_schema_to_the_model_output_subset() -> None:
    groups = resolve_selection_target_groups(
        output_vars=["electron_density", "temperature"],
        target_role_schema=_schema(),
    )

    assert {name: group.targets for name, group in groups.items()} == {
        "density": ("electron_density",),
        "temperature": ("temperature",),
    }


def test_group_selection_rejects_incomplete_or_overlapping_model_groups() -> None:
    with pytest.raises(ValueError, match="cover every output exactly once"):
        resolve_selection_target_groups(
            output_vars=["a", "b"],
            model_target_groups={"only_a": TargetGroup("only_a", ("a",), "test")},
        )
    with pytest.raises(ValueError, match="cover every output exactly once"):
        resolve_selection_target_groups(
            output_vars=["a", "b"],
            model_target_groups={
                "first": TargetGroup("first", ("a", "b"), "test"),
                "second": TargetGroup("second", ("a",), "test"),
            },
        )


def test_group_selection_weights_are_strict() -> None:
    groups = resolve_selection_target_groups(
        output_vars=["electron_density", "ion_density", "temperature", "potential"],
        target_role_schema=_schema(),
    )
    with pytest.raises(ValueError, match="unknown groups"):
        resolve_selection_group_weights({"bad": 1.0}, groups=groups, cfg_prefix="train.fno")
    with pytest.raises(ValueError, match="missing groups"):
        resolve_selection_group_weights({"density": 1.0}, groups=groups, cfg_prefix="train.fno")


def test_mainline_selection_contract_accepts_group_balance_mode() -> None:
    weights = validate_mainline_selection_contract(
        selection_cfg={"mode": "best_val_group_balance"},
        target_vars=["electron_density", "ion_density"],
        cfg_prefix="train.fno",
    )

    assert weights == {"electron_density": 0.5, "ion_density": 0.5}


def test_mainline_selection_contract_accepts_spatial_objective_mode() -> None:
    weights = validate_mainline_selection_contract(
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "spatial": {
                "point_weight": 1.0,
                "gradient_weight": 0.2,
                "boundary_weight": 0.0,
            },
        },
        target_vars=["electron_density", "ion_density"],
        cfg_prefix="train.fno",
    )

    assert weights == {"electron_density": 0.5, "ion_density": 0.5}


def test_unet_group_selection_restores_the_best_checkpoint(tmp_path, monkeypatch) -> None:
    model = UNetBaseline(
        input_dim=2,
        grid_shape=(3, 3),
        out_channels=2,
        output_keys=["electron_density", "temperature"],
        seed=3,
    )
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 3, dtype=np.float32),
        np.linspace(0.0, 1.0, 3, dtype=np.float32),
        indexing="ij",
    )
    model.set_static_spatial_features(np.stack([xx, yy], axis=-1))
    scores = iter([2.0, 1.0])

    def fake_group_score(**_kwargs):
        score = next(scores)
        return score, {
            "r2_electron_density_plasma": score,
            "r2_temperature_plasma": score,
            "r2_group_density_plasma": score,
            "r2_group_temperature_plasma": score,
        }

    restored: list[dict[str, np.ndarray]] = []
    original_restore = trainer_module._restore_model_state_numpy

    def spy_restore(target_model, state):
        restored.append(state)
        original_restore(target_model, state)

    monkeypatch.setattr(trainer_module, "group_plasma_balance_score", fake_group_score)
    monkeypatch.setattr(trainer_module, "_restore_model_state_numpy", spy_restore)
    cond = np.zeros((4, 2), dtype=np.float32)
    target = np.ones((4, 2, 3, 3), dtype=np.float32)
    out = Trainer(tmp_path / "group_restore").run_unet(
        model,
        cond[:2],
        target[:2],
        cond[2:],
        target[2:],
        epochs=2,
        lr=1.0e-2,
        physics_cfg={"enabled": False},
        selection_cfg={
            "mode": "best_val_group_balance",
            "warmup_epochs": 0,
            "eval_every_n_epochs": 1,
            "target_role_schema": {
                "targets": [
                    {"id": "electron_density", "field_family": "density"},
                    {"id": "temperature", "field_family": "temperature"},
                ]
            },
        },
    )

    assert len(restored) == 1
    assert out.history[0]["selected_epoch_flag"] == 1.0
    assert out.history[-1]["selection_mode_effective"] == "best_val_group_balance"
