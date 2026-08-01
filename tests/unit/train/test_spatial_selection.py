from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.core.target_groups import TargetGroup
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.train import trainer as trainer_module
from plasma_surrogate.train.selection import (
    SPATIAL_SELECTION_OBJECTIVE_VERSION,
    case_macro_spatial_objective,
    resolve_spatial_selection_config,
)
from plasma_surrogate.train.trainer import Trainer


def _point_only_cfg() -> dict:
    return {
        "spatial": {
            "point_weight": 1.0,
            "gradient_weight": 0.0,
            "boundary_weight": 0.0,
        },
        "case_aggregation": {
            "median_weight": 1.0,
            "p90_weight": 0.0,
            "worst_weight": 0.0,
        },
    }


def test_spatial_selection_reduces_each_case_before_tail_aggregation() -> None:
    target = np.zeros((2, 1, 2, 4), dtype=np.float32)
    pred = target.copy()
    pred[0, 0, :, :] = 1.0
    pred[1, 0, 0, 0] = 3.0
    # The first case has eight active cells and the second only one. A pooled
    # pixel objective would favor the first case; the product objective gives
    # both cases one vote before taking the median.
    mask = np.zeros((2, 2, 4), dtype=bool)
    mask[0] = True
    mask[1, 0, 0] = True

    score, parts = case_macro_spatial_objective(
        pred=pred,
        target=target,
        y_vars=["field"],
        plasma_mask=mask,
        distance_any=None,
        cfg=_point_only_cfg(),
    )

    assert score == pytest.approx(2.0)
    assert parts["spatial_point_median_field"] == pytest.approx(2.0)
    assert parts["selection_objective_version"] == SPATIAL_SELECTION_OBJECTIVE_VERSION


def test_spatial_selection_boundary_term_changes_checkpoint_order() -> None:
    target = np.zeros((1, 1, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)
    distance = np.array(
        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        dtype=np.float32,
    )
    boundary_error = target.copy()
    boundary_error[0, 0, 0, :] = 1.0
    interior_error = target.copy()
    interior_error[0, 0, 1, :] = 1.0
    cfg = {
        "spatial": {
            "point_weight": 0.0,
            "gradient_weight": 0.0,
            "boundary_weight": 1.0,
            "boundary_band_px": 1.0,
        },
        "case_aggregation": {
            "median_weight": 1.0,
            "p90_weight": 0.0,
            "worst_weight": 0.0,
        },
    }

    boundary_score, _ = case_macro_spatial_objective(
        pred=boundary_error,
        target=target,
        y_vars=["field"],
        plasma_mask=mask,
        distance_any=distance,
        cfg=cfg,
    )
    interior_score, _ = case_macro_spatial_objective(
        pred=interior_error,
        target=target,
        y_vars=["field"],
        plasma_mask=mask,
        distance_any=distance,
        cfg=cfg,
    )

    assert boundary_score == pytest.approx(1.0)
    assert interior_score == pytest.approx(0.0)


def test_spatial_selection_target_rms_gradient_is_scale_invariant() -> None:
    target = np.array([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]], dtype=np.float32)
    pred = target * 1.25
    cfg = {
        "spatial": {
            "point_weight": 0.0,
            "gradient_weight": 1.0,
            "gradient_normalization": "target_rms",
            "gradient_epsilon": 1.0e-6,
            "boundary_weight": 0.0,
        },
        "case_aggregation": {
            "median_weight": 1.0,
            "p90_weight": 0.0,
            "worst_weight": 0.0,
        },
    }
    score_a, _ = case_macro_spatial_objective(
        pred=pred,
        target=target,
        y_vars=["field"],
        plasma_mask=np.ones((2, 3), dtype=bool),
        distance_any=None,
        cfg=cfg,
    )
    score_b, _ = case_macro_spatial_objective(
        pred=pred * 10.0,
        target=target * 10.0,
        y_vars=["field"],
        plasma_mask=np.ones((2, 3), dtype=bool),
        distance_any=None,
        cfg=cfg,
    )

    assert score_a == pytest.approx(0.25)
    assert score_b == pytest.approx(score_a)


def test_spatial_selection_averages_target_families_not_family_size() -> None:
    target = np.zeros((1, 3, 1, 1), dtype=np.float32)
    pred = np.array([[[[1.0]], [[3.0]], [[5.0]]]], dtype=np.float32)
    groups = {
        "density": TargetGroup("density", ("ne", "ni"), "test"),
        "thermal": TargetGroup("thermal", ("te",), "test"),
    }

    score, parts = case_macro_spatial_objective(
        pred=pred,
        target=target,
        y_vars=["ne", "ni", "te"],
        plasma_mask=np.ones((1, 1), dtype=bool),
        distance_any=None,
        cfg=_point_only_cfg(),
        groups=groups,
        group_weights={"density": 1.0, "thermal": 1.0},
    )

    assert parts["spatial_group_density"] == pytest.approx(2.0)
    assert parts["spatial_group_thermal"] == pytest.approx(5.0)
    assert score == pytest.approx(3.5)


def test_spatial_selection_default_requires_distance_for_boundary_score() -> None:
    with pytest.raises(ValueError, match="requires distance_any"):
        case_macro_spatial_objective(
            pred=np.zeros((1, 1, 2, 2), dtype=np.float32),
            target=np.zeros((1, 1, 2, 2), dtype=np.float32),
            y_vars=["field"],
            plasma_mask=np.ones((2, 2), dtype=bool),
            distance_any=None,
        )


def test_spatial_selection_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        resolve_spatial_selection_config({"spatial": {"pixel_magic": 1.0}})


@pytest.mark.parametrize(
    "cfg,match",
    [
        (
            {"spatial": {"point_weight": 0.0, "gradient_weight": 0.0, "boundary_weight": 0.0}},
            "spatial weights",
        ),
        (
            {"case_aggregation": {"median_weight": 0.0, "p90_weight": 0.0, "worst_weight": 0.0}},
            "case_aggregation weights",
        ),
    ],
)
def test_spatial_selection_rejects_a_zero_objective(cfg: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_spatial_selection_config(cfg)


def test_spatial_selection_rejects_nonfinite_values_inside_active_mask() -> None:
    pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
    pred[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="inside the active mask"):
        case_macro_spatial_objective(
            pred=pred,
            target=np.zeros_like(pred),
            y_vars=["field"],
            plasma_mask=np.ones((2, 2), dtype=bool),
            distance_any=None,
            cfg=_point_only_cfg(),
        )


def test_unet_spatial_selection_minimizes_with_equal_family_weights(tmp_path, monkeypatch) -> None:
    model = UNetBaseline(
        input_dim=2,
        grid_shape=(3, 3),
        out_channels=2,
        output_keys=["density", "potential"],
        seed=5,
    )
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 3, dtype=np.float32),
        np.linspace(0.0, 1.0, 3, dtype=np.float32),
        indexing="ij",
    )
    model.set_static_spatial_features(np.stack([xx, yy], axis=-1))
    scores = iter([4.0, 1.0, 3.0])
    seen_group_weights: list[dict[str, float]] = []

    def fake_spatial_objective(**kwargs):
        seen_group_weights.append(dict(kwargs["group_weights"] or {}))
        score = next(scores)
        return score, {
            "spatial_density": score + 0.1,
            "spatial_potential": score + 0.2,
            "spatial_group_density": score + 0.1,
            "spatial_group_electrostatic": score + 0.2,
        }

    monkeypatch.setattr(trainer_module, "case_macro_spatial_objective", fake_spatial_objective)
    cond = np.zeros((4, 2), dtype=np.float32)
    target = np.ones((4, 2, 3, 3), dtype=np.float32)
    out = Trainer(tmp_path / "unet_spatial").run_unet(
        model,
        cond[:2],
        target[:2],
        cond[2:],
        target[2:],
        epochs=3,
        lr=1.0e-2,
        physics_cfg={"enabled": False},
        supervised_mask=np.ones((3, 3), dtype=np.float32),
        supervised_distance=np.ones((3, 3), dtype=np.float32),
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

    assert seen_group_weights == [
        {"density": 0.5, "electrostatic": 0.5},
        {"density": 0.5, "electrostatic": 0.5},
        {"density": 0.5, "electrostatic": 0.5},
    ]
    assert out.history[1]["selected_epoch_flag"] == 1.0
    assert out.history[-1]["selected_epoch_score"] == 1.0
    assert out.history[1]["selection_score_density"] == 1.1
    assert out.history[1]["selection_score_group_electrostatic"] == 1.2
