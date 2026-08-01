from __future__ import annotations

import numpy as np

from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.train.trainer import _is_effective_clip, train_one_epoch_global_physics


def _make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    cond = rng.normal(size=(6, 3)).astype(np.float32)
    y_field = rng.normal(size=(6, 3, 4, 4)).astype(np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:3, :] = 1.0
    dist = np.ones((4, 4), dtype=np.float32)
    return cond, y_field, mask, dist


def test_global_grad_scale_auto_active_pixels_applies_scale_gt_one():
    cond, y_field, mask, dist = _make_inputs(1)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=1, hidden=[16, 16], dropout=0.0)
    stats = train_one_epoch_global_physics(
        model,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
        grad_scale_cfg={"mode": "auto", "auto_ref": "active_pixels"},
    )
    assert float(stats["grad_scale_applied"]) > 1.0


def test_global_grad_scale_auto_power_half_uses_sqrt_active_pixels():
    cond, y_field, mask, dist = _make_inputs(11)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=11, hidden=[16, 16], dropout=0.0)
    stats = train_one_epoch_global_physics(
        model,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
        grad_scale_cfg={"mode": "auto", "auto_ref": "active_pixels", "auto_power": 0.5},
    )
    expected = float(np.sqrt(np.sum(mask > 0.5)))
    assert np.isclose(float(stats["grad_scale_applied"]), expected, rtol=1e-5, atol=1e-5)


def test_global_grad_clip_reduces_gradient_norm():
    cond, y_field, mask, dist = _make_inputs(2)
    model_no_clip = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=4, hidden=[16, 16], dropout=0.0)
    model_clip = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=4, hidden=[16, 16], dropout=0.0)

    stats_no_clip = train_one_epoch_global_physics(
        model_no_clip,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
        grad_scale_cfg={"mode": "fixed", "fixed_value": 500.0},
    )
    stats_clip = train_one_epoch_global_physics(
        model_clip,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
        grad_scale_cfg={"mode": "fixed", "fixed_value": 500.0},
        grad_clip_cfg={"mode": "global", "norm": 0.25},
    )
    assert float(stats_clip["grad_l2_total"]) <= float(stats_no_clip["grad_l2_total"])


def test_global_grad_clip_per_sample_applies_and_reports_clip_ratio():
    cond, y_field, mask, dist = _make_inputs(12)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=12, hidden=[16, 16], dropout=0.0)
    stats = train_one_epoch_global_physics(
        model,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
        grad_scale_cfg={"mode": "fixed", "fixed_value": 500.0},
        grad_clip_cfg={"mode": "per_sample", "norm": 0.25, "adaptive_by_dim": False},
    )
    assert float(stats["grad_norm_post_clip"]) <= float(stats["grad_norm_post_scale"])
    assert float(stats["clip_ratio"]) <= 1.0


def test_global_default_grad_scale_is_off_mode():
    cond, y_field, mask, dist = _make_inputs(3)
    model = GlobalMLP(input_dim=3, grid_shape=(4, 4), out_channels=3, seed=6, hidden=[16, 16], dropout=0.0)
    stats = train_one_epoch_global_physics(
        model,
        cond,
        y_field,
        lr=1e-3,
        physics_cfg={"enabled": False},
        supervised_mask=mask,
        supervised_distance=dist,
    )
    assert np.isclose(float(stats["grad_scale_applied"]), 1.0)


def test_is_effective_clip_lt_direction():
    assert _is_effective_clip(clip_ratio=0.8, threshold=0.99, direction="lt") is True
    assert _is_effective_clip(clip_ratio=1.0, threshold=0.99, direction="lt") is False


def test_is_effective_clip_gt_direction():
    assert _is_effective_clip(clip_ratio=1.0, threshold=0.99, direction="gt") is True
    assert _is_effective_clip(clip_ratio=0.8, threshold=0.99, direction="gt") is False
