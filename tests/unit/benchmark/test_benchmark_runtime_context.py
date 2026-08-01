from __future__ import annotations

from pathlib import Path

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.benchmark.runtime_context import build_benchmark_data_context
from tests._config_presets import default_target_transforms_four_field_example, runtime_table_only


def test_build_benchmark_data_context_shapes_and_split(tmp_path: Path):
    output_root = tmp_path / "bench"
    cfg = {
        "output_dir": str(output_root),
        "dataset": {"type": "synthetic", "n_cases": 9, "height": 8, "width": 8, "cond_dim": 3, "seed": 4},
        "profile": "m7_global_frozen_ref",
        "phi_mode": "direct",
        "runtime": runtime_table_only(),
        "split": {"seed": 2, "ratios": [0.6, 0.2, 0.2]},
        "preprocessing": {
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
    }

    profile_lock = BenchmarkRunner._resolve_profile_lock("m7_global_frozen_ref")
    ctx = build_benchmark_data_context(cfg=cfg, output_root=output_root, profile_lock=profile_lock)

    assert ctx.cond.shape[0] == 9
    assert ctx.y.shape == (9, 4, 8, 8)
    assert ctx.cond_scaled.shape == ctx.cond.shape
    assert ctx.y_scaled.shape == ctx.y.shape
    assert set(ctx.split.keys()) == {"train", "val", "test"}
    assert ctx.resolved["artifact_hashes"]["lock_hash"] == ctx.lock_hash


def test_build_benchmark_data_context_honors_benchmark_axis_mode_override(tmp_path: Path):
    cfg = {
        "output_dir": str(tmp_path / "bench"),
        "dataset": {"type": "synthetic", "n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 0},
        "profile": "m7_global_frozen_ref",
        "phi_mode": "direct",
        "axis_mode": "time",
        "runtime": runtime_table_only(),
        "preprocessing": {
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
    }
    # Profile lock requires steady for m7_global_frozen_ref; runner should block before context builder.
    profile_lock = BenchmarkRunner._resolve_profile_lock("m7_global_frozen_ref")
    assert profile_lock["axis_mode"] == "steady"
    # Context builder follows cfg.axis_mode override.
    ctx = build_benchmark_data_context(cfg=cfg, output_root=Path(cfg["output_dir"]), profile_lock=profile_lock)
    assert ctx.requested_axis_mode == "time"
