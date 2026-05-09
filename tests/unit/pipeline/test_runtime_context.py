from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_surrogate.cli.workflows import run_evaluate, run_infer, run_preprocess, run_train
from plasma_surrogate.pipeline.runtime_context import (
    build_infer_context,
    build_preprocess_context,
    build_train_context,
)
from tests._config_presets import default_target_transforms_ne_ni_te_phi, runtime_table_only


def _write_cfg(path: Path, run_dir: Path) -> None:
    cfg = {
        "run_dir": str(run_dir),
        "runtime": runtime_table_only(),
        "dataset": {"type": "synthetic", "n_cases": 8, "height": 8, "width": 8, "cond_dim": 3, "seed": 3},
        "preprocessing": {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "axis_schema": {"mode": "steady", "harmonics": 1},
            "scalers": {"target_transforms": default_target_transforms_ne_ni_te_phi()},
        },
        "model": {"name": "global_mlp", "phi_mode": "direct"},
        "train": {"epochs": 2, "lr": 0.01},
        "inference": {"single": {"enabled": False}, "batch": {"enabled": False}, "optimize": {"enabled": False}},
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def test_build_preprocess_context_returns_dataset_and_feature_meta(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    run_dir = tmp_path / "run"
    _write_cfg(cfg_path, run_dir)

    ctx = build_preprocess_context(cfg_path)
    assert ctx.run_dir == run_dir
    assert ctx.axis_mode == "steady"
    assert len(ctx.dataset.cases) == 8
    assert "feature_hash" in ctx.feature_meta


def test_build_train_context_bootstraps_preprocess(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    run_dir = tmp_path / "run"
    _write_cfg(cfg_path, run_dir)

    ctx = build_train_context(cfg_path, on_missing_preprocess=lambda: run_preprocess(cfg_path))
    assert ctx.bundle is not None
    split = ctx.bundle.split_random()
    assert set(split.keys()) == {"train", "val", "test"}


def test_build_infer_context_bootstraps_checkpoint(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    run_dir = tmp_path / "run"
    _write_cfg(cfg_path, run_dir)

    ctx = build_infer_context(
        cfg_path,
        on_missing_preprocess=lambda: run_preprocess(cfg_path),
        on_missing_checkpoint=lambda: run_train(cfg_path),
    )
    assert ctx.bundle is not None
    assert ctx.bundle.model is not None


def test_runtime_metadata_contract_flows_to_checkpoint_infer_and_evaluate(
    tmp_path: Path,
    assert_input_mode_metadata_keys,
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    run_dir = tmp_path / "run"
    _write_cfg(cfg_path, run_dir)

    run_preprocess(cfg_path)
    run_train(cfg_path)

    checkpoint_meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    assert_input_mode_metadata_keys(checkpoint_meta)
    assert checkpoint_meta["input_mode_effective"] == "table_only"
    assert checkpoint_meta["structure_feature_profile_effective"] == "none"
    assert checkpoint_meta["structure_descriptor_profile_effective"] == "none"
    assert checkpoint_meta["structure_latent_profile_effective"] == "none"
    assert checkpoint_meta["has_structure_inputs_effective"] is False

    infer_out = run_infer(cfg_path)
    infer_summary = json.loads(Path(infer_out["summary"]).read_text(encoding="utf-8"))
    assert_input_mode_metadata_keys(infer_summary)

    evaluate_out = run_evaluate(cfg_path)
    evaluate_summary = json.loads(Path(evaluate_out["summary"]).read_text(encoding="utf-8"))
    assert_input_mode_metadata_keys(evaluate_summary)
