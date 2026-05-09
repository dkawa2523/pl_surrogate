from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasma_surrogate.core.input_modes import build_input_mode_effective_metadata, input_mode_metadata_keys
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.checkpoint import save_checkpoint


def test_save_checkpoint_rejects_missing_required_input_mode_meta_keys(tmp_path: Path) -> None:
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0)
    with pytest.raises(ValueError, match="missing required input-mode keys"):
        save_checkpoint(
            model,
            tmp_path / "ckpt",
            extra_meta={"input_mode_effective": "table_only"},
        )


def test_save_checkpoint_accepts_required_input_mode_meta_keys(tmp_path: Path) -> None:
    model = GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0)
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": "table_only",
                "structure": {
                    "feature_profile": "none",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    ckpt_dir = save_checkpoint(
        model,
        tmp_path / "ckpt",
        extra_meta=meta,
    )
    loaded = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    assert loaded["input_mode_effective"] == "table_only"
    assert loaded["structure_adapter_mode_effective"] == "none"
    for key in input_mode_metadata_keys():
        assert key in loaded
