from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.features.structure_feature_registry import resolve_spatial_channels_for_feature_profile
from plasma_surrogate.preprocessing.runner import PreprocessRunner


def test_table_only_runtime_cfg_rejects_non_none_structure_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="table_only"):
        PreprocessRunner(
            cfg={},
            output_dir=tmp_path / "pre",
            runtime_cfg={
                "input_mode": "table_only",
                "structure": {
                    "feature_profile": "part_lite_v1",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                },
            },
        )


def test_table_only_rejects_channels_from_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="channels_from_profile"):
        PreprocessRunner(
            cfg={"coord_features": {"channels_from_profile": "part_lite_v1"}},
            output_dir=tmp_path / "pre",
            runtime_cfg={
                "input_mode": "table_only",
                "structure": {
                    "feature_profile": "none",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                },
            },
        )


def test_table_only_populates_effective_metadata(tmp_path: Path) -> None:
    pre = PreprocessRunner(
        cfg={},
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_only",
            "structure": {
                "feature_profile": "none",
                "descriptor_profile": "none",
                "latent_profile": "none",
                "adapter_mode": "none",
                "provider_mode": "fixed",
            },
        },
    )
    assert pre.runtime_input_mode_meta["input_mode_effective"] == "table_only"
    assert pre.runtime_input_mode_meta["has_structure_inputs_effective"] is False


def test_table_plus_structure_rejects_direct_channels(tmp_path: Path) -> None:
    pre = PreprocessRunner(
        cfg={},
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "part_lite_v1",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        },
    )
    with pytest.raises(ValueError, match="does not allow preprocessing.coord_features.channels"):
        pre._resolve_coord_feature_channels({"channels": ["x", "y"]})  # noqa: SLF001


def test_table_plus_structure_rejects_profile_mismatch(tmp_path: Path) -> None:
    pre = PreprocessRunner(
        cfg={},
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "part_lite_v1",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        },
    )
    with pytest.raises(ValueError, match="must match runtime.structure.feature_profile"):
        pre._resolve_coord_feature_channels({"channels_from_profile": "boundary_plus_v1"})  # noqa: SLF001


def test_table_plus_structure_resolves_channels_from_runtime_profile(tmp_path: Path) -> None:
    pre = PreprocessRunner(
        cfg={},
        output_dir=tmp_path / "pre",
        runtime_cfg={
            "input_mode": "table_plus_structure",
            "structure": {
                "feature_profile": "geom_v1_mainline",
                "descriptor_profile": "none",
                "latent_profile": "none",
            },
        },
    )
    channels = pre._resolve_coord_feature_channels({})  # noqa: SLF001
    assert channels == list(resolve_spatial_channels_for_feature_profile("geom_v1_mainline"))
