from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.core.input_modes import (
    INPUT_MODE_EFFECTIVE_KEY,
    build_input_mode_effective_metadata,
)
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def _table_only_meta() -> dict[str, object]:
    return build_input_mode_effective_metadata(
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


def test_inference_engine_accepts_matching_input_mode_metadata(tmp_path: Path, geometry_root: Path) -> None:
    meta = _table_only_meta()
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        input_mode="table_only",
        input_mode_meta=meta,
        checkpoint_input_mode_meta=dict(meta),
    )
    assert engine.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY] == "table_only"


def test_inference_engine_rejects_checkpoint_input_mode_mismatch(tmp_path: Path, geometry_root: Path) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta["structure_adapter_mode_effective"] = "auto"
    with pytest.raises(ValueError, match="input-mode metadata mismatch"):
        InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
            cond_schema=CondSchema(order=["c0", "c1", "c2"]),
            axis_schema=AxisSchema(mode="steady"),
            geometry_provider=FixedGeometryProvider(geometry_root),
            output_dir=tmp_path / "infer",
            input_mode="table_only",
            input_mode_meta=request_meta,
            checkpoint_input_mode_meta=checkpoint_meta,
        )


def test_inference_engine_allows_checkpoint_fallback_on_mismatch_when_enabled(
    tmp_path: Path,
    geometry_root: Path,
) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta["structure_adapter_mode_effective"] = "auto"
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        input_mode="table_only",
        strict_input_mode="error",
        allow_mode_fallback=True,
        input_mode_meta=request_meta,
        checkpoint_input_mode_meta=checkpoint_meta,
    )
    assert engine.input_mode_fallback_applied is True
    assert engine.input_mode_meta["structure_adapter_mode_effective"] == "auto"


def test_inference_engine_warn_mode_does_not_raise_on_mismatch(
    tmp_path: Path,
    geometry_root: Path,
) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta["structure_adapter_mode_effective"] = "auto"
    with pytest.warns(RuntimeWarning, match="input-mode metadata mismatch"):
        engine = InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
            cond_schema=CondSchema(order=["c0", "c1", "c2"]),
            axis_schema=AxisSchema(mode="steady"),
            geometry_provider=FixedGeometryProvider(geometry_root),
            output_dir=tmp_path / "infer",
            input_mode="table_only",
            strict_input_mode="warn",
            allow_mode_fallback=False,
            input_mode_meta=request_meta,
            checkpoint_input_mode_meta=checkpoint_meta,
        )
    assert engine.input_mode_fallback_applied is False
    assert engine.input_mode_meta["structure_adapter_mode_effective"] == "none"


def test_inference_engine_rejects_checkpoint_input_mode_missing_key(tmp_path: Path, geometry_root: Path) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta.pop("structure_feature_profile_effective")
    with pytest.raises(ValueError, match="missing required key"):
        InferenceEngine(
            model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
            cond_schema=CondSchema(order=["c0", "c1", "c2"]),
            axis_schema=AxisSchema(mode="steady"),
            geometry_provider=FixedGeometryProvider(geometry_root),
            output_dir=tmp_path / "infer",
            input_mode="table_only",
            input_mode_meta=request_meta,
            checkpoint_input_mode_meta=checkpoint_meta,
        )


def test_inference_engine_metadata_contains_all_effective_keys(
    tmp_path: Path,
    geometry_root: Path,
    assert_input_mode_metadata_keys,
) -> None:
    meta = _table_only_meta()
    engine = InferenceEngine(
        model=GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
        cond_schema=CondSchema(order=["c0", "c1", "c2"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "infer",
        input_mode="table_only",
        input_mode_meta=meta,
        checkpoint_input_mode_meta=dict(meta),
    )
    assert_input_mode_metadata_keys(engine.input_mode_meta)
