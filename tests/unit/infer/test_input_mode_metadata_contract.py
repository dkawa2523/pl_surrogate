from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.core.input_modes import (
    INPUT_MODE_EFFECTIVE_KEY,
    TARGET_SCHEMA_HASH_KEY,
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
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )


def _engine_kwargs(tmp_path: Path, geometry_root: Path, meta: dict[str, object]) -> dict[str, object]:
    return {
        "model": GlobalMLP(input_dim=3, grid_shape=(8, 8), seed=0),
        "cond_schema": CondSchema(order=["c0", "c1", "c2"]),
        "axis_schema": AxisSchema(mode="steady"),
        "geometry_provider": FixedGeometryProvider(geometry_root),
        "output_dir": tmp_path / "infer",
        "input_mode": "table_only",
        "input_mode_meta": meta,
    }


def test_inference_engine_accepts_matching_input_mode_metadata(tmp_path: Path, geometry_root: Path) -> None:
    meta = _table_only_meta()
    engine = InferenceEngine(
        **_engine_kwargs(tmp_path, geometry_root, meta),
        checkpoint_input_mode_meta=dict(meta),
    )
    assert engine.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY] == "table_only"


def test_inference_engine_rejects_checkpoint_input_mode_mismatch(tmp_path: Path, geometry_root: Path) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta["structure_adapter_mode_effective"] = "auto"
    with pytest.raises(ValueError, match="expected='auto'.*got='none'"):
        InferenceEngine(
            **_engine_kwargs(tmp_path, geometry_root, request_meta),
            checkpoint_input_mode_meta=checkpoint_meta,
        )


def test_inference_engine_rejects_checkpoint_input_mode_missing_key(tmp_path: Path, geometry_root: Path) -> None:
    request_meta = _table_only_meta()
    checkpoint_meta = dict(request_meta)
    checkpoint_meta.pop(TARGET_SCHEMA_HASH_KEY)
    with pytest.raises(ValueError, match=TARGET_SCHEMA_HASH_KEY):
        InferenceEngine(
            **_engine_kwargs(tmp_path, geometry_root, request_meta),
            checkpoint_input_mode_meta=checkpoint_meta,
        )


def test_inference_engine_metadata_contains_all_effective_keys(
    tmp_path: Path,
    geometry_root: Path,
    assert_input_mode_metadata_keys,
) -> None:
    meta = _table_only_meta()
    engine = InferenceEngine(
        **_engine_kwargs(tmp_path, geometry_root, meta),
        checkpoint_input_mode_meta=dict(meta),
    )
    assert_input_mode_metadata_keys(engine.input_mode_meta)
