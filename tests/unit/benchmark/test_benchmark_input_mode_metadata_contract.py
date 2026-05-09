from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasma_surrogate.benchmark import runner as runner_mod
from plasma_surrogate.core.input_modes import (
    build_input_mode_effective_metadata,
    load_checkpoint_metadata_with_input_mode,
)
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def _table_plus_structure_meta() -> dict[str, Any]:
    return build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": "table_plus_structure",
                "structure": {
                    "feature_profile": "geom_v1_mainline",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "grid_pack",
                    "provider_mode": "fixed",
                },
            }
        }
    )


def test_load_checkpoint_input_mode_meta_reads_effective_keys(tmp_path: Path, assert_input_mode_metadata_keys) -> None:
    meta = _table_plus_structure_meta()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({**meta, "extra": 1}), encoding="utf-8")

    _, loaded = load_checkpoint_metadata_with_input_mode(meta_path)
    assert loaded == meta
    assert_input_mode_metadata_keys(loaded)


def test_build_benchmark_inference_engine_passes_checkpoint_and_request_meta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeInferenceEngine:
        pass

    def _fake_builder(*args: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = dict(kwargs)
        return _FakeInferenceEngine()

    monkeypatch.setattr(runner_mod, "build_inference_engine", _fake_builder)
    input_mode_meta = _table_plus_structure_meta()
    checkpoint_meta_path = tmp_path / "meta.json"
    checkpoint_meta_path.write_text(json.dumps(input_mode_meta), encoding="utf-8")

    runner_mod._build_benchmark_inference_engine(
        model=object(),
        cond_schema=object(),
        axis_schema=object(),
        geometry_provider=object(),
        output_dir=tmp_path / "inference",
        transform_bundle=object(),
        cond_stats={},
        phi_mode="direct",
        phi_hybrid_steps=1,
        poisson_refine_iters=0,
        ood_cfg={},
        feature_store=object(),
        coord_scaler={},
        coord_feature_scaler={},
        coord_feature_pack={},
        coord_distance_transform_stats={},
        input_mode_meta=input_mode_meta,
        checkpoint_meta_path=checkpoint_meta_path,
        deeponet_head=None,
        grid_input_features_cfg={},
    )

    kwargs = dict(captured["kwargs"])
    assert kwargs["input_mode_meta"] == input_mode_meta
    assert kwargs["checkpoint_meta_path"] == checkpoint_meta_path


def test_inject_input_mode_metadata_into_row_adds_all_effective_keys(assert_input_mode_metadata_keys) -> None:
    row = {"model_id": "ffno"}
    meta = _table_plus_structure_meta()
    runner_mod._inject_input_mode_metadata_into_row(
        row=row,
        input_mode_meta=meta,
    )
    assert_input_mode_metadata_keys(row)
    for key, value in meta.items():
        assert row[key] == value


def test_inject_input_mode_metadata_into_row_rejects_missing_key() -> None:
    row = {"model_id": "ffno"}
    meta = _table_plus_structure_meta()
    meta.pop("geometry_provider_mode_effective")
    try:
        runner_mod._inject_input_mode_metadata_into_row(
            row=row,
            input_mode_meta=meta,
        )
    except ValueError as exc:
        assert "geometry_provider_mode_effective" in str(exc)
    else:
        raise AssertionError("expected ValueError when required key is missing")


def test_benchmark_built_inference_engine_rejects_geom_space_with_fixed_provider(
    tmp_path: Path,
    geometry_root: Path,
) -> None:
    input_mode_meta = _table_plus_structure_meta()
    checkpoint_meta_path = tmp_path / "meta.json"
    checkpoint_meta_path.write_text(json.dumps(input_mode_meta), encoding="utf-8")

    engine = runner_mod._build_benchmark_inference_engine(
        model=GlobalMLP(input_dim=2, grid_shape=(8, 8), seed=3),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=FixedGeometryProvider(geometry_root),
        output_dir=tmp_path / "inference",
        transform_bundle=None,
        cond_stats={},
        phi_mode="direct",
        phi_hybrid_steps=1,
        poisson_refine_iters=0,
        ood_cfg={},
        feature_store=None,
        coord_scaler={},
        coord_feature_scaler={},
        coord_feature_pack={},
        coord_distance_transform_stats={},
        input_mode_meta=input_mode_meta,
        checkpoint_meta_path=checkpoint_meta_path,
        deeponet_head=None,
        grid_input_features_cfg={},
    )
    try:
        engine.optimize_run(
            space={"c0": (0.0, 1.0)},
            geom_space={"part.p0.tx": (-0.2, 0.2)},
            n_trials=2,
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )
    except ValueError as exc:
        assert "provider_mode=parametric_parts" in str(exc)
    else:
        raise AssertionError("expected ValueError for fixed provider geom_space")
