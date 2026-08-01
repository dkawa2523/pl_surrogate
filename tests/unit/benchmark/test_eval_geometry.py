from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import plasma_surrogate.benchmark.runner as runner_module
from plasma_surrogate.benchmark.eval_geometry import materialize_evaluation_spatial_geometry
from plasma_surrogate.benchmark.runner import (
    BenchmarkModelResult,
    BenchmarkProbeResult,
    BenchmarkRunner,
)


def _case_context(*, masks: np.ndarray, distances: np.ndarray) -> SimpleNamespace:
    n_cases, h, w = masks.shape
    case_data = np.stack([masks, distances], axis=1).astype(np.float32)
    static_data = np.stack(
        [
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 99.0, dtype=np.float32),
        ],
        axis=0,
    )
    return SimpleNamespace(
        h=h,
        w=w,
        dataset=SimpleNamespace(cases=[{"case_id": f"case-{i}"} for i in range(n_cases)]),
        case_structure_feature_pack={
            "data": case_data,
            "channels": np.asarray(["mask_plasma", "distance_any"]),
            "case_ids": np.asarray([f"case-{i}" for i in range(n_cases)]),
        },
        static_spatial_feature_pack={
            "data": static_data,
            "channels": np.asarray(["mask_plasma", "distance_any"]),
        },
        coord_feature_pack=None,
    )


def test_materialize_evaluation_geometry_prefers_case_raw_channels_in_test_order() -> None:
    masks = np.asarray(
        [
            [[1, 1], [1, 0]],
            [[1, 0], [0, 0]],
            [[0, 1], [1, 1]],
        ],
        dtype=np.float32,
    )
    distances = np.asarray(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
            [[8, 9], [10, 11]],
        ],
        dtype=np.float32,
    )
    context = _case_context(masks=masks, distances=distances)
    geometry = materialize_evaluation_spatial_geometry(
        context=context,
        geom_ctx=SimpleNamespace(mask_plasma=np.ones((2, 2)), distance_any=np.zeros((2, 2))),
        eval_indices=np.asarray([2, 0], dtype=np.int64),
        metric_mask_enabled=True,
        expected_eval_count=2,
    )

    assert geometry.source == "case_structure_feature_pack"
    assert geometry.mask_plasma.shape == (2, 2, 2)
    assert geometry.distance_any.shape == (2, 2, 2)
    np.testing.assert_array_equal(geometry.mask_plasma, masks[[2, 0]])
    np.testing.assert_array_equal(geometry.metric_mask, masks[[2, 0]])
    np.testing.assert_array_equal(geometry.distance_any, distances[[2, 0]])
    np.testing.assert_array_equal(
        geometry.distance_signed,
        np.where(masks[[2, 0]] > 0.5, distances[[2, 0]], -distances[[2, 0]]),
    )


def test_materialize_evaluation_geometry_broadcasts_static_fallback() -> None:
    context = SimpleNamespace(
        h=2,
        w=2,
        dataset=SimpleNamespace(cases=[{"case_id": "a"}, {"case_id": "b"}]),
        case_structure_feature_pack={
            "data": np.zeros((2, 1, 2, 2), dtype=np.float32),
            "channels": np.asarray(["mask_coil"]),
            "case_ids": np.asarray(["a", "b"]),
        },
        static_spatial_feature_pack=None,
        coord_feature_pack=None,
    )
    mask = np.asarray([[1, 1], [1, 0]], dtype=np.float32)
    distance = np.asarray([[0, 1], [2, 3]], dtype=np.float32)
    geometry = materialize_evaluation_spatial_geometry(
        context=context,
        geom_ctx=SimpleNamespace(mask_plasma=mask, distance_any=distance),
        eval_indices=np.asarray([1, 0], dtype=np.int64),
        metric_mask_enabled=False,
        expected_eval_count=2,
    )

    assert geometry.source == "geometry_context"
    assert geometry.metric_mask is None
    np.testing.assert_array_equal(geometry.mask_plasma, np.repeat(mask[None, ...], 2, axis=0))
    np.testing.assert_array_equal(geometry.distance_any, np.repeat(distance[None, ...], 2, axis=0))


def test_evaluation_geometry_composes_static_wall_and_case_part_distance() -> None:
    part_sdf = np.asarray(
        [
            [[9.0, 2.0], [-1.0, 8.0]],
            [[4.0, -3.0], [7.0, 0.5]],
        ],
        dtype=np.float32,
    )
    context = SimpleNamespace(
        h=2,
        w=2,
        dataset=SimpleNamespace(cases=[{"case_id": "a"}, {"case_id": "b"}]),
        case_structure_feature_pack={
            "data": part_sdf[:, None],
            "channels": np.asarray(["part_sdf_nearest"]),
            "case_ids": np.asarray(["a", "b"]),
        },
        static_spatial_feature_pack={
            "data": np.stack(
                [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 5.0, dtype=np.float32)]
            ),
            "channels": np.asarray(["mask_plasma", "distance_any"]),
        },
        coord_feature_pack=None,
    )

    geometry = materialize_evaluation_spatial_geometry(
        context=context,
        geom_ctx=None,
        eval_indices=np.asarray([1, 0], dtype=np.int64),
        metric_mask_enabled=True,
        expected_eval_count=2,
        boundary_distance_channels=["distance_any", "part_sdf_nearest"],
    )

    assert geometry.source == "static_spatial_feature_pack+case_structure_feature_pack"
    np.testing.assert_array_equal(
        geometry.distance_any,
        np.asarray(
            [
                [[4.0, 3.0], [5.0, 0.5]],
                [[5.0, 2.0], [1.0, 5.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_evaluation_geometry_rejects_missing_configured_boundary_channel() -> None:
    context = _case_context(
        masks=np.ones((1, 2, 2), dtype=np.float32),
        distances=np.zeros((1, 2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing configured raw channel"):
        materialize_evaluation_spatial_geometry(
            context=context,
            geom_ctx=None,
            eval_indices=np.asarray([0], dtype=np.int64),
            metric_mask_enabled=True,
            boundary_distance_channels=["distance_any", "unknown_part_sdf"],
        )


def test_materialize_evaluation_geometry_rejects_partial_case_raw_contract() -> None:
    context = SimpleNamespace(
        h=2,
        w=2,
        dataset=SimpleNamespace(cases=[{"case_id": "a"}]),
        case_structure_feature_pack={
            "data": np.ones((1, 1, 2, 2), dtype=np.float32),
            "channels": np.asarray(["mask_plasma"]),
            "case_ids": np.asarray(["a"]),
        },
        static_spatial_feature_pack=None,
        coord_feature_pack=None,
    )
    with pytest.raises(ValueError, match="missing required raw channels"):
        materialize_evaluation_spatial_geometry(
            context=context,
            geom_ctx=SimpleNamespace(mask_plasma=np.ones((2, 2)), distance_any=np.zeros((2, 2))),
            eval_indices=np.asarray([0], dtype=np.int64),
            metric_mask_enabled=True,
            expected_eval_count=1,
        )


def test_materialize_evaluation_geometry_rejects_case_id_misalignment() -> None:
    context = _case_context(
        masks=np.ones((2, 2, 2), dtype=np.float32),
        distances=np.zeros((2, 2, 2), dtype=np.float32),
    )
    context.case_structure_feature_pack["case_ids"] = np.asarray(["case-1", "case-0"])
    with pytest.raises(ValueError, match="exactly match dataset order"):
        materialize_evaluation_spatial_geometry(
            context=context,
            geom_ctx=None,
            eval_indices=np.asarray([1], dtype=np.int64),
            metric_mask_enabled=True,
            expected_eval_count=1,
        )


class _FakeTransforms:
    def transform_cond(self, values):
        return np.asarray(values, dtype=np.float32)

    def transform_fields(self, values):
        return np.asarray(values, dtype=np.float32)

    def to_dict(self):
        return {"y_scalers": {}, "target_transforms": {}}


@pytest.mark.parametrize("lane", ["random", "interp", "extrap"])
def test_single_and_dual_axis_lanes_pass_case_aligned_geometry_to_eval_row(
    tmp_path,
    monkeypatch,
    lane: str,
) -> None:
    masks = np.asarray(
        [
            [[1, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[0, 1], [1, 1]],
        ],
        dtype=np.float32,
    )
    distances = np.asarray(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
            [[8, 9], [10, 11]],
        ],
        dtype=np.float32,
    )
    context = _case_context(masks=masks, distances=distances)
    context.n_cases = 3
    context.global_seed = 7
    context.cond = np.zeros((3, 1), dtype=np.float32)
    context.y = np.zeros((3, 1, 2, 2), dtype=np.float32)
    context.lock_hash = "lock"
    for name in (
        "deeponet_index",
        "deeponet_index_meta",
        "deeponet_poisson_index",
        "deeponet_poisson_meta",
        "deeponet_boundary_index",
        "deeponet_boundary_meta",
    ):
        setattr(context, name, {})
    context.coord_distance_transform_stats = {}
    protocol_transforms = {
        key: {"coord_feature_scaler": {"enabled": True}, "distance_transform_stats": {"mode": "raw"}}
        for key in ("interp", "extrap")
    }
    context.bundle = SimpleNamespace(
        schemas={"output_layout": {"vars": ["density"]}, "target_role_schema": {}},
        transforms={"protocol_transforms": protocol_transforms},
        transform_bundle=lambda *_args, **_kwargs: _FakeTransforms(),
    )
    geom_ctx = SimpleNamespace(
        mask_plasma=np.ones((2, 2), dtype=np.float32),
        distance_any=np.full((2, 2), 99.0, dtype=np.float32),
        distance_signed=np.full((2, 2), 99.0, dtype=np.float32),
        bc_dir_mask=np.zeros((2, 2), dtype=np.float32),
        regions={},
    )
    dispatch = {
        "model": object(),
        "history": [],
        "pred_eval": {"density": np.zeros((2, 1, 2, 2), dtype=np.float32)},
        "true_eval": {"density": np.zeros((2, 1, 2, 2), dtype=np.float32)},
        "metrics": {"density": 0.0},
        "r2_scores": {"density": 1.0},
        "extra_artifacts": {"effective_steps": 1},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner_module, "run_model_train_eval", lambda _ctx: dispatch)
    monkeypatch.setattr(
        runner_module,
        "resolve_effective_input_mode_metadata_for_model",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(runner_module, "_resolve_benchmark_potential_key", lambda **_kwargs: None)
    monkeypatch.setattr(runner_module, "_inject_input_mode_metadata_into_row", lambda **_kwargs: None)
    monkeypatch.setattr(runner_module, "_attach_primary_metric_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_module, "validation_selection_from_history", lambda _history: {})

    def _capture_eval_row(**kwargs):
        captured.update(kwargs)
        return {"model_id": "global_mlp"}

    monkeypatch.setattr(runner_module, "build_benchmark_eval_row", _capture_eval_row)
    monkeypatch.setattr(
        runner_module.BenchmarkProbe,
        "run",
        lambda *_args, **_kwargs: BenchmarkProbeResult(
            qoi={},
            diagnostics={},
            batch_rows=[],
            optimize=BenchmarkModelResult(status="skipped"),
        ),
    )

    runner = object.__new__(BenchmarkRunner)
    runner.benchmark_cfg = {"eval": {}, "inference": {}}
    runner.input_mode_meta = {}
    runner._write_model_runtime_artifacts = lambda **_kwargs: object()
    result = runner._run_single_split_model(
        context=context,
        profile_lock={},
        train_cfg={},
        loss_cfg={},
        curriculum_cfg={},
        physics_cfg={},
        geom_ctx=geom_ctx,
        quality_score_cfg={},
        metric_mask=np.ones((2, 2), dtype=np.float32),
        supervised_mask=None,
        supervised_distance=None,
        target_vars_for_score=["density"],
        region_band_cfg={},
        primary_metric="test_rmse_density",
        preprocess_report={},
        expected_lock_hash="lock",
        model_name="global_mlp",
        model_idx=0,
        model_dir=tmp_path / lane,
        tr_idx=np.asarray([1], dtype=np.int64),
        va_idx=np.asarray([1], dtype=np.int64),
        te_idx=np.asarray([2, 0], dtype=np.int64),
        scaler_split_name=lane,
    )

    assert result.row["model_id"] == "global_mlp"
    np.testing.assert_array_equal(captured["mask_plasma"], masks[[2, 0]])
    np.testing.assert_array_equal(captured["region_mask_plasma"], masks[[2, 0]])
    np.testing.assert_array_equal(captured["distance_any"], distances[[2, 0]])
    np.testing.assert_array_equal(
        captured["distance_signed"],
        np.where(masks[[2, 0]] > 0.5, distances[[2, 0]], -distances[[2, 0]]),
    )
