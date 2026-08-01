from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plasma_surrogate.benchmark import model_dispatch as mod


def _ctx(tmp_path: Path) -> mod.BenchmarkModelContext:
    zeros = np.zeros((1, 1), dtype=np.float32)
    idx = np.array([0], dtype=np.int64)
    return mod.BenchmarkModelContext(
        benchmark_cfg={
            "runtime": {
                "input_mode": "table_plus_structure",
                "structure": {
                    "feature_profile": "geom_v1_mainline",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "auto",
                    "provider_mode": "fixed",
                },
            }
        },
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name="deeponet_pod",
        model_dir=tmp_path / "m",
        global_seed=0,
        n_cases=1,
        h=8,
        w=8,
        y_vars=["ne"],
        cond_scaled=zeros,
        y=np.zeros((1, 1, 8, 8), dtype=np.float32),
        y_scaled=np.zeros((1, 1, 8, 8), dtype=np.float32),
        tr=idx,
        va=idx,
        te=idx,
        transforms=SimpleNamespace(),
        physics_cfg={},
        loss_cfg={},
        curriculum_cfg={},
        supervised_mask=None,
        supervised_distance=None,
        geom_ctx=None,
        deeponet_index={},
        deeponet_index_meta={},
        deeponet_poisson_index={},
        deeponet_poisson_meta={},
        deeponet_boundary_index={},
        deeponet_boundary_meta={},
        coord_feature_scaler={},
        coord_feature_pack=None,
        coord_distance_transform_stats={},
    )


def test_run_model_train_eval_passes_structure_adapter_mode_to_dispatch(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_train_predict(ctx):
        captured["ctx"] = ctx
        return SimpleNamespace(
            model=object(),
            history=[],
            pred_eval={},
            true_eval={},
            metrics={},
            r2_scores={},
            extra_artifacts={},
        )

    monkeypatch.setattr(mod, "run_model_train_predict", _fake_run_model_train_predict)
    mod.run_model_train_eval(_ctx(tmp_path))
    dispatch_ctx = captured["ctx"]
    assert getattr(dispatch_ctx, "structure_adapter_mode_effective") == "none"
