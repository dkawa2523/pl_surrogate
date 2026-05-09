from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
from plasma_surrogate.train.trainer import TrainOutput, Trainer
from tests._runtime_requirements import require_torch_runtime


class _IdentityTransforms:
    def __init__(self) -> None:
        self.target_transforms: dict[str, dict[str, Any]] = {}
        self.y_scalers: dict[str, Any] = {}

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}

    def inverse_fields(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)


def _ctx(tmp_path: Path) -> TrainDispatchContext:
    n, d, h, w = 8, 3, 4, 4
    rng = np.random.default_rng(31)
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    coord_data = np.stack(
        [
            np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1)),
            np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w)),
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.25, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    y_vars = ["ne", "ni", "Te", "phi"]
    return TrainDispatchContext(
        run_cfg={"train": {}},
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name="unet_operator_v2",
        model_dir=tmp_path / "dispatch_unet_operator_v2",
        global_seed=0,
        n_cases=n,
        h=h,
        w=w,
        y_vars=y_vars,
        cond_scaled=rng.normal(size=(n, d)).astype(np.float32),
        y=np.abs(rng.normal(size=(n, len(y_vars), h, w))).astype(np.float32),
        y_scaled=np.abs(rng.normal(size=(n, len(y_vars), h, w))).astype(np.float32),
        tr=np.arange(0, 5, dtype=np.int64),
        va=np.arange(5, 7, dtype=np.int64),
        te=np.arange(7, 8, dtype=np.int64),
        transforms=_IdentityTransforms(),
        physics_cfg={"enabled": False, "boundary_operator": {"enabled": False}},
        geom_ctx=None,
        deeponet_index={},
        deeponet_index_meta={},
        deeponet_poisson_index={},
        deeponet_poisson_meta={},
        deeponet_boundary_index={},
        deeponet_boundary_meta={},
        input_mode_effective="table_plus_structure",
        structure_adapter_mode_effective="auto",
        coord_feature_pack={"data": coord_data, "channels": np.asarray(channels)},
        coord_distance_transform_stats={},
        config_base_dir=tmp_path,
    )


def _valid_cfg() -> dict[str, Any]:
    return {
        "epochs": 1,
        "lr": 1e-3,
        "target_family": "allvars",
        "target_vars": ["ne", "ni", "Te", "phi"],
        "selection": {
            "mode": "best_val_allvars_balance",
            "weights": {"ne": 0.25, "ni": 0.25, "Te": 0.25, "phi": 0.25},
        },
        "input_features": {
            "mode": "geom_feature_pack",
            "require_pack": "error",
            "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            "distance_transform": {"mode": "raw"},
        },
        "model_cfg": {
            "backend": "torch",
            "unet_operator_v2_cfg": {
                "width": 8,
                "depth": 2,
                "blocks_per_level": 1,
                "max_width": 16,
                "dropout": 0.0,
                "kernel_size": 3,
                "downsample": "blur",
                "upsample": "bilinear",
                "use_film": True,
                "head_mode": "split_density_field",
            },
        },
    }


def test_unet_operator_v2_dispatch_accepts_valid_geom_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path)
    ctx.run_cfg = {"train": {"unet_operator_v2": _valid_cfg()}}
    monkeypatch.setattr(
        Trainer,
        "run_unet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == {"ne", "ni", "Te", "phi"}
    assert out.extra_artifacts["unet_contract_effective"]["unet_operator_v2_contract_effective"]["enabled"] is True
