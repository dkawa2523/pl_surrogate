from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.core.input_modes import (
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
)
from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
from plasma_surrogate.train.grid_contracts import resolve_geom_deeponet_siren_descriptor_contract
from plasma_surrogate.train.trainer import TrainOutput, Trainer


class _IdentityTransforms:
    def __init__(self) -> None:
        self.target_transforms: dict[str, dict[str, Any]] = {}
        self.y_scalers: dict[str, Any] = {}

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}

    def inverse_fields(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)



def _ctx(
    tmp_path: Path,
    *,
    input_mode: str = "table_plus_structure",
    adapter_mode: str = "auto",
    descriptor_profile: str = "struct_desc_v1",
) -> TrainDispatchContext:
    n, d, h, w = 8, 3, 4, 4
    rng = np.random.default_rng(121)
    cond = rng.normal(size=(n, d)).astype(np.float32)
    y_vars = ["density", "temperature", "potential", "flux"]
    y = np.abs(rng.normal(size=(n, len(y_vars), h, w))).astype(np.float32)
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
    descriptor_vec = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return TrainDispatchContext(
        run_cfg={"train": {}},
        profile_lock={"phi_mode": "direct", "primary_qoi_key": "Gamma_i"},
        model_idx=0,
        model_name="geom_deeponet_siren",
        model_dir=tmp_path / "dispatch_geom_deeponet_siren",
        global_seed=0,
        n_cases=n,
        h=h,
        w=w,
        y_vars=y_vars,
        cond_scaled=cond,
        y=y,
        y_scaled=y,
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
        input_mode_effective=input_mode,
        structure_adapter_mode_effective=adapter_mode,
        structure_descriptor_profile_effective=descriptor_profile,
        coord_feature_pack={"data": coord_data, "channels": np.asarray(channels)},
        structure_descriptor_pack={
            "vector": descriptor_vec,
            "feature_names": np.asarray([f"d{i}" for i in range(descriptor_vec.size)], dtype=object),
        },
        coord_distance_transform_stats={},
        config_base_dir=tmp_path,
    )


def _valid_cfg() -> dict[str, Any]:
    return {
        "epochs": 1,
        "lr": 1e-3,
        "target_family": "allvars",
        "target_vars": ["density", "temperature", "potential", "flux"],
        "selection": {
            "mode": "best_val_allvars_balance",
            "weights": {"density": 0.25, "temperature": 0.25, "potential": 0.25, "flux": 0.25},
        },
        "input_features": {
            "mode": "geom_feature_pack",
            "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            "distance_transform": {"mode": "raw"},
        },
        "model_cfg": {
            "backend": "torch",
            "geom_deeponet_siren_cfg": {
                "latent_dim": 12,
                "trunk_hidden": 16,
                "trunk_layers": 2,
                "branch_hidden": 24,
                "branch_layers": 2,
                "dropout": 0.0,
                "trunk_w0": 20.0,
            },
        },
    }


def test_geom_deeponet_siren_accepts_hybrid_descriptor_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    require_torch_runtime()
    ctx = _ctx(tmp_path, adapter_mode="auto", descriptor_profile="struct_desc_v1")
    ctx.run_cfg = {"train": {"geom_deeponet_siren": _valid_cfg()}}
    monkeypatch.setattr(
        Trainer,
        "run_unet",
        lambda self, model, cond_train, y_train, cond_val, y_val, **kwargs: TrainOutput(
            history=[{"epoch": 0.0, "train_loss": 0.0, "val_loss": 0.0}],
            model=model,
        ),
    )
    out = run_model_train_predict(ctx)
    assert set(out.metrics.keys()) == set(ctx.y_vars)
    assert int(out.extra_artifacts[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY]) == 4
    assert out.extra_artifacts[GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY] == "struct_desc_v1"


def test_geom_deeponet_siren_contract_preserves_case_descriptor_rows() -> None:
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    case_ids = np.asarray([f"case_{idx}" for idx in range(6)])
    resolved, meta = resolve_geom_deeponet_siren_descriptor_contract(
        input_mode="table_plus_structure",
        adapter_mode="hybrid_pack_descriptor",
        descriptor_profile="struct_desc_v2",
        descriptor_pack={
            "vectors": vectors,
            "case_ids": case_ids,
            "feature_names": np.asarray(["d0", "d1", "d2", "d3"]),
        },
    )

    np.testing.assert_array_equal(resolved, vectors)
    assert meta["descriptor_scope_effective"] == "case_specific"
    assert meta[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY] == 4
