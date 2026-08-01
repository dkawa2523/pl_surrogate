from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import plasma_surrogate.infer.engine as engine_mod
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    build_input_mode_effective_metadata,
)
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.features.structure_descriptors import build_structure_descriptor
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


class _StubPODModel:
    def __init__(self, *, expected_input_dim: int | None = None, h: int = 8, w: int = 8) -> None:
        self.expected_input_dim = expected_input_dim
        self.h = int(h)
        self.w = int(w)
        self.last_input_dim: int | None = None

    def to_meta(self) -> dict[str, Any]:
        return {"model_type": "deeponet_pod"}

    def predict_fields(self, cond_batch: np.ndarray, **_: Any) -> dict[str, np.ndarray]:
        arr = np.asarray(cond_batch, dtype=np.float32)
        self.last_input_dim = int(arr.shape[1])
        if self.expected_input_dim is not None and self.last_input_dim != int(self.expected_input_dim):
            raise ValueError(
                f"unexpected cond dim for stub deeponet_pod: got={self.last_input_dim}, "
                f"expected={int(self.expected_input_dim)}"
            )
        shape = (arr.shape[0], 1, self.h, self.w)
        return {
            "ne": np.zeros(shape, dtype=np.float32),
            "ni": np.zeros(shape, dtype=np.float32),
            "Te": np.zeros(shape, dtype=np.float32),
            "phi": np.zeros(shape, dtype=np.float32),
        }


def _write_parametric_geometry_root(root: Path) -> Path:
    geom = root / "geometry"
    geom.mkdir(parents=True, exist_ok=True)
    h, w = 8, 8
    mask = np.ones((h, w), dtype=np.float32)
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    np.save(geom / "mask_plasma.npy", mask)
    np.save(geom / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(geom / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))
    part_stack = np.zeros((2, h, w), dtype=np.float32)
    part_stack[0, 2:4, 2:4] = 1.0
    part_stack[1, 4:6, 5:7] = 1.0
    np.savez_compressed(
        geom / "parts_pack.npz",
        mask_stack=part_stack,
        part_ids=np.asarray(["p0", "p1"], dtype=object),
    )
    manifest = {
        "part_ids": ["p0", "p1"],
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p1.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
        },
    }
    (geom / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _table_plus_meta(*, adapter_mode: str, descriptor_profile: str, latent_profile: str) -> dict[str, Any]:
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": "table_plus_structure",
                "structure": {
                    "feature_profile": "geom_v1_mainline",
                    "descriptor_profile": descriptor_profile,
                    "latent_profile": latent_profile,
                    "adapter_mode": adapter_mode,
                    "provider_mode": "parametric_parts",
                },
            }
        }
    )
    meta[STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY] = adapter_mode
    return meta


def _descriptor_dim_for_default_geom(geometry_root: Path, profile: str) -> int:
    provider = build_geometry_provider(geometry_root, provider_mode="parametric_parts")
    geom = provider.get({"geom_id": "default", "geom_param": {}})
    descriptor = build_structure_descriptor(profile, geom)
    return int(descriptor.vector.shape[0])


def test_deeponet_pod_table_plus_descriptor_lane_passes_when_dim_matches(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    descriptor_profile = "struct_desc_v1"
    descriptor_dim = _descriptor_dim_for_default_geom(geometry_root, descriptor_profile)
    cond_schema = CondSchema(order=["c0", "c1", "c2"])
    model = _StubPODModel(expected_input_dim=len(cond_schema.order) + descriptor_dim)
    input_mode_meta = _table_plus_meta(
        adapter_mode="descriptor_branch",
        descriptor_profile=descriptor_profile,
        latent_profile="none",
    )
    checkpoint_meta = {
        DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: descriptor_dim,
        DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: descriptor_profile,
        DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: "none",
        DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
    }
    engine = InferenceEngine(
        model=model,
        cond_schema=cond_schema,
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=build_geometry_provider(geometry_root, provider_mode="parametric_parts"),
        output_dir=tmp_path / "infer",
        input_mode="table_plus_structure",
        input_mode_meta=input_mode_meta,
        checkpoint_input_mode_meta=dict(input_mode_meta),
        checkpoint_meta=checkpoint_meta,
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )
    out = engine.single_run(
        cond={"c0": 0.1, "c1": 0.2, "c2": 0.3},
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
    )
    assert "uniformity" in out.qoi
    assert model.last_input_dim == len(cond_schema.order) + descriptor_dim


def test_deeponet_pod_descriptor_dim_mismatch_is_rejected(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    descriptor_profile = "struct_desc_v1"
    descriptor_dim = _descriptor_dim_for_default_geom(geometry_root, descriptor_profile)
    input_mode_meta = _table_plus_meta(
        adapter_mode="descriptor_branch",
        descriptor_profile=descriptor_profile,
        latent_profile="none",
    )
    engine = InferenceEngine(
        model=_StubPODModel(),
        cond_schema=CondSchema(order=["c0", "c1"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=build_geometry_provider(geometry_root, provider_mode="parametric_parts"),
        output_dir=tmp_path / "infer",
        input_mode="table_plus_structure",
        input_mode_meta=input_mode_meta,
        checkpoint_input_mode_meta=dict(input_mode_meta),
        checkpoint_meta={
            DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: descriptor_dim + 1,
            DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: descriptor_profile,
            DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: "none",
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
        },
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )
    with pytest.raises(ValueError, match="descriptor dim mismatch"):
        engine.single_run(
            cond={"c0": 0.2, "c1": 0.4},
            geom={"geom_id": "default"},
            axis={"mode": "steady", "value": 0.0},
        )


def test_deeponet_pod_latent_hook_metadata_is_enabled_when_pack_present(tmp_path: Path) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    latent_pack = {
        "vector": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "feature_names": np.asarray(["z0", "z1", "z2"], dtype=object),
    }
    input_mode_meta = _table_plus_meta(
        adapter_mode="none",
        descriptor_profile="none",
        latent_profile="shape_ae_v1",
    )
    engine = InferenceEngine(
        model=_StubPODModel(),
        cond_schema=CondSchema(order=["c0"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=build_geometry_provider(geometry_root, provider_mode="parametric_parts"),
        output_dir=tmp_path / "infer",
        input_mode="table_plus_structure",
        input_mode_meta=input_mode_meta,
        checkpoint_input_mode_meta=dict(input_mode_meta),
        checkpoint_meta={
            DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: 0,
            DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: "none",
            DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: "shape_ae_v1",
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: True,
        },
        latent_feature_pack=latent_pack,
    )
    assert engine.pod_latent_hook_effective is True


def test_deeponet_pod_descriptor_is_cached_per_geom_ref(tmp_path: Path, monkeypatch) -> None:
    geometry_root = _write_parametric_geometry_root(tmp_path / "dataset")
    descriptor_profile = "struct_desc_v1"
    descriptor_dim = _descriptor_dim_for_default_geom(geometry_root, descriptor_profile)
    input_mode_meta = _table_plus_meta(
        adapter_mode="descriptor_branch",
        descriptor_profile=descriptor_profile,
        latent_profile="none",
    )
    engine = InferenceEngine(
        model=_StubPODModel(expected_input_dim=1 + descriptor_dim),
        cond_schema=CondSchema(order=["c0"]),
        axis_schema=AxisSchema(mode="steady"),
        geometry_provider=build_geometry_provider(geometry_root, provider_mode="parametric_parts"),
        output_dir=tmp_path / "infer",
        input_mode="table_plus_structure",
        input_mode_meta=input_mode_meta,
        checkpoint_input_mode_meta=dict(input_mode_meta),
        checkpoint_meta={
            DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY: descriptor_dim,
            DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY: descriptor_profile,
            DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY: "none",
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
        },
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
    )
    call_count = 0
    original_builder = engine_mod.build_structure_descriptor

    def _counted_builder(profile: Any, geom_ctx: Any):
        nonlocal call_count
        call_count += 1
        return original_builder(profile, geom_ctx)

    monkeypatch.setattr(engine_mod, "build_structure_descriptor", _counted_builder)
    engine.single_run(
        cond={"c0": 0.1},
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
    )
    engine.single_run(
        cond={"c0": 0.2},
        geom={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
    )
    assert call_count == 1
    engine.single_run(
        cond={"c0": 0.3},
        geom={"geom_id": "default", "geom_param": {"part.p0.tx": 0.1}},
        axis={"mode": "steady", "value": 0.0},
    )
    assert call_count == 2
