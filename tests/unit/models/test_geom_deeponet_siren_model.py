from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np

from plasma_surrogate.models.deeponet.geom_deeponet_siren import GeomDeepONetSIREN
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint

pytestmark = pytest.mark.torch_runtime


def _cfg() -> dict[str, object]:
    return {
        "latent_dim": 16,
        "trunk_hidden": 24,
        "trunk_layers": 2,
        "branch_hidden": 32,
        "branch_layers": 2,
        "dropout": 0.0,
        "trunk_w0": 20.0,
    }


def test_build_geom_deeponet_siren_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="geom_deeponet_siren",
        input_dim=7,
        grid_shape=(8, 8),
        model_cfg={"backend": "torch", "geom_deeponet_siren_cfg": _cfg()},
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, GeomDeepONetSIREN)
    assert model.output_keys == ["ne", "ni", "Te", "phi"]


def test_geom_deeponet_siren_forward_shape() -> None:
    require_torch_runtime()
    model = GeomDeepONetSIREN(
        input_dim=7,
        grid_shape=(8, 8),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        geom_deeponet_siren_cfg=_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 7)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (2, 4, 8, 8)
    assert set(pred.keys()) == {"ne", "ni", "Te", "phi"}
    assert np.isfinite(out).all()


def test_geom_deeponet_siren_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = GeomDeepONetSIREN(
        input_dim=7,
        grid_shape=(8, 8),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        geom_deeponet_siren_cfg=_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(2).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(3).normal(size=(2, 7)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_geom_deeponet_siren")
    loaded = load_checkpoint(tmp_path / "ckpt_geom_deeponet_siren")
    assert isinstance(loaded, GeomDeepONetSIREN)
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)
