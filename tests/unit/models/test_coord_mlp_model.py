from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.models.mlp.io import build_model_from_name, load_mlp_checkpoint, save_mlp_checkpoint


def _enable_torch() -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"


def _model_cfg() -> dict[str, object]:
    return {
        "cond_hidden": [16, 16],
        "latent_dim": 12,
        "decoder_hidden": [24, 24],
        "decoder_activation": "gelu",
        "embedding": {
            "type": "fourier",
            "n_frequencies": 4,
            "include_raw": True,
            "frequency_scale": 10.0,
        },
    }


def _siren_model_cfg() -> dict[str, object]:
    return {
        "cond_hidden": [16, 16],
        "latent_dim": 12,
        "decoder_hidden": [24, 24],
        "embedding": {"type": "none"},
        "siren": {"enabled": True, "w0_initial": 30.0, "w0_hidden": 1.0},
    }


def test_build_coord_mlp_fourier_smoke() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = build_model_from_name(
        model_name="coord_mlp_fourier",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_model_cfg(),
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, CoordMLPTorch)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]
    assert model.model_type == "coord_mlp_fourier"


def test_build_coord_mlp_siren_smoke() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = build_model_from_name(
        model_name="coord_mlp_siren",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_siren_model_cfg(),
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, CoordMLPTorch)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]
    assert model.model_type == "coord_mlp_siren"


def test_coord_mlp_static_spatial_features_shape_validation() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    with pytest.raises(ValueError, match="channels mismatch"):
        model.set_static_spatial_features(np.zeros((8, 8, 4), dtype=np.float32))


def test_coord_mlp_forward_shape_with_dynamic_output_keys() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    spatial = np.zeros((8, 8, 5), dtype=np.float32)
    spatial[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :]
    spatial[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    spatial[..., 2] = 1.0
    spatial[..., 3] = 0.5
    spatial[..., 4] = 0.25
    model.set_static_spatial_features(spatial)
    cond = np.ones((3, 3), dtype=np.float32)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (3, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}


def test_coord_mlp_siren_forward_shape_with_dynamic_output_keys() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_siren_model_cfg(),
    )
    spatial = np.zeros((8, 8, 5), dtype=np.float32)
    spatial[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :]
    spatial[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    spatial[..., 2] = 1.0
    spatial[..., 3] = 0.5
    spatial[..., 4] = 0.25
    model.set_static_spatial_features(spatial)
    cond = np.ones((3, 3), dtype=np.float32)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (3, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}


def test_coord_mlp_accepts_batched_spatial_features() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    base = np.zeros((8, 8, 5), dtype=np.float32)
    base[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :]
    base[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    spatial = np.stack([base, base + 0.1], axis=0).astype(np.float32)
    cond = np.ones((2, 3), dtype=np.float32)
    out = model.forward(cond, spatial_features=spatial)
    assert out.shape == (2, 2, 8, 8)


def test_coord_mlp_rejects_missing_spatial_features() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    with pytest.raises(ValueError, match="requires explicit geom_feature_pack spatial features"):
        model.forward(np.ones((1, 3), dtype=np.float32))


def test_coord_mlp_rejects_invalid_embedding_contract() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    bad_cfg = _model_cfg()
    bad_cfg["embedding"] = {"type": "siren", "n_frequencies": 4, "include_raw": True, "frequency_scale": 10.0}
    with pytest.raises(ValueError, match="must be fourier"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )


def test_coord_mlp_siren_rejects_invalid_contract() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    bad_cfg = _siren_model_cfg()
    bad_cfg["embedding"] = {"type": "fourier"}
    with pytest.raises(ValueError, match="embedding.type must be none"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"] = {"enabled": False, "w0_initial": 30.0, "w0_hidden": 1.0}
    with pytest.raises(ValueError, match="siren.enabled must be true"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["decoder_activation"] = "gelu"
    with pytest.raises(ValueError, match="decoder_activation is not used for SIREN"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["w0_initial"] = 0.0
    with pytest.raises(ValueError, match="w0_initial must be > 0"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["w0_hidden"] = 0.0
    with pytest.raises(ValueError, match="w0_hidden must be > 0"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )


def test_coord_mlp_checkpoint_roundtrip(tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_mlp_checkpoint(model, tmp_path / "ckpt")
    loaded = load_mlp_checkpoint(tmp_path / "ckpt")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CoordMLPTorch)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_coord_mlp_siren_checkpoint_roundtrip(tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_siren_model_cfg(),
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_mlp_checkpoint(model, tmp_path / "ckpt_siren")
    loaded = load_mlp_checkpoint(tmp_path / "ckpt_siren")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CoordMLPTorch)
    assert loaded.model_type == "coord_mlp_siren"
    assert loaded.model_cfg["embedding"] == {"type": "none"}
    assert "decoder_activation" not in loaded.model_cfg
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_coord_mlp_siren_uses_standard_cond_encoder_and_sine_point_decoder() -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_siren_model_cfg(),
    )
    cond_has_sine = any(type(module).__name__ == "_SineActivation" for module in model.net.cond_encoder)
    point_has_sine = any(type(module).__name__ == "_SineActivation" for module in model.net.point_decoder)
    assert cond_has_sine is False
    assert point_has_sine is True


def test_coord_mlp_changes_do_not_break_global_mlp_factory() -> None:
    model = build_model_from_name(
        model_name="global_mlp",
        input_dim=3,
        grid_shape=(4, 4),
        model_cfg={"hidden": [8], "dropout": 0.0},
        out_channels=2,
        output_keys=["density", "temperature"],
    )
    assert isinstance(model, GlobalMLP)
