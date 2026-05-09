from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np

from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline

pytestmark = pytest.mark.torch_runtime


def _spatial() -> np.ndarray:
    h = w = 8
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.stack(
        [
            np.tile(xx, (h, 1)),
            np.tile(yy[:, None], (1, w)),
            np.ones((h, w), dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.25, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def test_build_unetpp_attn_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unetpp_attn",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={
            "backend": "torch",
            "conv_cfg": {"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
            "output_heads": {"mode": "shared"},
        },
        out_channels=2,
        output_keys=["density", "temperature"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, UNetPPBaseline)
    assert model.model_type == "unetpp_attn"


def test_unetpp_attn_forward_shape_matches_unetpp_contract() -> None:
    require_torch_runtime()
    cond = np.ones((3, 3), dtype=np.float32)
    spatial = _spatial()
    base = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={"base_channels": 8, "depth": 2, "upsample_mode": "bilinear", "nested_skip": True},
        output_heads={"mode": "shared"},
    )
    attn = UNetPPBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        backend="torch",
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        conv_cfg={
            "base_channels": 8,
            "depth": 2,
            "upsample_mode": "bilinear",
            "nested_skip": True,
            "attention_cfg": {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"},
        },
        output_heads={"mode": "shared"},
    )
    base.set_static_spatial_features(spatial)
    attn.set_static_spatial_features(spatial)
    assert base.forward(cond).shape == attn.forward(cond).shape == (3, 2, 8, 8)
    assert set(attn.predict_fields(cond).keys()) == {"density", "temperature"}


def test_unetpp_attn_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unetpp_attn",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={
            "backend": "torch",
            "conv_cfg": {
                "base_channels": 8,
                "depth": 2,
                "upsample_mode": "bilinear",
                "nested_skip": True,
                "attention_cfg": {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"},
            },
            "output_heads": {"mode": "shared"},
        },
        out_channels=2,
        output_keys=["density", "temperature"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    spatial = _spatial()
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, UNetPPBaseline)
    assert loaded.model_type == "unetpp_attn"
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_unetpp_attn_forward_accepts_batched_spatial_features() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unetpp_attn",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={
            "backend": "torch",
            "conv_cfg": {
                "base_channels": 8,
                "depth": 2,
                "upsample_mode": "bilinear",
                "nested_skip": True,
                "attention_cfg": {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"},
            },
            "output_heads": {"mode": "shared"},
        },
        out_channels=2,
        output_keys=["density", "temperature"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    spatial = np.random.default_rng(5).normal(size=(2, 8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(6).normal(size=(2, 3)).astype(np.float32)
    out = model.forward(cond, spatial_features=spatial)
    assert out.shape == (2, 2, 8, 8)


@pytest.mark.parametrize(
    ("attention_cfg", "message"),
    [
        ({"enabled": True, "reduction": 0, "gate_activation": "sigmoid"}, "reduction must be >= 1"),
        ({"enabled": True, "reduction": 2, "gate_activation": "tanh"}, "gate_activation must be sigmoid"),
        ({"enabled": True, "reduction": 2, "gate_activation": "sigmoid"}, "deep_supervision.enabled=true"),
    ],
)
def test_unetpp_attn_rejects_invalid_attention_cfg(attention_cfg: dict[str, object], message: str) -> None:
    require_torch_runtime()
    conv_cfg = {
        "base_channels": 8,
        "depth": 2,
        "upsample_mode": "bilinear",
        "nested_skip": True,
        "attention_cfg": attention_cfg,
    }
    if "deep_supervision.enabled=true" in message:
        conv_cfg["deep_supervision"] = {"enabled": True}
    with pytest.raises(ValueError, match=message):
        UNetPPBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            backend="torch",
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            conv_cfg=conv_cfg,
            output_heads={"mode": "shared"},
        )
