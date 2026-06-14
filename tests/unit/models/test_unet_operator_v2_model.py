from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint
from plasma_surrogate.models.unet.operator_v2 import UNetOperatorV2
from tests._runtime_requirements import require_torch_runtime

pytestmark = pytest.mark.torch_runtime


def _cfg() -> dict[str, object]:
    return {
        "width": 12,
        "depth": 2,
        "blocks_per_level": 1,
        "max_width": 24,
        "dropout": 0.0,
        "kernel_size": 3,
        "downsample": "blur",
        "upsample": "bilinear",
        "use_film": True,
        "activation": "silu",
        "head_mode": "shared",
    }


def test_build_unet_operator_v2_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="unet_operator_v2",
        input_dim=3,
        grid_shape=(9, 8),
        model_cfg={"backend": "torch", "unet_operator_v2_cfg": _cfg()},
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, UNetOperatorV2)
    assert model.output_keys == ["ne", "ni", "Te", "phi"]
    assert model.unet_operator_v2_cfg["head_mode"] == "shared"


def test_unet_operator_v2_forward_shape() -> None:
    require_torch_runtime()
    model = UNetOperatorV2(
        input_dim=3,
        grid_shape=(9, 8),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        unet_operator_v2_cfg=_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(21).normal(size=(9, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(22).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (2, 4, 9, 8)
    assert set(pred.keys()) == {"ne", "ni", "Te", "phi"}
    assert np.isfinite(out).all()


def test_unet_operator_v2_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = UNetOperatorV2(
        input_dim=3,
        grid_shape=(9, 8),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        unet_operator_v2_cfg=_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(23).normal(size=(9, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(24).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_unet_operator_v2")
    loaded = load_checkpoint(tmp_path / "ckpt_unet_operator_v2")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, UNetOperatorV2)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_unet_operator_v2_rejects_invalid_head_mode() -> None:
    require_torch_runtime()
    bad = dict(_cfg())
    bad["head_mode"] = "too_many_heads"
    with pytest.raises(ValueError, match="head_mode"):
        UNetOperatorV2(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["ne", "Te"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            unet_operator_v2_cfg=bad,
            backend="torch",
        )
