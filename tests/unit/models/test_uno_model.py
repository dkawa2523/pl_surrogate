from __future__ import annotations

import pytest
import json
from pathlib import Path

import numpy as np

from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.uno.simple_uno import UNOBaseline

pytestmark = pytest.mark.torch_runtime


def _uno_cfg() -> dict[str, object]:
    return {"width": 16, "n_layers": 2, "dropout": 0.0}


def test_build_uno_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="u_no",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={"backend": "torch", "n_modes": 4, "uno_cfg": _uno_cfg()},
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, UNOBaseline)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]
    assert int(model.n_modes) == 4


def test_uno_forward_shape() -> None:
    require_torch_runtime()
    model = UNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        uno_cfg=_uno_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(8).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(7).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    out = model.forward(cond)
    pred = model.predict_fields(cond)
    assert out.shape == (2, 2, 8, 8)
    assert set(pred.keys()) == {"density", "temperature"}
    assert np.isfinite(out).all()


def test_uno_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = UNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        uno_cfg=_uno_cfg(),
        backend="torch",
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_uno")
    loaded = load_checkpoint(tmp_path / "ckpt_uno")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, UNOBaseline)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_uno_rejects_non_positive_n_modes() -> None:
    require_torch_runtime()
    with pytest.raises(ValueError, match="train.u_no.model_cfg.n_modes must be >= 1"):
        UNOBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            n_modes=0,
            uno_cfg=_uno_cfg(),
            backend="torch",
        )


def test_uno_checkpoint_load_rejects_non_positive_n_modes(tmp_path: Path) -> None:
    require_torch_runtime()
    model = UNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        uno_cfg=_uno_cfg(),
        backend="torch",
    )
    ckpt_dir = tmp_path / "ckpt_uno_invalid_n_modes"
    save_checkpoint(model, ckpt_dir)
    meta_path = ckpt_dir / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["n_modes"] = 0
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="train.u_no.model_cfg.n_modes must be >= 1"):
        load_checkpoint(ckpt_dir)
