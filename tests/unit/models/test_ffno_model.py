from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np

from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint

pytestmark = pytest.mark.torch_runtime


def _spectral_cfg() -> dict[str, object]:
    return {
        "width": 16,
        "n_layers": 2,
        "dealias_ratio": 0.67,
        "taper_alpha": 4.0,
        "skip_filter": "match_spectral",
        "factorized_cfg": {"enabled": True, "mode": "separable_1d", "share_weights": False},
    }


def _spectral_cfg_with_local_skip(*, enabled: bool, init_scale: float = 0.0) -> dict[str, object]:
    cfg = _spectral_cfg()
    cfg["local_skip_cfg"] = {"enabled": bool(enabled), "init_scale": float(init_scale)}
    return cfg


def test_build_ffno_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="ffno",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={"backend": "torch", "n_modes": 3, "spectral_cfg": _spectral_cfg()},
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, FFNOBaseline)
    assert model.output_keys == ["density", "temperature", "potential", "aux"]


def test_build_ffno_local_skip_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="ffno",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg={"backend": "torch", "n_modes": 3, "spectral_cfg": _spectral_cfg_with_local_skip(enabled=True)},
        out_channels=4,
        output_keys=["density", "temperature", "potential", "aux"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, FFNOBaseline)
    assert model.local_skip_enabled is True
    assert model.fno_impl_version == "factorized_separable_1d_v2_local_skip"


def test_build_ffno_axis_mix_smoke() -> None:
    require_torch_runtime()
    cfg = _spectral_cfg_with_local_skip(enabled=True)
    cfg["axis_mix_cfg"] = {"enabled": True, "init_h": 1.0, "init_w": 1.0}
    model = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=cfg,
    )
    assert model.axis_mix_enabled is True


def test_ffno_static_spatial_features_shape_validation() -> None:
    require_torch_runtime()
    model = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=_spectral_cfg(),
    )
    with pytest.raises(ValueError, match="shape"):
        model.set_static_spatial_features(np.zeros((8, 8, 4), dtype=np.float32))


def test_ffno_forward_shape_with_dynamic_output_keys() -> None:
    require_torch_runtime()
    model = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=_spectral_cfg(),
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


@pytest.mark.parametrize(
    ("local_skip_enabled", "expected_impl"),
    [
        (False, "factorized_separable_1d_v1"),
        (True, "factorized_separable_1d_v2_local_skip"),
    ],
)
def test_ffno_checkpoint_roundtrip(tmp_path: Path, local_skip_enabled: bool, expected_impl: str) -> None:
    require_torch_runtime()
    model = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=_spectral_cfg_with_local_skip(enabled=local_skip_enabled),
    )
    spatial = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(1).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, FFNOBaseline)
    assert loaded.fno_impl_version == expected_impl
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_fno_checkpoint_roundtrip_still_works_after_shared_spectral_cfg(tmp_path: Path) -> None:
    require_torch_runtime()
    model = FNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg={k: v for k, v in _spectral_cfg().items() if k != "factorized_cfg"},
    )
    spatial = np.random.default_rng(2).normal(size=(8, 8, 5)).astype(np.float32)
    cond = np.random.default_rng(3).normal(size=(2, 3)).astype(np.float32)
    model.set_static_spatial_features(spatial)
    pred_before = model.forward(cond)
    save_checkpoint(model, tmp_path / "ckpt_fno")
    loaded = load_checkpoint(tmp_path / "ckpt_fno")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, FNOBaseline)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("local_skip_enabled", [False, True])
def test_ffno_load_legacy_weights_without_axis_mix_scalars(local_skip_enabled: bool) -> None:
    require_torch_runtime()
    model = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=_spectral_cfg_with_local_skip(enabled=local_skip_enabled),
    )
    legacy_like_state = {k: v for k, v in model.state_dict_numpy().items() if not k.endswith(".beta_h") and not k.endswith(".beta_w")}
    target = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        spectral_cfg=_spectral_cfg_with_local_skip(enabled=local_skip_enabled),
    )
    target.load_state_dict_numpy(legacy_like_state)


@pytest.mark.parametrize(
    ("factorized_cfg", "message"),
    [
        ({"enabled": False, "mode": "separable_1d", "share_weights": False}, "enabled must be true"),
        ({"enabled": True, "mode": "separable_1d", "share_weights": True}, "share_weights must be false"),
    ],
)
def test_ffno_rejects_unsupported_factorized_v1_options(
    factorized_cfg: dict[str, object], message: str
) -> None:
    require_torch_runtime()
    with pytest.raises(ValueError, match=message):
        FFNOBaseline(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            n_modes=3,
            spectral_cfg={**_spectral_cfg(), "factorized_cfg": factorized_cfg},
        )


def test_ffno_matches_fno_forward_contract_shape() -> None:
    require_torch_runtime()
    channels = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
    cond = np.random.default_rng(7).normal(size=(2, 3)).astype(np.float32)
    spatial = np.random.default_rng(8).normal(size=(8, 8, len(channels))).astype(np.float32)
    fno = FNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=channels,
        n_modes=3,
        spectral_cfg={k: v for k, v in _spectral_cfg().items() if k != "factorized_cfg"},
    )
    ffno = FFNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=channels,
        n_modes=3,
        spectral_cfg=_spectral_cfg(),
    )
    out_fno = fno.forward(cond, spatial_features=spatial)
    out_ffno = ffno.forward(cond, spatial_features=spatial)
    assert out_fno.shape == out_ffno.shape == (2, 2, 8, 8)
    assert np.isfinite(out_fno).all()
    assert np.isfinite(out_ffno).all()
