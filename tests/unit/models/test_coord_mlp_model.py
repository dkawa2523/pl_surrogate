from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.infer.predictor import InferencePredictor
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.pod_deeponet_torch import fit_pod_basis_from_targets
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import CoordMLPPODResidual
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint



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
        "siren": {
            "enabled": True,
            "fusion": "split_add",
            "w0_initial": 10.0,
            "w0_hidden": 1.0,
            "fusion_cfg": {"cond_gain_init": 0.7, "point_gain_init": 1.3, "branch_norm": True},
        },
    }


def _pod_residual_cfg() -> dict[str, object]:
    return {
        "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True},
        "cond_hidden": [12, 12],
        "latent_dim": 8,
        "residual_hidden": [16, 16],
        "residual_activation": "gelu",
        "point_encoder": {
            "xy_fourier_frequencies": 2,
            "xy_frequency_scale": 10.0,
            "include_xy_raw": True,
            "include_aux_raw": True,
        },
        "coeff_loss_weight": 0.1,
        "residual_scale_init": 0.05,
    }


def _spatial_features(h: int = 8, w: int = 8) -> np.ndarray:
    spatial = np.zeros((h, w, 5), dtype=np.float32)
    spatial[..., 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    spatial[..., 1] = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    spatial[..., 2] = 1.0
    spatial[..., 3] = 0.5
    spatial[..., 4] = 0.25
    return spatial


def _pod_basis_bundle() -> object:
    rng = np.random.default_rng(123)
    y = rng.normal(size=(6, 4, 8, 8)).astype(np.float32)
    return fit_pod_basis_from_targets(
        y,
        output_keys=["ne", "ni", "Te", "phi"],
        requested_rank=3,
        center=True,
        per_var=True,
    )


def test_build_coord_mlp_fourier_smoke() -> None:
    require_torch_runtime()
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
    require_torch_runtime()
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


def test_build_coord_mlp_pod_residual_smoke() -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="coord_mlp_pod_residual",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_pod_residual_cfg(),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        pod_basis_bundle=_pod_basis_bundle(),
    )
    assert isinstance(model, CoordMLPPODResidual)
    assert model.output_keys == ["ne", "ni", "Te", "phi"]
    assert model.model_type == "coord_mlp_pod_residual"


def test_coord_mlp_static_spatial_features_shape_validation() -> None:
    require_torch_runtime()
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
    require_torch_runtime()
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
    require_torch_runtime()
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


def test_coord_mlp_pod_residual_forward_backward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="coord_mlp_pod_residual",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_pod_residual_cfg(),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        pod_basis_bundle=_pod_basis_bundle(),
    )
    model.set_static_spatial_features(_spatial_features())
    cond = np.ones((2, 3), dtype=np.float32)
    pred = model.forward_raw(cond, training=True)
    assert pred.shape == (2, 4, 8, 8)
    cached_out = model._torch_last_out
    cached_coeff = model._torch_last_coeff_norm
    aux_diag = model.auxiliary_loss_diagnostics(
        np.zeros_like(pred, dtype=np.float32),
        loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": 0.1}},
    )
    assert aux_diag["coeff_loss"] == aux_diag["loss_aux_coeff"]
    assert aux_diag["loss_aux_coeff_weighted"] == pytest.approx(0.1 * aux_diag["loss_aux_coeff"])
    assert aux_diag["loss_aux_total"] == aux_diag["loss_aux_coeff_weighted"]
    assert model._torch_last_out is cached_out
    assert model._torch_last_coeff_norm is cached_coeff
    diag = model.backward_raw(
        np.ones_like(pred, dtype=np.float32) * 0.01,
        lr=1.0e-3,
        target_raw=np.zeros_like(pred, dtype=np.float32),
        loss_cfg={"coord_mlp_pod_residual": {"coeff_loss_weight": 0.1}},
    )
    assert diag["coeff_loss"] >= 0.0
    for key in ("coeff_loss", "loss_aux_coeff", "loss_aux_coeff_weighted", "loss_aux_total"):
        assert diag[key] == pytest.approx(aux_diag[key])

    ckpt_dir = tmp_path / "coord_mlp_pod_residual_ckpt"
    save_checkpoint(model, ckpt_dir)
    loaded = load_checkpoint(ckpt_dir)
    assert isinstance(loaded, CoordMLPPODResidual)
    loaded.set_static_spatial_features(_spatial_features())
    out = loaded.forward_raw(cond)
    assert out.shape == (2, 4, 8, 8)


def test_coord_mlp_pod_residual_capability_routes_product_inference(tmp_path: Path) -> None:
    require_torch_runtime()
    model = build_model_from_name(
        model_name="coord_mlp_pod_residual",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_pod_residual_cfg(),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        pod_basis_bundle=_pod_basis_bundle(),
    )

    class _FeatureBuilder:
        def __init__(self) -> None:
            self.scaling_required = False

        def require_coord_feature_scaling_enabled(self) -> None:
            self.scaling_required = True

        @staticmethod
        def resolve_coord_feature_channels(candidate) -> list[str]:
            return list(candidate.input_feature_channels)

        @staticmethod
        def build_grid_feature_rows(geom, channels) -> np.ndarray:
            del geom
            assert channels == ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
            return _spatial_features().reshape(-1, len(channels))

    checkpoint = save_checkpoint(model, tmp_path / "pod_residual_inference")
    loaded = load_checkpoint(checkpoint)
    feature_builder = _FeatureBuilder()
    predictor = InferencePredictor(model=loaded, feature_builder=feature_builder)
    geom = type("Geom", (), {"mask_plasma": np.ones((8, 8), dtype=np.float32)})()
    fields = predictor.predict_fields(np.ones((3,), dtype=np.float32), geom)

    assert loaded.requires_spatial_features is True
    assert loaded.requires_scaled_spatial_features is True
    assert feature_builder.scaling_required is True
    assert set(fields) == {"ne", "ni", "Te", "phi"}
    assert all(np.asarray(field).shape == (1, 8, 8) for field in fields.values())


def test_coord_mlp_accepts_batched_spatial_features() -> None:
    require_torch_runtime()
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
    require_torch_runtime()
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
    require_torch_runtime()
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
    require_torch_runtime()
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
    bad_cfg["siren"] = {"enabled": False, "w0_initial": 10.0, "w0_hidden": 1.0}
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
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["fusion"] = "concat"
    with pytest.raises(ValueError, match="siren.fusion must be one of"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["fusion_cfg"]["cond_gain_init"] = 0.0
    with pytest.raises(ValueError, match="fusion_cfg.cond_gain_init must be > 0"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["fusion_cfg"]["point_gain_init"] = 0.0
    with pytest.raises(ValueError, match="fusion_cfg.point_gain_init must be > 0"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )
    bad_cfg = _siren_model_cfg()
    bad_cfg["siren"]["fusion_cfg"]["branch_norm"] = "true"
    with pytest.raises(ValueError, match="fusion_cfg.branch_norm must be a boolean"):
        CoordMLPTorch(
            input_dim=3,
            grid_shape=(8, 8),
            out_channels=2,
            output_keys=["density", "temperature"],
            input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
            model_cfg=bad_cfg,
        )


def test_coord_mlp_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
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
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CoordMLPTorch)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_coord_mlp_siren_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_torch_runtime()
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
    save_checkpoint(model, tmp_path / "ckpt_siren")
    loaded = load_checkpoint(tmp_path / "ckpt_siren")
    loaded.set_static_spatial_features(spatial)
    pred_after = loaded.forward(cond)
    assert isinstance(loaded, CoordMLPTorch)
    assert loaded.model_type == "coord_mlp_siren"
    assert loaded.model_cfg["embedding"] == {"type": "none"}
    assert loaded.model_cfg["siren"]["fusion"] == "split_add"
    assert loaded.model_cfg["siren"]["fusion_cfg"]["branch_norm"] is True
    assert loaded.model_cfg["siren"]["fusion_cfg"]["cond_gain_init"] == pytest.approx(0.7)
    assert loaded.model_cfg["siren"]["fusion_cfg"]["point_gain_init"] == pytest.approx(1.3)
    assert "decoder_activation" not in loaded.model_cfg
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6)


def test_coord_mlp_rejects_legacy_checkpoint_meta(tmp_path: Path) -> None:
    require_torch_runtime()
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    save_checkpoint(model, tmp_path / "ckpt_legacy")
    meta_path = tmp_path / "ckpt_legacy" / "meta.json"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("coord_mlp_impl_version", None)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="coord_mlp_impl_version=v4_siren_branch_balanced"):
        _ = load_checkpoint(tmp_path / "ckpt_legacy")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["coord_mlp_impl_version"] = "v3_siren_split_add"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="coord_mlp_impl_version=v4_siren_branch_balanced"):
        _ = load_checkpoint(tmp_path / "ckpt_legacy")


def test_coord_mlp_siren_uses_split_add_point_decoder() -> None:
    require_torch_runtime()
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_siren_model_cfg(),
    )
    cond_has_sine = any(type(module).__name__ == "_SineActivation" for module in model.net.cond_encoder)
    assert cond_has_sine is False
    assert type(model.net.point_decoder).__name__ == "_SirenSplitAddPointDecoder"


def test_coord_mlp_siren_gated_affine_decoder_forward_is_finite() -> None:
    require_torch_runtime()
    cfg = _siren_model_cfg()
    cfg["siren"]["fusion"] = "gated_affine"
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=cfg,
    )
    spatial = np.zeros((8, 8, 5), dtype=np.float32)
    spatial[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :]
    spatial[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    spatial[..., 2] = 1.0
    spatial[..., 3] = 0.5
    spatial[..., 4] = 0.25
    model.set_static_spatial_features(spatial)
    out = model.forward(np.ones((2, 3), dtype=np.float32))
    assert out.shape == (2, 2, 8, 8)
    assert np.isfinite(out).all()
    assert type(model.net.point_decoder).__name__ == "_SirenGatedAffinePointDecoder"


def test_coord_mlp_v4_has_decoder_input_norm_and_residual_head() -> None:
    require_torch_runtime()
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    assert hasattr(model.net, "decoder_input_norm")
    assert model.net.decoder_input_norm is not None
    assert hasattr(model.net, "residual_head")
    assert model.net.residual_head is not None
    assert hasattr(model.net, "residual_scale")
    assert model.coord_mlp_impl_version == "v4_siren_branch_balanced"


def test_coord_mlp_device_selection_matches_torch_runtime() -> None:
    require_torch_runtime()
    model = CoordMLPTorch(
        input_dim=3,
        grid_shape=(8, 8),
        out_channels=2,
        output_keys=["density", "temperature"],
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        model_cfg=_model_cfg(),
    )
    expected = "cuda" if bool(model.torch.cuda.is_available()) else "cpu"
    assert str(model.device.type) == expected


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
