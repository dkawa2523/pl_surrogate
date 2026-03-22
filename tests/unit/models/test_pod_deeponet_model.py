from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from plasma_surrogate.core.torch_backend import torch_runtime_available
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.deeponet.pod_deeponet_torch import (
    PODDeepONetTorch,
    POD_DEEPONET_IMPL_VERSION,
    fit_pod_basis_from_targets,
    normalize_pod_deeponet_model_cfg,
)
from plasma_surrogate.models.mlp.io import load_mlp_checkpoint, save_mlp_checkpoint


def _enable_torch() -> None:
    os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"


def _build_targets(n: int = 6, h: int = 4, w: int = 4) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    out = np.zeros((n, 2, h, w), dtype=np.float32)
    for i in range(n):
        out[i, 0] = (i + 1) * xv + 0.1 * yv
        out[i, 1] = (i + 1) * yv - 0.05 * xv
    return out


def test_fit_pod_basis_helper_clamps_rank() -> None:
    y = _build_targets(n=3, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=32,
        center=True,
        per_var=True,
    )
    assert set(bundle.basis_by_var.keys()) == {"density", "temperature"}
    assert bundle.basis_by_var["density"].shape == (3, 4, 4)
    assert bundle.mean_by_var["temperature"].shape == (4, 4)
    assert bundle.rank_by_var == {"density": 3, "temperature": 3}
    assert bundle.coeff_std_by_var["density"].shape == (3,)
    assert np.all(bundle.coeff_std_by_var["density"] > 0.0)


def test_normalize_pod_model_cfg_canonicalizes_hidden() -> None:
    cfg = normalize_pod_deeponet_model_cfg(
        {"hidden_dim": 16, "latent_dim": 12, "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True}},
        model_type="deeponet_pod",
    )
    assert cfg["hidden"] == [16, 12]
    assert "hidden_dim" not in cfg
    assert cfg["basis"]["fit_scope"] == "train_only"


def test_pod_deeponet_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    y = _build_targets(n=5, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=4,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden_dim": 16, "latent_dim": 12, "basis": {"rank": 4, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    cond = np.random.default_rng(3).normal(size=(2, 3)).astype(np.float32)
    pred = model.forward(cond)
    assert pred.shape == (2, 2, 4, 4)
    assert np.all(np.isfinite(pred))

    ckpt = save_mlp_checkpoint(model, tmp_path / "ckpt")
    loaded = load_mlp_checkpoint(ckpt)
    pred_loaded = loaded.forward(cond)
    assert pred_loaded.shape == pred.shape
    assert loaded.to_meta()["basis_rank_by_var"] == {"density": bundle.rank_by_var["density"], "temperature": bundle.rank_by_var["temperature"]}
    assert np.allclose(loaded.basis_bundle_numpy().basis_by_var["density"], bundle.basis_by_var["density"])
    assert loaded.to_meta()["impl_version"] == POD_DEEPONET_IMPL_VERSION
    assert "coeff_std_by_var" in loaded.to_meta()


def test_pod_deeponet_rejects_legacy_checkpoint_format(tmp_path: Path) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")
    y = _build_targets(n=4, h=4, w=4)
    bundle = fit_pod_basis_from_targets(
        y,
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden": [16, 12], "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    ckpt = save_mlp_checkpoint(model, tmp_path / "ckpt_legacy")
    meta_path = Path(ckpt) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.pop("impl_version", None)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with pytest.raises(ValueError, match="expected impl_version"):
        load_mlp_checkpoint(ckpt)


def test_pod_deeponet_inference_does_not_use_grid_spatial_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_torch()
    if not torch_runtime_available(refresh=True):
        pytest.skip("torch backend disabled for this environment")

    bundle = fit_pod_basis_from_targets(
        _build_targets(n=4, h=4, w=4),
        output_keys=["density", "temperature"],
        requested_rank=3,
        center=True,
        per_var=True,
    )
    model = PODDeepONetTorch(
        input_dim=3,
        grid_shape=(4, 4),
        out_channels=2,
        output_keys=["density", "temperature"],
        pod_basis_bundle=bundle,
        model_cfg={"hidden": [16, 12], "basis": {"rank": 3, "fit_scope": "train_only", "per_var": True, "center": True}},
    )
    engine = InferenceEngine(
        model=model,
        cond_schema=SimpleNamespace(),
        axis_schema=SimpleNamespace(mode="steady"),
        geometry_provider=SimpleNamespace(),
        output_dir=tmp_path / "infer",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("grid-spatial path should not be used for deeponet_pod")

    monkeypatch.setattr(engine, "_predict_grid_spatial_fields", _fail)
    out = engine._predict_fields(
        np.zeros((3,), dtype=np.float32),
        geom=SimpleNamespace(mask_plasma=np.ones((4, 4), dtype=np.float32)),
    )
    assert set(out.keys()) == {"density", "temperature"}
