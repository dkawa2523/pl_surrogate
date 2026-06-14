from __future__ import annotations

import pytest
import numpy as np

from plasma_surrogate.core.torch_backend import require_torch
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch

pytestmark = pytest.mark.torch_runtime


def test_deeponet_plasma_torch_forward_shapes():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["density", "temperature", "potential"],
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    v = torch.rand((2, 16, 5), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    out = model.forward(sensors={"x": x, "v": v}, query={"x": x, "f": f}, cond=cond)
    assert set(out.keys()) == {"density", "temperature", "potential"}
    assert tuple(out["potential"].shape) == (2, 16, 1)


def test_deeponet_plasma_torch_forward_geom_set_pool_shapes():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        sensor_pool_mode="set_mlp_pool",
        sensor_embed_dim=16,
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    v = torch.rand((2, 16, 5), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    out = model.forward(sensors={"x": x, "v": v}, query={"x": x, "f": f}, cond=cond)
    assert set(out.keys()) == {"ne", "ni", "Te", "phi"}
    assert tuple(out["ne"].shape) == (2, 16, 1)


def test_deeponet_plasma_torch_forward_fused_path_shapes():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        branch_mode="cond_only",
        output_path_mode="fused",
        output_path_dot_skip=0.25,
        output_path_fused_hidden_dim=32,
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    out = model.forward(sensors={"x": None, "v": None}, query={"x": x, "f": f}, cond=cond)
    assert set(out.keys()) == {"ne", "ni", "Te", "phi"}
    assert tuple(out["phi"].shape) == (2, 16, 1)


def test_deeponet_plasma_torch_forward_fused_global_local_shapes():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        branch_mode="cond_only",
        output_path_mode="fused",
        output_path_dot_skip=0.25,
        output_path_fused_hidden_dim=32,
        output_path_global_local_enabled=True,
        output_path_global_hidden_dim=16,
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    out = model.forward(sensors={"x": None, "v": None}, query={"x": x, "f": f}, cond=cond)
    assert set(out.keys()) == {"ne", "ni", "Te", "phi"}
    assert tuple(out["ne"].shape) == (2, 16, 1)
    vals = torch.cat([out["ne"], out["ni"], out["Te"], out["phi"]], dim=2)
    assert torch.isfinite(vals).all()


def test_deeponet_plasma_torch_rejects_invalid_dot_skip_mode():
    require_torch_runtime()
    with pytest.raises(ValueError, match="output_path_dot_skip_mode"):
        DeepONetPlasmaOperatorTorch(
            cond_dim=5,
            grid_shape=(8, 8),
            output_keys=["ne", "ni", "Te", "phi"],
            output_path_mode="fused",
            output_path_dot_skip_mode="invalid",
        )


def test_deeponet_plasma_torch_forward_fused_learned_per_var_dot_skip_shapes():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        branch_mode="cond_only",
        output_path_mode="fused",
        output_path_dot_skip=0.25,
        output_path_dot_skip_mode="learned_per_var",
        output_path_fused_hidden_dim=32,
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    out = model.forward(sensors={"x": None, "v": None}, query={"x": x, "f": f}, cond=cond)
    assert set(out.keys()) == {"ne", "ni", "Te", "phi"}
    assert tuple(out["phi"].shape) == (2, 16, 1)
    assert model.output_path_dot_skip_logits is not None
    assert model.to_meta()["output_path_dot_skip_mode"] == "learned_per_var"
    vals = torch.cat([out["ne"], out["ni"], out["Te"], out["phi"]], dim=2)
    assert torch.isfinite(vals).all()


def test_deeponet_plasma_torch_learned_per_var_dot_skip_state_roundtrip():
    require_torch_runtime()
    torch = require_torch()
    kwargs = {
        "cond_dim": 5,
        "grid_shape": (8, 8),
        "output_keys": ["ne", "ni", "Te", "phi"],
        "trunk_input_mode": "geom_feature_pack",
        "branch_mode": "cond_only",
        "output_path_mode": "fused",
        "output_path_dot_skip": 0.25,
        "output_path_dot_skip_mode": "learned_per_var",
        "output_path_fused_hidden_dim": 32,
    }
    model_a = DeepONetPlasmaOperatorTorch(**kwargs, seed=3)
    model_b = DeepONetPlasmaOperatorTorch(**kwargs, seed=19)
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    f = torch.rand((2, 16, 3), dtype=torch.float32)
    pred_a = model_a.forward(sensors={"x": None, "v": None}, query={"x": x, "f": f}, cond=cond)
    state = model_a.state_dict_numpy()
    model_b.load_state_dict_numpy(state)
    pred_b = model_b.forward(sensors={"x": None, "v": None}, query={"x": x, "f": f}, cond=cond)
    for key in ("ne", "ni", "Te", "phi"):
        np.testing.assert_allclose(
            pred_a[key].detach().cpu().numpy(),
            pred_b[key].detach().cpu().numpy(),
            atol=1e-6,
            rtol=1e-6,
        )


def test_deeponet_plasma_cond_only_skips_sensor_value_build():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        branch_mode="cond_only",
        output_path_mode="fused",
        output_path_fused_hidden_dim=32,
    )

    def _raise_sensor(*args, **kwargs):
        raise RuntimeError("sensor path must not be called in cond_only mode")

    model._sensor_values_from_geom = _raise_sensor  # type: ignore[method-assign]

    class _Geom:
        pass

    geom = _Geom()
    h, w = model.grid_shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    geom.coord_grid = np.stack([xv, yv], axis=0).astype(np.float32)
    geom.mask_plasma = np.ones((h, w), dtype=np.float32)
    geom.distance_signed = np.ones((h, w), dtype=np.float32)
    geom.distance_any = np.ones((h, w), dtype=np.float32)

    cond = torch.rand((2, 5), dtype=torch.float32)
    out = model.predict_fields_torch(cond, geom_ctx=geom)
    assert set(out.keys()) == {"ne", "ni", "Te", "phi"}


def test_deeponet_plasma_missing_query_features_policy_error_raises():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        trunk_input_mode="geom_feature_pack",
        branch_mode="cond_only",
        missing_geom_feature_policy="error",
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="query.f"):
        model.forward(sensors={"x": None, "v": None}, query={"x": x}, cond=cond)


def test_deeponet_plasma_missing_sensor_features_policy_error_raises():
    require_torch_runtime()
    torch = require_torch()
    model = DeepONetPlasmaOperatorTorch(
        cond_dim=5,
        grid_shape=(8, 8),
        output_keys=["ne", "ni", "Te", "phi"],
        branch_mode="moments",
        missing_geom_feature_policy="error",
    )
    cond = torch.rand((2, 5), dtype=torch.float32)
    x = torch.rand((2, 16, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="sensors.v"):
        model.forward(sensors={"x": x, "v": None}, query={"x": x}, cond=cond)
