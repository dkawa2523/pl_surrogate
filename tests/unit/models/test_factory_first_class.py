from __future__ import annotations

import pytest

from plasma_surrogate.core.model_specs import (
    MODEL_SPECS,
    PRODUCT_STATUS_FIRST_CLASS,
    get_model_spec,
    model_names_by_product_status,
)
from plasma_surrogate.models.cno.simple_cno import CNOBaseline
from plasma_surrogate.models.deeponet.geom_deeponet_siren import GeomDeepONetSIREN
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
from plasma_surrogate.models.factory import build_model_from_name
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline
from tests._runtime_requirements import require_torch_runtime


pytestmark = pytest.mark.torch_runtime


def _cfg_for(model_name: str) -> dict[str, object]:
    if model_name == "global_mlp":
        return {"hidden": [16, 16], "dropout": 0.0}
    if model_name == "unet":
        return {
            "backend": "torch",
            "conv_cfg": {"base_channels": 4, "depth": 1},
            "output_heads": {"mode": "shared"},
        }
    if model_name in {"unetpp", "unetpp_attn"}:
        return {
            "backend": "torch",
            "conv_cfg": {"base_channels": 4, "depth": 2},
            "output_heads": {"mode": "shared"},
        }
    if model_name in {"fno", "ffno"}:
        return {
            "backend": "torch",
            "n_modes": 2,
            "spectral_cfg": {"width": 8, "n_layers": 1},
        }
    if model_name == "u_no":
        return {
            "backend": "torch",
            "n_modes": 2,
            "uno_cfg": {"width": 8, "n_layers": 1, "dropout": 0.0},
        }
    if model_name == "cno":
        return {
            "backend": "torch",
            "cno_cfg": {"width": 8, "n_layers": 1, "dropout": 0.0, "kernel_size": 3},
        }
    if model_name == "deeponet_plasma":
        return {
            "latent_dim": 8,
            "hidden_dim": 16,
            "trunk_input_mode": "geom_feature_pack",
            "missing_geom_feature_policy": "error",
        }
    if model_name == "coord_mlp_siren":
        return {
            "cond_hidden": [12],
            "latent_dim": 8,
            "decoder_hidden": [16],
            "embedding": {"type": "none"},
            "siren": {
                "enabled": True,
                "fusion": "split_add",
                "w0_initial": 10.0,
                "w0_hidden": 1.0,
            },
        }
    if model_name == "geom_deeponet_siren":
        return {
            "backend": "torch",
            "geom_deeponet_siren_cfg": {
                "latent_dim": 8,
                "trunk_hidden": 16,
                "trunk_layers": 1,
                "branch_hidden": 16,
                "branch_layers": 1,
                "dropout": 0.0,
                "trunk_w0": 10.0,
            },
        }
    raise AssertionError(f"missing test cfg for {model_name}")


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("global_mlp", GlobalMLP),
        ("unet", UNetBaseline),
        ("unetpp", UNetPPBaseline),
        ("unetpp_attn", UNetPPBaseline),
        ("fno", FNOBaseline),
        ("ffno", FFNOBaseline),
        ("u_no", UNOBaseline),
        ("cno", CNOBaseline),
        ("deeponet_plasma", DeepONetPlasmaOperatorTorch),
        ("coord_mlp_siren", CoordMLPTorch),
        ("geom_deeponet_siren", GeomDeepONetSIREN),
    ],
)
def test_build_first_class_model_from_name(model_name: str, expected_type: type[object]) -> None:
    require_torch_runtime()
    input_dim = 7 if model_name in {"deeponet_plasma", "geom_deeponet_siren"} else 3
    model = build_model_from_name(
        model_name=model_name,
        input_dim=input_dim,
        grid_shape=(8, 8),
        model_cfg=_cfg_for(model_name),
        out_channels=4,
        output_keys=["density", "ion_density", "temperature", "potential"],
        unet_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
    )
    assert isinstance(model, expected_type)


def test_factory_first_class_set_matches_model_specs() -> None:
    first_class = set(model_names_by_product_status(PRODUCT_STATUS_FIRST_CLASS))
    assert first_class == {
        name
        for name, spec in MODEL_SPECS.items()
        if spec.product_status == PRODUCT_STATUS_FIRST_CLASS
    }
    for model_name in first_class:
        assert get_model_spec(model_name).family


def test_unknown_model_fails_through_model_specs() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        build_model_from_name(
            model_name="not_a_model",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg={},
        )
