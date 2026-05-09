from __future__ import annotations

from plasma_surrogate.core import model_families
from plasma_surrogate.core.model_input_policy import (
    MODEL_ALLOWED_ADAPTER_MODES,
    MODEL_AUTO_ADAPTER_MODE,
    MODEL_SUPPORTED_INPUT_MODES,
    STRUCTURE_PACK_REQUIRED_MODELS,
)
from plasma_surrogate.core.model_specs import MODEL_SPECS


def test_model_policy_tables_are_derived_from_model_specs() -> None:
    assert MODEL_SUPPORTED_INPUT_MODES == {name: spec.supported_input_modes for name, spec in MODEL_SPECS.items()}
    assert MODEL_ALLOWED_ADAPTER_MODES == {name: spec.allowed_adapter_modes for name, spec in MODEL_SPECS.items()}
    assert MODEL_AUTO_ADAPTER_MODE == {name: spec.auto_adapter_mode for name, spec in MODEL_SPECS.items()}
    assert STRUCTURE_PACK_REQUIRED_MODELS == tuple(
        name for name, spec in MODEL_SPECS.items() if spec.requires_structure_pack
    )


def test_model_family_constants_are_derived_from_model_specs() -> None:
    assert model_families.SPECTRAL_FAMILY_MODELS == ("fno", "ffno")
    assert model_families.UNET_FAMILY_MODELS == ("unet", "unetpp", "unetpp_attn", "unet_operator_v2")
    assert model_families.CNO_FAMILY_MODELS == ("cno", "cno_operator_unet")
    assert model_families.COND_ONLY_TORCH_MODELS == (
        "deeponet_pod",
        "deeponet_plasma_pod",
        "geom_deeponet_pod",
    )
    assert "deeponet_plasma" not in model_families.MAINLINE_GEOM_PACK_MODELS
