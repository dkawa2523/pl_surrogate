from __future__ import annotations

import pytest

from plasma_surrogate.core.input_modes import TABLE_ONLY, TABLE_PLUS_STRUCTURE
from plasma_surrogate.core.model_input_policy import (
    ADAPTER_AUTO,
    ADAPTER_COORD_PACK,
    ADAPTER_GRID_PACK,
    ADAPTER_HYBRID_PACK_DESCRIPTOR,
    ADAPTER_NONE,
    resolve_allowed_adapter_modes,
    resolve_effective_adapter_mode,
    resolve_supported_input_modes,
    validate_adapter_mode,
    validate_model_input_mode,
    validate_model_mode_adapter_policy,
)


@pytest.mark.parametrize(
    ("model_name", "expected_modes"),
    [
        ("global_mlp", (TABLE_ONLY,)),
        ("deeponet_pod", (TABLE_ONLY, TABLE_PLUS_STRUCTURE)),
        ("deeponet_plasma_pod", (TABLE_ONLY,)),
        ("geom_deeponet_pod", (TABLE_PLUS_STRUCTURE,)),
        ("unet", (TABLE_PLUS_STRUCTURE,)),
        ("unetpp", (TABLE_PLUS_STRUCTURE,)),
        ("unetpp_attn", (TABLE_PLUS_STRUCTURE,)),
        ("unet_operator_v2", (TABLE_PLUS_STRUCTURE,)),
        ("fno", (TABLE_PLUS_STRUCTURE,)),
        ("ffno", (TABLE_PLUS_STRUCTURE,)),
        ("coord_mlp_fourier", (TABLE_PLUS_STRUCTURE,)),
        ("coord_mlp_siren", (TABLE_PLUS_STRUCTURE,)),
        ("coord_mlp_pod_residual", (TABLE_PLUS_STRUCTURE,)),
        ("u_no", (TABLE_PLUS_STRUCTURE,)),
        ("cno", (TABLE_PLUS_STRUCTURE,)),
        ("cno_operator_unet", (TABLE_PLUS_STRUCTURE,)),
        ("geom_deeponet_siren", (TABLE_PLUS_STRUCTURE,)),
        ("deeponet_plasma", (TABLE_PLUS_STRUCTURE,)),
    ],
)
def test_resolve_supported_input_modes_by_model(model_name: str, expected_modes: tuple[str, ...]) -> None:
    assert resolve_supported_input_modes(model_name) == expected_modes


def test_validate_model_input_mode_rejects_unsupported_pair() -> None:
    with pytest.raises(ValueError, match="model/input_mode mismatch"):
        validate_model_input_mode("global_mlp", TABLE_PLUS_STRUCTURE)


def test_resolve_supported_input_modes_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_supported_input_modes("missing_model")


def test_validate_adapter_mode_success_and_failure() -> None:
    assert ADAPTER_GRID_PACK in set(resolve_allowed_adapter_modes("ffno"))
    validate_adapter_mode("ffno", TABLE_PLUS_STRUCTURE, ADAPTER_GRID_PACK)
    with pytest.raises(ValueError, match="model/adapter mismatch"):
        validate_adapter_mode("ffno", TABLE_PLUS_STRUCTURE, "descriptor_branch")


@pytest.mark.parametrize(
    "model_name",
    [
        "coord_mlp_fourier",
        "coord_mlp_siren",
        "coord_mlp_pod_residual",
        "deeponet_plasma",
    ],
)
def test_coord_pack_models_reject_unimplemented_descriptor_adapters(model_name: str) -> None:
    assert set(resolve_allowed_adapter_modes(model_name)) == {ADAPTER_COORD_PACK, ADAPTER_AUTO}
    with pytest.raises(ValueError, match="model/adapter mismatch"):
        validate_adapter_mode(
            model_name,
            TABLE_PLUS_STRUCTURE,
            ADAPTER_HYBRID_PACK_DESCRIPTOR,
        )


def test_resolve_effective_adapter_mode_auto_mapping() -> None:
    assert resolve_effective_adapter_mode("global_mlp", TABLE_ONLY, "auto") == ADAPTER_NONE
    assert resolve_effective_adapter_mode("deeponet_pod", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_NONE
    assert resolve_effective_adapter_mode("deeponet_plasma_pod", TABLE_ONLY, "auto") == ADAPTER_NONE
    assert resolve_effective_adapter_mode("geom_deeponet_pod", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_HYBRID_PACK_DESCRIPTOR
    assert resolve_effective_adapter_mode("fno", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_GRID_PACK
    assert resolve_effective_adapter_mode("unet_operator_v2", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_GRID_PACK
    assert resolve_effective_adapter_mode("u_no", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_GRID_PACK
    assert resolve_effective_adapter_mode("cno", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_GRID_PACK
    assert resolve_effective_adapter_mode("cno_operator_unet", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_GRID_PACK
    assert resolve_effective_adapter_mode("geom_deeponet_siren", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_HYBRID_PACK_DESCRIPTOR
    assert resolve_effective_adapter_mode("coord_mlp_fourier", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_COORD_PACK
    assert resolve_effective_adapter_mode("coord_mlp_pod_residual", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_COORD_PACK
    assert resolve_effective_adapter_mode("deeponet_plasma", TABLE_PLUS_STRUCTURE, "auto") == ADAPTER_COORD_PACK


def test_validate_model_mode_adapter_policy_rejects_table_only_non_none_adapter() -> None:
    with pytest.raises(ValueError, match="table_only"):
        validate_model_mode_adapter_policy(
            model_name="global_mlp",
            input_mode=TABLE_ONLY,
            adapter_mode=ADAPTER_GRID_PACK,
        )


def test_validate_model_mode_adapter_policy_rejects_pack_required_model_with_none() -> None:
    with pytest.raises(ValueError, match="structure-aware adapter_mode"):
        validate_model_mode_adapter_policy(
            model_name="fno",
            input_mode=TABLE_PLUS_STRUCTURE,
            adapter_mode=ADAPTER_NONE,
        )
