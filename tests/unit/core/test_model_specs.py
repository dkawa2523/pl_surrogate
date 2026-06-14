from __future__ import annotations

from plasma_surrogate.core import model_families
from plasma_surrogate.core.model_input_policy import (
    MODEL_ALLOWED_ADAPTER_MODES,
    MODEL_AUTO_ADAPTER_MODE,
    MODEL_SUPPORTED_INPUT_MODES,
    STRUCTURE_PACK_REQUIRED_MODELS,
)
from plasma_surrogate.core.model_specs import (
    MODEL_SPECS,
    PRODUCT_CATEGORY_BASELINE,
    PRODUCT_CATEGORY_COORDINATE_OPERATOR,
    PRODUCT_CATEGORY_EXPERIMENTAL_ARCHIVE,
    PRODUCT_CATEGORY_GRID_LOCAL,
    PRODUCT_CATEGORY_SPECTRAL_OPERATOR,
    PRODUCT_STATUS_EXPERIMENTAL,
    PRODUCT_STATUS_FIRST_CLASS,
    benchmark_scope_model_map,
    model_names_by_product_category,
    model_names_by_product_status,
)


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


def test_first_class_model_catalog_is_explicit() -> None:
    assert set(model_names_by_product_status(PRODUCT_STATUS_FIRST_CLASS)) == {
        "global_mlp",
        "unet",
        "unetpp",
        "unetpp_attn",
        "fno",
        "ffno",
        "u_no",
        "cno",
        "deeponet_plasma",
        "coord_mlp_siren",
        "geom_deeponet_siren",
    }
    assert set(model_names_by_product_status(PRODUCT_STATUS_EXPERIMENTAL)) == {
        "deeponet_pod",
        "deeponet_plasma_pod",
        "geom_deeponet_pod",
        "unet_operator_v2",
        "coord_mlp_fourier",
        "coord_mlp_pod_residual",
        "cno_operator_unet",
    }


def test_first_class_model_categories_are_explicit() -> None:
    assert set(model_names_by_product_category(PRODUCT_CATEGORY_BASELINE)) == {"global_mlp"}
    assert set(model_names_by_product_category(PRODUCT_CATEGORY_GRID_LOCAL)) == {
        "unet",
        "unetpp",
        "unetpp_attn",
    }
    assert set(model_names_by_product_category(PRODUCT_CATEGORY_SPECTRAL_OPERATOR)) == {
        "fno",
        "ffno",
        "u_no",
        "cno",
    }
    assert set(model_names_by_product_category(PRODUCT_CATEGORY_COORDINATE_OPERATOR)) == {
        "deeponet_plasma",
        "coord_mlp_siren",
        "geom_deeponet_siren",
    }
    assert set(model_names_by_product_category(PRODUCT_CATEGORY_EXPERIMENTAL_ARCHIVE)) == {
        "deeponet_pod",
        "deeponet_plasma_pod",
        "geom_deeponet_pod",
        "unet_operator_v2",
        "coord_mlp_fourier",
        "coord_mlp_pod_residual",
        "cno_operator_unet",
    }


def test_first_class_models_have_isolated_benchmark_scope() -> None:
    scope_models = {model for models in benchmark_scope_model_map().values() for model in models}
    assert set(model_names_by_product_status(PRODUCT_STATUS_FIRST_CLASS)).issubset(scope_models)
    assert all(MODEL_SPECS[name].benchmark_scope for name in model_names_by_product_status(PRODUCT_STATUS_FIRST_CLASS))
    assert not any(MODEL_SPECS[name].benchmark_scope for name in model_names_by_product_status(PRODUCT_STATUS_EXPERIMENTAL))
