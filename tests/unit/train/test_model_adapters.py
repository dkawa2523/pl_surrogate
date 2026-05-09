from __future__ import annotations

import pytest

from plasma_surrogate.core.model_specs import MODEL_SPECS
from plasma_surrogate.train.model_adapters import (
    MODEL_ADAPTERS,
    TRAIN_ADAPTER_DEEPONET_PLASMA,
    TRAIN_ADAPTER_GLOBAL_MLP,
    TRAIN_ADAPTER_GRID_TORCH,
    TRAIN_ADAPTER_POD_DEEPONET,
    resolve_train_model_adapter,
)


def test_train_model_adapters_cover_known_model_specs_once() -> None:
    mapped = [name for adapter in MODEL_ADAPTERS for name in adapter.model_names]
    assert sorted(mapped) == sorted(MODEL_SPECS)
    assert len(mapped) == len(set(mapped))


@pytest.mark.parametrize(
    ("model_name", "adapter_name"),
    [
        ("global_mlp", TRAIN_ADAPTER_GLOBAL_MLP),
        ("deeponet_pod", TRAIN_ADAPTER_POD_DEEPONET),
        ("deeponet_plasma_pod", TRAIN_ADAPTER_POD_DEEPONET),
        ("geom_deeponet_pod", TRAIN_ADAPTER_POD_DEEPONET),
        ("unetpp", TRAIN_ADAPTER_GRID_TORCH),
        ("ffno", TRAIN_ADAPTER_GRID_TORCH),
        ("coord_mlp_siren", TRAIN_ADAPTER_GRID_TORCH),
        ("coord_mlp_pod_residual", TRAIN_ADAPTER_GRID_TORCH),
        ("deeponet_plasma", TRAIN_ADAPTER_DEEPONET_PLASMA),
    ],
)
def test_resolve_train_model_adapter(model_name: str, adapter_name: str) -> None:
    adapter = resolve_train_model_adapter(model_name)
    assert adapter.name == adapter_name
    assert adapter.matches(model_name)


def test_resolve_train_model_adapter_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model in train dispatch"):
        resolve_train_model_adapter("old_experimental_model")
