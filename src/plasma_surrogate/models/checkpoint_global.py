"""Checkpoint serializer for the tabular GlobalMLP family."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.mlp.global_vector_mlp import (
    GLOBAL_VECTOR_MLP_IMPL_VERSION,
    GLOBAL_VECTOR_MLP_MODEL_TYPES,
    GlobalVectorMLP,
)

GLOBAL_MLP_CHECKPOINT_MODEL_TYPES = frozenset(
    {"global_mlp", *GLOBAL_VECTOR_MLP_MODEL_TYPES}
)

__all__ = [
    "GLOBAL_MLP_CHECKPOINT_MODEL_TYPES",
    "load_global_mlp_checkpoint_model",
    "load_global_mlp_checkpoint_weights",
    "make_global_mlp_checkpoint_meta",
]


def make_global_mlp_checkpoint_meta(model: Any) -> dict[str, Any] | None:
    if isinstance(model, GlobalVectorMLP):
        return model.to_meta()
    if not isinstance(model, GlobalMLP):
        return None
    output_keys = [str(v) for v in list(getattr(model, "output_keys", []))]
    if not output_keys:
        raise ValueError("checkpoint requires model.output_keys")
    return {
        "model_type": "global_mlp",
        "input_dim": model.input_dim,
        "grid_shape": list(model.grid_shape),
        "out_channels": model.out_channels,
        "output_keys": output_keys,
        "hidden": list(getattr(model, "hidden", [])),
        "dropout": float(getattr(model, "dropout", 0.0)),
        "weight_decay": float(getattr(model, "weight_decay", 0.0)),
        "arch_version": str(getattr(model, "arch_version", "linear_v1")),
    }


def load_global_mlp_checkpoint_model(meta: dict[str, Any]) -> GlobalMLP | GlobalVectorMLP | None:
    model_type = str(meta.get("model_type", "")).strip().lower()
    if model_type not in GLOBAL_MLP_CHECKPOINT_MODEL_TYPES:
        return None
    output_keys = [str(v) for v in list(meta.get("output_keys", []))]
    if not output_keys:
        raise ValueError("checkpoint meta requires output_keys")
    if model_type in GLOBAL_VECTOR_MLP_MODEL_TYPES:
        impl_version = str(meta.get("impl_version", "")).strip()
        if impl_version != GLOBAL_VECTOR_MLP_IMPL_VERSION:
            raise ValueError(
                f"Unsupported {model_type} impl_version: {impl_version!r}; "
                f"expected={GLOBAL_VECTOR_MLP_IMPL_VERSION!r}"
            )
        return GlobalVectorMLP(
            model_type=model_type,
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=output_keys,
            model_cfg=dict(meta.get("model_cfg", {})),
        )
    return GlobalMLP(
        input_dim=int(meta["input_dim"]),
        grid_shape=tuple(meta["grid_shape"]),
        out_channels=int(meta.get("out_channels", 3)),
        output_keys=output_keys,
        hidden=list(meta.get("hidden", [128, 128])),
        dropout=float(meta.get("dropout", 0.1)),
        weight_decay=float(meta.get("weight_decay", 0.0)),
    )


def load_global_mlp_checkpoint_weights(
    model: GlobalMLP | GlobalVectorMLP,
    weights: Any,
) -> None:
    if isinstance(model, GlobalMLP) and "W" in weights and "b" in weights:
        raise ValueError("legacy global_mlp checkpoint format is not supported; expected layer*.W/layer*.b")

    state = {k: np.asarray(weights[k], dtype=np.float32) for k in weights.files}
    model.load_state_dict_numpy(state)
