"""Train-dispatch adapter registry.

The adapter is the train-time execution lane for a model family.  Keep this
small and declarative so a new model can be wired by choosing the closest lane
instead of editing scattered top-level dispatch checks first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_surrogate.core.model_families import GRID_TORCH_MODELS, POD_DEEPONET_FAMILY_MODELS
from plasma_surrogate.core.model_specs import normalize_model_name


TRAIN_ADAPTER_DEEPONET_PLASMA = "deeponet_plasma"
TRAIN_ADAPTER_GLOBAL_MLP = "global_mlp"
TRAIN_ADAPTER_GRID_TORCH = "grid_torch"
TRAIN_ADAPTER_POD_DEEPONET = "pod_deeponet"


@dataclass(frozen=True)
class ModelAdapter:
    """Static train-dispatch lane for one or more model ids."""

    name: str
    model_names: tuple[str, ...]
    config_key: str
    runner_name: str

    def matches(self, model_name: Any) -> bool:
        return normalize_model_name(model_name) in self.model_names

    def run(self, ctx: Any) -> Any:
        from plasma_surrogate.train import model_dispatch

        runner = getattr(model_dispatch, self.runner_name)
        return runner(ctx, adapter=self)


MODEL_ADAPTERS: tuple[ModelAdapter, ...] = (
    ModelAdapter(
        name=TRAIN_ADAPTER_GLOBAL_MLP,
        model_names=("global_mlp",),
        config_key="global_mlp",
        runner_name="_run_global_mlp_train_predict",
    ),
    ModelAdapter(
        name=TRAIN_ADAPTER_POD_DEEPONET,
        model_names=tuple(POD_DEEPONET_FAMILY_MODELS),
        config_key="deeponet_pod",
        runner_name="_run_pod_deeponet_train_predict",
    ),
    ModelAdapter(
        name=TRAIN_ADAPTER_GRID_TORCH,
        model_names=tuple(GRID_TORCH_MODELS),
        config_key="model_name",
        runner_name="_run_grid_torch_adapter_train_predict",
    ),
    ModelAdapter(
        name=TRAIN_ADAPTER_DEEPONET_PLASMA,
        model_names=("deeponet_plasma",),
        config_key="deeponet_plasma",
        runner_name="_run_deeponet_plasma_train_predict",
    ),
)


def resolve_train_model_adapter(model_name: Any) -> ModelAdapter:
    name = normalize_model_name(model_name)
    for adapter in MODEL_ADAPTERS:
        if name in adapter.model_names:
            return adapter
    known = sorted({model for adapter in MODEL_ADAPTERS for model in adapter.model_names})
    raise ValueError(f"Unsupported model in train dispatch: {model_name!r}; known={known}")


__all__ = [
    "MODEL_ADAPTERS",
    "ModelAdapter",
    "TRAIN_ADAPTER_DEEPONET_PLASMA",
    "TRAIN_ADAPTER_GLOBAL_MLP",
    "TRAIN_ADAPTER_GRID_TORCH",
    "TRAIN_ADAPTER_POD_DEEPONET",
    "resolve_train_model_adapter",
]
