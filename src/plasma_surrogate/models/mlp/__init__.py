"""MLP model package."""

from __future__ import annotations

from importlib import import_module

__all__ = ["CoordMLPPODResidual", "CoordMLPTorch", "GlobalMLP", "GlobalVectorMLP"]

_EXPORT_MAP = {
    "CoordMLPPODResidual": "plasma_surrogate.models.mlp.coord_mlp_pod_residual",
    "CoordMLPTorch": "plasma_surrogate.models.mlp.coord_mlp_torch",
    "GlobalMLP": "plasma_surrogate.models.mlp.global_mlp",
    "GlobalVectorMLP": "plasma_surrogate.models.mlp.global_vector_mlp",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
