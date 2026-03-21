"""MLP model package with lazy exports to avoid import-time cycles."""

from __future__ import annotations

from importlib import import_module

__all__ = ["CoordMLPTorch", "GlobalMLP", "save_mlp_checkpoint", "load_mlp_checkpoint"]

_EXPORT_MAP = {
    "CoordMLPTorch": "plasma_surrogate.models.mlp.coord_mlp_torch",
    "GlobalMLP": "plasma_surrogate.models.mlp.global_mlp",
    "save_mlp_checkpoint": "plasma_surrogate.models.mlp.io",
    "load_mlp_checkpoint": "plasma_surrogate.models.mlp.io",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
