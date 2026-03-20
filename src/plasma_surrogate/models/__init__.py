"""Model package (lazy exports to avoid import-time cycles)."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "GlobalMLP",
    "UNetBaseline",
    "FNOBaseline",
    "DeepONetPlasmaOperatorTorch",
    "DeepONetPoissonHeadTorch",
    "BoundaryOperatorTorch",
]

_EXPORT_MAP = {
    "GlobalMLP": "plasma_surrogate.models.mlp.global_mlp",
    "UNetBaseline": "plasma_surrogate.models.unet.simple_unet",
    "FNOBaseline": "plasma_surrogate.models.fno.simple_fno",
    "DeepONetPlasmaOperatorTorch": "plasma_surrogate.models.deeponet.plasma_operator_torch",
    "DeepONetPoissonHeadTorch": "plasma_surrogate.models.deeponet.poisson_head_torch",
    "BoundaryOperatorTorch": "plasma_surrogate.models.deeponet.boundary_operator_torch",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
