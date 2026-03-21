"""Shared model-family constants used across train/build/benchmark paths."""

from __future__ import annotations

from typing import Iterable


UNET_FAMILY_MODELS: tuple[str, ...] = ("unet", "unetpp", "unetpp_attn")
UNETPP_FAMILY_MODELS: tuple[str, ...] = ("unetpp", "unetpp_attn")
SPECTRAL_FAMILY_MODELS: tuple[str, ...] = ("fno", "ffno")
COORD_MLP_FAMILY_MODELS: tuple[str, ...] = ("coord_mlp_fourier", "coord_mlp_siren")
POD_DEEPONET_FAMILY_MODELS: tuple[str, ...] = ("deeponet_pod",)
COND_ONLY_TORCH_MODELS: tuple[str, ...] = POD_DEEPONET_FAMILY_MODELS
GRID_TORCH_MODELS: tuple[str, ...] = UNET_FAMILY_MODELS + SPECTRAL_FAMILY_MODELS + COORD_MLP_FAMILY_MODELS
MAINLINE_GEOM_PACK_MODELS: tuple[str, ...] = SPECTRAL_FAMILY_MODELS + UNETPP_FAMILY_MODELS


def resolve_single_family_model(*, model_names: Iterable[str], family: Iterable[str]) -> str | None:
    family_set = {str(v) for v in family}
    active = [str(name) for name in model_names if str(name) in family_set]
    if len(active) > 1:
        raise ValueError(f"expected at most one active model in family {sorted(family_set)}, got {active}")
    return active[0] if active else None


__all__ = [
    "COND_ONLY_TORCH_MODELS",
    "GRID_TORCH_MODELS",
    "MAINLINE_GEOM_PACK_MODELS",
    "COORD_MLP_FAMILY_MODELS",
    "POD_DEEPONET_FAMILY_MODELS",
    "SPECTRAL_FAMILY_MODELS",
    "UNETPP_FAMILY_MODELS",
    "UNET_FAMILY_MODELS",
    "resolve_single_family_model",
]
