"""Shared model-family constants used across train/build/benchmark paths."""

from __future__ import annotations

from typing import Iterable

from plasma_surrogate.core.model_specs import model_names_by_family, model_names_where

GLOBAL_MLP_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("global_mlp")
UNET_FAMILY_MODELS: tuple[str, ...] = (
    model_names_by_family("unet")
    + model_names_by_family("unetpp")
    + model_names_by_family("unet_operator")
)
UNETPP_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("unetpp")
SPECTRAL_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("spectral")
COORD_MLP_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("coord_mlp")
UNO_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("uno")
CNO_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("cno")
POD_DEEPONET_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("pod_deeponet")
GEOM_DEEPONET_SIREN_FAMILY_MODELS: tuple[str, ...] = model_names_by_family("geom_deeponet_siren")
COND_ONLY_TORCH_MODELS: tuple[str, ...] = model_names_where("cond_only_torch")
GRID_TORCH_MODELS: tuple[str, ...] = model_names_where("grid_torch")
MAINLINE_GEOM_PACK_MODELS: tuple[str, ...] = model_names_where("mainline_geom_pack")


def resolve_single_family_model(*, model_names: Iterable[str], family: Iterable[str]) -> str | None:
    family_set = {str(v) for v in family}
    active = [str(name) for name in model_names if str(name) in family_set]
    if len(active) > 1:
        raise ValueError(f"expected at most one active model in family {sorted(family_set)}, got {active}")
    return active[0] if active else None


__all__ = [
    "COND_ONLY_TORCH_MODELS",
    "CNO_FAMILY_MODELS",
    "GLOBAL_MLP_FAMILY_MODELS",
    "GEOM_DEEPONET_SIREN_FAMILY_MODELS",
    "GRID_TORCH_MODELS",
    "MAINLINE_GEOM_PACK_MODELS",
    "COORD_MLP_FAMILY_MODELS",
    "POD_DEEPONET_FAMILY_MODELS",
    "SPECTRAL_FAMILY_MODELS",
    "UNO_FAMILY_MODELS",
    "UNETPP_FAMILY_MODELS",
    "UNET_FAMILY_MODELS",
    "resolve_single_family_model",
]
