"""Canonical model capability specifications.

The project has several execution surfaces: training, inference, benchmarks,
checkpointing, and docs.  Keep stable model capabilities here so new model
families do not require editing scattered policy tables first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from plasma_surrogate.core.input_modes import TABLE_ONLY, TABLE_PLUS_STRUCTURE


ADAPTER_AUTO = "auto"
ADAPTER_NONE = "none"
ADAPTER_GRID_PACK = "grid_pack"
ADAPTER_COORD_PACK = "coord_pack"
ADAPTER_DESCRIPTOR_BRANCH = "descriptor_branch"
ADAPTER_HYBRID_PACK_DESCRIPTOR = "hybrid_pack_descriptor"

ADAPTER_MODES: tuple[str, ...] = (
    ADAPTER_AUTO,
    ADAPTER_NONE,
    ADAPTER_GRID_PACK,
    ADAPTER_COORD_PACK,
    ADAPTER_DESCRIPTOR_BRANCH,
    ADAPTER_HYBRID_PACK_DESCRIPTOR,
)


@dataclass(frozen=True)
class ModelSpec:
    """Static capabilities for one model id."""

    name: str
    family: str
    supported_input_modes: tuple[str, ...]
    allowed_adapter_modes: tuple[str, ...]
    auto_adapter_mode: str
    requires_structure_pack: bool = False
    grid_torch: bool = False
    cond_only_torch: bool = False
    mainline_geom_pack: bool = False


def _spec(
    name: str,
    *,
    family: str,
    supported_input_modes: Iterable[str],
    allowed_adapter_modes: Iterable[str],
    auto_adapter_mode: str,
    requires_structure_pack: bool = False,
    grid_torch: bool = False,
    cond_only_torch: bool = False,
    mainline_geom_pack: bool = False,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family=family,
        supported_input_modes=tuple(supported_input_modes),
        allowed_adapter_modes=tuple(allowed_adapter_modes),
        auto_adapter_mode=auto_adapter_mode,
        requires_structure_pack=requires_structure_pack,
        grid_torch=grid_torch,
        cond_only_torch=cond_only_torch,
        mainline_geom_pack=mainline_geom_pack,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "global_mlp": _spec(
        "global_mlp",
        family="global_mlp",
        supported_input_modes=(TABLE_ONLY,),
        allowed_adapter_modes=(ADAPTER_NONE, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_NONE,
    ),
    "deeponet_pod": _spec(
        "deeponet_pod",
        family="pod_deeponet",
        supported_input_modes=(TABLE_ONLY, TABLE_PLUS_STRUCTURE),
        allowed_adapter_modes=(
            ADAPTER_NONE,
            ADAPTER_DESCRIPTOR_BRANCH,
            ADAPTER_HYBRID_PACK_DESCRIPTOR,
            ADAPTER_AUTO,
        ),
        auto_adapter_mode=ADAPTER_NONE,
        cond_only_torch=True,
    ),
    "deeponet_plasma_pod": _spec(
        "deeponet_plasma_pod",
        family="pod_deeponet",
        supported_input_modes=(TABLE_ONLY,),
        allowed_adapter_modes=(ADAPTER_NONE, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_NONE,
        cond_only_torch=True,
    ),
    "geom_deeponet_pod": _spec(
        "geom_deeponet_pod",
        family="pod_deeponet",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(
            ADAPTER_DESCRIPTOR_BRANCH,
            ADAPTER_HYBRID_PACK_DESCRIPTOR,
            ADAPTER_AUTO,
        ),
        auto_adapter_mode=ADAPTER_HYBRID_PACK_DESCRIPTOR,
        requires_structure_pack=True,
        cond_only_torch=True,
    ),
    "unet": _spec(
        "unet",
        family="unet",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
    ),
    "unetpp": _spec(
        "unetpp",
        family="unetpp",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "unetpp_attn": _spec(
        "unetpp_attn",
        family="unetpp",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "unet_operator_v2": _spec(
        "unet_operator_v2",
        family="unet_operator",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "fno": _spec(
        "fno",
        family="spectral",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "ffno": _spec(
        "ffno",
        family="spectral",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "coord_mlp_fourier": _spec(
        "coord_mlp_fourier",
        family="coord_mlp",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_COORD_PACK, ADAPTER_HYBRID_PACK_DESCRIPTOR, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_COORD_PACK,
        requires_structure_pack=True,
        grid_torch=True,
    ),
    "coord_mlp_siren": _spec(
        "coord_mlp_siren",
        family="coord_mlp",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_COORD_PACK, ADAPTER_HYBRID_PACK_DESCRIPTOR, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_COORD_PACK,
        requires_structure_pack=True,
        grid_torch=True,
    ),
    "coord_mlp_pod_residual": _spec(
        "coord_mlp_pod_residual",
        family="coord_mlp",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_COORD_PACK, ADAPTER_HYBRID_PACK_DESCRIPTOR, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_COORD_PACK,
        requires_structure_pack=True,
        grid_torch=True,
    ),
    "u_no": _spec(
        "u_no",
        family="uno",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "cno": _spec(
        "cno",
        family="cno",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "cno_operator_unet": _spec(
        "cno_operator_unet",
        family="cno",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_GRID_PACK, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_GRID_PACK,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "geom_deeponet_siren": _spec(
        "geom_deeponet_siren",
        family="geom_deeponet_siren",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(ADAPTER_HYBRID_PACK_DESCRIPTOR, ADAPTER_AUTO),
        auto_adapter_mode=ADAPTER_HYBRID_PACK_DESCRIPTOR,
        requires_structure_pack=True,
        grid_torch=True,
        mainline_geom_pack=True,
    ),
    "deeponet_plasma": _spec(
        "deeponet_plasma",
        family="deeponet_plasma",
        supported_input_modes=(TABLE_PLUS_STRUCTURE,),
        allowed_adapter_modes=(
            ADAPTER_COORD_PACK,
            ADAPTER_DESCRIPTOR_BRANCH,
            ADAPTER_HYBRID_PACK_DESCRIPTOR,
            ADAPTER_AUTO,
        ),
        auto_adapter_mode=ADAPTER_COORD_PACK,
        requires_structure_pack=True,
    ),
}


def normalize_model_name(model_name: Any) -> str:
    name = str(model_name).strip().lower()
    if not name:
        raise ValueError("model_name must be a non-empty string")
    return name


def get_model_spec(model_name: Any) -> ModelSpec:
    name = normalize_model_name(model_name)
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model spec: {model_name!r}; known={sorted(MODEL_SPECS)}") from exc


def model_names_by_family(family: str) -> tuple[str, ...]:
    return tuple(name for name, spec in MODEL_SPECS.items() if spec.family == family)


def model_names_where(attr: str) -> tuple[str, ...]:
    return tuple(name for name, spec in MODEL_SPECS.items() if bool(getattr(spec, attr)))


__all__ = [
    "ADAPTER_AUTO",
    "ADAPTER_COORD_PACK",
    "ADAPTER_DESCRIPTOR_BRANCH",
    "ADAPTER_GRID_PACK",
    "ADAPTER_HYBRID_PACK_DESCRIPTOR",
    "ADAPTER_MODES",
    "ADAPTER_NONE",
    "MODEL_SPECS",
    "ModelSpec",
    "get_model_spec",
    "model_names_by_family",
    "model_names_where",
    "normalize_model_name",
]
