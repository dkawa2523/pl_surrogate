"""Case-aligned raw geometry for spatial losses and checkpoint selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.boundary_distance import (
    DEFAULT_BOUNDARY_DISTANCE_CHANNELS,
    boundary_distance_channels_from_loss_cfg,
    compose_boundary_distance,
    normalize_boundary_distance_channels,
)
from plasma_surrogate.preprocessing.spatial_features import materialize_raw_spatial_batch
from plasma_surrogate.train.selection import SPATIAL_SELECTION_MODE, resolve_spatial_selection_config


def objective_boundary_distance_channels(
    *,
    loss_cfg: dict[str, Any] | None,
    selection_cfg: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Resolve the boundary channels only when an active objective consumes them."""

    supervised = dict(dict(loss_cfg or {}).get("supervised", {}) or {})
    spatial_loss = dict(supervised.get("spatial", {}) or {})
    active = float(spatial_loss.get("boundary_weight", 0.0)) > 0.0
    selection = dict(selection_cfg or {})
    if str(selection.get("mode", "last")).strip().lower() == SPATIAL_SELECTION_MODE:
        active = active or float(resolve_spatial_selection_config(selection)["boundary_weight"]) > 0.0
    return (
        boundary_distance_channels_from_loss_cfg(loss_cfg)
        if active
        else DEFAULT_BOUNDARY_DISTANCE_CHANNELS
    )


def required_raw_supervision_channels(
    *,
    loss_cfg: dict[str, Any] | None,
    selection_cfg: dict[str, Any] | None,
) -> list[str]:
    """Return raw channels that can affect training or checkpoint selection."""

    required: list[str] = []
    supervised = dict(dict(loss_cfg or {}).get("supervised", {}) or {})
    if str(supervised.get("mask", "none")).strip().lower() == "plasma_only":
        required.append("mask_plasma")
    spatial_loss = dict(supervised.get("spatial", {}) or {})
    boundary_channels = boundary_distance_channels_from_loss_cfg(loss_cfg)
    if float(spatial_loss.get("boundary_weight", 0.0)) > 0.0:
        required.extend(boundary_channels)

    selection = dict(selection_cfg or {})
    if str(selection.get("mode", "last")).strip().lower() == SPATIAL_SELECTION_MODE:
        required.append("mask_plasma")
        if float(resolve_spatial_selection_config(selection)["boundary_weight"]) > 0.0:
            required.extend(boundary_channels)
    return list(dict.fromkeys(required))


def slice_case_map(values: Any | None, indices: np.ndarray, *, key: str) -> np.ndarray | None:
    """Select case-aligned supervision without changing a static map."""

    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    local = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"{key} must be [H,W] or [N,H,W], got={arr.shape}")
    if int(arr.shape[0]) == 1:
        return np.broadcast_to(arr, (int(local.shape[0]), *arr.shape[1:])).astype(np.float32, copy=False)
    if local.size and int(np.max(local)) >= int(arr.shape[0]):
        raise IndexError(f"{key} cannot select case index from shape={arr.shape}")
    return arr[local]


def materialize_supervised_geometry(
    spatial_features: Any | None,
    indices: np.ndarray,
    *,
    mask: Any | None,
    distance_any: Any | None,
    distance_signed: Any | None = None,
    boundary_distance_channels: tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray | None]:
    """Use unscaled case geometry for loss/selection, never model-input rows."""

    boundary_channels = normalize_boundary_distance_channels(boundary_distance_channels)
    values: dict[str, Any | None] = {
        "mask_plasma": mask,
        "distance_any": distance_any,
        "distance_signed": distance_signed,
    }
    raw_callable = callable(getattr(spatial_features, "raw_batch", None))
    declared_channels = set(getattr(spatial_features, "static_channels", ()) or ()) | set(
        getattr(spatial_features, "case_channels", ()) or ()
    )
    if raw_callable and not declared_channels:
        raise ValueError(
            "raw spatial supervision sources must declare static_channels and/or "
            "case_channels; channel ownership cannot be inferred safely"
        )
    channels = [
        name
        for name, value in values.items()
        if value is not None or (raw_callable and name in declared_channels)
    ]
    if distance_any is not None or any(name in declared_channels for name in boundary_channels):
        channels.extend(boundary_channels)
    channels = list(dict.fromkeys(channels))
    if not channels:
        return {"mask": None, "distance_any": None, "distance_signed": None}
    if raw_callable:
        raw_channels = [name for name in channels if name in declared_channels]
    else:
        raw_channels = []
    fallback_channels = [name for name in channels if name not in raw_channels]
    by_name: dict[str, np.ndarray] = {}
    if raw_channels:
        raw = materialize_raw_spatial_batch(
            spatial_features,
            np.asarray(indices, dtype=np.int64),
            channels=raw_channels,
        )
        if raw is None:
            raise ValueError("raw spatial source did not materialize its declared supervision channels")
        by_name.update(
            {name: np.asarray(raw[..., idx], dtype=np.float32) for idx, name in enumerate(raw_channels)}
        )
    if fallback_channels:
        missing_fallback = [name for name in fallback_channels if values.get(name) is None]
        if missing_fallback:
            raise ValueError(
                "raw spatial supervision source is missing configured channels: "
                f"{missing_fallback}"
            )
        fallback = {name: values[name] for name in fallback_channels if values[name] is not None}
        fallback_raw = materialize_raw_spatial_batch(
            None,
            np.asarray(indices, dtype=np.int64),
            channels=fallback_channels,
            fallback=fallback,
        )
        if fallback_raw is None:
            raise ValueError("raw spatial supervision fallback could not be materialized")
        by_name.update(
            {
                name: np.asarray(fallback_raw[..., idx], dtype=np.float32)
                for idx, name in enumerate(fallback_channels)
            }
        )
    boundary_distance = None
    if any(name in by_name for name in boundary_channels):
        boundary_distance = compose_boundary_distance(
            by_name,
            channels=boundary_channels,
            key="supervised boundary distance",
        )
    return {
        "mask": by_name.get("mask_plasma"),
        "distance_any": boundary_distance,
        "distance_signed": by_name.get("distance_signed"),
    }


__all__ = [
    "materialize_supervised_geometry",
    "objective_boundary_distance_channels",
    "required_raw_supervision_channels",
    "slice_case_map",
]
