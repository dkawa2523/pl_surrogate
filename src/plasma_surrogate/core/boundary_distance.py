"""Canonical boundary-distance semantics shared by train and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_BOUNDARY_DISTANCE_CHANNELS: tuple[str, ...] = ("distance_any",)


def normalize_boundary_distance_channels(
    channels: Sequence[str] | None,
    *,
    key: str = "boundary_distance_channels",
) -> tuple[str, ...]:
    """Validate the ordered set of raw SDF/distance channels defining a boundary."""

    if channels is None:
        return DEFAULT_BOUNDARY_DISTANCE_CHANNELS
    if isinstance(channels, (str, bytes)):
        raise ValueError(f"{key} must be a non-empty list of channel names")
    out = tuple(str(value).strip() for value in channels)
    if not out or any(not value for value in out):
        raise ValueError(f"{key} must be a non-empty list of channel names")
    if len(set(out)) != len(out):
        raise ValueError(f"{key} must contain unique channel names; got={list(out)}")
    return out


def boundary_distance_channels_from_loss_cfg(
    loss_cfg: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Read the canonical boundary definition from a resolved loss config."""

    supervised = dict(dict(loss_cfg or {}).get("supervised", {}) or {})
    spatial = dict(supervised.get("spatial", {}) or {})
    return normalize_boundary_distance_channels(
        spatial.get("boundary_distance_channels"),
        key="train.loss.supervised.spatial.boundary_distance_channels",
    )


def compose_boundary_distance(
    channels_by_name: Mapping[str, Any],
    *,
    channels: Sequence[str] | None = None,
    key: str = "boundary_distance",
) -> np.ndarray:
    """Return distance to the nearest configured boundary as ``min(abs(SDF_i))``.

    Signed and unsigned distance fields can be mixed.  Every configured channel
    is mandatory and must be finite and exactly aligned, so a stale or partial
    geometry contract cannot silently change the loss/evaluation boundary.
    """

    names = normalize_boundary_distance_channels(channels)
    missing = [name for name in names if name not in channels_by_name]
    if missing:
        raise ValueError(f"{key} is missing configured raw channels: {missing}")
    arrays = [np.asarray(channels_by_name[name], dtype=np.float32) for name in names]
    expected = arrays[0].shape
    for name, array in zip(names, arrays, strict=True):
        if array.shape != expected:
            raise ValueError(
                f"{key} channel shape mismatch: expected={expected}, "
                f"{name}={array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key} channel {name!r} contains non-finite values")
    return np.minimum.reduce([np.abs(array) for array in arrays]).astype(np.float32, copy=False)


__all__ = [
    "DEFAULT_BOUNDARY_DISTANCE_CHANNELS",
    "boundary_distance_channels_from_loss_cfg",
    "compose_boundary_distance",
    "normalize_boundary_distance_channels",
]
