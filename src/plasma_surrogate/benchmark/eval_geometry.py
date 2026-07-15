"""Case-aligned raw geometry for benchmark evaluation.

Evaluation geometry is deliberately resolved independently from a model's
input-feature list.  A model may omit geometry channels while spatial metrics
still require the raw plasma mask and unsigned boundary distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.core.boundary_distance import (
    compose_boundary_distance,
    normalize_boundary_distance_channels,
)

_RAW_EVAL_CHANNELS: tuple[str, str] = ("mask_plasma", "distance_any")
_RAW_GEOMETRY_CHANNELS = frozenset((*_RAW_EVAL_CHANNELS, "distance_signed"))


@dataclass(frozen=True)
class EvaluationSpatialGeometry:
    """Raw, case-aligned ``[N,H,W]`` arrays used by evaluation."""

    mask_plasma: np.ndarray
    distance_any: np.ndarray
    distance_signed: np.ndarray
    metric_mask: np.ndarray | None
    source: str
    boundary_distance_channels: tuple[str, ...]


def _pack_channels(pack: dict[str, Any], *, pack_name: str) -> tuple[str, ...]:
    if "channels" not in pack:
        raise ValueError(f"{pack_name} is missing declared channels")
    raw = np.asarray(pack["channels"]).reshape(-1).tolist()
    channels = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw)
    if not channels:
        raise ValueError(f"{pack_name} declared channels must be non-empty")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{pack_name} declared channels must be unique: {channels}")
    return channels


def _dataset_case_ids(context: Any) -> tuple[str, ...]:
    dataset = getattr(context, "dataset", None)
    cases = list(getattr(dataset, "cases", []) or [])
    return tuple(str(case.get("case_id", index)) for index, case in enumerate(cases))


def _validate_indices(
    indices: np.ndarray,
    *,
    dataset_case_ids: tuple[str, ...],
    expected_eval_count: int | None,
) -> np.ndarray:
    out = np.asarray(indices, dtype=np.int64).reshape(-1)
    if expected_eval_count is not None and int(out.shape[0]) != int(expected_eval_count):
        raise ValueError(
            "evaluation geometry/test prediction count mismatch: "
            f"indices={int(out.shape[0])}, predictions={int(expected_eval_count)}"
        )
    if np.any(out < 0):
        raise IndexError("evaluation geometry indices must be non-negative")
    if dataset_case_ids and out.size and int(np.max(out)) >= len(dataset_case_ids):
        raise IndexError(
            "evaluation geometry index exceeds dataset case order: "
            f"max_index={int(np.max(out))}, n_cases={len(dataset_case_ids)}"
        )
    return out


def _case_geometry_from_pack(
    *,
    pack: dict[str, Any],
    indices: np.ndarray,
    dataset_case_ids: tuple[str, ...],
    spatial_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not pack:
        return None
    channels = _pack_channels(pack, pack_name="case_structure_feature_pack")
    declared_raw = set(channels) & _RAW_GEOMETRY_CHANNELS
    if not declared_raw:
        return None
    missing = sorted(set(_RAW_EVAL_CHANNELS) - set(channels))
    if missing:
        raise ValueError(
            "case_structure_feature_pack declares case-varying raw geometry but is missing "
            f"required raw channels: {missing}"
        )
    if "data" not in pack:
        raise ValueError("case_structure_feature_pack is missing data for declared raw channels")
    data = np.asarray(pack["data"], dtype=np.float32)
    if data.ndim != 4 or tuple(data.shape[2:]) != spatial_shape:
        raise ValueError(
            "case_structure_feature_pack data must be [N,C,H,W] for evaluation geometry: "
            f"expected spatial shape={spatial_shape}, got={data.shape}"
        )
    if int(data.shape[1]) != len(channels):
        raise ValueError(
            "case_structure_feature_pack declared channel/data count mismatch: "
            f"channels={len(channels)}, data_channels={int(data.shape[1])}"
        )
    if "case_ids" not in pack:
        raise ValueError(
            "case_structure_feature_pack must declare case_ids when it contains raw evaluation geometry"
        )
    raw_ids = np.asarray(pack["case_ids"]).reshape(-1).tolist()
    pack_case_ids = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_ids
    )
    if int(data.shape[0]) != len(pack_case_ids):
        raise ValueError(
            "case_structure_feature_pack case_ids/data row count mismatch: "
            f"case_ids={len(pack_case_ids)}, data_rows={int(data.shape[0])}"
        )
    if dataset_case_ids and pack_case_ids != dataset_case_ids:
        raise ValueError(
            "case_structure_feature_pack case_ids must exactly match dataset order for evaluation geometry"
        )
    if indices.size and int(np.max(indices)) >= int(data.shape[0]):
        raise IndexError(
            "evaluation geometry index exceeds case_structure_feature_pack rows: "
            f"max_index={int(np.max(indices))}, rows={int(data.shape[0])}"
        )
    mask_idx = channels.index("mask_plasma")
    distance_idx = channels.index("distance_any")
    return data[indices, mask_idx], data[indices, distance_idx]


def _static_geometry_from_pack(
    *,
    pack: dict[str, Any],
    pack_name: str,
    n_eval: int,
    spatial_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not pack:
        return None
    channels = _pack_channels(pack, pack_name=pack_name)
    if not set(_RAW_EVAL_CHANNELS).issubset(channels):
        return None
    if "data" not in pack:
        raise ValueError(f"{pack_name} is missing data for declared raw channels")
    data = np.asarray(pack["data"], dtype=np.float32)
    if data.ndim != 3 or tuple(data.shape[1:]) != spatial_shape:
        raise ValueError(
            f"{pack_name} data must be [C,H,W] for evaluation geometry: "
            f"expected spatial shape={spatial_shape}, got={data.shape}"
        )
    if int(data.shape[0]) != len(channels):
        raise ValueError(
            f"{pack_name} declared channel/data count mismatch: "
            f"channels={len(channels)}, data_channels={int(data.shape[0])}"
        )
    mask = np.broadcast_to(data[channels.index("mask_plasma")], (n_eval, *spatial_shape)).copy()
    distance = np.broadcast_to(data[channels.index("distance_any")], (n_eval, *spatial_shape)).copy()
    return mask, distance


def _static_geometry_from_context(
    *,
    geom_ctx: Any,
    n_eval: int,
    spatial_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if geom_ctx is None:
        raise ValueError(
            "evaluation requires raw mask_plasma and distance_any, but no spatial pack "
            "or geometry context is available"
        )
    if getattr(geom_ctx, "mask_plasma", None) is None or getattr(geom_ctx, "distance_any", None) is None:
        raise ValueError("geometry context is missing raw mask_plasma or distance_any for evaluation")
    mask_static = np.asarray(geom_ctx.mask_plasma, dtype=np.float32)
    distance_static = np.asarray(geom_ctx.distance_any, dtype=np.float32)
    if mask_static.shape != spatial_shape or distance_static.shape != spatial_shape:
        raise ValueError(
            "static evaluation geometry shape mismatch: "
            f"expected={spatial_shape}, mask={mask_static.shape}, distance_any={distance_static.shape}"
        )
    return (
        np.broadcast_to(mask_static, (n_eval, *spatial_shape)).copy(),
        np.broadcast_to(distance_static, (n_eval, *spatial_shape)).copy(),
    )


def _case_channel_from_pack(
    *,
    pack: dict[str, Any],
    channel: str,
    indices: np.ndarray,
    dataset_case_ids: tuple[str, ...],
    spatial_shape: tuple[int, int],
) -> np.ndarray | None:
    if not pack:
        return None
    channels = _pack_channels(pack, pack_name="case_structure_feature_pack")
    if channel not in channels:
        return None
    data = np.asarray(pack.get("data"), dtype=np.float32)
    if data.ndim != 4 or tuple(data.shape[2:]) != spatial_shape or int(data.shape[1]) != len(channels):
        raise ValueError(
            "case_structure_feature_pack data must be [N,C,H,W] and align with declared "
            f"channels for evaluation boundary distance; got={data.shape}"
        )
    if "case_ids" not in pack:
        raise ValueError(
            "case_structure_feature_pack must declare case_ids for case-varying boundary distance"
        )
    raw_ids = np.asarray(pack["case_ids"]).reshape(-1).tolist()
    pack_case_ids = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_ids
    )
    if len(pack_case_ids) != int(data.shape[0]):
        raise ValueError("case_structure_feature_pack case_ids/data row count mismatch")
    if dataset_case_ids and pack_case_ids != dataset_case_ids:
        raise ValueError(
            "case_structure_feature_pack case_ids must exactly match dataset order for evaluation geometry"
        )
    if indices.size and int(np.max(indices)) >= int(data.shape[0]):
        raise IndexError("evaluation boundary-distance index exceeds case_structure_feature_pack rows")
    return data[indices, channels.index(channel)]


def _static_channel_from_pack(
    *,
    pack: dict[str, Any],
    pack_name: str,
    channel: str,
    n_eval: int,
    spatial_shape: tuple[int, int],
) -> np.ndarray | None:
    if not pack:
        return None
    channels = _pack_channels(pack, pack_name=pack_name)
    if channel not in channels:
        return None
    data = np.asarray(pack.get("data"), dtype=np.float32)
    if data.ndim != 3 or tuple(data.shape[1:]) != spatial_shape or int(data.shape[0]) != len(channels):
        raise ValueError(
            f"{pack_name} data must be [C,H,W] and align with declared channels for "
            f"evaluation boundary distance; got={data.shape}"
        )
    return np.broadcast_to(data[channels.index(channel)], (n_eval, *spatial_shape)).copy()


def materialize_evaluation_spatial_geometry(
    *,
    context: Any,
    geom_ctx: Any,
    eval_indices: np.ndarray,
    metric_mask_enabled: bool,
    expected_eval_count: int | None = None,
    boundary_distance_channels: tuple[str, ...] | list[str] | None = None,
) -> EvaluationSpatialGeometry:
    """Resolve raw evaluation geometry in exact test-row order.

    Case-specific raw channels take precedence.  The mask uses the established
    raw geometry fallback, while the canonical boundary distance is composed as
    the minimum absolute value across all configured SDF/distance channels.
    """

    dataset_case_ids = _dataset_case_ids(context)
    indices = _validate_indices(
        eval_indices,
        dataset_case_ids=dataset_case_ids,
        expected_eval_count=expected_eval_count,
    )
    h = int(getattr(context, "h", 0))
    w = int(getattr(context, "w", 0))
    if h <= 0 or w <= 0:
        raise ValueError(f"evaluation spatial shape must be positive, got {(h, w)}")
    spatial_shape = (h, w)

    case_pack = dict(getattr(context, "case_structure_feature_pack", {}) or {})
    resolved = _case_geometry_from_pack(
        pack=case_pack,
        indices=indices,
        dataset_case_ids=dataset_case_ids,
        spatial_shape=spatial_shape,
    )
    source = "case_structure_feature_pack"
    if resolved is None:
        source = "static_spatial_feature_pack"
        resolved = _static_geometry_from_pack(
            pack=dict(getattr(context, "static_spatial_feature_pack", {}) or {}),
            pack_name=source,
            n_eval=int(indices.shape[0]),
            spatial_shape=spatial_shape,
        )
    if resolved is None:
        source = "coord_feature_pack"
        resolved = _static_geometry_from_pack(
            pack=dict(getattr(context, "coord_feature_pack", {}) or {}),
            pack_name=source,
            n_eval=int(indices.shape[0]),
            spatial_shape=spatial_shape,
        )
    if resolved is None:
        source = "geometry_context"
        resolved = _static_geometry_from_context(
            geom_ctx=geom_ctx,
            n_eval=int(indices.shape[0]),
            spatial_shape=spatial_shape,
        )

    mask, base_distance_any = (np.asarray(value, dtype=np.float32) for value in resolved)
    boundary_channels = normalize_boundary_distance_channels(
        boundary_distance_channels,
        key="evaluation boundary_distance_channels",
    )
    boundary_by_name: dict[str, np.ndarray] = {"distance_any": base_distance_any}
    source_parts = [source]
    for channel in boundary_channels:
        if channel in boundary_by_name:
            continue
        value = _case_channel_from_pack(
            pack=case_pack,
            channel=channel,
            indices=indices,
            dataset_case_ids=dataset_case_ids,
            spatial_shape=spatial_shape,
        )
        channel_source = "case_structure_feature_pack"
        if value is None:
            channel_source = "static_spatial_feature_pack"
            value = _static_channel_from_pack(
                pack=dict(getattr(context, "static_spatial_feature_pack", {}) or {}),
                pack_name=channel_source,
                channel=channel,
                n_eval=int(indices.shape[0]),
                spatial_shape=spatial_shape,
            )
        if value is None:
            channel_source = "coord_feature_pack"
            value = _static_channel_from_pack(
                pack=dict(getattr(context, "coord_feature_pack", {}) or {}),
                pack_name=channel_source,
                channel=channel,
                n_eval=int(indices.shape[0]),
                spatial_shape=spatial_shape,
            )
        if value is None and geom_ctx is not None and getattr(geom_ctx, channel, None) is not None:
            channel_source = "geometry_context"
            static = np.asarray(getattr(geom_ctx, channel), dtype=np.float32)
            if static.shape != spatial_shape:
                raise ValueError(
                    f"geometry context channel {channel!r} shape mismatch: "
                    f"expected={spatial_shape}, got={static.shape}"
                )
            value = np.broadcast_to(static, (int(indices.shape[0]), *spatial_shape)).copy()
        if value is None:
            raise ValueError(
                "evaluation boundary distance is missing configured raw channel: "
                f"{channel!r}"
            )
        boundary_by_name[channel] = np.asarray(value, dtype=np.float32)
        source_parts.append(channel_source)
    distance_any = compose_boundary_distance(
        boundary_by_name,
        channels=boundary_channels,
        key="evaluation boundary distance",
    )
    source = "+".join(dict.fromkeys(source_parts))
    expected_shape = (int(indices.shape[0]), h, w)
    if mask.shape != expected_shape or distance_any.shape != expected_shape:
        raise ValueError(
            "resolved evaluation geometry must align with test rows as [N,H,W]: "
            f"expected={expected_shape}, mask={mask.shape}, distance_any={distance_any.shape}"
        )
    if not np.all(np.isfinite(mask)) or not np.all(np.isfinite(distance_any)):
        raise ValueError("resolved evaluation mask_plasma/distance_any contains non-finite values")
    if np.any(distance_any < 0.0):
        raise ValueError("resolved evaluation distance_any must be non-negative")
    if any(not np.any(mask[case_idx] > 0.5) for case_idx in range(int(mask.shape[0]))):
        raise ValueError("resolved evaluation mask_plasma contains an empty case")

    distance_signed = np.where(mask > 0.5, distance_any, -distance_any).astype(np.float32)
    metric_mask = mask.copy() if bool(metric_mask_enabled) else None
    return EvaluationSpatialGeometry(
        mask_plasma=mask.copy(),
        distance_any=distance_any.copy(),
        distance_signed=distance_signed,
        metric_mask=metric_mask,
        source=source,
        boundary_distance_channels=boundary_channels,
    )
