"""Preprocessing runner for split/scaler/sampling artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DESCRIPTOR_PROFILE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    INPUT_MODE_EFFECTIVE_KEY,
    STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY,
    LATENT_PROFILE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
    attach_runtime_schema_hashes,
    build_input_mode_effective_metadata,
    build_runtime_schema_hashes,
    normalize_input_mode_cfg,
    validate_input_mode_cfg,
)
from plasma_surrogate.core.target_roles import build_target_role_schema
from plasma_surrogate.data.geometry_context import build_signed_distance_fields
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.features.structure_feature_registry import (
    resolve_spatial_channels_for_feature_profile,
    validate_coord_feature_channels,
)
from plasma_surrogate.features.structure_descriptors import build_structure_descriptor
from plasma_surrogate.features.geometry_feature_store import hash_json
from plasma_surrogate.preprocessing.sampling import (
    build_deeponet_indices,
    build_flattened_coords,
    build_patch_index,
    build_phase_wrap_pairs,
    build_point_pools,
    build_time_adjacent_pairs,
)
from plasma_surrogate.preprocessing.report import PreprocessReportBuilder
from plasma_surrogate.preprocessing.scalers import ScalerFactory, fit_scalers_train_only
from plasma_surrogate.preprocessing.schema import AxisSchema, ChannelMap, CondSchema
from plasma_surrogate.preprocessing.split_plan import SplitPlanBuilder
from plasma_surrogate.preprocessing.spatial_features import (
    ICP_PART_SDF_CHANNELS,
    ICP_PART_SDF_LITE_CASE_CHANNELS,
    ICP_STRUCT_CASE_CHANNELS,
    ICP_STRUCT_STATIC_CHANNELS,
    PART_SDF_SUMMARY_CHANNELS,
    PART_SOURCE_CHANNELS,
    PART_SOURCE_MEAN_CHANNELS,
    apply_distance_transform,
    derive_geom_feature_maps,
    part_sdf_summary_maps_from_stack,
    part_source_maps_from_stack,
    resolve_distance_transform_cfg,
    resolve_distance_transform_effective,
)


def _resolve_y_vars(cases: list[dict[str, Any]]) -> list[str]:
    keys = list(cases[0]["y"].keys())
    if len(keys) == 0:
        raise ValueError("Each case.y must include at least one target variable")
    return [str(k) for k in keys]


def _build_grid2d_field_layout(output_layout: dict[str, Any]) -> dict[str, Any]:
    shape = [int(v) for v in list(output_layout.get("shape", []))]
    vars_eff = [str(v) for v in list(output_layout.get("vars", []))]
    if len(shape) != 3:
        raise ValueError(f"grid2d field_layout requires output_layout.shape=[C,H,W], got={shape}")
    if int(shape[0]) != len(vars_eff):
        raise ValueError(
            "grid2d field_layout requires output_layout.shape[0] to match output_layout.vars length; "
            f"shape={shape}, vars={vars_eff}"
        )
    return {
        "version": 1,
        "layout_type": "grid2d",
        "vars": vars_eff,
        "shape": shape,
        "order": str(output_layout.get("order", "C")),
        "axes": ["channel", "y", "x"],
    }


def _coord_xy_maps(coord_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(coord_grid, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError(f"coord_grid must be rank-3, got shape={grid.shape}")
    if grid.shape[0] == 2:
        return grid[0], grid[1]
    if grid.shape[-1] == 2:
        return grid[..., 0], grid[..., 1]
    raise ValueError(f"coord_grid must be (2,H,W) or (H,W,2), got {grid.shape}")


PART_LITE_STATIC_CHANNELS: tuple[str, ...] = (
    *ICP_STRUCT_STATIC_CHANNELS,
    "normal_x",
    "normal_y",
    "curvature_proxy",
    "boundary_band",
)
PART_LITE_CASE_CHANNELS: tuple[str, ...] = PART_SDF_SUMMARY_CHANNELS
SMOOTH_STRUCTURE_STATIC_CHANNELS: tuple[str, ...] = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "boundary_band",
)
SMOOTH_STRUCTURE_CASE_CHANNELS: tuple[str, ...] = ("solid_proximity",)
PART_SOURCE_STATIC_CHANNELS: tuple[str, ...] = ("x", "y", "distance_signed")
PART_SOURCE_CASE_CHANNELS: tuple[str, ...] = PART_SOURCE_CHANNELS
ICP_COIL_SOURCE_STATIC_CHANNELS: tuple[str, ...] = ICP_STRUCT_STATIC_CHANNELS
ICP_COIL_SOURCE_CASE_CHANNELS: tuple[str, ...] = ("mask_coil",)
ICP_COIL_UNION_SDF_STATIC_CHANNELS: tuple[str, ...] = ICP_STRUCT_STATIC_CHANNELS
ICP_COIL_UNION_SDF_CASE_CHANNELS: tuple[str, ...] = ("part_sdf_union",)
ICP_COIL_SDF_SOURCE_STATIC_CHANNELS: tuple[str, ...] = ICP_STRUCT_STATIC_CHANNELS
ICP_COIL_SDF_SOURCE_CASE_CHANNELS: tuple[str, ...] = ("part_sdf_union", "part_source_sum")
ICP_COIL_SDF_SOURCE_MEAN_CASE_CHANNELS: tuple[str, ...] = PART_SOURCE_MEAN_CHANNELS
ICP_VACUUM_FIELD_STATIC_CHANNELS: tuple[str, ...] = ICP_STRUCT_STATIC_CHANNELS
ICP_VACUUM_FIELD_CASE_CHANNELS: tuple[str, ...] = (
    "vacuum_aphi_unit",
    "vacuum_br_unit",
    "vacuum_bz_unit",
    "vacuum_bmag_unit",
)
ICP_SDF_MEAN_VACUUM_CASE_CHANNELS: tuple[str, ...] = (
    *PART_SOURCE_MEAN_CHANNELS,
    *ICP_VACUUM_FIELD_CASE_CHANNELS,
)
ICP_SDF_MEAN_VACUUM_STRUCTURE_CASE_CHANNELS: tuple[str, ...] = (
    *ICP_SDF_MEAN_VACUUM_CASE_CHANNELS,
    "part_second_proximity",
    "part_competition",
    "solid_proximity",
)


def _is_mask_like_channel(name: str) -> bool:
    return str(name).startswith("mask_") or str(name) in {
        "valid_field_mask",
        "outside_mask",
        "coil_proximity",
        # Mean-reduced equal-strength source is bounded in [0, 1].  Keeping
        # that physical range avoids the very large z-scores produced when a
        # sparse coil field is fitted only on plasma/boundary pixels.
        "part_source_mean",
        "part_second_proximity",
        "part_competition",
        "solid_proximity",
    }


def _load_case_structure_npz(
    case: dict[str, Any],
    *,
    shape: tuple[int, int],
    require_mask_coil: bool = True,
) -> dict[str, np.ndarray]:
    raw_path = case.get("structure_npz")
    if raw_path is None:
        raise ValueError(
            "case-specific structure feature profiles require "
            "dataset rows with structure_npz_column"
        )
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"structure npz not found for case={case.get('case_id')}: {path}")
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        required_keys = ["mask_plasma", "valid_field_mask", "outside_mask"]
        if require_mask_coil:
            required_keys.append("mask_coil")
        for key in required_keys:
            if key not in data.files:
                raise ValueError(f"structure npz missing key={key!r} for case={case.get('case_id')}: {path}")
        for key in ("mask_plasma", "mask_coil", "valid_field_mask", "outside_mask"):
            if key not in data.files:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(
                    f"structure npz shape mismatch for case={case.get('case_id')} key={key}: "
                    f"expected={shape}, got={arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"structure npz contains non-finite values for case={case.get('case_id')} key={key}")
            out[key] = (arr > 0.5).astype(np.float32)
        if "part_mask_stack" in data.files:
            stack = np.asarray(data["part_mask_stack"], dtype=np.float32)
            if stack.ndim != 3 or tuple(stack.shape[1:]) != shape:
                raise ValueError(
                    f"structure npz shape mismatch for case={case.get('case_id')} key=part_mask_stack: "
                    f"expected=[P,{shape[0]},{shape[1]}], got={stack.shape}"
                )
            if not np.all(np.isfinite(stack)):
                raise ValueError(f"structure npz contains non-finite values for case={case.get('case_id')} key=part_mask_stack")
            out["part_mask_stack"] = (stack > 0.5).astype(np.float32)
        for key in ICP_PART_SDF_CHANNELS:
            if key not in data.files:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(
                    f"structure npz shape mismatch for case={case.get('case_id')} key={key}: "
                    f"expected={shape}, got={arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"structure npz contains non-finite values for case={case.get('case_id')} key={key}")
            out[key] = arr.astype(np.float32)
        for key in ICP_VACUUM_FIELD_CASE_CHANNELS:
            if key not in data.files:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(
                    f"structure npz shape mismatch for case={case.get('case_id')} key={key}: "
                    f"expected={shape}, got={arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"structure npz contains non-finite values for case={case.get('case_id')} key={key}")
            out[key] = arr
    return out


def _materialize_scaler_plasma_mask(
    cases: list[dict[str, Any]],
    *,
    static_mask: np.ndarray,
) -> np.ndarray:
    """Resolve raw masks in dataset row order for target-scaler fitting.

    Case structure inputs are the source used to build the compact spatial
    packs later in preprocessing.  Reading their raw masks here avoids fitting
    target statistics with the provider's reference geometry when each case has
    a different plasma domain.  Datasets without case structures retain the
    static-mask contract.
    """

    fallback = np.asarray(static_mask, dtype=np.float32)
    if fallback.ndim != 2:
        raise ValueError(f"static plasma mask must be [H,W], got {fallback.shape}")
    has_case_structure = [case.get("structure_npz") is not None for case in cases]
    if not any(has_case_structure):
        return fallback
    if not all(has_case_structure):
        missing = [
            str(case.get("case_id", index))
            for index, (case, present) in enumerate(zip(cases, has_case_structure, strict=True))
            if not present
        ]
        raise ValueError(
            "case-aligned plasma-mask scaler fitting requires structure_npz for every case; "
            f"missing_cases={missing[:8]}"
        )
    shape = (int(fallback.shape[0]), int(fallback.shape[1]))
    masks = [
        np.asarray(
            _load_case_structure_npz(case, shape=shape, require_mask_coil=False)["mask_plasma"],
            dtype=np.float32,
        )
        for case in cases
    ]
    return np.stack(masks, axis=0).astype(np.float32, copy=False)


def _scaler_mask_active_ratio(mask_plasma: np.ndarray, train_indices: np.ndarray) -> float:
    mask = np.asarray(mask_plasma, dtype=np.float32)
    if mask.ndim == 2:
        selected = mask
    elif mask.ndim == 3 and int(mask.shape[0]) == 1:
        selected = mask[0]
    elif mask.ndim == 3:
        selected = mask[np.asarray(train_indices, dtype=np.int64)]
    else:
        raise ValueError(f"scaler plasma mask must be [H,W], [1,H,W], or [N,H,W], got {mask.shape}")
    return float(np.mean(selected > 0.5))


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    if float(np.sum(binary, dtype=np.float32)) <= 0.0:
        return np.full(binary.shape, float(max(binary.shape)), dtype=np.float32)
    signed = build_signed_distance_fields(binary)
    return np.where(binary > 0.5, 0.0, np.abs(signed)).astype(np.float32)


def _build_split_spatial_feature_packs(
    *,
    cases: list[dict[str, Any]],
    channels: list[str],
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    train_indices: np.ndarray,
    coil_proximity_percentile: float,
    coil_proximity_tau_fixed: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape = tuple(int(v) for v in np.asarray(coord_x, dtype=np.float32).shape)
    coord_xy = np.stack(
        [
            np.asarray(coord_x, dtype=np.float32).reshape(-1),
            np.asarray(coord_y, dtype=np.float32).reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)
    struct_channels = [*ICP_STRUCT_STATIC_CHANNELS, *ICP_STRUCT_CASE_CHANNELS]
    part_sdf_channels = [*ICP_STRUCT_STATIC_CHANNELS, *ICP_PART_SDF_LITE_CASE_CHANNELS]
    part_lite_channels = [*PART_LITE_STATIC_CHANNELS, *PART_LITE_CASE_CHANNELS]
    smooth_structure_channels = [*SMOOTH_STRUCTURE_STATIC_CHANNELS, *SMOOTH_STRUCTURE_CASE_CHANNELS]
    part_source_channels = [*PART_SOURCE_STATIC_CHANNELS, *PART_SOURCE_CASE_CHANNELS]
    coil_source_channels = [*ICP_COIL_SOURCE_STATIC_CHANNELS, *ICP_COIL_SOURCE_CASE_CHANNELS]
    coil_union_sdf_channels = [*ICP_COIL_UNION_SDF_STATIC_CHANNELS, *ICP_COIL_UNION_SDF_CASE_CHANNELS]
    coil_sdf_source_channels = [*ICP_COIL_SDF_SOURCE_STATIC_CHANNELS, *ICP_COIL_SDF_SOURCE_CASE_CHANNELS]
    coil_sdf_source_mean_channels = [
        *ICP_COIL_SDF_SOURCE_STATIC_CHANNELS,
        *ICP_COIL_SDF_SOURCE_MEAN_CASE_CHANNELS,
    ]
    vacuum_field_channels = [*ICP_VACUUM_FIELD_STATIC_CHANNELS, *ICP_VACUUM_FIELD_CASE_CHANNELS]
    sdf_mean_vacuum_channels = [
        *ICP_COIL_SDF_SOURCE_STATIC_CHANNELS,
        *ICP_SDF_MEAN_VACUUM_CASE_CHANNELS,
    ]
    sdf_mean_vacuum_structure_channels = [
        *ICP_COIL_SDF_SOURCE_STATIC_CHANNELS,
        *ICP_SDF_MEAN_VACUUM_STRUCTURE_CASE_CHANNELS,
    ]
    if list(channels) == struct_channels:
        static_channels = ICP_STRUCT_STATIC_CHANNELS
        case_channels = ICP_STRUCT_CASE_CHANNELS
        feature_profile = "icp_struct_spatial_v1"
    elif list(channels) == part_sdf_channels:
        static_channels = ICP_STRUCT_STATIC_CHANNELS
        case_channels = ICP_PART_SDF_LITE_CASE_CHANNELS
        feature_profile = "icp_part_sdf_lite_v1"
    elif list(channels) == part_lite_channels:
        static_channels = PART_LITE_STATIC_CHANNELS
        case_channels = PART_LITE_CASE_CHANNELS
        feature_profile = "part_lite_v1"
    elif list(channels) == smooth_structure_channels:
        static_channels = SMOOTH_STRUCTURE_STATIC_CHANNELS
        case_channels = SMOOTH_STRUCTURE_CASE_CHANNELS
        feature_profile = "smooth_structure_v1"
    elif list(channels) == part_source_channels:
        static_channels = PART_SOURCE_STATIC_CHANNELS
        case_channels = PART_SOURCE_CASE_CHANNELS
        feature_profile = "part_source_v1"
    elif list(channels) == coil_source_channels:
        static_channels = ICP_COIL_SOURCE_STATIC_CHANNELS
        case_channels = ICP_COIL_SOURCE_CASE_CHANNELS
        feature_profile = "icp_coil_source_v1"
    elif list(channels) == coil_union_sdf_channels:
        static_channels = ICP_COIL_UNION_SDF_STATIC_CHANNELS
        case_channels = ICP_COIL_UNION_SDF_CASE_CHANNELS
        feature_profile = "icp_coil_union_sdf_v1"
    elif list(channels) == coil_sdf_source_channels:
        static_channels = ICP_COIL_SDF_SOURCE_STATIC_CHANNELS
        case_channels = ICP_COIL_SDF_SOURCE_CASE_CHANNELS
        feature_profile = "icp_coil_sdf_source_v2"
    elif list(channels) == coil_sdf_source_mean_channels:
        static_channels = ICP_COIL_SDF_SOURCE_STATIC_CHANNELS
        case_channels = ICP_COIL_SDF_SOURCE_MEAN_CASE_CHANNELS
        feature_profile = "icp_coil_sdf_source_mean_v3"
    elif list(channels) == vacuum_field_channels:
        static_channels = ICP_VACUUM_FIELD_STATIC_CHANNELS
        case_channels = ICP_VACUUM_FIELD_CASE_CHANNELS
        feature_profile = "icp_vacuum_field_v1"
    elif list(channels) == sdf_mean_vacuum_channels:
        static_channels = ICP_COIL_SDF_SOURCE_STATIC_CHANNELS
        case_channels = ICP_SDF_MEAN_VACUUM_CASE_CHANNELS
        feature_profile = "icp_coil_sdf_source_mean_vacuum_v1"
    elif list(channels) == sdf_mean_vacuum_structure_channels:
        static_channels = ICP_COIL_SDF_SOURCE_STATIC_CHANNELS
        case_channels = ICP_SDF_MEAN_VACUUM_STRUCTURE_CASE_CHANNELS
        feature_profile = "icp_coil_sdf_source_mean_vacuum_structure_v1"
    else:
        raise ValueError(
            "compact structure packs expect channels="
            f"{struct_channels}, {part_sdf_channels}, {part_lite_channels}, "
            f"{smooth_structure_channels}, {part_source_channels}, {coil_source_channels}, "
            f"{coil_union_sdf_channels}, {coil_sdf_source_channels}, "
            f"{coil_sdf_source_mean_channels}, {vacuum_field_channels}, "
            f"{sdf_mean_vacuum_channels}, or {sdf_mean_vacuum_structure_channels}; "
            f"got={list(channels)}"
        )

    static_map: dict[str, np.ndarray] | None = None
    case_maps: list[dict[str, np.ndarray]] = []
    train_distance_coil_values: list[np.ndarray] = []
    train_set = set(int(v) for v in np.asarray(train_indices, dtype=np.int64).reshape(-1).tolist())
    for idx, case in enumerate(cases):
        uses_coil_channels = any(name in set(case_channels) for name in ICP_STRUCT_CASE_CHANNELS)
        structure = _load_case_structure_npz(case, shape=shape, require_mask_coil=uses_coil_channels)
        mask_plasma = structure["mask_plasma"]
        distance_signed = build_signed_distance_fields(mask_plasma).astype(np.float32)
        distance_any = np.abs(distance_signed).astype(np.float32)

        current_static = {
            name: np.asarray(arr, dtype=np.float32).reshape(shape)
            for name, arr in derive_geom_feature_maps(
                coord_xy=coord_xy,
                distance_signed=distance_signed.reshape(-1),
                distance_any=distance_any.reshape(-1),
                mask_plasma=mask_plasma.reshape(-1),
                h=shape[0],
                w=shape[1],
            ).items()
        }
        if static_map is None:
            static_map = {name: np.asarray(current_static[name], dtype=np.float32) for name in static_channels}
        else:
            changed = [
                name
                for name in static_channels
                if not np.array_equal(np.asarray(current_static[name], dtype=np.float32), static_map[name])
            ]
            if changed:
                raise ValueError(
                    f"{feature_profile} compact packs require fixed static geometry features; "
                    f"case={case.get('case_id')} changed={changed}"
                )

        fmap: dict[str, np.ndarray] = {}
        if uses_coil_channels:
            distance_coil = _distance_to_mask(structure["mask_coil"]).astype(np.float32)
            fmap.update(
                {
                    "mask_coil": structure["mask_coil"],
                    "distance_coil": distance_coil,
                }
            )
        for name in ICP_PART_SDF_CHANNELS:
            if name in case_channels:
                if name not in structure:
                    raise ValueError(
                        f"{feature_profile} requires structure npz key={name!r} "
                        f"for case={case.get('case_id')}"
                    )
                fmap[name] = structure[name]
        for name in ICP_VACUUM_FIELD_CASE_CHANNELS:
            if name in case_channels:
                if name not in structure:
                    raise ValueError(
                        f"{feature_profile} requires structure npz key={name!r} "
                        f"for case={case.get('case_id')}"
                    )
                fmap[name] = structure[name]
        if feature_profile in {"part_lite_v1", "smooth_structure_v1"}:
            if "part_mask_stack" not in structure:
                raise ValueError(
                    f"{feature_profile} requires structure npz key='part_mask_stack' "
                    f"for case={case.get('case_id')}"
                )
            summaries = part_sdf_summary_maps_from_stack(structure["part_mask_stack"])
            fmap.update({name: summaries[name] for name in case_channels})
        if feature_profile in {
            "part_source_v1",
            "icp_coil_union_sdf_v1",
            "icp_coil_sdf_source_v2",
            "icp_coil_sdf_source_mean_v3",
            "icp_coil_sdf_source_mean_vacuum_v1",
            "icp_coil_sdf_source_mean_vacuum_structure_v1",
        }:
            if "part_mask_stack" not in structure:
                raise ValueError(
                    f"{feature_profile} requires structure npz key='part_mask_stack' "
                    f"for case={case.get('case_id')}"
                )
            source_maps = part_source_maps_from_stack(structure["part_mask_stack"])
            fmap.update({name: source_maps[name] for name in case_channels if name in source_maps})
        if feature_profile == "icp_coil_sdf_source_mean_vacuum_structure_v1":
            summaries = part_sdf_summary_maps_from_stack(structure["part_mask_stack"])
            fmap.update({name: summaries[name] for name in case_channels if name in summaries})
        case_maps.append(fmap)
        if uses_coil_channels and idx in train_set:
            sel = mask_plasma > 0.5
            vals = distance_coil[sel] if np.any(sel) else distance_coil.reshape(-1)
            train_distance_coil_values.append(np.asarray(vals, dtype=np.float32).reshape(-1))
    coil_tau: float | None = None
    if "coil_proximity" in set(case_channels):
        if coil_proximity_tau_fixed is not None:
            coil_tau = float(coil_proximity_tau_fixed)
            if not np.isfinite(coil_tau) or coil_tau <= 0.0:
                raise ValueError("coil_proximity_tau_fixed must be finite and > 0")
        else:
            if train_distance_coil_values:
                coil_values = np.concatenate(train_distance_coil_values, axis=0)
            else:
                coil_values = np.concatenate([m["distance_coil"].reshape(-1) for m in case_maps], axis=0)
            coil_values = coil_values[np.isfinite(coil_values)]
            if coil_values.size == 0:
                coil_values = np.asarray([float(max(shape))], dtype=np.float32)
            coil_tau = float(max(np.percentile(coil_values, float(coil_proximity_percentile)), 1.0e-3))
        for fmap in case_maps:
            fmap["coil_proximity"] = np.exp(-np.maximum(fmap["distance_coil"], 0.0) / coil_tau).astype(np.float32)
    if static_map is None:
        raise ValueError("cannot build compact spatial pack from empty cases")
    static_data = np.stack(
        [np.asarray(static_map[name], dtype=np.float32) for name in static_channels],
        axis=0,
    ).astype(np.float32)
    case_data = np.stack(
        [np.stack([np.asarray(fmap[name], dtype=np.float32) for name in case_channels], axis=0) for fmap in case_maps],
        axis=0,
    ).astype(np.float32)
    if not np.all(np.isfinite(static_data)) or not np.all(np.isfinite(case_data)):
        raise ValueError("compact case spatial feature packs contain non-finite values")
    channel_stats: dict[str, dict[str, Any]] = {}
    for channel_idx, name in enumerate(static_channels):
        values = np.asarray(static_data[channel_idx], dtype=np.float32)
        channel_stats[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "case_varying": False,
        }
    for channel_idx, name in enumerate(case_channels):
        values = np.asarray(case_data[:, channel_idx], dtype=np.float32)
        channel_stats[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "case_varying": bool(
                int(values.shape[0]) > 1
                and any(not np.array_equal(values[0], values[idx]) for idx in range(1, int(values.shape[0])))
            ),
        }
    for name in {"boundary_band", "solid_proximity", "coil_proximity"} & set(channel_stats):
        stats = channel_stats[name]
        if float(stats["min"]) < -1.0e-6 or float(stats["max"]) > 1.0 + 1.0e-6:
            raise ValueError(f"bounded spatial feature {name!r} must lie in [0,1]; got={stats}")
    case_feature_varies = bool(
        case_data.shape[0] > 1
        and any(not np.array_equal(case_data[0], case_data[idx]) for idx in range(1, case_data.shape[0]))
    )
    meta = {
        "case_ids": [str(c["case_id"]) for c in cases],
        "feature_profile": feature_profile,
        "logical_channels": list(channels),
        "static_channels": list(static_channels),
        "case_channels": list(case_channels),
        "static_shape": [int(v) for v in static_data.shape],
        "case_shape": [int(v) for v in case_data.shape],
        "logical_shape": [int(len(cases)), int(len(channels)), int(shape[0]), int(shape[1])],
        "storage": "split_static_case",
        "case_feature_varies": case_feature_varies,
        "channel_stats": channel_stats,
    }
    if coil_tau is not None:
        meta["coil_proximity_tau"] = coil_tau
        meta["coil_proximity_tau_source"] = (
            "fixed_nonlearned" if coil_proximity_tau_fixed is not None else "train_quantile"
        )
    return static_data, case_data, meta


@dataclass
class PreprocessOutput:
    split: dict[str, list[str]]
    cond_stats: dict[str, Any]
    hashes: dict[str, str]


@dataclass(frozen=True)
class PreprocessArtifacts:
    output_layout: dict[str, Any]
    target_role_schema: dict[str, Any]
    channel_map: dict[str, Any]
    coord_feature_pack_meta: dict[str, Any]
    static_spatial_feature_pack_meta: dict[str, Any]
    case_structure_feature_pack_meta: dict[str, Any]
    structure_descriptor_pack_meta: dict[str, Any]
    latent_feature_pack_meta: dict[str, Any]

    def runtime_schema_payload(self) -> dict[str, Any]:
        return {
            "output_layout": self.output_layout,
            "target_role_schema": self.target_role_schema,
            "channel_map": self.channel_map,
            "coord_feature_pack_meta": self.coord_feature_pack_meta,
            "static_spatial_feature_pack_meta": self.static_spatial_feature_pack_meta,
            "case_structure_feature_pack_meta": self.case_structure_feature_pack_meta,
            "structure_descriptor_pack_meta": self.structure_descriptor_pack_meta,
            "latent_feature_pack_meta": self.latent_feature_pack_meta,
        }


class PreprocessRunner:
    """Preprocessing runner that writes the shared mainline artifacts."""

    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str | Path,
        *,
        runtime_input_mode_meta: dict[str, Any] | None = None,
        runtime_cfg: dict[str, Any] | None = None,
        coord_distance_transform_cfg: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.store = ArtifactStore(self.output_dir)
        self.report_builder = PreprocessReportBuilder(self.store)
        self.runtime_cfg = dict(runtime_cfg or {})
        self.coord_distance_transform_cfg = resolve_distance_transform_cfg(coord_distance_transform_cfg)
        self.runtime_input_mode_meta = dict(runtime_input_mode_meta or {})
        if self.runtime_cfg:
            normalized = normalize_input_mode_cfg({"runtime": self.runtime_cfg})
            validate_input_mode_cfg(normalized)
            self.runtime_cfg = dict(normalized.get("runtime", {}))
            runtime_meta = build_input_mode_effective_metadata(normalized)
            existing_mode = str(self.runtime_input_mode_meta.get(INPUT_MODE_EFFECTIVE_KEY, "")).strip().lower()
            runtime_mode = str(runtime_meta.get(INPUT_MODE_EFFECTIVE_KEY, "")).strip().lower()
            if existing_mode and runtime_mode and existing_mode != runtime_mode:
                raise ValueError(
                    "runtime_input_mode_meta.input_mode_effective conflicts with runtime_cfg.input_mode: "
                    f"meta={existing_mode!r}, runtime={runtime_mode!r}"
                )
            self.runtime_input_mode_meta = {**runtime_meta, **self.runtime_input_mode_meta}
        self.input_mode_effective = str(
            self.runtime_input_mode_meta.get(INPUT_MODE_EFFECTIVE_KEY, TABLE_PLUS_STRUCTURE)
        ).strip().lower()
        self._validate_table_only_contract()

    @staticmethod
    def _coord_feature_usage(coord_features_cfg: dict[str, Any]) -> str:
        usage = str(coord_features_cfg.get("usage", "model_input")).strip().lower()
        if usage not in {"model_input", "supervision_only"}:
            raise ValueError(
                "preprocessing.coord_features.usage must be one of: "
                "model_input, supervision_only"
            )
        return usage

    def _validate_table_only_contract(self) -> None:
        if self.input_mode_effective != TABLE_ONLY:
            return
        structure = dict(self.runtime_cfg.get("structure", {}))
        feature_profile = str(structure.get("feature_profile", "none")).strip().lower()
        descriptor_profile = str(structure.get("descriptor_profile", "none")).strip().lower()
        latent_profile = str(structure.get("latent_profile", "none")).strip().lower()
        if feature_profile != "none" or descriptor_profile != "none" or latent_profile != "none":
            raise ValueError(
                "runtime.input_mode=table_only requires runtime.structure feature/descriptor/latent profiles to be none"
            )
        coord_features_cfg = dict(self.cfg.get("coord_features", {}))
        usage = self._coord_feature_usage(coord_features_cfg)
        channels_from_profile = str(coord_features_cfg.get("channels_from_profile", "")).strip().lower()
        if channels_from_profile and usage != "supervision_only":
            raise ValueError(
                "runtime.input_mode=table_only allows preprocessing.coord_features.channels_from_profile "
                "only when usage=supervision_only"
            )
        if usage == "supervision_only" and not channels_from_profile:
            raise ValueError(
                "preprocessing.coord_features.usage=supervision_only requires channels_from_profile"
            )
        if usage == "supervision_only" and not bool(coord_features_cfg.get("enabled", False)):
            raise ValueError(
                "preprocessing.coord_features.usage=supervision_only requires enabled=true"
            )

    def _resolve_runtime_feature_profile(self) -> str:
        profile = str(
            self.runtime_input_mode_meta.get(
                STRUCTURE_FEATURE_PROFILE_EFFECTIVE_KEY,
                dict(self.runtime_cfg.get("structure", {})).get("feature_profile", "none"),
            )
        ).strip().lower()
        return profile or "none"

    def _resolve_runtime_descriptor_profile(self) -> str:
        profile = str(
            self.runtime_input_mode_meta.get(
                DESCRIPTOR_PROFILE_KEY,
                dict(self.runtime_cfg.get("structure", {})).get("descriptor_profile", "none"),
            )
        ).strip().lower()
        return profile or "none"

    def _resolve_runtime_latent_profile(self) -> str:
        profile = str(
            self.runtime_input_mode_meta.get(
                LATENT_PROFILE_KEY,
                dict(self.runtime_cfg.get("structure", {})).get("latent_profile", "none"),
            )
        ).strip().lower()
        return profile or "none"

    def _build_descriptor_artifact(
        self,
        *,
        geom: Any,
        descriptor_profile: str,
        cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        case_structure_flags = [case.get("structure_npz") is not None for case in cases]
        if any(case_structure_flags) and not all(case_structure_flags):
            missing = [str(case.get("case_id")) for case, present in zip(cases, case_structure_flags) if not present]
            raise ValueError(
                "case-specific structure descriptor requires structure_npz for every case; "
                f"missing_cases={missing[:8]}"
            )
        reference_geom = geom
        if all(case_structure_flags) and cases:
            shape = tuple(int(v) for v in np.asarray(geom.mask_plasma, dtype=np.float32).shape)
            reference_case = cases[0]
            reference_structure = _load_case_structure_npz(
                reference_case,
                shape=shape,
                require_mask_coil=False,
            )
            if "part_mask_stack" not in reference_structure:
                raise ValueError(
                    "case-specific structure descriptor requires structure npz key='part_mask_stack'; "
                    f"case={reference_case.get('case_id')}"
                )
            reference_mask = np.asarray(reference_structure["mask_plasma"], dtype=np.float32)
            reference_distance_signed = build_signed_distance_fields(reference_mask).astype(np.float32)
            reference_stack = np.asarray(reference_structure["part_mask_stack"], dtype=np.float32)
            reference_regions = dict(getattr(geom, "regions", {}) or {})
            reference_regions["part_mask_stack"] = reference_stack
            reference_regions["solid_union_mask"] = np.maximum.reduce(reference_stack, axis=0).astype(np.float32)
            reference_geom = replace(
                geom,
                mask_plasma=reference_mask,
                distance_any=np.abs(reference_distance_signed).astype(np.float32),
                dist0=np.abs(reference_distance_signed).astype(np.float32),
                distance_signed=reference_distance_signed,
                regions=reference_regions,
            )
        descriptor = build_structure_descriptor(descriptor_profile, reference_geom)
        payload = descriptor.to_npz_payload()
        meta_payload = descriptor.to_meta_dict()
        if all(case_structure_flags) and cases:
            shape = tuple(int(v) for v in np.asarray(geom.mask_plasma, dtype=np.float32).shape)
            case_descriptors = []
            case_ids: list[str] = []
            for case in cases:
                case_id = str(case.get("case_id"))
                structure = _load_case_structure_npz(case, shape=shape, require_mask_coil=False)
                if "part_mask_stack" not in structure:
                    raise ValueError(
                        "case-specific structure descriptor requires structure npz key='part_mask_stack'; "
                        f"case={case_id}"
                    )
                mask_plasma = np.asarray(structure["mask_plasma"], dtype=np.float32)
                distance_signed = build_signed_distance_fields(mask_plasma).astype(np.float32)
                part_mask_stack = np.asarray(structure["part_mask_stack"], dtype=np.float32)
                regions = dict(getattr(geom, "regions", {}) or {})
                regions["part_mask_stack"] = part_mask_stack
                regions["solid_union_mask"] = np.maximum.reduce(part_mask_stack, axis=0).astype(np.float32)
                case_geom = replace(
                    geom,
                    mask_plasma=mask_plasma,
                    distance_any=np.abs(distance_signed).astype(np.float32),
                    dist0=np.abs(distance_signed).astype(np.float32),
                    distance_signed=distance_signed,
                    regions=regions,
                )
                case_descriptor = build_structure_descriptor(descriptor_profile, case_geom)
                if case_descriptor.feature_names != descriptor.feature_names:
                    raise ValueError(
                        "case-specific structure descriptor feature contract differs from provider geometry; "
                        f"case={case_id}, provider_dim={descriptor.vector.shape[0]}, "
                        f"case_dim={case_descriptor.vector.shape[0]}. "
                        "Use a fixed part-slot layout for case descriptors."
                    )
                case_descriptors.append(case_descriptor)
                case_ids.append(case_id)
            vectors = np.stack(
                [np.asarray(item.vector, dtype=np.float32) for item in case_descriptors],
                axis=0,
            ).astype(np.float32)
            if vectors.shape != (len(cases), int(descriptor.vector.shape[0])):
                raise RuntimeError(
                    "case-specific structure descriptor matrix shape mismatch: "
                    f"got={vectors.shape}, expected={(len(cases), int(descriptor.vector.shape[0]))}"
                )
            if len(set(case_ids)) != len(case_ids):
                raise ValueError("case-specific structure descriptor case_ids must be unique")
            payload.update(
                {
                    "vectors": vectors,
                    "case_ids": np.asarray(case_ids, dtype=str),
                    "n_parts_by_case": np.asarray(
                        [int(item.n_parts) for item in case_descriptors], dtype=np.int64
                    ),
                    "n_part_slots_by_case": np.asarray(
                        [int(item.n_part_slots) for item in case_descriptors], dtype=np.int64
                    ),
                }
            )
            active_counts = [int(item.n_parts) for item in case_descriptors]
            slot_counts = [int(item.n_part_slots) for item in case_descriptors]
            meta_payload.update(
                {
                    "case_specific": True,
                    "case_count": int(len(case_ids)),
                    "case_ids": case_ids,
                    "row_order": "dataset_case_order",
                    "n_parts_by_case_min": int(min(active_counts)),
                    "n_parts_by_case_max": int(max(active_counts)),
                    "n_part_slots_by_case": sorted(set(slot_counts)),
                }
            )
        else:
            meta_payload.update(
                {
                    "case_specific": False,
                    "case_count": 0,
                    "case_ids": [],
                    "row_order": "static_provider_geometry",
                }
            )
        pack_rel = "features/structure_descriptor_pack.npz"
        self.store.save_npz(pack_rel, **payload)
        pack_path = Path(pack_rel)
        meta_rel = str(pack_path.parent / f"{pack_path.stem}_meta.json")
        self.store.save_json(meta_rel, meta_payload)
        return {
            "structure_descriptor_pack_path": pack_rel,
            "structure_descriptor_pack_meta_path": meta_rel,
            "structure_descriptor_dim": int(meta_payload["descriptor_dim"]),
            "structure_descriptor_n_parts": int(meta_payload["n_parts"]),
            "structure_descriptor_n_part_slots": int(meta_payload["n_part_slots"]),
            "structure_descriptor_case_specific": bool(meta_payload["case_specific"]),
            "structure_descriptor_case_count": int(meta_payload["case_count"]),
        }

    def _resolve_latent_artifact(
        self,
        *,
        latent_profile: str,
        geometry_root: str | Path,
    ) -> dict[str, Any]:
        if latent_profile == "none":
            return {
                "latent_feature_pack_path": "",
                "latent_feature_pack_meta_path": "",
                "latent_feature_dim": 0,
                DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
            }
        dst_rel = "features/latent_feature_pack.npz"
        dst_path = self.output_dir / dst_rel
        if not dst_path.exists():
            base = Path(geometry_root)
            candidates = [
                base / "geometry" / "latent_feature_pack.npz",
                base / "latent_feature_pack.npz",
            ]
            src = next((path for path in candidates if path.exists()), None)
            if src is None:
                raise ValueError(
                    "runtime.structure.latent_profile is enabled but latent artifact is missing. "
                    "Expected one of: "
                    f"{[str(p) for p in candidates]} or existing {dst_path}"
                )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_path)
        with np.load(dst_path, allow_pickle=True) as data:
            if "vector" not in data.files:
                raise ValueError(
                    f"latent artifact must include 'vector' key: {dst_path}"
                )
            vector = np.asarray(data["vector"], dtype=np.float32).reshape(-1)
            if vector.size < 1:
                raise ValueError(f"latent artifact vector must be non-empty: {dst_path}")
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"latent artifact vector must be finite: {dst_path}")
            names_arr = data["feature_names"] if "feature_names" in data.files else np.asarray([], dtype=object)
            names = [str(v) for v in np.asarray(names_arr).reshape(-1).tolist()]
            if names and len(names) != int(vector.shape[0]):
                raise ValueError(
                    "latent artifact feature_names length mismatch: "
                    f"len(names)={len(names)}, dim={int(vector.shape[0])}"
                )
        meta_rel = "features/latent_feature_pack_meta.json"
        meta_payload = {
            "profile": str(latent_profile),
            "latent_dim": int(vector.shape[0]),
            "feature_names": names,
        }
        self.store.save_json(meta_rel, meta_payload)
        return {
            "latent_feature_pack_path": dst_rel,
            "latent_feature_pack_meta_path": meta_rel,
            "latent_feature_dim": int(vector.shape[0]),
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: True,
        }

    def _resolve_coord_feature_channels(self, coord_features_cfg: dict[str, Any]) -> list[str]:
        usage = self._coord_feature_usage(coord_features_cfg)
        if self.input_mode_effective == TABLE_PLUS_STRUCTURE:
            runtime_profile = self._resolve_runtime_feature_profile()
            if "channels" in coord_features_cfg:
                raise ValueError(
                    "runtime.input_mode=table_plus_structure does not allow "
                    "preprocessing.coord_features.channels; use runtime.structure.feature_profile"
                )
            channels_from_profile = str(coord_features_cfg.get("channels_from_profile", "")).strip().lower()
            if channels_from_profile and channels_from_profile != runtime_profile:
                raise ValueError(
                    "preprocessing.coord_features.channels_from_profile must match runtime.structure.feature_profile "
                    f"for table_plus_structure: expected={runtime_profile!r}, got={channels_from_profile!r}"
                )
            return list(resolve_spatial_channels_for_feature_profile(runtime_profile))

        if usage == "supervision_only":
            if "channels" in coord_features_cfg:
                raise ValueError(
                    "preprocessing.coord_features.usage=supervision_only does not allow direct "
                    "channels; use channels_from_profile"
                )
            profile = str(coord_features_cfg.get("channels_from_profile", "")).strip().lower()
            if not profile:
                raise ValueError(
                    "preprocessing.coord_features.usage=supervision_only requires channels_from_profile"
                )
            return list(resolve_spatial_channels_for_feature_profile(profile))

        coord_feature_channels_raw = coord_features_cfg.get(
            "channels",
            ["x", "y", "distance_signed", "distance_any", "mask_plasma"],
        )
        return list(validate_coord_feature_channels(coord_feature_channels_raw))

    def _save_split_artifacts(
        self,
        *,
        split: dict[str, Any],
        split_interp_marginal: dict[str, Any],
        split_interp_overlap: dict[str, Any],
        split_interp: dict[str, Any],
        split_extrap: dict[str, Any],
        split_structure_holdout: dict[str, Any],
        split_structure_holdout_meta: dict[str, Any],
    ) -> None:
        self.store.save_json("split/split_random_v1.json", split)
        self.store.save_json("split/split_interp_marginal_v1.json", split_interp_marginal)
        self.store.save_json("split/split_interp_overlap_v1.json", split_interp_overlap)
        self.store.save_json("split/split_interp_v1.json", split_interp)
        self.store.save_json("split/split_extrap_v1.json", split_extrap)
        self.store.save_json("split/split_structure_holdout_v1.json", split_structure_holdout)
        self.store.save_json("split/split_structure_holdout_meta_v1.json", split_structure_holdout_meta)

    def _save_transform_bundle(
        self,
        *,
        rel_root: str,
        transforms: Any,
        fit_split: str,
        fit_train_case_count: int,
        mask_active_ratio: float,
    ) -> None:
        cond_payload = transforms.cond_scaler.to_dict()
        cond_payload["cond_dim"] = transforms.cond_dim
        cond_payload["fit_policy"] = transforms.fit_policy
        cond_payload["fit_split"] = str(fit_split)
        cond_payload["fit_train_case_count"] = int(fit_train_case_count)
        cond_payload["mask_applied"] = transforms.mask_applied
        cond_payload["target_transforms"] = transforms.target_transforms
        self.store.save_json(f"{rel_root}/cond_scaler.json", cond_payload)
        self.store.save_json(
            f"{rel_root}/y_scalers.json",
            {key: value.to_dict() for key, value in transforms.y_scalers.items()},
        )
        self.store.save_json(
            f"{rel_root}/fit_policy.json",
            {
                "y_fit_policy": transforms.fit_policy,
                "fit_split": str(fit_split),
                "fit_train_case_count": int(fit_train_case_count),
                "mask_applied": bool(transforms.mask_applied),
                "mask_active_ratio": float(mask_active_ratio),
                "target_transforms": transforms.target_transforms,
            },
        )

    def _save_schema_artifacts(
        self,
        *,
        cond_schema: CondSchema,
        axis_schema: AxisSchema,
        channel_map_payload: dict[str, Any],
        output_layout_payload: dict[str, Any],
        field_layout_payload: dict[str, Any],
        target_role_schema_payload: dict[str, Any],
    ) -> None:
        self.store.save_json("schema/cond_schema.json", cond_schema.to_dict())
        self.store.save_json("schema/axis_schema.json", axis_schema.to_dict())
        self.store.save_json("schema/channel_map.json", channel_map_payload)
        self.store.save_json("schema/output_layout.json", output_layout_payload)
        self.store.save_json("schema/field_layout.json", field_layout_payload)
        self.store.save_json("schema/target_role_schema.json", target_role_schema_payload)

    def _save_runtime_schema_hashes(
        self,
        *,
        artifacts: PreprocessArtifacts,
    ) -> dict[str, str]:
        runtime_schema_hashes = build_runtime_schema_hashes(artifacts.runtime_schema_payload())
        self.store.save_json("validation/runtime_schema_hashes.json", runtime_schema_hashes)
        return runtime_schema_hashes

    def run(
        self,
        cases: list[dict[str, Any]],
        geometry_root: str | Path,
        target_metadata: list[dict[str, Any]] | None = None,
    ) -> PreprocessOutput:
        split_cfg = self.cfg.get("split", {})
        cond_order = list(self.cfg.get("cond_order", []))
        if not cond_order:
            cond_order = sorted(cases[0]["cond"].keys())
        cond_schema = CondSchema(order=cond_order)
        axis_schema = AxisSchema.from_dict(self.cfg.get("axis_schema", {"mode": "steady"}))
        split_plan = SplitPlanBuilder(split_cfg).build(cases=cases, cond_order=cond_order)
        split = split_plan.random
        split_interp_marginal = split_plan.interp_marginal
        split_interp_overlap = split_plan.interp_overlap
        split_interp = split_plan.interp
        split_extrap = split_plan.extrap
        split_structure_holdout = split_plan.structure_holdout
        split_structure_holdout_meta = split_plan.structure_holdout_meta
        extrap_cfg = dict(split_cfg.get("extrapolation", {}) or {})
        extrap_key = str(extrap_cfg.get("key", cond_order[0]))
        extrap_direction = str(extrap_cfg.get("direction", "high")).strip().lower()
        cond_matrix = []
        for c in cases:
            base = cond_schema.encode(c["cond"])
            axis_vec = axis_schema.encode(c.get("axis", 0.0))
            cond_matrix.append(np.concatenate([base, axis_vec], axis=0))
        cond_matrix = np.stack(cond_matrix, axis=0)

        y_vars = _resolve_y_vars(cases)
        y_by_var = {var: np.stack([np.asarray(c["y"][var], dtype=np.float32) for c in cases], axis=0) for var in y_vars}
        coord_grid_source = str(self.cfg.get("coord_grid_source", "normalized"))
        distance_contract_cfg = dict(self.cfg.get("distance_contract", {}))
        distance_contract_mode = str(distance_contract_cfg.get("require_negative_outside", "error")).strip().lower()
        if distance_contract_mode not in {"error", "off"}:
            raise ValueError(
                "preprocessing.distance_contract.require_negative_outside must be one of: error, off"
            )
        coord_contract_cfg = dict(self.cfg.get("coord_grid_contract", {}))
        coord_contract_mode = str(coord_contract_cfg.get("require_requested_source", "error")).strip().lower()
        if coord_contract_mode not in {"error", "off"}:
            raise ValueError(
                "preprocessing.coord_grid_contract.require_requested_source must be one of: error, off"
            )
        provider_mode = str(
            self.runtime_input_mode_meta.get(
                GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
                dict(self.runtime_cfg.get("structure", {})).get("provider_mode", "fixed"),
            )
        ).strip().lower() or "fixed"
        descriptor_profile = self._resolve_runtime_descriptor_profile()
        latent_profile = self._resolve_runtime_latent_profile()
        geom_provider = build_geometry_provider(
            geometry_root,
            provider_mode=provider_mode,
            coord_grid_source=coord_grid_source,
        )
        geom_ref: dict[str, Any] = {"geom_id": "default"}
        if (
            self.input_mode_effective == TABLE_PLUS_STRUCTURE
            and descriptor_profile != "none"
            and provider_mode == "parametric_parts"
        ):
            # Use default parametric-parts reconstruction so descriptor lane receives part masks.
            geom_ref["geom_param"] = {}
        geom = geom_provider.get(geom_ref)
        raw_coord_grid = np.asarray(geom.coord_grid, dtype=np.float32)
        if raw_coord_grid.ndim != 3:
            raise ValueError(f"Geometry coord_grid must be rank-3, got shape={raw_coord_grid.shape}")
        if raw_coord_grid.shape[0] == 2:
            coord_rows = raw_coord_grid.reshape(2, -1).T.astype(np.float32)
        elif raw_coord_grid.shape[-1] == 2:
            coord_rows = raw_coord_grid.reshape(-1, 2).astype(np.float32)
        else:
            raise ValueError(
                "Geometry coord_grid shape mismatch: expected (2,H,W) or (H,W,2), "
                f"got {raw_coord_grid.shape}"
            )
        coord_scaler_z = ScalerFactory.create("zscore").fit(coord_rows)
        coord_scaler_mm = ScalerFactory.create("minmax").fit(coord_rows)
        coord_rows_z = coord_scaler_z.transform(coord_rows).astype(np.float32)
        coord_rows_mm = coord_scaler_mm.transform(coord_rows).astype(np.float32)

        scaler_cfg = dict(self.cfg.get("scalers", {}))
        scaler_fit_split = str(scaler_cfg.get("fit_split", "random")).strip().lower()
        split_for_scalers_by_name = {
            "random": split,
            "casewise": split,
            "structure_holdout": split_structure_holdout,
            "interp": split_interp,
            "extrap": split_extrap,
        }
        if scaler_fit_split not in split_for_scalers_by_name:
            raise ValueError(
                "preprocessing.scalers.fit_split must be one of: "
                "random, interp, extrap, structure_holdout"
            )
        split_for_scalers = split_for_scalers_by_name[scaler_fit_split]
        case_to_idx = {c["case_id"]: i for i, c in enumerate(cases)}
        train_indices = np.array([case_to_idx[cid] for cid in split_for_scalers["train"]], dtype=np.int64)
        y_fit_policy = str(scaler_cfg.get("y_fit_policy", "all"))
        target_transforms_cfg = dict(scaler_cfg.get("target_transforms", {}))
        if "target_transform_policy" in scaler_cfg:
            raise ValueError(
                "preprocessing.scalers.target_transform_policy is removed; "
                "use preprocessing.scalers.target_transforms.<var>"
            )
        if len(target_transforms_cfg) == 0:
            raise ValueError(
                "preprocessing.scalers.target_transforms is required. "
                "Define per-var config: value_transform/scaler/fit_scope/clip for each target var."
            )
        missing_transform_vars = [name for name in y_vars if name not in target_transforms_cfg]
        if missing_transform_vars:
            raise ValueError(
                "preprocessing.scalers.target_transforms is incomplete. "
                f"Missing vars: {missing_transform_vars}"
            )
        cond_scaler_type = str(scaler_cfg.get("cond") or "zscore")
        y_scaler_type = str(scaler_cfg.get("target") or "zscore")
        plasma_scope_requested = y_fit_policy.strip().lower() == "plasma_only" or any(
            str(dict(target_transforms_cfg.get(var, {})).get("fit_scope", y_fit_policy)).strip().lower()
            == "plasma_only"
            for var in y_vars
        )
        scaler_mask_plasma = (
            _materialize_scaler_plasma_mask(cases, static_mask=np.asarray(geom.mask_plasma, dtype=np.float32))
            if plasma_scope_requested
            else np.asarray(geom.mask_plasma, dtype=np.float32)
        )
        transforms = fit_scalers_train_only(
            cond_matrix,
            y_by_var,
            train_indices,
            mask_plasma=scaler_mask_plasma,
            scaler_fit_policy=y_fit_policy,
            cond_scaler_type=cond_scaler_type,
            y_scaler_type=y_scaler_type,
            target_transforms=target_transforms_cfg,
        )
        protocol_transforms: dict[str, Any] = {}
        for protocol_name, protocol_split in split_for_scalers_by_name.items():
            if protocol_name == "casewise":
                continue
            protocol_train_indices = np.array(
                [case_to_idx[cid] for cid in protocol_split["train"]],
                dtype=np.int64,
            )
            protocol_transforms[protocol_name] = fit_scalers_train_only(
                cond_matrix,
                y_by_var,
                protocol_train_indices,
                mask_plasma=scaler_mask_plasma,
                scaler_fit_policy=y_fit_policy,
                cond_scaler_type=cond_scaler_type,
                y_scaler_type=y_scaler_type,
                target_transforms=target_transforms_cfg,
            )

        channel_map = ChannelMap.from_dict(self.cfg.get("channel_map", {"channels": []}))
        pools = build_point_pools(
            mask_plasma=geom.mask_plasma,
            distance_any=geom.distance_any,
            delta_edge=float(self.cfg.get("sampling", {}).get("delta_edge", 1.0)),
            delta_bulk=float(self.cfg.get("sampling", {}).get("delta_bulk", 3.0)),
            wafer_mask=geom.regions.get("wafer_mask"),
        )
        patches = build_patch_index(geom.mask_plasma.shape, (4, 4), (2, 2))

        self._save_split_artifacts(
            split=split,
            split_interp_marginal=split_interp_marginal,
            split_interp_overlap=split_interp_overlap,
            split_interp=split_interp,
            split_extrap=split_extrap,
            split_structure_holdout=split_structure_holdout,
            split_structure_holdout_meta=split_structure_holdout_meta,
        )
        featurization_root = self.cfg.get("featurization_root")
        x_grid = None
        channel_names: list[str] = []
        if featurization_root is not None:
            fdir = Path(str(featurization_root)) / "geometry_cache" / "default"
            x_grid_path = fdir / "x_grid.npy"
            names_path = fdir / "channel_names.npy"
            if x_grid_path.exists() and names_path.exists():
                x_grid = np.load(x_grid_path).astype(np.float32)
                channel_names = [str(n) for n in np.load(names_path, allow_pickle=True).tolist()]

        if len(channel_map.channels) == 0 and x_grid is not None and len(channel_names) == x_grid.shape[0]:
            channel_map = ChannelMap.from_dict(
                {
                    "channels": [
                        {
                            "name": name,
                            "source": f"x_grid[{i}]",
                            "normalize": "none" if ("mask" in name or "onehot" in name) else "zscore",
                            "role": "feature",
                        }
                        for i, name in enumerate(channel_names)
                    ]
                }
            )
        channel_map_payload = channel_map.to_dict()
        sample_shape = np.asarray(cases[0]["y"][y_vars[0]], dtype=np.float32).shape
        output_layout_payload = {
            "order": "C",
            "shape": [len(y_vars), int(sample_shape[0]), int(sample_shape[1])],
            "vars": y_vars,
        }
        field_layout_payload = _build_grid2d_field_layout(output_layout_payload)
        target_role_schema_payload = build_target_role_schema(target_metadata, y_vars=y_vars)
        self._save_schema_artifacts(
            cond_schema=cond_schema,
            axis_schema=axis_schema,
            channel_map_payload=channel_map_payload,
            output_layout_payload=output_layout_payload,
            field_layout_payload=field_layout_payload,
            target_role_schema_payload=target_role_schema_payload,
        )
        mask_active_ratio = _scaler_mask_active_ratio(scaler_mask_plasma, train_indices)
        self._save_transform_bundle(
            rel_root="scalers",
            transforms=transforms,
            fit_split=scaler_fit_split,
            fit_train_case_count=int(len(train_indices)),
            mask_active_ratio=mask_active_ratio,
        )
        for protocol_name, protocol_bundle in protocol_transforms.items():
            protocol_train_indices = np.array(
                [case_to_idx[cid] for cid in split_for_scalers_by_name[protocol_name]["train"]],
                dtype=np.int64,
            )
            protocol_train_count = len(protocol_train_indices)
            self._save_transform_bundle(
                rel_root=f"scalers/by_split/{protocol_name}",
                transforms=protocol_bundle,
                fit_split=protocol_name,
                fit_train_case_count=protocol_train_count,
                mask_active_ratio=_scaler_mask_active_ratio(
                    scaler_mask_plasma,
                    protocol_train_indices,
                ),
            )
        self.store.save_json(
            "scalers/coord_scaler.json",
            {
                "coord_dim": int(coord_rows.shape[1]),
                "status": "ok",
                "raw": {
                    "x_min": float(np.min(coord_rows[:, 0])),
                    "x_max": float(np.max(coord_rows[:, 0])),
                    "y_min": float(np.min(coord_rows[:, 1])),
                    "y_max": float(np.max(coord_rows[:, 1])),
                },
                "zscore": coord_scaler_z.to_dict(),
                "minmax": coord_scaler_mm.to_dict(),
                "none": {"type": "none"},
            },
        )
        xgrid_scalers: dict[str, Any] = {}
        if x_grid is not None and len(channel_names) == x_grid.shape[0]:
            for i, name in enumerate(channel_names):
                values = x_grid[i].reshape(-1, 1)
                normalize = "none" if ("mask" in name or "onehot" in name) else "zscore"
                scaler = ScalerFactory.create(normalize).fit(values)
                xgrid_scalers[name] = scaler.to_dict()
        self.store.save_json("scalers/xgrid_channel_scalers.json", xgrid_scalers)
        self.store.save_npz("sampling/point_pools/default.npz", **pools)
        raw_signed = getattr(geom, "distance_signed", None)
        if raw_signed is None:
            distance_signed = np.where(geom.mask_plasma > 0.5, geom.distance_any, -geom.distance_any).astype(np.float32)
        else:
            distance_signed = np.asarray(raw_signed, dtype=np.float32)
            if distance_signed.shape != geom.mask_plasma.shape:
                raise ValueError(
                    "GeometryContext.distance_signed shape mismatch: "
                    f"expected {geom.mask_plasma.shape}, got {distance_signed.shape}"
                )
        geom_sampling_dir = self.output_dir / "sampling" / "geometry"
        geom_sampling_dir.mkdir(parents=True, exist_ok=True)
        np.save(geom_sampling_dir / "distance_signed.npy", distance_signed)
        coord_features_cfg = dict(self.cfg.get("coord_features", {}))
        distance_stats_cfg = dict(coord_features_cfg.get("distance_transform_stats", {}))
        distance_stats_enabled = bool(distance_stats_cfg.get("enabled", False))
        distance_stats_fit_scope = str(distance_stats_cfg.get("fit_scope", "train_split")).strip().lower()
        if distance_stats_fit_scope not in {"train_split", "all"}:
            raise ValueError(
                "preprocessing.coord_features.distance_transform_stats.fit_scope must be one of: train_split, all"
            )
        distance_stats_mask_scope = str(distance_stats_cfg.get("mask_scope", "plasma_plus_band")).strip().lower()
        if distance_stats_mask_scope not in {"all", "plasma_only", "plasma_plus_band"}:
            raise ValueError(
                "preprocessing.coord_features.distance_transform_stats.mask_scope must be one of: "
                "all, plasma_only, plasma_plus_band"
            )
        distance_stats_band = float(distance_stats_cfg.get("chamber_band_px", 2.0))
        if distance_stats_band < 0.0:
            raise ValueError("preprocessing.coord_features.distance_transform_stats.chamber_band_px must be >= 0")
        signed_q_raw = float(distance_stats_cfg.get("signed_quantile", 0.75))
        proximity_q_raw = float(distance_stats_cfg.get("proximity_quantile", 0.50))

        def _as_percentile(raw: float, *, key: str) -> float:
            q = float(raw)
            if 0.0 < q <= 1.0:
                return q * 100.0
            if 1.0 < q <= 100.0:
                return q
            raise ValueError(
                f"preprocessing.coord_features.distance_transform_stats.{key} must be in (0,1] or (1,100]"
            )

        signed_pct = _as_percentile(signed_q_raw, key="signed_quantile")
        proximity_pct = _as_percentile(proximity_q_raw, key="proximity_quantile")
        stats_mask = np.ones_like(geom.mask_plasma, dtype=bool)
        if distance_stats_mask_scope == "plasma_only":
            stats_mask = geom.mask_plasma > 0.5
        elif distance_stats_mask_scope == "plasma_plus_band":
            stats_mask = np.asarray(distance_signed, dtype=np.float32) >= (-float(distance_stats_band))
        if distance_stats_fit_scope == "all":
            stats_mask = np.ones_like(stats_mask, dtype=bool)
        if not np.any(stats_mask):
            stats_mask = np.ones_like(stats_mask, dtype=bool)
        abs_signed_vals = np.abs(np.asarray(distance_signed, dtype=np.float32)[stats_mask])
        proximity_vals = np.asarray(geom.distance_any, dtype=np.float32)[stats_mask]
        signed_tanh_tau_auto = float(max(np.percentile(abs_signed_vals, signed_pct), 1e-3))
        proximity_tau_auto = float(max(np.percentile(proximity_vals, proximity_pct), 1e-3))
        distance_stats_payload = {
            "enabled": bool(distance_stats_enabled),
            "fit_scope": distance_stats_fit_scope,
            "mask_scope": distance_stats_mask_scope,
            "chamber_band_px": float(distance_stats_band),
            "signed_quantile": float(signed_q_raw),
            "proximity_quantile": float(proximity_q_raw),
            "signed_tanh_tau_auto": signed_tanh_tau_auto,
            "proximity_tau_auto": proximity_tau_auto,
        }
        self.store.save_json("scalers/distance_transform_stats.json", distance_stats_payload)
        coord_features_enabled = bool(coord_features_cfg.get("enabled", False))
        coord_features_scaling_cfg = dict(coord_features_cfg.get("scaling", {}))
        coord_features_scaling_enabled = bool(coord_features_scaling_cfg.get("enabled", False))
        coord_features_scaling_mode = str(coord_features_scaling_cfg.get("mode", "zscore")).strip().lower()
        if coord_features_scaling_mode not in {"none", "zscore", "minmax"}:
            raise ValueError("preprocessing.coord_features.scaling.mode must be one of: none, zscore, minmax")
        coord_features_scaling_fit_scope = str(
            coord_features_scaling_cfg.get("fit_scope", "train_split")
        ).strip().lower()
        if coord_features_scaling_fit_scope not in {"train_split", "all"}:
            raise ValueError("preprocessing.coord_features.scaling.fit_scope must be one of: train_split, all")
        coord_features_scaling_mask_scope = str(
            coord_features_scaling_cfg.get("mask_scope", "plasma_plus_band")
        ).strip().lower()
        if coord_features_scaling_mask_scope not in {"all", "plasma_only", "plasma_plus_band"}:
            raise ValueError(
                "preprocessing.coord_features.scaling.mask_scope must be one of: all, plasma_only, plasma_plus_band"
            )
        coord_features_scaling_band = float(coord_features_scaling_cfg.get("chamber_band_px", 2.0))
        if coord_features_scaling_band < 0.0:
            raise ValueError("preprocessing.coord_features.scaling.chamber_band_px must be >= 0")
        coord_feature_channels = self._resolve_coord_feature_channels(coord_features_cfg)
        coord_feature_rel_path = str(coord_features_cfg.get("output", "features/coord_feature_pack.npz"))
        coord_x, coord_y = _coord_xy_maps(raw_coord_grid)
        mask_plasma_map = np.asarray(geom.mask_plasma, dtype=np.float32)
        geom_shape = tuple(int(v) for v in mask_plasma_map.shape)
        regions = dict(getattr(geom, "regions", {}) or {})
        runtime_feature_profile = self._resolve_runtime_feature_profile()
        coord_feature_usage = self._coord_feature_usage(coord_features_cfg)
        supervision_feature_profile = str(
            coord_features_cfg.get("channels_from_profile", "")
        ).strip().lower()
        case_feature_profile = (
            supervision_feature_profile
            if coord_feature_usage == "supervision_only"
            else runtime_feature_profile
        )
        case_structure_available = any(case.get("structure_npz") is not None for case in cases)
        case_spatial_feature_enabled = bool(
            coord_features_enabled
            and (
                self.input_mode_effective == TABLE_PLUS_STRUCTURE
                or coord_feature_usage == "supervision_only"
            )
            and (
                case_feature_profile in {
                    "icp_struct_spatial_v1",
                    "icp_part_sdf_lite_v1",
                    "icp_coil_source_v1",
                    "icp_coil_union_sdf_v1",
                    "icp_coil_sdf_source_v2",
                    "icp_coil_sdf_source_mean_v3",
                    "icp_vacuum_field_v1",
                    "icp_coil_sdf_source_mean_vacuum_v1",
                    "icp_coil_sdf_source_mean_vacuum_structure_v1",
                }
                or (
                    case_feature_profile in {"part_lite_v1", "smooth_structure_v1", "part_source_v1"}
                    and case_structure_available
                )
            )
        )
        part_stack = None
        if (not case_spatial_feature_enabled) and any(name in PART_SDF_SUMMARY_CHANNELS for name in coord_feature_channels):
            part_stack = regions.get("part_mask_stack")
        coord_xy_flat = np.stack(
            [
                np.asarray(coord_x, dtype=np.float32).reshape(-1),
                np.asarray(coord_y, dtype=np.float32).reshape(-1),
            ],
            axis=1,
        ).astype(np.float32)
        coord_feature_maps = {
            name: np.asarray(arr, dtype=np.float32).reshape(geom_shape)
            for name, arr in derive_geom_feature_maps(
                coord_xy=coord_xy_flat,
                distance_signed=np.asarray(distance_signed, dtype=np.float32).reshape(-1),
                distance_any=np.asarray(geom.distance_any, dtype=np.float32).reshape(-1),
                mask_plasma=mask_plasma_map.reshape(-1),
                h=geom_shape[0],
                w=geom_shape[1],
                part_mask_stack=np.asarray(part_stack, dtype=np.float32) if part_stack is not None else None,
            ).items()
        }
        if not case_spatial_feature_enabled:
            missing_coord_feature_channels = [name for name in coord_feature_channels if name not in coord_feature_maps]
            if missing_coord_feature_channels:
                raise ValueError(
                    "preprocessing.coord_features requested channels that are unavailable for the current geometry: "
                    f"{missing_coord_feature_channels}"
                )
        static_spatial_feature_rel_path = str(
            coord_features_cfg.get("static_output", "features/static_spatial_feature_pack.npz")
        )
        case_structure_feature_rel_path = str(
            coord_features_cfg.get("case_structure_output", "features/case_structure_feature_pack.npz")
        )
        static_spatial_feature_meta_rel = ""
        case_structure_feature_meta_rel = ""
        case_spatial_feature_meta_rel = ""
        case_spatial_feature_shape: list[int] = []
        static_spatial_feature_shape: list[int] = []
        case_structure_feature_shape: list[int] = []
        static_spatial_feature_data: np.ndarray | None = None
        case_structure_feature_data: np.ndarray | None = None
        case_spatial_meta: dict[str, Any] = {}
        coord_feature_pack_meta_payload: dict[str, Any] = {}
        compact_fit_mask = np.ones_like(coord_x, dtype=bool)
        if case_spatial_feature_enabled:
            coil_tau_raw = coord_features_cfg.get("coil_proximity_tau_fixed", max(geom_shape))
            coil_tau_fixed = float(coil_tau_raw)
            if not np.isfinite(coil_tau_fixed) or coil_tau_fixed <= 0.0:
                raise ValueError("preprocessing.coord_features.coil_proximity_tau_fixed must be finite and > 0")
            static_spatial_feature_data, case_structure_feature_data, case_spatial_meta = (
                _build_split_spatial_feature_packs(
                    cases=cases,
                    channels=coord_feature_channels,
                    coord_x=coord_x,
                    coord_y=coord_y,
                    train_indices=train_indices,
                    coil_proximity_percentile=proximity_pct,
                    coil_proximity_tau_fixed=coil_tau_fixed,
                )
            )
            if bool(coord_features_cfg.get("require_case_variation", False)) and not bool(
                case_spatial_meta.get("case_feature_varies", False)
            ):
                raise ValueError(
                    "preprocessing.coord_features.require_case_variation=true but all case-specific "
                    "structure feature maps are identical; verify structure_npz routing and source geometry"
                )
            static_channels_effective = tuple(case_spatial_meta.get("static_channels", ICP_STRUCT_STATIC_CHANNELS))
            static_map = {
                name: static_spatial_feature_data[i]
                for i, name in enumerate(list(static_channels_effective))
            }
            signed_static = static_map["distance_signed"]
            any_static = np.asarray(
                static_map.get("distance_any", np.abs(signed_static)),
                dtype=np.float32,
            )
            # The plasma mask is supervision metadata, not a required model
            # input.  Minimal feature profiles therefore use the geometry
            # contract directly when fitting distance-feature statistics.
            mask_static = np.asarray(mask_plasma_map, dtype=np.float32) > 0.5
            stats_mask = np.ones_like(mask_static, dtype=bool)
            if distance_stats_mask_scope == "plasma_only":
                stats_mask = mask_static
            elif distance_stats_mask_scope == "plasma_plus_band":
                stats_mask = signed_static >= (-float(distance_stats_band))
            if not np.any(stats_mask):
                stats_mask = np.ones_like(stats_mask, dtype=bool)
            abs_signed_vals = np.abs(signed_static[stats_mask])
            proximity_vals = any_static[stats_mask]
            signed_tanh_tau_auto = float(max(np.percentile(abs_signed_vals, signed_pct), 1e-3))
            proximity_tau_auto = float(max(np.percentile(proximity_vals, proximity_pct), 1e-3))
            distance_stats_payload.update(
                {
                    "signed_tanh_tau_auto": signed_tanh_tau_auto,
                    "proximity_tau_auto": proximity_tau_auto,
                    "coil_proximity_tau": float(case_spatial_meta.get("coil_proximity_tau", proximity_tau_auto)),
                    "source": "static_spatial_feature_pack",
                }
            )
            self.store.save_json("scalers/distance_transform_stats.json", distance_stats_payload)
            compact_fit_mask = np.ones_like(mask_static, dtype=bool)
            if coord_features_scaling_mask_scope == "plasma_only":
                compact_fit_mask = mask_static
            elif coord_features_scaling_mask_scope == "plasma_plus_band":
                compact_fit_mask = signed_static >= (-float(coord_features_scaling_band))
            if coord_features_scaling_fit_scope == "all":
                compact_fit_mask = np.ones_like(compact_fit_mask, dtype=bool)
            if not np.any(compact_fit_mask):
                raise ValueError("coord feature scaler fit mask is empty")
            case_spatial_meta["scaling"] = {
                "enabled": bool(coord_features_scaling_enabled),
                "mode": str(coord_features_scaling_mode),
                "fit_scope": str(coord_features_scaling_fit_scope),
                "mask_scope": str(coord_features_scaling_mask_scope),
            }
            case_spatial_feature_shape = list(case_spatial_meta["logical_shape"])
            static_spatial_feature_shape = list(case_spatial_meta["static_shape"])
            case_structure_feature_shape = list(case_spatial_meta["case_shape"])
        coord_distance_transform_effective, coord_distance_transform_source = resolve_distance_transform_effective(
            self.coord_distance_transform_cfg,
            stats=distance_stats_payload,
        )
        fit_mask = np.ones_like(geom.mask_plasma, dtype=bool)
        if coord_features_scaling_mask_scope == "plasma_only":
            fit_mask = geom.mask_plasma > 0.5
        elif coord_features_scaling_mask_scope == "plasma_plus_band":
            fit_mask = np.asarray(distance_signed, dtype=np.float32) >= (-float(coord_features_scaling_band))
        if coord_features_scaling_fit_scope == "all":
            fit_mask = np.ones_like(fit_mask, dtype=bool)
        if not np.any(fit_mask):
            raise ValueError("coord feature scaler fit mask is empty")

        def _fit_coord_feature_scalers(
            fit_case_indices: np.ndarray,
            *,
            fit_split_name: str,
            force_case_train_scope: bool,
        ) -> dict[str, Any]:
            effective_fit_scope = (
                "train_split"
                if case_spatial_feature_enabled and force_case_train_scope
                else coord_features_scaling_fit_scope
            )
            payload: dict[str, Any] = {
                "contract_version": 3,
                "enabled": bool(coord_features_scaling_enabled),
                "mode": str(coord_features_scaling_mode),
                "input_space": "post_distance_transform",
                "distance_transform_effective": dict(coord_distance_transform_effective),
                "distance_transform_source": str(coord_distance_transform_source),
                "fit_scope": str(effective_fit_scope),
                "fit_scope_configured": str(coord_features_scaling_fit_scope),
                "fit_split": str(fit_split_name),
                "fit_train_case_count": int(len(fit_case_indices)),
                "train_only": bool(effective_fit_scope == "train_split"),
                "mask_scope": str(coord_features_scaling_mask_scope),
                "chamber_band_px": float(coord_features_scaling_band),
                "channels": {},
            }
            for name in coord_feature_channels:
                if case_spatial_feature_enabled:
                    if static_spatial_feature_data is None or case_structure_feature_data is None:
                        raise RuntimeError("compact case spatial feature pack was not initialized")
                    static_channels_effective = tuple(
                        case_spatial_meta.get("static_channels", ICP_STRUCT_STATIC_CHANNELS)
                    )
                    case_channels_effective = tuple(
                        case_spatial_meta.get("case_channels", ICP_STRUCT_CASE_CHANNELS)
                    )
                    if name in static_channels_effective:
                        static_idx = list(static_channels_effective).index(name)
                        values = static_spatial_feature_data[static_idx].reshape(-1, 1)
                        fit_values = values[compact_fit_mask.reshape(-1)]
                    elif name in case_channels_effective:
                        case_idx = list(case_channels_effective).index(name)
                        selected_case_indices = (
                            np.arange(case_structure_feature_data.shape[0], dtype=np.int64)
                            if effective_fit_scope == "all"
                            else np.asarray(fit_case_indices, dtype=np.int64)
                        )
                        values = case_structure_feature_data[selected_case_indices, case_idx].reshape(-1, 1)
                        repeated_mask = np.broadcast_to(
                            compact_fit_mask.reshape(1, -1),
                            (int(len(selected_case_indices)), int(compact_fit_mask.size)),
                        ).reshape(-1)
                        fit_values = values[repeated_mask]
                    else:
                        raise ValueError(f"compact case spatial feature channel is unavailable: {name}")
                else:
                    values = np.asarray(coord_feature_maps[name], dtype=np.float32).reshape(-1, 1)
                    fit_values = values[fit_mask.reshape(-1)]
                if (
                    coord_features_scaling_enabled
                    and coord_features_scaling_mode != "none"
                    and not _is_mask_like_channel(name)
                ):
                    scaler_fit_values, _ = apply_distance_transform(
                        fit_values,
                        channels=[name],
                        cfg=coord_distance_transform_effective,
                    )
                    scaler = ScalerFactory.create(coord_features_scaling_mode).fit(scaler_fit_values)
                else:
                    scaler = ScalerFactory.create("none").fit(values)
                payload["channels"][name] = scaler.to_dict()
            return payload

        coord_feature_scalers_payload = _fit_coord_feature_scalers(
            train_indices,
            fit_split_name=scaler_fit_split,
            force_case_train_scope=False,
        )
        self.store.save_json("scalers/coord_feature_scaler.json", coord_feature_scalers_payload)
        for protocol_name, protocol_split in split_for_scalers_by_name.items():
            if protocol_name == "casewise":
                continue
            protocol_train_indices = np.asarray(
                [case_to_idx[cid] for cid in protocol_split["train"]],
                dtype=np.int64,
            )
            protocol_coord_payload = _fit_coord_feature_scalers(
                protocol_train_indices,
                fit_split_name=protocol_name,
                force_case_train_scope=True,
            )
            self.store.save_json(
                f"scalers/by_split/{protocol_name}/coord_feature_scaler.json",
                protocol_coord_payload,
            )
            protocol_distance_payload = {
                **distance_stats_payload,
                "fit_split": protocol_name,
                "fit_train_case_count": int(len(protocol_train_indices)),
                "train_only": True,
            }
            self.store.save_json(
                f"scalers/by_split/{protocol_name}/distance_transform_stats.json",
                protocol_distance_payload,
            )
        if case_spatial_feature_enabled:
            if static_spatial_feature_data is None or case_structure_feature_data is None:
                raise RuntimeError("compact case spatial feature pack was not initialized")
            self.store.save_npz(
                static_spatial_feature_rel_path,
                data=static_spatial_feature_data,
                channels=np.asarray(list(case_spatial_meta.get("static_channels", ICP_STRUCT_STATIC_CHANNELS))),
            )
            self.store.save_npz(
                case_structure_feature_rel_path,
                data=case_structure_feature_data,
                channels=np.asarray(list(case_spatial_meta.get("case_channels", ICP_STRUCT_CASE_CHANNELS))),
                case_ids=np.asarray([str(c["case_id"]) for c in cases]),
            )
            static_path = Path(static_spatial_feature_rel_path)
            case_path = Path(case_structure_feature_rel_path)
            static_spatial_feature_meta_rel = str(static_path.parent / f"{static_path.stem}_meta.json")
            case_structure_feature_meta_rel = str(case_path.parent / f"{case_path.stem}_meta.json")
            self.store.save_json(static_spatial_feature_meta_rel, case_spatial_meta)
            self.store.save_json(case_structure_feature_meta_rel, case_spatial_meta)
        elif coord_features_enabled:
            coord_feature_data = np.stack([coord_feature_maps[name] for name in coord_feature_channels], axis=0).astype(np.float32)
            self.store.save_npz(coord_feature_rel_path, data=coord_feature_data, channels=np.asarray(coord_feature_channels))
            pack_path = Path(coord_feature_rel_path)
            meta_rel = str(pack_path.parent / f"{pack_path.stem}_meta.json")
            coord_feature_pack_meta_payload = {
                "enabled": True,
                "channels": coord_feature_channels,
                "shape": [int(v) for v in coord_feature_data.shape],
                "coord_source": str(getattr(geom, "coord_source", "unknown")),
                "scaling": {
                    "enabled": bool(coord_features_scaling_enabled),
                    "mode": str(coord_features_scaling_mode),
                },
            }
            self.store.save_json(meta_rel, coord_feature_pack_meta_payload)
        self.store.save_npz("sampling/patch_index/patches_train.npz", patches=patches)
        sample_ids = list(range(len(cases)))
        axis_values = [float(c.get("axis", 0.0)) for c in cases]
        phase_pairs = (
            build_phase_wrap_pairs(sample_ids, axis_values=axis_values) if axis_schema.mode == "phase_sincos" else []
        )
        time_pairs = build_time_adjacent_pairs(sample_ids, axis_values=axis_values) if axis_schema.mode == "time" else []
        self.store.save_json("sampling/pairs/phase_wrap_pairs.json", {"pairs": phase_pairs})
        self.store.save_json("sampling/pairs/time_adj_pairs.json", {"pairs": time_pairs})
        deeponet_cfg = self.cfg.get("sampling", {}).get("deeponet", {})
        deeponet_index_meta: dict[str, Any] = {}
        deeponet_index_hash = ""
        deeponet_task_hashes: dict[str, str] = {}

        def _build_task_payload(task_name: str, task_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
            h, w = geom.mask_plasma.shape
            flatten_order = str(task_cfg.get("flatten_order", "C"))
            if flatten_order not in {"C", "F"}:
                raise ValueError(f"sampling.deeponet.{task_name}.flatten_order must be 'C' or 'F'")
            n_sensors = int(task_cfg.get("n_sensors", 32))
            n_queries = int(task_cfg.get("n_queries", 64))
            seed = int(task_cfg.get("seed", 0))
            strategy = str(task_cfg.get("strategy", "uniform_fixed"))
            idx = build_deeponet_indices(
                n_points=int(h * w),
                n_sensors=n_sensors,
                n_queries=n_queries,
                seed=seed,
            )
            flat_coords = build_flattened_coords(geom.coord_grid, order=flatten_order)
            sensor_coords = flat_coords[idx["sensor_indices"]]
            query_coords = flat_coords[idx["query_indices"]]
            payload = {
                "sensor_indices": [int(v) for v in idx["sensor_indices"].tolist()],
                "query_indices": [int(v) for v in idx["query_indices"].tolist()],
                "seed": seed,
                "strategy": strategy,
            }
            if task_name == "boundary_operator":
                payload["primary_qoi_key"] = str(task_cfg.get("primary_qoi_key", "Gamma_i"))
            elif "primary_qoi_key" in task_cfg:
                payload["primary_qoi_key"] = str(task_cfg["primary_qoi_key"])
            meta = {
                "flatten_order": flatten_order,
                "grid_shape": [int(h), int(w)],
                "n_points": int(h * w),
                "coord_system": "cartesian",
                "task": task_name,
                "sampling_spec": {
                    "n_sensors": n_sensors,
                    "n_queries": n_queries,
                    "seed": seed,
                    "strategy": strategy,
                },
            }
            if task_name == "boundary_operator":
                meta["boundary_sampling_spec"] = {
                    "n_sensors": n_sensors,
                    "n_queries": n_queries,
                    "seed": seed,
                    "strategy": strategy,
                    "primary_qoi_key": str(payload["primary_qoi_key"]),
                }
            return payload, meta, sensor_coords, query_coords

        task_cfgs: dict[str, dict[str, Any]] = {}
        if bool(deeponet_cfg.get("enabled", False)):
            task_cfgs["default"] = dict(deeponet_cfg)
        for task_name, task_payload in dict(deeponet_cfg.get("tasks", {})).items():
            task_cfgs[str(task_name)] = dict(task_payload)

        if task_cfgs:
            deeponet_root = self.output_dir / "sampling" / "deeponet"
            deeponet_root.mkdir(parents=True, exist_ok=True)
            primary_task = "default" if "default" in task_cfgs else sorted(task_cfgs.keys())[0]
            for task_name, task_payload in sorted(task_cfgs.items()):
                payload, meta, sensor_coords, query_coords = _build_task_payload(task_name, task_payload)
                task_hash = hash_json(
                    {
                        "task": task_name,
                        "index": payload,
                        "meta": meta,
                        "primary_qoi_key": payload.get("primary_qoi_key", ""),
                    }
                )
                deeponet_task_hashes[task_name] = task_hash
                task_dir = deeponet_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                self.store.save_json(f"sampling/deeponet/{task_name}/sensor_query_index.json", payload)
                self.store.save_json(f"sampling/deeponet/{task_name}/index_meta.json", meta)
                np.save(task_dir / "sensor_coords.npy", sensor_coords.astype(np.float32))
                np.save(task_dir / "query_coords.npy", query_coords.astype(np.float32))
                if task_name == primary_task:
                    deeponet_index_meta = meta
                    deeponet_index_hash = task_hash

            self.store.save_json("sampling/deeponet/task_hashes.json", deeponet_task_hashes)

        cond_train = cond_matrix[train_indices, : len(cond_schema.order)]
        cond_stats = {
            key: {
                "mean": float(np.mean(cond_train[:, i])),
                "std": float(np.std(cond_train[:, i])),
                "min": float(np.min(cond_train[:, i])),
                "max": float(np.max(cond_train[:, i])),
            }
            for i, key in enumerate(cond_schema.order)
        }
        self.store.save_json("stats/cond_stats.json", cond_stats)
        y_stats = {}
        for var_idx, var in enumerate(y_vars):
            # Density magnitudes are O(1e18); float32 variance reduction can overflow even
            # though every source value is finite.  Statistics are reporting artifacts, so
            # reduce in float64 and preserve the exact training split.
            vals = np.asarray(y_by_var[var][train_indices], dtype=np.float64).reshape(-1)
            y_stats[var] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        self.store.save_json("stats/y_stats.json", y_stats)
        split_hash = hash_json(
            {
                "split_random": split,
                "split_extrap": split_extrap,
                "split_extrapolation_cfg": {
                    "key": extrap_key,
                    "direction": extrap_direction,
                    "holdout_ratio": float(extrap_cfg.get("holdout_ratio", 0.2)),
                    "val_ratio": float(extrap_cfg.get("val_ratio", 0.2)),
                },
                "split_structure_holdout": split_structure_holdout,
                "split_structure_holdout_meta": split_structure_holdout_meta,
                "scaler_fit_split": scaler_fit_split,
                "protocol_scaler_fit_splits": sorted(protocol_transforms),
            }
        )
        sampling_hash = hash_json(
            {
                "point_pools_shape": {k: int(v.shape[0]) for k, v in pools.items()},
                "patches": patches.tolist(),
                "phase_pairs": phase_pairs,
                "time_pairs": time_pairs,
                "deeponet_index_hash": deeponet_index_hash,
                "deeponet_index_meta": deeponet_index_meta,
                "deeponet_task_hashes": deeponet_task_hashes,
            }
        )
        self.store.save_json(
            "validation/repro_hashes.json",
            {
                "split_hash": split_hash,
                "sampling_hash": sampling_hash,
                "deeponet_index_hash": deeponet_index_hash,
                "deeponet_task_hashes": deeponet_task_hashes,
            },
        )
        outside = geom.mask_plasma <= 0.5
        if np.any(outside):
            negative_ratio = float(np.mean((distance_signed[outside] < 0.0).astype(np.float32)))
        else:
            negative_ratio = 1.0
        distance_contract_status = "ok"
        if negative_ratio <= 0.0:
            distance_contract_status = "invalid_no_negative_outside"
            if distance_contract_mode == "error":
                raise ValueError("distance contract violation: outside signed-distance has no negative values")
        coord_source_applied = str(getattr(geom, "coord_source", "unknown"))
        coord_source_requested = coord_grid_source
        coord_contract_status = "ok"
        if coord_source_applied != coord_source_requested:
            coord_contract_status = "mismatch"
            if coord_contract_mode == "error":
                raise ValueError(
                    "coord-grid contract violation: requested source "
                    f"{coord_source_requested} but applied {coord_source_applied}"
                )

        descriptor_artifact_meta = {
            "structure_descriptor_pack_path": "",
            "structure_descriptor_pack_meta_path": "",
            "structure_descriptor_dim": 0,
            "structure_descriptor_n_parts": 0,
            "structure_descriptor_n_part_slots": 0,
            "structure_descriptor_case_specific": False,
            "structure_descriptor_case_count": 0,
        }
        if self.input_mode_effective == TABLE_PLUS_STRUCTURE and descriptor_profile != "none":
            descriptor_artifact_meta = self._build_descriptor_artifact(
                geom=geom,
                descriptor_profile=descriptor_profile,
                cases=cases,
            )
        latent_artifact_meta = self._resolve_latent_artifact(
            latent_profile=latent_profile,
            geometry_root=geometry_root,
        ) if self.input_mode_effective == TABLE_PLUS_STRUCTURE else {
            "latent_feature_pack_path": "",
            "latent_feature_pack_meta_path": "",
            "latent_feature_dim": 0,
            DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY: False,
        }
        preprocess_artifacts = PreprocessArtifacts(
            output_layout=output_layout_payload,
            target_role_schema=target_role_schema_payload,
            channel_map=channel_map_payload,
            coord_feature_pack_meta=coord_feature_pack_meta_payload,
            static_spatial_feature_pack_meta=case_spatial_meta if case_spatial_feature_enabled else {},
            case_structure_feature_pack_meta=case_spatial_meta if case_spatial_feature_enabled else {},
            structure_descriptor_pack_meta=descriptor_artifact_meta if descriptor_profile != "none" else {},
            latent_feature_pack_meta=latent_artifact_meta if latent_profile != "none" else {},
        )
        runtime_schema_hashes = self._save_runtime_schema_hashes(
            artifacts=preprocess_artifacts,
        )
        self.runtime_input_mode_meta = attach_runtime_schema_hashes(
            self.runtime_input_mode_meta,
            schema_hashes=runtime_schema_hashes,
        )

        self.report_builder.save(
            cases=cases,
            coord_source_applied=coord_source_applied,
            coord_source_requested=coord_source_requested,
            coord_contract_status=coord_contract_status,
            negative_ratio=negative_ratio,
            distance_contract_status=distance_contract_status,
            coord_rows=coord_rows,
            coord_rows_z=coord_rows_z,
            coord_rows_mm=coord_rows_mm,
            scaler_fit_split=scaler_fit_split,
            protocol_scaler_fit_splits=sorted(protocol_transforms),
            structure_holdout_meta=split_structure_holdout_meta,
            train_indices=train_indices,
            target_transforms_cfg=target_transforms_cfg,
            y_vars=y_vars,
            coord_features_enabled=coord_features_enabled,
            coord_feature_channels=coord_feature_channels,
            coord_feature_rel_path=coord_feature_rel_path,
            coord_features_scaling_enabled=coord_features_scaling_enabled,
            coord_features_scaling_mode=coord_features_scaling_mode,
            case_spatial_feature_enabled=case_spatial_feature_enabled,
            case_spatial_feature_meta_rel=case_spatial_feature_meta_rel,
            case_spatial_feature_shape=case_spatial_feature_shape,
            static_spatial_feature_rel_path=static_spatial_feature_rel_path,
            static_spatial_feature_meta_rel=static_spatial_feature_meta_rel,
            static_spatial_feature_shape=static_spatial_feature_shape,
            case_structure_feature_rel_path=case_structure_feature_rel_path,
            case_structure_feature_meta_rel=case_structure_feature_meta_rel,
            case_structure_feature_shape=case_structure_feature_shape,
            distance_stats_enabled=distance_stats_enabled,
            signed_tanh_tau_auto=signed_tanh_tau_auto,
            proximity_tau_auto=proximity_tau_auto,
            signed_q_raw=signed_q_raw,
            proximity_q_raw=proximity_q_raw,
            descriptor_artifact_meta=descriptor_artifact_meta,
            latent_artifact_meta=latent_artifact_meta,
            runtime_input_mode_meta=self.runtime_input_mode_meta,
        )

        return PreprocessOutput(
            split=split,
            cond_stats=cond_stats,
            hashes={
                "split_hash": split_hash,
                "sampling_hash": sampling_hash,
                "deeponet_index_hash": deeponet_index_hash,
                **runtime_schema_hashes,
            },
        )
