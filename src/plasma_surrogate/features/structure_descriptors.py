"""Deterministic structure descriptor builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext, build_signed_distance_fields


STRUCT_DESC_V1 = "struct_desc_v1"
STRUCT_DESC_V2 = "struct_desc_v2"
STRUCT_DESC_LITE_V1 = "struct_desc_lite_v1"
STRUCTURE_DESCRIPTOR_PROFILES: tuple[str, ...] = (STRUCT_DESC_V1, STRUCT_DESC_V2, STRUCT_DESC_LITE_V1)

_GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "n_parts",
    "solid_area_frac",
    "plasma_area_frac",
    "min_part_gap",
    "mean_part_gap",
    "mean_distance_to_plasma_boundary",
)
_PART_FEATURE_NAMES: tuple[str, ...] = (
    "area_frac",
    "centroid_x",
    "centroid_y",
    "bbox_w",
    "bbox_h",
    "principal_angle",
    "perimeter",
    "min_gap_to_plasma",
    "min_gap_to_other_parts",
)


@dataclass(frozen=True)
class StructureDescriptorPack:
    profile: str
    vector: np.ndarray
    feature_names: tuple[str, ...]
    n_parts: int

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "vector": np.asarray(self.vector, dtype=np.float32).reshape(-1),
            "feature_names": np.asarray(list(self.feature_names), dtype=object),
            "profile": np.asarray([self.profile], dtype=object),
            "n_parts": np.asarray([int(self.n_parts)], dtype=np.int64),
        }

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "profile": str(self.profile),
            "descriptor_dim": int(self.vector.shape[0]),
            "n_parts": int(self.n_parts),
            "feature_names": list(self.feature_names),
        }


def _normalize_descriptor_profile_name(profile: Any) -> str:
    name = str(profile if profile is not None else "").strip().lower()
    if not name:
        raise ValueError("descriptor profile must be a non-empty string")
    if name not in set(STRUCTURE_DESCRIPTOR_PROFILES):
        raise ValueError(
            "unsupported descriptor profile: "
            f"{name!r}; supported={list(STRUCTURE_DESCRIPTOR_PROFILES)}"
        )
    return name


def _resolve_part_mask_stack(geom_ctx: GeometryContext, *, profile: str) -> np.ndarray:
    regions = dict(getattr(geom_ctx, "regions", {}) or {})
    if "part_mask_stack" not in regions:
        raise ValueError(
            f"{profile} requires GeometryContext.regions.part_mask_stack; "
            "descriptor lane is fail-fast when part masks are unavailable"
        )
    stack = np.asarray(regions["part_mask_stack"], dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"part_mask_stack must be [P,H,W], got shape={stack.shape}")
    if stack.shape[0] < 1:
        raise ValueError(f"part_mask_stack must contain at least one part for {profile}")
    if not np.all(np.isfinite(stack)):
        raise ValueError("part_mask_stack must contain only finite values")
    h, w = np.asarray(geom_ctx.mask_plasma, dtype=np.float32).shape
    if tuple(stack.shape[1:]) != (int(h), int(w)):
        raise ValueError(
            "part_mask_stack spatial shape mismatch: "
            f"expected={(int(h), int(w))}, got={tuple(int(v) for v in stack.shape[1:])}"
        )
    return (stack > 0.5).astype(np.float32)


def _resolve_union_solid(mask_stack: np.ndarray, geom_ctx: GeometryContext) -> np.ndarray:
    regions = dict(getattr(geom_ctx, "regions", {}) or {})
    if "solid_union_mask" in regions:
        union = (np.asarray(regions["solid_union_mask"], dtype=np.float32) > 0.5).astype(np.float32)
    else:
        union = np.maximum.reduce(mask_stack, axis=0).astype(np.float32)
    h, w = np.asarray(geom_ctx.mask_plasma, dtype=np.float32).shape
    if union.shape != (h, w):
        raise ValueError(
            "solid_union_mask shape mismatch: "
            f"expected={(int(h), int(w))}, got={tuple(int(v) for v in union.shape)}"
        )
    return union


def _resolve_coord_maps(geom_ctx: GeometryContext) -> tuple[np.ndarray, np.ndarray]:
    coord = np.asarray(geom_ctx.coord_grid, dtype=np.float32)
    if coord.ndim != 3:
        raise ValueError(f"coord_grid must be rank-3, got shape={coord.shape}")
    if coord.shape[0] == 2:
        return coord[0], coord[1]
    if coord.shape[-1] == 2:
        return coord[..., 0], coord[..., 1]
    raise ValueError(f"coord_grid must be (2,H,W) or (H,W,2), got shape={coord.shape}")


def _principal_angle(mask: np.ndarray, x_map: np.ndarray, y_map: np.ndarray) -> float:
    sel = mask > 0.5
    if int(np.sum(sel)) < 2:
        return 0.0
    x = np.asarray(x_map[sel], dtype=np.float32)
    y = np.asarray(y_map[sel], dtype=np.float32)
    x = x - np.mean(x, dtype=np.float32)
    y = y - np.mean(y, dtype=np.float32)
    cov = np.asarray(
        [
            [np.mean(x * x, dtype=np.float32), np.mean(x * y, dtype=np.float32)],
            [np.mean(x * y, dtype=np.float32), np.mean(y * y, dtype=np.float32)],
        ],
        dtype=np.float32,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=np.float32).reshape(2)
    return float(np.arctan2(float(major[1]), float(major[0])))


def _perimeter_proxy(mask: np.ndarray) -> float:
    m = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    h, w = m.shape
    boundary = np.zeros_like(m, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if m[i, j] <= 0.5:
                continue
            n0 = m[i - 1, j] if i > 0 else 0.0
            n1 = m[i + 1, j] if i < h - 1 else 0.0
            n2 = m[i, j - 1] if j > 0 else 0.0
            n3 = m[i, j + 1] if j < w - 1 else 0.0
            if (n0 <= 0.5) or (n1 <= 0.5) or (n2 <= 0.5) or (n3 <= 0.5):
                boundary[i, j] = 1.0
    return float(np.sum(boundary, dtype=np.float32))


def _min_gap_to_other_parts(mask_i: np.ndarray, union_others: np.ndarray) -> float:
    if float(np.sum(union_others, dtype=np.float32)) <= 0.0:
        return 0.0
    signed_others = build_signed_distance_fields(union_others.astype(np.float32))
    vals = np.abs(np.asarray(signed_others, dtype=np.float32)[mask_i > 0.5])
    if vals.size == 0:
        return 0.0
    return float(np.min(vals))


def _build_struct_desc(
    geom_ctx: GeometryContext,
    *,
    profile: str,
    normalize_lengths: bool,
) -> StructureDescriptorPack:
    mask_plasma = (np.asarray(geom_ctx.mask_plasma, dtype=np.float32) > 0.5).astype(np.float32)
    if mask_plasma.ndim != 2:
        raise ValueError(f"mask_plasma must be [H,W], got shape={mask_plasma.shape}")
    h, w = mask_plasma.shape
    total_area = float(max(int(h * w), 1))
    length_scale = float(max(int(h), int(w), 1))
    perimeter_scale = float(max(2 * (int(h) + int(w)), 1))
    part_mask_stack = _resolve_part_mask_stack(geom_ctx, profile=profile)
    n_parts = int(part_mask_stack.shape[0])
    union_solid = _resolve_union_solid(part_mask_stack, geom_ctx)
    x_map, y_map = _resolve_coord_maps(geom_ctx)
    distance_any = np.asarray(geom_ctx.distance_any, dtype=np.float32)
    if distance_any.shape != (h, w):
        raise ValueError(
            "distance_any shape mismatch: "
            f"expected={(int(h), int(w))}, got={tuple(int(v) for v in distance_any.shape)}"
        )
    distance_signed = np.asarray(geom_ctx.distance_signed, dtype=np.float32)
    if distance_signed.shape != (h, w):
        raise ValueError(
            "distance_signed shape mismatch: "
            f"expected={(int(h), int(w))}, got={tuple(int(v) for v in distance_signed.shape)}"
        )

    solid_area_frac = float(np.sum(union_solid > 0.5, dtype=np.float32) / total_area)
    plasma_area_frac = float(np.sum(mask_plasma > 0.5, dtype=np.float32) / total_area)
    solid_sel = union_solid > 0.5
    if np.any(solid_sel):
        mean_distance_to_plasma_boundary = float(np.mean(distance_any[solid_sel], dtype=np.float32))
    else:
        mean_distance_to_plasma_boundary = 0.0
    if normalize_lengths:
        mean_distance_to_plasma_boundary = float(mean_distance_to_plasma_boundary / length_scale)

    lite_profile = profile == STRUCT_DESC_LITE_V1
    per_part_rows: list[list[float]] = []
    per_part_gap_values: list[float] = []
    for i in range(n_parts):
        mask_i = (part_mask_stack[i] > 0.5).astype(np.float32)
        union_others = np.maximum.reduce(np.delete(part_mask_stack, i, axis=0), axis=0) if n_parts > 1 else np.zeros_like(mask_i)
        min_gap_to_other = _min_gap_to_other_parts(mask_i, union_others)
        if normalize_lengths:
            min_gap_to_other = float(min_gap_to_other / length_scale)
        per_part_gap_values.append(float(min_gap_to_other))
        if lite_profile:
            continue

        area = float(np.sum(mask_i, dtype=np.float32))
        area_frac = float(area / total_area)
        sel = mask_i > 0.5
        if np.any(sel):
            centroid_x = float(np.mean(x_map[sel], dtype=np.float32))
            centroid_y = float(np.mean(y_map[sel], dtype=np.float32))
            ys, xs = np.nonzero(sel)
            bbox_w = float((int(np.max(xs)) - int(np.min(xs)) + 1) / float(max(w, 1)))
            bbox_h = float((int(np.max(ys)) - int(np.min(ys)) + 1) / float(max(h, 1)))
            min_gap_to_plasma = float(np.min(np.abs(distance_signed[sel])))
        else:
            centroid_x = 0.0
            centroid_y = 0.0
            bbox_w = 0.0
            bbox_h = 0.0
            min_gap_to_plasma = 0.0
        principal_angle = _principal_angle(mask_i, x_map, y_map)
        perimeter = _perimeter_proxy(mask_i)
        if normalize_lengths:
            principal_angle = float(principal_angle / np.pi)
            perimeter = float(perimeter / perimeter_scale)
            min_gap_to_plasma = float(min_gap_to_plasma / length_scale)
        per_part_rows.append(
            [
                float(area_frac),
                float(centroid_x),
                float(centroid_y),
                float(bbox_w),
                float(bbox_h),
                float(principal_angle),
                float(perimeter),
                float(min_gap_to_plasma),
                float(min_gap_to_other),
            ]
        )

    min_part_gap = float(np.min(np.asarray(per_part_gap_values, dtype=np.float32))) if per_part_gap_values else 0.0
    mean_part_gap = float(np.mean(np.asarray(per_part_gap_values, dtype=np.float32))) if per_part_gap_values else 0.0
    global_vec = np.asarray(
        [
            float(n_parts),
            float(solid_area_frac),
            float(plasma_area_frac),
            float(min_part_gap),
            float(mean_part_gap),
            float(mean_distance_to_plasma_boundary),
        ],
        dtype=np.float32,
    )
    if lite_profile:
        vec = global_vec.astype(np.float32)
    else:
        part_vec = np.asarray(per_part_rows, dtype=np.float32).reshape(-1) if per_part_rows else np.zeros((0,), dtype=np.float32)
        vec = np.concatenate([global_vec, part_vec], axis=0).astype(np.float32)
    if vec.ndim != 1 or int(vec.shape[0]) < len(_GLOBAL_FEATURE_NAMES):
        raise RuntimeError(f"{profile} produced invalid descriptor shape")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{profile} produced non-finite descriptor values")
    feature_names = list(_GLOBAL_FEATURE_NAMES)
    if not lite_profile:
        for i in range(n_parts):
            prefix = f"part_{i:03d}"
            feature_names.extend(f"{prefix}.{name}" for name in _PART_FEATURE_NAMES)
    if len(feature_names) != int(vec.shape[0]):
        raise RuntimeError(
            f"{profile} feature-name length mismatch: "
            f"names={len(feature_names)}, vector_dim={int(vec.shape[0])}"
        )
    return StructureDescriptorPack(
        profile=profile,
        vector=vec,
        feature_names=tuple(feature_names),
        n_parts=int(n_parts),
    )


def build_struct_desc_v1(geom_ctx: GeometryContext) -> StructureDescriptorPack:
    return _build_struct_desc(geom_ctx, profile=STRUCT_DESC_V1, normalize_lengths=False)


def build_struct_desc_v2(geom_ctx: GeometryContext) -> StructureDescriptorPack:
    return _build_struct_desc(geom_ctx, profile=STRUCT_DESC_V2, normalize_lengths=True)


def build_struct_desc_lite_v1(geom_ctx: GeometryContext) -> StructureDescriptorPack:
    return _build_struct_desc(geom_ctx, profile=STRUCT_DESC_LITE_V1, normalize_lengths=True)


def build_structure_descriptor(profile: Any, geom_ctx: GeometryContext) -> StructureDescriptorPack:
    profile_name = _normalize_descriptor_profile_name(profile)
    if profile_name == STRUCT_DESC_V1:
        return build_struct_desc_v1(geom_ctx)
    if profile_name == STRUCT_DESC_V2:
        return build_struct_desc_v2(geom_ctx)
    if profile_name == STRUCT_DESC_LITE_V1:
        return build_struct_desc_lite_v1(geom_ctx)
    raise ValueError(f"unsupported descriptor profile: {profile_name!r}")


__all__ = [
    "STRUCT_DESC_V1",
    "STRUCT_DESC_V2",
    "STRUCT_DESC_LITE_V1",
    "STRUCTURE_DESCRIPTOR_PROFILES",
    "StructureDescriptorPack",
    "build_struct_desc_lite_v1",
    "build_struct_desc_v1",
    "build_struct_desc_v2",
    "build_structure_descriptor",
]
