"""Shared spatial region/boundary-type mask builders."""

from __future__ import annotations

from typing import Any

import numpy as np

VALID_TARGET_REGIONS = frozenset({"plasma_only", "all_domain"})


def normalize_target_region_by_var(
    raw: Any,
    *,
    target_vars: list[str] | tuple[str, ...] | set[str] | None = None,
    key_name: str = "target_region_by_var",
    reject_unknown: bool = False,
) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{key_name} must be a mapping")
    allowed = None if target_vars is None else {str(v) for v in target_vars}
    out: dict[str, str] = {}
    unknown: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if allowed is not None and name not in allowed:
            unknown.append(name)
            continue
        region = str(value).strip().lower()
        if region not in VALID_TARGET_REGIONS:
            raise ValueError(f"{key_name} values must be one of: plasma_only, all_domain")
        out[name] = region
    if reject_unknown and unknown:
        raise ValueError(f"{key_name} contains unknown vars: {sorted(unknown)}")
    return out


def target_region_for_var(
    region_by_var: dict[str, str] | None,
    var_name: str,
    *,
    default: str = "plasma_only",
) -> str:
    region = str((region_by_var or {}).get(str(var_name), default)).strip().lower()
    if region == "" and str(default).strip() == "":
        return ""
    if region not in VALID_TARGET_REGIONS:
        raise ValueError("target_region_by_var values must be one of: plasma_only, all_domain")
    return region


def to_bhw(arr: Any, *, key: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 2:
        out = out[None, ...]
    if out.ndim == 4 and int(out.shape[1]) == 1:
        out = out[:, 0]
    if out.ndim != 3:
        raise ValueError(f"{key} must be [H,W], [B,H,W], or [B,1,H,W], got {tuple(out.shape)}")
    return out.astype(np.float32)


def align_bhw_batch(arr: np.ndarray, *, batch_size: int, key: str) -> np.ndarray:
    out = to_bhw(arr, key=key)
    if int(out.shape[0]) == int(batch_size):
        return out
    if int(out.shape[0]) == 1 and int(batch_size) > 1:
        return np.repeat(out, int(batch_size), axis=0).astype(np.float32)
    raise ValueError(f"{key} batch mismatch: expected {batch_size}, got {int(out.shape[0])}")


def build_region_masks(
    *,
    mask_plasma: Any | None,
    distance_any: Any | None,
    distance_signed: Any | None,
    mode: str = "fixed_px",
    boundary_in_px: float = 2.0,
    mid_plasma_px: float = 10.0,
    deep_plasma_px: float = 10.0,
    boundary_q: float = 0.15,
    deep_q: float = 0.70,
) -> dict[str, np.ndarray]:
    if distance_signed is None:
        if distance_any is None or mask_plasma is None:
            raise ValueError("build_region_masks requires distance_signed or (distance_any + mask_plasma)")
        m = to_bhw(mask_plasma, key="mask_plasma")
        d_any = align_bhw_batch(distance_any, batch_size=int(m.shape[0]), key="distance_any")
        d_signed = np.where(m > 0.5, d_any, -d_any).astype(np.float32)
    else:
        d_signed = to_bhw(distance_signed, key="distance_signed")
        if mask_plasma is None:
            m = (d_signed >= 0.0).astype(np.float32)
        else:
            m = align_bhw_batch(mask_plasma, batch_size=int(d_signed.shape[0]), key="mask_plasma")

    mode_eff = str(mode or "fixed_px").strip().lower()
    if mode_eff not in {"fixed_px", "signed_quantile"}:
        raise ValueError("build_region_masks.mode must be one of: fixed_px, signed_quantile")
    if mode_eff == "fixed_px":
        boundary_thr = np.full((int(d_signed.shape[0]),), float(boundary_in_px), dtype=np.float32)
        deep_thr = np.full((int(d_signed.shape[0]),), float(deep_plasma_px), dtype=np.float32)
    else:
        bq = float(boundary_q)
        dq = float(deep_q)
        if not (0.0 < bq < 1.0) or not (0.0 < dq < 1.0):
            raise ValueError("build_region_masks boundary_q/deep_q must be within (0,1)")
        if bq >= dq:
            raise ValueError("build_region_masks requires boundary_q < deep_q")
        batch = int(d_signed.shape[0])
        boundary_thr = np.zeros((batch,), dtype=np.float32)
        deep_thr = np.zeros((batch,), dtype=np.float32)
        for bi in range(batch):
            plasma_vals = np.asarray(d_signed[bi][m[bi] > 0.5], dtype=np.float32)
            plasma_pos = plasma_vals[plasma_vals >= 0.0]
            if plasma_pos.size == 0:
                boundary_thr[bi] = 0.0
                deep_thr[bi] = 0.0
                continue
            b_val = float(np.quantile(plasma_pos, bq))
            d_val = float(np.quantile(plasma_pos, dq))
            d_val = max(d_val, b_val)
            boundary_thr[bi] = max(b_val, 0.0)
            deep_thr[bi] = max(d_val, boundary_thr[bi])

    boundary_cut = boundary_thr[:, None, None]
    deep_cut = deep_thr[:, None, None]
    mid_cut = np.where(mode_eff == "signed_quantile", deep_cut, float(mid_plasma_px))

    plasma = m > 0.5
    boundary_in = np.logical_and(plasma, np.logical_and(d_signed >= 0.0, d_signed <= boundary_cut))
    plasma_mid = np.logical_and(plasma, np.logical_and(d_signed > boundary_cut, d_signed <= mid_cut))
    plasma_deep = np.logical_and(plasma, d_signed > deep_cut)
    chamber_near = np.logical_and(d_signed < 0.0, d_signed >= -boundary_cut)
    chamber_far = d_signed < -np.maximum(deep_cut, 10.0)
    return {
        "boundary_in": boundary_in,
        "plasma_mid": plasma_mid,
        "plasma_deep": plasma_deep,
        "chamber_near": chamber_near,
        "chamber_far": chamber_far,
        "all_plasma": plasma,
        "boundary_threshold": boundary_thr.astype(np.float32),
        "deep_threshold": deep_thr.astype(np.float32),
    }


def build_boundary_type_masks(
    *,
    mask_plasma: Any,
    distance_any: Any,
    band_px: float = 2.0,
    bc_dir_mask: Any | None = None,
    wafer_mask: Any | None = None,
) -> dict[str, np.ndarray]:
    m = to_bhw(mask_plasma, key="mask_plasma")
    d_any = align_bhw_batch(distance_any, batch_size=int(m.shape[0]), key="distance_any")
    boundary_in = np.logical_and(m > 0.5, d_any <= float(band_px))

    if bc_dir_mask is None:
        bc = np.zeros_like(boundary_in, dtype=bool)
    else:
        bc_arr = align_bhw_batch(bc_dir_mask, batch_size=int(m.shape[0]), key="bc_dir_mask")
        bc = np.logical_and(boundary_in, bc_arr > 0.5)

    if wafer_mask is None:
        wafer = np.zeros_like(boundary_in, dtype=bool)
    else:
        wafer_arr = align_bhw_batch(wafer_mask, batch_size=int(m.shape[0]), key="wafer_mask")
        wafer = np.logical_and(boundary_in, wafer_arr > 0.5)

    interface = np.logical_and(boundary_in, np.logical_not(np.logical_or(bc, wafer)))
    return {
        "boundary_in": boundary_in,
        "interface": interface,
        "bc_dir": bc,
        "wafer": wafer,
    }
