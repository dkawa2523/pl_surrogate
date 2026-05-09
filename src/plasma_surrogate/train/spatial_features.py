"""Spatial feature helpers shared by grid and DeepONet training paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.features.structure_feature_registry import validate_coord_feature_channels
from plasma_surrogate.preprocessing.scalers import ScalerFactory


def coord_xy_rows_from_geom(geom_ctx: Any, *, h: int, w: int) -> tuple[np.ndarray, str]:
    if geom_ctx is None:
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1), "normalized"
    raw_obj = getattr(geom_ctx, "coord_grid", None)
    if raw_obj is None:
        raise ValueError("coord_source=geom_ctx requires geom_ctx.coord_grid")
    raw_grid = np.asarray(raw_obj, dtype=np.float32)
    if raw_grid.shape == (2, h, w):
        return np.stack([raw_grid[0].reshape(-1), raw_grid[1].reshape(-1)], axis=1).astype(np.float32), "geom_ctx"
    if raw_grid.shape == (h, w, 2):
        return raw_grid.reshape(-1, 2).astype(np.float32), "geom_ctx"
    raise ValueError(f"geom_ctx.coord_grid shape mismatch: got {raw_grid.shape}, expected (2,{h},{w})")


def resolve_coord_feature_channels(raw: Any) -> list[str]:
    if raw is None:
        return ["x", "y", "distance_signed", "distance_any", "mask_plasma"]
    channels = [str(v) for v in list(raw)] if isinstance(raw, list) else raw
    try:
        return list(validate_coord_feature_channels(channels))
    except ValueError as exc:
        raise ValueError(f"train.input_features.features is invalid: {exc}") from exc


def derive_geom_feature_maps(
    *,
    coord_xy: np.ndarray,
    distance_signed: np.ndarray,
    distance_any: np.ndarray,
    mask_plasma: np.ndarray,
    h: int,
    w: int,
) -> dict[str, np.ndarray]:
    signed_2d = np.asarray(distance_signed, dtype=np.float32).reshape(h, w)
    gy, gx = np.gradient(signed_2d, edge_order=1)
    gnorm = np.sqrt(gx**2 + gy**2).astype(np.float32)
    gnorm = np.where(gnorm < 1e-6, 1e-6, gnorm).astype(np.float32)
    normal_x = (gx / gnorm).reshape(-1).astype(np.float32)
    normal_y = (gy / gnorm).reshape(-1).astype(np.float32)
    dnx_dy, dnx_dx = np.gradient(gx / gnorm, edge_order=1)
    dny_dy, dny_dx = np.gradient(gy / gnorm, edge_order=1)
    curvature_proxy = (dnx_dx + dny_dy).reshape(-1).astype(np.float32)
    return {
        "x": coord_xy[:, 0],
        "y": coord_xy[:, 1],
        "distance_signed": np.asarray(distance_signed, dtype=np.float32).reshape(-1),
        "distance_any": np.asarray(distance_any, dtype=np.float32).reshape(-1),
        "mask_plasma": np.asarray(mask_plasma, dtype=np.float32).reshape(-1),
        "normal_x": normal_x,
        "normal_y": normal_y,
        "curvature_proxy": curvature_proxy,
    }


def apply_coord_feature_scaling(
    rows: np.ndarray,
    *,
    channels: list[str],
    coord_feature_scaler_artifact: dict[str, Any] | None,
) -> tuple[np.ndarray, str, bool]:
    raw = dict(coord_feature_scaler_artifact or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return np.asarray(rows, dtype=np.float32), "none", False
    ch_scalers = dict(raw.get("channels", {}))
    if len(ch_scalers) == 0:
        return np.asarray(rows, dtype=np.float32), "none", False
    out = np.asarray(rows, dtype=np.float32).copy()
    for i, name in enumerate(channels):
        payload = ch_scalers.get(name)
        if not isinstance(payload, dict) or len(payload) == 0:
            continue
        scaler = ScalerFactory.from_dict(payload)
        out[:, i : i + 1] = scaler.transform(out[:, i : i + 1]).astype(np.float32)
    return out.astype(np.float32), str(raw.get("mode", "custom")), True


def resolve_distance_transform_cfg(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    mode = str(cfg.get("mode", "raw")).strip().lower()
    if mode not in {"raw", "bounded", "bounded_auto"}:
        raise ValueError(
            "train.input_features.distance_transform.mode must be one of: raw, bounded, bounded_auto"
        )
    signed_tanh_tau = float(cfg.get("signed_tanh_tau", 8.0))
    proximity_tau = float(cfg.get("proximity_tau", 6.0))
    if signed_tanh_tau <= 0.0:
        raise ValueError("distance_transform.signed_tanh_tau must be > 0")
    if proximity_tau <= 0.0:
        raise ValueError("distance_transform.proximity_tau must be > 0")
    return {
        "mode": mode,
        "signed_tanh_tau": signed_tanh_tau,
        "proximity_tau": proximity_tau,
        "replace_distance_any": bool(cfg.get("replace_distance_any", True)),
    }


def resolve_distance_transform_effective(
    cfg: dict[str, Any],
    *,
    stats: dict[str, Any] | None,
    warnings_out: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    mode = str(cfg.get("mode", "raw")).strip().lower()
    out = dict(cfg)
    if mode != "bounded_auto":
        return out, "config"
    raw_stats = dict(stats or {})
    s_tau = float(raw_stats.get("signed_tanh_tau_auto", 0.0))
    p_tau = float(raw_stats.get("proximity_tau_auto", 0.0))
    if s_tau > 0.0 and p_tau > 0.0:
        out["signed_tanh_tau"] = s_tau
        out["proximity_tau"] = p_tau
        out["signed_quantile"] = float(raw_stats.get("signed_quantile", 0.75))
        out["proximity_quantile"] = float(raw_stats.get("proximity_quantile", 0.50))
        return out, "artifact"
    if warnings_out is not None:
        warnings_out.append(
            "bounded_auto fallback: preprocessing/scalers/distance_transform_stats.json is missing "
            "or invalid; using configured tau values"
        )
    return out, "fallback_default"


def apply_distance_transform(
    rows: np.ndarray,
    *,
    channels: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(rows, dtype=np.float32).copy()
    mode = str(cfg.get("mode", "raw")).strip().lower()
    effective = {
        "mode": mode,
        "signed_tanh_tau": float(cfg.get("signed_tanh_tau", 8.0)),
        "proximity_tau": float(cfg.get("proximity_tau", 6.0)),
        "replace_distance_any": bool(cfg.get("replace_distance_any", True)),
        "signed_quantile": float(cfg.get("signed_quantile", 0.75)),
        "proximity_quantile": float(cfg.get("proximity_quantile", 0.50)),
    }
    if mode == "raw":
        return out, effective

    signed_tau = max(float(effective["signed_tanh_tau"]), 1e-6)
    prox_tau = max(float(effective["proximity_tau"]), 1e-6)
    if "distance_signed" in channels:
        idx = channels.index("distance_signed")
        out[:, idx] = np.tanh(out[:, idx] / signed_tau).astype(np.float32)
    if "distance_any" in channels and bool(effective["replace_distance_any"]):
        idx = channels.index("distance_any")
        out[:, idx] = np.exp(-np.maximum(out[:, idx], 0.0) / prox_tau).astype(np.float32)
    return out.astype(np.float32), effective


def build_coord_feature_rows(
    *,
    channels: list[str],
    pack: dict[str, Any] | None,
    geom_ctx: Any,
    h: int,
    w: int,
) -> tuple[np.ndarray, str]:
    if pack:
        data = np.asarray(pack.get("data"), dtype=np.float32) if "data" in pack else None
        raw_channels = pack.get("channels")
        if data is not None and raw_channels is not None:
            pack_channels = [str(v) for v in np.asarray(raw_channels).reshape(-1).tolist()]
            if data.ndim == 3 and data.shape[1:] == (h, w):
                pack_map = {name: data[i] for i, name in enumerate(pack_channels) if i < data.shape[0]}
                if all(name in pack_map for name in channels):
                    stacked = np.stack([pack_map[name] for name in channels], axis=0).astype(np.float32)
                    return stacked.reshape(len(channels), -1).T.astype(np.float32), "preprocess_pack"
    coord_xy, _ = coord_xy_rows_from_geom(geom_ctx, h=h, w=w)
    if geom_ctx is None:
        distance_any = np.ones((h * w,), dtype=np.float32)
        distance_signed = np.ones((h * w,), dtype=np.float32)
        mask_plasma = np.ones((h * w,), dtype=np.float32)
    else:
        distance_any = np.asarray(getattr(geom_ctx, "distance_any"), dtype=np.float32).reshape(-1)
        raw_signed = getattr(geom_ctx, "distance_signed", None)
        if raw_signed is None:
            mask_tmp = np.asarray(getattr(geom_ctx, "mask_plasma"), dtype=np.float32).reshape(-1)
            distance_signed = np.where(mask_tmp > 0.5, distance_any, -distance_any).astype(np.float32)
        else:
            distance_signed = np.asarray(raw_signed, dtype=np.float32).reshape(-1)
        mask_plasma = np.asarray(getattr(geom_ctx, "mask_plasma"), dtype=np.float32).reshape(-1)
    mapping = derive_geom_feature_maps(
        coord_xy=coord_xy,
        distance_signed=distance_signed,
        distance_any=distance_any,
        mask_plasma=mask_plasma,
        h=h,
        w=w,
    )
    feats = np.stack([np.asarray(mapping[name], dtype=np.float32) for name in channels], axis=1).astype(np.float32)
    return feats, "runtime_geom"
