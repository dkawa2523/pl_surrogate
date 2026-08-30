"""Spatial feature helpers shared by preprocessing, training, and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import build_signed_distance_fields
from plasma_surrogate.features.structure_feature_registry import validate_coord_feature_channels
from plasma_surrogate.preprocessing.scalers import ScalerFactory


ICP_STRUCT_STATIC_CHANNELS: tuple[str, ...] = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
)
ICP_STRUCT_CASE_CHANNELS: tuple[str, ...] = ("mask_coil", "distance_coil", "coil_proximity")
ICP_PART_SDF_CHANNELS: tuple[str, ...] = tuple(f"sdf_coil_{i:02d}" for i in range(1, 7))
PART_SDF_SUMMARY_CHANNELS: tuple[str, ...] = (
    "part_sdf_nearest",
    "part_sdf_second",
    "part_gap_proxy",
    "solid_proximity",
)
PART_SOURCE_CHANNELS: tuple[str, ...] = ("part_sdf_union", "part_source_sum")
PART_SOURCE_MEAN_CHANNELS: tuple[str, ...] = ("part_sdf_union", "part_source_mean")
ICP_PART_SDF_LITE_CASE_CHANNELS: tuple[str, ...] = (*ICP_STRUCT_CASE_CHANNELS, *ICP_PART_SDF_CHANNELS)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    if float(np.sum(binary, dtype=np.float32)) <= 0.0:
        return np.full(binary.shape, float(max(binary.shape)), dtype=np.float32)
    signed = build_signed_distance_fields(binary)
    return np.where(binary > 0.5, 0.0, np.abs(signed)).astype(np.float32)


def sdf_from_part_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    if float(np.sum(binary, dtype=np.float32)) <= 0.0:
        return np.full(binary.shape, float(sum(binary.shape) + 1), dtype=np.float32)
    return (-build_signed_distance_fields(binary)).astype(np.float32)


def part_sdf_maps_from_stack(part_mask_stack: np.ndarray, *, slot_count: int = 6) -> dict[str, np.ndarray]:
    stack = np.asarray(part_mask_stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"part_mask_stack must be [P,H,W], got {stack.shape}")
    h, w = stack.shape[1:]
    fill = np.full((h, w), float(h + w + 1), dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for idx in range(int(slot_count)):
        name = f"sdf_coil_{idx + 1:02d}"
        if idx < int(stack.shape[0]):
            out[name] = sdf_from_part_mask(stack[idx]).astype(np.float32)
        else:
            out[name] = fill.copy()
    return out


def boundary_band_from_signed_distance(
    distance_signed: np.ndarray,
    *,
    tau: float = 2.0,
) -> np.ndarray:
    ds = np.asarray(distance_signed, dtype=np.float32)
    tau_eff = max(float(tau), 1.0e-6)
    out = np.exp(-np.abs(ds) / tau_eff).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def part_sdf_summary_maps_from_stack(
    part_mask_stack: np.ndarray,
    *,
    proximity_tau: float = 4.0,
) -> dict[str, np.ndarray]:
    stack = np.asarray(part_mask_stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"part_mask_stack must be [P,H,W], got {stack.shape}")
    h, w = stack.shape[1:]
    fill = np.full((h, w), float(h + w + 1), dtype=np.float32)
    active_stack = [
        stack[i]
        for i in range(int(stack.shape[0]))
        if np.any(np.asarray(stack[i], dtype=np.float32) > 0.5)
    ]
    if not active_stack:
        nearest = fill.copy()
        second = fill.copy()
    else:
        distances = [np.abs(sdf_from_part_mask(mask)).astype(np.float32) for mask in active_stack]
        sorted_dist = np.sort(np.stack(distances, axis=0), axis=0).astype(np.float32)
        nearest = sorted_dist[0].astype(np.float32)
        second = sorted_dist[1].astype(np.float32) if len(active_stack) >= 2 else fill.copy()
    gap_proxy = np.maximum(second - nearest, 0.0).astype(np.float32)
    tau_eff = max(float(proximity_tau), 1.0e-6)
    solid_proximity = np.exp(-np.maximum(nearest, 0.0) / tau_eff).astype(np.float32)
    second_proximity = np.exp(-np.maximum(second, 0.0) / tau_eff).astype(np.float32)
    part_competition = np.exp(-gap_proxy / tau_eff).astype(np.float32)
    return {
        "part_sdf_nearest": nearest.astype(np.float32),
        "part_sdf_second": second.astype(np.float32),
        "part_gap_proxy": gap_proxy.astype(np.float32),
        "solid_proximity": solid_proximity.astype(np.float32),
        "part_second_proximity": second_proximity.astype(np.float32),
        "part_competition": part_competition.astype(np.float32),
    }


def part_source_maps_from_stack(
    part_mask_stack: np.ndarray,
    *,
    source_tau: float = 4.0,
) -> dict[str, np.ndarray]:
    """Build order-invariant geometry and additive source maps for parts.

    ``part_sdf_union`` represents the generated solid geometry.  The source
    map sums equal-strength, smoothly decaying contributions from every active
    part, so adding or moving a part remains visible instead of being reduced
    to only the nearest-part distance.
    """

    stack = np.asarray(part_mask_stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"part_mask_stack must be [P,H,W], got {stack.shape}")
    tau = float(source_tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("source_tau must be finite and > 0")
    h, w = stack.shape[1:]
    active = [(stack[idx] > 0.5).astype(np.float32) for idx in range(stack.shape[0]) if np.any(stack[idx] > 0.5)]
    if not active:
        return {
            "part_sdf_union": np.full((h, w), float(h + w + 1), dtype=np.float32),
            "part_source_sum": np.zeros((h, w), dtype=np.float32),
            "part_source_mean": np.zeros((h, w), dtype=np.float32),
        }
    union = np.maximum.reduce(active).astype(np.float32)
    source = np.sum(
        np.stack([np.exp(-distance_to_mask(mask) / tau) for mask in active], axis=0),
        axis=0,
        dtype=np.float32,
    )
    return {
        "part_sdf_union": sdf_from_part_mask(union).astype(np.float32),
        "part_source_sum": np.asarray(source, dtype=np.float32),
        "part_source_mean": np.asarray(source / float(len(active)), dtype=np.float32),
    }


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
    part_mask_stack: np.ndarray | None = None,
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
    mapping = {
        "x": coord_xy[:, 0],
        "y": coord_xy[:, 1],
        "distance_signed": np.asarray(distance_signed, dtype=np.float32).reshape(-1),
        "distance_any": np.asarray(distance_any, dtype=np.float32).reshape(-1),
        "mask_plasma": np.asarray(mask_plasma, dtype=np.float32).reshape(-1),
        "normal_x": normal_x,
        "normal_y": normal_y,
        "curvature_proxy": curvature_proxy,
        "boundary_band": boundary_band_from_signed_distance(signed_2d).reshape(-1),
    }
    if part_mask_stack is not None:
        for name, arr in part_sdf_summary_maps_from_stack(part_mask_stack).items():
            mapping[name] = np.asarray(arr, dtype=np.float32).reshape(-1)
        for name, arr in part_source_maps_from_stack(part_mask_stack).items():
            mapping[name] = np.asarray(arr, dtype=np.float32).reshape(-1)
    return mapping


def apply_coord_feature_scaling(
    rows: np.ndarray,
    *,
    channels: list[str],
    coord_feature_scaler_artifact: dict[str, Any] | None,
    distance_transform_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str, bool]:
    raw = dict(coord_feature_scaler_artifact or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return np.asarray(rows, dtype=np.float32), "none", False
    ch_scalers = dict(raw.get("channels", {}))
    if len(ch_scalers) == 0:
        return np.asarray(rows, dtype=np.float32), "none", False
    if distance_transform_cfg is not None:
        validate_coord_feature_scaler_distance_transform(
            coord_feature_scaler_artifact=raw,
            distance_transform_cfg=distance_transform_cfg,
        )
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


def resolve_shared_distance_transform_cfg(
    *,
    train_cfg: dict[str, Any] | None,
    model_names: list[str] | tuple[str, ...],
    scaling_enabled: bool = True,
) -> dict[str, Any]:
    """Resolve the one distance-transform space shared by preprocessing scalers.

    Models without an ``input_features`` contract do not consume the coordinate
    feature scaler and are ignored.  A single scaler artifact cannot safely be
    shared by active models that request different distance transforms, so that
    configuration fails before preprocessing starts.
    """

    if not bool(scaling_enabled):
        return resolve_distance_transform_cfg(None)

    train = dict(train_cfg or {})
    resolved_by_model: list[tuple[str, dict[str, Any]]] = []
    for raw_name in model_names:
        model_name = str(raw_name).strip().lower()
        candidate_keys = [model_name]
        if model_name == "deeponet_plasma":
            candidate_keys.append("deeponet")
        model_cfg: dict[str, Any] | None = None
        for key in candidate_keys:
            raw_model_cfg = train.get(key)
            if isinstance(raw_model_cfg, dict) and isinstance(raw_model_cfg.get("input_features"), dict):
                model_cfg = dict(raw_model_cfg)
                break
        if model_cfg is None:
            continue
        input_features_cfg = dict(model_cfg["input_features"])
        resolved_by_model.append(
            (
                model_name,
                resolve_distance_transform_cfg(input_features_cfg.get("distance_transform")),
            )
        )

    if not resolved_by_model:
        return resolve_distance_transform_cfg(None)
    reference_name, reference_cfg = resolved_by_model[0]
    conflicts = [
        (name, cfg)
        for name, cfg in resolved_by_model[1:]
        if cfg != reference_cfg
    ]
    if conflicts:
        details = ", ".join(
            f"{name}={cfg}" for name, cfg in [(reference_name, reference_cfg), *conflicts]
        )
        raise ValueError(
            "active models require different input_features.distance_transform contracts, "
            f"but preprocessing writes one coord_feature_scaler artifact: {details}"
        )
    return reference_cfg


def validate_coord_feature_scaler_distance_transform(
    *,
    coord_feature_scaler_artifact: dict[str, Any] | None,
    distance_transform_cfg: dict[str, Any] | None,
) -> None:
    """Fail when a scaler is applied in an incompatible transform space."""

    artifact = dict(coord_feature_scaler_artifact or {})
    if not bool(artifact.get("enabled", False)) or not dict(artifact.get("channels", {})):
        return
    contract_version = int(artifact.get("contract_version", 0) or 0)
    input_space = str(artifact.get("input_space", "")).strip().lower()
    if contract_version < 2:
        requested = resolve_distance_transform_cfg(distance_transform_cfg)
        if requested["mode"] != "raw":
            raise ValueError(
                "legacy coord_feature_scaler has no reliable distance-transform input-space contract and "
                f"cannot be used with mode={requested['mode']!r}; rerun preprocessing and training"
            )
        return
    if input_space != "post_distance_transform":
        raise ValueError(
            "coord_feature_scaler input-space contract is unsupported; rerun preprocessing "
            "with the current distance-transform configuration"
        )
    stored_raw = artifact.get("distance_transform_effective")
    if not isinstance(stored_raw, dict):
        raise ValueError("coord_feature_scaler v2 is missing distance_transform_effective")
    stored = resolve_distance_transform_cfg(stored_raw)
    requested = resolve_distance_transform_cfg(distance_transform_cfg)
    exact_keys = ("mode", "replace_distance_any")
    float_keys = ("signed_tanh_tau", "proximity_tau")
    matches = all(stored[key] == requested[key] for key in exact_keys) and all(
        np.isclose(float(stored[key]), float(requested[key]), rtol=1.0e-7, atol=1.0e-9)
        for key in float_keys
    )
    if not matches:
        raise ValueError(
            "coord_feature_scaler was fitted in a different distance-transform space: "
            f"artifact={stored}, runtime={requested}; rerun preprocessing"
        )


def resolve_distance_transform_effective(
    cfg: dict[str, Any],
    *,
    stats: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    mode = str(cfg.get("mode", "raw")).strip().lower()
    out = dict(cfg)
    if mode != "bounded_auto":
        return out, "config"
    raw_stats = dict(stats or {})
    s_tau = float(raw_stats.get("signed_tanh_tau_auto", 0.0))
    p_tau = float(raw_stats.get("proximity_tau_auto", 0.0))
    if not (np.isfinite(s_tau) and np.isfinite(p_tau) and s_tau > 0.0 and p_tau > 0.0):
        raise ValueError("distance_transform.mode=bounded_auto requires preprocessing distance_transform_stats")
    out["signed_tanh_tau"] = s_tau
    out["proximity_tau"] = p_tau
    out["signed_quantile"] = float(raw_stats.get("signed_quantile", 0.75))
    out["proximity_quantile"] = float(raw_stats.get("proximity_quantile", 0.50))
    return out, "artifact"


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
    part_mask_stack = None
    if geom_ctx is not None:
        regions = dict(getattr(geom_ctx, "regions", {}) or {})
        if "part_mask_stack" in regions:
            part_mask_stack = np.asarray(regions["part_mask_stack"], dtype=np.float32)
    mapping = derive_geom_feature_maps(
        coord_xy=coord_xy,
        distance_signed=distance_signed,
        distance_any=distance_any,
        mask_plasma=mask_plasma,
        h=h,
        w=w,
        part_mask_stack=part_mask_stack,
    )
    feats = np.stack([np.asarray(mapping[name], dtype=np.float32) for name in channels], axis=1).astype(np.float32)
    return feats, "runtime_geom"


def _spatial_channel_names(channels: Sequence[str], *, key_name: str) -> tuple[str, ...]:
    if isinstance(channels, (str, bytes)):
        raise ValueError(f"{key_name} must be a sequence of channel names")
    names = tuple(str(value).strip() for value in channels)
    if not names or any(not name for name in names):
        raise ValueError(f"{key_name} must contain at least one non-empty channel name")
    return names


def _batch_indices(indices: Any, *, n_cases: int | None = None) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.size > 0 and raw.dtype.kind not in {"i", "u"}:
        raise ValueError("case indices must contain integers")
    out = raw.astype(np.int64, copy=False).reshape(-1)
    if np.any(out < 0):
        raise IndexError("case indices must be >= 0")
    if n_cases is not None and np.any(out >= int(n_cases)):
        raise IndexError(f"case index is out of range for n_cases={int(n_cases)}")
    return out


@dataclass(frozen=True)
class CaseSpatialFeatureSource:
    channels: tuple[str, ...]
    h: int
    w: int
    n_cases: int
    source: str
    static_data: np.ndarray | None = None
    static_channels: tuple[str, ...] = ()
    case_data: np.ndarray | None = None
    case_channels: tuple[str, ...] = ()
    case_indices: np.ndarray | None = None
    distance_transform_cfg: dict[str, Any] | None = None
    coord_feature_scaler_artifact: dict[str, Any] | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_cases, self.h, self.w, len(self.channels))

    @property
    def scaling_applied(self) -> bool:
        raw = dict(self.coord_feature_scaler_artifact or {})
        return bool(raw.get("enabled", False)) and bool(dict(raw.get("channels", {})))

    def _global_indices(self, indices: Any) -> np.ndarray:
        local = _batch_indices(indices, n_cases=self.n_cases)
        if self.case_indices is None:
            global_idx = local
        else:
            case_indices = np.asarray(self.case_indices, dtype=np.int64).reshape(-1)
            if int(case_indices.shape[0]) != int(self.n_cases):
                raise ValueError(
                    "case spatial feature case_indices length must match n_cases: "
                    f"case_indices={int(case_indices.shape[0])}, n_cases={int(self.n_cases)}"
                )
            global_idx = case_indices[local]
        if self.case_data is not None:
            case_data = np.asarray(self.case_data)
            if case_data.ndim < 1:
                raise ValueError("case spatial feature case_data must include a case axis")
            if np.any(global_idx < 0) or np.any(global_idx >= int(case_data.shape[0])):
                raise IndexError(
                    "case spatial feature global index is out of range for case_data: "
                    f"case_rows={int(case_data.shape[0])}"
                )
        return global_idx.astype(np.int64, copy=False)

    def subset(self, indices: np.ndarray) -> "CaseSpatialFeatureSource":
        global_idx = self._global_indices(indices)
        return CaseSpatialFeatureSource(
            channels=self.channels,
            h=self.h,
            w=self.w,
            n_cases=int(global_idx.shape[0]),
            source=self.source,
            static_data=self.static_data,
            static_channels=self.static_channels,
            case_data=self.case_data,
            case_channels=self.case_channels,
            case_indices=global_idx,
            distance_transform_cfg=self.distance_transform_cfg,
            coord_feature_scaler_artifact=self.coord_feature_scaler_artifact,
        )

    def raw_batch(self, indices: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        """Return untransformed geometry in requested ``[B,H,W,C]`` order."""

        requested = _spatial_channel_names(channels, key_name="raw spatial channels")
        global_idx = self._global_indices(indices)
        if self.static_data is None or self.case_data is None:
            raise ValueError("compact case spatial feature source is missing static or case data")
        static_data = np.asarray(self.static_data, dtype=np.float32)
        case_data = np.asarray(self.case_data, dtype=np.float32)
        if static_data.ndim != 3 or tuple(static_data.shape[1:]) != (self.h, self.w):
            raise ValueError(
                "static spatial feature data must be [C,H,W]: "
                f"expected spatial shape={(self.h, self.w)}, got={static_data.shape}"
            )
        if case_data.ndim != 4 or tuple(case_data.shape[2:]) != (self.h, self.w):
            raise ValueError(
                "case spatial feature data must be [N,C,H,W]: "
                f"expected spatial shape={(self.h, self.w)}, got={case_data.shape}"
            )
        if int(static_data.shape[0]) != len(self.static_channels):
            raise ValueError("static spatial feature data/channel count mismatch")
        if int(case_data.shape[1]) != len(self.case_channels):
            raise ValueError("case spatial feature data/channel count mismatch")
        static_to_idx = {name: i for i, name in enumerate(self.static_channels)}
        case_to_idx = {name: i for i, name in enumerate(self.case_channels)}
        missing = [name for name in requested if name not in static_to_idx and name not in case_to_idx]
        if missing:
            raise ValueError(f"raw spatial feature channel is unavailable: {missing}")

        out = np.empty((int(global_idx.shape[0]), self.h, self.w, len(requested)), dtype=np.float32)
        for out_idx, name in enumerate(requested):
            if name in static_to_idx:
                out[..., out_idx] = static_data[static_to_idx[name]]
            else:
                out[..., out_idx] = case_data[global_idx, case_to_idx[name]]
        return out

    def batch(self, indices: np.ndarray) -> np.ndarray:
        out = self.raw_batch(indices, self.channels)
        flat = out.reshape(-1, len(self.channels))
        flat, _ = apply_distance_transform(
            flat.astype(np.float32),
            channels=list(self.channels),
            cfg=dict(self.distance_transform_cfg or {"mode": "raw"}),
        )
        flat, _, _ = apply_coord_feature_scaling(
            flat.astype(np.float32),
            channels=list(self.channels),
            coord_feature_scaler_artifact=dict(self.coord_feature_scaler_artifact or {}),
            distance_transform_cfg=dict(self.distance_transform_cfg or {"mode": "raw"}),
        )
        return flat.reshape(out.shape).astype(np.float32)


def _load_pack_channels(pack: dict[str, Any], key: str = "channels") -> list[str]:
    return [str(v) for v in np.asarray(pack.get(key)).reshape(-1).tolist()]


def build_case_spatial_features(
    *,
    channels: list[str],
    static_pack: dict[str, Any] | None = None,
    case_pack: dict[str, Any] | None = None,
    h: int,
    w: int,
    distance_transform_cfg: dict[str, Any] | None = None,
    coord_feature_scaler_artifact: dict[str, Any] | None = None,
    expected_case_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> tuple[CaseSpatialFeatureSource | None, str]:
    if static_pack and case_pack and "data" in static_pack and "data" in case_pack:
        static_data = np.asarray(static_pack.get("data"), dtype=np.float32)
        case_data = np.asarray(case_pack.get("data"), dtype=np.float32)
        if static_data.ndim != 3 or tuple(static_data.shape[1:]) != (h, w):
            raise ValueError(f"static spatial feature pack data must be [C,H,W], got {static_data.shape}")
        if case_data.ndim != 4 or tuple(case_data.shape[2:]) != (h, w):
            raise ValueError(f"case structure feature pack data must be [N,C,H,W], got {case_data.shape}")
        raw_case_ids = case_pack.get("case_ids")
        pack_case_ids = (
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(raw_case_ids).reshape(-1).tolist()
            ]
            if raw_case_ids is not None
            else None
        )
        if pack_case_ids is not None and len(pack_case_ids) != int(case_data.shape[0]):
            raise ValueError(
                "case structure feature pack case_ids length must match data rows: "
                f"case_ids={len(pack_case_ids)}, data_rows={int(case_data.shape[0])}"
            )
        if expected_case_ids is not None:
            expected = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(expected_case_ids).reshape(-1).tolist()
            ]
            if pack_case_ids is None:
                raise ValueError(
                    "case structure feature pack must include case_ids when expected_case_ids is provided"
                )
            if pack_case_ids != expected:
                mismatch_idx = next(
                    (
                        idx
                        for idx, (actual, wanted) in enumerate(zip(pack_case_ids, expected, strict=False))
                        if actual != wanted
                    ),
                    min(len(pack_case_ids), len(expected)),
                )
                actual_value = pack_case_ids[mismatch_idx] if mismatch_idx < len(pack_case_ids) else "<missing>"
                expected_value = expected[mismatch_idx] if mismatch_idx < len(expected) else "<missing>"
                raise ValueError(
                    "case structure feature pack case_ids must exactly match dataset order: "
                    f"index={mismatch_idx}, pack={actual_value!r}, dataset={expected_value!r}, "
                    f"pack_count={len(pack_case_ids)}, dataset_count={len(expected)}"
                )
        static_channels = _load_pack_channels(static_pack)
        case_channels = _load_pack_channels(case_pack)
        available = set(static_channels) | set(case_channels)
        missing = [name for name in channels if name not in available]
        if missing:
            return None, f"missing_channels:{','.join(missing)}"
        if not np.all(np.isfinite(static_data)) or not np.all(np.isfinite(case_data)):
            raise ValueError("compact case spatial feature packs contain non-finite values")
        validate_coord_feature_scaler_distance_transform(
            coord_feature_scaler_artifact=coord_feature_scaler_artifact,
            distance_transform_cfg=distance_transform_cfg,
        )
        return (
            CaseSpatialFeatureSource(
                channels=tuple(channels),
                h=int(h),
                w=int(w),
                n_cases=int(case_data.shape[0]),
                source="compact_case_spatial_pack",
                static_data=static_data,
                static_channels=tuple(static_channels),
                case_data=case_data,
                case_channels=tuple(case_channels),
                distance_transform_cfg=dict(distance_transform_cfg or {"mode": "raw"}),
                coord_feature_scaler_artifact=dict(coord_feature_scaler_artifact or {}),
            ),
            "compact_case_spatial_pack",
        )

    return None, "missing"


def materialize_case_spatial_batch(spatial_features: Any, indices: np.ndarray) -> np.ndarray | None:
    if spatial_features is None:
        return None
    if hasattr(spatial_features, "batch"):
        return spatial_features.batch(indices)
    return np.asarray(spatial_features, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]


def _materialize_raw_spatial_fallback(
    fallback: Mapping[str, Any],
    indices: np.ndarray,
    *,
    channels: tuple[str, ...],
) -> np.ndarray:
    missing = [name for name in channels if name not in fallback]
    if missing:
        raise ValueError(f"raw spatial fallback is missing channels: {missing}")

    channel_batches: list[np.ndarray] = []
    spatial_shape: tuple[int, int] | None = None
    for name in channels:
        values = np.asarray(fallback[name], dtype=np.float32)
        if values.ndim == 2:
            selected = np.broadcast_to(values, (int(indices.shape[0]), *values.shape))
        elif values.ndim == 3:
            if int(values.shape[0]) == 1:
                selected = np.broadcast_to(
                    values,
                    (int(indices.shape[0]), *values.shape[1:]),
                )
            elif np.any(indices >= int(values.shape[0])):
                raise IndexError(
                    f"raw spatial fallback channel {name!r} cannot select case indices from "
                    f"shape={values.shape}"
                )
            else:
                selected = values[indices]
        else:
            raise ValueError(
                f"raw spatial fallback channel {name!r} must be [H,W] or [N,H,W], got {values.shape}"
            )
        current_shape = (int(selected.shape[1]), int(selected.shape[2]))
        if spatial_shape is None:
            spatial_shape = current_shape
        elif current_shape != spatial_shape:
            raise ValueError(
                "raw spatial fallback channels must have one spatial shape: "
                f"expected={spatial_shape}, channel={name!r}, got={current_shape}"
            )
        channel_batches.append(np.asarray(selected, dtype=np.float32))
    return np.stack(channel_batches, axis=-1).astype(np.float32, copy=False)


def materialize_raw_spatial_batch(
    spatial_features: Any,
    indices: np.ndarray,
    *,
    channels: Sequence[str],
    fallback: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """Materialize raw geometry without accidentally reusing transformed features.

    A source is used only when it exposes the explicit :meth:`raw_batch` API.
    Otherwise callers must supply a mapping whose values are static ``[H,W]``
    maps or full case-aligned ``[N,H,W]`` maps. Non-finite values are preserved
    for the supervised-loss active-mask contract to handle.
    """

    requested = _spatial_channel_names(channels, key_name="raw spatial channels")
    local = _batch_indices(indices)
    raw_batch = getattr(spatial_features, "raw_batch", None)
    if callable(raw_batch):
        out = np.asarray(raw_batch(local, requested), dtype=np.float32)
        if out.ndim != 4:
            raise ValueError(f"raw spatial source must return [B,H,W,C], got {out.shape}")
        if int(out.shape[0]) != int(local.shape[0]) or int(out.shape[-1]) != len(requested):
            raise ValueError(
                "raw spatial source returned an incompatible batch/channel shape: "
                f"expected B={int(local.shape[0])}, C={len(requested)}, got={out.shape}"
            )
        return out
    if fallback is None:
        return None
    if not isinstance(fallback, Mapping):
        raise ValueError("raw spatial fallback must be a channel-to-array mapping")
    return _materialize_raw_spatial_fallback(fallback, local, channels=requested)
