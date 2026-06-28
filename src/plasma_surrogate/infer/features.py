"""Inference-time spatial feature construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.features.structure_feature_registry import validate_coord_feature_channels
from plasma_surrogate.preprocessing.scalers import ScalerFactory
from plasma_surrogate.train.spatial_features import (
    apply_distance_transform,
    derive_geom_feature_maps,
    distance_to_mask,
    part_sdf_maps_from_stack,
    resolve_distance_transform_cfg,
    resolve_distance_transform_effective,
)


class InferenceFeatureBuilder:
    def __init__(
        self,
        *,
        geometry_provider: Any,
        coord_scaler: dict[str, Any],
        coord_feature_scaler: dict[str, Any],
        coord_feature_pack: dict[str, Any],
        coord_distance_transform_stats: dict[str, Any],
        coord_input_scaling_cfg: dict[str, Any],
        coord_input_features_cfg: dict[str, Any],
        grid_input_features_cfg: dict[str, Any],
    ):
        self.geometry_provider = geometry_provider
        self.coord_scaler = dict(coord_scaler or {})
        self.coord_feature_scaler = dict(coord_feature_scaler or {})
        self.coord_feature_pack = dict(coord_feature_pack or {})
        self.coord_distance_transform_stats = dict(coord_distance_transform_stats or {})
        self.coord_input_scaling_cfg = dict(coord_input_scaling_cfg or {})
        self.coord_input_features_cfg = dict(coord_input_features_cfg or {})
        self.grid_input_features_cfg = dict(grid_input_features_cfg or {})

    def transform_coord(self, coord: np.ndarray) -> np.ndarray:
        mode = str(self.coord_input_scaling_cfg.get("mode", "none")).strip().lower()
        if mode not in {"none", "zscore", "minmax"}:
            raise ValueError("train.input_features.coord_input_scaling.mode must be one of: none, zscore, minmax")
        source = str(self.coord_input_scaling_cfg.get("source", "preprocess")).strip().lower()
        if source not in {"preprocess", "runtime_fit"}:
            raise ValueError("train.input_features.coord_input_scaling.source must be one of: preprocess, runtime_fit")
        arr = np.asarray(coord, dtype=np.float32)
        if mode == "none":
            return arr
        if source == "preprocess":
            payload = self.coord_scaler.get(mode)
            if not isinstance(payload, dict) or not payload:
                raise ValueError(
                    "coord_input_scaling.source=preprocess requires preprocessing/scalers/coord_scaler.json "
                    f"with '{mode}' payload"
                )
            scaler = ScalerFactory.from_dict(payload)
            return scaler.transform(arr).astype(np.float32)
        scaler = ScalerFactory.create(mode).fit(arr)
        return scaler.transform(arr).astype(np.float32)

    @staticmethod
    def coord_xy_from_geom(geom: GeometryContext) -> np.ndarray:
        raw_coord = np.asarray(geom.coord_grid, dtype=np.float32)
        if raw_coord.ndim != 3:
            raise ValueError(f"geom.coord_grid must be rank-3, got {raw_coord.shape}")
        if raw_coord.shape[0] == 2:
            return raw_coord.reshape(2, -1).T.astype(np.float32)
        if raw_coord.shape[-1] == 2:
            return raw_coord.reshape(-1, 2).astype(np.float32)
        raise ValueError(
            "geom.coord_grid shape mismatch: "
            f"expected (2,H,W) or (H,W,2), got {raw_coord.shape}"
        )

    @staticmethod
    def resolve_coord_feature_channels(model: Any) -> list[str]:
        raw = getattr(model, "input_feature_channels", None)
        if not isinstance(raw, list) or len(raw) == 0:
            return ["x", "y"]
        channels = [str(v) for v in raw]
        try:
            return list(validate_coord_feature_channels(channels))
        except ValueError as exc:
            raise ValueError(f"Unsupported coord feature channels in checkpoint metadata: {channels}") from exc

    def build_coord_feature_rows(self, geom: GeometryContext, channels: list[str]) -> np.ndarray:
        h, w = geom.mask_plasma.shape
        pack = dict(self.coord_feature_pack or {})
        distance_transform_cfg = resolve_distance_transform_cfg(
            dict(self.coord_input_features_cfg).get("distance_transform")
        )
        distance_transform_cfg, _ = resolve_distance_transform_effective(
            distance_transform_cfg,
            stats=self.coord_distance_transform_stats,
        )
        scaler_cfg = dict(self.coord_feature_scaler or {})
        has_coord_feature_scaler = bool(dict(scaler_cfg.get("channels", {})))
        if pack:
            rows = self._rows_from_pack(pack=pack, channels=channels, h=h, w=w)
            if rows is not None:
                if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
                    idx_x = channels.index("x")
                    idx_y = channels.index("y")
                    rows[:, [idx_x, idx_y]] = self.transform_coord(rows[:, [idx_x, idx_y]])
                rows, _ = apply_distance_transform(rows, channels=channels, cfg=distance_transform_cfg)
                return self.transform_coord_features(rows, channels)

        if not bool(getattr(self.geometry_provider, "supports_geom_param", False)):
            raise ValueError("input-feature contract requires preprocessing coord_feature_pack")
        rows = self._build_runtime_coord_feature_rows(geom=geom, channels=channels)
        if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
            idx_x = channels.index("x")
            idx_y = channels.index("y")
            rows[:, [idx_x, idx_y]] = self.transform_coord(rows[:, [idx_x, idx_y]])
        rows, _ = apply_distance_transform(rows, channels=channels, cfg=distance_transform_cfg)
        return self.transform_coord_features(rows, channels)

    def transform_coord_features(self, rows: np.ndarray, channels: list[str]) -> np.ndarray:
        raw = dict(self.coord_feature_scaler or {})
        if not bool(raw.get("enabled", False)):
            return np.asarray(rows, dtype=np.float32)
        channel_scalers = dict(raw.get("channels", {}))
        if len(channel_scalers) == 0:
            return np.asarray(rows, dtype=np.float32)
        out = np.asarray(rows, dtype=np.float32).copy()
        for i, name in enumerate(channels):
            payload = channel_scalers.get(name)
            if not isinstance(payload, dict) or len(payload) == 0:
                continue
            scaler = ScalerFactory.from_dict(payload)
            out[:, i : i + 1] = scaler.transform(out[:, i : i + 1]).astype(np.float32)
        return out.astype(np.float32)

    def require_coord_feature_scaling_enabled(self) -> None:
        raw = dict(self.coord_feature_scaler or {})
        enabled = bool(raw.get("enabled", False))
        mode = str(raw.get("mode", "none")).strip().lower()
        if (not enabled) or mode == "none":
            raise ValueError(
                "coord_mlp requires preprocessing.coord_features.scaling.enabled=true "
                "(mode must not be none) for inference"
            )

    def build_grid_feature_rows(self, geom: GeometryContext, channels: list[str]) -> np.ndarray:
        h, w = geom.mask_plasma.shape
        cfg = dict(self.grid_input_features_cfg or {})
        mode = str(cfg.get("mode", "geom_feature_pack")).strip().lower()
        if mode != "geom_feature_pack":
            raise ValueError("train.<grid_model>.input_features.mode must be geom_feature_pack")
        distance_transform_cfg = resolve_distance_transform_cfg(dict(cfg.get("distance_transform") or {}))
        distance_transform_cfg, _ = resolve_distance_transform_effective(
            distance_transform_cfg,
            stats=self.coord_distance_transform_stats,
        )
        scaler_cfg = dict(self.coord_feature_scaler or {})
        has_coord_feature_scaler = bool(dict(scaler_cfg.get("channels", {})))
        rows = self._rows_from_pack(pack=dict(self.coord_feature_pack or {}), channels=channels, h=h, w=w)
        if rows is not None:
            if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
                idx_x = channels.index("x")
                idx_y = channels.index("y")
                rows[:, [idx_x, idx_y]] = self.transform_coord(rows[:, [idx_x, idx_y]])
            rows, _ = apply_distance_transform(rows, channels=channels, cfg=distance_transform_cfg)
            return self.transform_coord_features(rows, channels)
        return self.build_coord_feature_rows(geom, channels)

    @staticmethod
    def _rows_from_pack(
        *,
        pack: dict[str, Any],
        channels: list[str],
        h: int,
        w: int,
    ) -> np.ndarray | None:
        if not pack:
            return None
        data = np.asarray(pack.get("data"), dtype=np.float32) if "data" in pack else None
        raw_channels = pack.get("channels")
        if data is None or raw_channels is None or data.ndim != 3 or data.shape[1:] != (h, w):
            return None
        pack_channels = [str(v) for v in np.asarray(raw_channels).reshape(-1).tolist()]
        pack_map = {name: data[i] for i, name in enumerate(pack_channels) if i < data.shape[0]}
        if not all(name in pack_map for name in channels):
            return None
        stacked = np.stack([pack_map[name] for name in channels], axis=0).astype(np.float32)
        return stacked.reshape(len(channels), -1).T.astype(np.float32)

    def _build_runtime_coord_feature_rows(self, geom: GeometryContext, channels: list[str]) -> np.ndarray:
        h, w = geom.mask_plasma.shape
        coord_xy = InferenceFeatureBuilder.coord_xy_from_geom(geom)
        distance_any = np.asarray(geom.distance_any, dtype=np.float32).reshape(-1)
        raw_signed = getattr(geom, "distance_signed", None)
        if raw_signed is None:
            mask_tmp = np.asarray(geom.mask_plasma, dtype=np.float32).reshape(-1)
            distance_signed = np.where(mask_tmp > 0.5, distance_any, -distance_any).astype(np.float32)
        else:
            distance_signed = np.asarray(raw_signed, dtype=np.float32).reshape(-1)
        mask_plasma = np.asarray(geom.mask_plasma, dtype=np.float32).reshape(-1)
        regions = dict(getattr(geom, "regions", {}) or {})
        part_stack = regions.get("part_mask_stack")
        mapping = derive_geom_feature_maps(
            coord_xy=coord_xy,
            distance_signed=distance_signed,
            distance_any=distance_any,
            mask_plasma=mask_plasma,
            h=h,
            w=w,
            part_mask_stack=np.asarray(part_stack, dtype=np.float32) if part_stack is not None else None,
        )
        solid_union = regions.get("solid_union_mask")
        if solid_union is not None:
            mask_coil = (np.asarray(solid_union, dtype=np.float32) > 0.5).astype(np.float32)
            distance_coil = distance_to_mask(mask_coil).astype(np.float32)
            coil_tau = float(
                self.coord_distance_transform_stats.get(
                    "coil_proximity_tau",
                    self.coord_distance_transform_stats.get("proximity_tau_auto", max(h, w)),
                )
            )
            coil_tau = max(coil_tau, 1.0e-3)
            mapping["mask_coil"] = mask_coil.reshape(-1).astype(np.float32)
            mapping["distance_coil"] = distance_coil.reshape(-1).astype(np.float32)
            mapping["coil_proximity"] = np.exp(-np.maximum(distance_coil, 0.0) / coil_tau).reshape(-1).astype(np.float32)
        if part_stack is not None:
            for name, arr in part_sdf_maps_from_stack(np.asarray(part_stack, dtype=np.float32)).items():
                mapping[name] = np.asarray(arr, dtype=np.float32).reshape(-1)
        return np.stack([np.asarray(mapping[name], dtype=np.float32) for name in channels], axis=1).astype(np.float32)


__all__ = ["InferenceFeatureBuilder"]
