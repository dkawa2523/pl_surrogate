"""Preprocessing runner for split/scaler/sampling artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.features.geometry_feature_store import hash_json
from plasma_surrogate.preprocessing.sampling import (
    build_deeponet_indices,
    build_flattened_coords,
    build_patch_index,
    build_phase_wrap_pairs,
    build_point_pools,
    build_time_adjacent_pairs,
)
from plasma_surrogate.preprocessing.scalers import ScalerFactory, fit_scalers_train_only
from plasma_surrogate.preprocessing.schema import AxisSchema, ChannelMap, CondSchema
from plasma_surrogate.preprocessing.split import build_casewise_splits, build_pressure_extrap_split
from plasma_surrogate.preprocessing.split import (
    build_extrapolation_split,
    build_interpolation_overlap_split_with_status,
    build_interpolation_split,
)


_ALLOWED_COORD_FEATURE_CHANNELS = (
    "x",
    "y",
    "distance_signed",
    "distance_any",
    "mask_plasma",
    "normal_x",
    "normal_y",
    "curvature_proxy",
)


def _resolve_y_vars(cases: list[dict[str, Any]]) -> list[str]:
    keys = list(cases[0]["y"].keys())
    if len(keys) == 0:
        raise ValueError("Each case.y must include at least one target variable")
    return [str(k) for k in keys]


def _coord_xy_maps(coord_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(coord_grid, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError(f"coord_grid must be rank-3, got shape={grid.shape}")
    if grid.shape[0] == 2:
        return grid[0], grid[1]
    if grid.shape[-1] == 2:
        return grid[..., 0], grid[..., 1]
    raise ValueError(f"coord_grid must be (2,H,W) or (H,W,2), got {grid.shape}")


def _derive_geometry_feature_maps(
    *,
    distance_signed: np.ndarray,
) -> dict[str, np.ndarray]:
    signed = np.asarray(distance_signed, dtype=np.float32)
    gy, gx = np.gradient(signed, edge_order=1)
    gnorm = np.sqrt(gx**2 + gy**2).astype(np.float32)
    gnorm = np.where(gnorm < 1e-6, 1e-6, gnorm).astype(np.float32)
    normal_x = (gx / gnorm).astype(np.float32)
    normal_y = (gy / gnorm).astype(np.float32)
    dnx_dy, dnx_dx = np.gradient(normal_x, edge_order=1)
    dny_dy, dny_dx = np.gradient(normal_y, edge_order=1)
    curvature_proxy = (dnx_dx + dny_dy).astype(np.float32)
    return {
        "normal_x": normal_x,
        "normal_y": normal_y,
        "curvature_proxy": curvature_proxy,
    }


@dataclass
class PreprocessOutput:
    split: dict[str, list[str]]
    cond_stats: dict[str, Any]
    hashes: dict[str, str]


class PreprocessRunner:
    """Minimal implementation for cycle1 preprocessing outputs."""

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.store = ArtifactStore(self.output_dir)

    def run(self, cases: list[dict[str, Any]], geometry_root: str | Path) -> PreprocessOutput:
        split_cfg = self.cfg.get("split", {})
        ratios = split_cfg.get("ratios", [0.7, 0.15, 0.15])
        split_groups = [str(c.get("split_group", c["case_id"])) for c in cases]
        split = build_casewise_splits(
            [c["case_id"] for c in cases],
            split_groups=split_groups,
            seed=int(split_cfg.get("seed", 0)),
            ratios=tuple(ratios),
        )

        cond_order = list(self.cfg.get("cond_order", []))
        if not cond_order:
            cond_order = sorted(cases[0]["cond"].keys())
        cond_schema = CondSchema(order=cond_order)
        axis_schema = AxisSchema.from_dict(self.cfg.get("axis_schema", {"mode": "steady"}))
        cond_by_case = {c["case_id"]: c["cond"] for c in cases}
        extrap_key = str(split_cfg.get("pressure_extrap_key", cond_order[0]))
        split_extrap = build_pressure_extrap_split(
            [c["case_id"] for c in cases],
            cond_values=cond_by_case,
            key=extrap_key,
            holdout_ratio=float(split_cfg.get("pressure_extrap_holdout_ratio", 0.2)),
            val_ratio_within_remain=float(split_cfg.get("pressure_extrap_val_ratio", 0.2)),
        )
        interp_mode = str(split_cfg.get("interp_mode", "marginal")).strip().lower()
        if interp_mode not in {"marginal", "overlap"}:
            raise ValueError("split.interp_mode must be one of: marginal, overlap")
        split_interp_marginal = build_interpolation_split(
            [c["case_id"] for c in cases],
            cond_values=cond_by_case,
            keys=cond_order,
            seed=int(split_cfg.get("seed", 0)),
            ratios=tuple(ratios),
            mode="marginal",
        )
        split_interp_overlap_status = build_interpolation_overlap_split_with_status(
            [c["case_id"] for c in cases],
            cond_values=cond_by_case,
            keys=cond_order,
            seed=int(split_cfg.get("seed", 0)),
            ratios=tuple(ratios),
        )
        split_interp_overlap = dict(split_interp_overlap_status["split"])
        overlap_feasible = bool(split_interp_overlap_status.get("feasible", True))
        overlap_reason = str(split_interp_overlap_status.get("reason", ""))
        fallback_applied = bool(interp_mode == "overlap" and (not overlap_feasible))
        split_interp = split_interp_overlap if (interp_mode == "overlap" and overlap_feasible) else split_interp_marginal
        interp_status = {
            "requested_mode": interp_mode,
            "applied_mode": "marginal" if fallback_applied else interp_mode,
            "feasible": overlap_feasible,
            "reason": overlap_reason,
            "fallback_applied": fallback_applied,
        }
        split_extrap_v1 = build_extrapolation_split(
            [c["case_id"] for c in cases],
            cond_values=cond_by_case,
            key=extrap_key,
            holdout_ratio=float(split_cfg.get("pressure_extrap_holdout_ratio", 0.2)),
            val_ratio_within_remain=float(split_cfg.get("pressure_extrap_val_ratio", 0.2)),
        )

        cond_matrix = []
        for c in cases:
            base = cond_schema.encode(c["cond"])
            axis_vec = axis_schema.encode(c.get("axis", 0.0))
            cond_matrix.append(np.concatenate([base, axis_vec], axis=0))
        cond_matrix = np.stack(cond_matrix, axis=0)

        y_vars = _resolve_y_vars(cases)
        y_by_var = {var: np.stack([np.asarray(c["y"][var], dtype=np.float32) for c in cases], axis=0) for var in y_vars}
        coord_grid_source = str(self.cfg.get("coord_grid_source", "coord_grid"))
        distance_contract_cfg = dict(self.cfg.get("distance_contract", {}))
        distance_contract_mode = str(distance_contract_cfg.get("require_negative_outside", "warn")).strip().lower()
        if distance_contract_mode not in {"warn", "error", "off"}:
            raise ValueError(
                "preprocessing.distance_contract.require_negative_outside must be one of: warn, error, off"
            )
        coord_contract_cfg = dict(self.cfg.get("coord_grid_contract", {}))
        coord_contract_mode = str(coord_contract_cfg.get("require_requested_source", "warn")).strip().lower()
        if coord_contract_mode not in {"warn", "error", "off"}:
            raise ValueError(
                "preprocessing.coord_grid_contract.require_requested_source must be one of: warn, error, off"
            )
        geom = FixedGeometryProvider(geometry_root, coord_grid_source=coord_grid_source).get()
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

        case_to_idx = {c["case_id"]: i for i, c in enumerate(cases)}
        train_indices = np.array([case_to_idx[cid] for cid in split["train"]], dtype=np.int64)
        scaler_cfg = dict(self.cfg.get("scalers", {}))
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
        transforms = fit_scalers_train_only(
            cond_matrix,
            y_by_var,
            train_indices,
            mask_plasma=geom.mask_plasma,
            scaler_fit_policy=y_fit_policy,
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

        self.store.save_json("split/split_random_v1.json", split)
        self.store.save_json("split/split_interp_marginal_v1.json", split_interp_marginal)
        self.store.save_json("split/split_interp_overlap_v1.json", split_interp_overlap)
        self.store.save_json("split/split_interp_v1.json", split_interp)
        self.store.save_json("split/split_interp_status_v1.json", interp_status)
        self.store.save_json("split/split_pressure_extrap_v1.json", split_extrap)
        self.store.save_json("split/split_extrap_v1.json", split_extrap_v1)
        self.store.save_json("schema/cond_schema.json", cond_schema.to_dict())
        self.store.save_json("schema/axis_schema.json", axis_schema.to_dict())
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
        self.store.save_json("schema/channel_map.json", channel_map.to_dict())
        sample_shape = np.asarray(cases[0]["y"][y_vars[0]], dtype=np.float32).shape
        self.store.save_json(
            "schema/output_layout.json",
            {"order": "C", "shape": [len(y_vars), int(sample_shape[0]), int(sample_shape[1])], "vars": y_vars},
        )
        cond_scaler_payload = transforms.cond_scaler.to_dict()
        cond_scaler_payload["cond_dim"] = transforms.cond_dim
        cond_scaler_payload["fit_policy"] = transforms.fit_policy
        cond_scaler_payload["mask_applied"] = transforms.mask_applied
        cond_scaler_payload["target_transforms"] = transforms.target_transforms
        self.store.save_json("scalers/cond_scaler.json", cond_scaler_payload)
        self.store.save_json("scalers/y_scalers.json", {k: v.to_dict() for k, v in transforms.y_scalers.items()})
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
        self.store.save_json(
            "scalers/fit_policy.json",
            {
                "y_fit_policy": transforms.fit_policy,
                "mask_applied": bool(transforms.mask_applied),
                "mask_active_ratio": float(np.mean(geom.mask_plasma > 0.5)),
                "target_transforms": transforms.target_transforms,
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
        coord_feature_channels_raw = coord_features_cfg.get(
            "channels",
            ["x", "y", "distance_signed", "distance_any", "mask_plasma"],
        )
        if not isinstance(coord_feature_channels_raw, list) or len(coord_feature_channels_raw) == 0:
            raise ValueError("preprocessing.coord_features.channels must be a non-empty list")
        coord_feature_channels = [str(v) for v in coord_feature_channels_raw]
        unknown_channels = [v for v in coord_feature_channels if v not in _ALLOWED_COORD_FEATURE_CHANNELS]
        if unknown_channels:
            raise ValueError(
                "preprocessing.coord_features.channels contains unsupported entries: "
                f"{unknown_channels}; allowed={list(_ALLOWED_COORD_FEATURE_CHANNELS)}"
            )
        if len(set(coord_feature_channels)) != len(coord_feature_channels):
            raise ValueError("preprocessing.coord_features.channels must not contain duplicates")
        coord_feature_rel_path = str(coord_features_cfg.get("output", "features/coord_feature_pack.npz"))
        coord_x, coord_y = _coord_xy_maps(raw_coord_grid)
        coord_feature_maps = {
            "x": np.asarray(coord_x, dtype=np.float32),
            "y": np.asarray(coord_y, dtype=np.float32),
            "distance_signed": np.asarray(distance_signed, dtype=np.float32),
            "distance_any": np.asarray(geom.distance_any, dtype=np.float32),
            "mask_plasma": np.asarray(geom.mask_plasma, dtype=np.float32),
        }
        coord_feature_maps.update(_derive_geometry_feature_maps(distance_signed=distance_signed))
        coord_feature_scalers_payload: dict[str, Any] = {
            "enabled": bool(coord_features_scaling_enabled),
            "mode": str(coord_features_scaling_mode),
            "fit_scope": str(coord_features_scaling_fit_scope),
            "mask_scope": str(coord_features_scaling_mask_scope),
            "chamber_band_px": float(coord_features_scaling_band),
            "channels": {},
        }
        fit_mask = np.ones_like(geom.mask_plasma, dtype=bool)
        if coord_features_scaling_mask_scope == "plasma_only":
            fit_mask = geom.mask_plasma > 0.5
        elif coord_features_scaling_mask_scope == "plasma_plus_band":
            fit_mask = np.asarray(distance_signed, dtype=np.float32) >= (-float(coord_features_scaling_band))
        if coord_features_scaling_fit_scope == "all":
            fit_mask = np.ones_like(fit_mask, dtype=bool)
        if not np.any(fit_mask):
            fit_mask = np.ones_like(fit_mask, dtype=bool)
            coord_feature_scalers_payload["fit_scope_fallback"] = "all_pixels"
        for name in coord_feature_channels:
            values = np.asarray(coord_feature_maps[name], dtype=np.float32).reshape(-1, 1)
            if coord_features_scaling_enabled and coord_features_scaling_mode != "none" and name != "mask_plasma":
                scaler = ScalerFactory.create(coord_features_scaling_mode).fit(values[fit_mask.reshape(-1)])
            else:
                scaler = ScalerFactory.create("none").fit(values)
            coord_feature_scalers_payload["channels"][name] = scaler.to_dict()
        self.store.save_json("scalers/coord_feature_scaler.json", coord_feature_scalers_payload)
        if coord_features_enabled:
            coord_feature_data = np.stack([coord_feature_maps[name] for name in coord_feature_channels], axis=0).astype(np.float32)
            self.store.save_npz(coord_feature_rel_path, data=coord_feature_data, channels=np.asarray(coord_feature_channels))
            pack_path = Path(coord_feature_rel_path)
            meta_rel = str(pack_path.parent / f"{pack_path.stem}_meta.json")
            self.store.save_json(
                meta_rel,
                {
                    "enabled": True,
                    "channels": coord_feature_channels,
                    "shape": [int(v) for v in coord_feature_data.shape],
                    "coord_source": str(getattr(geom, "coord_source", "unknown")),
                    "scaling": {
                        "enabled": bool(coord_features_scaling_enabled),
                        "mode": str(coord_features_scaling_mode),
                    },
                },
            )
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
        deeponet_index_payload: dict[str, Any] = {}
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
            if "default" in task_cfgs:
                legacy_task = "default"
            elif "poisson_head" in task_cfgs:
                legacy_task = "poisson_head"
            else:
                legacy_task = sorted(task_cfgs.keys())[0]
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

                # Keep legacy single-task paths for backward compatibility.
                if task_name == legacy_task:
                    deeponet_index_payload = payload
                    deeponet_index_meta = meta
                    self.store.save_json("sampling/deeponet/sensor_query_index.json", payload)
                    self.store.save_json("sampling/deeponet/index_meta.json", meta)
                    np.save(deeponet_root / "sensor_coords.npy", sensor_coords.astype(np.float32))
                    np.save(deeponet_root / "query_coords.npy", query_coords.astype(np.float32))
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
            vals = y_by_var[var][train_indices].reshape(-1)
            y_stats[var] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        self.store.save_json("stats/y_stats.json", y_stats)
        split_hash = hash_json({"split_random": split, "split_pressure_extrap": split_extrap})
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
                raise ValueError(
                    "distance contract violation: outside signed-distance has no negative values. "
                    "Set preprocessing.distance_contract.require_negative_outside=warn|off to proceed."
                )
        coord_source_applied = str(getattr(geom, "coord_source", "unknown"))
        coord_source_requested = coord_grid_source
        coord_contract_status = "ok"
        if coord_source_applied != coord_source_requested:
            coord_contract_status = "fallback_applied"
            if coord_contract_mode == "error":
                raise ValueError(
                    "coord-grid contract violation: requested source "
                    f"{coord_source_requested} but applied {coord_source_applied}"
                )

        self.store.save_json(
            "validation/report.json",
            {
                "status": "ok",
                "n_cases": len(cases),
                "coord_grid_source": coord_source_applied,
                "coord_grid_source_requested": coord_source_requested,
                "coord_grid_source_applied": coord_source_applied,
                "coord_grid_contract_status": coord_contract_status,
                "distance_signed_negative_ratio": negative_ratio,
                "distance_contract_status": distance_contract_status,
                "coord_value_range_raw": {
                    "x": [float(np.min(coord_rows[:, 0])), float(np.max(coord_rows[:, 0]))],
                    "y": [float(np.min(coord_rows[:, 1])), float(np.max(coord_rows[:, 1]))],
                },
                "coord_value_range_scaled": {
                    "zscore": {
                        "x": [float(np.min(coord_rows_z[:, 0])), float(np.max(coord_rows_z[:, 0]))],
                        "y": [float(np.min(coord_rows_z[:, 1])), float(np.max(coord_rows_z[:, 1]))],
                    },
                    "minmax": {
                        "x": [float(np.min(coord_rows_mm[:, 0])), float(np.max(coord_rows_mm[:, 0]))],
                        "y": [float(np.min(coord_rows_mm[:, 1])), float(np.max(coord_rows_mm[:, 1]))],
                    },
                },
                "coord_scaler_status": "ok",
                "coord_features_enabled": bool(coord_features_enabled),
                "coord_feature_channels": coord_feature_channels,
                "coord_feature_pack_path": coord_feature_rel_path if coord_features_enabled else "",
                "coord_feature_scaling_enabled": bool(coord_features_scaling_enabled),
                "coord_feature_scaling_mode": str(coord_features_scaling_mode),
                "distance_transform_stats_path": "scalers/distance_transform_stats.json",
                "distance_transform_stats_enabled": bool(distance_stats_enabled),
                "distance_transform_tau_auto": {
                    "signed_tanh_tau_auto": float(signed_tanh_tau_auto),
                    "proximity_tau_auto": float(proximity_tau_auto),
                },
                "distance_transform_quantiles": {
                    "signed_quantile": float(signed_q_raw),
                    "proximity_quantile": float(proximity_q_raw),
                },
            },
        )

        return PreprocessOutput(
            split=split,
            cond_stats=cond_stats,
            hashes={
                "split_hash": split_hash,
                "sampling_hash": sampling_hash,
                "deeponet_index_hash": deeponet_index_hash,
            },
        )
