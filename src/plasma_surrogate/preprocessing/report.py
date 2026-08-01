"""Preprocess validation report builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore


class PreprocessReportBuilder:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def save(
        self,
        *,
        cases: list[dict[str, Any]],
        coord_source_applied: str,
        coord_source_requested: str,
        coord_contract_status: str,
        negative_ratio: float,
        distance_contract_status: str,
        coord_rows: np.ndarray,
        coord_rows_z: np.ndarray,
        coord_rows_mm: np.ndarray,
        scaler_fit_split: str,
        protocol_scaler_fit_splits: list[str],
        structure_holdout_meta: dict[str, Any],
        train_indices: np.ndarray,
        target_transforms_cfg: dict[str, Any],
        y_vars: list[str],
        coord_features_enabled: bool,
        coord_feature_channels: list[str],
        coord_feature_rel_path: str,
        coord_features_scaling_enabled: bool,
        coord_features_scaling_mode: str,
        case_spatial_feature_enabled: bool,
        case_spatial_feature_meta_rel: str,
        case_spatial_feature_shape: list[int],
        static_spatial_feature_rel_path: str,
        static_spatial_feature_meta_rel: str,
        static_spatial_feature_shape: list[int],
        case_structure_feature_rel_path: str,
        case_structure_feature_meta_rel: str,
        case_structure_feature_shape: list[int],
        distance_stats_enabled: bool,
        signed_tanh_tau_auto: float,
        proximity_tau_auto: float,
        signed_q_raw: float,
        proximity_q_raw: float,
        descriptor_artifact_meta: dict[str, Any],
        latent_artifact_meta: dict[str, Any],
        runtime_input_mode_meta: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "n_cases": len(cases),
            "coord_grid_source": coord_source_applied,
            "coord_grid_source_requested": coord_source_requested,
            "coord_grid_source_applied": coord_source_applied,
            "coord_grid_contract_status": coord_contract_status,
            "distance_signed_negative_ratio": float(negative_ratio),
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
            "scaler_fit_split": scaler_fit_split,
            "scaler_fit_train_case_count": int(len(train_indices)),
            "protocol_scaler_fit_splits": list(protocol_scaler_fit_splits),
            "structure_holdout": dict(structure_holdout_meta),
            "target_value_transform_effective": {
                name: str(dict(target_transforms_cfg.get(name, {})).get("value_transform", "identity"))
                for name in [str(var_name) for var_name in y_vars]
            },
            "coord_features_enabled": bool(coord_features_enabled),
            "coord_feature_channels": list(coord_feature_channels),
            "coord_feature_pack_path": (
                "" if case_spatial_feature_enabled else (coord_feature_rel_path if coord_features_enabled else "")
            ),
            "coord_feature_scaling_enabled": bool(coord_features_scaling_enabled),
            "coord_feature_scaling_mode": str(coord_features_scaling_mode),
            "case_spatial_pack_used": bool(case_spatial_feature_enabled),
            "case_spatial_feature_storage": "split_static_case" if case_spatial_feature_enabled else "",
            "case_spatial_feature_shape": list(case_spatial_feature_shape),
            "static_spatial_feature_pack_path": static_spatial_feature_rel_path if case_spatial_feature_enabled else "",
            "static_spatial_feature_pack_meta_path": static_spatial_feature_meta_rel,
            "static_spatial_feature_shape": list(static_spatial_feature_shape),
            "case_structure_feature_pack_path": case_structure_feature_rel_path if case_spatial_feature_enabled else "",
            "case_structure_feature_pack_meta_path": case_structure_feature_meta_rel,
            "case_structure_feature_shape": list(case_structure_feature_shape),
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
            **descriptor_artifact_meta,
            **latent_artifact_meta,
        }
        payload.update(runtime_input_mode_meta)
        self.store.save_json("validation/report.json", payload)
        return payload


__all__ = ["PreprocessReportBuilder"]
