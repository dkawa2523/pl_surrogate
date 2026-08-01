"""Inference model prediction dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.model_specs import get_model_spec
from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.infer.features import InferenceFeatureBuilder
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODDeepONetTorch
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP


class InferencePredictor:
    def __init__(self, *, model: Any, feature_builder: InferenceFeatureBuilder):
        self.model = model
        self.feature_builder = feature_builder

    def predict_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if isinstance(self.model, GlobalMLP):
            pred = self.model.predict_fields(cond_vec[None, :])
            return {k: v[0] for k, v in pred.items()}

        if self._model_capability("requires_spatial_features"):
            return self._predict_grid_spatial_fields(cond_vec, geom)

        if isinstance(self.model, PODDeepONetTorch):
            return self._predict_cond_only_fullfield_fields(cond_vec)

        if hasattr(self.model, "predict_fields"):
            try:
                pred = self.model.predict_fields(cond_vec[None, :], geom_ctx=geom)
            except TypeError:
                pred = self.model.predict_fields(cond_vec[None, :])
            return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

        raise TypeError("Unsupported model type")

    def _model_capability(self, name: str) -> bool:
        """Read stable input capabilities from the canonical model registry."""

        model_type = getattr(self.model, "model_type", None)
        if model_type is None:
            return bool(getattr(self.model, name, False))
        expected = bool(getattr(get_model_spec(model_type), name))
        declared = getattr(self.model, name, None)
        if declared is not None and bool(declared) != expected:
            raise RuntimeError(
                f"model capability drift for {model_type}.{name}: "
                f"registry={expected}, implementation={bool(declared)}"
            )
        return expected

    def _predict_grid_spatial_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if self._model_capability("requires_scaled_spatial_features"):
            self.feature_builder.require_coord_feature_scaling_enabled()
        channels = self.feature_builder.resolve_coord_feature_channels(self.model)
        spatial_rows = self.feature_builder.build_grid_feature_rows(geom, channels)
        h, w = geom.mask_plasma.shape
        spatial_map = spatial_rows.reshape(h, w, len(channels))[None, ...].astype(np.float32)
        pred = self.model.predict_fields(cond_vec[None, :], spatial_features=spatial_map)
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

    def _predict_cond_only_fullfield_fields(self, cond_vec: np.ndarray) -> dict[str, np.ndarray]:
        pred = self.model.predict_fields(cond_vec[None, :])
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

__all__ = ["InferencePredictor"]
