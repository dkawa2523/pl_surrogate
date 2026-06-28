"""Inference model prediction dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.infer.features import InferenceFeatureBuilder
from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet
from plasma_surrogate.models.cno.simple_cno import CNOBaseline
from plasma_surrogate.models.deeponet.geom_deeponet_siren import GeomDeepONetSIREN
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODDeepONetTorch
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline


class InferencePredictor:
    def __init__(self, *, model: Any, feature_builder: InferenceFeatureBuilder):
        self.model = model
        self.feature_builder = feature_builder

    def predict_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if isinstance(self.model, GlobalMLP):
            pred = self.model.predict_fields(cond_vec[None, :])
            return {k: v[0] for k, v in pred.items()}

        if isinstance(
            self.model,
            (
                UNetBaseline,
                UNetPPBaseline,
                FNOBaseline,
                FFNOBaseline,
                UNOBaseline,
                CNOBaseline,
                CNOOperatorUNet,
                CoordMLPTorch,
                GeomDeepONetSIREN,
            ),
        ):
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

    def _predict_grid_spatial_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if isinstance(self.model, CoordMLPTorch):
            self.feature_builder.require_coord_feature_scaling_enabled()
        channels = self.feature_builder.resolve_coord_feature_channels(self.model)
        spatial_rows = self.feature_builder.build_grid_feature_rows(geom, channels)
        h, w = geom.mask_plasma.shape
        spatial_map = spatial_rows.reshape(h, w, len(channels))[None, ...].astype(np.float32)
        if isinstance(self.model, (UNetBaseline, UNetPPBaseline)):
            pred = self.model.forward_features(cond_vec[None, :], spatial_features=spatial_map)
        else:
            pred = self.model.predict_fields(cond_vec[None, :], spatial_features=spatial_map)
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

    def _predict_cond_only_fullfield_fields(self, cond_vec: np.ndarray) -> dict[str, np.ndarray]:
        pred = self.model.predict_fields(cond_vec[None, :])
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}


__all__ = ["InferencePredictor"]
