"""Shared torch-side spatial wrapper helpers for unet-family grid field models."""

from __future__ import annotations

import numpy as np

from plasma_surrogate.models._torch_spatial_common import (
    _backward_raw_torch_step,
    _load_state_dict_numpy_torch,
    _resolve_batched_spatial_features,
    _resolve_torch_device,
    _state_dict_numpy_torch,
    _validate_static_spatial_features,
)


class _TorchSpatialFieldMixin:
    """Shared torch wrapper for models that consume [cond + spatial feature] maps."""

    _spatial_label = "grid"
    requires_spatial_features = True
    requires_scaled_spatial_features = False

    def _torch_forward(self, xt):
        return self.net(xt)

    def _torch_label(self) -> str:
        return str(getattr(self, "_spatial_label", type(self).__name__)).strip().lower()

    def _init_torch_device(self) -> None:
        self.device = _resolve_torch_device(self.torch)

    def _ensure_net_device(self) -> None:
        if getattr(self, "net", None) is not None:
            self.net.to(self.device)

    def set_static_spatial_features(self, spatial_features: np.ndarray | None) -> None:
        if spatial_features is None:
            self._static_spatial_features = None
            return
        self._static_spatial_features = _validate_static_spatial_features(
            spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=self._torch_label(),
        )

    def _resolve_spatial_features(self, cond: np.ndarray, spatial_features: np.ndarray | None) -> np.ndarray:
        return _resolve_batched_spatial_features(
            cond,
            spatial_features,
            static_spatial_features=self._static_spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=self._torch_label(),
            explicit_requirement_message=(
                f"{self._torch_label()} input_features requires explicit spatial features for channels "
                f"{self.input_feature_channels}"
            ),
        )

    def _feature_map(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(cond, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        h, w = self.grid_shape
        cond_map = np.repeat(x[:, None, None, :], h, axis=1)
        cond_map = np.repeat(cond_map, w, axis=2)
        spatial_map = self._resolve_spatial_features(x, spatial_features)
        return np.concatenate([cond_map, spatial_map], axis=-1).astype(np.float32)

    def feature_matrix(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        return fmap.reshape(-1, self.feature_dim).astype(np.float32)

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool, spatial_features: np.ndarray | None) -> np.ndarray:
        torch = self.torch
        self._ensure_net_device()
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        xt = torch.from_numpy(np.moveaxis(fmap, -1, 1).astype(np.float32)).to(self.device)
        if training:
            self.net.train()
            yt = self._torch_forward(xt)
            self._torch_last_in = xt
            self._torch_last_out = yt
        else:
            self.net.eval()
            with torch.no_grad():
                yt = self._torch_forward(xt)
            self._torch_last_in = None
            self._torch_last_out = None
        return yt.detach().cpu().numpy().astype(np.float32)

    def forward_features(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        y = self.forward_raw(cond, training=False, spatial_features=spatial_features)
        out = {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}
        if self.with_rho_eff_head:
            out["rho_eff"] = y[:, self.out_channels : self.out_channels + 1]
        return out

    def predict_fields(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return self.forward_features(cond, spatial_features=spatial_features)

    def _backward_raw_torch(
        self,
        grad_raw: np.ndarray,
        *,
        lr: float,
        apply_step: bool = True,
    ) -> dict[str, float]:
        if self._torch_last_out is None:
            raise RuntimeError(f"{type(self).__name__}.backward_raw called without torch forward cache")
        out = _backward_raw_torch_step(
            torch=self.torch,
            net=self.net,
            grad_raw=grad_raw,
            last_out=self._torch_last_out,
            lr=float(lr),
            step_reference=self._torch_step_reference,
            apply_step=bool(apply_step),
        )
        self._torch_last_out = None
        self._torch_last_in = None
        return out

    def _state_dict_numpy_torch(self) -> dict[str, np.ndarray]:
        return _state_dict_numpy_torch(self.net)

    def _load_state_dict_numpy_torch(self, state: dict[str, np.ndarray], *, legacy_message: str | None = None) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message=(
                str(legacy_message)
                if legacy_message
                else f"{type(self).__name__}(torch) state dict does not contain expected torch::* weights"
            ),
            device=self.device,
        )
        self._ensure_net_device()
