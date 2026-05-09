"""Shared torch wrapper for grid-to-field spectral baselines."""

from __future__ import annotations

from typing import Any

import numpy as np


class _TorchGridFieldBaseline:
    """Shared wrapper for torch-based grid field models with spatial features."""

    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int,
        output_keys: list[str] | None,
        seed: int,
        with_rho_eff_head: bool,
        head_mlp: dict[str, object] | None,
        input_feature_channels: list[str] | None,
        backend: str,
        cfg_prefix: str,
        default_output_keys: list[str],
        impl_version: str,
        head_arch_version: str,
    ) -> None:
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            base = list(default_output_keys)
            extra = [f"out_{i}" for i in range(max(0, self.out_channels - len(base)))]
            self.output_keys = (base + extra)[: self.out_channels]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        self.with_rho_eff_head = bool(with_rho_eff_head)
        self.raw_out_channels = self.out_channels + (1 if self.with_rho_eff_head else 0)
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError(f"{type(self).__name__} supports only backend='torch'")

        channels = list(input_feature_channels or ["x", "y"])
        if len(channels) == 0:
            raise ValueError(f"{cfg_prefix}.input_features.features must be a non-empty list")
        if len(set(channels)) != len(channels):
            raise ValueError(f"{cfg_prefix}.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.feature_dim = int(self.input_dim + self.spatial_feature_dim)
        self.head_mlp_cfg = dict(head_mlp or {})
        self.fno_impl_version = str(impl_version)
        self.head_arch_version = str(head_arch_version)

        h, w = self.grid_shape
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        self.coord = np.stack([xv, yv], axis=-1).astype(np.float32)
        self._static_spatial_features: np.ndarray | None = None

        from plasma_surrogate.core.torch_backend import require_torch

        torch = require_torch()
        torch.manual_seed(int(seed))
        self.torch = torch
        self.device = torch.device("cuda" if bool(torch.cuda.is_available()) else "cpu")
        self.net: Any = None
        self._torch_last_in = None
        self._torch_last_out = None

    def _ensure_net_device(self) -> None:
        if self.net is not None:
            self.net.to(self.device)

    def set_static_spatial_features(self, spatial_features: np.ndarray) -> None:
        arr = np.asarray(spatial_features, dtype=np.float32)
        h, w = self.grid_shape
        if arr.shape != (h, w, self.spatial_feature_dim):
            raise ValueError(
                f"{type(self).__name__} static spatial features must have shape "
                f"({h}, {w}, {self.spatial_feature_dim}), got={arr.shape}"
            )
        self._static_spatial_features = arr

    def _feature_map(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(cond, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        bsz = x.shape[0]
        h, w = self.grid_shape
        cond_map = np.repeat(x[:, None, None, :], h, axis=1)
        cond_map = np.repeat(cond_map, w, axis=2)
        if spatial_features is not None:
            spatial = np.asarray(spatial_features, dtype=np.float32)
        elif self._static_spatial_features is not None:
            spatial = self._static_spatial_features
        else:
            if self.spatial_feature_dim == 2:
                spatial = self.coord
            else:
                spatial = np.zeros((h, w, self.spatial_feature_dim), dtype=np.float32)
                spatial[..., :2] = self.coord
        if spatial.ndim == 3:
            if spatial.shape != (h, w, self.spatial_feature_dim):
                raise ValueError(
                    f"{type(self).__name__} spatial feature map shape mismatch: "
                    f"expected {(h, w, self.spatial_feature_dim)}, got={spatial.shape}"
                )
            spatial_batch = np.repeat(spatial[None, ...], bsz, axis=0)
        elif spatial.ndim == 4:
            if spatial.shape[1:] != (h, w, self.spatial_feature_dim):
                raise ValueError(
                    f"{type(self).__name__} spatial feature map shape mismatch: "
                    f"expected (*, {h}, {w}, {self.spatial_feature_dim}), got={spatial.shape}"
                )
            if spatial.shape[0] == 1 and bsz > 1:
                spatial_batch = np.repeat(spatial, bsz, axis=0)
            elif spatial.shape[0] == bsz:
                spatial_batch = spatial
            else:
                raise ValueError(
                    f"{type(self).__name__} spatial batch dimension mismatch: "
                    f"expected 1 or {bsz}, got={spatial.shape[0]}"
                )
        else:
            raise ValueError(
                f"{type(self).__name__} spatial feature map shape mismatch: "
                f"expected rank-3/4 spatial map, got={spatial.shape}"
            )
        return np.concatenate([cond_map, spatial_batch], axis=-1).astype(np.float32)

    def feature_matrix(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        return fmap.reshape(-1, self.feature_dim).astype(np.float32)

    def _forward_raw_torch(
        self,
        cond: np.ndarray,
        *,
        training: bool,
        spatial_features: np.ndarray | None,
    ) -> np.ndarray:
        torch = self.torch
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        self._ensure_net_device()
        xt = torch.from_numpy(np.moveaxis(fmap, -1, 1).astype(np.float32)).to(self.device)
        if training:
            self.net.train()
            yt = self.net(xt)
            self._torch_last_in = xt
            self._torch_last_out = yt
        else:
            self.net.eval()
            with torch.no_grad():
                yt = self.net(xt)
            self._torch_last_in = None
            self._torch_last_out = None
        return yt.detach().cpu().numpy().astype(np.float32)

    def forward_raw(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._forward_raw_torch(cond, training=bool(training), spatial_features=spatial_features)

    def forward_features(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        y = self.forward_raw(cond, training=False, spatial_features=spatial_features)
        out = {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}
        if self.with_rho_eff_head:
            out["rho_eff"] = y[:, self.out_channels : self.out_channels + 1]
        return out

    def forward(self, cond: np.ndarray, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)[:, : self.out_channels]

    def predict_fields(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return self.forward_features(cond, spatial_features=spatial_features)

    def _torch_step_reference(self):
        return self.net.in_proj.weight, self.net.post[-1].weight

    def backward_raw(
        self,
        grad_raw: np.ndarray,
        *,
        lr: float,
        weight_decay: float = 0.0,
        apply_step: bool = True,
        target_raw: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del target_raw, loss_cfg, weight_decay
        if self._torch_last_out is None:
            raise RuntimeError(f"{type(self).__name__}.backward_raw called without torch forward cache")
        torch = self.torch
        grad_t = torch.as_tensor(np.asarray(grad_raw, dtype=np.float32), device=self.device)
        params = [p for p in self.net.parameters() if p.requires_grad]
        if not params:
            return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}
        hidden_w, out_w = self._torch_step_reference()
        with torch.no_grad():
            w_hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
            w_out_prev = out_w.detach().clone() if out_w is not None else None
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        self._torch_last_out.backward(grad_t)
        step_hidden = 0.0
        step_out = 0.0
        if apply_step:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p -= float(lr) * p.grad
                hidden_w_cur, out_w_cur = self._torch_step_reference()
                if hidden_w_cur is not None and w_hidden_prev is not None:
                    dh = hidden_w_cur.detach() - w_hidden_prev
                    step_hidden = float(torch.linalg.norm(dh) / max(float(torch.linalg.norm(w_hidden_prev)), 1.0e-12))
                if out_w_cur is not None and w_out_prev is not None:
                    do = out_w_cur.detach() - w_out_prev
                    step_out = float(torch.linalg.norm(do) / max(float(torch.linalg.norm(w_out_prev)), 1.0e-12))
        self._torch_last_in = None
        self._torch_last_out = None
        return {"step_rel_hidden_mean": float(step_hidden), "step_rel_output": float(step_out)}

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, tensor in self.net.state_dict().items():
            out[f"torch::{name}"] = tensor.detach().cpu().numpy().astype(np.float32)
        return out

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        torch = self.torch
        state_t = {
            k.split("torch::", 1)[1]: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(self.device)
            for k, v in state.items()
            if str(k).startswith("torch::")
        }
        if not state_t:
            raise ValueError(
                f"legacy numpy {type(self).__name__} checkpoints are no longer supported; expected torch::* weights"
            )
        self._ensure_net_device()
        self.net.load_state_dict(state_t, strict=True)
