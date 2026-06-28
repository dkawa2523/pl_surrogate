"""Geometry-aware DeepONet + SIREN trunk baseline (torch-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import (
    _build_unit_coord_grid,
    _load_state_dict_numpy_torch,
    _resolve_batched_spatial_features,
    _resolve_torch_device,
    _state_dict_numpy_torch,
    _validate_static_spatial_features,
)


def normalize_geom_deeponet_siren_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    latent_dim = int(cfg.get("latent_dim", 48))
    trunk_hidden = int(cfg.get("trunk_hidden", 64))
    trunk_layers = int(cfg.get("trunk_layers", 3))
    branch_hidden = int(cfg.get("branch_hidden", 128))
    branch_layers = int(cfg.get("branch_layers", 2))
    dropout = float(cfg.get("dropout", 0.0))
    trunk_w0 = float(cfg.get("trunk_w0", 30.0))
    if latent_dim < 1:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.latent_dim must be >= 1")
    if trunk_hidden < 1:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.trunk_hidden must be >= 1")
    if trunk_layers < 1:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.trunk_layers must be >= 1")
    if branch_hidden < 1:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.branch_hidden must be >= 1")
    if branch_layers < 1:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.branch_layers must be >= 1")
    if not np.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.dropout must be in [0, 1)")
    if not np.isfinite(trunk_w0) or trunk_w0 <= 0.0:
        raise ValueError("train.geom_deeponet_siren.model_cfg.geom_deeponet_siren_cfg.trunk_w0 must be > 0")
    return {
        "latent_dim": int(latent_dim),
        "trunk_hidden": int(trunk_hidden),
        "trunk_layers": int(trunk_layers),
        "branch_hidden": int(branch_hidden),
        "branch_layers": int(branch_layers),
        "dropout": float(dropout),
        "trunk_w0": float(trunk_w0),
    }


class GeomDeepONetSIREN:
    """Experimental DeepONet variant with SIREN trunk over geometry features."""

    model_type = "geom_deeponet_siren"
    geom_deeponet_siren_impl_version = "geom_deeponet_siren_v1"

    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        input_feature_channels: list[str] | None = None,
        geom_deeponet_siren_cfg: dict[str, Any] | None = None,
        backend: str = "torch",
    ) -> None:
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            self.output_keys = [f"target_{i}" for i in range(self.out_channels)]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        self.with_rho_eff_head = bool(with_rho_eff_head)
        self.raw_out_channels = int(self.out_channels + (1 if self.with_rho_eff_head else 0))
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError("train.geom_deeponet_siren.model_cfg.backend must be torch")

        channels = [str(v) for v in list(input_feature_channels or ["x", "y"])]
        if len(channels) == 0:
            raise ValueError("train.geom_deeponet_siren.input_features.features must be non-empty")
        if len(set(channels)) != len(channels):
            raise ValueError("train.geom_deeponet_siren.input_features.features must not contain duplicates")
        self.input_feature_channels = channels
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.geom_deeponet_siren_cfg = normalize_geom_deeponet_siren_cfg(geom_deeponet_siren_cfg)
        self.coord_grid = _build_unit_coord_grid(self.grid_shape)
        self._static_spatial_features: np.ndarray | None = None
        self._torch_last_out = None
        self._torch_last_coeff = None

        from plasma_surrogate.core.torch_backend import require_torch

        self.torch = require_torch()
        self.device = _resolve_torch_device(self.torch)
        self.torch.manual_seed(int(seed))
        nn = self.torch.nn

        latent_dim = int(self.geom_deeponet_siren_cfg["latent_dim"])
        trunk_hidden = int(self.geom_deeponet_siren_cfg["trunk_hidden"])
        trunk_layers = int(self.geom_deeponet_siren_cfg["trunk_layers"])
        branch_hidden = int(self.geom_deeponet_siren_cfg["branch_hidden"])
        branch_layers = int(self.geom_deeponet_siren_cfg["branch_layers"])
        dropout = float(self.geom_deeponet_siren_cfg["dropout"])
        trunk_w0 = float(self.geom_deeponet_siren_cfg["trunk_w0"])

        class _Sine(nn.Module):
            def __init__(self, *, w0: float) -> None:
                super().__init__()
                self.w0 = float(w0)

            def forward(self, x):
                return self.torch.sin(self.w0 * x)

            @property
            def torch(self):
                import torch as _torch

                return _torch

        class _SirenMLP(nn.Module):
            def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, w0: float) -> None:
                super().__init__()
                layers: list[Any] = []
                dims = [int(in_dim)] + [int(hidden_dim)] * int(max(int(n_layers), 1))
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    layers.append(_Sine(w0=w0 if i == 0 else 1.0))
                layers.append(nn.Linear(dims[-1], int(out_dim)))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        class _BranchMLP(nn.Module):
            def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> None:
                super().__init__()
                layers: list[Any] = []
                dims = [int(in_dim)] + [int(hidden_dim)] * int(max(int(n_layers), 1))
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    layers.append(nn.GELU())
                    if float(dropout) > 0.0:
                        layers.append(nn.Dropout(float(dropout)))
                layers.append(nn.Linear(dims[-1], int(out_dim)))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        self.net = nn.Module()
        self.net.trunk = _SirenMLP(
            in_dim=int(self.spatial_feature_dim),
            hidden_dim=int(trunk_hidden),
            out_dim=int(latent_dim),
            n_layers=int(trunk_layers),
            w0=float(trunk_w0),
        )
        self.net.branch = _BranchMLP(
            in_dim=int(self.input_dim),
            hidden_dim=int(branch_hidden),
            out_dim=int(self.raw_out_channels * latent_dim),
            n_layers=int(branch_layers),
            dropout=float(dropout),
        )
        self.net.out_bias = nn.Parameter(self.torch.zeros((self.raw_out_channels,), dtype=self.torch.float32))
        self.net.to(self.device)

    def set_static_spatial_features(self, spatial_features: np.ndarray) -> None:
        arr = _validate_static_spatial_features(
            np.asarray(spatial_features, dtype=np.float32),
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=type(self).__name__,
        )
        self._static_spatial_features = arr.astype(np.float32, copy=True)

    def _resolve_spatial_batch(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return _resolve_batched_spatial_features(
            cond=np.asarray(cond, dtype=np.float32),
            spatial_features=np.asarray(spatial_features, dtype=np.float32) if spatial_features is not None else None,
            static_spatial_features=self._static_spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=type(self).__name__,
            explicit_requirement_message=(
                f"{type(self).__name__} requires geom_feature_pack spatial features; "
                "call set_static_spatial_features(...) or pass spatial_features to predict/forward"
            ),
        )

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool, spatial_features: np.ndarray | None) -> np.ndarray:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        if cond_arr.shape[1] != self.input_dim:
            raise ValueError(
                f"{type(self).__name__} cond dim mismatch: expected={self.input_dim}, got={cond_arr.shape[1]}"
            )
        spatial_batch = self._resolve_spatial_batch(cond_arr, spatial_features=spatial_features)
        bsz = int(cond_arr.shape[0])
        h, w = self.grid_shape
        npts = int(h * w)
        latent_dim = int(self.geom_deeponet_siren_cfg["latent_dim"])

        device = next(self.net.parameters()).device
        cond_t = self.torch.as_tensor(cond_arr, dtype=self.torch.float32, device=device)
        spatial_t = self.torch.as_tensor(spatial_batch, dtype=self.torch.float32, device=device)
        trunk_in = spatial_t.reshape(bsz * npts, self.spatial_feature_dim)

        if bool(training):
            self.net.train()
            branch_coeff = self.net.branch(cond_t).reshape(bsz, self.raw_out_channels, latent_dim)
            trunk_latent = self.net.trunk(trunk_in).reshape(bsz, npts, latent_dim)
            out = self.torch.einsum("bol,bnl->bon", branch_coeff, trunk_latent)
            out = out + self.net.out_bias.reshape(1, self.raw_out_channels, 1)
            out = out.reshape(bsz, self.raw_out_channels, h, w)
            self._torch_last_coeff = branch_coeff
            self._torch_last_out = out
            return out.detach().cpu().numpy().astype(np.float32)

        self.net.eval()
        with self.torch.no_grad():
            branch_coeff = self.net.branch(cond_t).reshape(bsz, self.raw_out_channels, latent_dim)
            trunk_latent = self.net.trunk(trunk_in).reshape(bsz, npts, latent_dim)
            out = self.torch.einsum("bol,bnl->bon", branch_coeff, trunk_latent)
            out = out + self.net.out_bias.reshape(1, self.raw_out_channels, 1)
            out = out.reshape(bsz, self.raw_out_channels, h, w)
        self._torch_last_coeff = None
        self._torch_last_out = None
        return out.detach().cpu().numpy().astype(np.float32)

    def forward_raw(self, cond: np.ndarray, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return self._forward_raw_torch(cond, training=bool(training), spatial_features=spatial_features)

    def forward(self, cond: np.ndarray, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)[:, : self.out_channels]

    def forward_features(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        y = self.forward_raw(cond, training=False, spatial_features=spatial_features)
        out = {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}
        if self.with_rho_eff_head:
            out["rho_eff"] = y[:, self.out_channels : self.out_channels + 1]
        return out

    def predict_fields(
        self,
        cond: np.ndarray,
        spatial_features: np.ndarray | None = None,
        geom_ctx: Any | None = None,
    ) -> dict[str, np.ndarray]:
        del geom_ctx
        return self.forward_features(cond, spatial_features=spatial_features)

    def _torch_step_reference(self):
        return self.net.branch.net[0].weight, self.net.trunk.net[-1].weight

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
        del weight_decay, target_raw, loss_cfg
        if self._torch_last_out is None:
            raise RuntimeError(f"{type(self).__name__}.backward_raw called without forward cache")
        grad_np = np.asarray(grad_raw, dtype=np.float32)
        grad_t = self.torch.as_tensor(
            grad_np,
            dtype=getattr(self._torch_last_out, "dtype", None) or self.torch.float32,
            device=getattr(self._torch_last_out, "device", None),
        )
        params = [p for p in self.net.parameters() if p.requires_grad]
        if not params:
            self._torch_last_out = None
            self._torch_last_coeff = None
            return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}
        hidden_w, out_w = self._torch_step_reference()
        with self.torch.no_grad():
            w_hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
            w_out_prev = out_w.detach().clone() if out_w is not None else None
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        self._torch_last_out.backward(grad_t)
        step_hidden = 0.0
        step_out = 0.0
        if bool(apply_step):
            with self.torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p -= float(lr) * p.grad
                hidden_cur, out_cur = self._torch_step_reference()
                if hidden_cur is not None and w_hidden_prev is not None:
                    dh = hidden_cur.detach() - w_hidden_prev
                    step_hidden = float(
                        self.torch.linalg.norm(dh) / max(float(self.torch.linalg.norm(w_hidden_prev)), 1.0e-12)
                    )
                if out_cur is not None and w_out_prev is not None:
                    do = out_cur.detach() - w_out_prev
                    step_out = float(
                        self.torch.linalg.norm(do) / max(float(self.torch.linalg.norm(w_out_prev)), 1.0e-12)
                    )
        self._torch_last_out = None
        self._torch_last_coeff = None
        return {"step_rel_hidden_mean": float(step_hidden), "step_rel_output": float(step_out)}

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return _state_dict_numpy_torch(self.net)

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message="GeomDeepONetSIREN state dict does not contain expected torch::* weights",
            device=self.device,
        )
        self.net.to(self.device)

    def to_meta(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "geom_deeponet_siren_impl_version": self.geom_deeponet_siren_impl_version,
            "input_dim": int(self.input_dim),
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "out_channels": int(self.out_channels),
            "output_keys": list(self.output_keys),
            "with_rho_eff_head": bool(self.with_rho_eff_head),
            "backend": str(self.backend),
            "input_feature_channels": list(self.input_feature_channels),
            "geom_deeponet_siren_cfg": dict(self.geom_deeponet_siren_cfg),
        }


__all__ = ["GeomDeepONetSIREN", "normalize_geom_deeponet_siren_cfg"]
