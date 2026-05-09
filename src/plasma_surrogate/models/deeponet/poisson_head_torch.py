"""DeepONet Poisson head for rho_eff -> phi prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.models._torch_spatial_common import _resolve_torch_device


class DeepONetPoissonHeadTorch:
    def __init__(
        self,
        deeponet_poisson: Any,
        sensor_indices: np.ndarray,
        query_indices: np.ndarray,
        flatten_order: str = "C",
        grid_shape: tuple[int, int] | None = None,
        freeze: bool = True,
    ) -> None:
        self._torch = require_torch()
        self.device = _resolve_torch_device(self._torch)
        self.net = deeponet_poisson
        if hasattr(self.net, "to"):
            self.net.to(self.device)
        elif hasattr(self.net, "device"):
            self.net.device = self.device
            if hasattr(self.net, "_move_modules_to_device"):
                self.net._move_modules_to_device()
        self.sensor_indices = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
        self.query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        self.flatten_order = str(flatten_order)
        self.grid_shape = None if grid_shape is None else (int(grid_shape[0]), int(grid_shape[1]))
        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_cache(
        cls,
        deeponet_poisson: Any,
        sensor_idx: np.ndarray,
        query_idx: np.ndarray,
        flatten_order: str,
        grid_shape: tuple[int, int],
        freeze: bool = True,
    ) -> "DeepONetPoissonHeadTorch":
        return cls(
            deeponet_poisson=deeponet_poisson,
            sensor_indices=np.asarray(sensor_idx, dtype=np.int64),
            query_indices=np.asarray(query_idx, dtype=np.int64),
            flatten_order=flatten_order,
            grid_shape=grid_shape,
            freeze=freeze,
        )

    def parameters(self):
        return self.net.parameters()

    def to(self, device):
        self.device = device
        if hasattr(self.net, "to"):
            self.net.to(device)
        elif hasattr(self.net, "device"):
            self.net.device = device
            if hasattr(self.net, "_move_modules_to_device"):
                self.net._move_modules_to_device()
        return self

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def to_meta(self) -> dict[str, Any]:
        return {
            "sensor_indices": [int(v) for v in self.sensor_indices.tolist()],
            "query_indices": [int(v) for v in self.query_indices.tolist()],
            "flatten_order": self.flatten_order,
            "grid_shape": list(self.grid_shape) if self.grid_shape is not None else None,
            "freeze": bool(self.freeze),
        }

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return self.net.state_dict_numpy()

    def load_state_dict_numpy(self, weights: dict[str, np.ndarray]) -> None:
        self.net.load_state_dict_numpy(weights)

    def _jacobi_refine(self, phi, rho_eff, mask_active, bc_mask, bc_val, iters: int):
        torch = self._torch
        if int(iters) <= 0:
            return phi
        out = phi
        for _ in range(int(iters)):
            p = torch.nn.functional.pad(out, (1, 1, 1, 1), mode="replicate")
            up = p[:, :, :-2, 1:-1]
            dn = p[:, :, 2:, 1:-1]
            lf = p[:, :, 1:-1, :-2]
            rg = p[:, :, 1:-1, 2:]
            nxt = 0.25 * (up + dn + lf + rg + rho_eff)
            out = mask_active * ((1.0 - bc_mask) * nxt + bc_mask * bc_val)
        return out

    def predict_phi(
        self,
        rho_eff: Any,
        cond_vec: Any,
        geom_ctx: Any,
        refine_iters: int = 0,
    ):
        torch = self._torch
        rho = torch.as_tensor(rho_eff, dtype=torch.float32, device=self.device)
        if rho.ndim == 3:
            rho = rho[:, None, ...]
        cond_t = torch.as_tensor(cond_vec, dtype=torch.float32, device=self.device)
        if cond_t.ndim == 1:
            cond_t = cond_t[None, :]
        if rho.shape[0] == 1 and cond_t.shape[0] > 1:
            rho = rho.expand(cond_t.shape[0], -1, -1, -1)
        bsz, _, h, w = rho.shape
        if self.grid_shape is not None and (h, w) != self.grid_shape:
            raise ValueError(f"grid_shape mismatch: expected={self.grid_shape}, actual={(h, w)}")

        coord_flat = torch.as_tensor(
            np.asarray(geom_ctx.coord_grid, dtype=np.float32).reshape(2, -1).T,
            dtype=torch.float32,
            device=self.device,
        )
        rho_flat = rho.reshape(bsz, 1, -1).permute(0, 2, 1)

        s_idx = torch.as_tensor(self.sensor_indices, dtype=torch.int64, device=self.device)
        q_idx = torch.as_tensor(self.query_indices, dtype=torch.int64, device=self.device)
        x_s = coord_flat[s_idx][None, ...].expand(bsz, -1, -1)
        v_s = rho_flat[:, s_idx, :]
        x_q = coord_flat[q_idx][None, ...].expand(bsz, -1, -1)
        pred = self.net.forward(sensors={"x": x_s, "v": v_s}, query={"x": x_q}, cond=cond_t)
        if "phi" not in pred:
            first_key = next(iter(pred.keys()))
            phi_q = pred[first_key]
        else:
            phi_q = pred["phi"]

        phi_full = torch.zeros((bsz, h * w, 1), dtype=torch.float32, device=self.device)
        phi_full[:, q_idx, :] = phi_q
        phi = phi_full.permute(0, 2, 1).reshape(bsz, 1, h, w)

        active = torch.as_tensor(
            np.asarray(geom_ctx.mask_plasma, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )[None, None, ...]
        bc_mask_np = np.zeros_like(np.asarray(geom_ctx.mask_plasma, dtype=np.float32))
        bc_val_np = np.zeros_like(np.asarray(geom_ctx.mask_plasma, dtype=np.float32))
        if geom_ctx.bc_dir_mask is not None:
            bc_mask_np = np.asarray(geom_ctx.bc_dir_mask, dtype=np.float32)
        if geom_ctx.bc_dir_value is not None:
            bc_val_np = np.asarray(geom_ctx.bc_dir_value, dtype=np.float32)
        bc_mask = torch.as_tensor(bc_mask_np, dtype=torch.float32, device=self.device)[None, None, ...]
        bc_val = torch.as_tensor(bc_val_np, dtype=torch.float32, device=self.device)[None, None, ...]
        phi = active * ((1.0 - bc_mask) * phi + bc_mask * bc_val)
        phi = self._jacobi_refine(
            phi=phi,
            rho_eff=rho,
            mask_active=active,
            bc_mask=bc_mask,
            bc_val=bc_val,
            iters=int(refine_iters),
        )
        return phi
