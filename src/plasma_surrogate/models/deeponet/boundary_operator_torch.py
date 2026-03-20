"""Boundary operator torch module (single primary QoI key)."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.torch_backend import require_torch


class BoundaryOperatorTorch:
    def __init__(
        self,
        primary_qoi_key: str = "Gamma_i",
        w_log_ne: float = 0.08,
        w_te: float = 0.06,
        w_en: float = 0.04,
        bias: float = 0.0,
        clamp: tuple[float, float] | None = None,
        freeze: bool = True,
    ) -> None:
        torch = require_torch()
        self._torch = torch
        self.primary_qoi_key = str(primary_qoi_key)
        self.w_log_ne = torch.nn.Parameter(torch.tensor(float(w_log_ne), dtype=torch.float32))
        self.w_te = torch.nn.Parameter(torch.tensor(float(w_te), dtype=torch.float32))
        self.w_en = torch.nn.Parameter(torch.tensor(float(w_en), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))
        self.clamp = None if clamp is None else (float(clamp[0]), float(clamp[1]))
        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def parameters(self):
        return [self.w_log_ne, self.w_te, self.w_en, self.bias]

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def to_meta(self) -> dict[str, Any]:
        out = {
            "primary_qoi_key": self.primary_qoi_key,
            "w_log_ne": float(self.w_log_ne.detach().cpu().item()),
            "w_te": float(self.w_te.detach().cpu().item()),
            "w_en": float(self.w_en.detach().cpu().item()),
            "bias": float(self.bias.detach().cpu().item()),
            "freeze": bool(self.freeze),
        }
        if self.clamp is not None:
            out["clamp"] = [self.clamp[0], self.clamp[1]]
        return out

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return {
            "w_log_ne": np.array([float(self.w_log_ne.detach().cpu().item())], dtype=np.float32),
            "w_te": np.array([float(self.w_te.detach().cpu().item())], dtype=np.float32),
            "w_en": np.array([float(self.w_en.detach().cpu().item())], dtype=np.float32),
            "bias": np.array([float(self.bias.detach().cpu().item())], dtype=np.float32),
        }

    def load_state_dict_numpy(self, weights: dict[str, np.ndarray]) -> None:
        torch = self._torch
        if "w_log_ne" in weights:
            self.w_log_ne.data.copy_(torch.as_tensor(float(np.asarray(weights["w_log_ne"]).reshape(-1)[0])))
        if "w_te" in weights:
            self.w_te.data.copy_(torch.as_tensor(float(np.asarray(weights["w_te"]).reshape(-1)[0])))
        if "w_en" in weights:
            self.w_en.data.copy_(torch.as_tensor(float(np.asarray(weights["w_en"]).reshape(-1)[0])))
        if "bias" in weights:
            self.bias.data.copy_(torch.as_tensor(float(np.asarray(weights["bias"]).reshape(-1)[0])))

    def _grad_mag(self, phi):
        torch = self._torch
        p = torch.as_tensor(phi, dtype=torch.float32)
        if p.ndim == 3 and p.shape[-1] == 1:
            # Pointwise inputs do not have a local stencil; keep EN neutral.
            return torch.zeros_like(p)
        dy = p[..., 1:, :] - p[..., :-1, :]
        dx = p[..., :, 1:] - p[..., :, :-1]
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 1), mode="replicate")
        dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode="replicate")
        return torch.sqrt(dx * dx + dy * dy + 1e-12)

    def _as_bm1(self, x: Any, sample_idx: Any | None = None):
        torch = self._torch
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 1:
            return t
        if t.ndim == 3:
            t = t[:, None, ...]
        if t.ndim != 4:
            raise ValueError(f"expected [B,1,H,W], [B,H,W], or [B,M,1], got {tuple(t.shape)}")
        if sample_idx is None:
            return t
        idx = torch.as_tensor(sample_idx, dtype=torch.int64, device=t.device).reshape(-1)
        bsz = t.shape[0]
        flat = t.reshape(bsz, 1, -1).permute(0, 2, 1)
        return flat[:, idx, :]

    def predict_target(
        self,
        log_ne: Any,
        te: Any,
        phi: Any,
        cond: Any | None = None,
        geom_ctx: Any | None = None,
        primary_qoi_key: str | None = None,
        sample_idx: Any | None = None,
    ) -> dict[str, Any]:
        del cond, geom_ctx
        torch = self._torch
        key = str(primary_qoi_key or self.primary_qoi_key)
        ln = self._as_bm1(log_ne, sample_idx=sample_idx)
        tt = self._as_bm1(te, sample_idx=sample_idx)
        p_full = self._as_bm1(phi, sample_idx=None)
        if sample_idx is not None:
            p = self._as_bm1(phi, sample_idx=sample_idx)
            e_mag = self._as_bm1(self._grad_mag(p_full), sample_idx=sample_idx)
        else:
            p = p_full
            e_mag = self._grad_mag(p).to(dtype=torch.float32, device=ln.device)
        out = self.w_log_ne * ln + self.w_te * tt + self.w_en * e_mag + self.bias
        if self.clamp is not None:
            out = torch.clamp(out, self.clamp[0], self.clamp[1])
        return {key: out}
