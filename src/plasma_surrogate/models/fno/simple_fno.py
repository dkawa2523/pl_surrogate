"""Minimal torch spectral FNO baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


class FNOBaseline:
    """Lightweight spectral FNO-style model (torch-only)."""

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        n_modes: int = 12,
        head_mlp: dict[str, object] | None = None,
        input_feature_channels: list[str] | None = None,
        spectral_cfg: dict[str, Any] | None = None,
    ):
        del head_mlp  # legacy arg kept for compatibility with existing configs.
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            base = ["ne", "Te", "phi"]
            extra = [f"out_{i}" for i in range(max(0, self.out_channels - len(base)))]
            self.output_keys = (base + extra)[: self.out_channels]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        self.with_rho_eff_head = bool(with_rho_eff_head)
        self.raw_out_channels = self.out_channels + (1 if self.with_rho_eff_head else 0)
        self.backend = "torch"
        self.n_modes = int(max(1, int(n_modes)))

        channels = list(input_feature_channels or ["x", "y"])
        if len(channels) == 0:
            raise ValueError("train.fno.input_features.features must be a non-empty list")
        if len(set(channels)) != len(channels):
            raise ValueError("train.fno.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.feature_dim = int(self.input_dim + self.spatial_feature_dim)
        self.head_arch_version = "spectral_v2"
        self.spectral_cfg = dict(spectral_cfg or {})
        self.dealias_ratio = float(np.clip(float(self.spectral_cfg.get("dealias_ratio", 1.0)), 0.05, 1.0))
        self.taper_alpha = float(max(float(self.spectral_cfg.get("taper_alpha", 0.0)), 0.0))
        self.skip_filter = str(self.spectral_cfg.get("skip_filter", "none")).strip().lower()
        if self.skip_filter not in {"none", "match_spectral"}:
            raise ValueError("train.fno.model_cfg.spectral_cfg.skip_filter must be one of: none, match_spectral")

        h, w = self.grid_shape
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        self.coord = np.stack([xv, yv], axis=-1).astype(np.float32)
        self._static_spatial_features: np.ndarray | None = None

        from plasma_surrogate.core.torch_backend import require_torch

        torch = require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))
        self.torch = torch

        class _SpectralConv2d(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, modes: int, *, dealias_ratio: float, taper_alpha: float):
                super().__init__()
                self.in_ch = int(in_ch)
                self.out_ch = int(out_ch)
                self.modes = int(max(modes, 1))
                self.dealias_ratio = float(np.clip(float(dealias_ratio), 0.05, 1.0))
                self.taper_alpha = float(max(float(taper_alpha), 0.0))
                scale = 1.0 / max(self.in_ch * self.out_ch, 1)
                self.weight_pos = nn.Parameter(scale * torch.randn(self.in_ch, self.out_ch, self.modes, self.modes, 2))
                self.weight_neg = nn.Parameter(scale * torch.randn(self.in_ch, self.out_ch, self.modes, self.modes, 2))
                self._mask_cache: dict[tuple[int, int, str, int], Any] = {}

            def _compl_mul(self, x_ft, w):
                return torch.einsum("bixy,ioxy->boxy", x_ft, w)

            def _spectral_filter(self, *, h: int, w_half: int, m1: int, m2: int, device: Any) -> Any:
                key = (int(h), int(w_half), int(m1), int(m2), str(device.type), int(device.index or -1))
                cached = self._mask_cache.get(key)
                if cached is not None:
                    return cached
                w = int(max((w_half - 1) * 2, 2))
                fy = torch.fft.fftfreq(h, d=1.0, device=device).abs() / 0.5
                fx = torch.fft.rfftfreq(w, d=1.0, device=device) / 0.5
                radial = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
                # NOTE:
                # `dealias_ratio` is interpreted as a ratio inside the retained FNO modes.
                # This avoids the previous near no-op behavior where low `n_modes` made
                # Nyquist-normalized thresholds too loose.
                max_fy = torch.max(fy[: max(int(m1), 1)])
                max_fx = torch.max(fx[: max(int(m2), 1)])
                support_radius = torch.sqrt(max_fy**2 + max_fx**2)
                cutoff = torch.clamp(self.dealias_ratio * support_radius, min=1.0e-6)
                hard = (radial <= cutoff).to(dtype=torch.float32)
                if self.taper_alpha > 0.0:
                    ratio = torch.clamp(radial / cutoff, min=0.0)
                    taper = torch.exp(-self.taper_alpha * ratio**4)
                    mask = hard * taper.to(dtype=torch.float32)
                else:
                    mask = hard
                self._mask_cache[key] = mask
                return mask

            def forward(self, x):
                bsz, _, h, w = x.shape
                x_ft = torch.fft.rfft2(x, norm="ortho")
                out_ft = torch.zeros(
                    (bsz, self.out_ch, h, (w // 2) + 1),
                    dtype=torch.cfloat,
                    device=x.device,
                )
                m1 = int(min(self.modes, h))
                m2 = int(min(self.modes, (w // 2) + 1))
                if m1 <= 0 or m2 <= 0:
                    return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
                w_pos = torch.view_as_complex(self.weight_pos[:, :, :m1, :m2, :].contiguous())
                w_neg = torch.view_as_complex(self.weight_neg[:, :, :m1, :m2, :].contiguous())
                out_ft[:, :, :m1, :m2] = self._compl_mul(x_ft[:, :, :m1, :m2], w_pos)
                out_ft[:, :, -m1:, :m2] = self._compl_mul(x_ft[:, :, -m1:, :m2], w_neg)
                out_ft = self._apply_filter_ft(out_ft, h=h, w=w, m1=m1, m2=m2)
                return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

            def _apply_filter_ft(self, x_ft, *, h: int, w: int, m1: int, m2: int):
                if self.dealias_ratio < 0.999 or self.taper_alpha > 0.0:
                    mask = self._spectral_filter(
                        h=h,
                        w_half=(w // 2) + 1,
                        m1=m1,
                        m2=m2,
                        device=x_ft.device,
                    )
                    x_ft = x_ft * mask[None, None, :, :]
                return x_ft

            def apply_filter_map(self, x):
                bsz, ch, h, w = x.shape
                if bsz <= 0 or ch <= 0:
                    return x
                m1 = int(min(self.modes, h))
                m2 = int(min(self.modes, (w // 2) + 1))
                if m1 <= 0 or m2 <= 0:
                    return x
                x_ft = torch.fft.rfft2(x, norm="ortho")
                x_ft = self._apply_filter_ft(x_ft, h=h, w=w, m1=m1, m2=m2)
                return torch.fft.irfft2(x_ft, s=(h, w), norm="ortho")

        class _SpectralBlock(nn.Module):
            def __init__(
                self,
                width: int,
                modes: int,
                *,
                dealias_ratio: float,
                taper_alpha: float,
                skip_filter: str,
            ):
                super().__init__()
                self.spec = _SpectralConv2d(
                    width,
                    width,
                    modes,
                    dealias_ratio=dealias_ratio,
                    taper_alpha=taper_alpha,
                )
                self.skip = nn.Conv2d(width, width, kernel_size=1)
                self.skip_filter = str(skip_filter).strip().lower()

            def forward(self, x):
                spec_out = self.spec(x)
                skip_out = self.skip(x)
                if self.skip_filter == "match_spectral":
                    skip_out = self.spec.apply_filter_map(skip_out)
                return torch.nn.functional.gelu(spec_out + skip_out)

        class _SpectralFNONet(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                width: int,
                n_layers: int,
                n_modes: int,
                dropout: float,
                dealias_ratio: float,
                taper_alpha: float,
                skip_filter: str,
            ):
                super().__init__()
                self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
                self.blocks = nn.ModuleList(
                    [
                        _SpectralBlock(
                            width,
                            n_modes,
                            dealias_ratio=dealias_ratio,
                            taper_alpha=taper_alpha,
                            skip_filter=skip_filter,
                        )
                        for _ in range(max(int(n_layers), 1))
                    ]
                )
                self.post = nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(float(max(dropout, 0.0))),
                    nn.Conv2d(width, out_channels, kernel_size=1),
                )

            def forward(self, x):
                h = self.in_proj(x)
                for block in self.blocks:
                    h = block(h)
                return self.post(h)

        width = int(max(int(self.spectral_cfg.get("width", 64)), 8))
        n_layers = int(max(int(self.spectral_cfg.get("n_layers", 4)), 1))
        dropout = float(np.clip(float(self.spectral_cfg.get("dropout", 0.0)), 0.0, 0.9))
        self.net = _SpectralFNONet(
            in_channels=int(self.feature_dim),
            out_channels=int(self.raw_out_channels),
            width=width,
            n_layers=n_layers,
            n_modes=int(self.n_modes),
            dropout=dropout,
            dealias_ratio=float(self.dealias_ratio),
            taper_alpha=float(self.taper_alpha),
            skip_filter=str(self.skip_filter),
        )
        self._torch_width = int(width)
        self._torch_layers = int(n_layers)
        self._torch_last_in = None
        self._torch_last_out = None

    def set_static_spatial_features(self, spatial_features: np.ndarray) -> None:
        arr = np.asarray(spatial_features, dtype=np.float32)
        h, w = self.grid_shape
        if arr.shape != (h, w, self.spatial_feature_dim):
            raise ValueError(
                "FNO static spatial features must have shape "
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
        if spatial.shape != (h, w, self.spatial_feature_dim):
            raise ValueError(
                "FNO spatial feature map shape mismatch: "
                f"expected {(h, w, self.spatial_feature_dim)}, got={spatial.shape}"
            )
        spatial_batch = np.repeat(spatial[None, ...], bsz, axis=0)
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
        xt = torch.from_numpy(np.moveaxis(fmap, -1, 1).astype(np.float32))
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
    ) -> dict[str, float]:
        del weight_decay
        if self._torch_last_out is None:
            raise RuntimeError("FNOBaseline.backward_raw called without torch forward cache")
        torch = self.torch
        grad_t = torch.as_tensor(np.asarray(grad_raw, dtype=np.float32))
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
            k.split("torch::", 1)[1]: torch.from_numpy(np.asarray(v, dtype=np.float32))
            for k, v in state.items()
            if str(k).startswith("torch::")
        }
        if not state_t:
            raise ValueError("legacy numpy FNO checkpoints are no longer supported; expected torch::* weights")
        self.net.load_state_dict(state_t, strict=True)
