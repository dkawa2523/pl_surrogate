"""Minimal torch spectral FNO baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.fno._spectral_cfg import normalize_common_spectral_cfg
from plasma_surrogate.models.fno._torch_grid_base import _TorchGridFieldBaseline


class FNOBaseline(_TorchGridFieldBaseline):
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
        backend: str = "torch",
    ):
        super().__init__(
            input_dim=input_dim,
            grid_shape=grid_shape,
            out_channels=out_channels,
            output_keys=output_keys,
            seed=seed,
            with_rho_eff_head=with_rho_eff_head,
            head_mlp=head_mlp,
            input_feature_channels=input_feature_channels,
            backend=backend,
            cfg_prefix="train.fno",
            default_output_keys=["ne", "Te", "phi"],
            impl_version="spectral_v2",
            head_arch_version="linear_v1",
        )
        self.n_modes = int(max(1, int(n_modes)))
        self.spectral_cfg, spectral_common = normalize_common_spectral_cfg(
            spectral_cfg,
            cfg_prefix="train.fno",
        )
        self.dealias_ratio = float(spectral_common["dealias_ratio"])
        self.taper_alpha = float(spectral_common["taper_alpha"])
        self.skip_filter = str(spectral_common["skip_filter"])
        torch = self.torch
        nn = torch.nn

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

        width = int(spectral_common["width"])
        n_layers = int(spectral_common["n_layers"])
        dropout = float(spectral_common["dropout"])
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
