"""Factorized FNO baseline with separable 1D spectral blocks."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.fno._spectral_cfg import normalize_common_spectral_cfg
from plasma_surrogate.models.fno._torch_grid_base import _TorchGridFieldBaseline
from plasma_surrogate.models.heads.role_grouped import (
    build_role_grouped_conv2d_head,
    is_grouped_output_head_mode,
)


class FFNOBaseline(_TorchGridFieldBaseline):
    """Lightweight factorized FNO-style model (torch-only)."""

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
        output_heads: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
    ):
        normalized_spectral_cfg, spectral_common = normalize_common_spectral_cfg(
            spectral_cfg,
            cfg_prefix="train.ffno",
        )
        factorized_cfg = dict(normalized_spectral_cfg.get("factorized_cfg", {}))
        factorized_enabled = bool(factorized_cfg.get("enabled", True))
        if not factorized_enabled:
            raise ValueError("train.ffno.model_cfg.spectral_cfg.factorized_cfg.enabled must be true for ffno v1")
        factorized_mode = str(factorized_cfg.get("mode", "separable_1d")).strip().lower()
        if factorized_mode != "separable_1d":
            raise ValueError("train.ffno.model_cfg.spectral_cfg.factorized_cfg.mode must be separable_1d")
        share_weights = bool(factorized_cfg.get("share_weights", False))
        if share_weights:
            raise ValueError("train.ffno.model_cfg.spectral_cfg.factorized_cfg.share_weights must be false for ffno v1")
        local_skip_cfg = dict(normalized_spectral_cfg.get("local_skip_cfg", {}))
        if "kernel_size" in local_skip_cfg and int(local_skip_cfg.get("kernel_size", 3)) != 3:
            raise ValueError("train.ffno.model_cfg.spectral_cfg.local_skip_cfg.kernel_size must be 3")
        local_skip_enabled = bool(local_skip_cfg.get("enabled", False))
        local_skip_init_scale = float(local_skip_cfg.get("init_scale", 0.0))
        axis_mix_cfg = dict(normalized_spectral_cfg.get("axis_mix_cfg", {}))
        axis_mix_enabled = bool(axis_mix_cfg.get("enabled", False))
        axis_mix_init_h = float(axis_mix_cfg.get("init_h", 1.0))
        axis_mix_init_w = float(axis_mix_cfg.get("init_w", 1.0))
        normalized_spectral_cfg["factorized_cfg"] = {
            "enabled": True,
            "mode": "separable_1d",
            "share_weights": False,
        }
        normalized_spectral_cfg["local_skip_cfg"] = {
            "enabled": bool(local_skip_enabled),
            "init_scale": float(local_skip_init_scale),
        }
        normalized_spectral_cfg["axis_mix_cfg"] = {
            "enabled": bool(axis_mix_enabled),
            "init_h": float(axis_mix_init_h),
            "init_w": float(axis_mix_init_w),
        }
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
            cfg_prefix="train.ffno",
            default_output_keys=[],
            impl_version=(
                "factorized_separable_1d_v2_local_skip" if local_skip_enabled else "factorized_separable_1d_v1"
            ),
            head_arch_version="linear_v1",
            output_heads=output_heads,
            target_role_schema=target_role_schema,
        )
        self.n_modes = int(max(1, int(n_modes)))
        self.spectral_cfg = dict(normalized_spectral_cfg)
        self.dealias_ratio = float(spectral_common["dealias_ratio"])
        self.taper_alpha = float(spectral_common["taper_alpha"])
        self.skip_filter = str(spectral_common["skip_filter"])
        self.factorized_cfg = dict(normalized_spectral_cfg["factorized_cfg"])
        self.factorized_enabled = bool(factorized_enabled)
        self.factorized_mode = str(factorized_mode)
        self.share_weights = bool(share_weights)
        self.local_skip_cfg = dict(normalized_spectral_cfg["local_skip_cfg"])
        self.local_skip_enabled = bool(local_skip_enabled)
        self.local_skip_init_scale = float(local_skip_init_scale)
        self.axis_mix_cfg = dict(normalized_spectral_cfg["axis_mix_cfg"])
        self.axis_mix_enabled = bool(axis_mix_enabled)
        self.axis_mix_init_h = float(axis_mix_init_h)
        self.axis_mix_init_w = float(axis_mix_init_w)
        torch = self.torch
        nn = torch.nn
        output_head_group_options = dict(getattr(self, "output_head_group_options", {}) or {})

        class _FactorizedSpectralConv2d(nn.Module):
            def __init__(
                self,
                in_ch: int,
                out_ch: int,
                modes: int,
                *,
                dealias_ratio: float,
                taper_alpha: float,
            ):
                super().__init__()
                self.in_ch = int(in_ch)
                self.out_ch = int(out_ch)
                self.modes = int(max(modes, 1))
                self.dealias_ratio = float(np.clip(float(dealias_ratio), 0.05, 1.0))
                self.taper_alpha = float(max(float(taper_alpha), 0.0))
                scale = 1.0 / max(self.in_ch * self.out_ch, 1)
                self.weight_h = nn.Parameter(scale * torch.randn(self.in_ch, self.out_ch, self.modes, 2))
                self.weight_w = nn.Parameter(scale * torch.randn(self.in_ch, self.out_ch, self.modes, 2))
                self._mask_cache: dict[tuple[str, int, int, str, int], Any] = {}

            def _spectral_filter_1d(self, *, axis: str, size: int, modes: int, device: Any) -> Any:
                key = (str(axis), int(size), int(modes), str(device.type), int(device.index or -1))
                cached = self._mask_cache.get(key)
                if cached is not None:
                    return cached
                freq = torch.fft.rfftfreq(size, d=1.0, device=device) / 0.5
                max_freq = torch.max(freq[: max(int(modes), 1)])
                cutoff = torch.clamp(self.dealias_ratio * max_freq, min=1.0e-6)
                hard = (freq <= cutoff).to(dtype=torch.float32)
                if self.taper_alpha > 0.0:
                    ratio = torch.clamp(freq / cutoff, min=0.0)
                    taper = torch.exp(-self.taper_alpha * ratio**4)
                    mask = hard * taper.to(dtype=torch.float32)
                else:
                    mask = hard
                self._mask_cache[key] = mask
                return mask

            def _forward_axis(self, x, *, axis: str):
                if axis == "h":
                    size = int(x.shape[-2])
                    x_ft = torch.fft.rfft(x, dim=-2, norm="ortho")
                    out_ft = torch.zeros(
                        (x.shape[0], self.out_ch, (size // 2) + 1, x.shape[-1]),
                        dtype=torch.cfloat,
                        device=x.device,
                    )
                    modes = int(min(self.modes, (size // 2) + 1))
                    if modes <= 0:
                        return torch.fft.irfft(out_ft, n=size, dim=-2, norm="ortho")
                    w_axis = torch.view_as_complex(self.weight_h[:, :, :modes, :].contiguous())
                    out_ft[:, :, :modes, :] = torch.einsum("bimw,iom->bomw", x_ft[:, :, :modes, :], w_axis)
                    if self.dealias_ratio < 0.999 or self.taper_alpha > 0.0:
                        mask = self._spectral_filter_1d(axis="h", size=size, modes=modes, device=x.device)
                        out_ft = out_ft * mask[None, None, :, None]
                    return torch.fft.irfft(out_ft, n=size, dim=-2, norm="ortho")
                size = int(x.shape[-1])
                x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
                out_ft = torch.zeros(
                    (x.shape[0], self.out_ch, x.shape[-2], (size // 2) + 1),
                    dtype=torch.cfloat,
                    device=x.device,
                )
                modes = int(min(self.modes, (size // 2) + 1))
                if modes <= 0:
                    return torch.fft.irfft(out_ft, n=size, dim=-1, norm="ortho")
                w_axis = torch.view_as_complex(self.weight_w[:, :, :modes, :].contiguous())
                out_ft[:, :, :, :modes] = torch.einsum("bihm,iom->bohm", x_ft[:, :, :, :modes], w_axis)
                if self.dealias_ratio < 0.999 or self.taper_alpha > 0.0:
                    mask = self._spectral_filter_1d(axis="w", size=size, modes=modes, device=x.device)
                    out_ft = out_ft * mask[None, None, None, :]
                return torch.fft.irfft(out_ft, n=size, dim=-1, norm="ortho")

            def _apply_filter_axis(self, x, *, axis: str):
                if axis == "h":
                    size = int(x.shape[-2])
                    x_ft = torch.fft.rfft(x, dim=-2, norm="ortho")
                    modes = int(min(self.modes, (size // 2) + 1))
                    if modes <= 0 or (self.dealias_ratio >= 0.999 and self.taper_alpha <= 0.0):
                        return x
                    mask = self._spectral_filter_1d(axis="h", size=size, modes=modes, device=x.device)
                    x_ft = x_ft * mask[None, None, :, None]
                    return torch.fft.irfft(x_ft, n=size, dim=-2, norm="ortho")
                size = int(x.shape[-1])
                x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
                modes = int(min(self.modes, (size // 2) + 1))
                if modes <= 0 or (self.dealias_ratio >= 0.999 and self.taper_alpha <= 0.0):
                    return x
                mask = self._spectral_filter_1d(axis="w", size=size, modes=modes, device=x.device)
                x_ft = x_ft * mask[None, None, None, :]
                return torch.fft.irfft(x_ft, n=size, dim=-1, norm="ortho")

            def forward_components(self, x):
                return self._forward_axis(x, axis="h"), self._forward_axis(x, axis="w")

            def forward(self, x):
                h_out, w_out = self.forward_components(x)
                return h_out + w_out

            def apply_filter_map(self, x):
                out = self._apply_filter_axis(x, axis="h")
                out = self._apply_filter_axis(out, axis="w")
                return out

        class _FactorizedSpectralBlock(nn.Module):
            def __init__(
                self,
                width: int,
                modes: int,
                *,
                dealias_ratio: float,
                taper_alpha: float,
                skip_filter: str,
                local_skip_enabled: bool,
                local_skip_init_scale: float,
                axis_mix_enabled: bool,
                axis_mix_init_h: float,
                axis_mix_init_w: float,
            ):
                super().__init__()
                self.spec = _FactorizedSpectralConv2d(
                    width,
                    width,
                    modes,
                    dealias_ratio=dealias_ratio,
                    taper_alpha=taper_alpha,
                )
                self.skip = nn.Conv2d(width, width, kernel_size=1)
                self.skip_filter = str(skip_filter).strip().lower()
                self.beta_h = nn.Parameter(torch.tensor(float(axis_mix_init_h), dtype=torch.float32))
                self.beta_w = nn.Parameter(torch.tensor(float(axis_mix_init_w), dtype=torch.float32))
                self.axis_mix_gate = float(1.0 if axis_mix_enabled else 0.0)
                self.local_skip_enabled = bool(local_skip_enabled)
                if self.local_skip_enabled:
                    self.local_skip = nn.Conv2d(
                        width,
                        width,
                        kernel_size=3,
                        padding=1,
                        groups=width,
                        bias=True,
                    )
                    self.local_skip_alpha = nn.Parameter(torch.tensor(float(local_skip_init_scale), dtype=torch.float32))
                else:
                    self.local_skip = None
                    self.local_skip_alpha = None

            def forward(self, x):
                spec_h, spec_w = self.spec.forward_components(x)
                skip_out = self.skip(x)
                if self.skip_filter == "match_spectral":
                    skip_out = self.spec.apply_filter_map(skip_out)
                spec_out = spec_h + spec_w
                if self.axis_mix_gate > 0.0:
                    spec_out = (self.beta_h * spec_h) + (self.beta_w * spec_w)
                out = spec_out + skip_out
                if self.local_skip_enabled and self.local_skip is not None and self.local_skip_alpha is not None:
                    out = out + self.local_skip_alpha * self.local_skip(skip_out)
                return torch.nn.functional.gelu(out)

        class _FactorizedFNONet(nn.Module):
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
                local_skip_enabled: bool,
                local_skip_init_scale: float,
                axis_mix_enabled: bool,
                axis_mix_init_h: float,
                axis_mix_init_w: float,
                head_mode: str,
                output_keys: list[str],
                target_groups: dict[str, Any],
                with_rho_eff_head: bool,
            ):
                super().__init__()
                self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
                self.blocks = nn.ModuleList(
                    [
                        _FactorizedSpectralBlock(
                            width,
                            n_modes,
                            dealias_ratio=dealias_ratio,
                            taper_alpha=taper_alpha,
                            skip_filter=skip_filter,
                            local_skip_enabled=local_skip_enabled,
                            local_skip_init_scale=local_skip_init_scale,
                            axis_mix_enabled=axis_mix_enabled,
                            axis_mix_init_h=axis_mix_init_h,
                            axis_mix_init_w=axis_mix_init_w,
                        )
                        for _ in range(max(int(n_layers), 1))
                    ]
                )
                if is_grouped_output_head_mode(head_mode):
                    self.post = nn.Sequential(
                        nn.Conv2d(width, width, kernel_size=1),
                        nn.GELU(),
                        nn.Dropout(float(max(dropout, 0.0))),
                    )
                    self.head = build_role_grouped_conv2d_head(
                        torch=torch,
                        in_channels=width,
                        output_keys=output_keys,
                        target_groups=target_groups,
                        group_options=output_head_group_options,
                        with_rho_eff_head=with_rho_eff_head,
                    )
                else:
                    self.post = nn.Sequential(
                        nn.Conv2d(width, width, kernel_size=1),
                        nn.GELU(),
                        nn.Dropout(float(max(dropout, 0.0))),
                        nn.Conv2d(width, out_channels, kernel_size=1),
                    )
                    self.head = None

            def forward(self, x):
                h = self.in_proj(x)
                for block in self.blocks:
                    h = block(h)
                h = self.post(h)
                if self.head is not None:
                    return self.head(h)
                return h

        width = int(spectral_common["width"])
        n_layers = int(spectral_common["n_layers"])
        dropout = float(spectral_common["dropout"])
        self.net = _FactorizedFNONet(
            in_channels=int(self.feature_dim),
            out_channels=int(self.raw_out_channels),
            width=width,
            n_layers=n_layers,
            n_modes=int(self.n_modes),
            dropout=dropout,
            dealias_ratio=float(self.dealias_ratio),
            taper_alpha=float(self.taper_alpha),
            skip_filter=str(self.skip_filter),
            local_skip_enabled=bool(self.local_skip_enabled),
            local_skip_init_scale=float(self.local_skip_init_scale),
            axis_mix_enabled=bool(self.axis_mix_enabled),
            axis_mix_init_h=float(self.axis_mix_init_h),
            axis_mix_init_w=float(self.axis_mix_init_w),
            head_mode=str(self.output_heads_mode),
            output_keys=list(self.output_keys),
            target_groups=dict(self.target_groups),
            with_rho_eff_head=bool(self.with_rho_eff_head),
        )
        self._torch_width = int(width)
        self._torch_layers = int(n_layers)

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        torch = self.torch
        state_t = {
            k.split("torch::", 1)[1]: torch.from_numpy(np.asarray(v, dtype=np.float32))
            for k, v in state.items()
            if str(k).startswith("torch::")
        }
        if not state_t:
            raise ValueError("legacy numpy FFNO checkpoints are no longer supported; expected torch::* weights")
        missing, unexpected = self.net.load_state_dict(state_t, strict=False)
        if unexpected:
            raise ValueError(f"unexpected FFNO checkpoint keys: {unexpected}")
        allowed_missing = {k for k in missing if str(k).endswith(".beta_h") or str(k).endswith(".beta_w")}
        disallowed_missing = [k for k in missing if k not in allowed_missing]
        if disallowed_missing:
            raise ValueError(f"missing required FFNO checkpoint keys: {disallowed_missing}")
