"""Minimal UNO-lite torch grid model."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.fno._torch_grid_base import _TorchGridFieldBaseline
from plasma_surrogate.models.heads.role_grouped import (
    build_role_grouped_conv2d_head,
    is_grouped_output_head_mode,
)


def normalize_uno_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    width = int(cfg.get("width", 64))
    n_layers = int(cfg.get("n_layers", 4))
    dropout = float(cfg.get("dropout", 0.0))
    if width < 1:
        raise ValueError("train.u_no.model_cfg.uno_cfg.width must be >= 1")
    if n_layers < 1:
        raise ValueError("train.u_no.model_cfg.uno_cfg.n_layers must be >= 1")
    if not np.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise ValueError("train.u_no.model_cfg.uno_cfg.dropout must be finite and in [0, 1)")
    return {
        "width": int(width),
        "n_layers": int(n_layers),
        "dropout": float(dropout),
    }


def normalize_uno_n_modes(raw_n_modes: Any) -> int:
    n_modes = int(raw_n_modes)
    if n_modes < 1:
        raise ValueError("train.u_no.model_cfg.n_modes must be >= 1")
    return n_modes


class UNOBaseline(_TorchGridFieldBaseline):
    """UNO-lite model with low-frequency global mixing + local convolution."""

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
        uno_cfg: dict[str, Any] | None = None,
        backend: str = "torch",
        output_heads: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
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
            cfg_prefix="train.u_no",
            default_output_keys=[],
            impl_version="uno_lite_v1",
            head_arch_version="linear_v1",
            output_heads=output_heads,
            target_role_schema=target_role_schema,
        )
        self.n_modes = normalize_uno_n_modes(n_modes)
        self.uno_cfg = normalize_uno_cfg(uno_cfg)
        width = int(self.uno_cfg["width"])
        n_layers = int(self.uno_cfg["n_layers"])
        dropout = float(self.uno_cfg["dropout"])

        torch = self.torch
        nn = torch.nn

        class _LowFreqMix2d(nn.Module):
            def __init__(self, channels: int, n_modes: int) -> None:
                super().__init__()
                self.channels = int(channels)
                self.n_modes = normalize_uno_n_modes(n_modes)
                scale = 1.0 / max(self.channels, 1)
                self.weight_pos = nn.Parameter(
                    scale * torch.randn(self.channels, self.channels, self.n_modes, self.n_modes, 2)
                )
                self.weight_neg = nn.Parameter(
                    scale * torch.randn(self.channels, self.channels, self.n_modes, self.n_modes, 2)
                )

            @staticmethod
            def _compl_mul(x_ft, w):
                return torch.einsum("bixy,ioxy->boxy", x_ft, w)

            def forward(self, x):
                bsz, _, h, w = x.shape
                x_ft = torch.fft.rfft2(x, norm="ortho")
                out_ft = torch.zeros((bsz, self.channels, h, (w // 2) + 1), dtype=torch.cfloat, device=x.device)
                m1 = int(min(self.n_modes, h))
                m2 = int(min(self.n_modes, (w // 2) + 1))
                if m1 <= 0 or m2 <= 0:
                    return x
                w_pos = torch.view_as_complex(self.weight_pos[:, :, :m1, :m2, :].contiguous())
                w_neg = torch.view_as_complex(self.weight_neg[:, :, :m1, :m2, :].contiguous())
                out_ft[:, :, :m1, :m2] = self._compl_mul(x_ft[:, :, :m1, :m2], w_pos)
                out_ft[:, :, -m1:, :m2] = self._compl_mul(x_ft[:, :, -m1:, :m2], w_neg)
                return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

        class _UNOBlock(nn.Module):
            def __init__(self, width: int, n_modes: int, dropout: float) -> None:
                super().__init__()
                self.global_mix = _LowFreqMix2d(width, n_modes)
                self.local_mix = nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                )
                self.skip = nn.Conv2d(width, width, kernel_size=1)
                self.norm = nn.GroupNorm(num_groups=1, num_channels=width)
                self.drop = nn.Dropout(float(max(dropout, 0.0)))

            def forward(self, x):
                y = self.global_mix(x) + self.local_mix(x)
                y = self.norm(y)
                y = self.drop(y)
                return torch.nn.functional.gelu(y + self.skip(x))

        class _UNONet(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                width: int,
                n_layers: int,
                n_modes: int,
                dropout: float,
                head_mode: str,
                output_keys: list[str],
                target_groups: dict[str, Any],
                with_rho_eff_head: bool,
            ):
                super().__init__()
                self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
                self.blocks = nn.ModuleList(
                    [_UNOBlock(width=width, n_modes=n_modes, dropout=dropout) for _ in range(max(int(n_layers), 1))]
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

        self.net = _UNONet(
            in_channels=int(self.feature_dim),
            out_channels=int(self.raw_out_channels),
            width=width,
            n_layers=n_layers,
            n_modes=int(self.n_modes),
            dropout=dropout,
            head_mode=str(self.output_heads_mode),
            output_keys=list(self.output_keys),
            target_groups=dict(self.target_groups),
            with_rho_eff_head=bool(self.with_rho_eff_head),
        )
        self._torch_width = int(width)
        self._torch_layers = int(n_layers)
