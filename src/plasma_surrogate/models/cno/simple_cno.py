"""Minimal CNO-lite torch grid model."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.fno._torch_grid_base import _TorchGridFieldBaseline


def normalize_cno_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    width = int(cfg.get("width", 64))
    n_layers = int(cfg.get("n_layers", 4))
    dropout = float(cfg.get("dropout", 0.0))
    kernel_size = int(cfg.get("kernel_size", 3))
    if width < 1:
        raise ValueError("train.cno.model_cfg.cno_cfg.width must be >= 1")
    if n_layers < 1:
        raise ValueError("train.cno.model_cfg.cno_cfg.n_layers must be >= 1")
    if not np.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise ValueError("train.cno.model_cfg.cno_cfg.dropout must be finite and in [0, 1)")
    if kernel_size < 1:
        raise ValueError("train.cno.model_cfg.cno_cfg.kernel_size must be >= 1")
    if kernel_size % 2 == 0:
        raise ValueError("train.cno.model_cfg.cno_cfg.kernel_size must be odd")
    return {
        "width": int(width),
        "n_layers": int(n_layers),
        "dropout": float(dropout),
        "kernel_size": int(kernel_size),
    }


class CNOBaseline(_TorchGridFieldBaseline):
    """CNO-lite baseline with local convolutional mixing."""

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        head_mlp: dict[str, object] | None = None,
        input_feature_channels: list[str] | None = None,
        cno_cfg: dict[str, Any] | None = None,
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
            cfg_prefix="train.cno",
            default_output_keys=["ne", "Te", "phi"],
            impl_version="cno_lite_v1",
            head_arch_version="linear_v1",
        )
        self.cno_cfg = normalize_cno_cfg(cno_cfg)
        width = int(self.cno_cfg["width"])
        n_layers = int(self.cno_cfg["n_layers"])
        dropout = float(self.cno_cfg["dropout"])
        kernel_size = int(self.cno_cfg["kernel_size"])
        padding = int(kernel_size // 2)

        torch = self.torch
        nn = torch.nn

        class _CNOBlock(nn.Module):
            def __init__(self, width: int, kernel_size: int, padding: int, dropout: float) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding)
                self.conv2 = nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding)
                self.skip = nn.Conv2d(width, width, kernel_size=1)
                self.norm = nn.GroupNorm(num_groups=1, num_channels=width)
                self.drop = nn.Dropout(float(max(dropout, 0.0)))

            def forward(self, x):
                y = torch.nn.functional.gelu(self.conv1(x))
                y = self.conv2(y)
                y = self.norm(y)
                y = self.drop(y)
                return torch.nn.functional.gelu(y + self.skip(x))

        class _CNONet(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                width: int,
                n_layers: int,
                kernel_size: int,
                padding: int,
                dropout: float,
            ):
                super().__init__()
                self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
                self.blocks = nn.ModuleList(
                    [
                        _CNOBlock(
                            width=width,
                            kernel_size=kernel_size,
                            padding=padding,
                            dropout=dropout,
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

        self.net = _CNONet(
            in_channels=int(self.feature_dim),
            out_channels=int(self.raw_out_channels),
            width=width,
            n_layers=n_layers,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout,
        )
        self._torch_width = int(width)
        self._torch_layers = int(n_layers)
        self._torch_kernel_size = int(kernel_size)
