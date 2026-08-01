"""Operator U-Net variant for the CNO model family."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.fno._torch_grid_base import _TorchGridFieldBaseline


def normalize_cno_operator_unet_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    width = int(cfg.get("width", 48))
    depth = int(cfg.get("depth", 3))
    blocks_per_level = int(cfg.get("blocks_per_level", 2))
    max_width = int(cfg.get("max_width", 192))
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.0))
    downsample = str(cfg.get("downsample", "blur")).strip().lower()
    upsample = str(cfg.get("upsample", "bilinear")).strip().lower()
    use_film = bool(cfg.get("use_film", True))

    if width < 1:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.width must be >= 1")
    if depth < 2:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.depth must be >= 2")
    if blocks_per_level < 1:
        raise ValueError(
            "train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.blocks_per_level must be >= 1"
        )
    if max_width < width:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.max_width must be >= width")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.kernel_size must be odd and >= 1")
    if not np.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.dropout must be in [0, 1)")
    if downsample not in {"blur", "conv"}:
        raise ValueError("train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.downsample must be one of: blur, conv")
    if upsample not in {"bilinear", "nearest"}:
        raise ValueError(
            "train.cno_operator_unet.model_cfg.cno_operator_unet_cfg.upsample must be one of: bilinear, nearest"
        )

    return {
        "width": width,
        "depth": depth,
        "blocks_per_level": blocks_per_level,
        "max_width": max_width,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "downsample": downsample,
        "upsample": upsample,
        "use_film": use_film,
    }


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CNOOperatorUNet(_TorchGridFieldBaseline):
    """Multi-scale CNO-style operator U-Net.

    This is intentionally a separate model id from ``cno``.  The existing CNO
    remains a CNO-lite comparison baseline, while this class adds the missing
    operator U-Net ingredients: anti-aliased downsampling, resolution-restoring
    upsampling, skip fusion, and condition FiLM at each scale.
    """

    model_type = "cno_operator_unet"

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
        cno_operator_unet_cfg: dict[str, Any] | None = None,
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
            cfg_prefix="train.cno_operator_unet",
            default_output_keys=[f"target_{i}" for i in range(int(out_channels))],
            impl_version="cno_operator_unet_v1",
            head_arch_version="operator_unet_v1",
        )
        self.cno_operator_unet_cfg = normalize_cno_operator_unet_cfg(cno_operator_unet_cfg)

        torch = self.torch
        nn = torch.nn
        cfg = self.cno_operator_unet_cfg
        width = int(cfg["width"])
        depth = int(cfg["depth"])
        blocks_per_level = int(cfg["blocks_per_level"])
        max_width = int(cfg["max_width"])
        kernel_size = int(cfg["kernel_size"])
        padding = kernel_size // 2
        dropout = float(cfg["dropout"])
        use_film = bool(cfg["use_film"])
        upsample_mode = str(cfg["upsample"])
        downsample_mode = str(cfg["downsample"])
        widths = [min(width * (2**level), max_width) for level in range(depth)]

        class _FiLMResidualBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
                self.norm1 = nn.GroupNorm(_group_count(out_ch), out_ch)
                self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
                self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
                self.drop = nn.Dropout(dropout)
                self.film = nn.Linear(int(input_dim), 2 * out_ch) if use_film and int(input_dim) > 0 else None
                if self.film is not None:
                    nn.init.zeros_(self.film.weight)
                    nn.init.zeros_(self.film.bias)

            def forward(self, x, cond):
                y = torch.nn.functional.gelu(self.norm1(self.conv1(x)))
                y = self.norm2(self.conv2(y))
                if self.film is not None and cond is not None:
                    gamma, beta = self.film(cond).chunk(2, dim=1)
                    y = y * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                y = self.drop(y)
                return torch.nn.functional.gelu(y + self.skip(x))

        class _Stage(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, n_blocks: int) -> None:
                super().__init__()
                blocks = [_FiLMResidualBlock(in_ch, out_ch)]
                blocks.extend(_FiLMResidualBlock(out_ch, out_ch) for _ in range(max(n_blocks - 1, 0)))
                self.blocks = nn.ModuleList(blocks)

            def forward(self, x, cond):
                h = x
                for block in self.blocks:
                    h = block(h, cond)
                return h

        class _Down(nn.Module):
            def __init__(self, in_ch: int, out_ch: int) -> None:
                super().__init__()
                self.mode = downsample_mode
                if self.mode == "blur":
                    self.blur = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
                    kernel = torch.tensor(
                        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                        dtype=torch.float32,
                    )
                    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
                    with torch.no_grad():
                        self.blur.weight.copy_(kernel)
                    self.blur.weight.requires_grad_(False)
                else:
                    self.blur = nn.Identity()
                self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.proj(self.blur(x))

        class _Up(nn.Module):
            def __init__(self, in_ch: int, out_ch: int) -> None:
                super().__init__()
                self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

            def forward(self, x, size: tuple[int, int]):
                kwargs: dict[str, Any] = {"size": size, "mode": upsample_mode}
                if upsample_mode == "bilinear":
                    kwargs["align_corners"] = False
                y = torch.nn.functional.interpolate(x, **kwargs)
                return self.proj(y)

        class _OperatorUNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Conv2d(int(self_feature_dim), widths[0], kernel_size=1)
                self.encoders = nn.ModuleList()
                self.downs = nn.ModuleList()
                in_ch = widths[0]
                for level, out_ch in enumerate(widths):
                    self.encoders.append(_Stage(in_ch, out_ch, blocks_per_level))
                    if level < depth - 1:
                        self.downs.append(_Down(out_ch, widths[level + 1]))
                    in_ch = widths[level + 1] if level < depth - 1 else out_ch
                self.bottleneck = _Stage(widths[-1], widths[-1], blocks_per_level)
                self.ups = nn.ModuleList()
                self.decoders = nn.ModuleList()
                cur_ch = widths[-1]
                for skip_ch in reversed(widths[:-1]):
                    self.ups.append(_Up(cur_ch, skip_ch))
                    self.decoders.append(_Stage(skip_ch * 2, skip_ch, blocks_per_level))
                    cur_ch = skip_ch
                self.head = nn.Sequential(
                    nn.GroupNorm(_group_count(widths[0]), widths[0]),
                    nn.GELU(),
                    nn.Conv2d(widths[0], int(out_channels_raw), kernel_size=1),
                )

            def forward(self, x):
                cond = x[:, : int(input_dim)].mean(dim=(-2, -1)) if int(input_dim) > 0 else None
                h = self.in_proj(x)
                skips = []
                for idx, enc in enumerate(self.encoders):
                    h = enc(h, cond)
                    skips.append(h)
                    if idx < len(self.downs):
                        h = self.downs[idx](h)
                h = self.bottleneck(h, cond)
                for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
                    h = up(h, skip.shape[-2:])
                    h = dec(torch.cat([h, skip], dim=1), cond)
                return self.head(h)

            def step_reference(self):
                return self.in_proj.weight, self.head[-1].weight

        self_feature_dim = int(self.feature_dim)
        out_channels_raw = int(self.raw_out_channels)
        self.net = _OperatorUNet()
        self._ensure_net_device()
        self._torch_widths = list(widths)
        self._torch_depth = int(depth)
        self._torch_blocks_per_level = int(blocks_per_level)
        self._torch_kernel_size = int(kernel_size)
        self._torch_downsample = str(downsample_mode)
        self._torch_upsample = str(upsample_mode)
        self._torch_use_film = bool(use_film)

    def _torch_step_reference(self):
        return self.net.step_reference()


__all__ = ["CNOOperatorUNet", "normalize_cno_operator_unet_cfg"]
