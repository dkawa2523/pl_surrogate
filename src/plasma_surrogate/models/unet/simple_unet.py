"""UNet baseline model with optional real Conv backend (numpy/torch)."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import _build_unit_coord_grid
from plasma_surrogate.models.heads.role_grouped import (
    GROUPED_OUTPUT_HEAD_MODES,
    OUTPUT_HEAD_MODE_CAUSAL_EM,
    build_role_grouped_conv2d_head,
    configure_output_head_metadata,
    is_grouped_output_head_mode,
)
from plasma_surrogate.models.unet._torch_spatial_base import (
    _TorchSpatialFieldMixin,
)

class UNetBaseline(_TorchSpatialFieldMixin):
    """
    UNet baseline with two backends:
    - numpy: per-pixel head on [cond + spatial features]
    - torch: real 2D conv encoder-decoder with skip connection
    """

    _spatial_label = "unet"

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        head_mlp: dict[str, object] | None = None,
        backend: str = "numpy",
        input_feature_channels: list[str] | None = None,
        conv_cfg: dict[str, Any] | None = None,
        output_heads: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
    ):
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            self.output_keys = [f"target_{i}" for i in range(self.out_channels)]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        self.with_rho_eff_head = bool(with_rho_eff_head)
        self.raw_out_channels = self.out_channels + (1 if self.with_rho_eff_head else 0)
        self.backend = str(backend).strip().lower()
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("train.unet.model_cfg.backend must be one of: numpy, torch")

        channels = list(input_feature_channels or ["x", "y"])
        if len(channels) == 0:
            raise ValueError("train.unet.input_features.features must be a non-empty list")
        if len(set(channels)) != len(channels):
            raise ValueError("train.unet.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.feature_dim = int(self.input_dim + self.spatial_feature_dim)
        configure_output_head_metadata(
            self,
            output_heads=dict(output_heads or {}),
            output_keys=list(self.output_keys),
            target_role_schema=dict(target_role_schema or {}),
            cfg_prefix="train.unet",
        )

        self._cache: dict[str, object] = {}
        self._static_spatial_features: np.ndarray | None = None

        self.coord = _build_unit_coord_grid(self.grid_shape)

        if self.backend == "numpy":
            if self.output_heads_mode != "shared":
                raise ValueError("train.unet.model_cfg.output_heads non-shared modes require backend=torch")
            self._init_numpy(seed=seed, head_mlp=head_mlp)
        else:
            self._init_torch(seed=seed, conv_cfg=conv_cfg, output_heads=self.output_heads)

    def _init_numpy(self, *, seed: int, head_mlp: dict[str, object] | None) -> None:
        rng = np.random.default_rng(seed)
        self.head_mlp = dict(head_mlp or {})
        self.head_enabled = bool(self.head_mlp.get("enabled", False))
        self.head_arch_version = "linear_v1"
        if self.head_enabled:
            self.head_hidden = list(self.head_mlp.get("hidden", [128, 128]))
            self.head_activation = str(self.head_mlp.get("activation", "tanh")).strip().lower()
            self.head_dropout = float(np.clip(float(self.head_mlp.get("dropout", 0.0)), 0.0, 0.9))
            dims = [self.feature_dim, *self.head_hidden, self.raw_out_channels]
            self.weights: list[np.ndarray] = []
            self.biases: list[np.ndarray] = []
            for fan_in, fan_out in zip(dims[:-1], dims[1:]):
                scale = np.sqrt(2.0 / max(int(fan_in), 1))
                self.weights.append((rng.normal(size=(fan_in, fan_out)) * scale).astype(np.float32))
                self.biases.append(np.zeros((fan_out,), dtype=np.float32))
            self.head_arch_version = "mlp_head_v1"
            self.W = self.weights[-1]
            self.b = self.biases[-1]
        else:
            self.head_hidden = []
            self.head_activation = "tanh"
            self.head_dropout = 0.0
            self.W = rng.normal(0.0, 0.03, size=(self.feature_dim, self.raw_out_channels)).astype(np.float32)
            self.b = np.zeros((self.raw_out_channels,), dtype=np.float32)
            self.weights = [self.W]
            self.biases = [self.b]

    def _init_torch(self, *, seed: int, conv_cfg: dict[str, Any] | None, output_heads: dict[str, Any] | None) -> None:
        from plasma_surrogate.core.torch_backend import require_torch

        torch = require_torch()
        nn = torch.nn
        cfg = dict(conv_cfg or {})
        architecture = str(cfg.get("architecture", "unet")).strip().lower()
        if architecture not in {"unet", "response_scale"}:
            raise ValueError("train.unet.model_cfg.conv_cfg.architecture must be one of: response_scale, unet")
        response_scale_cfg = dict(cfg.get("response_scale_cfg", {}) or {})
        radial_padding_mode = str(
            response_scale_cfg.get("radial_padding_mode", "zero")
        ).strip().lower()
        if radial_padding_mode not in {"zero", "axisymmetric_neumann"}:
            raise ValueError(
                "train.unet.model_cfg.conv_cfg.response_scale_cfg.radial_padding_mode "
                "must be one of: axisymmetric_neumann, zero"
            )
        base_channels = int(cfg.get("base_channels", 32))
        if base_channels <= 0:
            raise ValueError("train.unet.model_cfg.base_channels must be > 0")
        depth = int(cfg.get("depth", 1))
        if depth not in {1, 2}:
            raise ValueError("train.unet.model_cfg.conv_cfg.depth must be one of: 1, 2")
        upsample_mode = str(cfg.get("upsample_mode", "deconv")).strip().lower()
        if upsample_mode not in {"deconv", "resize_conv"}:
            raise ValueError("train.unet.model_cfg.upsample_mode must be one of: deconv, resize_conv")
        mid_channels = int(max(base_channels * 2, 2))
        bot_channels = int(max(base_channels * 4, 4))
        in_channels = int(self.input_dim + self.spatial_feature_dim)
        head_cfg = dict(output_heads or {})
        output_heads_mode = str(head_cfg.get("mode", "shared")).strip().lower()
        if output_heads_mode not in {"shared", OUTPUT_HEAD_MODE_CAUSAL_EM, *GROUPED_OUTPUT_HEAD_MODES}:
            raise ValueError(
                "train.unet.model_cfg.output_heads.mode must be one of: "
                "causal_em, custom_groups, role_grouped, shared"
            )
        if architecture == "response_scale":
            if output_heads_mode != "shared":
                raise ValueError("response_scale architecture requires output_heads.mode=shared")
            if self.with_rho_eff_head:
                raise ValueError("response_scale architecture does not support rho_eff head")
            required_targets = {"ne", "ni", "Te", "phi"}
            if set(self.output_keys) != required_targets or len(self.output_keys) != len(required_targets):
                raise ValueError(
                    "response_scale architecture requires output_keys to contain exactly ne, ni, Te, phi"
                )

        driver_indices: list[int] = []
        response_indices: list[int] = []
        causal_hidden_channels = int(head_cfg.get("hidden_channels", base_channels))
        causal_detach_driver = bool(head_cfg.get("detach_driver", False))
        if output_heads_mode == OUTPUT_HEAD_MODE_CAUSAL_EM:
            if self.with_rho_eff_head:
                raise ValueError("train.unet.model_cfg.output_heads.mode=causal_em does not support rho_eff head")
            driver_targets = [str(v) for v in list(head_cfg.get("driver_targets", []))]
            response_targets = [str(v) for v in list(head_cfg.get("response_targets", []))]
            if not driver_targets:
                raise ValueError("train.unet.model_cfg.output_heads.driver_targets must be non-empty")
            if not response_targets:
                response_targets = [key for key in self.output_keys if key not in set(driver_targets)]
            if len(set(driver_targets)) != len(driver_targets) or len(set(response_targets)) != len(response_targets):
                raise ValueError("train.unet.model_cfg.output_heads causal target lists must not contain duplicates")
            overlap = sorted(set(driver_targets) & set(response_targets))
            if overlap:
                raise ValueError(f"causal_em driver_targets and response_targets overlap: {overlap}")
            unknown = sorted((set(driver_targets) | set(response_targets)) - set(self.output_keys))
            missing = [key for key in self.output_keys if key not in set(driver_targets + response_targets)]
            if unknown or missing:
                raise ValueError(
                    "causal_em targets must cover output_keys exactly; "
                    f"unknown={unknown}, missing={missing}"
                )
            if causal_hidden_channels <= 0:
                raise ValueError("train.unet.model_cfg.output_heads.hidden_channels must be > 0")
            driver_indices = [self.output_keys.index(key) for key in driver_targets]
            response_indices = [self.output_keys.index(key) for key in response_targets]

        class _ConvUNetDepth1Backbone(nn.Module):
            def __init__(self, in_ch: int, base_ch: int, mid_ch: int, upsample_mode: str):
                super().__init__()
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.down = nn.Sequential(
                    nn.Conv2d(base_ch, mid_ch, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.upsample_mode = str(upsample_mode)
                if self.upsample_mode == "deconv":
                    self.up = nn.ConvTranspose2d(mid_ch, base_ch, kernel_size=2, stride=2)
                    self.up_refine = None
                else:
                    self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
                    self.up_refine = nn.Conv2d(mid_ch, base_ch, kernel_size=3, padding=1)
                self.dec = nn.Sequential(
                    nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                e1 = self.enc1(x)
                z = self.down(e1)
                u = self.up(z)
                if self.up_refine is not None:
                    u = self.up_refine(u)
                if u.shape[-2:] != e1.shape[-2:]:
                    u = nn.functional.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
                return self.dec(torch.cat([u, e1], dim=1))

        class _ConvUNetDepth2Backbone(nn.Module):
            def __init__(self, in_ch: int, base_ch: int, mid_ch: int, bot_ch: int, upsample_mode: str):
                super().__init__()
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.down1 = nn.Sequential(
                    nn.Conv2d(base_ch, mid_ch, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.enc2 = nn.Sequential(
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.down2 = nn.Sequential(
                    nn.Conv2d(mid_ch, bot_ch, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.upsample_mode = str(upsample_mode)
                if self.upsample_mode == "deconv":
                    self.up2 = nn.ConvTranspose2d(bot_ch, mid_ch, kernel_size=2, stride=2)
                    self.up2_refine = None
                    self.up1 = nn.ConvTranspose2d(mid_ch, base_ch, kernel_size=2, stride=2)
                    self.up1_refine = None
                else:
                    self.up2 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
                    self.up2_refine = nn.Conv2d(bot_ch, mid_ch, kernel_size=3, padding=1)
                    self.up1 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
                    self.up1_refine = nn.Conv2d(mid_ch, base_ch, kernel_size=3, padding=1)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.dec1 = nn.Sequential(
                    nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                e1 = self.enc1(x)
                z1 = self.down1(e1)
                e2 = self.enc2(z1)
                z2 = self.down2(e2)
                u2 = self.up2(z2)
                if self.up2_refine is not None:
                    u2 = self.up2_refine(u2)
                if u2.shape[-2:] != e2.shape[-2:]:
                    u2 = nn.functional.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
                d2 = self.dec2(torch.cat([u2, e2], dim=1))
                u1 = self.up1(d2)
                if self.up1_refine is not None:
                    u1 = self.up1_refine(u1)
                if u1.shape[-2:] != e1.shape[-2:]:
                    u1 = nn.functional.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
                return self.dec1(torch.cat([u1, e1], dim=1))

        class _ConvUNetHead(nn.Module):
            def __init__(
                self,
                base_ch: int,
                out_ch: int,
                mode: str,
                with_rho_eff_head: bool,
                *,
                driver_indices: list[int],
                response_indices: list[int],
                causal_hidden_channels: int,
                causal_detach_driver: bool,
            ):
                super().__init__()
                self.mode = mode
                self.with_rho_eff_head = bool(with_rho_eff_head)
                self.driver_indices = list(driver_indices)
                self.response_indices = list(response_indices)
                self.causal_detach_driver = bool(causal_detach_driver)
                if self.mode == "shared":
                    self.out = nn.Conv2d(base_ch, out_ch + (1 if self.with_rho_eff_head else 0), kernel_size=1)
                    self.role_grouped = None
                    self.driver_head = None
                    self.response_head = None
                elif self.mode == OUTPUT_HEAD_MODE_CAUSAL_EM:
                    self.out = None
                    self.role_grouped = None
                    self.driver_head = nn.Conv2d(base_ch, len(self.driver_indices), kernel_size=1)
                    self.response_head = nn.Sequential(
                        nn.Conv2d(
                            base_ch + len(self.driver_indices),
                            int(causal_hidden_channels),
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.GELU(),
                        nn.Conv2d(int(causal_hidden_channels), len(self.response_indices), kernel_size=1),
                    )
                else:
                    self.out = None
                    self.driver_head = None
                    self.response_head = None
                    if not is_grouped_output_head_mode(self.mode):
                        raise ValueError(
                            "train.unet.model_cfg.output_heads.mode must be one of: "
                            "causal_em, custom_groups, role_grouped, shared"
                        )
                    self.role_grouped = build_role_grouped_conv2d_head(
                        torch=torch,
                        in_channels=base_ch,
                        output_keys=list(self_output_keys),
                        target_groups=dict(self_target_groups),
                        group_options=dict(self_output_head_group_options),
                        with_rho_eff_head=self.with_rho_eff_head,
                    )

            def forward(self, feat, *, output_keys: list[str]):
                del output_keys
                if self.mode == "shared":
                    return self.out(feat)
                if self.mode == OUTPUT_HEAD_MODE_CAUSAL_EM:
                    driver = self.driver_head(feat)
                    response_driver = driver.detach() if self.causal_detach_driver else driver
                    response = self.response_head(torch.cat([feat, response_driver], dim=1))
                    out = feat.new_empty(
                        (feat.shape[0], len(self.driver_indices) + len(self.response_indices), *feat.shape[-2:])
                    )
                    for source_index, target_index in enumerate(self.driver_indices):
                        out[:, target_index : target_index + 1] = driver[:, source_index : source_index + 1]
                    for source_index, target_index in enumerate(self.response_indices):
                        out[:, target_index : target_index + 1] = response[:, source_index : source_index + 1]
                    return out
                return self.role_grouped(feat)

            def step_reference(self):
                if self.out is not None:
                    return self.out.weight
                if self.driver_head is not None:
                    return self.driver_head.weight
                if self.role_grouped is not None:
                    return self.role_grouped.step_reference()
                return None

        class _ConvUNetModel(nn.Module):
            def __init__(
                self,
                *,
                in_ch: int,
                base_ch: int,
                mid_ch: int,
                bot_ch: int,
                depth: int,
                out_ch: int,
                head_mode: str,
                with_rho_eff_head: bool,
                upsample_mode: str,
                driver_indices: list[int],
                response_indices: list[int],
                causal_hidden_channels: int,
                causal_detach_driver: bool,
            ):
                super().__init__()
                if depth == 1:
                    self.backbone = _ConvUNetDepth1Backbone(in_ch, base_ch, mid_ch, upsample_mode)
                else:
                    self.backbone = _ConvUNetDepth2Backbone(in_ch, base_ch, mid_ch, bot_ch, upsample_mode)
                self.head = _ConvUNetHead(
                    base_ch,
                    out_ch,
                    head_mode,
                    with_rho_eff_head,
                    driver_indices=driver_indices,
                    response_indices=response_indices,
                    causal_hidden_channels=causal_hidden_channels,
                    causal_detach_driver=causal_detach_driver,
                )
                self.head_mode = head_mode

            def forward(self, x, *, output_keys: list[str]):
                feat = self.backbone(x)
                return self.head(feat, output_keys=output_keys)

        class _ResponseScaleModel(nn.Module):
            """Skip-free coarse decoder with a positive quasi-neutral density head."""

            def __init__(
                self,
                *,
                in_ch: int,
                base_ch: int,
                output_keys: list[str],
                mask_channel_index: int,
                target_affine: dict[str, dict[str, float]],
                density_imbalance_fraction: float,
                radial_padding_mode: str,
            ):
                super().__init__()
                self.output_keys = list(output_keys)
                self.mask_channel_index = int(mask_channel_index)
                self.density_imbalance_fraction = float(density_imbalance_fraction)
                self.radial_padding_mode = str(radial_padding_mode)
                channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]

                class _AxisymmetricNeumannConv2d(nn.Module):
                    """3x3 convolution with cell-centered reflection at r=0.

                    ICP stores the first radial cell at positive r.  Its ghost
                    cell across the symmetry axis must therefore copy the
                    first interior cell.  The outer-r and both axial sides
                    retain the existing zero-padding contract.
                    """

                    def __init__(
                        self,
                        in_channels: int,
                        out_channels: int,
                        *,
                        stride: int = 1,
                        dilation: int = 1,
                    ) -> None:
                        super().__init__()
                        self.radial_pad = int(dilation)
                        self.axial_pad = int(dilation)
                        self.conv = nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=int(stride),
                            padding=0,
                            dilation=int(dilation),
                        )

                    @property
                    def weight(self):
                        return self.conv.weight

                    @property
                    def bias(self):
                        return self.conv.bias

                    def forward(self, value):
                        if value.shape[-1] < self.radial_pad:
                            raise ValueError("radial grid is too narrow for axisymmetric padding")
                        reflected = value[..., : self.radial_pad].flip(-1)
                        padded = torch.cat((reflected, value), dim=-1)
                        padded = nn.functional.pad(
                            padded,
                            (0, self.radial_pad, self.axial_pad, self.axial_pad),
                            mode="constant",
                            value=0.0,
                        )
                        return self.conv(padded)

                def conv3(
                    in_channels: int,
                    out_channels: int,
                    *,
                    stride: int = 1,
                    dilation: int = 1,
                ):
                    if self.radial_padding_mode == "axisymmetric_neumann":
                        return _AxisymmetricNeumannConv2d(
                            in_channels,
                            out_channels,
                            stride=stride,
                            dilation=dilation,
                        )
                    return nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=dilation,
                        dilation=dilation,
                    )

                blocks = []
                previous = int(in_ch)
                for channel in channels:
                    blocks.append(
                        nn.Sequential(
                            conv3(previous, channel, stride=2),
                            nn.GELU(),
                            conv3(channel, channel),
                            nn.GELU(),
                        )
                    )
                    previous = channel
                self.encoder = nn.ModuleList(blocks)
                self.bottleneck = nn.Sequential(
                    conv3(previous, previous, dilation=2),
                    nn.GELU(),
                    conv3(previous, previous),
                    nn.GELU(),
                )
                self.decode_1 = nn.Sequential(
                    conv3(previous, base_ch * 2),
                    nn.GELU(),
                )
                self.decode_2 = nn.Sequential(
                    conv3(base_ch * 2, base_ch),
                    nn.GELU(),
                )
                self.density_shape = nn.Conv2d(base_ch, 1, kernel_size=1)
                self.density_imbalance = nn.Conv2d(base_ch, 1, kernel_size=1)
                self.density_amplitude = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(previous, 1, kernel_size=1),
                )
                self.other_head = nn.Conv2d(base_ch, 2, kernel_size=1)

                means = []
                scales = []
                for name in ("ne", "ni"):
                    affine = dict(target_affine.get(name, {}) or {})
                    mean = float(affine.get("mean", float("nan")))
                    scale = float(affine.get("scale", float("nan")))
                    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
                        raise ValueError(f"response_scale requires finite positive target affine for {name}")
                    means.append(mean)
                    scales.append(scale)
                self.register_buffer("density_means", torch.tensor(means, dtype=torch.float32).reshape(1, 2, 1, 1))
                self.register_buffer("density_scales", torch.tensor(scales, dtype=torch.float32).reshape(1, 2, 1, 1))
                self.register_buffer(
                    "density_reference_scale",
                    torch.tensor(float(np.mean(scales)), dtype=torch.float32),
                )

            def forward(self, x, *, output_keys: list[str]):
                full_size = x.shape[-2:]
                encoded = x
                for block in self.encoder:
                    encoded = block(encoded)
                encoded = self.bottleneck(encoded)
                amplitude = nn.functional.softplus(self.density_amplitude(encoded)) + 1.0e-6

                decoded = nn.functional.interpolate(encoded, scale_factor=2.0, mode="bilinear", align_corners=False)
                decoded = self.decode_1(decoded)
                decoded = nn.functional.interpolate(decoded, scale_factor=2.0, mode="bilinear", align_corners=False)
                decoded = self.decode_2(decoded)
                mask = nn.functional.interpolate(
                    x[:, self.mask_channel_index : self.mask_channel_index + 1],
                    size=decoded.shape[-2:],
                    mode="area",
                ).clamp(0.0, 1.0)
                positive_shape = nn.functional.softplus(self.density_shape(decoded)) + 1.0e-6
                normalizer = torch.sum(positive_shape * mask, dim=(2, 3), keepdim=True) / torch.clamp(
                    torch.sum(mask, dim=(2, 3), keepdim=True), min=1.0
                )
                density_shape = positive_shape / torch.clamp(normalizer, min=1.0e-6)
                common_density = self.density_reference_scale * amplitude * density_shape
                imbalance = self.density_imbalance_fraction * torch.tanh(self.density_imbalance(decoded))
                density_physical = torch.cat(
                    [common_density * (1.0 - imbalance), common_density * (1.0 + imbalance)],
                    dim=1,
                )
                density_standard = (density_physical - self.density_means) / self.density_scales
                other = self.other_head(decoded)
                low = decoded.new_empty((decoded.shape[0], len(output_keys), *decoded.shape[-2:]))
                low[:, output_keys.index("ne") : output_keys.index("ne") + 1] = density_standard[:, 0:1]
                low[:, output_keys.index("ni") : output_keys.index("ni") + 1] = density_standard[:, 1:2]
                low[:, output_keys.index("Te") : output_keys.index("Te") + 1] = other[:, 0:1]
                low[:, output_keys.index("phi") : output_keys.index("phi") + 1] = other[:, 1:2]
                return nn.functional.interpolate(low, size=full_size, mode="bilinear", align_corners=False)

            def step_reference(self):
                return self.encoder[0][0].weight, self.other_head.weight

        self.torch = torch
        self._init_torch_device()
        self.output_heads_mode = output_heads_mode
        self_output_keys = list(self.output_keys)
        self_target_groups = dict(getattr(self, "target_groups", {}) or {})
        self_output_head_group_options = dict(getattr(self, "output_head_group_options", {}) or {})
        if architecture == "response_scale":
            if "mask_plasma" not in self.input_feature_channels:
                raise ValueError("response_scale architecture requires mask_plasma input feature")
            target_affine = dict(response_scale_cfg.get("target_affine", {}) or {})
            imbalance_fraction = float(response_scale_cfg.get("density_imbalance_fraction", 0.05))
            if not np.isfinite(imbalance_fraction) or not (0.0 <= imbalance_fraction < 1.0):
                raise ValueError("response_scale density_imbalance_fraction must be in [0,1)")
            self.net = _ResponseScaleModel(
                in_ch=in_channels,
                base_ch=base_channels,
                output_keys=self_output_keys,
                mask_channel_index=self.input_dim + self.input_feature_channels.index("mask_plasma"),
                target_affine=target_affine,
                density_imbalance_fraction=imbalance_fraction,
                radial_padding_mode=radial_padding_mode,
            )
        else:
            self.net = _ConvUNetModel(
                in_ch=in_channels,
                base_ch=base_channels,
                mid_ch=mid_channels,
                bot_ch=bot_channels,
                depth=depth,
                out_ch=self.out_channels,
                head_mode=output_heads_mode,
                with_rho_eff_head=self.with_rho_eff_head,
                upsample_mode=upsample_mode,
                driver_indices=driver_indices,
                response_indices=response_indices,
                causal_hidden_channels=causal_hidden_channels,
                causal_detach_driver=causal_detach_driver,
            )
        self._ensure_net_device()
        self.net.train()
        self._torch_seed = int(seed)
        self._torch_base_channels = int(base_channels)
        self._torch_depth = int(depth)
        self._torch_upsample_mode = str(upsample_mode)
        self._torch_architecture = str(architecture)
        self._torch_response_scale_cfg = dict(response_scale_cfg)
        self._torch_output_heads_mode = str(output_heads_mode)
        self._torch_last_out = None
        self._torch_last_in = None
        self.head_enabled = True
        self.head_hidden = []
        self.head_activation = "relu"
        self.head_dropout = 0.0
        self.head_arch_version = (
            "response_scale_quasineutral_v1"
            if architecture == "response_scale"
            else (
                "conv_unet_causal_em_v1"
                if output_heads_mode == OUTPUT_HEAD_MODE_CAUSAL_EM
                else "conv_unet_v1"
            )
        )

    def set_static_spatial_features(self, spatial_features: np.ndarray | None) -> None:
        return _TorchSpatialFieldMixin.set_static_spatial_features(self, spatial_features)

    def _resolve_spatial_features(self, cond: np.ndarray, spatial_features: np.ndarray | None) -> np.ndarray:
        return _TorchSpatialFieldMixin._resolve_spatial_features(self, cond, spatial_features)

    def _feature_map(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return _TorchSpatialFieldMixin._feature_map(self, cond, spatial_features=spatial_features)

    def feature_matrix(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return _TorchSpatialFieldMixin.feature_matrix(self, cond, spatial_features=spatial_features)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.head_activation == "relu":
            return np.maximum(z, 0.0).astype(np.float32)
        return np.tanh(z).astype(np.float32)

    def _activate_grad(self, z: np.ndarray) -> np.ndarray:
        if self.head_activation == "relu":
            return (z > 0.0).astype(np.float32)
        t = np.tanh(z).astype(np.float32)
        return 1.0 - t * t

    def _forward_raw_numpy(self, cond: np.ndarray, *, training: bool, spatial_features: np.ndarray | None) -> np.ndarray:
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        bsz = fmap.shape[0]
        x = fmap.reshape(-1, self.feature_dim).astype(np.float32)
        if not self.head_enabled:
            y = x @ self.W + self.b
            self._cache = {"x": x}
            return np.moveaxis(y.reshape(bsz, *self.grid_shape, self.raw_out_channels), -1, 1).astype(np.float32)

        acts: list[np.ndarray] = [x]
        zs: list[np.ndarray] = []
        drops: list[np.ndarray] = []
        a = x
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            zs.append(z)
            if i < n_layers - 1:
                a = self._activate(z)
                if training and self.head_dropout > 0.0:
                    keep = 1.0 - self.head_dropout
                    dm = (np.random.rand(*a.shape) < keep).astype(np.float32) / max(keep, 1e-6)
                    a = a * dm
                else:
                    dm = np.ones_like(a, dtype=np.float32)
                drops.append(dm)
            else:
                a = z.astype(np.float32)
                drops.append(np.ones_like(a, dtype=np.float32))
            acts.append(a)
        self._cache = {"acts": acts, "zs": zs, "drops": drops}
        return np.moveaxis(a.reshape(bsz, *self.grid_shape, self.raw_out_channels), -1, 1).astype(np.float32)

    def _torch_forward(self, xt):
        return self.net(xt, output_keys=self.output_keys)

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool, spatial_features: np.ndarray | None) -> np.ndarray:
        return _TorchSpatialFieldMixin._forward_raw_torch(
            self,
            cond,
            training=bool(training),
            spatial_features=spatial_features,
        )

    def _forward_raw(self, cond: np.ndarray, *, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        if self.backend == "torch":
            return self._forward_raw_torch(cond, training=bool(training), spatial_features=spatial_features)
        return self._forward_raw_numpy(cond, training=bool(training), spatial_features=spatial_features)

    def forward_raw(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._forward_raw(cond, training=bool(training), spatial_features=spatial_features)

    def forward_features(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if self.backend == "torch":
            return _TorchSpatialFieldMixin.forward_features(self, cond, spatial_features=spatial_features)
        y = self._forward_raw(cond, training=False, spatial_features=spatial_features)
        out = {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}
        if self.with_rho_eff_head:
            out["rho_eff"] = y[:, self.out_channels : self.out_channels + 1]
        return out

    def forward(self, cond: np.ndarray, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return self._forward_raw(cond, training=bool(training), spatial_features=spatial_features)[:, : self.out_channels].astype(
            np.float32
        )

    def predict_fields(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return _TorchSpatialFieldMixin.predict_fields(self, cond, spatial_features=spatial_features)

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
        del target_raw, loss_cfg
        if self.backend == "torch":
            return self._backward_raw_torch(grad_raw, lr=lr, apply_step=apply_step)
        return self._backward_raw_numpy(grad_raw, lr=lr, weight_decay=weight_decay)

    def _torch_step_reference(self):
        if self.backend != "torch":
            return None, None
        if str(getattr(self, "_torch_architecture", "unet")) == "response_scale":
            return self.net.step_reference()
        hidden = self.net.backbone.enc1[0].weight
        out = self.net.head.step_reference()
        return hidden, out

    def _backward_raw_torch(self, grad_raw: np.ndarray, *, lr: float, apply_step: bool = True) -> dict[str, float]:
        return _TorchSpatialFieldMixin._backward_raw_torch(self, grad_raw, lr=float(lr), apply_step=bool(apply_step))

    def _backward_raw_numpy(self, grad_raw: np.ndarray, *, lr: float, weight_decay: float = 0.0) -> dict[str, float]:
        grad_y = np.moveaxis(np.asarray(grad_raw, dtype=np.float32), 1, -1).reshape(-1, self.raw_out_channels)
        wd = float(max(weight_decay, 0.0))
        if not self.head_enabled:
            x = np.asarray(self._cache.get("x"), dtype=np.float32)
            if x.size == 0:
                raise RuntimeError("UNetBaseline.backward_raw called without forward cache")
            w_prev = self.W.copy()
            grad_w = x.T @ grad_y
            grad_b = np.sum(grad_y, axis=0)
            if wd > 0.0:
                grad_w = grad_w + wd * w_prev
            step_rel_out = float(np.linalg.norm(lr * grad_w) / max(np.linalg.norm(w_prev), 1e-12))
            self.W = self.W - float(lr) * grad_w.astype(np.float32)
            self.b = self.b - float(lr) * grad_b.astype(np.float32)
            return {"step_rel_hidden_mean": 0.0, "step_rel_output": step_rel_out}

        acts = self._cache.get("acts")
        zs = self._cache.get("zs")
        drops = self._cache.get("drops")
        if not isinstance(acts, list) or not isinstance(zs, list) or not isinstance(drops, list):
            raise RuntimeError("UNetBaseline.backward_raw called without forward cache")
        step_hidden: list[float] = []
        step_out = 0.0
        grad = grad_y
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = np.asarray(acts[i], dtype=np.float32)
            w_prev = self.weights[i].copy()
            grad_w = a_prev.T @ grad
            grad_b = np.sum(grad, axis=0)
            if wd > 0.0:
                grad_w = grad_w + wd * w_prev
            step = float(np.linalg.norm(lr * grad_w) / max(np.linalg.norm(w_prev), 1e-12))
            if i == len(self.weights) - 1:
                step_out = step
            else:
                step_hidden.append(step)
            self.weights[i] = self.weights[i] - float(lr) * grad_w.astype(np.float32)
            self.biases[i] = self.biases[i] - float(lr) * grad_b.astype(np.float32)
            if i > 0:
                grad = grad @ w_prev.T
                grad = grad * self._activate_grad(np.asarray(zs[i - 1], dtype=np.float32))
                grad = grad * np.asarray(drops[i - 1], dtype=np.float32)
        self.W = self.weights[-1]
        self.b = self.biases[-1]
        return {
            "step_rel_hidden_mean": float(np.mean(step_hidden)) if step_hidden else 0.0,
            "step_rel_output": float(step_out),
        }

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self.backend == "torch":
            return self._state_dict_numpy_torch()
        if self.head_enabled:
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                out[f"layer{i}.W"] = np.asarray(w, dtype=np.float32)
                out[f"layer{i}.b"] = np.asarray(b, dtype=np.float32)
        else:
            out["W"] = np.asarray(self.W, dtype=np.float32)
            out["b"] = np.asarray(self.b, dtype=np.float32)
        return out

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        if self.backend == "torch":
            self._load_state_dict_numpy_torch(state)
            return

        if "layer0.W" in state and "layer0.b" in state:
            w_list: list[np.ndarray] = []
            b_list: list[np.ndarray] = []
            i = 0
            while f"layer{i}.W" in state and f"layer{i}.b" in state:
                w_list.append(np.asarray(state[f"layer{i}.W"], dtype=np.float32))
                b_list.append(np.asarray(state[f"layer{i}.b"], dtype=np.float32))
                i += 1
            self.weights = w_list
            self.biases = b_list
            self.head_enabled = True
            self.head_arch_version = "mlp_head_v1_loaded"
            self.W = self.weights[-1]
            self.b = self.biases[-1]
            return
        if "W" in state and "b" in state:
            self.W = np.asarray(state["W"], dtype=np.float32)
            self.b = np.asarray(state["b"], dtype=np.float32)
            self.weights = [self.W]
            self.biases = [self.b]
            self.head_enabled = False
            self.head_arch_version = "linear_v1_loaded"
            return
        raise ValueError("UNetBaseline state dict does not contain expected weights")
