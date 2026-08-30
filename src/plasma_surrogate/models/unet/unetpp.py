"""Torch UNet++ baseline model with optional skip-gated attention."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import _build_unit_coord_grid
from plasma_surrogate.models.heads.role_grouped import (
    build_role_grouped_conv2d_head,
    configure_output_head_metadata,
    is_grouped_output_head_mode,
)
from plasma_surrogate.models.unet._torch_spatial_base import (
    _TorchSpatialFieldMixin,
)


def _resolve_unetpp_identity(conv_cfg: dict[str, Any] | None) -> tuple[str, str]:
    cfg = dict(conv_cfg or {})
    attention_cfg = dict(cfg.get("attention_cfg", {}))
    if bool(attention_cfg.get("enabled", False)):
        return "unetpp_attn", "train.unetpp_attn"
    return "unetpp", "train.unetpp"


def _build_unetpp_modules(
    *,
    torch,
    nn,
    in_ch: int,
    base_ch: int,
    mid_ch: int,
    bot_ch: int,
    out_ch: int,
    with_rho_eff_head: bool,
    upsample_mode: str,
    attention_enabled: bool,
    attention_reduction: int,
    head_mode: str,
    output_keys: list[str],
    target_groups: dict[str, Any],
    group_options: dict[str, Any] | None = None,
    architecture: str = "unetpp",
    response_scale_cfg: dict[str, Any] | None = None,
    mask_channel_index: int | None = None,
):
    response_cfg = dict(response_scale_cfg or {})
    radial_padding_mode = str(response_cfg.get("radial_padding_mode", "zero"))
    downsampling_stem = str(response_cfg.get("downsampling_stem", "fixed_area_v1")).strip().lower()

    class _AxisymmetricNeumannConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=int(stride), padding=0
            )

        @property
        def weight(self):
            return self.conv.weight

        def forward(self, value):
            reflected = value[..., :1]
            padded = torch.cat((reflected, value), dim=-1)
            padded = nn.functional.pad(padded, (0, 1, 1, 1), mode="constant", value=0.0)
            return self.conv(padded)

    def _conv3(in_channels: int, out_channels: int, *, stride: int = 1):
        if radial_padding_mode == "axisymmetric_neumann":
            return _AxisymmetricNeumannConv2d(in_channels, out_channels, stride=stride)
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=int(stride), padding=1
        )

    class _DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                _conv3(in_channels, out_channels),
                nn.ReLU(inplace=True),
                _conv3(out_channels, out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _Downsample(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                _conv3(in_channels, out_channels, stride=2),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _UpsampleModule(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, mode: str):
            super().__init__()
            self.mode = str(mode)
            if self.mode == "deconv":
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                self.refine = None
            else:
                self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
                self.refine = _conv3(in_channels, out_channels)

        def forward(self, x, size: tuple[int, int]):
            y = self.up(x)
            if self.refine is not None:
                y = self.refine(y)
            if y.shape[-2:] != size:
                y = nn.functional.interpolate(y, size=size, mode="bilinear", align_corners=False)
            return y

    class _AttentionGate2dModule(nn.Module):
        def __init__(self, skip_ch: int, gate_ch: int, reduction: int):
            super().__init__()
            inter_ch = max(int(skip_ch) // int(reduction), 1)
            self.theta_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1)
            self.phi_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1)
            self.psi = nn.Conv2d(inter_ch, skip_ch, kernel_size=1)
            self.act = nn.ReLU(inplace=True)
            self.gate = nn.Sigmoid()

        def forward(self, skip, gate):
            attn = self.theta_x(skip) + self.phi_g(gate)
            attn = self.act(attn)
            return skip * self.gate(self.psi(attn))

    class _UNetPPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = str(head_mode)
            self.response_scale = str(architecture) == "response_scale"
            if self.response_scale:
                if self.mode != "shared":
                    raise ValueError("UNet++ response_scale architecture requires output_heads.mode=shared")
                if with_rho_eff_head:
                    raise ValueError("UNet++ response_scale architecture does not support rho_eff head")
                required = {"ne", "ni", "Te", "phi"}
                if set(output_keys) != required or len(output_keys) != 4:
                    raise ValueError(
                        "UNet++ response_scale architecture requires output_keys to contain exactly ne, ni, Te, phi"
                    )
                target_affine = dict(response_cfg.get("target_affine", {}) or {})
                means: list[float] = []
                scales: list[float] = []
                for name in ("ne", "ni"):
                    affine = dict(target_affine.get(name, {}) or {})
                    mean = float(affine.get("mean", float("nan")))
                    scale = float(affine.get("scale", float("nan")))
                    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
                        raise ValueError(f"UNet++ response_scale requires finite positive target affine for {name}")
                    means.append(mean)
                    scales.append(scale)
                imbalance = float(response_cfg.get("density_imbalance_fraction", 0.05))
                if not np.isfinite(imbalance) or not (0.0 <= imbalance < 1.0):
                    raise ValueError("UNet++ response_scale density_imbalance_fraction must be in [0,1)")
                self.imbalance_fraction = imbalance
                self.density_shape = nn.Conv2d(base_ch, 1, kernel_size=1)
                self.density_imbalance = nn.Conv2d(base_ch, 1, kernel_size=1)
                self.density_amplitude = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Conv2d(base_ch, 1, kernel_size=1)
                )
                self.other_head = nn.Conv2d(base_ch, 2, kernel_size=1)
                self.register_buffer(
                    "density_means", torch.tensor(means, dtype=torch.float32).reshape(1, 2, 1, 1)
                )
                self.register_buffer(
                    "density_scales", torch.tensor(scales, dtype=torch.float32).reshape(1, 2, 1, 1)
                )
                self.register_buffer(
                    "density_reference_scale", torch.tensor(float(np.mean(scales)), dtype=torch.float32)
                )
                self.out = None
                self.role_grouped = None
            elif is_grouped_output_head_mode(self.mode):
                self.out = None
                self.role_grouped = build_role_grouped_conv2d_head(
                    torch=torch,
                    in_channels=base_ch,
                    output_keys=list(output_keys),
                    target_groups=dict(target_groups),
                    group_options=dict(group_options or {}),
                    with_rho_eff_head=bool(with_rho_eff_head),
                )
            else:
                self.out = nn.Conv2d(base_ch, out_ch + (1 if with_rho_eff_head else 0), kernel_size=1)
                self.role_grouped = None

        def forward(self, feat, mask=None, full_size=None):
            if self.response_scale:
                if mask is None or full_size is None:
                    raise RuntimeError("UNet++ response_scale head requires plasma mask and full grid size")
                low_size = (max(int(full_size[0]) // 4, 1), max(int(full_size[1]) // 4, 1))
                low_feat = nn.functional.interpolate(feat, size=low_size, mode="area")
                low_mask = nn.functional.interpolate(mask, size=low_size, mode="area").clamp(0.0, 1.0)
                positive_shape = nn.functional.softplus(self.density_shape(low_feat)) + 1.0e-6
                normalizer = torch.sum(positive_shape * low_mask, dim=(2, 3), keepdim=True) / torch.clamp(
                    torch.sum(low_mask, dim=(2, 3), keepdim=True), min=1.0
                )
                shape = positive_shape / torch.clamp(normalizer, min=1.0e-6)
                amplitude = nn.functional.softplus(self.density_amplitude(low_feat)) + 1.0e-6
                common = self.density_reference_scale * amplitude * shape
                imbalance = self.imbalance_fraction * torch.tanh(self.density_imbalance(low_feat))
                physical = torch.cat((common * (1.0 - imbalance), common * (1.0 + imbalance)), dim=1)
                density = (physical - self.density_means) / self.density_scales
                other = self.other_head(low_feat)
                low = low_feat.new_empty((low_feat.shape[0], len(output_keys), *low_size))
                low[:, output_keys.index("ne") : output_keys.index("ne") + 1] = density[:, 0:1]
                low[:, output_keys.index("ni") : output_keys.index("ni") + 1] = density[:, 1:2]
                low[:, output_keys.index("Te") : output_keys.index("Te") + 1] = other[:, 0:1]
                low[:, output_keys.index("phi") : output_keys.index("phi") + 1] = other[:, 1:2]
                return nn.functional.interpolate(low, size=full_size, mode="bilinear", align_corners=False)
            if self.role_grouped is not None:
                return self.role_grouped(feat)
            return self.out(feat)

        def step_reference(self):
            if self.response_scale:
                return self.other_head.weight
            if self.role_grouped is not None:
                return self.role_grouped.step_reference()
            return self.out.weight

    class _UNetPPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.response_stem = None
            if str(architecture) == "response_scale" and downsampling_stem == "learned_stride2_v1":
                self.response_stem = nn.Sequential(
                    _conv3(in_ch, base_ch, stride=2),
                    nn.GELU(),
                    _conv3(base_ch, base_ch, stride=2),
                    nn.GELU(),
                )
                x00_in_ch = base_ch
            else:
                x00_in_ch = in_ch
            self.x00 = _DoubleConv(x00_in_ch, base_ch)
            self.down0 = _Downsample(base_ch, mid_ch)
            self.x10 = _DoubleConv(mid_ch, mid_ch)
            self.down1 = _Downsample(mid_ch, bot_ch)
            self.x20 = _DoubleConv(bot_ch, bot_ch)

            self.up_10_to_00 = _UpsampleModule(mid_ch, base_ch, upsample_mode)
            self.up_20_to_10 = _UpsampleModule(bot_ch, mid_ch, upsample_mode)
            self.up_11_to_00 = _UpsampleModule(mid_ch, base_ch, upsample_mode)

            self.x01 = _DoubleConv(base_ch * 2, base_ch)
            self.x11 = _DoubleConv(mid_ch * 2, mid_ch)
            self.x02 = _DoubleConv(base_ch * 3, base_ch)
            self.head = _UNetPPHead()
            self.attention_enabled = bool(attention_enabled)
            if self.attention_enabled:
                self.gate_x00_from_x10 = _AttentionGate2dModule(base_ch, base_ch, attention_reduction)
                self.gate_x10_from_x20 = _AttentionGate2dModule(mid_ch, mid_ch, attention_reduction)
                self.gate_x00_from_x11 = _AttentionGate2dModule(base_ch, base_ch, attention_reduction)
                self.gate_x01_from_x11 = _AttentionGate2dModule(base_ch, base_ch, attention_reduction)
            else:
                self.gate_x00_from_x10 = None
                self.gate_x10_from_x20 = None
                self.gate_x00_from_x11 = None
                self.gate_x01_from_x11 = None

        def forward(self, x):
            full_size = x.shape[-2:]
            model_input = x
            response_mask = None
            if str(architecture) == "response_scale":
                native_size = (max(int(full_size[0]) // 4, 4), max(int(full_size[1]) // 4, 4))
                if mask_channel_index is not None:
                    response_mask = nn.functional.interpolate(
                        x[:, mask_channel_index : mask_channel_index + 1], size=native_size, mode="area"
                    )
                if self.response_stem is not None:
                    model_input = self.response_stem(x)
                    if model_input.shape[-2:] != native_size:
                        model_input = nn.functional.interpolate(
                            model_input, size=native_size, mode="bilinear", align_corners=False
                        )
                else:
                    model_input = nn.functional.interpolate(x, size=native_size, mode="area")
            x00 = self.x00(model_input)
            x10 = self.x10(self.down0(x00))
            x20 = self.x20(self.down1(x10))

            up_10_to_00 = self.up_10_to_00(x10, x00.shape[-2:])
            skip_x00_for_x01 = self.gate_x00_from_x10(x00, up_10_to_00) if self.attention_enabled else x00
            x01 = self.x01(torch.cat([skip_x00_for_x01, up_10_to_00], dim=1))

            up_20_to_10 = self.up_20_to_10(x20, x10.shape[-2:])
            skip_x10_for_x11 = self.gate_x10_from_x20(x10, up_20_to_10) if self.attention_enabled else x10
            x11 = self.x11(torch.cat([skip_x10_for_x11, up_20_to_10], dim=1))

            up_11_to_00 = self.up_11_to_00(x11, x00.shape[-2:])
            skip_x00_for_x02 = self.gate_x00_from_x11(x00, up_11_to_00) if self.attention_enabled else x00
            skip_x01_for_x02 = self.gate_x01_from_x11(x01, up_11_to_00) if self.attention_enabled else x01
            x02 = self.x02(torch.cat([skip_x00_for_x02, skip_x01_for_x02, up_11_to_00], dim=1))
            mask = response_mask
            if str(architecture) != "response_scale" and mask_channel_index is not None:
                mask = model_input[:, mask_channel_index : mask_channel_index + 1]
            return self.head(x02, mask=mask, full_size=full_size)

        def step_reference(self):
            return self.x00.block[0].weight, self.head.step_reference()

    return _UNetPPModel()


class UNetPPBaseline(_TorchSpatialFieldMixin):
    """Depth-2 UNet++ wrapper that matches the existing unet-like API."""

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        head_mlp: dict[str, object] | None = None,
        backend: str = "torch",
        input_feature_channels: list[str] | None = None,
        conv_cfg: dict[str, Any] | None = None,
        output_heads: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
    ) -> None:
        self.model_type, self._config_prefix = _resolve_unetpp_identity(conv_cfg)
        self._spatial_label = self.model_type
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
        if self.backend != "torch":
            raise ValueError(f"{self._config_prefix}.model_cfg.backend must be torch")

        channels = list(input_feature_channels or ["x", "y"])
        if len(channels) == 0:
            raise ValueError(f"{self._config_prefix}.input_features.features must be a non-empty list")
        if len(set(channels)) != len(channels):
            raise ValueError(f"{self._config_prefix}.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.feature_dim = int(self.input_dim + self.spatial_feature_dim)
        configure_output_head_metadata(
            self,
            output_heads=dict(output_heads or {}),
            output_keys=list(self.output_keys),
            target_role_schema=dict(target_role_schema or {}),
            cfg_prefix=self._config_prefix,
        )
        self.head_mlp = dict(head_mlp or {})
        self._static_spatial_features: np.ndarray | None = None
        self._cache: dict[str, object] = {}
        self.coord = _build_unit_coord_grid(self.grid_shape)

        self._init_torch(seed=seed, conv_cfg=conv_cfg, output_heads=self.output_heads)

    def _init_torch(self, *, seed: int, conv_cfg: dict[str, Any] | None, output_heads: dict[str, Any] | None) -> None:
        del output_heads
        from plasma_surrogate.core.torch_backend import require_torch

        torch = require_torch()
        nn = torch.nn
        cfg = dict(conv_cfg or {})
        architecture = str(cfg.get("architecture", "unetpp")).strip().lower()
        if architecture not in {"unetpp", "response_scale"}:
            raise ValueError(
                f"{self._config_prefix}.model_cfg.conv_cfg.architecture must be one of: response_scale, unetpp"
            )
        response_scale_cfg = dict(cfg.get("response_scale_cfg", {}) or {})
        downsampling_stem = str(
            response_scale_cfg.get("downsampling_stem", "fixed_area_v1")
        ).strip().lower()
        if downsampling_stem not in {"fixed_area_v1", "learned_stride2_v1"}:
            raise ValueError(
                f"{self._config_prefix}.model_cfg.conv_cfg.response_scale_cfg.downsampling_stem "
                "must be one of: fixed_area_v1, learned_stride2_v1"
            )
        radial_padding_mode = str(
            response_scale_cfg.get("radial_padding_mode", "zero")
        ).strip().lower()
        if radial_padding_mode not in {"zero", "axisymmetric_neumann"}:
            raise ValueError(
                f"{self._config_prefix}.model_cfg.conv_cfg.response_scale_cfg.radial_padding_mode "
                "must be one of: axisymmetric_neumann, zero"
            )
        base_channels = int(cfg.get("base_channels", 32))
        if base_channels <= 0:
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.base_channels must be > 0")
        depth = int(cfg.get("depth", 2))
        if depth != 2:
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.depth must be 2 in phase 1a")
        nested_skip = bool(cfg.get("nested_skip", True))
        if not nested_skip:
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.nested_skip must be true")
        upsample_mode = str(cfg.get("upsample_mode", "bilinear")).strip().lower()
        if upsample_mode not in {"bilinear", "deconv", "resize_conv"}:
            raise ValueError(
                f"{self._config_prefix}.model_cfg.conv_cfg.upsample_mode must be one of: bilinear, deconv, resize_conv"
            )
        deep_supervision_cfg = dict(cfg.get("deep_supervision", {}))
        if bool(deep_supervision_cfg.get("enabled", False)):
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.deep_supervision.enabled=true is not supported")
        attention_cfg = dict(cfg.get("attention_cfg", {}))
        attention_enabled = bool(attention_cfg.get("enabled", False))
        attention_reduction = int(attention_cfg.get("reduction", 2))
        if attention_reduction < 1:
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.attention_cfg.reduction must be >= 1")
        gate_activation = str(attention_cfg.get("gate_activation", "sigmoid")).strip().lower()
        if gate_activation != "sigmoid":
            raise ValueError(f"{self._config_prefix}.model_cfg.conv_cfg.attention_cfg.gate_activation must be sigmoid")

        mode_effective = "bilinear" if upsample_mode == "resize_conv" else upsample_mode
        mid_channels = int(max(base_channels * 2, 2))
        bot_channels = int(max(base_channels * 4, 4))
        in_channels = int(self.input_dim + self.spatial_feature_dim)

        torch.manual_seed(int(seed))
        self.torch = torch
        self._init_torch_device()
        self.net = _build_unetpp_modules(
            torch=torch,
            nn=nn,
            in_ch=in_channels,
            base_ch=base_channels,
            mid_ch=mid_channels,
            bot_ch=bot_channels,
            out_ch=self.out_channels,
            with_rho_eff_head=self.with_rho_eff_head,
            upsample_mode=mode_effective,
            attention_enabled=attention_enabled,
            attention_reduction=attention_reduction,
            head_mode=str(self.output_heads_mode),
            output_keys=list(self.output_keys),
            target_groups=dict(self.target_groups),
            group_options=dict(getattr(self, "output_head_group_options", {}) or {}),
            architecture=architecture,
            response_scale_cfg=response_scale_cfg,
            mask_channel_index=(
                self.input_dim + self.input_feature_channels.index("mask_plasma")
                if "mask_plasma" in self.input_feature_channels
                else None
            ),
        )
        self._ensure_net_device()
        self.net.train()
        self._torch_seed = int(seed)
        self._torch_base_channels = int(base_channels)
        self._torch_depth = int(depth)
        self._torch_upsample_mode = str(mode_effective)
        self._torch_nested_skip = bool(nested_skip)
        self._torch_attention_enabled = bool(attention_enabled)
        self._torch_attention_reduction = int(attention_reduction)
        self._torch_attention_gate_activation = str(gate_activation)
        self._torch_architecture = str(architecture)
        self._torch_response_scale_cfg = dict(response_scale_cfg)
        self._torch_conv_cfg = {
            "base_channels": int(base_channels),
            "depth": int(depth),
            "upsample_mode": str(mode_effective),
            "nested_skip": bool(nested_skip),
            "architecture": str(architecture),
            "response_scale_cfg": dict(response_scale_cfg),
            "attention_cfg": {
                "enabled": bool(attention_enabled),
                "reduction": int(attention_reduction),
                "gate_activation": str(gate_activation),
            },
            "deep_supervision": {"enabled": False, **deep_supervision_cfg},
        }
        self._torch_output_heads_mode = str(self.output_heads_mode)
        self._torch_last_out = None
        self._torch_last_in = None
        self.head_enabled = True
        self.head_hidden = []
        self.head_activation = "relu"
        self.head_dropout = 0.0
        self.model_type = "unetpp_attn" if attention_enabled else "unetpp"
        self._spatial_label = self.model_type
        self._config_prefix = f"train.{self.model_type}"
        self.head_arch_version = (
            "response_scale_unetpp_attn_v1"
            if architecture == "response_scale" and attention_enabled
            else (
                "response_scale_unetpp_v1"
                if architecture == "response_scale"
                else ("conv_unetpp_attn_v1" if attention_enabled else "conv_unetpp_v1")
            )
        )

    def _torch_step_reference(self):
        return self.net.step_reference()

    def forward_raw(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._forward_raw_torch(cond, training=bool(training), spatial_features=spatial_features)

    def forward(self, cond: np.ndarray, training: bool = False, spatial_features: np.ndarray | None = None) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)[:, : self.out_channels]

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
        return self._backward_raw_torch(grad_raw, lr=float(lr), apply_step=bool(apply_step))

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return self._state_dict_numpy_torch()

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        self._load_state_dict_numpy_torch(state)
