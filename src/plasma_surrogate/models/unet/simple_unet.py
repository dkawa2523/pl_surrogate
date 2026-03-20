"""UNet baseline model with optional real Conv backend (numpy/torch)."""

from __future__ import annotations

from typing import Any

import numpy as np


class UNetBaseline:
    """
    UNet baseline with two backends:
    - numpy: legacy per-pixel head on [cond + spatial features]
    - torch: real 2D conv encoder-decoder with skip connection
    """

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
    ):
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
        self.output_heads = dict(output_heads or {})
        self.output_heads_mode = str(self.output_heads.get("mode", "shared")).strip().lower()
        if self.output_heads_mode not in {"shared", "split_density_field"}:
            raise ValueError("train.unet.model_cfg.output_heads.mode must be one of: shared, split_density_field")

        self._cache: dict[str, object] = {}
        self._static_spatial_features: np.ndarray | None = None

        h, w = self.grid_shape
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        self.coord = np.stack([xv, yv], axis=-1).astype(np.float32)  # [H,W,2]

        if self.backend == "numpy":
            if self.output_heads_mode != "shared":
                raise ValueError("train.unet.model_cfg.output_heads.mode=split_density_field requires backend=torch")
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
        if output_heads_mode not in {"shared", "split_density_field"}:
            raise ValueError("train.unet.model_cfg.output_heads.mode must be one of: shared, split_density_field")
        if output_heads_mode == "split_density_field":
            allowed = {"ne", "ni", "log_ne", "log_ni", "Te", "phi"}
            missing = [name for name in self.output_keys if name not in allowed]
            if missing:
                raise ValueError(
                    "train.unet.model_cfg.output_heads.mode=split_density_field supports "
                    f"only [ne, ni, log_ne, log_ni, Te, phi], got unsupported keys: {missing}"
                )

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
            def __init__(self, base_ch: int, out_ch: int, mode: str, with_rho_eff_head: bool):
                super().__init__()
                self.mode = mode
                self.with_rho_eff_head = bool(with_rho_eff_head)
                if self.mode == "shared":
                    self.out = nn.Conv2d(base_ch, out_ch + (1 if self.with_rho_eff_head else 0), kernel_size=1)
                else:
                    self.out_density = nn.Conv2d(base_ch, 2, kernel_size=1)
                    self.out_field = nn.Conv2d(base_ch, 2, kernel_size=1)
                    self.out_rho = nn.Conv2d(base_ch, 1, kernel_size=1) if self.with_rho_eff_head else None

            def forward(self, feat, *, output_keys: list[str]):
                if self.mode == "shared":
                    return self.out(feat)
                density = self.out_density(feat)
                field = self.out_field(feat)
                name_to_tensor = {
                    "ne": density[:, 0:1],
                    "ni": density[:, 1:2],
                    "log_ne": density[:, 0:1],
                    "log_ni": density[:, 1:2],
                    "Te": field[:, 0:1],
                    "phi": field[:, 1:2],
                }
                ordered = []
                for name in output_keys:
                    if name not in name_to_tensor:
                        raise ValueError(
                            "split_density_field head requires output_keys among "
                            f"[ne, ni, log_ne, log_ni, Te, phi], got {output_keys}"
                        )
                    ordered.append(name_to_tensor[name])
                out = torch.cat(ordered, dim=1)
                if self.with_rho_eff_head and self.out_rho is not None:
                    out = torch.cat([out, self.out_rho(feat)], dim=1)
                return out

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
            ):
                super().__init__()
                if depth == 1:
                    self.backbone = _ConvUNetDepth1Backbone(in_ch, base_ch, mid_ch, upsample_mode)
                else:
                    self.backbone = _ConvUNetDepth2Backbone(in_ch, base_ch, mid_ch, bot_ch, upsample_mode)
                self.head = _ConvUNetHead(base_ch, out_ch, head_mode, with_rho_eff_head)
                self.head_mode = head_mode

            def forward(self, x, *, output_keys: list[str]):
                feat = self.backbone(x)
                return self.head(feat, output_keys=output_keys)

        self.torch = torch
        self.output_heads_mode = output_heads_mode
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
        )
        self.net.train()
        self._torch_seed = int(seed)
        self._torch_base_channels = int(base_channels)
        self._torch_depth = int(depth)
        self._torch_upsample_mode = str(upsample_mode)
        self._torch_output_heads_mode = str(output_heads_mode)
        self._torch_last_out = None
        self._torch_last_in = None
        self.head_enabled = True
        self.head_hidden = []
        self.head_activation = "relu"
        self.head_dropout = 0.0
        self.head_arch_version = "conv_unet_v1"

    def set_static_spatial_features(self, spatial_features: np.ndarray | None) -> None:
        if spatial_features is None:
            self._static_spatial_features = None
            return
        arr = np.asarray(spatial_features, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"unet spatial features must be [H,W,C], got {arr.shape}")
        h, w = self.grid_shape
        if tuple(arr.shape[:2]) != (h, w):
            raise ValueError(
                f"unet spatial features shape mismatch: expected {(h, w, arr.shape[2])}, got {arr.shape}"
            )
        if int(arr.shape[2]) != int(self.spatial_feature_dim):
            raise ValueError(
                "unet spatial feature channels mismatch: "
                f"expected {self.spatial_feature_dim}, got {int(arr.shape[2])}"
            )
        self._static_spatial_features = arr.astype(np.float32)

    def _resolve_spatial_features(self, cond: np.ndarray, spatial_features: np.ndarray | None) -> np.ndarray:
        x = np.asarray(cond, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        bsz = int(x.shape[0])
        h, w = self.grid_shape
        src = spatial_features if spatial_features is not None else self._static_spatial_features
        if src is None:
            if self.spatial_feature_dim != 2:
                raise ValueError(
                    "unet input_features requires explicit spatial features for channels "
                    f"{self.input_feature_channels}"
                )
            return np.repeat(self.coord[None, ...], bsz, axis=0).astype(np.float32)
        arr = np.asarray(src, dtype=np.float32)
        if arr.ndim == 3:
            if tuple(arr.shape[:2]) != (h, w):
                raise ValueError(f"unet spatial features shape mismatch: expected {(h, w)}, got {arr.shape[:2]}")
            if int(arr.shape[2]) != int(self.spatial_feature_dim):
                raise ValueError(
                    f"unet spatial feature channels mismatch: expected {self.spatial_feature_dim}, got {arr.shape[2]}"
                )
            return np.repeat(arr[None, ...], bsz, axis=0).astype(np.float32)
        if arr.ndim == 4:
            if tuple(arr.shape[1:3]) != (h, w):
                raise ValueError(f"unet spatial features shape mismatch: expected {(h, w)}, got {arr.shape[1:3]}")
            if int(arr.shape[3]) != int(self.spatial_feature_dim):
                raise ValueError(
                    f"unet spatial feature channels mismatch: expected {self.spatial_feature_dim}, got {arr.shape[3]}"
                )
            if int(arr.shape[0]) == bsz:
                return arr.astype(np.float32)
            if int(arr.shape[0]) == 1:
                return np.repeat(arr, bsz, axis=0).astype(np.float32)
            raise ValueError(
                f"unet spatial features batch mismatch: cond batch={bsz}, features batch={arr.shape[0]}"
            )
        raise ValueError(f"unet spatial features must be [H,W,C] or [B,H,W,C], got {arr.shape}")

    def _feature_map(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(cond, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        bsz = x.shape[0]
        h, w = self.grid_shape
        cond_map = np.repeat(x[:, None, None, :], h, axis=1)
        cond_map = np.repeat(cond_map, w, axis=2)
        spatial_map = self._resolve_spatial_features(x, spatial_features)
        return np.concatenate([cond_map, spatial_map], axis=-1).astype(np.float32)  # [B,H,W,F]

    def feature_matrix(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> np.ndarray:
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        return fmap.reshape(-1, self.feature_dim).astype(np.float32)

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

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool, spatial_features: np.ndarray | None) -> np.ndarray:
        torch = self.torch
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        x = np.moveaxis(fmap, -1, 1).astype(np.float32)  # [B,F,H,W]
        xt = torch.from_numpy(x)
        if training:
            self.net.train()
            yt = self.net(xt, output_keys=self.output_keys)
            self._torch_last_in = xt
            self._torch_last_out = yt
        else:
            self.net.eval()
            with torch.no_grad():
                yt = self.net(xt, output_keys=self.output_keys)
            self._torch_last_in = None
            self._torch_last_out = None
        return yt.detach().cpu().numpy().astype(np.float32)

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
        return self.forward_features(cond, spatial_features=spatial_features)

    def backward_raw(
        self,
        grad_raw: np.ndarray,
        *,
        lr: float,
        weight_decay: float = 0.0,
        apply_step: bool = True,
    ) -> dict[str, float]:
        if self.backend == "torch":
            return self._backward_raw_torch(grad_raw, lr=lr, apply_step=apply_step)
        return self._backward_raw_numpy(grad_raw, lr=lr, weight_decay=weight_decay)

    def _torch_step_reference(self):
        if self.backend != "torch":
            return None, None
        hidden = self.net.backbone.enc1[0].weight
        if getattr(self.net, "head_mode", "shared") == "shared":
            out = self.net.head.out.weight
        else:
            out = self.net.head.out_field.weight
        return hidden, out

    def _backward_raw_torch(self, grad_raw: np.ndarray, *, lr: float, apply_step: bool = True) -> dict[str, float]:
        if self._torch_last_out is None:
            raise RuntimeError("UNetBaseline.backward_raw called without torch forward cache")
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
                    step_hidden = float(torch.linalg.norm(dh) / max(float(torch.linalg.norm(w_hidden_prev)), 1e-12))
                if out_w_cur is not None and w_out_prev is not None:
                    do = out_w_cur.detach() - w_out_prev
                    step_out = float(torch.linalg.norm(do) / max(float(torch.linalg.norm(w_out_prev)), 1e-12))
        self._torch_last_out = None
        self._torch_last_in = None
        return {"step_rel_hidden_mean": step_hidden, "step_rel_output": step_out}

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
            for name, tensor in self.net.state_dict().items():
                out[f"torch::{name}"] = tensor.detach().cpu().numpy().astype(np.float32)
            return out
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
            torch = self.torch
            state_t = {
                k.split("torch::", 1)[1]: torch.from_numpy(np.asarray(v, dtype=np.float32))
                for k, v in state.items()
                if str(k).startswith("torch::")
            }
            if not state_t:
                raise ValueError("UNetBaseline(torch) state dict does not contain expected torch::* weights")
            self.net.load_state_dict(state_t, strict=True)
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
