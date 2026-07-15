"""POD-coarse plus coordinate residual decoder for full-field regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._auxiliary_losses import (
    coefficient_aux_loss_diagnostics,
    coefficient_aux_loss_tensor,
    project_masked_pod_coefficients_torch,
)
from plasma_surrogate.models._torch_spatial_common import (
    _load_state_dict_numpy_torch,
    _resolve_batched_spatial_features,
    _resolve_torch_device,
    _state_dict_numpy_torch,
    _validate_static_spatial_features,
)
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODBasisBundle
from plasma_surrogate.models.mlp.coord_mlp_torch import (
    _activation_module,
    _resolve_hidden_list,
    _softplus_inverse,
)

COORD_MLP_POD_RESIDUAL_IMPL_VERSION = "coord_mlp_pod_residual_v1"

try:  # pragma: no cover - torch availability is checked at runtime.
    import torch as _torch_mod
    import torch.nn as _torch_nn
except Exception:  # pragma: no cover - torch is optional at import time.
    _torch_mod = None
    _torch_nn = None


def _cfg_prefix() -> str:
    return "train.coord_mlp_pod_residual"


def _resolve_positive_float(raw: Any, *, default: float, cfg_key: str) -> float:
    value = float(default if raw is None else raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{cfg_key} must be > 0")
    return float(value)


def _resolve_nonnegative_float(raw: Any, *, default: float, cfg_key: str) -> float:
    value = float(default if raw is None else raw)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{cfg_key} must be finite and >= 0")
    return float(value)


def _resolve_basis_cfg(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    prefix = _cfg_prefix()
    cfg = dict(raw_cfg.get("basis", {}))
    rank = int(cfg.get("rank", 32))
    if rank < 1:
        raise ValueError(f"{prefix}.model_cfg.basis.rank must be >= 1")
    fit_scope = str(cfg.get("fit_scope", "train_only")).strip().lower()
    if fit_scope != "train_only":
        raise ValueError(f"{prefix}.model_cfg.basis.fit_scope must be train_only")
    per_var = bool(cfg.get("per_var", True))
    if not per_var:
        raise ValueError(f"{prefix}.model_cfg.basis.per_var must be true")
    return {
        "rank": int(rank),
        "fit_scope": "train_only",
        "per_var": True,
        "center": bool(cfg.get("center", True)),
    }


def normalize_coord_mlp_pod_residual_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the public config for ``coord_mlp_pod_residual``.

    The model intentionally exposes only a small set of knobs: POD rank,
    condition branch size, residual branch size, and xy Fourier encoding.
    """

    cfg = dict(raw_cfg or {})
    prefix = _cfg_prefix()
    point_raw = dict(cfg.get("point_encoder", {}))
    xy_freq = int(point_raw.get("xy_fourier_frequencies", 8))
    if xy_freq < 1:
        raise ValueError(f"{prefix}.model_cfg.point_encoder.xy_fourier_frequencies must be >= 1")
    xy_scale = _resolve_positive_float(
        point_raw.get("xy_frequency_scale"),
        default=10.0,
        cfg_key=f"{prefix}.model_cfg.point_encoder.xy_frequency_scale",
    )
    latent_dim = int(cfg.get("latent_dim", 128))
    if latent_dim < 1:
        raise ValueError(f"{prefix}.model_cfg.latent_dim must be >= 1")
    coeff_loss_weight = _resolve_nonnegative_float(
        cfg.get("coeff_loss_weight"),
        default=0.1,
        cfg_key=f"{prefix}.model_cfg.coeff_loss_weight",
    )
    residual_scale_init = _resolve_positive_float(
        cfg.get("residual_scale_init"),
        default=0.05,
        cfg_key=f"{prefix}.model_cfg.residual_scale_init",
    )
    residual_activation = str(cfg.get("residual_activation", "gelu")).strip().lower()
    # Validate activation through the shared helper without creating a module here.
    if residual_activation not in {"relu", "gelu", "tanh"}:
        raise ValueError(f"{prefix}.model_cfg.residual_activation must be one of: relu, gelu, tanh")
    encoder_cfg = {
        "type": "xy_fourier_aux_raw",
        "xy_fourier_frequencies": int(xy_freq),
        "xy_frequency_scale": float(xy_scale),
        "include_xy_raw": bool(point_raw.get("include_xy_raw", True)),
        "include_aux_raw": bool(point_raw.get("include_aux_raw", True)),
    }
    return {
        "basis": _resolve_basis_cfg(cfg),
        "cond_hidden": _resolve_hidden_list(
            cfg.get("cond_hidden"),
            default=[128, 128],
            cfg_key=f"{prefix}.model_cfg.cond_hidden",
        ),
        "latent_dim": int(latent_dim),
        "residual_hidden": _resolve_hidden_list(
            cfg.get("residual_hidden"),
            default=[256, 256, 256],
            cfg_key=f"{prefix}.model_cfg.residual_hidden",
        ),
        "residual_activation": residual_activation,
        "point_encoder": encoder_cfg,
        "decoder_input_norm": {
            "enabled": bool(dict(cfg.get("decoder_input_norm", {})).get("enabled", True)),
        },
        "coeff_loss_weight": float(coeff_loss_weight),
        "residual_scale_init": float(residual_scale_init),
        # Keep the resolved contract shape close to the existing Coord MLP family.
        "embedding": dict(encoder_cfg),
        "siren": {"enabled": False},
    }


if _torch_nn is not None and _torch_mod is not None:  # pragma: no branch - definition guard only.

    class _XyFourierAuxPointEncoder(_torch_nn.Module):
        def __init__(
            self,
            *,
            spatial_feature_dim: int,
            xy_indices: tuple[int, int],
            include_xy_raw: bool,
            include_aux_raw: bool,
            n_frequencies: int,
            frequency_scale: float,
        ) -> None:
            super().__init__()
            self.spatial_feature_dim = int(spatial_feature_dim)
            self.xy_indices = tuple(int(v) for v in xy_indices)
            self.include_xy_raw = bool(include_xy_raw)
            self.include_aux_raw = bool(include_aux_raw)
            self.n_frequencies = int(n_frequencies)
            scales = [
                (float(2**k) * float(np.pi)) / float(frequency_scale)
                for k in range(self.n_frequencies)
            ]
            self.register_buffer(
                "_fourier_scales",
                _torch_mod.as_tensor(np.asarray(scales, dtype=np.float32)),
                persistent=False,
            )

        def output_dim(self) -> int:
            raw_xy = 2 if self.include_xy_raw else 0
            aux_dim = max(self.spatial_feature_dim - 2, 0) if self.include_aux_raw else 0
            return int(raw_xy + aux_dim + (2 * self.n_frequencies * 2))

        def forward(self, spatial_t):
            xy = spatial_t[..., list(self.xy_indices)]
            pieces = []
            if self.include_xy_raw:
                pieces.append(xy)
            aux_indices = [idx for idx in range(self.spatial_feature_dim) if idx not in set(self.xy_indices)]
            if self.include_aux_raw and aux_indices:
                pieces.append(spatial_t[..., aux_indices])
            scaled = xy.unsqueeze(-2) * self._fourier_scales.view(1, 1, 1, -1, 1)
            pieces.append(_torch_mod.sin(scaled).reshape(*xy.shape[:3], -1))
            pieces.append(_torch_mod.cos(scaled).reshape(*xy.shape[:3], -1))
            return _torch_mod.cat(pieces, dim=-1)


    class _CoordMLPPODResidualNet(_torch_nn.Module):
        def __init__(
            self,
            *,
            cond_encoder: _torch_nn.Module,
            coeff_heads: _torch_nn.ModuleDict,
            point_encoder: _torch_nn.Module,
            residual_decoder: _torch_nn.Module,
            decoder_input_norm: _torch_nn.Module | None,
            latent_dim: int,
            out_channels: int,
            residual_scale_init: float,
        ) -> None:
            super().__init__()
            self.cond_encoder = cond_encoder
            self.coeff_heads = coeff_heads
            self.point_encoder = point_encoder
            self.residual_decoder = residual_decoder
            self.decoder_input_norm = decoder_input_norm
            self.latent_dim = int(latent_dim)
            self.out_channels = int(out_channels)
            self._residual_scale_raw = _torch_nn.Parameter(
                _torch_mod.tensor([_softplus_inverse(float(residual_scale_init))], dtype=_torch_mod.float32)
            )

        def residual_scale(self):
            return _torch_nn.functional.softplus(self._residual_scale_raw)

        def encode_cond(self, cond_t):
            return self.cond_encoder(cond_t)

        def predict_coeff_norm(self, latent_t, basis_keys: list[str]):
            return _torch_mod.cat([self.coeff_heads[str(name)](latent_t) for name in basis_keys], dim=1)

        def decode_residual(self, latent_t, spatial_t):
            bsz, h, w, _ = spatial_t.shape
            point_t = self.point_encoder(spatial_t)
            latent_grid = latent_t[:, None, None, :].expand(bsz, h, w, self.latent_dim)
            dec_in = _torch_mod.cat([latent_grid, point_t], dim=-1)
            flat = dec_in.reshape(bsz * h * w, dec_in.shape[-1])
            if self.decoder_input_norm is not None:
                flat = self.decoder_input_norm(flat)
            residual = self.residual_decoder(flat).reshape(bsz, h, w, self.out_channels)
            return residual.permute(0, 3, 1, 2).contiguous()


else:  # pragma: no cover - only used when torch is unavailable at import time.
    _XyFourierAuxPointEncoder = None
    _CoordMLPPODResidualNet = None


def _build_mlp(
    *,
    torch: Any,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation_name: str,
):
    nn = torch.nn
    layers: list[Any] = []
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    last = len(dims) - 2
    for idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(int(fan_in), int(fan_out)))
        if idx != last:
            layers.append(_activation_module(nn, activation_name, cfg_prefix=_cfg_prefix()))
    return nn.Sequential(*layers)


class CoordMLPPODResidual:
    """Condition-POD coarse model with a geometry-aware coordinate residual."""

    model_type = "coord_mlp_pod_residual"
    coord_mlp_impl_version = COORD_MLP_POD_RESIDUAL_IMPL_VERSION
    requires_spatial_features = True
    requires_scaled_spatial_features = True

    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 4,
        output_keys: list[str] | None = None,
        input_feature_channels: list[str] | None = None,
        pod_basis_bundle: PODBasisBundle,
        model_cfg: dict[str, Any] | None = None,
        seed: int = 0,
        backend: str = "torch",
    ) -> None:
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.out_channels = int(out_channels)
        self.output_keys = list(output_keys or [f"out_{i}" for i in range(self.out_channels)])[: self.out_channels]
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError(f"{_cfg_prefix()}.model_cfg.backend must be torch")
        self.model_cfg = normalize_coord_mlp_pod_residual_cfg(model_cfg)
        channels = list(input_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
        if len(channels) == 0:
            raise ValueError(f"{_cfg_prefix()}.input_features.features must be a non-empty list")
        if len(set(str(v) for v in channels)) != len(channels):
            raise ValueError(f"{_cfg_prefix()}.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        if "x" in self.input_feature_channels and "y" in self.input_feature_channels:
            self.xy_indices = (
                self.input_feature_channels.index("x"),
                self.input_feature_channels.index("y"),
            )
        elif self.spatial_feature_dim >= 2:
            self.xy_indices = (0, 1)
        else:
            raise ValueError(f"{_cfg_prefix()}.input_features.features must include at least x/y-like channels")
        self._static_spatial_features: np.ndarray | None = None
        self._torch_last_out = None
        self._torch_last_coeff_norm = None

        missing_basis = [name for name in self.output_keys if name not in pod_basis_bundle.basis_by_var]
        missing_mean = [name for name in self.output_keys if name not in pod_basis_bundle.mean_by_var]
        if missing_basis or missing_mean:
            raise ValueError(
                "coord_mlp_pod_residual missing POD basis/mean for output keys: "
                f"basis={missing_basis}, mean={missing_mean}"
            )
        h, w = self.grid_shape
        self.basis_keys = [str(v) for v in self.output_keys]
        self.basis_rank_by_var: dict[str, int] = {}
        total_rank = 0
        for name in self.basis_keys:
            basis_arr = np.asarray(pod_basis_bundle.basis_by_var[name], dtype=np.float32)
            mean_arr = np.asarray(pod_basis_bundle.mean_by_var[name], dtype=np.float32)
            if basis_arr.ndim != 3 or tuple(basis_arr.shape[1:]) != (h, w):
                raise ValueError(f"coord_mlp_pod_residual basis[{name}] must be [rank,H,W], got {basis_arr.shape}")
            if mean_arr.shape != (h, w):
                raise ValueError(f"coord_mlp_pod_residual mean[{name}] must be [H,W], got {mean_arr.shape}")
            rank_eff = int(pod_basis_bundle.rank_by_var[name])
            self.basis_rank_by_var[name] = rank_eff
            total_rank += rank_eff
        self.coeff_dim = int(total_rank)
        self._coeff_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in self.basis_keys:
            stop = offset + int(self.basis_rank_by_var[name])
            self._coeff_slices[name] = (offset, stop)
            offset = stop

        from plasma_surrogate.core.torch_backend import require_torch

        self.torch = require_torch()
        self.device = _resolve_torch_device(self.torch)
        self.torch.manual_seed(int(seed))
        nn = self.torch.nn
        point_cfg = dict(self.model_cfg["point_encoder"])
        if _XyFourierAuxPointEncoder is None or _CoordMLPPODResidualNet is None:
            raise RuntimeError("CoordMLPPODResidual requires torch modules to be available")
        point_encoder = _XyFourierAuxPointEncoder(
            spatial_feature_dim=self.spatial_feature_dim,
            xy_indices=self.xy_indices,
            include_xy_raw=bool(point_cfg["include_xy_raw"]),
            include_aux_raw=bool(point_cfg["include_aux_raw"]),
            n_frequencies=int(point_cfg["xy_fourier_frequencies"]),
            frequency_scale=float(point_cfg["xy_frequency_scale"]),
        )
        cond_encoder = _build_mlp(
            torch=self.torch,
            input_dim=self.input_dim,
            hidden_dims=list(self.model_cfg["cond_hidden"]),
            output_dim=int(self.model_cfg["latent_dim"]),
            activation_name="gelu",
        )
        residual_input_dim = int(self.model_cfg["latent_dim"]) + int(point_encoder.output_dim())
        residual_decoder = _build_mlp(
            torch=self.torch,
            input_dim=residual_input_dim,
            hidden_dims=list(self.model_cfg["residual_hidden"]),
            output_dim=self.out_channels,
            activation_name=str(self.model_cfg["residual_activation"]),
        )
        decoder_input_norm = (
            nn.LayerNorm(residual_input_dim) if bool(dict(self.model_cfg["decoder_input_norm"]).get("enabled", True)) else None
        )
        coeff_heads = nn.ModuleDict(
            {
                str(name): nn.Linear(int(self.model_cfg["latent_dim"]), int(self.basis_rank_by_var[name]))
                for name in self.basis_keys
            }
        )
        self.net = _CoordMLPPODResidualNet(
            cond_encoder=cond_encoder,
            coeff_heads=coeff_heads,
            point_encoder=point_encoder,
            residual_decoder=residual_decoder,
            decoder_input_norm=decoder_input_norm,
            latent_dim=int(self.model_cfg["latent_dim"]),
            out_channels=self.out_channels,
            residual_scale_init=float(self.model_cfg["residual_scale_init"]),
        )
        for name in self.basis_keys:
            self.net.register_buffer(
                f"_pod_basis__{name}",
                self.torch.as_tensor(pod_basis_bundle.basis_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
            self.net.register_buffer(
                f"_pod_mean__{name}",
                self.torch.as_tensor(pod_basis_bundle.mean_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
            self.net.register_buffer(
                f"_pod_coeff_std__{name}",
                self.torch.as_tensor(pod_basis_bundle.coeff_std_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
        self.net.to(self.device)

    def _basis_tensor(self, name: str):
        return getattr(self.net, f"_pod_basis__{name}")

    def _mean_tensor(self, name: str):
        return getattr(self.net, f"_pod_mean__{name}")

    def _coeff_std_tensor(self, name: str):
        return getattr(self.net, f"_pod_coeff_std__{name}")

    def basis_bundle_numpy(self) -> PODBasisBundle:
        return PODBasisBundle.from_dicts(
            basis_by_var={
                name: self._basis_tensor(name).detach().cpu().numpy().astype(np.float32)
                for name in self.basis_keys
            },
            mean_by_var={
                name: self._mean_tensor(name).detach().cpu().numpy().astype(np.float32)
                for name in self.basis_keys
            },
            rank_by_var=dict(self.basis_rank_by_var),
            coeff_std_by_var={
                name: self._coeff_std_tensor(name).detach().cpu().numpy().astype(np.float32)
                for name in self.basis_keys
            },
        )

    def set_static_spatial_features(self, spatial_features: np.ndarray | None) -> None:
        if spatial_features is None:
            self._static_spatial_features = None
            return
        self._static_spatial_features = _validate_static_spatial_features(
            spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=self.model_type,
        )

    def _resolve_spatial_features(self, cond: np.ndarray, spatial_features: np.ndarray | None) -> np.ndarray:
        return _resolve_batched_spatial_features(
            cond,
            spatial_features,
            static_spatial_features=self._static_spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=self.model_type,
            explicit_requirement_message=(
                "coord_mlp_pod_residual requires geom_feature_pack spatial features; "
                "set_static_spatial_features(...) or pass spatial_features=..."
            ),
        )

    def _reconstruct_from_coeff_norm(self, coeff_norm_t):
        fields = []
        for name in self.basis_keys:
            start, stop = self._coeff_slices[name]
            coeff_norm = coeff_norm_t[:, start:stop]
            coeff_raw = coeff_norm * self._coeff_std_tensor(name).reshape(1, -1)
            basis_t = self._basis_tensor(name).reshape(int(stop - start), -1)
            mean_t = self._mean_tensor(name).reshape(1, -1)
            field_flat = self.torch.matmul(coeff_raw, basis_t) + mean_t
            fields.append(field_flat.reshape(int(coeff_norm_t.shape[0]), *self.grid_shape))
        return self.torch.stack(fields, dim=1)

    def _projection_mask_torch(self, supervised_mask: Any | None, target_t):
        if supervised_mask is None:
            if bool(self.torch.any(~self.torch.isfinite(target_t)).detach().cpu().item()):
                raise ValueError("coord_mlp_pod_residual target_raw must contain only finite values")
            return None
        mask = self.torch.as_tensor(supervised_mask, dtype=self.torch.float32, device=target_t.device)
        if mask.ndim == 2:
            mask = mask[None, ...]
        if mask.ndim == 4 and int(mask.shape[1]) == 1:
            mask = mask[:, 0]
        if mask.ndim != 3:
            raise ValueError(
                "coord_mlp_pod_residual supervised_mask must be [H,W], [B,H,W], or [B,1,H,W]"
            )
        if int(mask.shape[0]) == 1 and int(target_t.shape[0]) > 1:
            mask = mask.expand(int(target_t.shape[0]), -1, -1)
        expected = (int(target_t.shape[0]), *self.grid_shape)
        if tuple(int(value) for value in mask.shape) != expected:
            raise ValueError(
                "coord_mlp_pod_residual supervised_mask mismatch: "
                f"expected={expected}, got={tuple(mask.shape)}"
            )
        if bool(self.torch.any(~self.torch.isfinite(mask)).detach().cpu().item()):
            raise ValueError("coord_mlp_pod_residual supervised_mask must contain only finite values")
        active = mask > 0.0
        invalid = active[:, None, :, :] & ~self.torch.isfinite(target_t)
        if bool(self.torch.any(invalid).detach().cpu().item()):
            raise ValueError("coord_mlp_pod_residual target_raw is non-finite inside supervised_mask")
        return active.reshape(int(target_t.shape[0]), -1)

    def _project_target_coeff_norm_torch(self, target_t, *, supervised_mask: Any | None = None):
        active_flat = self._projection_mask_torch(supervised_mask, target_t)
        chunks = []
        for var_idx, name in enumerate(self.basis_keys):
            flat = target_t[:, var_idx].reshape(int(target_t.shape[0]), -1)
            centered = flat - self._mean_tensor(name).reshape(1, -1)
            basis_t = self._basis_tensor(name).reshape(int(self.basis_rank_by_var[name]), -1)
            coeff_raw = project_masked_pod_coefficients_torch(
                centered,
                basis_t,
                active_mask=active_flat,
            )
            chunks.append(coeff_raw / self._coeff_std_tensor(name).reshape(1, -1))
        return self.torch.cat(chunks, dim=1)

    def _resolve_coeff_loss_weight(self, loss_cfg: dict[str, Any] | None) -> float:
        weight = float(self.model_cfg["coeff_loss_weight"])
        if isinstance(loss_cfg, dict):
            weight = float(
                dict(loss_cfg.get("coord_mlp_pod_residual", {})).get("coeff_loss_weight", weight)
            )
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("coord_mlp_pod_residual coeff_loss_weight must be finite and >= 0")
        return float(weight)

    def _cached_coeff_loss_tensor(
        self,
        target_raw: np.ndarray,
        *,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ):
        if self._torch_last_coeff_norm is None:
            raise RuntimeError("CoordMLPPODResidual auxiliary loss requires a cached training forward")
        target_arr = np.asarray(target_raw, dtype=np.float32)
        if target_arr.ndim != 4:
            raise ValueError(f"coord_mlp_pod_residual target_raw must be [B,C,H,W], got {target_arr.shape}")
        if int(target_arr.shape[1]) != len(self.basis_keys):
            raise ValueError(
                "coord_mlp_pod_residual target_raw channel mismatch: "
                f"expected {len(self.basis_keys)}, got {target_arr.shape[1]}"
            )
        if int(target_arr.shape[0]) != int(self._torch_last_coeff_norm.shape[0]):
            raise ValueError(
                "coord_mlp_pod_residual target_raw batch mismatch: "
                f"expected {int(self._torch_last_coeff_norm.shape[0])}, got {int(target_arr.shape[0])}"
            )
        if tuple(int(v) for v in target_arr.shape[2:]) != tuple(self.grid_shape):
            raise ValueError(
                "coord_mlp_pod_residual target_raw grid mismatch: "
                f"expected {self.grid_shape}, got {target_arr.shape[2:]}"
            )
        target_t = self.torch.as_tensor(target_arr, dtype=self.torch.float32, device=self.device)
        coeff_target = self._project_target_coeff_norm_torch(
            target_t,
            supervised_mask=supervised_mask,
        )
        return coefficient_aux_loss_tensor(
            self._torch_last_coeff_norm,
            coeff_target,
            basis_keys=self.basis_keys,
            coeff_slices=self._coeff_slices,
            loss_cfg=loss_cfg,
        )

    def auxiliary_loss_diagnostics(
        self,
        target_raw: np.ndarray,
        *,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Inspect cached coefficient loss without changing caches or gradients."""

        weight = self._resolve_coeff_loss_weight(loss_cfg)
        with self.torch.no_grad():
            coeff_loss, by_target, by_group = self._cached_coeff_loss_tensor(
                target_raw,
                supervised_mask=supervised_mask,
                loss_cfg=loss_cfg,
            )
            coeff_loss_value = float(coeff_loss.detach().cpu().item())
        return coefficient_aux_loss_diagnostics(
            coeff_loss=coeff_loss_value,
            coeff_loss_weight=weight,
            coeff_loss_by_target={
                name: float(value.detach().cpu().item()) for name, value in by_target.items()
            },
            coeff_loss_by_group={
                name: float(value.detach().cpu().item()) for name, value in by_group.items()
            },
        )

    def evaluate_auxiliary_losses(
        self,
        cond: np.ndarray,
        target_raw: np.ndarray,
        *,
        spatial_features: np.ndarray | None = None,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Evaluate coefficient supervision without touching training state.

        Unlike :meth:`auxiliary_loss_diagnostics`, this validation-time API
        predicts fresh coefficients from ``cond`` and therefore does not
        require, consume, or replace a cached training forward.
        """

        weight = self._resolve_coeff_loss_weight(loss_cfg)
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        if cond_arr.ndim != 2:
            raise ValueError(
                "coord_mlp_pod_residual cond must be [B,input_dim], "
                f"got {cond_arr.shape}"
            )
        if int(cond_arr.shape[0]) < 1:
            raise ValueError("coord_mlp_pod_residual cond batch must be non-empty")
        if int(cond_arr.shape[1]) != int(self.input_dim):
            raise ValueError(
                "coord_mlp_pod_residual cond feature mismatch: "
                f"expected {self.input_dim}, got {cond_arr.shape[1]}"
            )
        if not bool(np.isfinite(cond_arr).all()):
            raise ValueError("coord_mlp_pod_residual cond must contain only finite values")

        target_arr = np.asarray(target_raw, dtype=np.float32)
        if target_arr.ndim != 4:
            raise ValueError(
                "coord_mlp_pod_residual target_raw must be [B,C,H,W], "
                f"got {target_arr.shape}"
            )
        if int(target_arr.shape[0]) != int(cond_arr.shape[0]):
            raise ValueError(
                "coord_mlp_pod_residual target_raw batch mismatch: "
                f"expected {cond_arr.shape[0]}, got {target_arr.shape[0]}"
            )
        if int(target_arr.shape[1]) != len(self.basis_keys):
            raise ValueError(
                "coord_mlp_pod_residual target_raw channel mismatch: "
                f"expected {len(self.basis_keys)}, got {target_arr.shape[1]}"
            )
        if tuple(int(v) for v in target_arr.shape[2:]) != tuple(self.grid_shape):
            raise ValueError(
                "coord_mlp_pod_residual target_raw grid mismatch: "
                f"expected {self.grid_shape}, got {target_arr.shape[2:]}"
            )

        # Spatial features do not enter the coefficient head, but validating
        # them here keeps this public evaluation entry point consistent with
        # the model's complete input contract.
        self._resolve_spatial_features(cond_arr, spatial_features)

        cached_out = self._torch_last_out
        cached_coeff_norm = self._torch_last_coeff_norm
        module_modes = [(module, bool(module.training)) for module in self.net.modules()]
        try:
            self.net.eval()
            with self.torch.no_grad():
                cond_t = self.torch.as_tensor(
                    cond_arr,
                    dtype=self.torch.float32,
                    device=self.device,
                )
                target_t = self.torch.as_tensor(
                    target_arr,
                    dtype=self.torch.float32,
                    device=self.device,
                )
                latent_t = self.net.encode_cond(cond_t)
                coeff_pred_t = self.net.predict_coeff_norm(latent_t, self.basis_keys)
                coeff_target_t = self._project_target_coeff_norm_torch(
                    target_t,
                    supervised_mask=supervised_mask,
                )
                coeff_loss, by_target, by_group = coefficient_aux_loss_tensor(
                    coeff_pred_t,
                    coeff_target_t,
                    basis_keys=self.basis_keys,
                    coeff_slices=self._coeff_slices,
                    loss_cfg=loss_cfg,
                )
                coeff_loss_value = float(coeff_loss.detach().cpu().item())
                target_values = {
                    name: float(value.detach().cpu().item()) for name, value in by_target.items()
                }
                group_values = {
                    name: float(value.detach().cpu().item()) for name, value in by_group.items()
                }
        finally:
            for module, was_training in module_modes:
                module.training = was_training
            self._torch_last_out = cached_out
            self._torch_last_coeff_norm = cached_coeff_norm

        return coefficient_aux_loss_diagnostics(
            coeff_loss=coeff_loss_value,
            coeff_loss_weight=weight,
            coeff_loss_by_target=target_values,
            coeff_loss_by_group=group_values,
        )

    def forward_raw(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        spatial_arr = self._resolve_spatial_features(cond_arr, spatial_features)
        cond_t = self.torch.as_tensor(cond_arr, dtype=self.torch.float32, device=self.device)
        spatial_t = self.torch.as_tensor(spatial_arr, dtype=self.torch.float32, device=self.device)
        if bool(training):
            self.net.train()
            latent = self.net.encode_cond(cond_t)
            coeff_norm = self.net.predict_coeff_norm(latent, self.basis_keys)
            coarse = self._reconstruct_from_coeff_norm(coeff_norm)
            residual = self.net.decode_residual(latent, spatial_t)
            self._torch_last_coeff_norm = coeff_norm
            self._torch_last_out = coarse + self.net.residual_scale() * residual
            return self._torch_last_out.detach().cpu().numpy().astype(np.float32)
        self.net.eval()
        with self.torch.no_grad():
            latent = self.net.encode_cond(cond_t)
            coeff_norm = self.net.predict_coeff_norm(latent, self.basis_keys)
            coarse = self._reconstruct_from_coeff_norm(coeff_norm)
            residual = self.net.decode_residual(latent, spatial_t)
            out = coarse + self.net.residual_scale() * residual
        self._torch_last_coeff_norm = None
        self._torch_last_out = None
        return out.detach().cpu().numpy().astype(np.float32)

    def forward(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)

    def forward_features(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        y = self.forward_raw(cond, training=False, spatial_features=spatial_features)
        return {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}

    def predict_fields(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return self.forward_features(cond, spatial_features=spatial_features)

    def _torch_step_reference(self):
        first = None
        last = None
        for module in self.net.cond_encoder:
            if hasattr(module, "weight"):
                first = module.weight
                break
        for module in reversed(list(self.net.residual_decoder)):
            if hasattr(module, "weight"):
                last = module.weight
                break
        return first, last

    def backward_raw(
        self,
        grad_raw: np.ndarray,
        *,
        lr: float,
        weight_decay: float = 0.0,
        apply_step: bool = True,
        target_raw: np.ndarray | None = None,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del weight_decay
        if self._torch_last_out is None or self._torch_last_coeff_norm is None:
            raise RuntimeError("CoordMLPPODResidual.backward_raw called without torch forward cache")
        grad_t = self.torch.as_tensor(
            np.asarray(grad_raw, dtype=np.float32),
            dtype=getattr(self._torch_last_out, "dtype", None) or self.torch.float32,
            device=getattr(self._torch_last_out, "device", None),
        )
        params = [p for p in self.net.parameters() if p.requires_grad]
        hidden_w, out_w = self._torch_step_reference()
        with self.torch.no_grad():
            hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
            out_prev = out_w.detach().clone() if out_w is not None else None
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        coeff_loss_weight = self._resolve_coeff_loss_weight(loss_cfg)
        use_coeff_loss = bool(target_raw is not None and coeff_loss_weight > 0.0)
        coeff_loss = None
        coeff_loss_by_target: dict[str, Any] = {}
        coeff_loss_by_group: dict[str, Any] = {}
        if target_raw is not None:
            coeff_loss, coeff_loss_by_target, coeff_loss_by_group = self._cached_coeff_loss_tensor(
                target_raw,
                supervised_mask=supervised_mask,
                loss_cfg=loss_cfg,
            )
        self._torch_last_out.backward(grad_t, retain_graph=use_coeff_loss)
        coeff_loss_val = 0.0
        if coeff_loss is not None:
            coeff_loss_val = float(coeff_loss.detach().cpu().item())
        if use_coeff_loss and coeff_loss is not None:
            (float(coeff_loss_weight) * coeff_loss).backward()
        step_hidden = 0.0
        step_out = 0.0
        if bool(apply_step):
            with self.torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p -= float(lr) * p.grad
                hidden_cur, out_cur = self._torch_step_reference()
                if hidden_cur is not None and hidden_prev is not None:
                    step_hidden = float(
                        self.torch.linalg.norm(hidden_cur.detach() - hidden_prev)
                        / max(float(self.torch.linalg.norm(hidden_prev)), 1.0e-12)
                    )
                if out_cur is not None and out_prev is not None:
                    step_out = float(
                        self.torch.linalg.norm(out_cur.detach() - out_prev)
                        / max(float(self.torch.linalg.norm(out_prev)), 1.0e-12)
                    )
        self._torch_last_out = None
        self._torch_last_coeff_norm = None
        return {
            "step_rel_hidden_mean": float(step_hidden),
            "step_rel_output": float(step_out),
            **coefficient_aux_loss_diagnostics(
                coeff_loss=coeff_loss_val,
                coeff_loss_weight=coeff_loss_weight,
                coeff_loss_by_target={
                    name: float(value.detach().cpu().item())
                    for name, value in coeff_loss_by_target.items()
                },
                coeff_loss_by_group={
                    name: float(value.detach().cpu().item())
                    for name, value in coeff_loss_by_group.items()
                },
            ),
        }

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        state = _state_dict_numpy_torch(self.net)
        basis_bundle = self.basis_bundle_numpy()
        for name in self.basis_keys:
            state[f"basis::{name}"] = np.asarray(basis_bundle.basis_by_var[name], dtype=np.float32).copy()
            state[f"mean::{name}"] = np.asarray(basis_bundle.mean_by_var[name], dtype=np.float32).copy()
            state[f"coeff_std::{name}"] = np.asarray(basis_bundle.coeff_std_by_var[name], dtype=np.float32).copy()
        return state

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message="CoordMLPPODResidual state dict does not contain expected torch::* weights",
            device=self.device,
        )
        self.net.to(self.device)
        for name in self.basis_keys:
            for prefix, tensor_getter in (
                ("basis", self._basis_tensor),
                ("mean", self._mean_tensor),
                ("coeff_std", self._coeff_std_tensor),
            ):
                key = f"{prefix}::{name}"
                if key not in state:
                    raise ValueError(f"coord_mlp_pod_residual checkpoint missing {key}")
                arr = np.asarray(state[key], dtype=np.float32)
                tensor_getter(name).data.copy_(
                    self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device)
                )

    def to_meta(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "coord_mlp_impl_version": self.coord_mlp_impl_version,
            "input_dim": int(self.input_dim),
            "grid_shape": [int(v) for v in self.grid_shape],
            "out_channels": int(self.out_channels),
            "output_keys": list(self.output_keys),
            "backend": str(self.backend),
            "input_feature_channels": list(self.input_feature_channels),
            "model_cfg": dict(self.model_cfg),
            "basis_keys": list(self.basis_keys),
            "basis_rank_by_var": {str(k): int(v) for k, v in self.basis_rank_by_var.items()},
        }


__all__ = [
    "COORD_MLP_POD_RESIDUAL_IMPL_VERSION",
    "CoordMLPPODResidual",
    "normalize_coord_mlp_pod_residual_cfg",
]
