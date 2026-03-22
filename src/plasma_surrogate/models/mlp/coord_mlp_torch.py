"""Torch coord-MLP decoder family for Fourier and SIREN full-grid decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import (
    _backward_raw_torch_step,
    _build_unit_coord_grid,
    _load_state_dict_numpy_torch,
    _resolve_batched_spatial_features,
    _state_dict_numpy_torch,
    _validate_static_spatial_features,
)

try:  # pragma: no cover - exercised through runtime-backed tests.
    import torch as _torch_mod
    import torch.nn as _torch_nn
except Exception:  # pragma: no cover - torch is optional at import time.
    _torch_mod = None
    _torch_nn = None


def _cfg_prefix(model_type: str) -> str:
    return f"train.{str(model_type).strip().lower()}"


def _activation_module(nn: Any, name: str, *, cfg_prefix: str):
    mode = str(name).strip().lower()
    if mode == "relu":
        return nn.ReLU()
    if mode == "gelu":
        return nn.GELU()
    if mode == "tanh":
        return nn.Tanh()
    raise ValueError(f"{cfg_prefix}.model_cfg.decoder_activation must be one of: relu, gelu, tanh")


def _resolve_hidden_list(raw: Any, *, default: list[int], cfg_key: str) -> list[int]:
    values = list(default if raw is None else raw)
    if len(values) == 0:
        raise ValueError(f"{cfg_key} must be a non-empty list")
    out = [int(v) for v in values]
    if any(v <= 0 for v in out):
        raise ValueError(f"{cfg_key} must contain only positive integers")
    return out


def _resolve_positive_float(raw: Any, *, default: float, cfg_key: str) -> float:
    value = float(default if raw is None else raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{cfg_key} must be > 0")
    return float(value)


def _resolve_strict_bool(raw: Any, *, default: bool, cfg_key: str) -> bool:
    value = default if raw is None else raw
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{cfg_key} must be a boolean")


def _softplus_inverse(value: float) -> float:
    x = float(value)
    # Stable inverse of softplus for strictly positive targets.
    if x > 20.0:
        return x
    return float(np.log(np.expm1(x)))


def _siren_init_linear(*, torch: Any, linear: Any, w0: float, is_first: bool) -> None:
    fan_in = int(linear.in_features)
    if fan_in <= 0:
        raise ValueError("SIREN layer fan_in must be > 0")
    bound = (1.0 / float(fan_in)) if bool(is_first) else (np.sqrt(6.0 / float(fan_in)) / float(w0))
    with torch.no_grad():
        linear.weight.uniform_(-float(bound), float(bound))
        if linear.bias is not None:
            linear.bias.uniform_(-float(bound), float(bound))


def _resolve_coord_mlp_model_type(cfg: dict[str, Any]) -> str:
    siren_cfg = dict(cfg.get("siren", {}))
    embedding_type = str(dict(cfg.get("embedding", {})).get("type", "")).strip().lower()
    if bool(siren_cfg.get("enabled", False)) or embedding_type == "none":
        return "coord_mlp_siren"
    return "coord_mlp_fourier"


def _normalize_coord_mlp_model_cfg(
    *,
    model_name: str | None,
    raw_cfg: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    cfg_local = dict(raw_cfg or {})
    requested = str(model_name).strip().lower() if model_name is not None else ""
    inferred = _resolve_coord_mlp_model_type(cfg_local)
    if requested in {"coord_mlp_fourier", "coord_mlp_siren"}:
        model_type = requested
    else:
        model_type = inferred
    if model_type == "coord_mlp_fourier" and bool(dict(cfg_local.get("siren", {})).get("enabled", False)):
        raise ValueError("train.coord_mlp_fourier.model_cfg.siren.enabled must be false")
    return _resolve_coord_mlp_config(cfg_local, model_type=model_type)


def _resolve_coord_mlp_config(cfg: dict[str, Any], *, model_type: str | None = None) -> tuple[str, dict[str, Any]]:
    cfg_local = dict(cfg or {})
    resolved_type = str(model_type).strip().lower() if model_type is not None else _resolve_coord_mlp_model_type(cfg_local)
    cfg_prefix = _cfg_prefix(resolved_type)

    raw_embedding = dict(cfg_local.get("embedding", {}))
    default_embedding_type = "none" if resolved_type == "coord_mlp_siren" else "fourier"
    raw_embedding_type = str(raw_embedding.get("type", default_embedding_type)).strip().lower() or default_embedding_type

    siren_cfg = dict(cfg_local.get("siren", {}))
    if resolved_type == "coord_mlp_siren":
        if raw_embedding_type != "none":
            raise ValueError(f"{cfg_prefix}.model_cfg.embedding.type must be none")
        if not bool(siren_cfg.get("enabled", False)):
            raise ValueError(f"{cfg_prefix}.model_cfg.siren.enabled must be true")
        if "decoder_activation" in cfg_local:
            raise ValueError(f"{cfg_prefix}.model_cfg.decoder_activation is not used for SIREN; remove it")
        fusion = str(siren_cfg.get("fusion", "split_add")).strip().lower()
        if fusion != "split_add":
            raise ValueError(f"{cfg_prefix}.model_cfg.siren.fusion must be split_add")
        w0_initial = _resolve_positive_float(
            siren_cfg.get("w0_initial"),
            default=10.0,
            cfg_key=f"{cfg_prefix}.model_cfg.siren.w0_initial",
        )
        w0_hidden = _resolve_positive_float(
            siren_cfg.get("w0_hidden"),
            default=1.0,
            cfg_key=f"{cfg_prefix}.model_cfg.siren.w0_hidden",
        )
        fusion_cfg_raw = dict(siren_cfg.get("fusion_cfg", {}))
        cond_gain_init = _resolve_positive_float(
            fusion_cfg_raw.get("cond_gain_init"),
            default=0.7,
            cfg_key=f"{cfg_prefix}.model_cfg.siren.fusion_cfg.cond_gain_init",
        )
        point_gain_init = _resolve_positive_float(
            fusion_cfg_raw.get("point_gain_init"),
            default=1.3,
            cfg_key=f"{cfg_prefix}.model_cfg.siren.fusion_cfg.point_gain_init",
        )
        branch_norm = _resolve_strict_bool(
            fusion_cfg_raw.get("branch_norm"),
            default=True,
            cfg_key=f"{cfg_prefix}.model_cfg.siren.fusion_cfg.branch_norm",
        )
        embedding_cfg = {"type": "none"}
        siren_cfg = {
            "enabled": True,
            "fusion": "split_add",
            "w0_initial": float(w0_initial),
            "w0_hidden": float(w0_hidden),
            "fusion_cfg": {
                "cond_gain_init": float(cond_gain_init),
                "point_gain_init": float(point_gain_init),
                "branch_norm": bool(branch_norm),
            },
        }
    else:
        if raw_embedding_type != "fourier":
            raise ValueError(f"{cfg_prefix}.model_cfg.embedding.type must be fourier")
        embedding_cfg = {
            "type": "fourier",
            "n_frequencies": int(raw_embedding.get("n_frequencies", 8)),
            "include_raw": bool(raw_embedding.get("include_raw", True)),
            "frequency_scale": float(raw_embedding.get("frequency_scale", 10.0)),
        }
        if embedding_cfg["n_frequencies"] <= 0:
            raise ValueError(f"{cfg_prefix}.model_cfg.embedding.n_frequencies must be > 0")
        if not np.isfinite(embedding_cfg["frequency_scale"]) or embedding_cfg["frequency_scale"] <= 0.0:
            raise ValueError(f"{cfg_prefix}.model_cfg.embedding.frequency_scale must be > 0")
        siren_cfg = {
            "enabled": False,
            "fusion": "split_add",
            "w0_initial": float(siren_cfg.get("w0_initial", 10.0)),
            "w0_hidden": float(siren_cfg.get("w0_hidden", 1.0)),
            "fusion_cfg": {
                "cond_gain_init": 0.7,
                "point_gain_init": 1.3,
                "branch_norm": True,
            },
        }

    out = {
        "cond_hidden": _resolve_hidden_list(
            cfg_local.get("cond_hidden"),
            default=[128, 128],
            cfg_key=f"{cfg_prefix}.model_cfg.cond_hidden",
        ),
        "latent_dim": int(cfg_local.get("latent_dim", 128)),
        "decoder_hidden": _resolve_hidden_list(
            cfg_local.get("decoder_hidden"),
            default=[256, 256, 256],
            cfg_key=f"{cfg_prefix}.model_cfg.decoder_hidden",
        ),
        "embedding": embedding_cfg,
        "siren": siren_cfg,
        "decoder_input_norm": {
            "enabled": bool(dict(cfg_local.get("decoder_input_norm", {})).get("enabled", True)),
        },
        "residual_head": {
            "enabled": bool(dict(cfg_local.get("residual_head", {})).get("enabled", True)),
            "init_scale": float(dict(cfg_local.get("residual_head", {})).get("init_scale", 0.0)),
        },
    }
    if resolved_type == "coord_mlp_fourier":
        out["decoder_activation"] = str(cfg_local.get("decoder_activation", "gelu")).strip().lower()
    if out["latent_dim"] <= 0:
        raise ValueError(f"{cfg_prefix}.model_cfg.latent_dim must be > 0")
    if not np.isfinite(float(out["residual_head"]["init_scale"])):
        raise ValueError(f"{cfg_prefix}.model_cfg.residual_head.init_scale must be finite")
    return resolved_type, out


def _standard_hidden_mlp(
    *,
    nn: Any,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation_name: str,
    cfg_prefix: str,
):
    layers: list[Any] = []
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    last_linear_idx = len(dims) - 2
    for linear_idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(int(fan_in), int(fan_out)))
        if linear_idx != last_linear_idx:
            layers.append(_activation_module(nn, activation_name, cfg_prefix=cfg_prefix))
    return nn.Sequential(*layers)


def _siren_hidden_mlp(
    *,
    torch: Any,
    nn: Any,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    w0_initial: float,
    w0_hidden: float,
):
    layers: list[Any] = []
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    last_linear_idx = len(dims) - 2
    for linear_idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        linear = nn.Linear(int(fan_in), int(fan_out))
        _siren_init_linear(
            torch=torch,
            linear=linear,
            w0=(w0_initial if linear_idx == 0 else w0_hidden),
            is_first=(linear_idx == 0),
        )
        layers.append(linear)
        if linear_idx != last_linear_idx:
            layers.append(
                _sine_activation_module(
                    nn=nn,
                    torch=torch,
                    w0=(w0_initial if linear_idx == 0 else w0_hidden),
                )
            )
    return nn.Sequential(*layers)


def _build_sequential_mlp(
    *,
    torch: Any,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation_name: str,
    siren_cfg: dict[str, Any],
    cfg_prefix: str,
):
    nn = torch.nn
    siren_enabled = bool(dict(siren_cfg).get("enabled", False))
    if siren_enabled:
        return _siren_hidden_mlp(
            torch=torch,
            nn=nn,
            input_dim=int(input_dim),
            hidden_dims=list(hidden_dims),
            output_dim=int(output_dim),
            w0_initial=float(dict(siren_cfg).get("w0_initial", 10.0)),
            w0_hidden=float(dict(siren_cfg).get("w0_hidden", 1.0)),
        )
    return _standard_hidden_mlp(
        nn=nn,
        input_dim=int(input_dim),
        hidden_dims=list(hidden_dims),
        output_dim=int(output_dim),
        activation_name=str(activation_name),
        cfg_prefix=cfg_prefix,
    )


def _sine_activation_module(*, nn: Any, torch: Any, w0: float):
    if _torch_nn is not None and nn is _torch_nn and _torch_mod is not None and torch is _torch_mod:
        return _SineActivation(w0=float(w0))

    class _FallbackSineActivation(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w0 = float(w0)

        def forward(self, x):
            return torch.sin(float(self.w0) * x)

    return _FallbackSineActivation()


if _torch_nn is not None and _torch_mod is not None:  # pragma: no branch - definition guard only.
    class _SineActivation(_torch_nn.Module):
        def __init__(self, *, w0: float) -> None:
            super().__init__()
            self.w0 = float(w0)

        def forward(self, x):
            return _torch_mod.sin(float(self.w0) * x)


    class _IdentityPointEncoder(_torch_nn.Module):
        def __init__(self, *, spatial_feature_dim: int) -> None:
            super().__init__()
            self.spatial_feature_dim = int(spatial_feature_dim)

        def output_dim(self) -> int:
            return int(self.spatial_feature_dim)

        def forward(self, spatial_t):
            return spatial_t


    class _FourierPointEncoder(_torch_nn.Module):
        def __init__(
            self,
            *,
            spatial_feature_dim: int,
            include_raw: bool,
            n_frequencies: int,
            frequency_scale: float,
        ) -> None:
            super().__init__()
            self.spatial_feature_dim = int(spatial_feature_dim)
            self.include_raw = bool(include_raw)
            self.n_frequencies = int(n_frequencies)
            self.frequency_scale = float(frequency_scale)
            scales = [
                (float(2**k) * float(np.pi)) / float(self.frequency_scale)
                for k in range(self.n_frequencies)
            ]
            self.register_buffer(
                "_fourier_scales",
                _torch_mod.as_tensor(np.asarray(scales, dtype=np.float32)),
                persistent=False,
            )

        def output_dim(self) -> int:
            raw = self.spatial_feature_dim if self.include_raw else 0
            return int(raw + (2 * self.n_frequencies * self.spatial_feature_dim))

        def forward(self, spatial_t):
            pieces = []
            if self.include_raw:
                pieces.append(spatial_t)
            scaled = spatial_t.unsqueeze(-2) * self._fourier_scales.view(1, 1, 1, -1, 1)
            sin_part = _torch_mod.sin(scaled).reshape(*spatial_t.shape[:3], -1)
            cos_part = _torch_mod.cos(scaled).reshape(*spatial_t.shape[:3], -1)
            pieces.extend([sin_part, cos_part])
            return _torch_mod.cat(pieces, dim=-1)


    class _CoordMLPNet(_torch_nn.Module):
        def __init__(
            self,
            *,
            model_type: str,
            out_channels: int,
            latent_dim: int,
            point_encoder: _torch_nn.Module,
            cond_encoder: _torch_nn.Module,
            point_decoder: _torch_nn.Module,
            decoder_input_norm: _torch_nn.Module | None,
            residual_head: _torch_nn.Module | None,
            residual_scale: Any | None,
            embedding_type: str,
            siren_enabled: bool,
        ) -> None:
            super().__init__()
            self.model_type = str(model_type)
            self.out_channels = int(out_channels)
            self.latent_dim = int(latent_dim)
            self.embedding_type = str(embedding_type)
            self.siren_enabled = bool(siren_enabled)
            self.point_encoder = point_encoder
            self.cond_encoder = cond_encoder
            self.point_decoder = point_decoder
            self.decoder_input_norm = decoder_input_norm
            self.residual_head = residual_head
            self.residual_scale = residual_scale

        def forward(self, cond_t, spatial_t):
            bsz, h, w, _ = spatial_t.shape
            z_cond = self.cond_encoder(cond_t)
            z_point = self.point_encoder(spatial_t)
            z_cond_expanded = z_cond[:, None, None, :].expand(bsz, h, w, self.latent_dim)
            dec_in = _torch_mod.cat([z_cond_expanded, z_point], dim=-1)
            flat = dec_in.reshape(bsz * h * w, dec_in.shape[-1])
            if self.decoder_input_norm is not None:
                flat = self.decoder_input_norm(flat)
            flat_out = self.point_decoder(flat)
            out = flat_out.reshape(bsz, h, w, self.out_channels)
            if self.residual_head is not None:
                res = self.residual_head(z_cond).reshape(bsz, 1, 1, self.out_channels).expand(bsz, h, w, self.out_channels)
                scale = self.residual_scale if self.residual_scale is not None else out.new_tensor(0.0)
                out = out + scale * res
            return out.permute(0, 3, 1, 2).contiguous()


    class _SirenSplitAddPointDecoder(_torch_nn.Module):
        def __init__(
            self,
            *,
            cond_dim: int,
            point_dim: int,
            hidden_dims: list[int],
            out_dim: int,
            w0_initial: float,
            w0_hidden: float,
            cond_gain_init: float,
            point_gain_init: float,
            branch_norm: bool,
        ) -> None:
            super().__init__()
            if len(hidden_dims) == 0:
                raise ValueError("train.coord_mlp_siren.model_cfg.decoder_hidden must be non-empty")
            self.cond_dim = int(cond_dim)
            self.point_dim = int(point_dim)
            self.w0_initial = float(w0_initial)
            self.w0_hidden = float(w0_hidden)
            self.branch_norm = bool(branch_norm)
            self.cond_norm = _torch_nn.LayerNorm(self.cond_dim) if self.branch_norm else _torch_nn.Identity()
            self.point_norm = _torch_nn.LayerNorm(self.point_dim) if self.branch_norm else _torch_nn.Identity()
            first_dim = int(hidden_dims[0])
            self.first_cond = _torch_nn.Linear(self.cond_dim, first_dim, bias=False)
            self.first_point = _torch_nn.Linear(self.point_dim, first_dim, bias=False)
            self.first_bias = _torch_nn.Parameter(_torch_mod.zeros((first_dim,), dtype=_torch_mod.float32))
            # Positive-constrained branch gains keep cond/point contribution balancing stable.
            self._cond_gain_raw = _torch_nn.Parameter(
                _torch_mod.tensor([_softplus_inverse(float(cond_gain_init))], dtype=_torch_mod.float32)
            )
            self._point_gain_raw = _torch_nn.Parameter(
                _torch_mod.tensor([_softplus_inverse(float(point_gain_init))], dtype=_torch_mod.float32)
            )
            _siren_init_linear(
                torch=_torch_mod,
                linear=self.first_cond,
                w0=self.w0_initial,
                is_first=True,
            )
            _siren_init_linear(
                torch=_torch_mod,
                linear=self.first_point,
                w0=self.w0_initial,
                is_first=True,
            )

            self.hidden_layers = _torch_nn.ModuleList()
            prev = first_dim
            for nxt in [int(v) for v in hidden_dims[1:]]:
                linear = _torch_nn.Linear(prev, nxt)
                _siren_init_linear(
                    torch=_torch_mod,
                    linear=linear,
                    w0=self.w0_hidden,
                    is_first=False,
                )
                self.hidden_layers.append(linear)
                prev = nxt
            self.out_layer = _torch_nn.Linear(prev, int(out_dim))
            _siren_init_linear(
                torch=_torch_mod,
                linear=self.out_layer,
                w0=self.w0_hidden,
                is_first=False,
            )

        def forward(self, flat):
            cond = flat[..., : self.cond_dim]
            point = flat[..., self.cond_dim : self.cond_dim + self.point_dim]
            cond = self.cond_norm(cond)
            point = self.point_norm(point)
            cond_gain = _torch_nn.functional.softplus(self._cond_gain_raw)
            point_gain = _torch_nn.functional.softplus(self._point_gain_raw)
            h = _torch_mod.sin(
                float(self.w0_initial)
                * (cond_gain * self.first_cond(cond) + point_gain * self.first_point(point) + self.first_bias)
            )
            for linear in self.hidden_layers:
                h = _torch_mod.sin(float(self.w0_hidden) * linear(h))
            return self.out_layer(h)


else:  # pragma: no cover - only used when torch is unavailable at import time.
    _SineActivation = None
    _IdentityPointEncoder = None
    _FourierPointEncoder = None
    _CoordMLPNet = None
    _SirenSplitAddPointDecoder = None


def _build_coord_mlp_net(
    *,
    torch: Any,
    model_type: str,
    input_dim: int,
    out_channels: int,
    spatial_feature_dim: int,
    cond_hidden: list[int],
    latent_dim: int,
    decoder_hidden: list[int],
    decoder_activation: str,
    embedding_cfg: dict[str, Any],
    siren_cfg: dict[str, Any],
    decoder_input_norm_cfg: dict[str, Any],
    residual_head_cfg: dict[str, Any],
):
    emb_type = str(embedding_cfg.get("type", "fourier")).strip().lower()
    embedding_include_raw = bool(embedding_cfg.get("include_raw", True))
    embedding_n_frequencies = int(embedding_cfg.get("n_frequencies", 8))
    embedding_frequency_scale = float(embedding_cfg.get("frequency_scale", 10.0))
    cfg_prefix = _cfg_prefix(model_type)
    nn = torch.nn
    siren_enabled = bool(dict(siren_cfg).get("enabled", False))
    cond_encoder = _build_sequential_mlp(
        torch=torch,
        input_dim=int(input_dim),
        hidden_dims=list(cond_hidden),
        output_dim=int(latent_dim),
        activation_name="gelu",
        siren_cfg={"enabled": False},
        cfg_prefix=cfg_prefix,
    )
    if emb_type == "none":
        if _IdentityPointEncoder is None:
            raise RuntimeError("CoordMLPTorch requires torch point encoders to be available")
        point_encoder = _IdentityPointEncoder(spatial_feature_dim=int(spatial_feature_dim))
    else:
        if _FourierPointEncoder is None:
            raise RuntimeError("CoordMLPTorch requires torch point encoders to be available")
        point_encoder = _FourierPointEncoder(
            spatial_feature_dim=int(spatial_feature_dim),
            include_raw=bool(embedding_include_raw),
            n_frequencies=int(embedding_n_frequencies),
            frequency_scale=float(embedding_frequency_scale),
        )
    if siren_enabled:
        fusion = str(dict(siren_cfg).get("fusion", "split_add")).strip().lower()
        if fusion != "split_add":
            raise ValueError(f"{cfg_prefix}.model_cfg.siren.fusion must be split_add")
        if _SirenSplitAddPointDecoder is None:
            raise RuntimeError("CoordMLPTorch requires torch SIREN decoder modules to be available")
        point_decoder = _SirenSplitAddPointDecoder(
            cond_dim=int(latent_dim),
            point_dim=int(point_encoder.output_dim()),
            hidden_dims=list(decoder_hidden),
            out_dim=int(out_channels),
            w0_initial=float(dict(siren_cfg).get("w0_initial", 10.0)),
            w0_hidden=float(dict(siren_cfg).get("w0_hidden", 1.0)),
            cond_gain_init=float(dict(dict(siren_cfg).get("fusion_cfg", {})).get("cond_gain_init", 0.7)),
            point_gain_init=float(dict(dict(siren_cfg).get("fusion_cfg", {})).get("point_gain_init", 1.3)),
            branch_norm=bool(dict(dict(siren_cfg).get("fusion_cfg", {})).get("branch_norm", True)),
        )
    else:
        point_decoder = _build_sequential_mlp(
            torch=torch,
            input_dim=int(latent_dim) + int(point_encoder.output_dim()),
            hidden_dims=list(decoder_hidden),
            output_dim=int(out_channels),
            activation_name=str(decoder_activation),
            siren_cfg=dict(siren_cfg),
            cfg_prefix=cfg_prefix,
        )
    decoder_input_norm = nn.LayerNorm(int(latent_dim) + int(point_encoder.output_dim())) if bool(
        dict(decoder_input_norm_cfg).get("enabled", True)
    ) else None
    residual_head = None
    residual_scale = None
    if bool(dict(residual_head_cfg).get("enabled", True)):
        residual_head = nn.Linear(int(latent_dim), int(out_channels))
        residual_scale = nn.Parameter(
            torch.full((1,), float(dict(residual_head_cfg).get("init_scale", 0.0)), dtype=torch.float32)
        )
    if _CoordMLPNet is None:
        raise RuntimeError("CoordMLPTorch requires torch modules to be available")
    return _CoordMLPNet(
        model_type=str(model_type),
        out_channels=int(out_channels),
        latent_dim=int(latent_dim),
        point_encoder=point_encoder,
        cond_encoder=cond_encoder,
        point_decoder=point_decoder,
        decoder_input_norm=decoder_input_norm,
        residual_head=residual_head,
        residual_scale=residual_scale,
        embedding_type=str(emb_type),
        siren_enabled=bool(siren_enabled),
    )


class CoordMLPTorch:
    """Geometry-aware full-field coordinate decoder family."""

    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        input_feature_channels: list[str] | None = None,
        model_cfg: dict[str, Any] | None = None,
        seed: int = 0,
        backend: str = "torch",
    ) -> None:
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            base = ["ne", "Te", "phi"]
            extra = [f"out_{i}" for i in range(max(0, self.out_channels - len(base)))]
            self.output_keys = (base + extra)[: self.out_channels]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        cfg = dict(model_cfg or {})
        self.model_type, self.model_cfg = _normalize_coord_mlp_model_cfg(
            model_name=None,
            raw_cfg=cfg,
        )
        cfg_prefix = _cfg_prefix(self.model_type)
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError(f"{cfg_prefix}.model_cfg.backend must be torch")
        channels = list(input_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
        if len(channels) == 0:
            raise ValueError(f"{cfg_prefix}.input_features.features must be a non-empty list")
        if len(set(channels)) != len(channels):
            raise ValueError(f"{cfg_prefix}.input_features.features must not contain duplicates")
        self.input_feature_channels = [str(v) for v in channels]
        self.spatial_feature_dim = int(len(self.input_feature_channels))
        self.raw_out_channels = int(self.out_channels)
        self.coord_mlp_impl_version = "v4_siren_branch_balanced"
        self.coord = _build_unit_coord_grid(self.grid_shape)
        self._static_spatial_features: np.ndarray | None = None
        self._torch_last_out = None
        self._torch_last_cond = None
        self._torch_last_spatial = None
        from plasma_surrogate.core.torch_backend import require_torch

        torch = require_torch()
        torch.manual_seed(int(seed))
        self.torch = torch
        self.net = _build_coord_mlp_net(
            torch=torch,
            model_type=str(self.model_type),
            input_dim=self.input_dim,
            out_channels=self.out_channels,
            spatial_feature_dim=self.spatial_feature_dim,
            cond_hidden=list(self.model_cfg["cond_hidden"]),
            latent_dim=int(self.model_cfg["latent_dim"]),
            decoder_hidden=list(self.model_cfg["decoder_hidden"]),
            decoder_activation=str(self.model_cfg.get("decoder_activation", "gelu")),
            embedding_cfg=dict(self.model_cfg["embedding"]),
            siren_cfg=dict(self.model_cfg["siren"]),
            decoder_input_norm_cfg=dict(self.model_cfg.get("decoder_input_norm", {})),
            residual_head_cfg=dict(self.model_cfg.get("residual_head", {})),
        )
        self.device = torch.device("cuda" if bool(torch.cuda.is_available()) else "cpu")
        self.net.to(self.device)

    def set_static_spatial_features(self, spatial_features: np.ndarray | None) -> None:
        if spatial_features is None:
            self._static_spatial_features = None
            return
        self._static_spatial_features = _validate_static_spatial_features(
            spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=str(self.model_type),
        )

    def _resolve_spatial_features(self, cond: np.ndarray, spatial_features: np.ndarray | None) -> np.ndarray:
        return _resolve_batched_spatial_features(
            cond,
            spatial_features,
            static_spatial_features=self._static_spatial_features,
            grid_shape=self.grid_shape,
            spatial_feature_dim=self.spatial_feature_dim,
            label=str(self.model_type),
            explicit_requirement_message=(
                f"{self.model_type} requires explicit geom_feature_pack spatial features; "
                "set_static_spatial_features(...) or pass spatial_features=..."
            ),
        )

    def _build_decoder_input(
        self,
        cond: np.ndarray,
        spatial_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        spatial_arr = self._resolve_spatial_features(cond_arr, spatial_features)
        return cond_arr.astype(np.float32), spatial_arr.astype(np.float32)

    def _forward_raw_torch(
        self,
        cond: np.ndarray,
        *,
        training: bool,
        spatial_features: np.ndarray | None,
    ) -> np.ndarray:
        torch = self.torch
        cond_arr, spatial_arr = self._build_decoder_input(cond, spatial_features=spatial_features)
        cond_t = torch.as_tensor(cond_arr, dtype=torch.float32, device=self.device)
        spatial_t = torch.as_tensor(spatial_arr, dtype=torch.float32, device=self.device)
        if bool(training):
            self.net.train()
            yt = self.net(cond_t, spatial_t)
            self._torch_last_cond = cond_t
            self._torch_last_spatial = spatial_t
            self._torch_last_out = yt
        else:
            self.net.eval()
            with torch.no_grad():
                yt = self.net(cond_t, spatial_t)
            self._torch_last_cond = None
            self._torch_last_spatial = None
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
        return {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}

    def forward(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)

    def predict_fields(self, cond: np.ndarray, spatial_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return self.forward_features(cond, spatial_features=spatial_features)

    def _torch_step_reference(self):
        cond_first = None
        decoder_last = None
        for module in self.net.cond_encoder:
            if hasattr(module, "weight"):
                cond_first = module.weight
                break
        point_decoder = self.net.point_decoder
        if hasattr(point_decoder, "out_layer") and hasattr(point_decoder.out_layer, "weight"):
            decoder_last = point_decoder.out_layer.weight
        elif hasattr(point_decoder, "__iter__"):
            for module in reversed(list(point_decoder)):
                if hasattr(module, "weight"):
                    decoder_last = module.weight
                    break
        elif hasattr(point_decoder, "weight"):
            decoder_last = point_decoder.weight
        return cond_first, decoder_last

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
            raise RuntimeError("CoordMLPTorch.backward_raw called without torch forward cache")
        out = _backward_raw_torch_step(
            torch=self.torch,
            net=self.net,
            grad_raw=grad_raw,
            last_out=self._torch_last_out,
            lr=float(lr),
            step_reference=self._torch_step_reference,
            apply_step=bool(apply_step),
        )
        self._torch_last_cond = None
        self._torch_last_spatial = None
        self._torch_last_out = None
        return out

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return _state_dict_numpy_torch(self.net)

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message="CoordMLPTorch state dict does not contain expected torch::* weights",
        )


__all__ = ["CoordMLPTorch", "_normalize_coord_mlp_model_cfg"]
