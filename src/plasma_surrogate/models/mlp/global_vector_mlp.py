"""Shared vector-to-field DNN for residual and dense-connectivity baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import (
    _load_state_dict_numpy_torch,
    _resolve_torch_device,
    _state_dict_numpy_torch,
)


GLOBAL_VECTOR_MLP_MODEL_TYPES = ("global_resmlp", "global_densemlp")
GLOBAL_VECTOR_MLP_IMPL_VERSION = "global_vector_mlp_v1"


def normalize_global_vector_mlp_cfg(
    model_cfg: dict[str, Any] | None,
    *,
    model_type: str,
) -> dict[str, Any]:
    """Return the small, shared configuration contract for both backbones."""

    name = str(model_type).strip().lower()
    if name not in GLOBAL_VECTOR_MLP_MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {list(GLOBAL_VECTOR_MLP_MODEL_TYPES)}; got={model_type!r}"
        )
    cfg = dict(model_cfg or {})
    shared_keys = {
        "width",
        "blocks",
        "dropout",
        "weight_decay",
        "activation",
        "backend",
    }
    variant_keys = {"residual_scale"} if name == "global_resmlp" else {"growth_rate"}
    unknown_keys = sorted(set(cfg) - shared_keys - variant_keys)
    if unknown_keys:
        raise ValueError(
            f"train.{name}.model_cfg contains unsupported keys: {unknown_keys}"
        )
    width = int(cfg.get("width", 64))
    blocks = int(cfg.get("blocks", 2))
    dropout = float(cfg.get("dropout", 0.0))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    activation = str(cfg.get("activation", "gelu")).strip().lower()
    backend = str(cfg.get("backend", "torch")).strip().lower()

    if width < 1:
        raise ValueError(f"train.{name}.model_cfg.width must be >= 1")
    if blocks < 1:
        raise ValueError(f"train.{name}.model_cfg.blocks must be >= 1")
    if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"train.{name}.model_cfg.dropout must be finite and in [0, 1)")
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(f"train.{name}.model_cfg.weight_decay must be finite and >= 0")
    if activation not in {"gelu", "silu"}:
        raise ValueError(f"train.{name}.model_cfg.activation must be one of: gelu, silu")
    if backend != "torch":
        raise ValueError(f"train.{name}.model_cfg.backend must be torch")

    normalized = {
        "width": width,
        "blocks": blocks,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "activation": activation,
        "backend": backend,
    }
    if name == "global_resmlp":
        residual_scale = float(cfg.get("residual_scale", 1.0))
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError(
                "train.global_resmlp.model_cfg.residual_scale must be finite and > 0"
            )
        normalized["residual_scale"] = residual_scale
    else:
        growth_rate = int(cfg.get("growth_rate", 16))
        if growth_rate < 1:
            raise ValueError("train.global_densemlp.model_cfg.growth_rate must be >= 1")
        normalized["growth_rate"] = growth_rate
    return normalized


def _build_network(
    *,
    torch: Any,
    model_type: str,
    input_dim: int,
    output_dim: int,
    cfg: dict[str, Any],
) -> Any:
    nn = torch.nn
    width = int(cfg["width"])
    blocks = int(cfg["blocks"])
    growth_rate = int(cfg.get("growth_rate", 16))
    dropout = float(cfg["dropout"])
    residual_scale = float(cfg.get("residual_scale", 1.0))
    activation_name = str(cfg["activation"])

    def activation() -> Any:
        return nn.GELU() if activation_name == "gelu" else nn.SiLU()

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.linear1 = nn.Linear(width, width)
            self.linear2 = nn.Linear(width, width)
            self.activation = activation()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            branch = self.linear1(self.norm(x))
            branch = self.activation(branch)
            branch = self.dropout(branch)
            branch = self.linear2(branch)
            return x + residual_scale * branch

    class DenseBlock(nn.Module):
        def __init__(self, in_features: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(in_features)
            self.linear = nn.Linear(in_features, growth_rate)
            self.activation = activation()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            new_features = self.linear(self.norm(x))
            new_features = self.activation(new_features)
            new_features = self.dropout(new_features)
            return torch.cat((x, new_features), dim=-1)

    class VectorFieldNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(input_dim, width), activation())
            if model_type == "global_resmlp":
                self.blocks = nn.ModuleList(ResidualBlock() for _ in range(blocks))
                self.compression = nn.Identity()
            else:
                dense_blocks = []
                in_features = width
                for _ in range(blocks):
                    dense_blocks.append(DenseBlock(in_features))
                    in_features += growth_rate
                self.blocks = nn.ModuleList(dense_blocks)
                # Keep the field head width independent of DenseNet growth.
                self.compression = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, width),
                    activation(),
                )
            self.output = nn.Linear(width, output_dim)

        def forward_features(self, x: Any) -> Any:
            features = self.stem(x)
            for block in self.blocks:
                features = block(features)
            return self.compression(features)

        def forward(self, x: Any) -> Any:
            return self.output(self.forward_features(x))

    return VectorFieldNet()


class GlobalVectorMLP:
    """One train/checkpoint contract with residual or dense connectivity."""

    requires_spatial_features = False
    requires_scaled_spatial_features = False
    impl_version = GLOBAL_VECTOR_MLP_IMPL_VERSION

    def __init__(
        self,
        *,
        model_type: str,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        model_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.model_type = str(model_type).strip().lower()
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.out_channels = int(out_channels)
        self.output_keys = list(
            output_keys or [f"target_{idx}" for idx in range(self.out_channels)]
        )
        if self.input_dim < 1:
            raise ValueError("GlobalVectorMLP input_dim must be >= 1")
        if len(self.grid_shape) != 2 or min(self.grid_shape) < 1:
            raise ValueError("GlobalVectorMLP grid_shape must contain two positive dimensions")
        if self.out_channels < 1:
            raise ValueError("GlobalVectorMLP out_channels must be >= 1")
        if len(self.output_keys) != self.out_channels:
            raise ValueError(
                "GlobalVectorMLP output_keys length must match out_channels: "
                f"keys={len(self.output_keys)}, channels={self.out_channels}"
            )

        self.model_cfg = normalize_global_vector_mlp_cfg(
            model_cfg,
            model_type=self.model_type,
        )
        self.backend = "torch"
        self.width = int(self.model_cfg["width"])
        self.blocks = int(self.model_cfg["blocks"])
        self.growth_rate = int(self.model_cfg.get("growth_rate", 0))
        self.dropout = float(self.model_cfg["dropout"])
        self.weight_decay = float(self.model_cfg["weight_decay"])
        self.residual_scale = float(self.model_cfg.get("residual_scale", 0.0))
        self.output_dim = int(self.out_channels * self.grid_shape[0] * self.grid_shape[1])

        from plasma_surrogate.core.torch_backend import require_torch

        self.torch = require_torch()
        self.device = _resolve_torch_device(self.torch)
        self.torch.manual_seed(int(seed))
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.manual_seed_all(int(seed))
        self.net = _build_network(
            torch=self.torch,
            model_type=self.model_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            cfg=self.model_cfg,
        ).to(self.device)
        self._last_out = None

    def _validate_cond(self, cond: np.ndarray) -> np.ndarray:
        arr = np.asarray(cond, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or int(arr.shape[1]) != self.input_dim:
            raise ValueError(
                f"{self.model_type} cond must be [B,{self.input_dim}], got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{self.model_type} cond must contain only finite values")
        return arr

    def forward(self, cond: np.ndarray, training: bool = False) -> np.ndarray:
        arr = self._validate_cond(cond)
        cond_t = self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device)
        if bool(training):
            self.net.train()
            self._last_out = self.net(cond_t)
            return self._last_out.detach().cpu().numpy().astype(np.float32)
        self.net.eval()
        self._last_out = None
        with self.torch.no_grad():
            return self.net(cond_t).detach().cpu().numpy().astype(np.float32)

    def backward(
        self,
        grad_output: np.ndarray,
        *,
        lr: float,
        weight_decay: float | None = None,
        layer_lr_multipliers: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if self._last_out is None:
            raise RuntimeError(f"{self.model_type}.backward called without a training forward")
        grad = np.asarray(grad_output, dtype=np.float32)
        if tuple(grad.shape) != tuple(self._last_out.shape):
            raise ValueError(
                f"{self.model_type} grad shape mismatch: expected={tuple(self._last_out.shape)}, "
                f"got={grad.shape}"
            )
        grad_t = self.torch.as_tensor(
            grad,
            dtype=self._last_out.dtype,
            device=self._last_out.device,
        )
        self.net.zero_grad(set_to_none=True)
        self._last_out.backward(grad_t)

        wd = self.weight_decay if weight_decay is None else float(max(weight_decay, 0.0))
        multipliers = dict(layer_lr_multipliers or {})
        hidden_mult = float(multipliers.get("hidden", 1.0))
        output_mult = float(multipliers.get("output", 1.0))
        output_ids = {id(parameter) for parameter in self.net.output.parameters()}
        totals = {
            "hidden_update_sq": 0.0,
            "hidden_param_sq": 0.0,
            "output_update_sq": 0.0,
            "output_param_sq": 0.0,
        }
        with self.torch.no_grad():
            for parameter in self.net.parameters():
                if parameter.grad is None:
                    continue
                is_output = id(parameter) in output_ids
                grad_eff = parameter.grad
                if wd > 0.0 and int(parameter.ndim) > 1:
                    grad_eff = grad_eff + wd * parameter
                lr_eff = float(lr) * (output_mult if is_output else hidden_mult)
                update = lr_eff * grad_eff
                prefix = "output" if is_output else "hidden"
                totals[f"{prefix}_update_sq"] += float(self.torch.sum(update * update).item())
                totals[f"{prefix}_param_sq"] += float(
                    self.torch.sum(parameter * parameter).item()
                )
                parameter.sub_(update)
        self._last_out = None
        return {
            "step_rel_hidden_mean": float(
                np.sqrt(totals["hidden_update_sq"])
                / max(np.sqrt(totals["hidden_param_sq"]), 1.0e-12)
            ),
            "step_rel_output": float(
                np.sqrt(totals["output_update_sq"])
                / max(np.sqrt(totals["output_param_sq"]), 1.0e-12)
            ),
        }

    def extract_hidden(self, cond: np.ndarray) -> np.ndarray:
        arr = self._validate_cond(cond)
        self.net.eval()
        with self.torch.no_grad():
            cond_t = self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device)
            hidden = self.net.forward_features(cond_t)
            return hidden.detach().cpu().numpy().astype(np.float32)

    def fit_output_layer_ridge(
        self,
        hidden: np.ndarray,
        target_flat: np.ndarray,
        *,
        ridge: float = 1.0e-4,
    ) -> float:
        hmat = np.asarray(hidden, dtype=np.float32)
        ymat = np.asarray(target_flat, dtype=np.float32)
        if hmat.ndim != 2 or int(hmat.shape[1]) != self.width:
            raise ValueError(f"hidden must be [N,{self.width}], got {hmat.shape}")
        if ymat.ndim != 2 or int(ymat.shape[1]) != self.output_dim:
            raise ValueError(f"target_flat must be [N,{self.output_dim}], got {ymat.shape}")
        if int(hmat.shape[0]) != int(ymat.shape[0]):
            raise ValueError("hidden and target_flat batch sizes must match")

        design = np.concatenate(
            [hmat, np.ones((hmat.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        gram = design.T @ design
        ridge_value = float(max(ridge, 0.0))
        if ridge_value > 0.0:
            regularizer = np.eye(gram.shape[0], dtype=np.float32)
            regularizer[-1, -1] = 0.0
            gram = gram + ridge_value * regularizer
        beta = np.linalg.solve(gram, design.T @ ymat).astype(np.float32)
        with self.torch.no_grad():
            self.net.output.weight.copy_(
                self.torch.as_tensor(beta[:-1].T, dtype=self.torch.float32, device=self.device)
            )
            self.net.output.bias.copy_(
                self.torch.as_tensor(beta[-1], dtype=self.torch.float32, device=self.device)
            )
        singular_values = np.linalg.svd(design, compute_uv=False)
        return float(
            singular_values[0] / max(float(singular_values[-1]), 1.0e-12)
        )

    def predict_fields(self, cond: np.ndarray) -> dict[str, np.ndarray]:
        flat = self.forward(cond, training=False)
        fields = flat.reshape(flat.shape[0], self.out_channels, *self.grid_shape)
        return {
            name: fields[:, idx : idx + 1]
            for idx, name in enumerate(self.output_keys)
        }

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return _state_dict_numpy_torch(self.net)

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message=(
                f"{self.model_type} state dict does not contain expected torch::* weights"
            ),
            device=self.device,
        )
        self.net.to(self.device)
        self._last_out = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "impl_version": self.impl_version,
            "input_dim": self.input_dim,
            "grid_shape": list(self.grid_shape),
            "out_channels": self.out_channels,
            "output_keys": list(self.output_keys),
            "model_cfg": dict(self.model_cfg),
        }


__all__ = [
    "GLOBAL_VECTOR_MLP_IMPL_VERSION",
    "GLOBAL_VECTOR_MLP_MODEL_TYPES",
    "GlobalVectorMLP",
    "normalize_global_vector_mlp_cfg",
]
