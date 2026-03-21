"""Experimental POD-based DeepONet full-field torch wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.models._torch_spatial_common import (
    _backward_raw_torch_step,
    _load_state_dict_numpy_torch,
    _state_dict_numpy_torch,
)


def _cfg_prefix(model_type: str) -> str:
    return f"train.{str(model_type).strip().lower()}"


def _resolve_basis_cfg(model_cfg: dict[str, Any], *, cfg_prefix: str) -> dict[str, Any]:
    basis_cfg = dict(model_cfg.get("basis", {}))
    rank = int(basis_cfg.get("rank", 32))
    if rank < 1:
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.rank must be >= 1")
    fit_scope = str(basis_cfg.get("fit_scope", "train_only")).strip().lower()
    if fit_scope != "train_only":
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.fit_scope must be train_only")
    per_var = bool(basis_cfg.get("per_var", True))
    if not per_var:
        raise ValueError(f"{cfg_prefix}.model_cfg.basis.per_var must be true")
    center = bool(basis_cfg.get("center", True))
    return {
        "rank": int(rank),
        "fit_scope": fit_scope,
        "per_var": per_var,
        "center": center,
    }


def _resolve_hidden_dims(cfg: dict[str, Any], *, cfg_prefix: str) -> list[int]:
    hidden_raw = cfg.get("hidden")
    if hidden_raw is None:
        hidden_dim = int(cfg.get("hidden_dim", 128))
        latent_dim = int(cfg.get("latent_dim", 128))
        hidden = [hidden_dim, latent_dim]
    else:
        hidden = [int(v) for v in list(hidden_raw)]
        if len(hidden) == 1:
            hidden.append(int(cfg.get("latent_dim", hidden[0])))
    if len(hidden) == 0:
        raise ValueError(f"{cfg_prefix}.model_cfg.hidden must be a non-empty list")
    if any(v < 1 for v in hidden):
        raise ValueError(f"{cfg_prefix}.model_cfg.hidden must contain only positive integers")
    return hidden


def normalize_pod_deeponet_model_cfg(
    raw_cfg: dict[str, Any] | None,
    *,
    model_type: str = "deeponet_pod",
) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    cfg_prefix = _cfg_prefix(model_type)
    hidden = _resolve_hidden_dims(cfg, cfg_prefix=cfg_prefix)
    basis_cfg = _resolve_basis_cfg(cfg, cfg_prefix=cfg_prefix)
    return {
        "hidden": list(hidden),
        "latent_dim": int(hidden[-1]),
        "basis": basis_cfg,
    }


@dataclass(frozen=True)
class PODBasisBundle:
    basis_by_var: dict[str, np.ndarray]
    mean_by_var: dict[str, np.ndarray]
    rank_by_var: dict[str, int]

    @classmethod
    def from_dicts(
        cls,
        *,
        basis_by_var: dict[str, np.ndarray],
        mean_by_var: dict[str, np.ndarray],
        rank_by_var: dict[str, int] | None = None,
    ) -> "PODBasisBundle":
        basis_out: dict[str, np.ndarray] = {}
        mean_out: dict[str, np.ndarray] = {}
        rank_out: dict[str, int] = {}
        rank_meta = dict(rank_by_var or {})
        for name, arr in dict(basis_by_var or {}).items():
            key = str(name)
            basis_arr = np.asarray(arr, dtype=np.float32)
            if basis_arr.ndim != 3:
                raise ValueError(f"deeponet_pod basis[{key}] must be [rank,H,W], got {basis_arr.shape}")
            rank_eff = int(rank_meta.get(key, basis_arr.shape[0]))
            if rank_eff != int(basis_arr.shape[0]):
                raise ValueError(
                    f"deeponet_pod basis rank mismatch for {key}: rank_meta={rank_eff}, basis={basis_arr.shape[0]}"
                )
            if rank_eff < 1:
                raise ValueError(f"deeponet_pod basis rank for {key} must be >= 1")
            basis_out[key] = basis_arr.astype(np.float32, copy=True)
            rank_out[key] = int(rank_eff)
        for name, arr in dict(mean_by_var or {}).items():
            mean_out[str(name)] = np.asarray(arr, dtype=np.float32).copy()
        missing_mean = sorted(set(basis_out.keys()) - set(mean_out.keys()))
        if missing_mean:
            raise ValueError(f"deeponet_pod missing mean for basis vars: {missing_mean}")
        return cls(basis_by_var=basis_out, mean_by_var=mean_out, rank_by_var=rank_out)


def fit_pod_basis_from_targets(
    y_train_scaled: np.ndarray,
    *,
    output_keys: list[str],
    requested_rank: int,
    center: bool,
    per_var: bool,
) -> PODBasisBundle:
    if not bool(per_var):
        raise ValueError("deeponet_pod v1 requires per_var=true")
    arr = np.asarray(y_train_scaled, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"y_train_scaled must be [N,C,H,W], got {arr.shape}")
    n_samples, n_targets, h, w = arr.shape
    if n_samples < 1:
        raise ValueError("deeponet_pod basis fitting requires at least one train sample")
    if int(n_targets) != len(list(output_keys)):
        raise ValueError(
            f"deeponet_pod basis fitting target mismatch: got C={n_targets}, output_keys={list(output_keys)}"
        )
    basis_by_var: dict[str, np.ndarray] = {}
    mean_by_var: dict[str, np.ndarray] = {}
    rank_by_var: dict[str, int] = {}
    max_rank = max(int(requested_rank), 1)
    flat_dim = int(h * w)
    for idx, var_name in enumerate(output_keys):
        snapshots = arr[:, idx].reshape(n_samples, flat_dim).astype(np.float32)
        mean_flat = (
            np.mean(snapshots, axis=0, dtype=np.float32) if bool(center) else np.zeros((flat_dim,), dtype=np.float32)
        )
        centered = snapshots - mean_flat[None, :]
        rank_eff = int(min(max_rank, n_samples, flat_dim))
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis_flat = np.asarray(vh[:rank_eff], dtype=np.float32)
        basis_by_var[str(var_name)] = basis_flat.reshape(rank_eff, h, w).astype(np.float32)
        mean_by_var[str(var_name)] = mean_flat.reshape(h, w).astype(np.float32)
        rank_by_var[str(var_name)] = int(rank_eff)
    return PODBasisBundle.from_dicts(
        basis_by_var=basis_by_var,
        mean_by_var=mean_by_var,
        rank_by_var=rank_by_var,
    )


class PODDeepONetTorch:
    """Reduced-order torch wrapper that predicts POD coefficients from cond."""

    model_type = "deeponet_pod"

    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int,
        output_keys: list[str] | None,
        pod_basis_bundle: PODBasisBundle | None = None,
        pod_basis: dict[str, np.ndarray] | None = None,
        pod_mean: dict[str, np.ndarray] | None = None,
        model_cfg: dict[str, Any] | None = None,
        basis_rank_by_var: dict[str, int] | None = None,
        seed: int = 0,
        backend: str = "torch",
    ) -> None:
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.out_channels = int(out_channels)
        self.output_keys = list(output_keys or [f"out_{i}" for i in range(self.out_channels)])
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError("train.deeponet_pod.model_cfg.backend must be torch")
        self.model_cfg = normalize_pod_deeponet_model_cfg(model_cfg, model_type=self.model_type)
        if list(self.output_keys) != list(self.output_keys[: self.out_channels]):
            self.output_keys = list(self.output_keys[: self.out_channels])
        basis_bundle = pod_basis_bundle or PODBasisBundle.from_dicts(
            basis_by_var=dict(pod_basis or {}),
            mean_by_var=dict(pod_mean or {}),
            rank_by_var=dict(basis_rank_by_var or {}),
        )
        self.basis_keys = [str(v) for v in self.output_keys]
        missing_basis = [name for name in self.basis_keys if name not in basis_bundle.basis_by_var]
        missing_mean = [name for name in self.basis_keys if name not in basis_bundle.mean_by_var]
        if missing_basis:
            raise ValueError(f"deeponet_pod missing basis for output keys: {missing_basis}")
        if missing_mean:
            raise ValueError(f"deeponet_pod missing mean for output keys: {missing_mean}")

        h, w = self.grid_shape
        self.basis_rank_by_var: dict[str, int] = {}
        total_rank = 0
        for name in self.basis_keys:
            basis_arr = np.asarray(basis_bundle.basis_by_var[name], dtype=np.float32)
            mean_arr = np.asarray(basis_bundle.mean_by_var[name], dtype=np.float32)
            if basis_arr.ndim != 3 or tuple(basis_arr.shape[1:]) != (h, w):
                raise ValueError(f"deeponet_pod basis[{name}] must be [rank,H,W], got {basis_arr.shape}")
            if mean_arr.shape != (h, w):
                raise ValueError(f"deeponet_pod mean[{name}] must be [H,W], got {mean_arr.shape}")
            rank_eff = int(basis_bundle.rank_by_var[name])
            self.basis_rank_by_var[name] = int(rank_eff)
            total_rank += int(rank_eff)
        self.coeff_dim = int(total_rank)
        self._coeff_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in self.basis_keys:
            stop = offset + int(self.basis_rank_by_var[name])
            self._coeff_slices[name] = (offset, stop)
            offset = stop
        if offset != self.coeff_dim:
            raise RuntimeError("deeponet_pod coefficient slice construction failed")

        from plasma_surrogate.core.torch_backend import require_torch

        self.torch = require_torch()
        self.torch.manual_seed(int(seed))
        nn = self.torch.nn
        dims = [self.input_dim, *list(self.model_cfg["hidden"]), self.coeff_dim]
        layers: list[Any] = []
        for idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(int(fan_in), int(fan_out)))
            if idx != len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        for name in self.basis_keys:
            self.net.register_buffer(
                f"_pod_basis__{name}",
                self.torch.as_tensor(basis_bundle.basis_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
            self.net.register_buffer(
                f"_pod_mean__{name}",
                self.torch.as_tensor(basis_bundle.mean_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
        self._torch_last_out = None

    def _basis_tensor(self, name: str):
        return getattr(self.net, f"_pod_basis__{name}")

    def _mean_tensor(self, name: str):
        return getattr(self.net, f"_pod_mean__{name}")

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
        )

    def _reconstruct_from_coeff(self, coeff_t):
        bsz = int(coeff_t.shape[0])
        fields = []
        for name in self.basis_keys:
            start, stop = self._coeff_slices[name]
            coeff_var = coeff_t[:, start:stop]
            basis_t = self._basis_tensor(name).reshape(int(stop - start), -1)
            mean_t = self._mean_tensor(name).reshape(1, -1)
            field_flat = self.torch.matmul(coeff_var, basis_t) + mean_t
            fields.append(field_flat.reshape(bsz, *self.grid_shape))
        return self.torch.stack(fields, dim=1)

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool) -> np.ndarray:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        cond_t = self.torch.from_numpy(cond_arr.astype(np.float32))
        if bool(training):
            self.net.train()
            coeff_t = self.net(cond_t)
            self._torch_last_out = self._reconstruct_from_coeff(coeff_t)
        else:
            self.net.eval()
            with self.torch.no_grad():
                coeff_t = self.net(cond_t)
                self._torch_last_out = None
                return self._reconstruct_from_coeff(coeff_t).detach().cpu().numpy().astype(np.float32)
        return self._torch_last_out.detach().cpu().numpy().astype(np.float32)

    def forward_raw(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del spatial_features
        return self._forward_raw_torch(cond, training=bool(training))

    def forward(
        self,
        cond: np.ndarray,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward_raw(cond, training=bool(training), spatial_features=spatial_features)

    def forward_features(
        self,
        cond: np.ndarray,
        spatial_features: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        pred = self.forward(cond, training=False, spatial_features=spatial_features)
        return {name: pred[:, i : i + 1] for i, name in enumerate(self.output_keys)}

    def predict_fields(
        self,
        cond: np.ndarray,
        spatial_features: np.ndarray | None = None,
        geom_ctx: Any | None = None,
    ) -> dict[str, np.ndarray]:
        del geom_ctx
        return self.forward_features(cond, spatial_features=spatial_features)

    def _torch_step_reference(self):
        first = None
        last = None
        for module in self.net:
            if hasattr(module, "weight"):
                first = module.weight
                break
        for module in reversed(list(self.net)):
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
    ) -> dict[str, float]:
        del weight_decay
        if self._torch_last_out is None:
            raise RuntimeError("PODDeepONetTorch.backward_raw called without torch forward cache")
        out = _backward_raw_torch_step(
            torch=self.torch,
            net=self.net,
            grad_raw=grad_raw,
            last_out=self._torch_last_out,
            lr=float(lr),
            step_reference=self._torch_step_reference,
            apply_step=bool(apply_step),
        )
        self._torch_last_out = None
        return out

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        state = _state_dict_numpy_torch(self.net)
        basis_bundle = self.basis_bundle_numpy()
        for name in self.basis_keys:
            state[f"basis::{name}"] = np.asarray(basis_bundle.basis_by_var[name], dtype=np.float32).copy()
            state[f"mean::{name}"] = np.asarray(basis_bundle.mean_by_var[name], dtype=np.float32).copy()
        return state

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        _load_state_dict_numpy_torch(
            state,
            torch=self.torch,
            net=self.net,
            empty_message="PODDeepONetTorch state dict does not contain expected torch::* weights",
        )
        for name in self.basis_keys:
            basis_key = f"basis::{name}"
            mean_key = f"mean::{name}"
            if basis_key in state:
                arr = np.asarray(state[basis_key], dtype=np.float32)
                self._basis_tensor(name).data.copy_(self.torch.as_tensor(arr, dtype=self.torch.float32))
            if mean_key in state:
                arr = np.asarray(state[mean_key], dtype=np.float32)
                self._mean_tensor(name).data.copy_(self.torch.as_tensor(arr, dtype=self.torch.float32))

    def to_meta(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "input_dim": int(self.input_dim),
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "out_channels": int(self.out_channels),
            "output_keys": list(self.output_keys),
            "backend": str(self.backend),
            "model_cfg": dict(self.model_cfg),
            "basis_keys": list(self.basis_keys),
            "basis_rank_by_var": {str(k): int(v) for k, v in self.basis_rank_by_var.items()},
        }


__all__ = [
    "PODBasisBundle",
    "PODDeepONetTorch",
    "fit_pod_basis_from_targets",
    "normalize_pod_deeponet_model_cfg",
]
