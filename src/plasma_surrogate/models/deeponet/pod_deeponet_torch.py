"""Experimental POD-based DeepONet full-field torch wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from plasma_surrogate.models._auxiliary_losses import (
    coefficient_aux_loss_diagnostics,
    coefficient_aux_loss_tensor,
    project_masked_pod_coefficients_torch,
)
from plasma_surrogate.models._torch_spatial_common import (
    _load_state_dict_numpy_torch,
    _resolve_torch_device,
    _state_dict_numpy_torch,
)


POD_DEEPONET_IMPL_VERSION = "deeponet_pod_v2_coeffnorm"


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
    coeff_loss_weight = float(cfg.get("coeff_loss_weight", 0.1))
    if not np.isfinite(coeff_loss_weight) or coeff_loss_weight < 0.0:
        raise ValueError(f"{cfg_prefix}.model_cfg.coeff_loss_weight must be finite and >= 0")
    return {
        "hidden": list(hidden),
        "latent_dim": int(hidden[-1]),
        "basis": basis_cfg,
        "coeff_loss_weight": float(coeff_loss_weight),
    }


@dataclass(frozen=True)
class PODBasisBundle:
    basis_by_var: dict[str, np.ndarray]
    mean_by_var: dict[str, np.ndarray]
    rank_by_var: dict[str, int]
    coeff_std_by_var: dict[str, np.ndarray]

    @classmethod
    def from_dicts(
        cls,
        *,
        basis_by_var: dict[str, np.ndarray],
        mean_by_var: dict[str, np.ndarray],
        rank_by_var: dict[str, int] | None = None,
        coeff_std_by_var: dict[str, np.ndarray] | None = None,
    ) -> "PODBasisBundle":
        basis_out: dict[str, np.ndarray] = {}
        mean_out: dict[str, np.ndarray] = {}
        rank_out: dict[str, int] = {}
        std_out: dict[str, np.ndarray] = {}
        rank_meta = dict(rank_by_var or {})
        std_meta = dict(coeff_std_by_var or {})
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
            std_arr = np.asarray(std_meta.get(key, np.ones((rank_eff,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if int(std_arr.shape[0]) != int(rank_eff):
                raise ValueError(
                    f"deeponet_pod coeff_std rank mismatch for {key}: coeff_std={std_arr.shape[0]}, rank={rank_eff}"
                )
            std_arr = np.where(std_arr > np.float32(1.0e-6), std_arr, np.float32(1.0)).astype(np.float32)
            std_out[key] = std_arr.copy()
        for name, arr in dict(mean_by_var or {}).items():
            mean_out[str(name)] = np.asarray(arr, dtype=np.float32).copy()
        missing_mean = sorted(set(basis_out.keys()) - set(mean_out.keys()))
        if missing_mean:
            raise ValueError(f"deeponet_pod missing mean for basis vars: {missing_mean}")
        return cls(
            basis_by_var=basis_out,
            mean_by_var=mean_out,
            rank_by_var=rank_out,
            coeff_std_by_var=std_out,
        )


def fit_pod_basis_from_targets(
    y_train_scaled: np.ndarray,
    *,
    output_keys: list[str],
    requested_rank: int,
    center: bool,
    per_var: bool,
    active_mask: np.ndarray | None = None,
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
    if active_mask is None:
        mask = np.ones((n_samples, h, w), dtype=bool)
    else:
        mask_raw = np.asarray(active_mask)
        if mask_raw.ndim == 2:
            mask_raw = np.broadcast_to(mask_raw[None, ...], (n_samples, h, w))
        elif mask_raw.ndim == 3 and int(mask_raw.shape[0]) == 1 and n_samples > 1:
            mask_raw = np.broadcast_to(mask_raw, (n_samples, h, w))
        if mask_raw.shape != (n_samples, h, w):
            raise ValueError(
                "deeponet_pod active_mask must align to [N,H,W]; "
                f"expected={(n_samples, h, w)}, got={mask_raw.shape}"
            )
        if not np.all(np.isfinite(mask_raw)):
            raise ValueError("deeponet_pod active_mask must contain only finite values")
        mask = np.asarray(mask_raw > 0.0, dtype=bool)
    empty_cases = np.flatnonzero(np.sum(mask, axis=(1, 2)) <= 0)
    if empty_cases.size:
        raise ValueError(
            "deeponet_pod basis fitting has empty active masks: "
            f"case_indices={empty_cases.astype(int).tolist()}"
        )
    basis_by_var: dict[str, np.ndarray] = {}
    mean_by_var: dict[str, np.ndarray] = {}
    rank_by_var: dict[str, int] = {}
    coeff_std_by_var: dict[str, np.ndarray] = {}
    max_rank = max(int(requested_rank), 1)
    flat_dim = int(h * w)
    for idx, var_name in enumerate(output_keys):
        snapshots = arr[:, idx].reshape(n_samples, flat_dim).astype(np.float32)
        mask_flat = mask.reshape(n_samples, flat_dim)
        invalid_active = mask_flat & ~np.isfinite(snapshots)
        if np.any(invalid_active):
            raise ValueError(
                "deeponet_pod basis fitting contains non-finite active targets: "
                f"target={var_name}, count={int(np.sum(invalid_active))}"
            )
        if bool(center):
            counts = np.sum(mask_flat, axis=0, dtype=np.float32)
            sums = np.sum(np.where(mask_flat, snapshots, 0.0), axis=0, dtype=np.float32)
            mean_flat = np.divide(
                sums,
                np.maximum(counts, 1.0),
                out=np.zeros_like(sums, dtype=np.float32),
            ).astype(np.float32)
        else:
            mean_flat = np.zeros((flat_dim,), dtype=np.float32)
        # Inactive cells contribute zero centered energy.  This keeps arbitrary
        # solid-region fill values out of the SVD while retaining the union of
        # all train-case plasma supports.
        centered = np.where(mask_flat, snapshots - mean_flat[None, :], 0.0).astype(np.float32)
        rank_eff = int(min(max_rank, n_samples, flat_dim))
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis_flat = np.asarray(vh[:rank_eff], dtype=np.float32)
        coeff_raw = centered @ basis_flat.T
        coeff_std = np.std(coeff_raw, axis=0, dtype=np.float32).astype(np.float32)
        coeff_std = np.where(coeff_std > np.float32(1.0e-6), coeff_std, np.float32(1.0)).astype(np.float32)
        basis_by_var[str(var_name)] = basis_flat.reshape(rank_eff, h, w).astype(np.float32)
        mean_by_var[str(var_name)] = mean_flat.reshape(h, w).astype(np.float32)
        rank_by_var[str(var_name)] = int(rank_eff)
        coeff_std_by_var[str(var_name)] = coeff_std.reshape(rank_eff).astype(np.float32)
    return PODBasisBundle.from_dicts(
        basis_by_var=basis_by_var,
        mean_by_var=mean_by_var,
        rank_by_var=rank_by_var,
        coeff_std_by_var=coeff_std_by_var,
    )


class PODDeepONetTorch:
    """Reduced-order torch wrapper that predicts POD coefficients from cond."""

    model_type = "deeponet_pod"
    pod_impl_version = POD_DEEPONET_IMPL_VERSION
    requires_spatial_features = False
    requires_scaled_spatial_features = False

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
        model_type: str = "deeponet_pod",
    ) -> None:
        self.model_type = str(model_type).strip().lower() or "deeponet_pod"
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.out_channels = int(out_channels)
        self.output_keys = list(output_keys or [f"out_{i}" for i in range(self.out_channels)])
        self.backend = str(backend).strip().lower()
        if self.backend != "torch":
            raise ValueError(f"{_cfg_prefix(self.model_type)}.model_cfg.backend must be torch")
        self.model_cfg = normalize_pod_deeponet_model_cfg(model_cfg, model_type=self.model_type)
        self.coeff_loss_weight = float(self.model_cfg.get("coeff_loss_weight", 0.1))
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
        self.device = _resolve_torch_device(self.torch)
        self.torch.manual_seed(int(seed))
        nn = self.torch.nn
        hidden = [int(v) for v in list(self.model_cfg["hidden"])]
        if len(hidden) == 0:
            raise ValueError("deeponet_pod model_cfg.hidden must be non-empty")
        self.net = nn.Module()
        trunk_layers: list[Any] = []
        dims = [self.input_dim, *hidden]
        for idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            trunk_layers.append(nn.Linear(int(fan_in), int(fan_out)))
            if idx != len(dims) - 2:
                trunk_layers.append(nn.GELU())
        self.net.trunk = nn.Sequential(*trunk_layers)
        self.net.var_heads = nn.ModuleDict(
            {str(name): nn.Linear(int(hidden[-1]), int(self.basis_rank_by_var[name])) for name in self.basis_keys}
        )
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
            self.net.register_buffer(
                f"_pod_coeff_std__{name}",
                self.torch.as_tensor(basis_bundle.coeff_std_by_var[name], dtype=self.torch.float32),
                persistent=False,
            )
        self.net.to(self.device)
        self._torch_last_out = None
        self._torch_last_coeff_norm = None

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

    def _reconstruct_from_coeff_norm(self, coeff_norm_t):
        bsz = int(coeff_norm_t.shape[0])
        fields = []
        for name in self.basis_keys:
            start, stop = self._coeff_slices[name]
            coeff_norm = coeff_norm_t[:, start:stop]
            coeff_var = coeff_norm * self._coeff_std_tensor(name).reshape(1, -1)
            basis_t = self._basis_tensor(name).reshape(int(stop - start), -1)
            mean_t = self._mean_tensor(name).reshape(1, -1)
            field_flat = self.torch.matmul(coeff_var, basis_t) + mean_t
            fields.append(field_flat.reshape(bsz, *self.grid_shape))
        return self.torch.stack(fields, dim=1)

    def _predict_coeff_norm_torch(self, cond_t):
        z = self.net.trunk(cond_t)
        chunks = [self.net.var_heads[name](z) for name in self.basis_keys]
        return self.torch.cat(chunks, dim=1)

    def _projection_mask_torch(self, supervised_mask: Any | None, target_t):
        if supervised_mask is None:
            if bool(self.torch.any(~self.torch.isfinite(target_t)).detach().cpu().item()):
                raise ValueError("deeponet_pod target_raw must contain only finite values")
            return None
        mask = self.torch.as_tensor(
            supervised_mask,
            dtype=self.torch.float32,
            device=target_t.device,
        )
        if mask.ndim == 2:
            mask = mask[None, ...]
        if mask.ndim == 4 and int(mask.shape[1]) == 1:
            mask = mask[:, 0]
        if mask.ndim != 3:
            raise ValueError(
                "deeponet_pod supervised_mask must be [H,W], [B,H,W], or [B,1,H,W]"
            )
        if int(mask.shape[0]) == 1 and int(target_t.shape[0]) > 1:
            mask = mask.expand(int(target_t.shape[0]), -1, -1)
        expected = (int(target_t.shape[0]), *self.grid_shape)
        if tuple(int(value) for value in mask.shape) != expected:
            raise ValueError(
                f"deeponet_pod supervised_mask mismatch: expected={expected}, got={tuple(mask.shape)}"
            )
        if bool(self.torch.any(~self.torch.isfinite(mask)).detach().cpu().item()):
            raise ValueError("deeponet_pod supervised_mask must contain only finite values")
        active = mask > 0.0
        invalid = active[:, None, :, :] & ~self.torch.isfinite(target_t)
        if bool(self.torch.any(invalid).detach().cpu().item()):
            raise ValueError("deeponet_pod target_raw is non-finite inside supervised_mask")
        return active.reshape(int(target_t.shape[0]), -1)

    def _project_target_coeff_norm_torch(self, target_t, *, supervised_mask: Any | None = None):
        active_flat = self._projection_mask_torch(supervised_mask, target_t)
        chunks = []
        for var_idx, name in enumerate(self.basis_keys):
            flat = target_t[:, var_idx].reshape(int(target_t.shape[0]), -1)
            mean_t = self._mean_tensor(name).reshape(1, -1)
            centered = flat - mean_t
            basis_t = self._basis_tensor(name).reshape(int(self.basis_rank_by_var[name]), -1)
            coeff_raw = project_masked_pod_coefficients_torch(
                centered,
                basis_t,
                active_mask=active_flat,
            )
            coeff_norm = coeff_raw / self._coeff_std_tensor(name).reshape(1, -1)
            chunks.append(coeff_norm)
        return self.torch.cat(chunks, dim=1)

    def _resolve_coeff_loss_weight(self, loss_cfg: dict[str, Any] | None) -> float:
        weight = float(self.coeff_loss_weight)
        if isinstance(loss_cfg, dict):
            weight = float(dict(loss_cfg.get("deeponet_pod", {})).get("coeff_loss_weight", weight))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("deeponet_pod coeff_loss_weight must be finite and >= 0")
        return float(weight)

    def _validate_auxiliary_evaluation_inputs(
        self,
        cond: np.ndarray,
        target_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        if cond_arr.ndim != 2:
            raise ValueError(f"deeponet_pod cond must be [B,D], got {cond_arr.shape}")
        if int(cond_arr.shape[0]) < 1:
            raise ValueError("deeponet_pod cond must contain at least one case")
        if int(cond_arr.shape[1]) != int(self.input_dim):
            raise ValueError(
                f"deeponet_pod cond feature mismatch: expected {self.input_dim}, got {cond_arr.shape[1]}"
            )
        if not np.all(np.isfinite(cond_arr)):
            raise ValueError("deeponet_pod cond must contain only finite values")

        target_arr = np.asarray(target_raw, dtype=np.float32)
        if target_arr.ndim != 4:
            raise ValueError(f"deeponet_pod target_raw must be [B,C,H,W], got {target_arr.shape}")
        if int(target_arr.shape[0]) != int(cond_arr.shape[0]):
            raise ValueError(
                "deeponet_pod target_raw batch mismatch: "
                f"expected {int(cond_arr.shape[0])}, got {int(target_arr.shape[0])}"
            )
        if int(target_arr.shape[1]) != int(len(self.basis_keys)):
            raise ValueError(
                f"deeponet_pod target_raw channel mismatch: expected {len(self.basis_keys)}, "
                f"got {target_arr.shape[1]}"
            )
        if tuple(int(v) for v in target_arr.shape[2:]) != tuple(self.grid_shape):
            raise ValueError(
                f"deeponet_pod target_raw grid mismatch: expected {self.grid_shape}, got {target_arr.shape[2:]}"
            )
        return cond_arr.astype(np.float32, copy=False), target_arr.astype(np.float32, copy=False)

    def evaluate_auxiliary_losses(
        self,
        cond: np.ndarray,
        target_raw: np.ndarray,
        *,
        spatial_features: np.ndarray | None = None,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Evaluate coefficient diagnostics without consuming training caches or gradients."""

        del spatial_features  # POD coefficients depend on conditions only.
        cond_arr, target_arr = self._validate_auxiliary_evaluation_inputs(cond, target_raw)
        weight = self._resolve_coeff_loss_weight(loss_cfg)
        module_modes = [(module, bool(module.training)) for module in self.net.modules()]
        cached_out = self._torch_last_out
        cached_coeff = self._torch_last_coeff_norm
        device = next(self.net.parameters()).device
        try:
            self.net.eval()
            with self.torch.no_grad():
                cond_t = self.torch.as_tensor(cond_arr, dtype=self.torch.float32, device=device)
                target_t = self.torch.as_tensor(target_arr, dtype=self.torch.float32, device=device)
                coeff_pred = self._predict_coeff_norm_torch(cond_t)
                coeff_target = self._project_target_coeff_norm_torch(
                    target_t,
                    supervised_mask=supervised_mask,
                )
                coeff_loss, by_target, by_group = coefficient_aux_loss_tensor(
                    coeff_pred,
                    coeff_target,
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
            self._torch_last_coeff_norm = cached_coeff
        return coefficient_aux_loss_diagnostics(
            coeff_loss=coeff_loss_value,
            coeff_loss_weight=weight,
            coeff_loss_by_target=target_values,
            coeff_loss_by_group=group_values,
        )

    def _cached_coeff_loss_tensor(
        self,
        target_raw: np.ndarray,
        *,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ):
        if self._torch_last_coeff_norm is None:
            raise RuntimeError("PODDeepONetTorch auxiliary loss requires a cached training forward")
        target_arr = np.asarray(target_raw, dtype=np.float32)
        if target_arr.ndim != 4:
            raise ValueError(f"deeponet_pod target_raw must be [B,C,H,W], got {target_arr.shape}")
        if int(target_arr.shape[1]) != int(len(self.basis_keys)):
            raise ValueError(
                f"deeponet_pod target_raw channel mismatch: expected {len(self.basis_keys)}, got {target_arr.shape[1]}"
            )
        if int(target_arr.shape[0]) != int(self._torch_last_coeff_norm.shape[0]):
            raise ValueError(
                "deeponet_pod target_raw batch mismatch: "
                f"expected {int(self._torch_last_coeff_norm.shape[0])}, got {int(target_arr.shape[0])}"
            )
        if tuple(int(v) for v in target_arr.shape[2:]) != tuple(self.grid_shape):
            raise ValueError(
                f"deeponet_pod target_raw grid mismatch: expected {self.grid_shape}, got {target_arr.shape[2:]}"
            )
        target_t = self.torch.as_tensor(
            target_arr,
            dtype=self.torch.float32,
            device=getattr(self._torch_last_coeff_norm, "device", None),
        )
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

    def _forward_raw_torch(self, cond: np.ndarray, *, training: bool) -> np.ndarray:
        cond_arr = np.asarray(cond, dtype=np.float32)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr[None, :]
        device = next(self.net.parameters()).device
        cond_t = self.torch.as_tensor(cond_arr.astype(np.float32), dtype=self.torch.float32, device=device)
        if bool(training):
            self.net.train()
            coeff_norm_t = self._predict_coeff_norm_torch(cond_t)
            self._torch_last_coeff_norm = coeff_norm_t
            self._torch_last_out = self._reconstruct_from_coeff_norm(coeff_norm_t)
        else:
            self.net.eval()
            with self.torch.no_grad():
                self._torch_last_out = None
                self._torch_last_coeff_norm = None
                coeff_norm_t = self._predict_coeff_norm_torch(cond_t)
                return self._reconstruct_from_coeff_norm(coeff_norm_t).detach().cpu().numpy().astype(np.float32)
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
        for module in self.net.modules():
            if hasattr(module, "weight"):
                first = module.weight
                break
        for module in reversed(list(self.net.modules())):
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
        if self._torch_last_out is None:
            raise RuntimeError("PODDeepONetTorch.backward_raw called without torch forward cache")
        if self._torch_last_coeff_norm is None:
            raise RuntimeError("PODDeepONetTorch.backward_raw missing coeff cache")
        grad_np = np.asarray(grad_raw, dtype=np.float32)
        grad_t = self.torch.as_tensor(
            grad_np,
            dtype=getattr(self._torch_last_out, "dtype", None) or self.torch.float32,
            device=getattr(self._torch_last_out, "device", None),
        )
        params = [p for p in self.net.parameters() if p.requires_grad]
        if not params:
            self._torch_last_out = None
            self._torch_last_coeff_norm = None
            return {
                "step_rel_hidden_mean": 0.0,
                "step_rel_output": 0.0,
                **coefficient_aux_loss_diagnostics(
                    coeff_loss=0.0,
                    coeff_loss_weight=self._resolve_coeff_loss_weight(loss_cfg),
                ),
            }
        hidden_w, out_w = self._torch_step_reference()
        with self.torch.no_grad():
            w_hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
            w_out_prev = out_w.detach().clone() if out_w is not None else None
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
        self._torch_last_out.backward(grad_t, retain_graph=bool(use_coeff_loss))
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
                if hidden_cur is not None and w_hidden_prev is not None:
                    dh = hidden_cur.detach() - w_hidden_prev
                    step_hidden = float(
                        self.torch.linalg.norm(dh) / max(float(self.torch.linalg.norm(w_hidden_prev)), 1.0e-12)
                    )
                if out_cur is not None and w_out_prev is not None:
                    do = out_cur.detach() - w_out_prev
                    step_out = float(
                        self.torch.linalg.norm(do) / max(float(self.torch.linalg.norm(w_out_prev)), 1.0e-12)
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
            empty_message="PODDeepONetTorch state dict does not contain expected torch::* weights",
            device=self.device,
        )
        self.net.to(self.device)
        for name in self.basis_keys:
            basis_key = f"basis::{name}"
            mean_key = f"mean::{name}"
            coeff_std_key = f"coeff_std::{name}"
            if basis_key in state:
                arr = np.asarray(state[basis_key], dtype=np.float32)
                self._basis_tensor(name).data.copy_(self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device))
            if mean_key in state:
                arr = np.asarray(state[mean_key], dtype=np.float32)
                self._mean_tensor(name).data.copy_(self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device))
            if coeff_std_key in state:
                arr = np.asarray(state[coeff_std_key], dtype=np.float32).reshape(-1)
                self._coeff_std_tensor(name).data.copy_(self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device))
            else:
                raise ValueError(
                    "legacy deeponet_pod checkpoint format is not supported; missing coeff_std::<var> entries"
                )

    def to_meta(self) -> dict[str, Any]:
        basis_bundle = self.basis_bundle_numpy()
        return {
            "model_type": self.model_type,
            "impl_version": self.pod_impl_version,
            "input_dim": int(self.input_dim),
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "out_channels": int(self.out_channels),
            "output_keys": list(self.output_keys),
            "backend": str(self.backend),
            "model_cfg": dict(self.model_cfg),
            "basis_keys": list(self.basis_keys),
            "basis_rank_by_var": {str(k): int(v) for k, v in self.basis_rank_by_var.items()},
            "coeff_std_by_var": {
                str(k): np.asarray(v, dtype=np.float32).reshape(-1).tolist()
                for k, v in basis_bundle.coeff_std_by_var.items()
            },
        }


__all__ = [
    "PODBasisBundle",
    "PODDeepONetTorch",
    "POD_DEEPONET_IMPL_VERSION",
    "fit_pod_basis_from_targets",
    "normalize_pod_deeponet_model_cfg",
]
