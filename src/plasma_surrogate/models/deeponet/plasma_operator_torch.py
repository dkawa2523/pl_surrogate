"""Minimal torch DeepONet operator for product training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.models._torch_spatial_common import _resolve_torch_device


def _safe_logit(prob: float) -> float:
    p = float(np.clip(float(prob), 1.0e-4, 1.0 - 1.0e-4))
    return float(np.log(p / (1.0 - p)))


class DeepONetPlasmaOperatorTorch:
    """Lightweight DeepONet-like operator with branch/trunk factorization."""

    model_type = "deeponet_plasma"
    requires_spatial_features = True
    requires_scaled_spatial_features = False

    def __init__(
        self,
        cond_dim: int,
        grid_shape: tuple[int, int],
        output_keys: list[str] | None = None,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        sensor_indices: np.ndarray | None = None,
        query_indices: np.ndarray | None = None,
        flatten_order: str = "C",
        sensor_feature_names: list[str] | None = None,
        query_feature_names: list[str] | None = None,
        trunk_input_mode: str = "geom_feature_pack",
        sensor_pool_mode: str = "moments",
        sensor_embed_dim: int = 32,
        branch_mode: str = "moments",
        trunk_fourier_n_freq: int = 1,
        trunk_fourier_mode: str = "symmetric",
        trunk_cond_modulation: str = "none",
        trunk_cond_mod_hidden: int = 64,
        residual_head_enabled: bool = False,
        residual_head_hidden_dim: int = 64,
        residual_head_scale_init: float = 0.0,
        residual_head_gain_mode: str = "learned",
        residual_head_gain_value: float = 1.0,
        latent_layer_norm: bool = False,
        output_path_mode: str = "dot",
        output_path_dot_skip: float = 0.25,
        output_path_dot_skip_mode: str = "fixed",
        output_path_fused_hidden_dim: int = 96,
        output_path_global_local_enabled: bool = False,
        output_path_global_hidden_dim: int = 64,
        missing_geom_feature_policy: str = "error",
        seed: int = 0,
    ) -> None:
        torch = require_torch()
        self._torch = torch
        self.device = _resolve_torch_device(torch)
        self.cond_dim = int(cond_dim)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.flatten_order = str(flatten_order)
        if self.flatten_order not in {"C", "F"}:
            raise ValueError("flatten_order must be 'C' or 'F'")

        self.output_keys = list(output_keys or ["target_0", "target_1", "target_2", "target_3"])
        self.out_dim = len(self.output_keys)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_points = int(self.grid_shape[0] * self.grid_shape[1])
        self.sensor_feature_names = list(sensor_feature_names or ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
        # Use the same public channel contract as the other spatial models.  The
        # rows themselves are deliberately not checkpointed: inference rebuilds
        # them from the run bundle's coordinate pack and transform artifacts.
        self.input_feature_channels = list(self.sensor_feature_names)
        self.sensor_feature_dim = int(max(len(self.sensor_feature_names), 1))
        derived_query_features = [name for name in self.sensor_feature_names if name not in {"x", "y"}]
        self.query_feature_names = list(
            derived_query_features if query_feature_names is None else query_feature_names
        )
        missing_query_features = [
            name for name in self.query_feature_names if name not in set(self.sensor_feature_names)
        ]
        if missing_query_features:
            raise ValueError(
                "query_feature_names must be a subset of sensor_feature_names; "
                f"missing={missing_query_features}"
            )
        self.trunk_input_mode = str(trunk_input_mode).strip().lower()
        if self.trunk_input_mode != "geom_feature_pack":
            raise ValueError("trunk_input_mode must be geom_feature_pack")
        self.sensor_pool_mode = str(sensor_pool_mode).strip().lower()
        if self.sensor_pool_mode not in {"moments", "set_mlp_pool"}:
            raise ValueError("sensor_pool_mode must be one of: moments, set_mlp_pool")
        self.trunk_fourier_n_freq = int(max(int(trunk_fourier_n_freq), 1))
        self.trunk_fourier_mode = str(trunk_fourier_mode).strip().lower()
        if self.trunk_fourier_mode != "symmetric":
            raise ValueError("trunk_fourier_mode must be symmetric")
        self.trunk_cond_modulation = str(trunk_cond_modulation).strip().lower()
        if self.trunk_cond_modulation not in {"none", "film"}:
            raise ValueError("trunk_cond_modulation must be one of: none, film")
        self.trunk_cond_mod_hidden = int(max(int(trunk_cond_mod_hidden), 8))
        self.residual_head_enabled = bool(residual_head_enabled)
        self.residual_head_hidden_dim = int(max(int(residual_head_hidden_dim), 8))
        self.residual_head_scale_init = float(residual_head_scale_init)
        self.residual_head_gain_mode = str(residual_head_gain_mode).strip().lower()
        if self.residual_head_gain_mode not in {"learned", "fixed"}:
            raise ValueError("residual_head_gain_mode must be one of: learned, fixed")
        self.residual_head_gain_value = float(residual_head_gain_value)
        self.latent_layer_norm = bool(latent_layer_norm)
        self.output_path_mode = str(output_path_mode).strip().lower()
        if self.output_path_mode not in {"dot", "fused"}:
            raise ValueError("output_path_mode must be one of: dot, fused")
        self.output_path_dot_skip = float(output_path_dot_skip)
        if not np.isfinite(self.output_path_dot_skip):
            raise ValueError("output_path_dot_skip must be finite")
        self.output_path_dot_skip_mode = str(output_path_dot_skip_mode).strip().lower()
        if self.output_path_dot_skip_mode not in {"fixed", "learned_per_var"}:
            raise ValueError("output_path_dot_skip_mode must be one of: fixed, learned_per_var")
        self.output_path_fused_hidden_dim = int(max(int(output_path_fused_hidden_dim), 8))
        self.output_path_global_local_enabled = bool(output_path_global_local_enabled)
        self.output_path_global_hidden_dim = int(max(int(output_path_global_hidden_dim), 8))
        if str(missing_geom_feature_policy).strip().lower() != "error":
            raise ValueError("missing_geom_feature_policy must be error")
        self.missing_geom_feature_policy = "error"
        self.branch_mode = str(branch_mode).strip().lower()
        if self.branch_mode not in {"moments", "set_mlp_pool", "cond_only"}:
            raise ValueError("branch_mode must be one of: moments, set_mlp_pool, cond_only")
        if self.branch_mode == "set_mlp_pool":
            self.sensor_pool_mode = "set_mlp_pool"
        if self.branch_mode == "cond_only":
            self.sensor_pool_mode = "moments"
        self.sensor_embed_dim = int(max(int(sensor_embed_dim), 4))
        self.sensor_indices = (
            np.asarray(sensor_indices, dtype=np.int64).reshape(-1) if sensor_indices is not None else np.arange(min(32, self.n_points), dtype=np.int64)
        )
        self.query_indices = (
            np.asarray(query_indices, dtype=np.int64).reshape(-1) if query_indices is not None else np.arange(self.n_points, dtype=np.int64)
        )
        self._validate_indices(self.sensor_indices, "sensor_indices")
        self._validate_indices(self.query_indices, "query_indices")

        self._coord_flat = self._build_coord_flat(self.grid_shape, self.flatten_order)
        self._torch.manual_seed(int(seed))
        nn = self._torch.nn
        if self.branch_mode == "cond_only":
            self.sensor_encoder = None
            branch_input_dim = self.cond_dim
        elif self.sensor_pool_mode == "set_mlp_pool":
            self.sensor_encoder = nn.Sequential(
                nn.Linear(self.sensor_feature_dim, self.sensor_embed_dim),
                nn.Tanh(),
                nn.Linear(self.sensor_embed_dim, self.sensor_embed_dim),
                nn.Tanh(),
            )
            branch_input_dim = self.cond_dim + 2 * self.sensor_embed_dim
        else:
            self.sensor_encoder = None
            branch_input_dim = self.cond_dim + 4 * self.sensor_feature_dim
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.out_dim * self.latent_dim),
        )
        trunk_base_dim = 2 + 4 * int(self.trunk_fourier_n_freq)
        trunk_input_dim = trunk_base_dim + len(self.query_feature_names)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        if self.output_path_mode == "fused":
            self.fused_head = nn.Sequential(
                nn.Linear(3 * self.latent_dim, self.output_path_fused_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.output_path_fused_hidden_dim, 1),
            )
            if self.output_path_dot_skip_mode == "learned_per_var":
                self.output_path_dot_skip_logits = nn.Parameter(
                    self._torch.full(
                        (self.out_dim,),
                        _safe_logit(self.output_path_dot_skip),
                        dtype=self._torch.float32,
                        device=self.device,
                    )
                )
            else:
                self.output_path_dot_skip_logits = None
        else:
            self.fused_head = None
            self.output_path_dot_skip_logits = None
        if self.output_path_global_local_enabled:
            self.global_head = nn.Sequential(
                nn.Linear(self.cond_dim, self.output_path_global_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.output_path_global_hidden_dim, self.out_dim),
            )
        else:
            self.global_head = None
        if self.latent_layer_norm:
            self.branch_latent_norm = nn.LayerNorm(self.latent_dim)
            self.trunk_latent_norm = nn.LayerNorm(self.latent_dim)
        else:
            self.branch_latent_norm = None
            self.trunk_latent_norm = None
        if self.trunk_cond_modulation == "film":
            self.trunk_film = nn.Sequential(
                nn.Linear(self.cond_dim, self.trunk_cond_mod_hidden),
                nn.Tanh(),
                nn.Linear(self.trunk_cond_mod_hidden, 2 * self.latent_dim),
            )
        else:
            self.trunk_film = None
        if self.residual_head_enabled:
            self.residual_cond = nn.Sequential(
                nn.Linear(self.cond_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(self.latent_dim + self.hidden_dim, self.residual_head_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.residual_head_hidden_dim, self.out_dim),
            )
            if self.residual_head_gain_mode == "learned":
                self.residual_scale = nn.Parameter(
                    self._torch.full(
                        (1,),
                        float(self.residual_head_scale_init),
                        dtype=self._torch.float32,
                        device=self.device,
                    )
                )
            else:
                self.residual_scale = None
        else:
            self.residual_cond = None
            self.residual_head = None
            self.residual_scale = None
        self.out_bias = nn.Parameter(self._torch.zeros((self.out_dim,), dtype=self._torch.float32, device=self.device))
        self._attached_poisson_head: Any | None = None
        self._attached_boundary_operator: Any | None = None
        self._static_feature_rows: np.ndarray | None = None
        self._static_feature_channels: list[str] = []
        self._move_modules_to_device()

    @staticmethod
    def _build_coord_flat(shape: tuple[int, int], order: str) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        coord = np.stack([xv, yv], axis=-1)
        return coord.reshape(-1, 2, order=order).astype(np.float32)

    def _validate_indices(self, idx: np.ndarray, name: str) -> None:
        if idx.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if int(np.min(idx)) < 0 or int(np.max(idx)) >= self.n_points:
            raise ValueError(f"{name} out of range for n_points={self.n_points}")

    def _handle_missing_geom_features(self, *, role: str, missing_names: list[str]) -> None:
        names = tuple(sorted(str(v) for v in list(missing_names) if str(v)))
        if len(names) == 0:
            return
        msg = (
            f"DeepONet geom-feature missing for {role}: missing={list(names)}; "
            "policy=error"
        )
        raise ValueError(msg)

    def named_parameters(self):
        seen: set[int] = set()

        def _yield_named(module: Any | None, prefix: str):
            if module is None or not hasattr(module, "named_parameters"):
                return
            for name, p in module.named_parameters():
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                yield f"{prefix}{name}", p

        if self.sensor_encoder is not None:
            yield from _yield_named(self.sensor_encoder, "sensor_encoder.")
        yield from _yield_named(self.branch, "branch.")
        if self.fused_head is not None:
            yield from _yield_named(self.fused_head, "fused_head.")
        if self.global_head is not None:
            yield from _yield_named(self.global_head, "global_head.")
        if self.branch_latent_norm is not None:
            yield from _yield_named(self.branch_latent_norm, "branch_latent_norm.")
        yield from _yield_named(self.trunk, "trunk.")
        if self.trunk_latent_norm is not None:
            yield from _yield_named(self.trunk_latent_norm, "trunk_latent_norm.")
        if self.trunk_film is not None:
            yield from _yield_named(self.trunk_film, "trunk_film.")
        if self.residual_cond is not None:
            yield from _yield_named(self.residual_cond, "residual_cond.")
        if self.residual_head is not None:
            yield from _yield_named(self.residual_head, "residual_head.")
        if self.residual_scale is not None:
            pid = id(self.residual_scale)
            if pid not in seen:
                seen.add(pid)
                yield "residual_scale", self.residual_scale
        pid = id(self.out_bias)
        if pid not in seen:
            seen.add(pid)
            yield "out_bias", self.out_bias
        if self.output_path_dot_skip_logits is not None:
            pid = id(self.output_path_dot_skip_logits)
            if pid not in seen:
                seen.add(pid)
                yield "output_path_dot_skip_logits", self.output_path_dot_skip_logits
        if self._attached_poisson_head is not None and hasattr(self._attached_poisson_head, "named_parameters"):
            yield from _yield_named(self._attached_poisson_head, "poisson_head.")
        if self._attached_boundary_operator is not None and hasattr(self._attached_boundary_operator, "named_parameters"):
            yield from _yield_named(self._attached_boundary_operator, "boundary_operator.")

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def _move_modules_to_device(self) -> None:
        module_names = (
            "sensor_encoder",
            "branch",
            "trunk",
            "fused_head",
            "global_head",
            "branch_latent_norm",
            "trunk_latent_norm",
            "trunk_film",
            "residual_cond",
            "residual_head",
        )
        for name in module_names:
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "to"):
                module.to(self.device)
        for name in ("residual_scale", "out_bias", "output_path_dot_skip_logits"):
            param = getattr(self, name, None)
            if param is not None and hasattr(param, "data"):
                param.data = param.data.to(self.device)
        for name in ("_attached_poisson_head", "_attached_boundary_operator"):
            obj = getattr(self, name, None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(self.device)

    def to(self, device):
        self.device = device
        self._move_modules_to_device()
        return self

    def train(self) -> None:
        if self.sensor_encoder is not None:
            self.sensor_encoder.train()
        self.branch.train()
        self.trunk.train()
        if self.fused_head is not None:
            self.fused_head.train()
        if self.global_head is not None:
            self.global_head.train()
        if self.trunk_film is not None:
            self.trunk_film.train()
        if self.branch_latent_norm is not None:
            self.branch_latent_norm.train()
        if self.trunk_latent_norm is not None:
            self.trunk_latent_norm.train()
        if self.residual_cond is not None:
            self.residual_cond.train()
        if self.residual_head is not None:
            self.residual_head.train()
        if self._attached_poisson_head is not None:
            self._attached_poisson_head.train()
        if self._attached_boundary_operator is not None:
            self._attached_boundary_operator.train()

    def eval(self) -> None:
        if self.sensor_encoder is not None:
            self.sensor_encoder.eval()
        self.branch.eval()
        self.trunk.eval()
        if self.fused_head is not None:
            self.fused_head.eval()
        if self.global_head is not None:
            self.global_head.eval()
        if self.trunk_film is not None:
            self.trunk_film.eval()
        if self.branch_latent_norm is not None:
            self.branch_latent_norm.eval()
        if self.trunk_latent_norm is not None:
            self.trunk_latent_norm.eval()
        if self.residual_cond is not None:
            self.residual_cond.eval()
        if self.residual_head is not None:
            self.residual_head.eval()
        if self._attached_poisson_head is not None:
            self._attached_poisson_head.eval()
        if self._attached_boundary_operator is not None:
            self._attached_boundary_operator.eval()

    def attach_poisson_head(self, head: Any) -> None:
        self._attached_poisson_head = head
        if hasattr(head, "to"):
            head.to(self.device)

    def attach_boundary_operator(self, op: Any) -> None:
        self._attached_boundary_operator = op
        if hasattr(op, "to"):
            op.to(self.device)

    def set_static_spatial_features(
        self,
        rows: np.ndarray,
        *,
        channels: list[str] | None = None,
    ) -> None:
        arr = np.asarray(rows, dtype=np.float32)
        effective_channels = list(self.input_feature_channels if channels is None else channels)
        if arr.ndim != 2:
            raise ValueError("DeepONet static feature rows must be rank-2 [n_points, n_channels]")
        if int(arr.shape[0]) != int(self.n_points):
            raise ValueError(
                f"DeepONet static feature rows n_points mismatch: expected={self.n_points}, got={arr.shape[0]}"
            )
        if len(effective_channels) != int(arr.shape[1]):
            raise ValueError(
                "DeepONet static feature rows channel count mismatch: "
                f"expected={arr.shape[1]} from rows, got={len(effective_channels)} channels"
            )
        self._static_feature_rows = arr.copy()
        self._static_feature_channels = [str(v) for v in effective_channels]

    def _spatial_feature_tensor(
        self,
        spatial_features: Any | None,
        *,
        batch_size: int,
        dtype: Any,
        device: Any,
    ) -> tuple[Any | None, list[str]]:
        raw = self._static_feature_rows if spatial_features is None else spatial_features
        channels = (
            list(self._static_feature_channels)
            if spatial_features is None
            else list(self.input_feature_channels)
        )
        if raw is None or not channels:
            return None, []
        if self._torch.is_tensor(raw):
            rows = raw.to(device=device, dtype=dtype)
        else:
            rows = self._torch.as_tensor(np.asarray(raw, dtype=np.float32), dtype=dtype, device=device)
        h, w = self.grid_shape
        n_points = int(h * w)
        if rows.ndim == 2:
            rows = rows[None, ...]
        elif rows.ndim == 3 and tuple(int(v) for v in rows.shape[:2]) == (h, w):
            rows = rows.reshape(1, n_points, int(rows.shape[-1]))
        elif rows.ndim == 4 and tuple(int(v) for v in rows.shape[1:3]) == (h, w):
            rows = rows.reshape(int(rows.shape[0]), n_points, int(rows.shape[-1]))
        elif rows.ndim != 3:
            raise ValueError(
                "DeepONet spatial features must be [N,C], [H,W,C], [B,N,C], or [B,H,W,C]; "
                f"got shape={tuple(int(v) for v in rows.shape)}"
            )
        if int(rows.shape[1]) != n_points:
            raise ValueError(
                "DeepONet spatial feature point count mismatch: "
                f"expected={n_points}, got={int(rows.shape[1])}"
            )
        if int(rows.shape[2]) != len(channels):
            raise ValueError(
                "DeepONet spatial feature channel count mismatch: "
                f"expected={len(channels)}, got={int(rows.shape[2])}"
            )
        if int(rows.shape[0]) == 1 and int(batch_size) > 1:
            rows = rows.expand(int(batch_size), -1, -1)
        elif int(rows.shape[0]) != int(batch_size):
            raise ValueError(
                "DeepONet spatial feature batch mismatch: "
                f"expected=1 or {int(batch_size)}, got={int(rows.shape[0])}"
            )
        return rows, channels

    @property
    def poisson_head(self) -> Any | None:
        return self._attached_poisson_head

    @property
    def boundary_operator(self) -> Any | None:
        return self._attached_boundary_operator

    def _trunk_features(self, query: dict[str, Any], x_q):
        qx = x_q[..., 0]
        qy = x_q[..., 1]
        base_parts = [qx, qy]
        for k in range(1, int(self.trunk_fourier_n_freq) + 1):
            kk = float(k)
            base_parts.extend(
                [
                    self._torch.sin(2.0 * np.pi * kk * qx),
                    self._torch.cos(2.0 * np.pi * kk * qx),
                    self._torch.sin(2.0 * np.pi * kk * qy),
                    self._torch.cos(2.0 * np.pi * kk * qy),
                ]
            )
        base = self._torch.stack(base_parts, dim=-1)
        qf = query.get("f")
        if qf is None:
            self._handle_missing_geom_features(
                role="query.f",
                missing_names=list(self.query_feature_names),
            )
        qf = self._torch.as_tensor(qf, dtype=x_q.dtype, device=x_q.device)
        if qf.ndim == 2:
            qf = qf[None, ...].expand(int(x_q.shape[0]), -1, -1)
        if int(qf.shape[0]) != int(x_q.shape[0]) or int(qf.shape[1]) != int(x_q.shape[1]):
            raise ValueError("query['f'] shape mismatch in DeepONet trunk features")
        if int(qf.shape[-1]) != len(self.query_feature_names):
            raise ValueError(
                "query['f'] feature dim mismatch: "
                f"expected={len(self.query_feature_names)}, got={int(qf.shape[-1])}"
            )
        return self._torch.cat([base, qf], dim=-1)

    def _branch_features(self, sensors: dict[str, Any], cond):
        if self.branch_mode == "cond_only":
            return cond
        sv = sensors.get("v")
        if sv is None:
            self._handle_missing_geom_features(
                role="sensors.v",
                missing_names=list(self.sensor_feature_names),
            )
        sv = self._torch.as_tensor(sv, dtype=cond.dtype, device=cond.device)
        if sv.ndim == 2:
            sv = sv[:, :, None]
        if int(sv.shape[-1]) != self.sensor_feature_dim:
            raise ValueError(
                "sensors['v'] feature dim mismatch: "
                f"expected={self.sensor_feature_dim}, got={int(sv.shape[-1])}"
            )
        if self.sensor_pool_mode == "set_mlp_pool":
            if self.sensor_encoder is None:
                raise RuntimeError("sensor_encoder is required for sensor_pool_mode=set_mlp_pool")
            bsz = int(sv.shape[0])
            n_s = int(sv.shape[1])
            enc = self.sensor_encoder(sv.reshape(bsz * n_s, self.sensor_feature_dim)).reshape(
                bsz, n_s, self.sensor_embed_dim
            )
            pooled = self._torch.cat([enc.mean(dim=1), enc.std(dim=1, unbiased=False)], dim=1)
        else:
            pooled = self._torch.stack(
                [
                    sv.mean(dim=1),
                    sv.std(dim=1, unbiased=False),
                    sv.amax(dim=1),
                    sv.amin(dim=1),
                ],
                dim=1,
            ).reshape(int(sv.shape[0]), 4 * self.sensor_feature_dim)
        return self._torch.cat([cond, pooled], dim=1)

    @staticmethod
    def _as_hw_array(raw: Any, *, h: int, w: int) -> np.ndarray | None:
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape == (h, w):
            return arr
        if arr.shape == (1, h, w):
            return arr[0]
        if arr.shape == (h, w, 1):
            return arr[:, :, 0]
        return None

    def _sensor_values_from_geom(self, *, geom_ctx: Any, coord_flat: np.ndarray, sensor_indices: np.ndarray, bsz: int, dtype, device):
        h, w = self.grid_shape
        feature_map: dict[str, np.ndarray] = {
            "x": np.asarray(coord_flat[:, 0], dtype=np.float32),
            "y": np.asarray(coord_flat[:, 1], dtype=np.float32),
        }
        for key in ("mask_plasma", "distance_signed", "distance_any"):
            raw = getattr(geom_ctx, key, None) if geom_ctx is not None else None
            if raw is None and geom_ctx is not None and key == "mask_plasma" and hasattr(geom_ctx, "regions"):
                raw = dict(getattr(geom_ctx, "regions", {}) or {}).get("plasma_mask")
            hw_arr = self._as_hw_array(raw, h=h, w=w)
            if hw_arr is not None:
                feature_map[key] = hw_arr.reshape(-1).astype(np.float32)
        missing = [name for name in self.sensor_feature_names if name not in feature_map]
        self._handle_missing_geom_features(role="geom_ctx.sensor_features", missing_names=missing)
        cols = [np.asarray(feature_map[name], dtype=np.float32) for name in self.sensor_feature_names]
        stacked = np.stack(cols, axis=1)
        sampled = stacked[np.asarray(sensor_indices, dtype=np.int64)]
        sampled_t = self._torch.as_tensor(sampled, dtype=dtype, device=device)
        return sampled_t[None, ...].expand(int(bsz), -1, -1)

    def _query_values_from_geom(self, *, geom_ctx: Any, coord_flat: np.ndarray, query_indices: np.ndarray, bsz: int, dtype, device):
        h, w = self.grid_shape
        feature_map: dict[str, np.ndarray] = {}
        for key in self.query_feature_names:
            raw = getattr(geom_ctx, key, None) if geom_ctx is not None else None
            if raw is None and geom_ctx is not None and key == "mask_plasma" and hasattr(geom_ctx, "regions"):
                raw = dict(getattr(geom_ctx, "regions", {}) or {}).get("plasma_mask")
            hw_arr = self._as_hw_array(raw, h=h, w=w)
            if hw_arr is not None:
                feature_map[key] = hw_arr.reshape(-1).astype(np.float32)
        missing = [name for name in self.query_feature_names if name not in feature_map]
        self._handle_missing_geom_features(role="geom_ctx.query_features", missing_names=missing)
        cols = [np.asarray(feature_map[name], dtype=np.float32) for name in self.query_feature_names]
        stacked = np.stack(cols, axis=1)
        sampled = stacked[np.asarray(query_indices, dtype=np.int64)]
        sampled_t = self._torch.as_tensor(sampled, dtype=dtype, device=device)
        return sampled_t[None, ...].expand(int(bsz), -1, -1)

    def forward(self, sensors: dict[str, Any], query: dict[str, Any], cond):
        torch = self._torch
        self._move_modules_to_device()
        cond_t = torch.as_tensor(cond, dtype=torch.float32, device=self.device)
        x_q = torch.as_tensor(query["x"], dtype=torch.float32, device=self.device)
        if x_q.ndim == 2:
            x_q = x_q[None, ...].expand(cond_t.shape[0], -1, -1)
        branch_in = self._branch_features(sensors, cond_t)
        trunk_in = self._trunk_features(query, x_q)
        b_lat = self.branch(branch_in).reshape(cond_t.shape[0], self.out_dim, self.latent_dim)
        t_lat = self.trunk(trunk_in.reshape(-1, trunk_in.shape[-1])).reshape(cond_t.shape[0], x_q.shape[1], self.latent_dim)
        if self.branch_latent_norm is not None:
            b_lat = self.branch_latent_norm(b_lat.reshape(-1, self.latent_dim)).reshape(
                cond_t.shape[0], self.out_dim, self.latent_dim
            )
        if self.trunk_latent_norm is not None:
            t_lat = self.trunk_latent_norm(t_lat.reshape(-1, self.latent_dim)).reshape(
                cond_t.shape[0], x_q.shape[1], self.latent_dim
            )
        if self.trunk_film is not None:
            film = self.trunk_film(cond_t).reshape(cond_t.shape[0], 2, self.latent_dim)
            gamma = 1.0 + film[:, 0][:, None, :]
            beta = film[:, 1][:, None, :]
            t_lat = gamma * t_lat + beta
        out_dot = torch.einsum("bcl,bql->bqc", b_lat, t_lat)
        if self.fused_head is not None:
            bsz = int(cond_t.shape[0])
            n_q = int(x_q.shape[1])
            lat = int(self.latent_dim)
            out_dim = int(self.out_dim)
            chunk = 256
            out_fused = out_dot.new_zeros((bsz, n_q, out_dim))
            for start in range(0, n_q, chunk):
                stop = min(start + chunk, n_q)
                q_lat = t_lat[:, start:stop, :]
                q_len = int(stop - start)
                b_ctx = b_lat[:, None, :, :].expand(bsz, q_len, out_dim, lat)
                q_ctx = q_lat[:, :, None, :].expand(bsz, q_len, out_dim, lat)
                f_in = torch.cat([b_ctx, q_ctx, b_ctx * q_ctx], dim=3)
                pred = self.fused_head(f_in.reshape(-1, int(f_in.shape[-1]))).reshape(bsz, q_len, out_dim)
                out_fused[:, start:stop, :] = pred
            if self.output_path_dot_skip_logits is not None:
                dot_skip = torch.sigmoid(self.output_path_dot_skip_logits).reshape(1, 1, out_dim)
            else:
                dot_skip = out_dot.new_full((1, 1, out_dim), float(self.output_path_dot_skip))
            out = dot_skip * out_dot + out_fused
        else:
            out = out_dot
        if self.global_head is not None:
            out = out + self.global_head(cond_t)[:, None, :]
        out = out + self.out_bias[None, None, :]
        if self.residual_head is not None and self.residual_cond is not None:
            cond_lat = self.residual_cond(cond_t)[:, None, :].expand(-1, int(x_q.shape[1]), -1)
            r_in = torch.cat([t_lat, cond_lat], dim=2)
            delta = self.residual_head(r_in.reshape(-1, int(r_in.shape[-1]))).reshape(
                int(cond_t.shape[0]), int(x_q.shape[1]), int(self.out_dim)
            )
            if self.residual_head_gain_mode == "learned":
                gain = self.residual_scale[0] if self.residual_scale is not None else delta.new_tensor(1.0)
            else:
                gain = delta.new_tensor(float(self.residual_head_gain_value))
            out = out + gain * delta
        result: dict[str, Any] = {}
        for i, key in enumerate(self.output_keys):
            result[key] = out[:, :, i : i + 1]
        return result

    def predict_fields_torch(
        self,
        cond_t,
        geom_ctx: Any | None,
        spatial_features: Any | None = None,
    ) -> dict[str, Any]:
        torch = self._torch
        cond_t = cond_t.to(self.device)
        bsz = int(cond_t.shape[0])
        rows, spatial_channels = self._spatial_feature_tensor(
            spatial_features,
            batch_size=bsz,
            dtype=torch.float32,
            device=self.device,
        )
        if geom_ctx is not None:
            coord = torch.as_tensor(
                np.asarray(geom_ctx.coord_grid, dtype=np.float32).reshape(2, -1).T,
                dtype=torch.float32,
                device=self.device,
            )
            x_q = coord[None, ...].expand(bsz, -1, -1)
        else:
            if rows is None or not spatial_channels:
                raise ValueError(
                    "geom_ctx is required unless preprocessed DeepONet spatial features are available"
                )
            ch_pos_initial = {name: i for i, name in enumerate(spatial_channels)}
            missing_coord = [name for name in ("x", "y") if name not in ch_pos_initial]
            if missing_coord:
                raise ValueError(
                    "DeepONet spatial features must contain x/y when geom_ctx is absent; "
                    f"missing={missing_coord}"
                )
            x_q = rows[:, :, [ch_pos_initial["x"], ch_pos_initial["y"]]]
            coord = x_q[0]
        query_idx_np = np.arange(int(coord.shape[0]), dtype=np.int64)
        v_s = None
        q_f = None
        sensor_idx_np: np.ndarray | None = None
        x_s = None
        if rows is not None and spatial_channels:
            ch_pos = {name: i for i, name in enumerate(spatial_channels)}
            if self.trunk_fourier_mode == "symmetric" and "x" in ch_pos and "y" in ch_pos:
                xy = rows[:, :, [int(ch_pos["x"]), int(ch_pos["y"])]]
                xy_min = xy.amin(dim=1, keepdim=True)
                xy_max = xy.amax(dim=1, keepdim=True)
                xy_span = self._torch.clamp(xy_max - xy_min, min=1.0e-6)
                xy_norm = (xy - xy_min) / xy_span
                x_q = xy_norm
            sensor_pos = [ch_pos[name] for name in self.sensor_feature_names if name in ch_pos]
            missing_sensor_names = [name for name in self.sensor_feature_names if name not in ch_pos]
            query_pos = [ch_pos[name] for name in self.query_feature_names if name in ch_pos]
            missing_query_names = [name for name in self.query_feature_names if name not in ch_pos]
            if self.branch_mode != "cond_only":
                self._handle_missing_geom_features(
                    role="static_feature_rows.sensor",
                    missing_names=missing_sensor_names,
                )
                sensor_idx_np = np.asarray(self.sensor_indices, dtype=np.int64).reshape(-1)
                sensor_idx = torch.as_tensor(sensor_idx_np, dtype=torch.int64, device=coord.device)
                x_s = x_q[:, sensor_idx, :]
            if self.branch_mode != "cond_only" and sensor_idx_np is not None:
                if len(sensor_pos) != self.sensor_feature_dim:
                    raise ValueError(
                        "static_feature_rows.sensor feature dim mismatch: "
                        f"expected={self.sensor_feature_dim}, got={len(sensor_pos)}"
                    )
                sensor_idx = torch.as_tensor(sensor_idx_np, dtype=torch.int64, device=coord.device)
                sampled = rows[:, sensor_idx, :][:, :, sensor_pos]
                v_s = sampled
            self._handle_missing_geom_features(
                role="static_feature_rows.query",
                missing_names=missing_query_names,
            )
            if len(query_pos) != len(self.query_feature_names):
                raise ValueError(
                    "static_feature_rows.query feature dim mismatch: "
                    f"expected={len(self.query_feature_names)}, got={len(query_pos)}"
                )
            q_f = rows[:, :, query_pos]
        if self.branch_mode != "cond_only" and sensor_idx_np is None:
            sensor_idx_np = np.asarray(self.sensor_indices, dtype=np.int64).reshape(-1)
            sensor_idx = torch.as_tensor(sensor_idx_np, dtype=torch.int64, device=coord.device)
            x_s = x_q[:, sensor_idx, :]
        if v_s is None and self.branch_mode != "cond_only" and sensor_idx_np is not None:
            if geom_ctx is None:
                raise ValueError(
                    "DeepONet sensor values require geom_ctx when they are absent from spatial features"
                )
            v_s = self._sensor_values_from_geom(
                geom_ctx=geom_ctx,
                coord_flat=np.asarray(x_q[0].detach().cpu().numpy(), dtype=np.float32),
                sensor_indices=sensor_idx_np,
                bsz=bsz,
                dtype=coord.dtype,
                device=coord.device,
            )
        if q_f is None:
            if geom_ctx is None:
                raise ValueError(
                    "DeepONet query features require geom_ctx when they are absent from spatial features"
                )
            q_f = self._query_values_from_geom(
                geom_ctx=geom_ctx,
                coord_flat=np.asarray(x_q[0].detach().cpu().numpy(), dtype=np.float32),
                query_indices=query_idx_np,
                bsz=bsz,
                dtype=coord.dtype,
                device=coord.device,
            )
        pred = self.forward(sensors={"x": x_s, "v": v_s}, query={"x": x_q, "f": q_f}, cond=cond_t)
        h, w = self.grid_shape
        out = {}
        for key, val in pred.items():
            out[key] = val.permute(0, 2, 1).reshape(bsz, 1, h, w)
        return out

    def predict_fields(
        self,
        cond_vec: np.ndarray,
        geom_ctx: Any | None = None,
        cache_key: str = "poisson_head_v1",
        spatial_features: Any | None = None,
    ) -> dict[str, np.ndarray]:
        del cache_key
        torch = self._torch
        cond_t = torch.as_tensor(np.asarray(cond_vec, dtype=np.float32), dtype=torch.float32, device=self.device)
        if cond_t.ndim == 1:
            cond_t = cond_t[None, :]
        self.eval()
        with torch.no_grad():
            pred = self.predict_fields_torch(
                cond_t,
                geom_ctx=geom_ctx,
                spatial_features=spatial_features,
            )
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in pred.items()}

    def predict_phi(self, rho_eff, cond_vec, geom_ctx: Any, refine_iters: int = 0):
        if self._attached_poisson_head is None:
            raise RuntimeError("deeponet_plasma has no attached poisson_head")
        return self._attached_poisson_head.predict_phi(
            rho_eff=rho_eff,
            cond_vec=cond_vec,
            geom_ctx=geom_ctx,
            refine_iters=refine_iters,
        )

    def to_meta(self) -> dict[str, Any]:
        meta = {
            "model_type": self.model_type,
            "cond_dim": int(self.cond_dim),
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "flatten_order": self.flatten_order,
            "latent_dim": int(self.latent_dim),
            "hidden_dim": int(self.hidden_dim),
            "output_keys": list(self.output_keys),
            "sensor_indices": [int(v) for v in self.sensor_indices.tolist()],
            "query_indices": [int(v) for v in self.query_indices.tolist()],
            "sensor_feature_names": list(self.sensor_feature_names),
            "input_feature_channels": list(self.input_feature_channels),
            "sensor_feature_dim": int(self.sensor_feature_dim),
            "trunk_input_mode": str(self.trunk_input_mode),
            "trunk_fourier_n_freq": int(self.trunk_fourier_n_freq),
            "trunk_fourier_mode": str(self.trunk_fourier_mode),
            "trunk_cond_modulation": str(self.trunk_cond_modulation),
            "trunk_cond_mod_hidden": int(self.trunk_cond_mod_hidden),
            "residual_head_enabled": bool(self.residual_head_enabled),
            "residual_head_hidden_dim": int(self.residual_head_hidden_dim),
            "residual_head_scale_init": float(self.residual_head_scale_init),
            "residual_head_gain_mode": str(self.residual_head_gain_mode),
            "residual_head_gain_value": float(self.residual_head_gain_value),
            "output_path_mode": str(self.output_path_mode),
            "output_path_dot_skip": float(self.output_path_dot_skip),
            "output_path_dot_skip_mode": str(self.output_path_dot_skip_mode),
            "output_path_fused_hidden_dim": int(self.output_path_fused_hidden_dim),
            "output_path_global_local_enabled": bool(self.output_path_global_local_enabled),
            "output_path_global_hidden_dim": int(self.output_path_global_hidden_dim),
            "sensor_pool_mode": str(self.sensor_pool_mode),
            "branch_mode": str(self.branch_mode),
            "sensor_embed_dim": int(self.sensor_embed_dim),
            "query_feature_names": list(self.query_feature_names),
            "latent_layer_norm": bool(self.latent_layer_norm),
            "missing_geom_feature_policy": str(self.missing_geom_feature_policy),
        }
        if self._attached_poisson_head is not None:
            meta["poisson_head"] = self._attached_poisson_head.to_meta()
        if self._attached_boundary_operator is not None:
            meta["boundary_operator"] = self._attached_boundary_operator.to_meta()
        return meta

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k, v in self.branch.state_dict().items():
            out[f"branch::{k}"] = v.detach().cpu().numpy()
        if self.sensor_encoder is not None:
            for k, v in self.sensor_encoder.state_dict().items():
                out[f"sensor_encoder::{k}"] = v.detach().cpu().numpy()
        for k, v in self.trunk.state_dict().items():
            out[f"trunk::{k}"] = v.detach().cpu().numpy()
        if self.fused_head is not None:
            for k, v in self.fused_head.state_dict().items():
                out[f"fused_head::{k}"] = v.detach().cpu().numpy()
        if self.global_head is not None:
            for k, v in self.global_head.state_dict().items():
                out[f"global_head::{k}"] = v.detach().cpu().numpy()
        if self.trunk_film is not None:
            for k, v in self.trunk_film.state_dict().items():
                out[f"trunk_film::{k}"] = v.detach().cpu().numpy()
        if self.residual_cond is not None:
            for k, v in self.residual_cond.state_dict().items():
                out[f"residual_cond::{k}"] = v.detach().cpu().numpy()
        if self.residual_head is not None:
            for k, v in self.residual_head.state_dict().items():
                out[f"residual_head::{k}"] = v.detach().cpu().numpy()
        if self.residual_scale is not None:
            out["residual_scale"] = self.residual_scale.detach().cpu().numpy()
        if self.output_path_dot_skip_logits is not None:
            out["output_path_dot_skip_logits"] = self.output_path_dot_skip_logits.detach().cpu().numpy()
        out["out_bias"] = self.out_bias.detach().cpu().numpy()
        if self._attached_poisson_head is not None:
            for k, v in self._attached_poisson_head.state_dict_numpy().items():
                out[f"poisson_head::{k}"] = v
        if self._attached_boundary_operator is not None:
            for k, v in self._attached_boundary_operator.state_dict_numpy().items():
                out[f"boundary_operator::{k}"] = v
        return out

    def load_state_dict_numpy(self, weights: dict[str, np.ndarray]) -> None:
        torch = self._torch
        device = self.device
        branch_state = {
            k.split("branch::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("branch::")
        }
        sensor_state = {
            k.split("sensor_encoder::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("sensor_encoder::")
        }
        trunk_state = {
            k.split("trunk::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("trunk::")
        }
        fused_head_state = {
            k.split("fused_head::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("fused_head::")
        }
        global_head_state = {
            k.split("global_head::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("global_head::")
        }
        trunk_film_state = {
            k.split("trunk_film::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("trunk_film::")
        }
        residual_cond_state = {
            k.split("residual_cond::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("residual_cond::")
        }
        residual_head_state = {
            k.split("residual_head::", 1)[1]: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in weights.items()
            if k.startswith("residual_head::")
        }
        self.branch.load_state_dict(branch_state, strict=True)
        if self.sensor_encoder is not None and sensor_state:
            self.sensor_encoder.load_state_dict(sensor_state, strict=True)
        self.trunk.load_state_dict(trunk_state, strict=True)
        if self.fused_head is not None and fused_head_state:
            self.fused_head.load_state_dict(fused_head_state, strict=True)
        if self.global_head is not None and global_head_state:
            self.global_head.load_state_dict(global_head_state, strict=True)
        if self.trunk_film is not None and trunk_film_state:
            self.trunk_film.load_state_dict(trunk_film_state, strict=True)
        if self.residual_cond is not None and residual_cond_state:
            self.residual_cond.load_state_dict(residual_cond_state, strict=True)
        if self.residual_head is not None and residual_head_state:
            self.residual_head.load_state_dict(residual_head_state, strict=True)
        if self.residual_scale is not None and "residual_scale" in weights:
            self.residual_scale.data.copy_(torch.as_tensor(weights["residual_scale"], dtype=torch.float32, device=device))
        if self.output_path_dot_skip_logits is not None and "output_path_dot_skip_logits" in weights:
            self.output_path_dot_skip_logits.data.copy_(
                torch.as_tensor(weights["output_path_dot_skip_logits"], dtype=torch.float32, device=device)
            )
        if "out_bias" in weights:
            self.out_bias.data.copy_(torch.as_tensor(weights["out_bias"], dtype=torch.float32, device=device))
        self._move_modules_to_device()
