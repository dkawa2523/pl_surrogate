"""Global-MLP baseline with lightweight multilayer backbone."""

from __future__ import annotations

from typing import Any

import numpy as np


class GlobalMLP:
    """Condition-to-field model with optional hidden layers."""

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        hidden: list[int] | None = None,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
    ):
        self.input_dim = int(input_dim)
        self.grid_shape = tuple(grid_shape)
        self.out_channels = int(out_channels)
        if output_keys is None:
            base = ["log_ne", "Te", "phi"]
            extra = [f"out_{i}" for i in range(max(0, self.out_channels - len(base)))]
            self.output_keys = (base + extra)[: self.out_channels]
        else:
            self.output_keys = list(output_keys)[: self.out_channels]
        self.output_dim = self.out_channels * self.grid_shape[0] * self.grid_shape[1]
        self.hidden = list(hidden if hidden is not None else [128, 128])
        self.dropout = float(np.clip(float(dropout), 0.0, 0.9))
        self.weight_decay = float(max(weight_decay, 0.0))
        self.arch_version = "mlp_v2"

        dims = [self.input_dim, *self.hidden, self.output_dim]
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(fan_in, 1))
            self.weights.append((rng.normal(size=(fan_in, fan_out)) * scale).astype(np.float32))
            self.biases.append(np.zeros((fan_out,), dtype=np.float32))
        self._cache: dict[str, Any] = {}

    def _forward_internal(self, cond: np.ndarray, *, training: bool) -> np.ndarray:
        x = np.asarray(cond, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        acts = [x]
        zs: list[np.ndarray] = []
        drops: list[np.ndarray] = []
        a = x
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            zs.append(z)
            if i < n_layers - 1:
                a = np.tanh(z).astype(np.float32)
                if training and self.dropout > 0.0:
                    keep = 1.0 - self.dropout
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
        return a

    def forward(self, cond: np.ndarray, training: bool = False) -> np.ndarray:
        return self._forward_internal(cond, training=bool(training))

    def backward(
        self,
        grad_output: np.ndarray,
        *,
        lr: float,
        weight_decay: float | None = None,
        layer_lr_multipliers: dict[str, float] | None = None,
    ) -> dict[str, float]:
        grad = np.asarray(grad_output, dtype=np.float32)
        cache = self._cache
        if not cache:
            raise RuntimeError("GlobalMLP.backward called without forward cache")
        acts: list[np.ndarray] = cache["acts"]
        zs: list[np.ndarray] = cache["zs"]
        drops: list[np.ndarray] = cache["drops"]
        wd = self.weight_decay if weight_decay is None else float(max(weight_decay, 0.0))
        mult_cfg = dict(layer_lr_multipliers or {})
        hidden_mult = float(mult_cfg.get("hidden", 1.0))
        output_mult = float(mult_cfg.get("output", 1.0))
        output_step_rel = 0.0
        hidden_step_rels: list[float] = []
        n_layers = len(self.weights)
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = acts[i]
            w_prev = self.weights[i].copy()
            grad_w = a_prev.T @ grad
            grad_b = np.sum(grad, axis=0)
            if wd > 0.0:
                grad_w = grad_w + wd * w_prev
            lr_mult = output_mult if i == n_layers - 1 else hidden_mult
            lr_eff = float(lr) * float(lr_mult)
            step_rel = lr_eff * float(np.linalg.norm(grad_w)) / max(float(np.linalg.norm(w_prev)), 1e-12)
            if i == n_layers - 1:
                output_step_rel = float(step_rel)
            else:
                hidden_step_rels.append(float(step_rel))
            self.weights[i] = self.weights[i] - lr_eff * grad_w.astype(np.float32)
            self.biases[i] = self.biases[i] - lr_eff * grad_b.astype(np.float32)
            if i > 0:
                # Backprop must use pre-update weights for this layer.
                grad = grad @ w_prev.T
                tanh_z = np.tanh(zs[i - 1]).astype(np.float32)
                grad = grad * (1.0 - tanh_z * tanh_z)
                grad = grad * drops[i - 1]
        hidden_mean = float(np.mean(hidden_step_rels)) if hidden_step_rels else 0.0
        return {
            "step_rel_hidden_mean": hidden_mean,
            "step_rel_output": float(output_step_rel),
        }

    def predict_fields(self, cond: np.ndarray) -> dict[str, np.ndarray]:
        y = self.forward(cond, training=False)
        bsz = y.shape[0]
        h, w = self.grid_shape
        y = y.reshape(bsz, self.out_channels, h, w)
        return {name: y[:, i : i + 1] for i, name in enumerate(self.output_keys)}

    def extract_hidden(self, cond: np.ndarray) -> np.ndarray:
        """Return the final hidden activations (or input if no hidden layers)."""

        self._forward_internal(cond, training=False)
        acts = self._cache.get("acts", [])
        if len(acts) < 2:
            raise RuntimeError("GlobalMLP forward cache is empty")
        if len(self.weights) <= 1:
            return np.asarray(acts[0], dtype=np.float32)
        return np.asarray(acts[-2], dtype=np.float32)

    def fit_output_layer_ridge(
        self,
        hidden: np.ndarray,
        target_flat: np.ndarray,
        *,
        ridge: float = 1e-4,
    ) -> float:
        """
        Refit only the output layer with ridge-regularized least squares.

        Returns:
            Condition number of the augmented design matrix.
        """

        hmat = np.asarray(hidden, dtype=np.float32)
        ymat = np.asarray(target_flat, dtype=np.float32)
        if hmat.ndim != 2:
            raise ValueError(f"hidden must be 2D [N,H], got {hmat.shape}")
        if ymat.ndim != 2:
            raise ValueError(f"target_flat must be 2D [N,D], got {ymat.shape}")
        if hmat.shape[0] != ymat.shape[0]:
            raise ValueError(f"batch size mismatch: hidden={hmat.shape[0]}, target={ymat.shape[0]}")
        if hmat.shape[1] != self.weights[-1].shape[0]:
            raise ValueError(
                f"hidden dim mismatch: hidden={hmat.shape[1]}, expected={self.weights[-1].shape[0]}"
            )
        if ymat.shape[1] != self.weights[-1].shape[1]:
            raise ValueError(
                f"target dim mismatch: target={ymat.shape[1]}, expected={self.weights[-1].shape[1]}"
            )

        x = np.concatenate([hmat, np.ones((hmat.shape[0], 1), dtype=np.float32)], axis=1)
        lam = float(max(ridge, 0.0))
        xtx = x.T @ x
        if lam > 0.0:
            reg = np.eye(xtx.shape[0], dtype=np.float32)
            reg[-1, -1] = 0.0  # do not penalize bias term
            xtx = xtx + lam * reg
        xty = x.T @ ymat
        beta = np.linalg.solve(xtx, xty).astype(np.float32)
        self.weights[-1] = beta[:-1]
        self.biases[-1] = beta[-1]

        s = np.linalg.svd(x, compute_uv=False)
        cond = float(s[0] / max(float(s[-1]), 1e-12))
        return cond

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out[f"layer{i}.W"] = np.asarray(w, dtype=np.float32)
            out[f"layer{i}.b"] = np.asarray(b, dtype=np.float32)
        return out

    def load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None:
        loaded_w: list[np.ndarray] = []
        loaded_b: list[np.ndarray] = []
        i = 0
        while f"layer{i}.W" in state and f"layer{i}.b" in state:
            loaded_w.append(np.asarray(state[f"layer{i}.W"], dtype=np.float32))
            loaded_b.append(np.asarray(state[f"layer{i}.b"], dtype=np.float32))
            i += 1
        if len(loaded_w) == 0:
            raise ValueError("GlobalMLP state dict does not contain layer*.W/layer*.b")
        self.weights = loaded_w
        self.biases = loaded_b
