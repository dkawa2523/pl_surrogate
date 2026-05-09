"""Shared private helpers for torch models with explicit spatial feature maps."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _build_unit_coord_grid(grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = tuple(grid_shape)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yv, xv = np.meshgrid(yy, xx, indexing="ij")
    return np.stack([xv, yv], axis=-1).astype(np.float32)


def _validate_static_spatial_features(
    spatial_features: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    spatial_feature_dim: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(spatial_features, dtype=np.float32)
    h, w = tuple(grid_shape)
    if arr.ndim != 3:
        raise ValueError(f"{label} spatial features must be [H,W,C], got {arr.shape}")
    if tuple(arr.shape[:2]) != (h, w):
        raise ValueError(f"{label} spatial features shape mismatch: expected {(h, w, arr.shape[2])}, got {arr.shape}")
    if int(arr.shape[2]) != int(spatial_feature_dim):
        raise ValueError(
            f"{label} spatial feature channels mismatch: expected {int(spatial_feature_dim)}, got {int(arr.shape[2])}"
        )
    return arr.astype(np.float32)


def _resolve_batched_spatial_features(
    cond: np.ndarray,
    spatial_features: np.ndarray | None,
    *,
    static_spatial_features: np.ndarray | None,
    grid_shape: tuple[int, int],
    spatial_feature_dim: int,
    label: str,
    coord_grid: np.ndarray | None = None,
    allow_coord_fallback: bool = False,
    explicit_requirement_message: str | None = None,
) -> np.ndarray:
    x = np.asarray(cond, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    bsz = int(x.shape[0])
    h, w = tuple(grid_shape)
    src = spatial_features if spatial_features is not None else static_spatial_features
    if src is None:
        if allow_coord_fallback and coord_grid is not None:
            return np.repeat(np.asarray(coord_grid, dtype=np.float32)[None, ...], bsz, axis=0).astype(np.float32)
        if explicit_requirement_message:
            raise ValueError(str(explicit_requirement_message))
        raise ValueError(f"{label} input_features requires explicit spatial features")

    arr = np.asarray(src, dtype=np.float32)
    if arr.ndim == 3:
        if tuple(arr.shape[:2]) != (h, w):
            raise ValueError(f"{label} spatial features shape mismatch: expected {(h, w)}, got {arr.shape[:2]}")
        if int(arr.shape[2]) != int(spatial_feature_dim):
            raise ValueError(
                f"{label} spatial feature channels mismatch: expected {int(spatial_feature_dim)}, got {int(arr.shape[2])}"
            )
        return np.repeat(arr[None, ...], bsz, axis=0).astype(np.float32)
    if arr.ndim == 4:
        if tuple(arr.shape[1:3]) != (h, w):
            raise ValueError(f"{label} spatial features shape mismatch: expected {(h, w)}, got {arr.shape[1:3]}")
        if int(arr.shape[3]) != int(spatial_feature_dim):
            raise ValueError(
                f"{label} spatial feature channels mismatch: expected {int(spatial_feature_dim)}, got {int(arr.shape[3])}"
            )
        if int(arr.shape[0]) == bsz:
            return arr.astype(np.float32)
        if int(arr.shape[0]) == 1:
            return np.repeat(arr, bsz, axis=0).astype(np.float32)
        raise ValueError(f"{label} spatial features batch mismatch: cond batch={bsz}, features batch={arr.shape[0]}")
    raise ValueError(f"{label} spatial features must be [H,W,C] or [B,H,W,C], got {arr.shape}")


def _state_dict_numpy_torch(net: Any) -> dict[str, np.ndarray]:
    return {
        f"torch::{name}": tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in net.state_dict().items()
    }


def _resolve_torch_device(torch: Any) -> Any:
    return torch.device("cuda" if bool(torch.cuda.is_available()) else "cpu")


def _load_state_dict_numpy_torch(
    state: dict[str, np.ndarray],
    *,
    torch: Any,
    net: Any,
    empty_message: str,
    device: Any | None = None,
) -> None:
    state_t = {}
    for k, v in state.items():
        if not str(k).startswith("torch::"):
            continue
        tensor = torch.from_numpy(np.asarray(v, dtype=np.float32))
        if device is not None:
            tensor = tensor.to(device)
        state_t[k.split("torch::", 1)[1]] = tensor
    if not state_t:
        raise ValueError(empty_message)
    net.load_state_dict(state_t, strict=True)


def _backward_raw_torch_step(
    *,
    torch: Any,
    net: Any,
    grad_raw: np.ndarray,
    last_out: Any,
    lr: float,
    step_reference: Callable[[], tuple[Any, Any]],
    apply_step: bool = True,
) -> dict[str, float]:
    if last_out is None:
        raise RuntimeError("torch backward called without forward cache")
    grad_np = np.asarray(grad_raw, dtype=np.float32)
    grad_t = torch.as_tensor(
        grad_np,
        dtype=getattr(last_out, "dtype", None) or torch.float32,
        device=getattr(last_out, "device", None),
    )
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}
    hidden_w, out_w = step_reference()
    with torch.no_grad():
        w_hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
        w_out_prev = out_w.detach().clone() if out_w is not None else None
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    last_out.backward(grad_t)
    step_hidden = 0.0
    step_out = 0.0
    if bool(apply_step):
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= float(lr) * p.grad
            hidden_cur, out_cur = step_reference()
            if hidden_cur is not None and w_hidden_prev is not None:
                dh = hidden_cur.detach() - w_hidden_prev
                step_hidden = float(torch.linalg.norm(dh) / max(float(torch.linalg.norm(w_hidden_prev)), 1.0e-12))
            if out_cur is not None and w_out_prev is not None:
                do = out_cur.detach() - w_out_prev
                step_out = float(torch.linalg.norm(do) / max(float(torch.linalg.norm(w_out_prev)), 1.0e-12))
    return {"step_rel_hidden_mean": float(step_hidden), "step_rel_output": float(step_out)}


__all__ = [
    "_backward_raw_torch_step",
    "_build_unit_coord_grid",
    "_load_state_dict_numpy_torch",
    "_resolve_batched_spatial_features",
    "_resolve_torch_device",
    "_state_dict_numpy_torch",
    "_validate_static_spatial_features",
]
