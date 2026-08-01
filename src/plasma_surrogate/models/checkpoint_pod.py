"""Checkpoint serializer for POD-DeepONet models."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.deeponet.pod_deeponet_torch import (
    PODBasisBundle,
    PODDeepONetTorch,
    POD_DEEPONET_IMPL_VERSION,
    normalize_pod_deeponet_model_cfg,
)

POD_DEEPONET_CHECKPOINT_MODEL_TYPES = frozenset(
    {"deeponet_pod", "deeponet_plasma_pod", "geom_deeponet_pod"}
)

__all__ = [
    "POD_DEEPONET_CHECKPOINT_MODEL_TYPES",
    "load_pod_deeponet_checkpoint_model",
    "make_pod_deeponet_checkpoint_meta",
]


def make_pod_deeponet_checkpoint_meta(model: Any) -> dict[str, Any] | None:
    if not isinstance(model, PODDeepONetTorch):
        return None
    basis_bundle = model.basis_bundle_numpy()
    model_type = str(getattr(model, "model_type", "deeponet_pod")).strip().lower() or "deeponet_pod"
    return {
        "model_type": model_type,
        "impl_version": str(getattr(model, "pod_impl_version", POD_DEEPONET_IMPL_VERSION)),
        "input_dim": model.input_dim,
        "grid_shape": list(model.grid_shape),
        "out_channels": model.out_channels,
        "output_keys": list(getattr(model, "output_keys", [])),
        "backend": str(getattr(model, "backend", "torch")),
        "model_cfg": dict(getattr(model, "model_cfg", {})),
        "basis_keys": list(getattr(model, "basis_keys", [])),
        "basis_rank_by_var": dict(getattr(model, "basis_rank_by_var", {})),
        "coeff_std_by_var": {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1).tolist()
            for k, v in basis_bundle.coeff_std_by_var.items()
        },
    }


def load_pod_deeponet_checkpoint_model(meta: dict[str, Any], weights: Any) -> PODDeepONetTorch | None:
    model_type = str(meta.get("model_type", "")).strip().lower()
    if model_type not in POD_DEEPONET_CHECKPOINT_MODEL_TYPES:
        return None

    impl_version = str(meta.get("impl_version", "")).strip().lower()
    if impl_version != str(POD_DEEPONET_IMPL_VERSION).strip().lower():
        raise ValueError(
            "legacy deeponet_pod checkpoint format is not supported; "
            f"expected impl_version={POD_DEEPONET_IMPL_VERSION}; "
            "retrain or re-export checkpoint with current code"
        )

    basis_keys = [str(v) for v in list(meta.get("basis_keys", meta.get("output_keys", [])))]
    coeff_std_meta = dict(meta.get("coeff_std_by_var", {}))
    coeff_std_state = {
        str(name): np.asarray(weights[f"coeff_std::{name}"], dtype=np.float32)
        for name in basis_keys
        if f"coeff_std::{name}" in weights
    }
    coeff_std_by_var = coeff_std_state if coeff_std_state else coeff_std_meta
    missing_std_src = [name for name in basis_keys if str(name) not in set(str(k) for k in coeff_std_by_var)]
    if missing_std_src:
        raise ValueError(
            "legacy deeponet_pod checkpoint format is not supported; "
            f"missing coeff_std for vars={missing_std_src}"
        )

    basis_bundle = PODBasisBundle.from_dicts(
        basis_by_var={
            str(name): np.asarray(weights[f"basis::{name}"], dtype=np.float32)
            for name in basis_keys
            if f"basis::{name}" in weights
        },
        mean_by_var={
            str(name): np.asarray(weights[f"mean::{name}"], dtype=np.float32)
            for name in basis_keys
            if f"mean::{name}" in weights
        },
        rank_by_var=dict(meta.get("basis_rank_by_var", {})),
        coeff_std_by_var={
            str(name): np.asarray(coeff_std_by_var[str(name)], dtype=np.float32)
            for name in basis_keys
            if str(name) in coeff_std_by_var
        },
    )
    return PODDeepONetTorch(
        input_dim=int(meta["input_dim"]),
        grid_shape=tuple(meta["grid_shape"]),
        out_channels=int(meta.get("out_channels", len(meta.get("output_keys", [])))),
        output_keys=list(meta.get("output_keys", [])),
        pod_basis_bundle=basis_bundle,
        model_cfg=normalize_pod_deeponet_model_cfg(
            dict(meta.get("model_cfg", {})),
            model_type=model_type,
        ),
        backend=str(meta.get("backend", "torch")),
        model_type=model_type,
    )
