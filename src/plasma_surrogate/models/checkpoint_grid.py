"""Checkpoint serializers for grid and coordinate model families."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet, normalize_cno_operator_unet_cfg
from plasma_surrogate.models.cno.simple_cno import CNOBaseline, normalize_cno_cfg
from plasma_surrogate.models.deeponet.geom_deeponet_siren import (
    GeomDeepONetSIREN,
    normalize_geom_deeponet_siren_cfg,
)
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODBasisBundle
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import (
    COORD_MLP_POD_RESIDUAL_IMPL_VERSION,
    CoordMLPPODResidual,
    normalize_coord_mlp_pod_residual_cfg,
)
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch, _normalize_coord_mlp_model_cfg
from plasma_surrogate.models.unet.operator_v2 import UNetOperatorV2, normalize_unet_operator_v2_cfg
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline, normalize_uno_cfg


GRID_CHECKPOINT_MODEL_TYPES = frozenset(
    {
        "unet",
        "unetpp",
        "unetpp_attn",
        "unet_operator_v2",
        "fno",
        "ffno",
        "u_no",
        "cno",
        "cno_operator_unet",
        "geom_deeponet_siren",
        "coord_mlp_fourier",
        "coord_mlp_siren",
        "coord_mlp_pod_residual",
    }
)

def make_grid_checkpoint_meta(model: Any) -> dict[str, Any] | None:
    """Return checkpoint metadata for a grid family model, or None if unsupported."""

    if isinstance(model, UNetBaseline):
        return {
            "model_type": "unet",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["log_ne", "Te", "phi"])),
            "backend": str(getattr(model, "backend", "numpy")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "head_mlp": {
                "enabled": bool(getattr(model, "head_enabled", False)),
                "hidden": list(getattr(model, "head_hidden", [])),
                "activation": str(getattr(model, "head_activation", "tanh")),
                "dropout": float(getattr(model, "head_dropout", 0.0)),
            },
            "conv_cfg": {
                "base_channels": int(getattr(model, "_torch_base_channels", 32)),
                "depth": int(getattr(model, "_torch_depth", 1)),
                "upsample_mode": str(getattr(model, "_torch_upsample_mode", "deconv")),
            },
            "output_heads": {
                "mode": str(getattr(model, "output_heads_mode", "shared")),
            },
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }

    if isinstance(model, UNetPPBaseline):
        model_type = str(getattr(model, "model_type", "unetpp")).strip().lower()
        if model_type not in {"unetpp", "unetpp_attn"}:
            raise TypeError(f"Unsupported UNetPPBaseline model_type for checkpoint: {model_type}")
        return {
            "model_type": model_type,
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["log_ne", "Te", "phi"])),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "head_mlp": {
                "enabled": bool(getattr(model, "head_enabled", True)),
                "hidden": list(getattr(model, "head_hidden", [])),
                "activation": str(getattr(model, "head_activation", "relu")),
                "dropout": float(getattr(model, "head_dropout", 0.0)),
            },
            "conv_cfg": dict(
                getattr(
                    model,
                    "_torch_conv_cfg",
                    {
                        "base_channels": int(getattr(model, "_torch_base_channels", 32)),
                        "depth": int(getattr(model, "_torch_depth", 2)),
                        "upsample_mode": str(getattr(model, "_torch_upsample_mode", "bilinear")),
                        "nested_skip": bool(getattr(model, "_torch_nested_skip", True)),
                        "deep_supervision": {"enabled": False},
                    },
                )
            ),
            "output_heads": {
                "mode": str(getattr(model, "output_heads_mode", "shared")),
            },
            "head_arch_version": str(getattr(model, "head_arch_version", "conv_unetpp_v1")),
        }

    if isinstance(model, UNetOperatorV2):
        return {
            "model_type": "unet_operator_v2",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(
                getattr(model, "input_feature_channels", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
            ),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "unet_operator_v2_cfg": dict(getattr(model, "unet_operator_v2_cfg", {})),
            "unet_operator_v2_impl_version": str(
                getattr(model, "fno_impl_version", "unet_operator_v2_v1")
            ),
            "head_arch_version": str(getattr(model, "head_arch_version", "unet_operator_v2_v1")),
        }

    if isinstance(model, FNOBaseline):
        return {
            "model_type": "fno",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "n_modes": int(getattr(model, "n_modes", 2)),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "spectral_cfg": dict(getattr(model, "spectral_cfg", {})),
            "fno_impl_version": str(getattr(model, "fno_impl_version", "spectral_v2")),
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }

    if isinstance(model, FFNOBaseline):
        return {
            "model_type": "ffno",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "n_modes": int(getattr(model, "n_modes", 2)),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "spectral_cfg": dict(getattr(model, "spectral_cfg", {})),
            "fno_impl_version": str(getattr(model, "fno_impl_version", "factorized_separable_1d_v1")),
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }

    if isinstance(model, UNOBaseline):
        return {
            "model_type": "u_no",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "n_modes": int(getattr(model, "n_modes", 12)),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "uno_cfg": dict(getattr(model, "uno_cfg", {})),
            "uno_impl_version": str(getattr(model, "fno_impl_version", "uno_lite_v1")),
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }

    if isinstance(model, CNOBaseline):
        return {
            "model_type": "cno",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "cno_cfg": dict(getattr(model, "cno_cfg", {})),
            "cno_impl_version": str(getattr(model, "fno_impl_version", "cno_lite_v1")),
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }

    if isinstance(model, CNOOperatorUNet):
        return {
            "model_type": "cno_operator_unet",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "head_mlp": dict(getattr(model, "head_mlp_cfg", {})),
            "cno_operator_unet_cfg": dict(getattr(model, "cno_operator_unet_cfg", {})),
            "cno_operator_unet_impl_version": str(
                getattr(model, "fno_impl_version", "cno_operator_unet_v1")
            ),
            "head_arch_version": str(getattr(model, "head_arch_version", "operator_unet_v1")),
        }

    if isinstance(model, GeomDeepONetSIREN):
        return {
            "model_type": "geom_deeponet_siren",
            "input_dim": int(model.input_dim),
            "grid_shape": list(model.grid_shape),
            "out_channels": int(model.out_channels),
            "output_keys": list(getattr(model, "output_keys", ["ne", "ni", "Te", "phi"])),
            "with_rho_eff_head": bool(getattr(model, "with_rho_eff_head", False)),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", ["x", "y"])),
            "geom_deeponet_siren_cfg": dict(getattr(model, "geom_deeponet_siren_cfg", {})),
            "geom_deeponet_siren_impl_version": str(
                getattr(model, "geom_deeponet_siren_impl_version", "geom_deeponet_siren_v1")
            ),
        }

    if isinstance(model, CoordMLPTorch):
        return {
            "model_type": str(getattr(model, "model_type", "coord_mlp_fourier")),
            "coord_mlp_impl_version": str(getattr(model, "coord_mlp_impl_version", "v4_siren_branch_balanced")),
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", [])),
            "backend": str(getattr(model, "backend", "torch")),
            "input_feature_channels": list(getattr(model, "input_feature_channels", [])),
            "model_cfg": dict(getattr(model, "model_cfg", {})),
        }

    if isinstance(model, CoordMLPPODResidual):
        return dict(model.to_meta())

    return None


def _pod_basis_bundle_from_grid_weights(meta: dict[str, Any], weights: Any) -> PODBasisBundle:
    basis_keys = [str(v) for v in list(meta.get("basis_keys", meta.get("output_keys", [])))]
    return PODBasisBundle.from_dicts(
        basis_by_var={
            name: np.asarray(weights[f"basis::{name}"], dtype=np.float32)
            for name in basis_keys
            if weights is not None and f"basis::{name}" in weights
        },
        mean_by_var={
            name: np.asarray(weights[f"mean::{name}"], dtype=np.float32)
            for name in basis_keys
            if weights is not None and f"mean::{name}" in weights
        },
        rank_by_var=dict(meta.get("basis_rank_by_var", {})),
        coeff_std_by_var={
            name: np.asarray(weights[f"coeff_std::{name}"], dtype=np.float32)
            for name in basis_keys
            if weights is not None and f"coeff_std::{name}" in weights
        },
    )


def load_grid_checkpoint_model(meta: dict[str, Any], weights: Any | None = None) -> Any | None:
    """Build a grid family model from checkpoint metadata, or None if unsupported."""

    model_type = str(meta.get("model_type", "")).strip().lower()
    if model_type not in GRID_CHECKPOINT_MODEL_TYPES:
        return None

    if model_type == "unet":
        return UNetBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["log_ne", "Te", "phi"])),
            backend=str(meta.get("backend", "numpy")),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            head_mlp=dict(meta.get("head_mlp", {})),
            conv_cfg=dict(meta.get("conv_cfg", {})),
            output_heads=dict(meta.get("output_heads", {})),
        )

    if model_type in {"unetpp", "unetpp_attn"}:
        conv_cfg = dict(meta.get("conv_cfg", {}))
        if model_type == "unetpp_attn":
            attention_cfg = dict(conv_cfg.get("attention_cfg", {}))
            attention_cfg["enabled"] = True
            attention_cfg.setdefault("reduction", 2)
            attention_cfg.setdefault("gate_activation", "sigmoid")
            conv_cfg["attention_cfg"] = attention_cfg
        return UNetPPBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["log_ne", "Te", "phi"])),
            backend=str(meta.get("backend", "torch")),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            head_mlp=dict(meta.get("head_mlp", {})),
            conv_cfg=conv_cfg,
            output_heads=dict(meta.get("output_heads", {})),
        )

    if model_type == "unet_operator_v2":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy UNet operator v2 checkpoints are no longer supported")
        impl = str(meta.get("unet_operator_v2_impl_version", "")).strip().lower()
        if impl not in {"unet_operator_v2_v1"}:
            raise ValueError("legacy UNet operator v2 checkpoint format is not supported")
        return UNetOperatorV2(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(
                meta.get("input_feature_channels", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
            ),
            unet_operator_v2_cfg=normalize_unet_operator_v2_cfg(
                dict(meta.get("unet_operator_v2_cfg", {}))
            ),
            backend=backend,
        )

    if model_type == "fno":
        backend = str(meta.get("backend", "numpy")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy FNO checkpoints are no longer supported")
        impl = str(meta.get("fno_impl_version", "")).strip().lower()
        if impl not in {"spectral_v2"}:
            raise ValueError("legacy FNO checkpoint format is not supported")
        return FNOBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            n_modes=int(meta.get("n_modes", 2)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            spectral_cfg=dict(meta.get("spectral_cfg", {})),
            backend=backend,
        )

    if model_type == "ffno":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy FFNO checkpoints are no longer supported")
        impl = str(meta.get("fno_impl_version", "")).strip().lower()
        if impl not in {"factorized_separable_1d_v1", "factorized_separable_1d_v2_local_skip"}:
            raise ValueError("legacy FFNO checkpoint format is not supported")
        return FFNOBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            n_modes=int(meta.get("n_modes", 2)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            spectral_cfg=dict(meta.get("spectral_cfg", {})),
            backend=backend,
        )

    if model_type == "u_no":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy U-NO checkpoints are no longer supported")
        impl = str(meta.get("uno_impl_version", "")).strip().lower()
        if impl not in {"uno_lite_v1"}:
            raise ValueError("legacy U-NO checkpoint format is not supported")
        return UNOBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            n_modes=int(meta.get("n_modes", 12)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            uno_cfg=normalize_uno_cfg(dict(meta.get("uno_cfg", {}))),
            backend=backend,
        )

    if model_type == "cno":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy CNO checkpoints are no longer supported")
        impl = str(meta.get("cno_impl_version", "")).strip().lower()
        if impl not in {"cno_lite_v1"}:
            raise ValueError("legacy CNO checkpoint format is not supported")
        return CNOBaseline(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            cno_cfg=normalize_cno_cfg(dict(meta.get("cno_cfg", {}))),
            backend=backend,
        )

    if model_type == "cno_operator_unet":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy CNO operator U-Net checkpoints are no longer supported")
        impl = str(meta.get("cno_operator_unet_impl_version", "")).strip().lower()
        if impl not in {"cno_operator_unet_v1"}:
            raise ValueError("legacy CNO operator U-Net checkpoint format is not supported")
        return CNOOperatorUNet(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            head_mlp=dict(meta.get("head_mlp", {})),
            input_feature_channels=list(meta.get("input_feature_channels", ["x", "y"])),
            cno_operator_unet_cfg=normalize_cno_operator_unet_cfg(
                dict(meta.get("cno_operator_unet_cfg", {}))
            ),
            backend=backend,
        )

    if model_type == "geom_deeponet_siren":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy geom_deeponet_siren checkpoints are no longer supported")
        impl = str(meta.get("geom_deeponet_siren_impl_version", "")).strip().lower()
        if impl not in {"geom_deeponet_siren_v1"}:
            raise ValueError("legacy geom_deeponet_siren checkpoint format is not supported")
        return GeomDeepONetSIREN(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            with_rho_eff_head=bool(meta.get("with_rho_eff_head", False)),
            input_feature_channels=list(
                meta.get("input_feature_channels", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
            ),
            geom_deeponet_siren_cfg=normalize_geom_deeponet_siren_cfg(
                dict(meta.get("geom_deeponet_siren_cfg", {}))
            ),
            backend=backend,
        )

    if model_type == "coord_mlp_pod_residual":
        impl_version = str(meta.get("coord_mlp_impl_version", "")).strip().lower()
        if impl_version != COORD_MLP_POD_RESIDUAL_IMPL_VERSION:
            raise ValueError(
                "legacy coord_mlp_pod_residual checkpoint format is not supported; "
                f"expected coord_mlp_impl_version={COORD_MLP_POD_RESIDUAL_IMPL_VERSION}"
            )
        if weights is None:
            raise ValueError("coord_mlp_pod_residual checkpoint loading requires weights.npz")
        basis_bundle = _pod_basis_bundle_from_grid_weights(meta, weights)
        return CoordMLPPODResidual(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", len(meta.get("output_keys", [])))),
            output_keys=list(meta.get("output_keys", [])),
            input_feature_channels=list(
                meta.get("input_feature_channels", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
            ),
            pod_basis_bundle=basis_bundle,
            model_cfg=normalize_coord_mlp_pod_residual_cfg(dict(meta.get("model_cfg", {}))),
            backend=str(meta.get("backend", "torch")),
        )

    impl_version = str(meta.get("coord_mlp_impl_version", "")).strip().lower()
    if impl_version != "v4_siren_branch_balanced":
        raise ValueError(
            "legacy coord_mlp checkpoint format is not supported; "
            "expected coord_mlp_impl_version=v4_siren_branch_balanced; "
            "retrain or re-export checkpoint with current code"
        )
    _, model_cfg = _normalize_coord_mlp_model_cfg(
        model_name=model_type,
        raw_cfg=dict(meta.get("model_cfg", {})),
    )
    return CoordMLPTorch(
        input_dim=int(meta["input_dim"]),
        grid_shape=tuple(meta["grid_shape"]),
        out_channels=int(meta.get("out_channels", 3)),
        output_keys=list(meta.get("output_keys", [])),
        input_feature_channels=list(
            meta.get("input_feature_channels", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
        ),
        model_cfg=model_cfg,
        backend=str(meta.get("backend", "torch")),
    )
