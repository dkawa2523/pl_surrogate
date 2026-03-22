"""Checkpoint I/O for cycle1/m3 baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.model_families import COORD_MLP_FAMILY_MODELS, UNETPP_FAMILY_MODELS
from plasma_surrogate.models.deeponet.pod_deeponet_torch import (
    PODBasisBundle,
    PODDeepONetTorch,
    POD_DEEPONET_IMPL_VERSION,
    normalize_pod_deeponet_model_cfg,
)
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch, _normalize_coord_mlp_model_cfg
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline


def _is_deeponet_plasma_torch_model(model: Any) -> bool:
    try:
        meta = model.to_meta()
    except Exception:  # pragma: no cover - defensive for non-deeponet models.
        return False
    return isinstance(meta, dict) and str(meta.get("model_type", "")) == "deeponet_plasma_torch"


def _build_unetpp_family_model(
    *,
    model_name: str,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    with_rho: bool,
    unet_feature_channels: list[str] | None,
    seed: int,
) -> UNetPPBaseline:
    conv_cfg = dict(cfg.get("conv_cfg", {}))
    if "base_channels" in cfg and "base_channels" not in conv_cfg:
        conv_cfg["base_channels"] = int(cfg.get("base_channels", 32))
    if "upsample_mode" in cfg and "upsample_mode" not in conv_cfg:
        conv_cfg["upsample_mode"] = str(cfg.get("upsample_mode"))
    if str(model_name).strip().lower() == "unetpp_attn":
        attention_cfg = dict(conv_cfg.get("attention_cfg", {}))
        attention_cfg["enabled"] = True
        attention_cfg.setdefault("reduction", 2)
        attention_cfg.setdefault("gate_activation", "sigmoid")
        conv_cfg["attention_cfg"] = attention_cfg
    output_heads_cfg = dict(cfg.get("output_heads", {}))
    return UNetPPBaseline(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        head_mlp=dict(cfg.get("head_mlp", {})),
        backend=str(cfg.get("backend", "torch")),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        conv_cfg=conv_cfg,
        output_heads=output_heads_cfg,
        seed=int(seed),
    )


def _build_coord_mlp_family_model(
    *,
    model_name: str,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    unet_feature_channels: list[str] | None,
    seed: int,
) -> CoordMLPTorch:
    model_key = str(model_name).strip().lower()
    if model_key not in COORD_MLP_FAMILY_MODELS:
        raise ValueError(f"Unsupported coord-MLP model family member: {model_name}")
    _, cfg_local = _normalize_coord_mlp_model_cfg(model_name=model_key, raw_cfg=dict(cfg))
    return CoordMLPTorch(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        input_feature_channels=list(
            unet_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
        ),
        model_cfg=cfg_local,
        seed=int(seed),
        backend="torch",
    )


def _build_pod_deeponet_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    seed: int,
    pod_basis_bundle: PODBasisBundle | None = None,
) -> PODDeepONetTorch:
    cfg_local = normalize_pod_deeponet_model_cfg(cfg, model_type="deeponet_pod")
    return PODDeepONetTorch(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        pod_basis_bundle=pod_basis_bundle,
        model_cfg=cfg_local,
        seed=int(seed),
        backend=str(dict(cfg).get("backend", "torch")),
    )


def build_model_from_name(
    *,
    model_name: str,
    input_dim: int,
    grid_shape: tuple[int, int],
    model_cfg: dict[str, Any] | None = None,
    seed: int = 0,
    phi_mode: str = "direct",
    out_channels: int = 3,
    output_keys: list[str] | None = None,
    coord_feature_dim: int = 2,
    unet_feature_channels: list[str] | None = None,
    pod_basis_bundle: PODBasisBundle | None = None,
) -> Any:
    """Factory for non-DeepONet model branches shared by train/benchmark."""

    cfg = dict(model_cfg or {})
    with_rho = bool(cfg.get("rho_eff_head", str(phi_mode) == "poisson_hybrid"))
    name = str(model_name)
    if name == "global_mlp":
        return GlobalMLP(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            seed=int(seed),
            hidden=list(cfg.get("hidden", [128, 128])),
            dropout=float(cfg.get("dropout", 0.1)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    if name == "unet":
        conv_cfg = dict(cfg.get("conv_cfg", {}))
        if "base_channels" in cfg and "base_channels" not in conv_cfg:
            conv_cfg["base_channels"] = int(cfg.get("base_channels", 32))
        if "upsample_mode" in cfg and "upsample_mode" not in conv_cfg:
            conv_cfg["upsample_mode"] = str(cfg.get("upsample_mode"))
        output_heads_cfg = dict(cfg.get("output_heads", {}))
        return UNetBaseline(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho_eff_head=with_rho,
            head_mlp=dict(cfg.get("head_mlp", {})),
            backend=str(cfg.get("backend", "numpy")),
            input_feature_channels=list(unet_feature_channels or ["x", "y"]),
            conv_cfg=conv_cfg,
            output_heads=output_heads_cfg,
            seed=int(seed),
        )
    if name in UNETPP_FAMILY_MODELS:
        return _build_unetpp_family_model(
            model_name=name,
            cfg=cfg,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho=with_rho,
            unet_feature_channels=unet_feature_channels,
            seed=int(seed),
        )
    if name in COORD_MLP_FAMILY_MODELS:
        return _build_coord_mlp_family_model(
            model_name=name,
            cfg=cfg,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            unet_feature_channels=unet_feature_channels,
            seed=int(seed),
        )
    if name == "deeponet_pod":
        return _build_pod_deeponet_model(
            cfg=cfg,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            seed=int(seed),
            pod_basis_bundle=pod_basis_bundle,
        )
    if name == "fno":
        return FNOBaseline(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho_eff_head=with_rho,
            n_modes=int(cfg.get("fno_n_modes", cfg.get("n_modes", 2))),
            head_mlp=dict(cfg.get("head_mlp", {})),
            input_feature_channels=list(unet_feature_channels or ["x", "y"]),
            spectral_cfg=dict(cfg.get("spectral_cfg", {})),
            seed=int(seed),
            backend=str(cfg.get("backend", "torch")),
        )
    if name == "ffno":
        return FFNOBaseline(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho_eff_head=with_rho,
            n_modes=int(cfg.get("fno_n_modes", cfg.get("n_modes", 2))),
            head_mlp=dict(cfg.get("head_mlp", {})),
            input_feature_channels=list(unet_feature_channels or ["x", "y"]),
            spectral_cfg=dict(cfg.get("spectral_cfg", {})),
            seed=int(seed),
            backend=str(cfg.get("backend", "torch")),
        )
    raise ValueError(f"Unsupported model.name: {model_name}")


def resolve_deeponet_stages(
    deeponet_cfg: dict[str, Any],
    *,
    default_epochs: int,
    default_lr: float,
) -> list[dict[str, Any]]:
    cfg = dict(deeponet_cfg or {})
    has_s1 = "stage1" in cfg
    has_s2 = "stage2" in cfg
    s1 = dict(cfg.get("stage1", {}))
    s2 = dict(cfg.get("stage2", {}))

    # Plain mainline default: keep training contract single-stage unless stage blocks are explicit.
    if not has_s1 and not has_s2:
        return [
            {
                "name": "stage1",
                "epochs": int(max(1, int(default_epochs))),
                "lr": float(default_lr),
                "freeze_poisson_head": False,
                "freeze_boundary_operator": False,
                "refine_iters": 0,
            }
        ]

    stages: list[dict[str, Any]] = [
        {
            "name": "stage1",
            "epochs": int(s1.get("epochs", max(1, int(default_epochs)))),
            "lr": float(s1.get("lr", float(default_lr))),
            "freeze_poisson_head": bool(s1.get("freeze_poisson_head", True)),
            "freeze_boundary_operator": bool(s1.get("freeze_boundary_operator", True)),
            "refine_iters": int(s1.get("refine_iters", 0)),
        }
    ]
    if has_s2:
        stages.append(
            {
                "name": "stage2",
                "epochs": int(s2.get("epochs", max(1, int(default_epochs)))),
                "lr": float(s2.get("lr", max(float(default_lr) * 0.5, 1e-4))),
                "freeze_poisson_head": bool(s2.get("freeze_poisson_head", False)),
                "freeze_boundary_operator": bool(s2.get("freeze_boundary_operator", False)),
                "refine_iters": int(s2.get("refine_iters", 1)),
            }
        )
    return stages


def save_mlp_checkpoint(model: Any, ckpt_dir: str | Path) -> Path:
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, GlobalMLP):
        meta = {
            "model_type": "global_mlp",
            "input_dim": model.input_dim,
            "grid_shape": list(model.grid_shape),
            "out_channels": model.out_channels,
            "output_keys": list(getattr(model, "output_keys", ["log_ne", "Te", "phi"])),
            "hidden": list(getattr(model, "hidden", [])),
            "dropout": float(getattr(model, "dropout", 0.0)),
            "weight_decay": float(getattr(model, "weight_decay", 0.0)),
            "arch_version": str(getattr(model, "arch_version", "linear_v1")),
        }
    elif isinstance(model, UNetBaseline):
        meta = {
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
    elif isinstance(model, UNetPPBaseline):
        model_type = str(getattr(model, "model_type", "unetpp")).strip().lower()
        if model_type not in {"unetpp", "unetpp_attn"}:
            raise TypeError(f"Unsupported UNetPPBaseline model_type for checkpoint: {model_type}")
        meta = {
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
    elif isinstance(model, CoordMLPTorch):
        meta = {
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
    elif isinstance(model, PODDeepONetTorch):
        basis_bundle = model.basis_bundle_numpy()
        meta = {
            "model_type": "deeponet_pod",
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
    elif isinstance(model, FNOBaseline):
        meta = {
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
    elif isinstance(model, FFNOBaseline):
        meta = {
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
            "fno_impl_version": str(
                getattr(model, "fno_impl_version", "factorized_separable_1d_v1")
            ),
            "head_arch_version": str(getattr(model, "head_arch_version", "linear_v1")),
        }
    elif _is_deeponet_plasma_torch_model(model):
        meta = model.to_meta()
    else:
        raise TypeError(f"Unsupported model type for checkpoint: {type(model)}")

    if _is_deeponet_plasma_torch_model(model):
        np.savez_compressed(ckpt_path / "weights.npz", **model.state_dict_numpy())
    elif hasattr(model, "state_dict_numpy"):
        np.savez_compressed(ckpt_path / "weights.npz", **model.state_dict_numpy())
    else:
        np.savez_compressed(ckpt_path / "weights.npz", W=model.W, b=model.b)
    with (ckpt_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return ckpt_path


def load_mlp_checkpoint(ckpt_dir: str | Path) -> Any:
    ckpt_path = Path(ckpt_dir)
    with (ckpt_path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    weights = np.load(ckpt_path / "weights.npz")
    model_type = meta.get("model_type")

    if model_type == "global_mlp":
        model = GlobalMLP(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", 3)),
            output_keys=list(meta.get("output_keys", ["log_ne", "Te", "phi"])),
            hidden=list(meta.get("hidden", [128, 128])),
            dropout=float(meta.get("dropout", 0.1)),
            weight_decay=float(meta.get("weight_decay", 0.0)),
        )
    elif model_type == "unet":
        model = UNetBaseline(
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
    elif model_type in {"unetpp", "unetpp_attn"}:
        conv_cfg = dict(meta.get("conv_cfg", {}))
        if model_type == "unetpp_attn":
            attention_cfg = dict(conv_cfg.get("attention_cfg", {}))
            attention_cfg["enabled"] = True
            attention_cfg.setdefault("reduction", 2)
            attention_cfg.setdefault("gate_activation", "sigmoid")
            conv_cfg["attention_cfg"] = attention_cfg
        model = UNetPPBaseline(
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
    elif model_type == "fno":
        backend = str(meta.get("backend", "numpy")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy FNO checkpoints are no longer supported")
        impl = str(meta.get("fno_impl_version", "")).strip().lower()
        if impl not in {"spectral_v2"}:
            raise ValueError("legacy FNO checkpoint format is not supported")
        model = FNOBaseline(
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
    elif model_type == "ffno":
        backend = str(meta.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("legacy numpy FFNO checkpoints are no longer supported")
        impl = str(meta.get("fno_impl_version", "")).strip().lower()
        if impl not in {"factorized_separable_1d_v1", "factorized_separable_1d_v2_local_skip"}:
            raise ValueError("legacy FFNO checkpoint format is not supported")
        model = FFNOBaseline(
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
    elif model_type in {"coord_mlp_fourier", "coord_mlp_siren"}:
        impl_version = str(meta.get("coord_mlp_impl_version", "")).strip().lower()
        if impl_version != "v4_siren_branch_balanced":
            raise ValueError(
                "legacy coord_mlp checkpoint format is not supported; "
                "expected coord_mlp_impl_version=v4_siren_branch_balanced; "
                "retrain or re-export checkpoint with current code"
            )
        _, model_cfg = _normalize_coord_mlp_model_cfg(
            model_name=str(model_type),
            raw_cfg=dict(meta.get("model_cfg", {})),
        )
        model = CoordMLPTorch(
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
    elif model_type == "deeponet_pod":
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
        missing_std_src = [name for name in basis_keys if str(name) not in set(str(k) for k in coeff_std_by_var.keys())]
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
        model = PODDeepONetTorch(
            input_dim=int(meta["input_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            out_channels=int(meta.get("out_channels", len(meta.get("output_keys", [])))),
            output_keys=list(meta.get("output_keys", [])),
            pod_basis_bundle=basis_bundle,
            model_cfg=normalize_pod_deeponet_model_cfg(
                dict(meta.get("model_cfg", {})),
                model_type="deeponet_pod",
            ),
            backend=str(meta.get("backend", "torch")),
        )
    elif model_type in {"hybrid_unet_fno", "hybrid_unet_fno_residual"}:
        raise ValueError(f"{model_type} checkpoints are no longer supported")
    elif model_type == "deeponet_plasma_torch":
        from plasma_surrogate.models.deeponet.boundary_operator_torch import BoundaryOperatorTorch
        from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
        from plasma_surrogate.models.deeponet.poisson_head_torch import DeepONetPoissonHeadTorch

        model = DeepONetPlasmaOperatorTorch(
            cond_dim=int(meta["cond_dim"]),
            grid_shape=tuple(meta["grid_shape"]),
            output_keys=list(meta.get("output_keys", ["ne", "ni", "Te", "phi"])),
            latent_dim=int(meta.get("latent_dim", 32)),
            hidden_dim=int(meta.get("hidden_dim", 64)),
            sensor_indices=np.asarray(meta.get("sensor_indices", []), dtype=np.int64),
            query_indices=np.asarray(meta.get("query_indices", []), dtype=np.int64),
            flatten_order=str(meta.get("flatten_order", "C")),
            sensor_feature_names=list(meta.get("sensor_feature_names", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])),
            trunk_input_mode=str(meta.get("trunk_input_mode", "legacy_xy_fourier")),
            sensor_pool_mode=str(meta.get("sensor_pool_mode", "moments")),
            sensor_embed_dim=int(meta.get("sensor_embed_dim", 32)),
            branch_mode=str(meta.get("branch_mode", "moments")),
            trunk_fourier_n_freq=int(meta.get("trunk_fourier_n_freq", 1)),
            trunk_fourier_mode=str(meta.get("trunk_fourier_mode", "legacy")),
            trunk_cond_modulation=str(meta.get("trunk_cond_modulation", "none")),
            trunk_cond_mod_hidden=int(meta.get("trunk_cond_mod_hidden", 64)),
            residual_head_enabled=bool(meta.get("residual_head_enabled", False)),
            residual_head_hidden_dim=int(meta.get("residual_head_hidden_dim", 64)),
            residual_head_scale_init=float(meta.get("residual_head_scale_init", 0.0)),
            residual_head_gain_mode=str(meta.get("residual_head_gain_mode", "learned")),
            residual_head_gain_value=float(meta.get("residual_head_gain_value", 1.0)),
            latent_layer_norm=bool(meta.get("latent_layer_norm", False)),
            output_path_mode=str(meta.get("output_path_mode", "dot")),
            output_path_dot_skip=float(meta.get("output_path_dot_skip", 0.25)),
            output_path_fused_hidden_dim=int(meta.get("output_path_fused_hidden_dim", 96)),
            output_path_global_local_enabled=bool(meta.get("output_path_global_local_enabled", False)),
            output_path_global_hidden_dim=int(meta.get("output_path_global_hidden_dim", 64)),
        )
        ph_meta = meta.get("poisson_head")
        if isinstance(ph_meta, dict):
            ph_net = DeepONetPlasmaOperatorTorch(
                cond_dim=int(meta["cond_dim"]),
                grid_shape=tuple(meta["grid_shape"]),
                output_keys=["phi"],
                latent_dim=int(meta.get("latent_dim", 32)),
                hidden_dim=int(meta.get("hidden_dim", 64)),
                sensor_indices=np.asarray(ph_meta.get("sensor_indices", meta.get("sensor_indices", [])), dtype=np.int64),
                query_indices=np.asarray(ph_meta.get("query_indices", meta.get("query_indices", [])), dtype=np.int64),
                flatten_order=str(ph_meta.get("flatten_order", meta.get("flatten_order", "C"))),
                sensor_feature_names=list(ph_meta.get("sensor_feature_names", meta.get("sensor_feature_names", ["x", "y", "mask_plasma", "distance_signed", "distance_any"]))),
                trunk_input_mode=str(ph_meta.get("trunk_input_mode", meta.get("trunk_input_mode", "legacy_xy_fourier"))),
                sensor_pool_mode=str(ph_meta.get("sensor_pool_mode", meta.get("sensor_pool_mode", "moments"))),
                sensor_embed_dim=int(ph_meta.get("sensor_embed_dim", meta.get("sensor_embed_dim", 32))),
                branch_mode=str(ph_meta.get("branch_mode", meta.get("branch_mode", "moments"))),
                trunk_fourier_n_freq=int(ph_meta.get("trunk_fourier_n_freq", meta.get("trunk_fourier_n_freq", 1))),
                trunk_fourier_mode=str(ph_meta.get("trunk_fourier_mode", meta.get("trunk_fourier_mode", "legacy"))),
                trunk_cond_modulation=str(ph_meta.get("trunk_cond_modulation", meta.get("trunk_cond_modulation", "none"))),
                trunk_cond_mod_hidden=int(ph_meta.get("trunk_cond_mod_hidden", meta.get("trunk_cond_mod_hidden", 64))),
                residual_head_enabled=bool(ph_meta.get("residual_head_enabled", meta.get("residual_head_enabled", False))),
                residual_head_hidden_dim=int(ph_meta.get("residual_head_hidden_dim", meta.get("residual_head_hidden_dim", 64))),
                residual_head_scale_init=float(ph_meta.get("residual_head_scale_init", meta.get("residual_head_scale_init", 0.0))),
                residual_head_gain_mode=str(
                    ph_meta.get("residual_head_gain_mode", meta.get("residual_head_gain_mode", "learned"))
                ),
                residual_head_gain_value=float(
                    ph_meta.get("residual_head_gain_value", meta.get("residual_head_gain_value", 1.0))
                ),
                latent_layer_norm=bool(ph_meta.get("latent_layer_norm", meta.get("latent_layer_norm", False))),
                output_path_mode=str(ph_meta.get("output_path_mode", meta.get("output_path_mode", "dot"))),
                output_path_dot_skip=float(ph_meta.get("output_path_dot_skip", meta.get("output_path_dot_skip", 0.25))),
                output_path_fused_hidden_dim=int(
                    ph_meta.get("output_path_fused_hidden_dim", meta.get("output_path_fused_hidden_dim", 96))
                ),
                output_path_global_local_enabled=bool(
                    ph_meta.get(
                        "output_path_global_local_enabled",
                        meta.get("output_path_global_local_enabled", False),
                    )
                ),
                output_path_global_hidden_dim=int(
                    ph_meta.get(
                        "output_path_global_hidden_dim",
                        meta.get("output_path_global_hidden_dim", 64),
                    )
                ),
            )
            ph = DeepONetPoissonHeadTorch.from_cache(
                deeponet_poisson=ph_net,
                sensor_idx=np.asarray(ph_meta.get("sensor_indices", []), dtype=np.int64),
                query_idx=np.asarray(ph_meta.get("query_indices", []), dtype=np.int64),
                flatten_order=str(ph_meta.get("flatten_order", "C")),
                grid_shape=tuple(ph_meta.get("grid_shape", meta["grid_shape"])),
                freeze=bool(ph_meta.get("freeze", True)),
            )
            model.attach_poisson_head(ph)
        bo_meta = meta.get("boundary_operator")
        if isinstance(bo_meta, dict):
            bo = BoundaryOperatorTorch(
                primary_qoi_key=str(bo_meta.get("primary_qoi_key", "Gamma_i")),
                w_log_ne=float(bo_meta.get("w_log_ne", 0.08)),
                w_te=float(bo_meta.get("w_te", 0.06)),
                w_en=float(bo_meta.get("w_en", 0.04)),
                bias=float(bo_meta.get("bias", 0.0)),
                clamp=tuple(bo_meta["clamp"]) if bo_meta.get("clamp") is not None else None,
                freeze=bool(bo_meta.get("freeze", True)),
            )
            model.attach_boundary_operator(bo)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    if model_type == "deeponet_plasma_torch":
        model.load_state_dict_numpy({k: np.asarray(weights[k], dtype=np.float32) for k in weights.files})
        ph = getattr(model, "poisson_head", None)
        if ph is not None:
            ph_weights = {
                k.split("poisson_head::", 1)[1]: np.asarray(weights[k], dtype=np.float32)
                for k in weights.files
                if k.startswith("poisson_head::")
            }
            if ph_weights:
                ph.load_state_dict_numpy(ph_weights)
        bo = getattr(model, "boundary_operator", None)
        if bo is not None:
            bo_weights = {
                k.split("boundary_operator::", 1)[1]: np.asarray(weights[k], dtype=np.float32)
                for k in weights.files
                if k.startswith("boundary_operator::")
            }
            if bo_weights:
                bo.load_state_dict_numpy(bo_weights)
    elif model_type in {"global_mlp"} and hasattr(model, "load_state_dict_numpy"):
        if "W" in weights and "b" in weights:
            legacy_w = np.asarray(weights["W"], dtype=np.float32)
            legacy_b = np.asarray(weights["b"], dtype=np.float32)
            if legacy_w.ndim != 2 or legacy_b.ndim != 1:
                raise ValueError(f"legacy checkpoint has invalid shapes: W={legacy_w.shape} b={legacy_b.shape}")
            model.weights = [legacy_w]
            model.biases = [legacy_b]
            if hasattr(model, "hidden"):
                model.hidden = []
            if hasattr(model, "arch_version"):
                model.arch_version = "linear_v1_legacy_loaded"
        else:
            state = {k: np.asarray(weights[k], dtype=np.float32) for k in weights.files}
            model.load_state_dict_numpy(state)
    elif model_type in {"unet", "unetpp", "unetpp_attn", "fno", "ffno", "coord_mlp_fourier", "coord_mlp_siren", "deeponet_pod"} and hasattr(
        model, "load_state_dict_numpy"
    ):
        state = {k: np.asarray(weights[k], dtype=np.float32) for k in weights.files}
        model.load_state_dict_numpy(state)
    else:
        model.W = np.asarray(weights["W"], dtype=np.float32)
        model.b = np.asarray(weights["b"], dtype=np.float32)
    return model
