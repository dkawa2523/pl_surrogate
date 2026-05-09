"""Model factory shared by training and benchmark orchestration."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.model_families import (
    CNO_FAMILY_MODELS,
    COORD_MLP_FAMILY_MODELS,
    GEOM_DEEPONET_SIREN_FAMILY_MODELS,
    POD_DEEPONET_FAMILY_MODELS,
    UNETPP_FAMILY_MODELS,
    UNO_FAMILY_MODELS,
)
from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet, normalize_cno_operator_unet_cfg
from plasma_surrogate.models.cno.simple_cno import CNOBaseline, normalize_cno_cfg
from plasma_surrogate.models.deeponet.geom_deeponet_siren import (
    GeomDeepONetSIREN,
    normalize_geom_deeponet_siren_cfg,
)
from plasma_surrogate.models.deeponet.pod_deeponet_torch import (
    PODBasisBundle,
    PODDeepONetTorch,
    normalize_pod_deeponet_model_cfg,
)
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import (
    CoordMLPPODResidual,
    normalize_coord_mlp_pod_residual_cfg,
)
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch, _normalize_coord_mlp_model_cfg
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.unet.operator_v2 import UNetOperatorV2, normalize_unet_operator_v2_cfg
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline, normalize_uno_cfg

__all__ = ["build_model_from_name"]


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
        output_heads=dict(cfg.get("output_heads", {})),
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
    pod_basis_bundle: PODBasisBundle | None = None,
) -> Any:
    model_key = str(model_name).strip().lower()
    if model_key not in COORD_MLP_FAMILY_MODELS:
        raise ValueError(f"Unsupported coord-MLP model family member: {model_name}")
    if model_key == "coord_mlp_pod_residual":
        basis_bundle = pod_basis_bundle
        if basis_bundle is None:
            raise ValueError("coord_mlp_pod_residual requires a train-split POD basis bundle")
        return CoordMLPPODResidual(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            input_feature_channels=list(
                unet_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
            ),
            pod_basis_bundle=basis_bundle,
            model_cfg=normalize_coord_mlp_pod_residual_cfg(dict(cfg)),
            seed=int(seed),
            backend=str(dict(cfg).get("backend", "torch")),
        )
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


def _build_uno_family_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    with_rho: bool,
    unet_feature_channels: list[str] | None,
    seed: int,
) -> UNOBaseline:
    backend = str(cfg.get("backend", "torch")).strip().lower()
    if backend != "torch":
        raise ValueError("train.u_no.model_cfg.backend must be torch")
    return UNOBaseline(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        n_modes=int(cfg.get("fno_n_modes", cfg.get("n_modes", 12))),
        head_mlp=dict(cfg.get("head_mlp", {})),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        uno_cfg=normalize_uno_cfg(dict(cfg.get("uno_cfg", {}))),
        seed=int(seed),
        backend=backend,
    )


def _build_cno_family_model(
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
) -> Any:
    backend = str(cfg.get("backend", "torch")).strip().lower()
    if backend != "torch":
        raise ValueError(f"train.{model_name}.model_cfg.backend must be torch")
    if str(model_name).strip().lower() == "cno_operator_unet":
        return CNOOperatorUNet(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho_eff_head=with_rho,
            head_mlp=dict(cfg.get("head_mlp", {})),
            input_feature_channels=list(unet_feature_channels or ["x", "y"]),
            cno_operator_unet_cfg=normalize_cno_operator_unet_cfg(
                dict(cfg.get("cno_operator_unet_cfg", {}))
            ),
            seed=int(seed),
            backend=backend,
        )
    return CNOBaseline(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        head_mlp=dict(cfg.get("head_mlp", {})),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        cno_cfg=normalize_cno_cfg(dict(cfg.get("cno_cfg", {}))),
        seed=int(seed),
        backend=backend,
    )


def _build_geom_deeponet_siren_family_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    with_rho: bool,
    unet_feature_channels: list[str] | None,
    seed: int,
) -> GeomDeepONetSIREN:
    backend = str(cfg.get("backend", "torch")).strip().lower()
    if backend != "torch":
        raise ValueError("train.geom_deeponet_siren.model_cfg.backend must be torch")
    return GeomDeepONetSIREN(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        geom_deeponet_siren_cfg=normalize_geom_deeponet_siren_cfg(
            dict(cfg.get("geom_deeponet_siren_cfg", {}))
        ),
        seed=int(seed),
        backend=backend,
    )


def _build_pod_deeponet_model(
    *,
    model_name: str,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    seed: int,
    pod_basis_bundle: PODBasisBundle | None = None,
) -> PODDeepONetTorch:
    model_type = str(model_name).strip().lower()
    cfg_local = normalize_pod_deeponet_model_cfg(cfg, model_type=model_type)
    return PODDeepONetTorch(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        pod_basis_bundle=pod_basis_bundle,
        model_cfg=cfg_local,
        seed=int(seed),
        backend=str(dict(cfg).get("backend", "torch")),
        model_type=model_type,
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
    """Build a model instance from a public model name."""

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
            output_heads=dict(cfg.get("output_heads", {})),
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
    if name == "unet_operator_v2":
        backend = str(cfg.get("backend", "torch")).strip().lower()
        if backend != "torch":
            raise ValueError("train.unet_operator_v2.model_cfg.backend must be torch")
        return UNetOperatorV2(
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho_eff_head=with_rho,
            head_mlp=dict(cfg.get("head_mlp", {})),
            input_feature_channels=list(
                unet_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
            ),
            unet_operator_v2_cfg=normalize_unet_operator_v2_cfg(
                dict(cfg.get("unet_operator_v2_cfg", {}))
            ),
            seed=int(seed),
            backend=backend,
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
            pod_basis_bundle=pod_basis_bundle,
        )
    if name in UNO_FAMILY_MODELS:
        return _build_uno_family_model(
            cfg=cfg,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho=with_rho,
            unet_feature_channels=unet_feature_channels,
            seed=int(seed),
        )
    if name in CNO_FAMILY_MODELS:
        return _build_cno_family_model(
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
    if name in GEOM_DEEPONET_SIREN_FAMILY_MODELS:
        return _build_geom_deeponet_siren_family_model(
            cfg=cfg,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            with_rho=with_rho,
            unet_feature_channels=unet_feature_channels,
            seed=int(seed),
        )
    if name in POD_DEEPONET_FAMILY_MODELS:
        return _build_pod_deeponet_model(
            model_name=name,
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
