"""Model factory shared by training and benchmark orchestration."""

from __future__ import annotations

from typing import Any, Callable

from plasma_surrogate.core.model_specs import get_model_spec
from plasma_surrogate.core.model_families import COORD_MLP_FAMILY_MODELS
from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet, normalize_cno_operator_unet_cfg
from plasma_surrogate.models.cno.simple_cno import CNOBaseline, normalize_cno_cfg
from plasma_surrogate.models.deeponet.geom_deeponet_siren import (
    GeomDeepONetSIREN,
    normalize_geom_deeponet_siren_cfg,
)
from plasma_surrogate.models.deeponet.plasma_operator_torch import DeepONetPlasmaOperatorTorch
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
from plasma_surrogate.models.mlp.global_vector_mlp import (
    GLOBAL_VECTOR_MLP_MODEL_TYPES,
    GlobalVectorMLP,
)
from plasma_surrogate.models.heads.role_grouped import (
    ROLE_GROUPED_OUTPUT_HEAD_MODELS,
    is_grouped_output_head_mode,
)
from plasma_surrogate.models.unet.operator_v2 import UNetOperatorV2, normalize_unet_operator_v2_cfg
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline, normalize_uno_cfg

__all__ = ["build_model_from_name"]

_MODEL_BUILDER = Callable[..., Any]


def _build_global_mlp_family_model(
    *,
    model_name: str,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    seed: int,
    **_: Any,
) -> GlobalMLP | GlobalVectorMLP:
    model_key = str(model_name).strip().lower()
    if model_key in GLOBAL_VECTOR_MLP_MODEL_TYPES:
        return GlobalVectorMLP(
            model_type=model_key,
            input_dim=int(input_dim),
            grid_shape=tuple(grid_shape),
            out_channels=int(out_channels),
            output_keys=output_keys,
            model_cfg=cfg,
            seed=int(seed),
        )
    if model_key != "global_mlp":
        raise ValueError(f"Unsupported global-MLP family member: {model_name}")
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


def _build_unet_family_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    with_rho: bool,
    unet_feature_channels: list[str] | None,
    seed: int,
    **_: Any,
) -> UNetBaseline:
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
        backend=str(cfg.get("backend", "torch")),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        conv_cfg=conv_cfg,
        output_heads=dict(cfg.get("output_heads", {})),
        target_role_schema=dict(cfg.get("target_role_schema", {})),
        seed=int(seed),
    )


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
    **_: Any,
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
        target_role_schema=dict(cfg.get("target_role_schema", {})),
        seed=int(seed),
    )


def _build_unet_operator_family_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    with_rho: bool,
    unet_feature_channels: list[str] | None,
    seed: int,
    **_: Any,
) -> UNetOperatorV2:
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
        unet_operator_v2_cfg=normalize_unet_operator_v2_cfg(dict(cfg.get("unet_operator_v2_cfg", {}))),
        seed=int(seed),
        backend=backend,
    )


def _build_spectral_family_model(
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
    **_: Any,
) -> FNOBaseline | FFNOBaseline:
    model_key = str(model_name).strip().lower()
    model_cls: type[FNOBaseline] | type[FFNOBaseline]
    if model_key == "fno":
        model_cls = FNOBaseline
    elif model_key == "ffno":
        model_cls = FFNOBaseline
    else:
        raise ValueError(f"Unsupported spectral model family member: {model_name}")
    return model_cls(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        n_modes=int(cfg.get("fno_n_modes", cfg.get("n_modes", 2))),
        head_mlp=dict(cfg.get("head_mlp", {})),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        spectral_cfg=dict(cfg.get("spectral_cfg", {})),
        output_heads=dict(cfg.get("output_heads", {})),
        target_role_schema=dict(cfg.get("target_role_schema", {})),
        operator_response_cfg=dict(cfg.get("operator_response_cfg", {})),
        seed=int(seed),
        backend=str(cfg.get("backend", "torch")),
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
    **_: Any,
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
    **_: Any,
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
        output_heads=dict(cfg.get("output_heads", {})),
        target_role_schema=dict(cfg.get("target_role_schema", {})),
        operator_response_cfg=dict(cfg.get("operator_response_cfg", {})),
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
    **_: Any,
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
        output_heads=dict(cfg.get("output_heads", {})),
        target_role_schema=dict(cfg.get("target_role_schema", {})),
        operator_response_cfg=dict(cfg.get("operator_response_cfg", {})),
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
    **_: Any,
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


def _build_deeponet_plasma_family_model(
    *,
    cfg: dict[str, Any],
    input_dim: int,
    grid_shape: tuple[int, int],
    out_channels: int,
    output_keys: list[str] | None,
    unet_feature_channels: list[str] | None,
    seed: int,
    **_: Any,
) -> DeepONetPlasmaOperatorTorch:
    model_cfg = dict(cfg.get("model_cfg", cfg))
    residual_head_cfg = dict(model_cfg.get("residual_head", {}))
    output_path_cfg = dict(model_cfg.get("output_path", {}))
    keys = list(output_keys or [f"target_{idx}" for idx in range(int(out_channels))])
    return DeepONetPlasmaOperatorTorch(
        cond_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        output_keys=keys,
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        sensor_feature_names=list(
            unet_feature_channels or ["x", "y", "mask_plasma", "distance_signed", "distance_any"]
        ),
        trunk_input_mode=str(model_cfg.get("trunk_input_mode", "geom_feature_pack")),
        sensor_pool_mode=str(model_cfg.get("sensor_pool_mode", "moments")),
        sensor_embed_dim=int(model_cfg.get("sensor_embed_dim", 32)),
        branch_mode=str(model_cfg.get("branch_mode", "moments")),
        trunk_fourier_n_freq=int(model_cfg.get("trunk_fourier_n_freq", 1)),
        trunk_fourier_mode=str(model_cfg.get("trunk_fourier_mode", "symmetric")),
        trunk_cond_modulation=str(model_cfg.get("trunk_cond_modulation", "none")),
        trunk_cond_mod_hidden=int(model_cfg.get("trunk_cond_mod_hidden", 64)),
        residual_head_enabled=bool(residual_head_cfg.get("enabled", False)),
        residual_head_hidden_dim=int(residual_head_cfg.get("hidden_dim", 64)),
        residual_head_scale_init=float(residual_head_cfg.get("scale_init", 0.0)),
        residual_head_gain_mode=str(residual_head_cfg.get("gain_mode", "learned")),
        residual_head_gain_value=float(residual_head_cfg.get("gain_value", 1.0)),
        latent_layer_norm=bool(model_cfg.get("latent_layer_norm", False)),
        output_path_mode=str(output_path_cfg.get("mode", "dot")),
        output_path_dot_skip=float(output_path_cfg.get("dot_skip", 0.25)),
        output_path_dot_skip_mode=str(output_path_cfg.get("dot_skip_mode", "fixed")),
        output_path_fused_hidden_dim=int(output_path_cfg.get("fused_hidden_dim", 96)),
        output_path_global_local_enabled=bool(output_path_cfg.get("global_local", {}).get("enabled", False)),
        output_path_global_hidden_dim=int(output_path_cfg.get("global_hidden_dim", 64)),
        missing_geom_feature_policy=str(model_cfg.get("missing_geom_feature_policy", "error")),
        seed=int(seed),
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
    pod_descriptor_normalization_stats: dict[str, Any] | None = None,
    **_: Any,
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
        descriptor_normalization_stats=pod_descriptor_normalization_stats,
        seed=int(seed),
        backend=str(dict(cfg).get("backend", "torch")),
        model_type=model_type,
    )


_FAMILY_BUILDERS: dict[str, _MODEL_BUILDER] = {
    "global_mlp": _build_global_mlp_family_model,
    "unet": _build_unet_family_model,
    "unetpp": _build_unetpp_family_model,
    "unet_operator": _build_unet_operator_family_model,
    "spectral": _build_spectral_family_model,
    "coord_mlp": _build_coord_mlp_family_model,
    "uno": _build_uno_family_model,
    "cno": _build_cno_family_model,
    "geom_deeponet_siren": _build_geom_deeponet_siren_family_model,
    "deeponet_plasma": _build_deeponet_plasma_family_model,
    "pod_deeponet": _build_pod_deeponet_model,
}


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
    pod_descriptor_normalization_stats: dict[str, Any] | None = None,
) -> Any:
    """Build a model instance from a public model name."""

    cfg = dict(model_cfg or {})
    with_rho = bool(cfg.get("rho_eff_head", str(phi_mode) == "poisson_hybrid"))
    spec = get_model_spec(model_name)
    output_heads_mode = str(dict(cfg.get("output_heads", {}) or {}).get("mode", "shared")).strip().lower()
    if is_grouped_output_head_mode(output_heads_mode) and spec.name not in ROLE_GROUPED_OUTPUT_HEAD_MODELS:
        raise ValueError(
            f"output_heads.mode={output_heads_mode} is supported only for "
            f"{sorted(ROLE_GROUPED_OUTPUT_HEAD_MODELS)}; got model={spec.name!r}"
        )
    builder = _FAMILY_BUILDERS.get(spec.family)
    if builder is None:
        raise ValueError(f"Unsupported model family: model={spec.name}, family={spec.family}")
    return builder(
        model_name=spec.name,
        cfg=cfg,
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        coord_feature_dim=int(coord_feature_dim),
        with_rho=with_rho,
        unet_feature_channels=unet_feature_channels,
        seed=int(seed),
        pod_basis_bundle=pod_basis_bundle,
        pod_descriptor_normalization_stats=pod_descriptor_normalization_stats,
    )
