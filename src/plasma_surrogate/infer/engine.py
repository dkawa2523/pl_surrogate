"""Inference engine for cycle1 MLP baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY,
    INPUT_MODE_FALLBACK_APPLIED_KEY,
    INPUT_MODE_EFFECTIVE_KEY,
    INPUT_MODES,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
    normalize_strict_input_mode,
    resolve_input_mode_metadata_contract,
    input_mode_metadata_keys,
)
from plasma_surrogate.core.model_input_policy import (
    ADAPTER_DESCRIPTOR_BRANCH,
    ADAPTER_HYBRID_PACK_DESCRIPTOR,
    ADAPTER_NONE,
    validate_model_input_mode,
)
from plasma_surrogate.core.model_families import POD_DEEPONET_FAMILY_MODELS
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.data.geometry_provider import GeometryProviderLike
from plasma_surrogate.eval.metrics import bc_mae, boundary_band_mask, boundary_gamma_proxy, poisson_residual_norm, uniformity
from plasma_surrogate.features.structure_descriptors import build_structure_descriptor
from plasma_surrogate.features.structure_feature_registry import validate_coord_feature_channels
from plasma_surrogate.infer.optimize import OptimizeRunner, validate_optimize_geom_contract
from plasma_surrogate.models.cno.operator_unet import CNOOperatorUNet
from plasma_surrogate.models.cno.simple_cno import CNOBaseline
from plasma_surrogate.models.fno.factorized_fno import FFNOBaseline
from plasma_surrogate.models.fno.simple_fno import FNOBaseline
from plasma_surrogate.models.uno.simple_uno import UNOBaseline
from plasma_surrogate.models.deeponet.geom_deeponet_siren import GeomDeepONetSIREN
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODDeepONetTorch
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP
from plasma_surrogate.models.mlp.coord_mlp_torch import CoordMLPTorch
from plasma_surrogate.models.heads.plasma_head import PlasmaHead
from plasma_surrogate.models.unet.simple_unet import UNetBaseline
from plasma_surrogate.models.unet.unetpp import UNetPPBaseline
from plasma_surrogate.preprocessing.scalers import ScalerFactory, TransformBundle
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema
from plasma_surrogate.train.losses import boundary_operator_loss, boundary_operator_target, poisson_residual

@dataclass
class InferenceResult:
    fields_model: dict[str, np.ndarray]
    fields_phys: dict[str, np.ndarray]
    derived: dict[str, np.ndarray]
    qoi: dict[str, float]
    diagnostics: dict[str, Any]
    warnings: list[str]
    case_key: str = ""


class InferenceEngine:
    @staticmethod
    def _normalize_effective_meta_value(*, key: str, value: Any) -> Any:
        if key == HAS_STRUCTURE_INPUTS_EFFECTIVE_KEY:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "f", "no", "n", "off"}:
                return False
            raise ValueError(f"invalid boolean value for {key}: {value!r}")
        text = str(value).strip().lower()
        if not text:
            raise ValueError(f"empty value for {key} is not allowed")
        return text

    @classmethod
    def _normalize_effective_meta_dict(cls, raw: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in input_mode_metadata_keys():
            if raw is None or key not in raw:
                continue
            out[key] = cls._normalize_effective_meta_value(key=key, value=raw[key])
        return out

    @staticmethod
    def _pick_uniformity_target(
        fields_phys: dict[str, np.ndarray],
        *,
        preferred_keys: list[str] | None = None,
    ) -> tuple[str | None, np.ndarray | None]:
        excluded = {"rho_eff", "e_theta", "e_mag", "phi_raw", "phi_refined"}
        for key in [str(v) for v in (preferred_keys or [])]:
            if key in fields_phys and key not in excluded:
                return key, np.asarray(fields_phys[key], dtype=np.float32)[0]
        for key in sorted(fields_phys.keys()):
            if key in excluded:
                continue
            return key, np.asarray(fields_phys[key], dtype=np.float32)[0]
        return None, None

    @staticmethod
    def _resolve_symbol_key(
        fields_phys: dict[str, np.ndarray],
        *,
        configured: Any,
        aliases: tuple[str, ...],
    ) -> str | None:
        if isinstance(configured, str):
            cand = str(configured).strip()
            if cand and cand in fields_phys:
                return cand
        if isinstance(configured, list):
            for item in configured:
                cand = str(item).strip()
                if cand and cand in fields_phys:
                    return cand
        for name in aliases:
            if name in fields_phys:
                return name
        return None

    def _resolve_boundary_operator_inputs(self, fields_phys: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        bo_cfg = dict(self.ood_cfg.get("boundary_operator", {}))
        physics_cfg = dict(self.ood_cfg.get("physics", {}))
        symbols: dict[str, Any] = {}
        for raw in [physics_cfg.get("symbols"), bo_cfg.get("symbols")]:
            if isinstance(raw, dict):
                symbols.update({str(k): v for k, v in raw.items()})
        density_key = self._resolve_symbol_key(
            fields_phys,
            configured=symbols.get("density"),
            aliases=("ne", "log_ne"),
        )
        temperature_key = self._resolve_symbol_key(
            fields_phys,
            configured=symbols.get("temperature"),
            aliases=("Te",),
        )
        potential_key = self._resolve_symbol_key(
            fields_phys,
            configured=symbols.get("potential"),
            aliases=("phi",),
        )
        missing: list[str] = []
        if density_key is None:
            missing.append("density")
        if temperature_key is None:
            missing.append("temperature")
        if potential_key is None:
            missing.append("potential")
        physics_enabled = bool(physics_cfg.get("enabled", False))
        boundary_enabled = bool(bo_cfg.get("enabled", False))
        if missing and (physics_enabled or boundary_enabled):
            raise ValueError(
                "inference physics symbols unresolved. "
                f"missing={missing}; available={sorted(fields_phys.keys())}. "
                "Set inference.ood.physics.symbols or inference.ood.boundary_operator.symbols."
            )
        if missing:
            return None
        density_arr = np.asarray(fields_phys[density_key], dtype=np.float32)
        if density_key.startswith("log_"):
            log_density = density_arr
        else:
            log_density = np.log10(np.maximum(density_arr, np.float32(1.0e-30))).astype(np.float32)
        return {
            "log_density": log_density,
            "temperature": np.asarray(fields_phys[temperature_key], dtype=np.float32),
            "potential": np.asarray(fields_phys[potential_key], dtype=np.float32),
        }

    def __init__(
        self,
        model: Any,
        cond_schema: CondSchema,
        axis_schema: AxisSchema,
        geometry_provider: GeometryProviderLike,
        output_dir: str | Path,
        transform_bundle: TransformBundle | None = None,
        cond_stats: dict[str, Any] | None = None,
        phi_mode: str = "direct",
        phi_hybrid_steps: int = 3,
        poisson_refine_iters: int = 0,
        ood_cfg: dict[str, Any] | None = None,
        feature_store: Any = None,
        deeponet_head: Any | None = None,
        coord_scaler: dict[str, Any] | None = None,
        coord_feature_scaler: dict[str, Any] | None = None,
        coord_feature_pack: dict[str, Any] | None = None,
        coord_distance_transform_stats: dict[str, Any] | None = None,
        coord_input_scaling_cfg: dict[str, Any] | None = None,
        coord_input_features_cfg: dict[str, Any] | None = None,
        grid_input_features_cfg: dict[str, Any] | None = None,
        unet_input_features_cfg: dict[str, Any] | None = None,
        input_mode: str = TABLE_PLUS_STRUCTURE,
        strict_input_mode: str = "error",
        allow_mode_fallback: bool = False,
        input_mode_meta: dict[str, Any] | None = None,
        checkpoint_input_mode_meta: dict[str, Any] | None = None,
        checkpoint_meta: dict[str, Any] | None = None,
        structure_descriptor_pack: dict[str, Any] | None = None,
        latent_feature_pack: dict[str, Any] | None = None,
    ):
        self.model = model
        self.cond_schema = cond_schema
        self.axis_schema = axis_schema
        self.geometry_provider = geometry_provider
        self.store = ArtifactStore(output_dir)
        self.transforms = transform_bundle
        self.cond_stats = cond_stats or {}
        self.phi_mode = str(phi_mode)
        self.plasma_head = PlasmaHead(mode=self.phi_mode, jacobi_iters=int(phi_hybrid_steps))
        self.poisson_refine_iters = int(poisson_refine_iters)
        self.ood_cfg = ood_cfg or {}
        self.feature_store = feature_store
        self.deeponet_head = deeponet_head
        self.coord_scaler = dict(coord_scaler or {})
        self.coord_feature_scaler = dict(coord_feature_scaler or {})
        self.coord_feature_pack = dict(coord_feature_pack or {})
        self.coord_distance_transform_stats = dict(coord_distance_transform_stats or {})
        self.coord_input_scaling_cfg = dict(coord_input_scaling_cfg or {})
        self.coord_input_features_cfg = dict(coord_input_features_cfg or {})
        self.grid_input_features_cfg = dict(grid_input_features_cfg or unet_input_features_cfg or {})
        self.unet_input_features_cfg = dict(self.grid_input_features_cfg)
        self.strict_input_mode = normalize_strict_input_mode(strict_input_mode, default="error")
        self.allow_mode_fallback = bool(allow_mode_fallback)
        self.input_mode_contract_warnings: list[str] = []
        self.input_mode_fallback_applied = False
        self.input_mode = str(input_mode or TABLE_PLUS_STRUCTURE).strip().lower()
        if self.input_mode not in set(INPUT_MODES):
            raise ValueError(f"inference input_mode must be one of {list(INPUT_MODES)}; got={self.input_mode!r}")
        self.input_mode_meta = self._normalize_effective_meta_dict(input_mode_meta)
        self._checkpoint_input_mode_meta_provided = checkpoint_input_mode_meta is not None
        self.checkpoint_input_mode_meta = self._normalize_effective_meta_dict(checkpoint_input_mode_meta)
        self.checkpoint_meta = dict(checkpoint_meta or {})
        self.structure_descriptor_pack = dict(structure_descriptor_pack or {})
        self.latent_feature_pack = dict(latent_feature_pack or {})
        self._pod_descriptor_cache: dict[str, np.ndarray] = {}
        self._geom_deeponet_siren_descriptor_cache: dict[str, np.ndarray] = {}
        self.geom_deeponet_siren_descriptor_dim_effective = int(
            self.checkpoint_meta.get(GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY, 0) or 0
        )
        if INPUT_MODE_EFFECTIVE_KEY not in self.input_mode_meta:
            self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY] = self.input_mode
        self._validate_input_mode_metadata_contract()
        resolved_mode = str(self.input_mode_meta.get(INPUT_MODE_EFFECTIVE_KEY, self.input_mode)).strip().lower()
        if resolved_mode and resolved_mode in set(INPUT_MODES) and resolved_mode != self.input_mode:
            if self.allow_mode_fallback:
                msg = (
                    "inference input_mode arg differs from resolved metadata; "
                    f"arg={self.input_mode!r}, resolved={resolved_mode!r}; fallback adopts resolved metadata mode"
                )
                self.input_mode_contract_warnings.append(msg)
                self.input_mode_fallback_applied = True
                self.input_mode = resolved_mode
            else:
                raise ValueError(
                    "inference request input_mode metadata mismatch: "
                    f"{INPUT_MODE_EFFECTIVE_KEY}={self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY]!r} "
                    f"but input_mode arg={self.input_mode!r}"
                )
        if str(self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY]).strip().lower() != self.input_mode:
            raise ValueError(
                "inference request input_mode metadata mismatch: "
                f"{INPUT_MODE_EFFECTIVE_KEY}={self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY]!r} "
                f"but input_mode arg={self.input_mode!r}"
            )
        self.provider_mode_effective = str(
            self.input_mode_meta.get(
                GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
                getattr(self.geometry_provider, "provider_mode", "fixed"),
            )
        ).strip().lower() or "fixed"
        provider_mode_runtime = str(getattr(self.geometry_provider, "provider_mode", "fixed")).strip().lower() or "fixed"
        if provider_mode_runtime != self.provider_mode_effective:
            raise ValueError(
                "inference provider mode mismatch between runtime metadata and provider instance: "
                f"meta={self.provider_mode_effective!r}, provider={provider_mode_runtime!r}"
            )
        self.structure_adapter_mode_effective = str(
            self.input_mode_meta.get(STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY, "auto")
        ).strip().lower() or "auto"
        self.structure_descriptor_profile_effective = str(
            self.input_mode_meta.get(STRUCTURE_DESCRIPTOR_PROFILE_EFFECTIVE_KEY, "none")
        ).strip().lower() or "none"
        self.structure_latent_profile_effective = str(
            self.input_mode_meta.get(STRUCTURE_LATENT_PROFILE_EFFECTIVE_KEY, "none")
        ).strip().lower() or "none"
        self.pod_descriptor_dim_effective = int(self.checkpoint_meta.get(DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY, 0) or 0)
        self.pod_latent_hook_effective = bool(self.checkpoint_meta.get(DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY, False))
        self._validate_model_input_mode_contract()
        self._validate_pod_descriptor_and_latent_contract()
        self._validate_geom_deeponet_siren_descriptor_contract()

    def _validate_input_mode_metadata_contract(self) -> None:
        if not self._checkpoint_input_mode_meta_provided:
            return
        resolved_meta, contract_warnings, fallback_applied = resolve_input_mode_metadata_contract(
            request_meta=self.input_mode_meta,
            checkpoint_meta=self.checkpoint_input_mode_meta,
            strict_input_mode=self.strict_input_mode,
            allow_mode_fallback=self.allow_mode_fallback,
            keys=input_mode_metadata_keys(),
            context="inference input-mode metadata",
        )
        self.input_mode_meta = self._normalize_effective_meta_dict(resolved_meta)
        self.input_mode_contract_warnings.extend(contract_warnings)
        if fallback_applied:
            self.input_mode_fallback_applied = True

    @staticmethod
    def _resolve_model_type_name(model: Any) -> str:
        if hasattr(model, "to_meta"):
            meta = model.to_meta()
            if not isinstance(meta, dict):
                raise TypeError(f"{type(model).__name__}.to_meta() must return a dict")
            name = str(meta.get("model_type", "")).strip().lower()
            if name:
                return name
        model_type_attr = str(getattr(model, "model_type", "")).strip().lower()
        if model_type_attr:
            return model_type_attr
        class_name = str(type(model).__name__).strip().lower()
        aliases = {
            "globalmlp": "global_mlp",
            "unetbaseline": "unet",
            "unetppbaseline": "unetpp",
            "fnobaseline": "fno",
            "ffnobaseline": "ffno",
            "unobaseline": "u_no",
            "cnobaseline": "cno",
            "cnooperatorunet": "cno_operator_unet",
            "coordmlptorch": "coord_mlp_fourier",
            "poddeeponettorch": "deeponet_pod",
            "geomdeeponetsiren": "geom_deeponet_siren",
        }
        return aliases.get(class_name, class_name)

    def _validate_model_input_mode_contract(self) -> None:
        if self.input_mode != TABLE_ONLY:
            return
        model_type = self._resolve_model_type_name(self.model)
        try:
            validate_model_input_mode(model_type, TABLE_ONLY)
        except ValueError as exc:
            msg = str(exc)
            if "Unsupported model for input-mode policy" in msg:
                return
            raise ValueError(
                "inference model/input_mode contract violation: "
                f"model_type={model_type!r}, input_mode={TABLE_ONLY!r}; detail={msg}"
            ) from exc

    def _is_pod_model(self) -> bool:
        if isinstance(self.model, PODDeepONetTorch):
            return True
        model_type = self._resolve_model_type_name(self.model)
        return model_type in POD_DEEPONET_FAMILY_MODELS

    def _validate_pod_descriptor_and_latent_contract(self) -> None:
        if not self._is_pod_model():
            return
        mode = self.input_mode
        adapter = self.structure_adapter_mode_effective
        descriptor_profile = self.structure_descriptor_profile_effective
        latent_profile = self.structure_latent_profile_effective
        uses_descriptor_adapter = adapter in {ADAPTER_DESCRIPTOR_BRANCH, ADAPTER_HYBRID_PACK_DESCRIPTOR}

        if mode == TABLE_ONLY:
            if descriptor_profile != "none" or latent_profile != "none":
                raise ValueError(
                    "runtime.input_mode=table_only requires descriptor_profile/latent_profile to be none for deeponet_pod"
                )
            if adapter != ADAPTER_NONE:
                raise ValueError(
                    "runtime.input_mode=table_only requires adapter_mode_effective='none' for deeponet_pod"
                )
            self.pod_descriptor_dim_effective = 0
            self.pod_latent_hook_effective = False
            return

        if descriptor_profile != "none":
            if not uses_descriptor_adapter:
                raise ValueError(
                    "deeponet_pod descriptor profile requires adapter_mode_effective in "
                    "{descriptor_branch, hybrid_pack_descriptor}"
                )
            if self.provider_mode_effective != "parametric_parts":
                raise ValueError(
                    "deeponet_pod descriptor lane requires geometry provider_mode_effective='parametric_parts' "
                    f"for inference; got={self.provider_mode_effective!r}"
                )
            ckpt_descriptor_profile = str(
                self.checkpoint_meta.get(DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY, descriptor_profile)
            ).strip().lower()
            if ckpt_descriptor_profile and ckpt_descriptor_profile != descriptor_profile:
                raise ValueError(
                    "descriptor profile mismatch between runtime request and checkpoint metadata: "
                    f"request={descriptor_profile!r}, checkpoint={ckpt_descriptor_profile!r}"
                )
            if DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY not in self.checkpoint_meta:
                raise ValueError(
                    "checkpoint metadata missing required key for descriptor lane: "
                    f"{DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY!r}"
                )
            descriptor_dim = int(self.checkpoint_meta[DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY])
            if descriptor_dim <= 0:
                raise ValueError(
                    "checkpoint descriptor metadata is invalid: "
                    f"{DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY}={descriptor_dim!r}"
                )
            self.pod_descriptor_dim_effective = int(descriptor_dim)
        elif uses_descriptor_adapter:
            raise ValueError(
                "adapter_mode_effective requires runtime.structure.descriptor_profile != none for deeponet_pod"
            )
        else:
            self.pod_descriptor_dim_effective = 0

        if latent_profile != "none":
            load_vector_from_pack(
                pack=self.latent_feature_pack,
                pack_name="latent_feature_pack",
            )
            ckpt_latent_profile = str(
                self.checkpoint_meta.get(DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY, latent_profile)
            ).strip().lower()
            if ckpt_latent_profile and ckpt_latent_profile != latent_profile:
                raise ValueError(
                    "latent profile mismatch between runtime request and checkpoint metadata: "
                    f"request={latent_profile!r}, checkpoint={ckpt_latent_profile!r}"
                )
            if DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY in self.checkpoint_meta:
                if not bool(self.checkpoint_meta[DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY]):
                    raise ValueError(
                        "checkpoint metadata indicates latent hook disabled while runtime latent profile is enabled"
                    )
            self.pod_latent_hook_effective = True
        else:
            self.pod_latent_hook_effective = False

    def _is_geom_deeponet_siren_model(self) -> bool:
        if isinstance(self.model, GeomDeepONetSIREN):
            return True
        model_type = self._resolve_model_type_name(self.model)
        return model_type == "geom_deeponet_siren"

    def _validate_geom_deeponet_siren_descriptor_contract(self) -> None:
        if not self._is_geom_deeponet_siren_model():
            return
        if self.input_mode != TABLE_PLUS_STRUCTURE:
            raise ValueError("geom_deeponet_siren requires runtime.input_mode=table_plus_structure for inference")
        if self.structure_adapter_mode_effective != ADAPTER_HYBRID_PACK_DESCRIPTOR:
            raise ValueError(
                "geom_deeponet_siren requires runtime.structure.adapter_mode_effective="
                "'hybrid_pack_descriptor' for inference"
            )
        descriptor_profile = self.structure_descriptor_profile_effective
        if descriptor_profile == "none":
            raise ValueError("geom_deeponet_siren requires runtime.structure.descriptor_profile != none for inference")
        if self.provider_mode_effective != "parametric_parts":
            raise ValueError(
                "geom_deeponet_siren descriptor lane requires geometry provider_mode_effective='parametric_parts' "
                f"for inference; got={self.provider_mode_effective!r}"
            )
        ckpt_descriptor_profile = str(
            self.checkpoint_meta.get(GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY, descriptor_profile)
        ).strip().lower()
        if ckpt_descriptor_profile and ckpt_descriptor_profile != descriptor_profile:
            raise ValueError(
                "geom_deeponet_siren descriptor profile mismatch between runtime request and checkpoint metadata: "
                f"request={descriptor_profile!r}, checkpoint={ckpt_descriptor_profile!r}"
            )
        if GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY not in self.checkpoint_meta:
            raise ValueError(
                "checkpoint metadata missing required key for geom_deeponet_siren descriptor lane: "
                f"{GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY!r}"
            )
        descriptor_dim = int(self.checkpoint_meta[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY])
        if descriptor_dim <= 0:
            raise ValueError(
                "checkpoint descriptor metadata is invalid for geom_deeponet_siren: "
                f"{GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY}={descriptor_dim!r}"
            )
        self.geom_deeponet_siren_descriptor_dim_effective = int(descriptor_dim)

    @staticmethod
    def _descriptor_cache_key(geom_ref: dict[str, Any]) -> str:
        return json.dumps(dict(geom_ref or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _resolve_pod_descriptor_vector(self, *, geom_ref: dict[str, Any], geom_ctx: GeometryContext) -> np.ndarray:
        cache_key = self._descriptor_cache_key(geom_ref)
        cached = self._pod_descriptor_cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32).reshape(-1)
        descriptor = build_structure_descriptor(self.structure_descriptor_profile_effective, geom_ctx)
        desc_vec = np.asarray(descriptor.vector, dtype=np.float32).reshape(-1)
        expected_dim = int(self.pod_descriptor_dim_effective)
        if expected_dim > 0 and int(desc_vec.shape[0]) != expected_dim:
            raise ValueError(
                "deeponet_pod descriptor dim mismatch between checkpoint and runtime geometry: "
                f"checkpoint={expected_dim}, runtime={int(desc_vec.shape[0])}"
            )
        self._pod_descriptor_cache[cache_key] = np.asarray(desc_vec, dtype=np.float32).reshape(-1)
        return desc_vec

    def _augment_pod_cond_vector(
        self,
        cond_vec: np.ndarray,
        *,
        geom_ref: dict[str, Any],
        geom_ctx: GeometryContext,
    ) -> np.ndarray:
        if not self._is_pod_model():
            return np.asarray(cond_vec, dtype=np.float32)
        if self.structure_descriptor_profile_effective == "none":
            return np.asarray(cond_vec, dtype=np.float32)
        desc_vec = self._resolve_pod_descriptor_vector(geom_ref=geom_ref, geom_ctx=geom_ctx)
        return np.concatenate([np.asarray(cond_vec, dtype=np.float32), desc_vec], axis=0).astype(np.float32)

    def _resolve_geom_deeponet_siren_descriptor_vector(
        self,
        *,
        geom_ref: dict[str, Any],
        geom_ctx: GeometryContext,
    ) -> np.ndarray:
        cache_key = self._descriptor_cache_key(geom_ref)
        cached = self._geom_deeponet_siren_descriptor_cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32).reshape(-1)
        descriptor = build_structure_descriptor(self.structure_descriptor_profile_effective, geom_ctx)
        desc_vec = np.asarray(descriptor.vector, dtype=np.float32).reshape(-1)
        expected_dim = int(self.geom_deeponet_siren_descriptor_dim_effective)
        if expected_dim > 0 and int(desc_vec.shape[0]) != expected_dim:
            raise ValueError(
                "geom_deeponet_siren descriptor dim mismatch between checkpoint and runtime geometry: "
                f"checkpoint={expected_dim}, runtime={int(desc_vec.shape[0])}"
            )
        self._geom_deeponet_siren_descriptor_cache[cache_key] = np.asarray(desc_vec, dtype=np.float32).reshape(-1)
        return desc_vec

    def _augment_geom_deeponet_siren_cond_vector(
        self,
        cond_vec: np.ndarray,
        *,
        geom_ref: dict[str, Any],
        geom_ctx: GeometryContext,
    ) -> np.ndarray:
        if not self._is_geom_deeponet_siren_model():
            return np.asarray(cond_vec, dtype=np.float32)
        desc_vec = self._resolve_geom_deeponet_siren_descriptor_vector(geom_ref=geom_ref, geom_ctx=geom_ctx)
        return np.concatenate([np.asarray(cond_vec, dtype=np.float32), desc_vec], axis=0).astype(np.float32)

    @staticmethod
    def _resolve_distance_transform_cfg(raw: dict[str, Any] | None) -> dict[str, Any]:
        cfg = dict(raw or {})
        mode = str(cfg.get("mode", "raw")).strip().lower()
        if mode not in {"raw", "bounded", "bounded_auto"}:
            raise ValueError(
                "train.input_features.distance_transform.mode must be one of: raw, bounded, bounded_auto"
            )
        signed_tanh_tau = float(cfg.get("signed_tanh_tau", 8.0))
        proximity_tau = float(cfg.get("proximity_tau", 6.0))
        if signed_tanh_tau <= 0.0:
            raise ValueError("distance_transform.signed_tanh_tau must be > 0")
        if proximity_tau <= 0.0:
            raise ValueError("distance_transform.proximity_tau must be > 0")
        return {
            "mode": mode,
            "signed_tanh_tau": signed_tanh_tau,
            "proximity_tau": proximity_tau,
            "replace_distance_any": bool(cfg.get("replace_distance_any", True)),
        }

    @staticmethod
    def _resolve_distance_transform_effective(
        cfg: dict[str, Any],
        stats: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mode = str(cfg.get("mode", "raw")).strip().lower()
        out = dict(cfg)
        if mode != "bounded_auto":
            return out
        raw_stats = dict(stats or {})
        s_tau = float(raw_stats.get("signed_tanh_tau_auto", 0.0))
        p_tau = float(raw_stats.get("proximity_tau_auto", 0.0))
        if s_tau > 0.0 and p_tau > 0.0:
            out["signed_tanh_tau"] = s_tau
            out["proximity_tau"] = p_tau
            out["signed_quantile"] = float(raw_stats.get("signed_quantile", 0.75))
            out["proximity_quantile"] = float(raw_stats.get("proximity_quantile", 0.50))
        return out

    @staticmethod
    def _apply_distance_transform(rows: np.ndarray, channels: list[str], cfg: dict[str, Any]) -> np.ndarray:
        out = np.asarray(rows, dtype=np.float32).copy()
        mode = str(cfg.get("mode", "raw")).strip().lower()
        if mode == "raw":
            return out
        signed_tau = max(float(cfg.get("signed_tanh_tau", 8.0)), 1e-6)
        prox_tau = max(float(cfg.get("proximity_tau", 6.0)), 1e-6)
        if "distance_signed" in channels:
            idx = channels.index("distance_signed")
            out[:, idx] = np.tanh(out[:, idx] / signed_tau).astype(np.float32)
        if "distance_any" in channels and bool(cfg.get("replace_distance_any", True)):
            idx = channels.index("distance_any")
            out[:, idx] = np.exp(-np.maximum(out[:, idx], 0.0) / prox_tau).astype(np.float32)
        return out.astype(np.float32)

    def _transform_coord(self, coord: np.ndarray) -> np.ndarray:
        mode = str(self.coord_input_scaling_cfg.get("mode", "none")).strip().lower()
        if mode not in {"none", "zscore", "minmax"}:
            raise ValueError("train.input_features.coord_input_scaling.mode must be one of: none, zscore, minmax")
        source = str(self.coord_input_scaling_cfg.get("source", "preprocess")).strip().lower()
        if source not in {"preprocess", "runtime_fit"}:
            raise ValueError("train.input_features.coord_input_scaling.source must be one of: preprocess, runtime_fit")
        arr = np.asarray(coord, dtype=np.float32)
        if mode == "none":
            return arr
        if source == "preprocess":
            payload = self.coord_scaler.get(mode)
            if not isinstance(payload, dict) or not payload:
                raise ValueError(
                    "coord_input_scaling.source=preprocess requires preprocessing/scalers/coord_scaler.json "
                    f"with '{mode}' payload"
                )
            scaler = ScalerFactory.from_dict(payload)
            return scaler.transform(arr).astype(np.float32)
        scaler = ScalerFactory.create(mode).fit(arr)
        return scaler.transform(arr).astype(np.float32)

    def _build_cond_vector(self, cond: dict[str, Any], axis: dict[str, Any]) -> np.ndarray:
        base = self.cond_schema.encode(cond)
        axis_vec = self.axis_schema.encode(float(axis.get("value", 0.0)))
        cond_vec = np.concatenate([base, axis_vec], axis=0).astype(np.float32)
        if self.transforms is not None:
            return self.transforms.transform_cond(cond_vec[None, :])[0]
        return cond_vec

    @staticmethod
    def _coord_xy_from_geom(geom: GeometryContext) -> np.ndarray:
        raw_coord = np.asarray(geom.coord_grid, dtype=np.float32)
        if raw_coord.ndim != 3:
            raise ValueError(f"geom.coord_grid must be rank-3, got {raw_coord.shape}")
        if raw_coord.shape[0] == 2:
            return raw_coord.reshape(2, -1).T.astype(np.float32)
        if raw_coord.shape[-1] == 2:
            return raw_coord.reshape(-1, 2).astype(np.float32)
        raise ValueError(
            "geom.coord_grid shape mismatch: "
            f"expected (2,H,W) or (H,W,2), got {raw_coord.shape}"
        )

    @staticmethod
    def _resolve_coord_feature_channels(model: Any) -> list[str]:
        raw = getattr(model, "input_feature_channels", None)
        if not isinstance(raw, list) or len(raw) == 0:
            return ["x", "y"]
        channels = [str(v) for v in raw]
        try:
            return list(validate_coord_feature_channels(channels))
        except ValueError as exc:
            raise ValueError(f"Unsupported coord feature channels in checkpoint metadata: {channels}") from exc

    def _build_coord_feature_rows(self, geom: GeometryContext, channels: list[str]) -> np.ndarray:
        h, w = geom.mask_plasma.shape
        pack = dict(self.coord_feature_pack or {})
        require_pack = str(self.coord_input_features_cfg.get("require_pack", "off")).strip().lower()
        if require_pack not in {"off", "warn", "error"}:
            raise ValueError("train.input_features.require_pack must be one of: off, warn, error")
        distance_transform_cfg = self._resolve_distance_transform_cfg(
            dict(self.coord_input_features_cfg).get("distance_transform")
        )
        distance_transform_cfg = self._resolve_distance_transform_effective(
            distance_transform_cfg,
            self.coord_distance_transform_stats,
        )
        scaler_cfg = dict(self.coord_feature_scaler or {})
        has_coord_feature_scaler = bool(dict(scaler_cfg.get("channels", {})))
        if pack:
            data = np.asarray(pack.get("data"), dtype=np.float32) if "data" in pack else None
            raw_channels = pack.get("channels")
            if data is not None and raw_channels is not None and data.ndim == 3 and data.shape[1:] == (h, w):
                pack_channels = [str(v) for v in np.asarray(raw_channels).reshape(-1).tolist()]
                pack_map = {name: data[i] for i, name in enumerate(pack_channels) if i < data.shape[0]}
                if all(name in pack_map for name in channels):
                    stacked = np.stack([pack_map[name] for name in channels], axis=0).astype(np.float32)
                    rows = stacked.reshape(len(channels), -1).T.astype(np.float32)
                    if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
                        idx_x = channels.index("x")
                        idx_y = channels.index("y")
                        rows[:, [idx_x, idx_y]] = self._transform_coord(rows[:, [idx_x, idx_y]])
                    rows = self._apply_distance_transform(rows, channels, distance_transform_cfg)
                    rows = self._transform_coord_features(rows, channels)
                    return rows

        if require_pack == "error":
            raise ValueError("input-feature contract requires preprocess feature_pack, but pack is missing")
        coord_xy = self._coord_xy_from_geom(geom)
        distance_any = np.asarray(geom.distance_any, dtype=np.float32).reshape(-1)
        raw_signed = getattr(geom, "distance_signed", None)
        if raw_signed is None:
            mask_tmp = np.asarray(geom.mask_plasma, dtype=np.float32).reshape(-1)
            distance_signed = np.where(mask_tmp > 0.5, distance_any, -distance_any).astype(np.float32)
        else:
            distance_signed = np.asarray(raw_signed, dtype=np.float32).reshape(-1)
        mask_plasma = np.asarray(geom.mask_plasma, dtype=np.float32).reshape(-1)
        signed_2d = distance_signed.reshape(h, w)
        gy, gx = np.gradient(signed_2d, edge_order=1)
        gnorm = np.sqrt(gx**2 + gy**2).astype(np.float32)
        gnorm = np.where(gnorm < 1e-6, 1e-6, gnorm).astype(np.float32)
        normal_x = (gx / gnorm).reshape(-1).astype(np.float32)
        normal_y = (gy / gnorm).reshape(-1).astype(np.float32)
        dnx_dy, dnx_dx = np.gradient(gx / gnorm, edge_order=1)
        dny_dy, dny_dx = np.gradient(gy / gnorm, edge_order=1)
        curvature_proxy = (dnx_dx + dny_dy).reshape(-1).astype(np.float32)
        mapping = {
            "x": coord_xy[:, 0],
            "y": coord_xy[:, 1],
            "distance_signed": distance_signed,
            "distance_any": distance_any,
            "mask_plasma": mask_plasma,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "curvature_proxy": curvature_proxy,
        }
        rows = np.stack([np.asarray(mapping[name], dtype=np.float32) for name in channels], axis=1).astype(np.float32)
        if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
            idx_x = channels.index("x")
            idx_y = channels.index("y")
            rows[:, [idx_x, idx_y]] = self._transform_coord(rows[:, [idx_x, idx_y]])
        rows = self._apply_distance_transform(rows, channels, distance_transform_cfg)
        rows = self._transform_coord_features(rows, channels)
        return rows

    def _transform_coord_features(self, rows: np.ndarray, channels: list[str]) -> np.ndarray:
        raw = dict(self.coord_feature_scaler or {})
        if not bool(raw.get("enabled", False)):
            return np.asarray(rows, dtype=np.float32)
        channel_scalers = dict(raw.get("channels", {}))
        if len(channel_scalers) == 0:
            return np.asarray(rows, dtype=np.float32)
        out = np.asarray(rows, dtype=np.float32).copy()
        for i, name in enumerate(channels):
            payload = channel_scalers.get(name)
            if not isinstance(payload, dict) or len(payload) == 0:
                continue
            scaler = ScalerFactory.from_dict(payload)
            out[:, i : i + 1] = scaler.transform(out[:, i : i + 1]).astype(np.float32)
        return out.astype(np.float32)

    def _require_coord_feature_scaling_enabled(self) -> None:
        raw = dict(self.coord_feature_scaler or {})
        enabled = bool(raw.get("enabled", False))
        mode = str(raw.get("mode", "none")).strip().lower()
        if (not enabled) or mode == "none":
            raise ValueError(
                "coord_mlp requires preprocessing.coord_features.scaling.enabled=true "
                "(mode must not be none) for inference"
            )

    def _build_grid_feature_rows(self, geom: GeometryContext, channels: list[str]) -> np.ndarray:
        h, w = geom.mask_plasma.shape
        pack = dict(self.coord_feature_pack or {})
        cfg = dict(self.grid_input_features_cfg or {})
        mode = str(cfg.get("mode", "geom_feature_pack")).strip().lower()
        if mode not in {"legacy_xy", "geom_feature_pack"}:
            raise ValueError("train.<grid_model>.input_features.mode must be one of: legacy_xy, geom_feature_pack")
        if mode == "legacy_xy" and list(channels) == ["x", "y"]:
            yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
            xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
            yv, xv = np.meshgrid(yy, xx, indexing="ij")
            return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)
        require_pack = str(cfg.get("require_pack", "off")).strip().lower()
        if require_pack not in {"off", "warn", "error"}:
            raise ValueError("train.<grid_model>.input_features.require_pack must be one of: off, warn, error")
        distance_transform_cfg = self._resolve_distance_transform_cfg(dict(cfg.get("distance_transform") or {}))
        distance_transform_cfg = self._resolve_distance_transform_effective(
            distance_transform_cfg,
            self.coord_distance_transform_stats,
        )
        scaler_cfg = dict(self.coord_feature_scaler or {})
        has_coord_feature_scaler = bool(dict(scaler_cfg.get("channels", {})))
        if pack:
            data = np.asarray(pack.get("data"), dtype=np.float32) if "data" in pack else None
            raw_channels = pack.get("channels")
            if data is not None and raw_channels is not None and data.ndim == 3 and data.shape[1:] == (h, w):
                pack_channels = [str(v) for v in np.asarray(raw_channels).reshape(-1).tolist()]
                pack_map = {name: data[i] for i, name in enumerate(pack_channels) if i < data.shape[0]}
                if all(name in pack_map for name in channels):
                    stacked = np.stack([pack_map[name] for name in channels], axis=0).astype(np.float32)
                    rows = stacked.reshape(len(channels), -1).T.astype(np.float32)
                    if (not has_coord_feature_scaler) and {"x", "y"}.issubset(set(channels)):
                        idx_x = channels.index("x")
                        idx_y = channels.index("y")
                        rows[:, [idx_x, idx_y]] = self._transform_coord(rows[:, [idx_x, idx_y]])
                    rows = self._apply_distance_transform(rows, channels, distance_transform_cfg)
                    rows = self._transform_coord_features(rows, channels)
                    return rows
        if require_pack == "error":
            raise ValueError("unet input-feature contract requires preprocess coord_feature_pack, but pack is missing")
        return self._build_coord_feature_rows(geom, channels)

    def _predict_grid_spatial_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if isinstance(self.model, CoordMLPTorch):
            self._require_coord_feature_scaling_enabled()
        channels = self._resolve_coord_feature_channels(self.model)
        spatial_rows = self._build_grid_feature_rows(geom, channels)
        h, w = geom.mask_plasma.shape
        spatial_map = spatial_rows.reshape(h, w, len(channels))[None, ...].astype(np.float32)
        if isinstance(self.model, (UNetBaseline, UNetPPBaseline)):
            pred = self.model.forward_features(cond_vec[None, :], spatial_features=spatial_map)
        else:
            pred = self.model.predict_fields(cond_vec[None, :], spatial_features=spatial_map)
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

    def _predict_cond_only_fullfield_fields(self, cond_vec: np.ndarray) -> dict[str, np.ndarray]:
        pred = self.model.predict_fields(cond_vec[None, :])
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

    def _validate_geom_ref(self, geom: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(geom, dict):
            raise TypeError("geom must be a dict with {'geom_id': ...} or {'geom_param': {...}}")
        if "geom_param" in geom:
            if self.input_mode == TABLE_ONLY:
                raise ValueError("runtime.input_mode=table_only does not allow geom_ref.geom_param")
            if not bool(getattr(self.geometry_provider, "supports_geom_param", False)):
                raise ValueError(
                    "geom_ref.geom_param requires provider_mode=parametric_parts; "
                    f"got provider_mode={getattr(self.geometry_provider, 'provider_mode', 'fixed')!r}"
                )
            gp = geom["geom_param"]
            if not isinstance(gp, dict):
                raise TypeError("geom_param must be dict")
            norm: dict[str, float] = {}
            for key, value in sorted(gp.items()):
                fval = float(value)
                if not np.isfinite(fval):
                    raise ValueError(f"geom_param[{key!r}] must be finite; got={value!r}")
                norm[str(key)] = fval
            return {"geom_id": str(geom.get("geom_id", "default")), "geom_param": norm}
        out = {"geom_id": str(geom.get("geom_id", "default"))}
        if (
            (self._is_pod_model() or self._is_geom_deeponet_siren_model())
            and self.input_mode == TABLE_PLUS_STRUCTURE
            and self.structure_descriptor_profile_effective != "none"
            and bool(getattr(self.geometry_provider, "supports_geom_param", False))
        ):
            # Force parametric_parts reconstruction so part_mask_stack is populated for descriptor lane.
            out["geom_param"] = {}
        return out

    @staticmethod
    def _case_key(cond: dict[str, Any], axis: dict[str, Any], geom: dict[str, Any]) -> str:
        payload = {
            "cond": {k: float(v) for k, v in sorted(cond.items())},
            "axis": {"mode": str(axis.get("mode", "steady")), "value": float(axis.get("value", 0.0))},
            "geom": geom,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def _predict_fields(self, cond_vec: np.ndarray, geom: GeometryContext) -> dict[str, np.ndarray]:
        if isinstance(self.model, GlobalMLP):
            pred = self.model.predict_fields(cond_vec[None, :])
            return {k: v[0] for k, v in pred.items()}

        if isinstance(
            self.model,
            (
                UNetBaseline,
                UNetPPBaseline,
                FNOBaseline,
                FFNOBaseline,
                UNOBaseline,
                CNOBaseline,
                CNOOperatorUNet,
                CoordMLPTorch,
                GeomDeepONetSIREN,
            ),
        ):
            return self._predict_grid_spatial_fields(cond_vec, geom)

        if isinstance(self.model, PODDeepONetTorch):
            return self._predict_cond_only_fullfield_fields(cond_vec)

        if hasattr(self.model, "predict_fields"):
            try:
                pred = self.model.predict_fields(cond_vec[None, :], geom_ctx=geom)
            except TypeError:
                pred = self.model.predict_fields(cond_vec[None, :])
            return {k: np.asarray(v[0], dtype=np.float32) for k, v in pred.items()}

        raise TypeError("Unsupported model type")

    @staticmethod
    def _compute_derived(phi: np.ndarray) -> dict[str, np.ndarray]:
        gy, gx = np.gradient(phi[0], edge_order=1)
        e_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)
        return {"E_mag": e_mag[None, ...]}

    def _get_geom_ctx(self, geom_ref: dict[str, Any], axis: dict[str, Any]) -> GeometryContext:
        if self.feature_store is not None:
            axis_value = float(axis.get("value", 0.0))
            axis_mode = str(axis.get("mode", "steady"))
            return self.feature_store.get_context(
                geom_ref=geom_ref,
                axis_value=axis_value,
                axis_mode=axis_mode,
                geometry_provider=self.geometry_provider,
            )
        return self.geometry_provider.get(geom_ref)

    def _ood_warnings(self, cond: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        for key, stats in self.cond_stats.items():
            if key not in cond:
                continue
            lo = float(stats.get("min", -np.inf))
            hi = float(stats.get("max", np.inf))
            val = float(cond[key])
            if val < lo or val > hi:
                warnings.append(f"ood:cond:{key}:{val:.6g} outside [{lo:.6g},{hi:.6g}]")
        limit = float(self.ood_cfg.get("poisson_residual_limit", np.inf))
        if float(diagnostics.get("poisson_residual_norm", 0.0)) > limit:
            warnings.append("ood:poisson_residual")
        bo_cfg = self.ood_cfg.get("boundary_operator", {})
        bo_limit = bo_cfg.get("loss_limit")
        if bo_limit is not None and float(diagnostics.get("boundary_operator_proxy_loss", 0.0)) > float(bo_limit):
            warnings.append("ood:boundary_operator")
        return warnings

    def _resolve_axis_samples(self, axis_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        mode = str(axis_cfg.get("mode", self.axis_schema.mode))
        if mode != self.axis_schema.mode:
            raise ValueError(f"axis mode mismatch: expected={self.axis_schema.mode} got={mode}")
        aggregation = str(axis_cfg.get("aggregation", "single"))
        if aggregation == "single":
            return [{"mode": mode, "value": float(axis_cfg.get("value", 0.0))}]
        if aggregation not in {"window_mean", "window_max"}:
            raise ValueError(f"Unsupported axis aggregation: {aggregation}")
        if mode not in {"time", "phase_sincos"}:
            raise ValueError(f"axis aggregation '{aggregation}' requires time/phase_sincos mode, got={mode}")
        window = axis_cfg.get("window", [0.0, 1.0])
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("axis.window must be [lo, hi]")
        lo, hi = float(window[0]), float(window[1])
        if lo > hi:
            raise ValueError("axis.window requires lo <= hi")
        n_points = int(axis_cfg.get("n_points", 5))
        if n_points < 1:
            raise ValueError("axis.n_points must be >= 1")
        values = np.linspace(lo, hi, n_points, dtype=np.float32)
        if mode == "phase_sincos":
            # Keep phase in [0, 1) for periodic axis encoding.
            values = np.mod(values, 1.0)
        return [{"mode": mode, "value": float(v)} for v in values.tolist()]

    @staticmethod
    def _is_numeric_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating))

    @staticmethod
    def _aggregate_results(results: list[InferenceResult], mode: str) -> InferenceResult:
        if len(results) == 0:
            raise ValueError("No results to aggregate")
        if mode == "window_max":
            best = max(results, key=lambda r: float(r.qoi.get("uniformity", 0.0)))
            return InferenceResult(
                fields_model={k: np.asarray(v, dtype=np.float32) for k, v in best.fields_model.items()},
                fields_phys={k: np.asarray(v, dtype=np.float32) for k, v in best.fields_phys.items()},
                derived={k: np.asarray(v, dtype=np.float32) for k, v in best.derived.items()},
                qoi={
                    k: (float(v) if InferenceEngine._is_numeric_scalar(v) else v)
                    for k, v in best.qoi.items()
                },
                diagnostics={k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in best.diagnostics.items()},
                warnings=sorted(set(best.warnings)),
                case_key=str(best.case_key),
            )

        # window_mean
        mean_fields_model = {
            k: np.mean(np.stack([np.asarray(r.fields_model[k], dtype=np.float32) for r in results], axis=0), axis=0)
            for k in results[0].fields_model.keys()
        }
        mean_fields_phys = {
            k: np.mean(np.stack([np.asarray(r.fields_phys[k], dtype=np.float32) for r in results], axis=0), axis=0)
            for k in results[0].fields_phys.keys()
        }
        mean_derived = {
            k: np.mean(np.stack([np.asarray(r.derived[k], dtype=np.float32) for r in results], axis=0), axis=0)
            for k in results[0].derived.keys()
        }
        mean_qoi: dict[str, Any] = {}
        for key in results[0].qoi.keys():
            first = results[0].qoi[key]
            if InferenceEngine._is_numeric_scalar(first):
                mean_qoi[key] = float(np.mean([float(r.qoi[key]) for r in results]))
            else:
                mean_qoi[key] = first
        mean_diag = {
            k: float(np.mean([float(r.diagnostics[k]) for r in results]))
            for k in results[0].diagnostics.keys()
            if isinstance(results[0].diagnostics[k], (int, float, np.floating))
        }
        warnings = sorted(set(w for r in results for w in r.warnings))
        return InferenceResult(
            fields_model=mean_fields_model,
            fields_phys=mean_fields_phys,
            derived=mean_derived,
            qoi=mean_qoi,
            diagnostics=mean_diag,
            warnings=warnings,
            case_key="",
        )

    def single_run_aggregated(self, cond: dict[str, Any], geom: dict[str, Any], axis: dict[str, Any]) -> InferenceResult:
        aggregation = str(axis.get("aggregation", "single"))
        axis_samples = self._resolve_axis_samples(axis)
        if aggregation == "single":
            return self.single_run(cond=cond, geom=geom, axis=axis_samples[0])
        runs = [self.single_run(cond=cond, geom=geom, axis=sample) for sample in axis_samples]
        return self._aggregate_results(runs, aggregation)

    def single_run(self, cond: dict[str, Any], geom: dict[str, Any], axis: dict[str, Any]) -> InferenceResult:
        geom_ref = self._validate_geom_ref(geom)
        geom_ctx = self._get_geom_ctx(geom_ref, axis=axis)
        cond_vec = self._build_cond_vector(cond, axis)
        cond_vec = self._augment_pod_cond_vector(cond_vec, geom_ref=geom_ref, geom_ctx=geom_ctx)
        cond_vec = self._augment_geom_deeponet_siren_cond_vector(cond_vec, geom_ref=geom_ref, geom_ctx=geom_ctx)
        raw_pred = self._predict_fields(cond_vec, geom_ctx)

        fields_model = {k: np.asarray(v, dtype=np.float32) for k, v in raw_pred.items() if k != "rho_eff"}

        if self.transforms is not None:
            fields_phys = self.transforms.inverse_field_dict(fields_model)
        else:
            fields_phys = {k: np.asarray(v, dtype=np.float32) for k, v in fields_model.items()}
        if "rho_eff" in raw_pred:
            fields_phys["rho_eff"] = np.asarray(raw_pred["rho_eff"], dtype=np.float32)
        head_in = {}
        for key, value in fields_phys.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None, ...]
            head_in[key] = arr
        head_out, head_aux = self.plasma_head.apply(
            head_in,
            geom_ctx=geom_ctx,
            refine_iters=self.poisson_refine_iters,
            deeponet_head=self.deeponet_head,
            cond_vec=cond_vec,
        )
        fields_phys = {}
        for key, value in head_out.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            fields_phys[key] = arr
        for key, value in head_aux.items():
            if key not in fields_phys:
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                fields_phys[key] = arr
        derived = self._compute_derived(fields_phys["phi"])

        wafer = geom_ctx.regions.get("wafer_mask")
        preferred_uniformity = self.ood_cfg.get("uniformity_target")
        preferred_uniformity_keys = [str(preferred_uniformity)] if isinstance(preferred_uniformity, str) else None
        qoi_target_key, qoi_target = self._pick_uniformity_target(fields_phys, preferred_keys=preferred_uniformity_keys)
        if qoi_target is None:
            qoi = {"uniformity": 0.0}
        else:
            if wafer is not None:
                vals = qoi_target[wafer > 0.5]
            else:
                vals = qoi_target[geom_ctx.mask_plasma > 0.5]
            qoi = {"uniformity": uniformity(vals)}
            qoi["uniformity_target"] = str(qoi_target_key)
        bo_cfg = self.ood_cfg.get("boundary_operator", {})
        delta_edge = float(bo_cfg.get("delta_edge", 1.5))
        wafer_only = bool(bo_cfg.get("wafer_only", False))
        band_mask = boundary_band_mask(
            mask_plasma=geom_ctx.mask_plasma,
            distance_any=geom_ctx.distance_any,
            delta_edge=delta_edge,
            wafer_mask=wafer,
            wafer_only=wafer_only,
        )
        op_inputs = self._resolve_boundary_operator_inputs(fields_phys)
        if op_inputs is not None:
            gamma_vals = boundary_gamma_proxy(
                log_ne=op_inputs["log_density"][0],
                te=op_inputs["temperature"][0],
                phi=op_inputs["potential"][0],
                mask_band=band_mask,
            )
            if gamma_vals.size > 0:
                qoi["boundary_gamma_mean"] = float(np.mean(gamma_vals))
                qoi["boundary_gamma_uniformity"] = uniformity(gamma_vals)
            else:
                qoi["boundary_gamma_mean"] = 0.0
                qoi["boundary_gamma_uniformity"] = 0.0
        else:
            qoi["boundary_gamma_mean"] = 0.0
            qoi["boundary_gamma_uniformity"] = 0.0

        diagnostics = {
            "poisson_residual_norm": float(poisson_residual_norm(fields_phys["phi"][0], eps=geom_ctx.eps)),
            "bc_phi_mae": float(
                bc_mae(
                    fields_phys["phi"][0],
                    bc_mask=np.zeros_like(geom_ctx.mask_plasma) if geom_ctx.bc_dir_mask is None else geom_ctx.bc_dir_mask,
                    bc_value=np.zeros_like(geom_ctx.mask_plasma) if geom_ctx.bc_dir_value is None else geom_ctx.bc_dir_value,
                )
            ),
            "n_active": int(np.sum(geom_ctx.mask_plasma > 0.5)),
            "boundary_band_n": int(np.sum(band_mask > 0.5)),
            "boundary_operator_proxy_loss": 0.0,
            "coord_split_merge_effective": False,
        }
        if op_inputs is not None:
            diagnostics["boundary_operator_proxy_loss"] = float(
                boundary_operator_loss(
                    phi=op_inputs["potential"],
                    log_ne=op_inputs["log_density"],
                    te=op_inputs["temperature"],
                    mask_band=band_mask,
                    mode=str(bo_cfg.get("mode", "proxy")),
                    target_coeffs=bo_cfg.get("target_coeffs"),
                    prior_coeffs=bo_cfg.get("prior_coeffs"),
                    operator_handle=bo_cfg.get("operator_handle"),
                    external_operator_handle=bo_cfg.get("external_operator_handle"),
                    target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
                )
            )
        rhs_map = None
        if "rho_eff" in fields_phys:
            rhs_map = -np.asarray(fields_phys["rho_eff"], dtype=np.float32)
        poisson_map = poisson_residual(fields_phys["phi"], rhs=rhs_map)[0].astype(np.float32)
        phi_for_bo = op_inputs["potential"] if op_inputs is not None else fields_phys["phi"]
        bo_target = np.zeros_like(phi_for_bo[0], dtype=np.float32)
        if op_inputs is not None:
            bo_target = boundary_operator_target(
                log_ne=op_inputs["log_density"],
                te=op_inputs["temperature"],
                phi=op_inputs["potential"],
                mode=str(bo_cfg.get("mode", "proxy")),
                target_coeffs=bo_cfg.get("target_coeffs"),
                prior_coeffs=bo_cfg.get("prior_coeffs"),
                operator_handle=bo_cfg.get("operator_handle"),
                external_operator_handle=bo_cfg.get("external_operator_handle"),
                target_clamp=tuple(bo_cfg["target_clamp"]) if bo_cfg.get("target_clamp") is not None else None,
            )[0].astype(np.float32)
        bo_residual = ((fields_phys["phi"][0] - bo_target) * band_mask).astype(np.float32)
        diagnostics["poisson_residual_map_l2"] = float(np.sqrt(np.mean(poisson_map**2)))
        diagnostics["boundary_operator_residual_map_l2"] = float(np.sqrt(np.mean(bo_residual**2)))

        warnings = sorted(set([*self.input_mode_contract_warnings, *self._ood_warnings(cond=cond, diagnostics=diagnostics)]))
        cond_warnings = [w for w in warnings if w.startswith("ood:cond:")]
        physics_warnings = [w for w in warnings if not w.startswith("ood:cond:")]
        ood_report = {
            "warnings": warnings,
            "cond_warnings": cond_warnings,
            "physics_warnings": physics_warnings,
            INPUT_MODE_FALLBACK_APPLIED_KEY: bool(self.input_mode_fallback_applied),
            "axis_context": {
                "mode": str(axis.get("mode", self.axis_schema.mode)),
                "value": float(axis.get("value", 0.0)),
                "aggregation": str(axis.get("aggregation", "single")),
                "window": axis.get("window"),
                "n_points": axis.get("n_points"),
            },
            "limits": {
                "cond_stats": self.cond_stats,
                "poisson_residual_limit": self.ood_cfg.get("poisson_residual_limit"),
                "boundary_operator": self.ood_cfg.get("boundary_operator", {}),
            },
        }

        case_key = self._case_key(cond=cond, axis=axis, geom=geom_ref)
        self.store.save_npz(f"single/{case_key}/fields_model.npz", **fields_model)
        self.store.save_npz(f"single/{case_key}/fields_phys.npz", **fields_phys)
        self.store.save_npz(f"single/{case_key}/derived.npz", **derived)
        self.store.save_json(f"single/{case_key}/qoi.json", qoi)
        self.store.save_json(f"single/{case_key}/diagnostics.json", diagnostics)
        self.store.save_json(f"single/{case_key}/ood_report.json", ood_report)
        self.store.save_npz(
            f"single/{case_key}/diagnostics_maps.npz",
            poisson_residual_map=poisson_map,
            boundary_band_mask=band_mask.astype(np.float32),
            boundary_operator_target=bo_target,
            boundary_operator_residual_map=bo_residual,
        )
        if warnings:
            self.store.save_json(f"single/{case_key}/warnings.json", {"warnings": warnings})

        return InferenceResult(
            fields_model=fields_model,
            fields_phys=fields_phys,
            derived=derived,
            qoi=qoi,
            diagnostics=diagnostics,
            warnings=warnings,
            case_key=case_key,
        )

    def batch_run(self, conds: list[dict[str, Any]], geom: dict[str, Any], axis: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cond in conds:
            result = self.single_run_aggregated(cond=cond, geom=geom, axis=axis)
            row = dict(cond)
            row.update(result.qoi)
            rows.append(row)

        header = sorted({k for row in rows for k in row.keys()})
        csv_rows = [[row.get(h, "") for h in header] for row in rows]
        self.store.save_csv("batch/summary.csv", header, csv_rows)
        return rows

    def optimize_run(
        self,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]] | None,
        n_trials: int,
        geom: dict[str, Any],
        axis: dict[str, Any],
        seed: int = 0,
        backend: str = "random",
        backend_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_mode = str(getattr(self.geometry_provider, "provider_mode", "fixed")).strip().lower()
        geom_space_norm = validate_optimize_geom_contract(
            input_mode=self.input_mode,
            provider_mode=provider_mode,
            geom_space=geom_space,
            geom_ref=geom,
            label_prefix="inference.optimize",
        )
        runner = OptimizeRunner(self)
        result = runner.run(
            space=space,
            geom_space=geom_space_norm,
            n_trials=n_trials,
            geom_ref=geom,
            axis=axis,
            seed=seed,
            backend=backend,
            backend_cfg=backend_cfg,
        )
        self.store.save_json(
            "optimize/best.json",
            {
                "best_cond": result.best_cond,
                "best_geom_param": result.best_geom_param,
                "best_value": result.best_value,
            },
        )
        self.store.save_json(
            "optimize/summary.json",
            {
                "backend": result.backend,
                "backend_cfg": result.backend_cfg,
                "seed": int(seed),
                "n_trials": int(n_trials),
                "objective_key": result.objective_key,
                "best_value": float(result.best_value),
                "geom_space_enabled_effective": bool(len(geom_space_norm) > 0),
                "geom_param_keys_effective": sorted(list(result.best_geom_param.keys())),
                "invalid_trial_count": int(result.invalid_trial_count),
            },
        )
        rows = []
        for i, t in enumerate(result.trials):
            row = {
                "trial": i,
                **{k: v for k, v in t.items() if k not in {"cond", "geom_param"}},
                **dict(t.get("cond", {})),
                **dict(t.get("geom_param", {})),
            }
            rows.append(row)
        header = sorted({k for row in rows for k in row.keys()})
        self.store.save_csv("optimize/trials.csv", header, [["" if r.get(k) is None else r.get(k, "") for k in header] for r in rows])
        return {
            "best_cond": result.best_cond,
            "best_geom_param": result.best_geom_param,
            "best_value": result.best_value,
            "invalid_trial_count": int(result.invalid_trial_count),
        }
