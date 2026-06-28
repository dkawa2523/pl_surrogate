"""Inference engine for mainline surrogate models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.contracts import (
    validate_geom_deeponet_siren_descriptor_contract,
    validate_pod_descriptor_latent_contract,
)
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DESCRIPTOR_PROFILE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    INPUT_MODE_EFFECTIVE_KEY,
    INPUT_MODES,
    LATENT_PROFILE_KEY,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
)
from plasma_surrogate.core.model_families import POD_DEEPONET_FAMILY_MODELS
from plasma_surrogate.core.vector_pack import load_vector_from_pack
from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys
from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.data.geometry_provider import GeometryProviderLike
from plasma_surrogate.features.structure_descriptors import build_structure_descriptor
from plasma_surrogate.infer.batch import aggregate_inference_results
from plasma_surrogate.infer.contracts import (
    normalize_effective_meta_dict,
    resolve_model_type_name,
    validate_inference_runtime_metadata,
    validate_model_input_mode_contract,
    validate_output_head_metadata_contract,
)
from plasma_surrogate.infer.derived_fields import (
    compute_configured_derived_fields,
    compute_default_derived_fields,
)
from plasma_surrogate.infer.diagnostics import collect_inference_diagnostics
from plasma_surrogate.infer.features import InferenceFeatureBuilder
from plasma_surrogate.infer.optimize import OptimizeRunner, validate_optimize_geom_contract
from plasma_surrogate.infer.output_writer import diagnostics_maps_enabled, write_single_run_outputs
from plasma_surrogate.infer.predictor import InferencePredictor
from plasma_surrogate.infer.qoi import compute_inference_qoi
from plasma_surrogate.models.deeponet.geom_deeponet_siren import GeomDeepONetSIREN
from plasma_surrogate.models.deeponet.pod_deeponet_torch import PODDeepONetTorch
from plasma_surrogate.models.heads.plasma_head import PlasmaHead
from plasma_surrogate.preprocessing.scalers import TransformBundle
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _config_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    try:
        values = list(raw)
    except TypeError:
        values = [raw]
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _config_bool(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean config value: {raw!r}")


def _normalize_derived_fields_config(raw: Any) -> tuple[list[dict[str, Any]] | None, bool | None]:
    if raw is None:
        return None, None
    strict: bool | None = None
    items = raw
    if isinstance(raw, dict):
        if "strict" in raw:
            strict = _config_bool(raw.get("strict"))
        for key in ("items", "fields", "definitions"):
            if key in raw:
                items = raw.get(key)
                break
        else:
            raise ValueError("inference.derived_fields dict must contain items, fields, or definitions")
    if isinstance(items, (str, bytes)):
        raise ValueError("inference.derived_fields must be a list of field configs")
    try:
        values = list(items)
    except TypeError as exc:
        raise ValueError("inference.derived_fields must be a list of field configs") from exc
    out: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("each inference.derived_fields item must be a mapping")
        out.append(dict(item))
    return out, strict


@dataclass
class InferenceResult:
    fields_model: dict[str, np.ndarray]
    fields_phys: dict[str, np.ndarray]
    derived: dict[str, np.ndarray]
    qoi: dict[str, float]
    diagnostics: dict[str, Any]
    case_key: str = ""


class InferenceEngine:
    @staticmethod
    def _postprocess_positive_fields(
        fields: dict[str, np.ndarray],
        *,
        positive_vars: list[str] | None,
        floor: float,
    ) -> dict[str, np.ndarray]:
        names = {str(v) for v in (positive_vars or [])}
        if not names:
            return fields
        out: dict[str, np.ndarray] = {}
        floor_value = float(floor)
        if not np.isfinite(floor_value):
            raise ValueError("postprocess.positive_floor must be finite")
        for key, value in fields.items():
            arr = np.asarray(value, dtype=np.float32)
            if key in names:
                arr = np.where(np.isfinite(arr), arr, floor_value).astype(np.float32)
                arr = np.maximum(arr, floor_value).astype(np.float32)
            out[key] = arr
        return out

    def _combined_physics_symbols(self) -> dict[str, Any]:
        bo_cfg = _dict_or_empty(self.ood_cfg.get("boundary_operator"))
        physics_cfg = _dict_or_empty(self.ood_cfg.get("physics"))
        symbols: dict[str, Any] = {}
        for raw in [physics_cfg.get("symbols"), bo_cfg.get("symbols")]:
            if isinstance(raw, dict):
                symbols.update({str(k): v for k, v in raw.items()})
        return symbols

    def _resolve_physics_keys(
        self,
        fields_phys: dict[str, np.ndarray],
        *,
        required: tuple[str, ...],
        context: str,
    ) -> dict[str, str]:
        return resolve_physics_symbol_keys(
            [str(k) for k in fields_phys.keys()],
            symbols=self._combined_physics_symbols(),
            target_role_schema=self.target_role_schema,
            required=required,
            context=context,
        )

    def _resolve_boundary_operator_inputs(self, fields_phys: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        bo_cfg = _dict_or_empty(self.ood_cfg.get("boundary_operator"))
        physics_cfg = _dict_or_empty(self.ood_cfg.get("physics"))
        physics_enabled = bool(physics_cfg.get("enabled", False))
        boundary_enabled = bool(bo_cfg.get("enabled", False))
        if not (physics_enabled or boundary_enabled):
            return None
        resolved = self._resolve_physics_keys(
            fields_phys,
            required=("density", "temperature", "potential"),
            context="inference physics",
        )
        density_key = resolved["density"]
        temperature_key = resolved["temperature"]
        potential_key = resolved["potential"]
        return {
            "density": np.asarray(fields_phys[density_key], dtype=np.float32),
            "temperature": np.asarray(fields_phys[temperature_key], dtype=np.float32),
            "potential": np.asarray(fields_phys[potential_key], dtype=np.float32),
        }

    def _resolve_potential_field(self, fields_phys: dict[str, np.ndarray]) -> np.ndarray:
        resolved = self._resolve_physics_keys(
            fields_phys,
            required=("potential",),
            context="inference diagnostics",
        )
        return np.asarray(fields_phys[resolved["potential"]], dtype=np.float32)

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
        input_mode_meta: dict[str, Any] | None = None,
        checkpoint_input_mode_meta: dict[str, Any] | None = None,
        checkpoint_meta: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
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
        self.derived_fields_cfg, derived_fields_strict = _normalize_derived_fields_config(
            self.ood_cfg.get("derived_fields")
        )
        if derived_fields_strict is None:
            derived_fields_strict = _config_bool(self.ood_cfg.get("derived_fields_strict"), default=False)
        self.derived_fields_strict = bool(derived_fields_strict)
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
        self.feature_builder = InferenceFeatureBuilder(
            geometry_provider=self.geometry_provider,
            coord_scaler=self.coord_scaler,
            coord_feature_scaler=self.coord_feature_scaler,
            coord_feature_pack=self.coord_feature_pack,
            coord_distance_transform_stats=self.coord_distance_transform_stats,
            coord_input_scaling_cfg=self.coord_input_scaling_cfg,
            coord_input_features_cfg=self.coord_input_features_cfg,
            grid_input_features_cfg=self.grid_input_features_cfg,
        )
        self.predictor = InferencePredictor(model=self.model, feature_builder=self.feature_builder)
        self.input_mode = str(input_mode or TABLE_PLUS_STRUCTURE).strip().lower()
        if self.input_mode not in set(INPUT_MODES):
            raise ValueError(f"inference input_mode must be one of {list(INPUT_MODES)}; got={self.input_mode!r}")
        self.input_mode_meta = normalize_effective_meta_dict(input_mode_meta)
        self._checkpoint_input_mode_meta_provided = checkpoint_input_mode_meta is not None
        self.checkpoint_input_mode_meta = normalize_effective_meta_dict(checkpoint_input_mode_meta)
        self.checkpoint_meta = dict(checkpoint_meta or {})
        self.target_role_schema = dict(target_role_schema or {})
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
            raise ValueError(
                f"inference input_mode: key={INPUT_MODE_EFFECTIVE_KEY!r}, "
                f"expected={self.input_mode!r}, got={resolved_mode!r}"
            )
        if str(self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY]).strip().lower() != self.input_mode:
            raise ValueError(
                f"inference input_mode: key={INPUT_MODE_EFFECTIVE_KEY!r}, "
                f"expected={self.input_mode!r}, got={self.input_mode_meta[INPUT_MODE_EFFECTIVE_KEY]!r}"
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
                "inference provider mode: "
                f"key={GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY!r}, "
                f"expected={self.provider_mode_effective!r}, got={provider_mode_runtime!r}"
            )
        self.structure_adapter_mode_effective = str(
            self.input_mode_meta.get(STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY, "auto")
        ).strip().lower() or "auto"
        self.structure_descriptor_profile_effective = str(
            self.input_mode_meta.get(DESCRIPTOR_PROFILE_KEY, "none")
        ).strip().lower() or "none"
        self.structure_latent_profile_effective = str(
            self.input_mode_meta.get(LATENT_PROFILE_KEY, "none")
        ).strip().lower() or "none"
        self.pod_descriptor_dim_effective = int(self.checkpoint_meta.get(DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY, 0) or 0)
        self.pod_latent_hook_effective = bool(self.checkpoint_meta.get(DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY, False))
        self._validate_model_input_mode_contract()
        validate_output_head_metadata_contract(
            model=self.model,
            checkpoint_meta=self.checkpoint_meta,
            target_role_schema=self.target_role_schema,
        )
        self._validate_pod_descriptor_and_latent_contract()
        self._validate_geom_deeponet_siren_descriptor_contract()

    def _validate_input_mode_metadata_contract(self) -> None:
        if not self._checkpoint_input_mode_meta_provided:
            return
        self.input_mode_meta = validate_inference_runtime_metadata(
            request_meta=self.input_mode_meta,
            checkpoint_meta=self.checkpoint_input_mode_meta,
            context="inference runtime_metadata",
        )

    def _validate_model_input_mode_contract(self) -> None:
        validate_model_input_mode_contract(model=self.model, input_mode=self.input_mode)

    def _is_pod_model(self) -> bool:
        if isinstance(self.model, PODDeepONetTorch):
            return True
        model_type = resolve_model_type_name(self.model)
        return model_type in POD_DEEPONET_FAMILY_MODELS

    def _validate_pod_descriptor_and_latent_contract(self) -> None:
        if not self._is_pod_model():
            return
        meta = validate_pod_descriptor_latent_contract(
            input_mode=self.input_mode,
            adapter_mode=self.structure_adapter_mode_effective,
            descriptor_profile=self.structure_descriptor_profile_effective,
            latent_profile=self.structure_latent_profile_effective,
            provider_mode=self.provider_mode_effective,
            checkpoint_meta=self.checkpoint_meta,
            require_checkpoint_metadata=True,
        )
        self.pod_descriptor_dim_effective = int(meta[DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY])
        self.pod_latent_hook_effective = bool(meta[DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY])
        if self.pod_latent_hook_effective:
            load_vector_from_pack(
                pack=self.latent_feature_pack,
                pack_name="latent_feature_pack",
            )

    def _is_geom_deeponet_siren_model(self) -> bool:
        if isinstance(self.model, GeomDeepONetSIREN):
            return True
        model_type = resolve_model_type_name(self.model)
        return model_type == "geom_deeponet_siren"

    def _validate_geom_deeponet_siren_descriptor_contract(self) -> None:
        if not self._is_geom_deeponet_siren_model():
            return
        meta = validate_geom_deeponet_siren_descriptor_contract(
            input_mode=self.input_mode,
            adapter_mode=self.structure_adapter_mode_effective,
            descriptor_profile=self.structure_descriptor_profile_effective,
            provider_mode=self.provider_mode_effective,
            checkpoint_meta=self.checkpoint_meta,
            require_checkpoint_metadata=True,
        )
        self.geom_deeponet_siren_descriptor_dim_effective = int(
            meta[GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY]
        )

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

    def _build_cond_vector(self, cond: dict[str, Any], axis: dict[str, Any]) -> np.ndarray:
        base = self.cond_schema.encode(cond)
        axis_vec = self.axis_schema.encode(float(axis.get("value", 0.0)))
        cond_vec = np.concatenate([base, axis_vec], axis=0).astype(np.float32)
        if self.transforms is not None:
            return self.transforms.transform_cond(cond_vec[None, :])[0]
        return cond_vec

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

    def _compute_derived_fields(self, fields_phys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.derived_fields_cfg is not None:
            return compute_configured_derived_fields(
                fields_phys,
                self.derived_fields_cfg,
                symbols=self._combined_physics_symbols(),
                target_role_schema=self.target_role_schema,
                strict=self.derived_fields_strict,
            )
        return compute_default_derived_fields(
            fields_phys,
            symbols=self._combined_physics_symbols(),
            target_role_schema=self.target_role_schema,
            strict=False,
        )

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

    def single_run_aggregated(
        self,
        cond: dict[str, Any],
        geom: dict[str, Any],
        axis: dict[str, Any],
        save_outputs: bool = True,
    ) -> InferenceResult:
        aggregation = str(axis.get("aggregation", "single"))
        axis_samples = self._resolve_axis_samples(axis)
        if aggregation == "single":
            return self.single_run(cond=cond, geom=geom, axis=axis_samples[0], save_outputs=save_outputs)
        runs = [
            self.single_run(cond=cond, geom=geom, axis=sample, save_outputs=save_outputs)
            for sample in axis_samples
        ]
        return aggregate_inference_results(runs, aggregation)

    def _prepare_single_run_case(
        self,
        *,
        cond: dict[str, Any],
        geom: dict[str, Any],
        axis: dict[str, Any],
    ) -> tuple[dict[str, Any], GeometryContext, np.ndarray]:
        geom_ref = self._validate_geom_ref(geom)
        geom_ctx = self._get_geom_ctx(geom_ref, axis=axis)
        cond_vec = self._build_cond_vector(cond, axis)
        cond_vec = self._augment_pod_cond_vector(cond_vec, geom_ref=geom_ref, geom_ctx=geom_ctx)
        cond_vec = self._augment_geom_deeponet_siren_cond_vector(cond_vec, geom_ref=geom_ref, geom_ctx=geom_ctx)
        return geom_ref, geom_ctx, cond_vec

    def _postprocess_fields(
        self,
        *,
        raw_pred: dict[str, np.ndarray],
        geom_ctx: GeometryContext,
        cond_vec: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None, dict[str, np.ndarray]]:
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
        potential_key: str | None = None
        if self.phi_mode == "direct":
            try:
                potential_key = self._resolve_physics_keys(
                    fields_phys,
                    required=("potential",),
                    context="inference potential postprocess",
                )["potential"]
            except ValueError:
                potential_key = None
        else:
            potential_key = self._resolve_physics_keys(
                fields_phys,
                required=("potential",),
                context="inference potential postprocess",
            )["potential"]
        if potential_key is None:
            head_out, head_aux = head_in, {}
        else:
            head_out, head_aux = self.plasma_head.apply(
                head_in,
                geom_ctx=geom_ctx,
                refine_iters=self.poisson_refine_iters,
                deeponet_head=self.deeponet_head,
                cond_vec=cond_vec,
                potential_key=potential_key,
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

        post_cfg = _dict_or_empty(self.ood_cfg.get("postprocess"))
        floor_raw = post_cfg.get("positive_floor")
        if floor_raw is None:
            floor_raw = post_cfg.get("floor")
        if floor_raw is None:
            floor_raw = 1.0e-30
        fields_phys = self._postprocess_positive_fields(
            fields_phys,
            positive_vars=_config_string_list(post_cfg.get("positive_vars")),
            floor=float(floor_raw),
        )
        try:
            potential_field = self._resolve_potential_field(fields_phys)
        except ValueError:
            potential_field = None
        derived = self._compute_derived_fields(fields_phys)
        return fields_model, fields_phys, potential_field, derived

    def single_run(
        self,
        cond: dict[str, Any],
        geom: dict[str, Any],
        axis: dict[str, Any],
        save_outputs: bool = True,
    ) -> InferenceResult:
        geom_ref, geom_ctx, cond_vec = self._prepare_single_run_case(cond=cond, geom=geom, axis=axis)
        raw_pred = self.predictor.predict_fields(cond_vec, geom_ctx)
        fields_model, fields_phys, potential_field, derived = self._postprocess_fields(
            raw_pred=raw_pred,
            geom_ctx=geom_ctx,
            cond_vec=cond_vec,
        )
        qoi, band_mask, op_inputs, bo_cfg = compute_inference_qoi(
            fields_phys=fields_phys,
            geom_ctx=geom_ctx,
            potential_field=potential_field,
            ood_cfg=self.ood_cfg,
            target_role_schema=self.target_role_schema,
            resolve_boundary_operator_inputs=self._resolve_boundary_operator_inputs,
        )
        save_diagnostics_maps = diagnostics_maps_enabled(self.ood_cfg)
        diagnostics, diagnostics_maps = collect_inference_diagnostics(
            fields_phys=fields_phys,
            geom_ctx=geom_ctx,
            potential_field=potential_field,
            band_mask=band_mask,
            op_inputs=op_inputs,
            bo_cfg=bo_cfg,
            include_maps=save_diagnostics_maps,
        )

        case_key = self._case_key(cond=cond, axis=axis, geom=geom_ref)
        if save_outputs:
            write_single_run_outputs(
                store=self.store,
                case_key=case_key,
                fields_model=fields_model,
                fields_phys=fields_phys,
                derived=derived,
                qoi=qoi,
                diagnostics=diagnostics,
                diagnostics_maps=diagnostics_maps,
                save_diagnostics_maps=save_diagnostics_maps,
            )

        return InferenceResult(
            fields_model=fields_model,
            fields_phys=fields_phys,
            derived=derived,
            qoi=qoi,
            diagnostics=diagnostics,
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
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
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
            objective_cfg=objective_cfg,
            constraints_cfg=constraints_cfg,
            output_cfg=output_cfg,
        )
        output_effective = dict(result.output_cfg or {"save_fields": "all", "top_k": 3})
        if output_effective.get("save_fields") == "top_k":
            top_k = int(output_effective.get("top_k", 3))

            def _record_value(record: dict[str, Any]) -> float:
                try:
                    value = float(record.get("objective_value"))
                except (TypeError, ValueError):
                    return float("inf")
                if np.isfinite(value):
                    return value
                return float("inf")

            ranked = sorted(
                result.trials,
                key=lambda r: (0 if bool(r.get("feasible", False)) else 1, _record_value(r)),
            )
            seen: set[str] = set()
            for trial in ranked:
                cond_payload = dict(trial.get("cond", {}) or {})
                geom_param_payload = dict(trial.get("geom_param", {}) or {})
                signature = json.dumps(
                    {"cond": cond_payload, "geom_param": geom_param_payload},
                    sort_keys=True,
                    default=str,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                geom_payload = dict(geom or {"geom_id": "default"})
                if geom_param_payload:
                    geom_payload.pop("geom_param", None)
                    geom_payload["geom_param"] = {
                        str(k): float(v) for k, v in sorted(geom_param_payload.items())
                    }
                self.single_run_aggregated(
                    cond=cond_payload,
                    geom=geom_payload,
                    axis=axis,
                    save_outputs=True,
                )
                if len(seen) >= top_k:
                    break
        self.store.save_json(
            "optimize/best.json",
            {
                "best_cond": result.best_cond,
                "best_geom_param": result.best_geom_param,
                "status": result.status,
                "objective_value": result.objective_value,
                "search_value": result.search_value,
                "feasible": bool(result.feasible),
                "violated_constraints": list(result.violated_constraints),
                "constraint_violation_total": result.constraint_violation_total,
                "objective_mode": result.objective_mode,
            },
        )
        feasible_trial_count = sum(1 for trial in result.trials if bool(trial.get("feasible", False)))
        self.store.save_json(
            "optimize/summary.json",
            {
                "backend": result.backend,
                "backend_cfg": result.backend_cfg,
                "seed": int(seed),
                "n_trials": int(n_trials),
                "status": result.status,
                "objective_mode": result.objective_mode,
                "objective_value": result.objective_value,
                "search_value": result.search_value,
                "feasible": bool(result.feasible),
                "violated_constraints": list(result.violated_constraints),
                "constraint_violation_total": result.constraint_violation_total,
                "geom_space_enabled_effective": bool(len(geom_space_norm) > 0),
                "geom_param_keys_effective": sorted(list(result.best_geom_param.keys())),
                "invalid_trial_count": int(result.invalid_trial_count),
                "feasible_trial_count": int(feasible_trial_count),
                "output": output_effective,
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

        def _csv_value(value: Any) -> Any:
            if value is None:
                return ""
            if isinstance(value, list):
                return ";".join(str(v) for v in value)
            return value

        self.store.save_csv("optimize/trials.csv", header, [[_csv_value(r.get(k)) for k in header] for r in rows])
        return {
            "best_cond": result.best_cond,
            "best_geom_param": result.best_geom_param,
            "status": result.status,
            "objective_value": result.objective_value,
            "search_value": result.search_value,
            "feasible": bool(result.feasible),
            "violated_constraints": list(result.violated_constraints),
            "constraint_violation_total": result.constraint_violation_total,
            "objective_mode": result.objective_mode,
            "invalid_trial_count": int(result.invalid_trial_count),
        }
