"""Small product-contract value objects shared by runtime entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_PENDING_HASH = "pending"
_TABLE_ONLY = "table_only"
_TABLE_PLUS_STRUCTURE = "table_plus_structure"
_INPUT_MODES = {_TABLE_ONLY, _TABLE_PLUS_STRUCTURE}
_ADAPTER_NONE = "none"
_ADAPTER_DESCRIPTOR_BRANCH = "descriptor_branch"
_ADAPTER_HYBRID_PACK_DESCRIPTOR = "hybrid_pack_descriptor"
_RUNTIME_REQUIRED_METADATA_KEYS = (
    "input_mode_effective",
    "structure_feature_profile_effective",
    "structure_adapter_mode_effective",
    "geometry_provider_mode_effective",
    "target_schema_hash",
    "feature_schema_hash",
)
_RUNTIME_OPTIONAL_METADATA_KEYS = ("descriptor_profile", "latent_profile")
_DEEPONET_POD_DESCRIPTOR_DIM_KEY = "deeponet_pod_descriptor_dim_effective"
_DEEPONET_POD_DESCRIPTOR_PROFILE_KEY = "deeponet_pod_descriptor_profile_effective"
_DEEPONET_POD_LATENT_PROFILE_KEY = "deeponet_pod_latent_profile_effective"
_DEEPONET_POD_LATENT_HOOK_KEY = "deeponet_pod_latent_hook_effective"
_GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY = "geom_deeponet_siren_descriptor_dim_effective"
_GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_KEY = "geom_deeponet_siren_descriptor_profile_effective"


def _norm_text(value: Any, *, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    return text if text else default


def _norm_optional_profile(value: Any) -> str | None:
    text = _norm_text(value, default="none")
    return None if text == "none" else text


@dataclass(frozen=True)
class RuntimeContract:
    input_mode: str
    structure_feature_profile: str
    structure_adapter_mode: str
    geometry_provider_mode: str
    target_schema_hash: str
    feature_schema_hash: str
    descriptor_profile: str | None = None
    latent_profile: str | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any] | None,
        *,
        context: str = "runtime_contract",
        allow_pending_schema_hashes: bool = False,
    ) -> "RuntimeContract":
        raw = dict(metadata or {})
        required = {
            "input_mode_effective": "input_mode",
            "structure_feature_profile_effective": "structure_feature_profile",
            "structure_adapter_mode_effective": "structure_adapter_mode",
            "geometry_provider_mode_effective": "geometry_provider_mode",
            "target_schema_hash": "target_schema_hash",
            "feature_schema_hash": "feature_schema_hash",
        }
        missing = [key for key in required if key not in raw or str(raw.get(key, "")).strip() == ""]
        if missing:
            raise ValueError(f"{context}: missing required runtime metadata keys: {missing}")
        target_hash = str(raw["target_schema_hash"]).strip()
        feature_hash = str(raw["feature_schema_hash"]).strip()
        if not allow_pending_schema_hashes and _PENDING_HASH in {target_hash.lower(), feature_hash.lower()}:
            raise ValueError(
                f"{context}: runtime schema hashes must be concrete before train/infer/benchmark; "
                f"target_schema_hash={target_hash!r}, feature_schema_hash={feature_hash!r}"
            )
        input_mode = _norm_text(raw["input_mode_effective"], default="")
        if input_mode not in _INPUT_MODES:
            raise ValueError(f"{context}: input_mode_effective must be one of {sorted(_INPUT_MODES)}; got={input_mode!r}")
        return cls(
            input_mode=input_mode,
            structure_feature_profile=_norm_text(raw["structure_feature_profile_effective"], default="none"),
            structure_adapter_mode=_norm_text(raw["structure_adapter_mode_effective"], default="auto"),
            geometry_provider_mode=_norm_text(raw["geometry_provider_mode_effective"], default="fixed"),
            target_schema_hash=target_hash,
            feature_schema_hash=feature_hash,
            descriptor_profile=_norm_optional_profile(raw.get("descriptor_profile")),
            latent_profile=_norm_optional_profile(raw.get("latent_profile")),
        )

    def to_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "input_mode_effective": self.input_mode,
            "structure_feature_profile_effective": self.structure_feature_profile,
            "structure_adapter_mode_effective": self.structure_adapter_mode,
            "geometry_provider_mode_effective": self.geometry_provider_mode,
            "target_schema_hash": self.target_schema_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }
        if self.descriptor_profile:
            out["descriptor_profile"] = self.descriptor_profile
        if self.latent_profile:
            out["latent_profile"] = self.latent_profile
        return out


def validate_runtime_metadata_pair(
    *,
    request_meta: dict[str, Any] | None,
    checkpoint_meta: dict[str, Any] | None,
    keys: tuple[str, ...] | None = None,
    context: str = "runtime_metadata",
) -> dict[str, Any]:
    """Validate request/checkpoint runtime metadata through RuntimeContract."""

    req = dict(request_meta or {})
    ckpt = dict(checkpoint_meta or {})
    if not ckpt:
        return req
    required = tuple(keys or _RUNTIME_REQUIRED_METADATA_KEYS)
    optional = tuple(key for key in _RUNTIME_OPTIONAL_METADATA_KEYS if key in req or key in ckpt)
    for key in (*required, *optional):
        if key not in ckpt:
            raise ValueError(f"{context}: key={key!r}, expected=<missing>, got={req.get(key)!r}")
        if key not in req:
            raise ValueError(f"{context}: key={key!r}, expected={ckpt.get(key)!r}, got=<missing>")
        if req[key] != ckpt[key]:
            raise ValueError(f"{context}: key={key!r}, expected={ckpt[key]!r}, got={req[key]!r}")

    if set(_RUNTIME_REQUIRED_METADATA_KEYS).issubset(req) and set(_RUNTIME_REQUIRED_METADATA_KEYS).issubset(ckpt):
        req_contract = RuntimeContract.from_metadata(
            req,
            context=f"{context} request",
            allow_pending_schema_hashes=True,
        )
        ckpt_contract = RuntimeContract.from_metadata(
            ckpt,
            context=f"{context} checkpoint",
            allow_pending_schema_hashes=True,
        )
        if req_contract != ckpt_contract:
            raise ValueError(
                f"{context}: runtime contract mismatch; "
                f"expected={ckpt_contract.to_metadata()!r}, got={req_contract.to_metadata()!r}"
            )
    return req


def _pod_table_only_metadata(*, desc_profile: str, lat_profile: str, adapter: str) -> dict[str, Any]:
    if desc_profile != "none" or lat_profile != "none":
        raise ValueError(
            "runtime.input_mode=table_only requires descriptor_profile/latent_profile to be none for deeponet_pod"
        )
    if adapter != _ADAPTER_NONE:
        raise ValueError("runtime.input_mode=table_only requires adapter_mode_effective='none' for deeponet_pod")
    return {
        _DEEPONET_POD_DESCRIPTOR_DIM_KEY: 0,
        _DEEPONET_POD_DESCRIPTOR_PROFILE_KEY: "none",
        _DEEPONET_POD_LATENT_PROFILE_KEY: "none",
        _DEEPONET_POD_LATENT_HOOK_KEY: False,
    }


def _validate_pod_descriptor_lane(
    *,
    desc_profile: str,
    adapter: str,
    provider: str | None,
    ckpt: dict[str, Any],
    require_checkpoint_metadata: bool,
) -> int:
    uses_descriptor_adapter = adapter in {_ADAPTER_DESCRIPTOR_BRANCH, _ADAPTER_HYBRID_PACK_DESCRIPTOR}
    descriptor_dim = int(ckpt.get(_DEEPONET_POD_DESCRIPTOR_DIM_KEY, 0) or 0)
    if desc_profile == "none" and uses_descriptor_adapter:
        raise ValueError("adapter_mode_effective requires runtime.structure.descriptor_profile != none for deeponet_pod")
    if desc_profile == "none":
        return 0
    if not uses_descriptor_adapter:
        raise ValueError(
            "deeponet_pod descriptor profile requires adapter_mode_effective in "
            "{descriptor_branch, hybrid_pack_descriptor}"
        )
    if provider is not None and provider != "parametric_parts":
        raise ValueError(
            "deeponet_pod descriptor lane requires geometry provider_mode_effective='parametric_parts' "
            f"for inference; got={provider!r}"
        )
    ckpt_descriptor_profile = _norm_text(ckpt.get(_DEEPONET_POD_DESCRIPTOR_PROFILE_KEY, desc_profile))
    if ckpt_descriptor_profile and ckpt_descriptor_profile != desc_profile:
        raise ValueError(
            "descriptor profile mismatch between runtime request and checkpoint metadata: "
            f"request={desc_profile!r}, checkpoint={ckpt_descriptor_profile!r}"
        )
    if require_checkpoint_metadata and _DEEPONET_POD_DESCRIPTOR_DIM_KEY not in ckpt:
        raise ValueError(
            "checkpoint metadata missing required key for descriptor lane: "
            f"{_DEEPONET_POD_DESCRIPTOR_DIM_KEY!r}"
        )
    if require_checkpoint_metadata and descriptor_dim <= 0:
        raise ValueError(
            "checkpoint descriptor metadata is invalid: "
            f"{_DEEPONET_POD_DESCRIPTOR_DIM_KEY}={descriptor_dim!r}"
        )
    return int(descriptor_dim)


def _validate_pod_latent_lane(*, lat_profile: str, ckpt: dict[str, Any]) -> bool:
    if lat_profile == "none":
        return False
    ckpt_latent_profile = _norm_text(ckpt.get(_DEEPONET_POD_LATENT_PROFILE_KEY, lat_profile))
    if ckpt_latent_profile and ckpt_latent_profile != lat_profile:
        raise ValueError(
            "latent profile mismatch between runtime request and checkpoint metadata: "
            f"request={lat_profile!r}, checkpoint={ckpt_latent_profile!r}"
        )
    if _DEEPONET_POD_LATENT_HOOK_KEY in ckpt and not bool(ckpt[_DEEPONET_POD_LATENT_HOOK_KEY]):
        raise ValueError("checkpoint metadata indicates latent hook disabled while runtime latent profile is enabled")
    return True


def validate_pod_descriptor_latent_contract(
    *,
    input_mode: Any,
    adapter_mode: Any,
    descriptor_profile: Any,
    latent_profile: Any,
    provider_mode: Any | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
    require_checkpoint_metadata: bool = False,
) -> dict[str, Any]:
    """Validate the DeepONet-POD descriptor/latent runtime lane."""

    mode = _norm_text(input_mode, default=_TABLE_PLUS_STRUCTURE)
    adapter = _norm_text(adapter_mode, default="auto")
    desc_profile = _norm_text(descriptor_profile, default="none")
    lat_profile = _norm_text(latent_profile, default="none")
    provider = _norm_text(provider_mode, default="fixed") if provider_mode is not None else None
    ckpt = dict(checkpoint_meta or {})

    if mode == _TABLE_ONLY:
        return _pod_table_only_metadata(desc_profile=desc_profile, lat_profile=lat_profile, adapter=adapter)
    if mode != _TABLE_PLUS_STRUCTURE:
        raise ValueError(f"unsupported runtime input mode for deeponet_pod: {mode!r}")

    descriptor_dim = _validate_pod_descriptor_lane(
        desc_profile=desc_profile,
        adapter=adapter,
        provider=provider,
        ckpt=ckpt,
        require_checkpoint_metadata=require_checkpoint_metadata,
    )
    latent_hook = _validate_pod_latent_lane(lat_profile=lat_profile, ckpt=ckpt)
    return {
        _DEEPONET_POD_DESCRIPTOR_DIM_KEY: int(descriptor_dim),
        _DEEPONET_POD_DESCRIPTOR_PROFILE_KEY: desc_profile,
        _DEEPONET_POD_LATENT_PROFILE_KEY: lat_profile,
        _DEEPONET_POD_LATENT_HOOK_KEY: bool(latent_hook),
    }


def validate_geom_deeponet_siren_descriptor_contract(
    *,
    input_mode: Any,
    adapter_mode: Any,
    descriptor_profile: Any,
    provider_mode: Any,
    checkpoint_meta: dict[str, Any] | None = None,
    require_checkpoint_metadata: bool = False,
) -> dict[str, Any]:
    """Validate the Geom-DeepONet-SIREN descriptor runtime lane."""

    mode = _norm_text(input_mode, default=_TABLE_PLUS_STRUCTURE)
    adapter = _norm_text(adapter_mode, default="auto")
    desc_profile = _norm_text(descriptor_profile, default="none")
    provider = _norm_text(provider_mode, default="fixed")
    ckpt = dict(checkpoint_meta or {})
    if mode != _TABLE_PLUS_STRUCTURE:
        raise ValueError("geom_deeponet_siren requires runtime.input_mode=table_plus_structure for inference")
    if adapter != _ADAPTER_HYBRID_PACK_DESCRIPTOR:
        raise ValueError(
            "geom_deeponet_siren requires runtime.structure.adapter_mode_effective="
            "'hybrid_pack_descriptor' for inference"
        )
    if desc_profile == "none":
        raise ValueError("geom_deeponet_siren requires runtime.structure.descriptor_profile != none for inference")
    if provider != "parametric_parts":
        raise ValueError(
            "geom_deeponet_siren descriptor lane requires geometry provider_mode_effective='parametric_parts' "
            f"for inference; got={provider!r}"
        )
    ckpt_descriptor_profile = _norm_text(ckpt.get(_GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_KEY, desc_profile))
    if ckpt_descriptor_profile and ckpt_descriptor_profile != desc_profile:
        raise ValueError(
            "geom_deeponet_siren descriptor profile mismatch between runtime request and checkpoint metadata: "
            f"request={desc_profile!r}, checkpoint={ckpt_descriptor_profile!r}"
        )
    if require_checkpoint_metadata and _GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY not in ckpt:
        raise ValueError(
            "checkpoint metadata missing required key for geom_deeponet_siren descriptor lane: "
            f"{_GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY!r}"
        )
    descriptor_dim = int(ckpt.get(_GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY, 0) or 0)
    if require_checkpoint_metadata and descriptor_dim <= 0:
        raise ValueError(
            "checkpoint descriptor metadata is invalid for geom_deeponet_siren: "
            f"{_GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY}={descriptor_dim!r}"
        )
    return {
        _GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_KEY: int(descriptor_dim),
        _GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_KEY: desc_profile,
    }


@dataclass(frozen=True)
class EvaluationProtocol:
    mode: str = "single"
    primary_split: str = "interp"
    primary_metric: str = "surrogate_quality_score"
    objective_mode: str = "min"
    target_vars: tuple[str, ...] = ()
    region_bands: dict[str, Any] = field(default_factory=dict)
    scope: str = "common"
    interp_weight: float = 0.5
    extrap_weight: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scope": self.scope,
            "primary_split": self.primary_split,
            "primary_metric": self.primary_metric,
            "objective_mode": self.objective_mode,
            "target_vars": list(self.target_vars),
            "region_bands": dict(self.region_bands),
            "interp_weight": float(self.interp_weight),
            "extrap_weight": float(self.extrap_weight),
        }


@dataclass(frozen=True)
class TrainArtifacts:
    run_dir: Path
    checkpoint_dir: Path
    eval_dir: Path
    split_name: str
    runtime_contract: RuntimeContract

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "eval_dir": str(self.eval_dir),
            "split_name": self.split_name,
            "runtime_contract": self.runtime_contract.to_metadata(),
        }


@dataclass(frozen=True)
class ProductManifest:
    """Structured product manifest for train/benchmark/infer artifacts."""

    input_mode: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mode": dict(self.input_mode),
            "model": dict(self.model),
            "physics": dict(self.physics),
            "training": dict(self.training),
            "artifacts": dict(self.artifacts),
        }


def build_product_manifest(
    *,
    split: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
) -> ProductManifest:
    """Build the nested product manifest from an existing resolved payload."""

    payload = dict(resolved or {})
    input_mode_keys = {
        "input_mode_effective",
        "structure_feature_profile_effective",
        "structure_adapter_mode_effective",
        "geometry_provider_mode_effective",
        "target_schema_hash",
        "feature_schema_hash",
        "descriptor_profile",
        "latent_profile",
    }
    training_keys = {
        "cv",
        "effective_steps_per_model",
        "loss_protocol_effective",
        "loss_protocol_version",
        "loss_protocol_definition_hash",
        "loss_protocol_effective_config",
    }
    physics_prefixes = ("physics_",)
    physics_keys = {
        "diagnostics_effective",
        "spatial_error_audit_effective",
        "spatial_error_boundary_type_breakdown_effective",
        "spatial_error_region_bands_effective",
    }
    model_contracts: dict[str, Any] = {}
    raw_model_contracts = payload.get("model_contracts")
    if isinstance(raw_model_contracts, dict):
        model_contracts.update({str(key): value for key, value in raw_model_contracts.items()})
    model = {"contracts": model_contracts} if model_contracts else {}
    physics = {
        key: value
        for key, value in payload.items()
        if key in physics_keys or any(str(key).startswith(prefix) for prefix in physics_prefixes)
    }
    artifacts: dict[str, Any] = {"split": dict(split or {})}
    if "split" in payload:
        artifacts["split_config"] = payload["split"]
    if "artifact_hashes" in payload:
        artifacts["artifact_hashes"] = payload["artifact_hashes"]
    return ProductManifest(
        input_mode={key: payload[key] for key in sorted(input_mode_keys) if key in payload},
        model=model,
        physics=physics,
        training={key: payload[key] for key in sorted(training_keys) if key in payload},
        artifacts=artifacts,
    )


__all__ = [
    "EvaluationProtocol",
    "ProductManifest",
    "RuntimeContract",
    "TrainArtifacts",
    "build_product_manifest",
    "validate_geom_deeponet_siren_descriptor_contract",
    "validate_pod_descriptor_latent_contract",
    "validate_runtime_metadata_pair",
]
