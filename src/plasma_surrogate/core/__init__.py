"""Core contracts and helpers."""

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.contracts import (
    EvaluationProtocol,
    ProductManifest,
    RuntimeContract,
    TrainArtifacts,
    build_product_manifest,
    validate_geom_deeponet_siren_descriptor_contract,
    validate_pod_descriptor_latent_contract,
    validate_runtime_metadata_pair,
)
from plasma_surrogate.core.input_modes import (
    DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY,
    DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY,
    EFFECTIVE_RUNTIME_METADATA_KEYS,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY,
    GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY,
    INPUT_MODES,
    TABLE_ONLY,
    TABLE_PLUS_STRUCTURE,
    build_input_mode_effective_metadata,
    descriptor_latent_metadata_keys,
    extract_descriptor_latent_metadata,
    extract_input_mode_metadata,
    merge_effective_runtime_metadata,
    normalize_input_mode_cfg,
    validate_input_mode_cfg,
)
from plasma_surrogate.core.model_input_policy import (
    resolve_allowed_adapter_modes,
    resolve_effective_adapter_mode,
    resolve_effective_input_mode_metadata_for_model,
    resolve_supported_input_modes,
    validate_adapter_mode,
    validate_model_mode_adapter_policy,
    validate_model_input_mode,
)
from plasma_surrogate.core.model_specs import ModelSpec, get_model_spec
from plasma_surrogate.core.registry import Registry
from plasma_surrogate.core.run_bundle import RunBundle, RunBundleLoader, require_artifacts
from plasma_surrogate.core.synthetic_data import SyntheticDataset, build_synthetic_dataset
from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups
from plasma_surrogate.core.task_spec import TaskSpecV1

__all__ = [
    "ArtifactStore",
    "DEEPONET_POD_DESCRIPTOR_DIM_EFFECTIVE_KEY",
    "DEEPONET_POD_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "DEEPONET_POD_LATENT_HOOK_EFFECTIVE_KEY",
    "DEEPONET_POD_LATENT_PROFILE_EFFECTIVE_KEY",
    "EFFECTIVE_RUNTIME_METADATA_KEYS",
    "EvaluationProtocol",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_DIM_EFFECTIVE_KEY",
    "GEOM_DEEPONET_SIREN_DESCRIPTOR_PROFILE_EFFECTIVE_KEY",
    "INPUT_MODES",
    "Registry",
    "ProductManifest",
    "RuntimeContract",
    "RunBundle",
    "RunBundleLoader",
    "TABLE_ONLY",
    "TABLE_PLUS_STRUCTURE",
    "build_input_mode_effective_metadata",
    "descriptor_latent_metadata_keys",
    "extract_descriptor_latent_metadata",
    "extract_input_mode_metadata",
    "merge_effective_runtime_metadata",
    "ModelSpec",
    "require_artifacts",
    "normalize_input_mode_cfg",
    "resolve_allowed_adapter_modes",
    "resolve_effective_adapter_mode",
    "resolve_effective_input_mode_metadata_for_model",
    "resolve_supported_input_modes",
    "resolve_target_groups",
    "get_model_spec",
    "SyntheticDataset",
    "TargetGroup",
    "TaskSpecV1",
    "TrainArtifacts",
    "build_product_manifest",
    "build_synthetic_dataset",
    "validate_adapter_mode",
    "validate_geom_deeponet_siren_descriptor_contract",
    "validate_input_mode_cfg",
    "validate_model_mode_adapter_policy",
    "validate_model_input_mode",
    "validate_pod_descriptor_latent_contract",
    "validate_runtime_metadata_pair",
]
