"""Core contracts and helpers."""

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.registry import Registry
from plasma_surrogate.core.run_bundle import RunBundle, RunBundleLoader, require_artifacts
from plasma_surrogate.core.synthetic_data import SyntheticDataset, build_synthetic_dataset
from plasma_surrogate.core.task_spec import TaskSpecV1

__all__ = [
    "ArtifactStore",
    "Registry",
    "RunBundle",
    "RunBundleLoader",
    "require_artifacts",
    "SyntheticDataset",
    "TaskSpecV1",
    "build_synthetic_dataset",
]
