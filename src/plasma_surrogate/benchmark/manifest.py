"""Benchmark manifest writer."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.contracts import build_product_manifest


class BenchmarkManifestWriter:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def save_manifest(
        self,
        *,
        split: dict[str, Any],
        resolved: dict[str, Any],
        include_manifest: bool = False,
    ) -> None:
        if include_manifest:
            manifest = build_product_manifest(split=split, resolved=resolved).as_dict()
            self.store.save_json("manifest.json", manifest)


__all__ = ["BenchmarkManifestWriter"]
