from __future__ import annotations

import json

from plasma_surrogate.benchmark.manifest import BenchmarkManifestWriter
from plasma_surrogate.core.artifact_store import ArtifactStore


def test_benchmark_manifest_writer_uses_nested_manifest_without_flat_resolved(tmp_path):
    writer = BenchmarkManifestWriter(ArtifactStore(tmp_path))
    writer.save_manifest(
        split={"train": ["a"]},
        resolved={
            "input_mode_effective": "table_only",
            "loss_protocol_effective": "plasma_surrogate_v2",
            "artifact_hashes": {"lock_hash": "abc"},
        },
        include_manifest=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["input_mode"]["input_mode_effective"] == "table_only"
    assert manifest["training"]["loss_protocol_effective"] == "plasma_surrogate_v2"
    assert manifest["artifacts"]["split"] == {"train": ["a"]}
    assert manifest["artifacts"]["artifact_hashes"]["lock_hash"] == "abc"
    assert "resolved" not in manifest
    assert "split" not in manifest
    assert "compatibility" not in manifest
    assert not (tmp_path / "resolved_benchmark.json").exists()
