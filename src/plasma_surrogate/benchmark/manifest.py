"""Benchmark manifest writer."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.contracts import build_product_manifest


def _run_git(args: list[str], *, cwd: Path) -> bytes | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return bytes(proc.stdout)


def _hash_untracked_files(*, root: Path, paths_raw: bytes) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for raw_path in paths_raw.split(b"\0"):
        if not raw_path:
            continue
        rel = raw_path.decode("utf-8", errors="surrogateescape")
        path = root / rel
        if not path.is_file():
            continue
        count += 1
        digest.update(raw_path)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), count


def collect_source_provenance(*, cwd: Path | None = None) -> dict[str, Any]:
    """Collect enough source/runtime identity to detect non-reproducible runs.

    The manifest deliberately stores hashes of dirty state rather than embedding
    a potentially large or sensitive working-tree diff.
    """

    workdir = Path(cwd or Path.cwd()).resolve()
    root_raw = _run_git(["rev-parse", "--show-toplevel"], cwd=workdir)
    commit_raw = _run_git(["rev-parse", "HEAD"], cwd=workdir)
    status_raw = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=workdir,
    )
    diff_raw = _run_git(["diff", "--binary", "HEAD", "--"], cwd=workdir)
    untracked_raw = _run_git(
        ["ls-files", "-z", "--others", "--exclude-standard"],
        cwd=workdir,
    )
    available = root_raw is not None and commit_raw is not None and status_raw is not None
    payload: dict[str, Any] = {
        "git_available": bool(available),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if available:
        git_root = Path(root_raw.decode("utf-8", errors="replace").strip()).resolve()
        untracked_sha256, untracked_count = _hash_untracked_files(
            root=git_root,
            paths_raw=untracked_raw or b"",
        )
        payload.update(
            {
                "git_root": str(git_root),
                "git_commit": commit_raw.decode("ascii", errors="replace").strip(),
                "git_dirty": bool(status_raw),
                "git_status_sha256": hashlib.sha256(status_raw).hexdigest(),
                "git_tracked_diff_sha256": hashlib.sha256(diff_raw or b"").hexdigest(),
                "git_untracked_content_sha256": untracked_sha256,
                "git_untracked_file_count": int(untracked_count),
            }
        )
    return payload


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
            manifest.setdefault("artifacts", {})["source_provenance"] = collect_source_provenance()
            self.store.save_json("manifest.json", manifest)


__all__ = ["BenchmarkManifestWriter", "collect_source_provenance"]
