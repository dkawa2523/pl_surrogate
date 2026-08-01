"""Shared training diagnostics writers."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.core.artifact_store import ArtifactStore


def save_optimization_diagnostics(
    *,
    store: ArtifactStore,
    rows: list[dict[str, Any]],
    priority: list[str],
) -> None:
    if not rows:
        return
    available = {str(k) for row in rows for k in row.keys()}
    header = [key for key in priority if key in available]
    header.extend(sorted(key for key in available if key not in set(header)))
    data = [[row.get(key, 0.0) for key in header] for row in rows]
    store.save_csv("scalars/optimization_diagnostics.csv", header, data)


__all__ = ["save_optimization_diagnostics"]
