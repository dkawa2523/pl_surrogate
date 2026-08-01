"""Benchmark metric table writers."""

from __future__ import annotations

from typing import Any

from plasma_surrogate.benchmark.planning import build_core_leaderboard_header
from plasma_surrogate.core.artifact_store import ArtifactStore


_DIAGNOSTIC_PAYLOAD_KEYS = ("diagnostics", "_diagnostics")


def _split_explicit_diagnostics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for row in rows:
        core = dict(row)
        diagnostics: dict[str, Any] = {}
        for key in _DIAGNOSTIC_PAYLOAD_KEYS:
            raw = core.pop(key, None)
            if isinstance(raw, dict):
                diagnostics.update(raw)
        core_rows.append(core)
        if diagnostics:
            diagnostic_rows.append({"model_id": row.get("model_id", ""), **diagnostics})
    return core_rows, diagnostic_rows


def save_leaderboard_metric_tables(
    *,
    store: ArtifactStore,
    leaderboard: list[dict[str, Any]],
    full_header: list[str],
) -> Any:
    core_rows, diagnostic_rows = _split_explicit_diagnostics(leaderboard)
    core_keys = {key for row in core_rows for key in row.keys()}
    core_header = build_core_leaderboard_header(full_header=full_header, core_keys=core_keys)
    leaderboard_path = store.save_csv(
        "leaderboard.csv",
        core_header,
        [[row.get(h, "") for h in core_header] for row in (core_rows or leaderboard)],
    )
    store.save_csv("core_metrics.csv", core_header, [[row.get(h, "") for h in core_header] for row in core_rows])
    if diagnostic_rows:
        diagnostic_header = sorted({key for row in diagnostic_rows for key in row.keys()})
        store.save_csv(
            "diagnostics/diagnostics.csv",
            diagnostic_header,
            [[row.get(h, "") for h in diagnostic_header] for row in diagnostic_rows],
        )
    return leaderboard_path


__all__ = ["save_leaderboard_metric_tables"]
