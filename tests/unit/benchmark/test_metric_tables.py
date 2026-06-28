from __future__ import annotations

from plasma_surrogate.benchmark.metric_tables import save_leaderboard_metric_tables
from plasma_surrogate.core.artifact_store import ArtifactStore


def test_metric_tables_split_core_diagnostics_without_legacy_full_metrics(tmp_path):
    store = ArtifactStore(tmp_path)
    leaderboard = [
        {
            "model_id": "m",
            "score": 1.0,
            "_diagnostics": {"physics_poisson_residual_norm": 0.2},
        }
    ]

    path = save_leaderboard_metric_tables(
        store=store,
        leaderboard=leaderboard,
        full_header=["model_id", "score", "physics_poisson_residual_norm"],
    )

    assert path == tmp_path / "leaderboard.csv"
    assert (tmp_path / "core_metrics.csv").exists()
    assert (tmp_path / "diagnostics" / "diagnostics.csv").exists()
    assert not (tmp_path / "compat" / "legacy_full_metrics.csv").exists()
    assert not (tmp_path / "legacy_full_metrics.csv").exists()
