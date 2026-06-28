from __future__ import annotations

import csv

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.train.artifact_writers import (
    save_numpy_optimization_diagnostics,
    save_torch_optimization_diagnostics,
)


def _read_header(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def test_numpy_trainer_optimization_diagnostics_use_dynamic_header(tmp_path):
    save_numpy_optimization_diagnostics(
        store=ArtifactStore(tmp_path),
        rows=[
            {
                "epoch": 0.0,
                "grad_l2_total": 1.0,
                "selection_score_density": 0.5,
            }
        ],
    )

    header = _read_header(tmp_path / "scalars" / "optimization_diagnostics.csv")
    assert "selection_score_density" in header
    assert "region_balance_applied" not in header
    assert "boundary_weight_effective_ratio" not in header
    assert "loss_boundary_in" not in header


def test_torch_trainer_optimization_diagnostics_use_dynamic_header(tmp_path):
    save_torch_optimization_diagnostics(
        store=ArtifactStore(tmp_path),
        rows=[
            {
                "epoch": 0.0,
                "grad_l2_total": 1.0,
                "custom_diag": 2.0,
            }
        ],
    )

    header = _read_header(tmp_path / "scalars" / "optimization_diagnostics.csv")
    assert "custom_diag" in header
    assert "head_refresh_applied" not in header
