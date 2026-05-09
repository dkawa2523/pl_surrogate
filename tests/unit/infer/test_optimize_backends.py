from __future__ import annotations

import builtins
import csv
from pathlib import Path

import pytest

from plasma_surrogate.cli.workflows import _resolve_optimize_backend
from plasma_surrogate.infer.optimize import OptimizeRunner, cond_space_from_stats


class _FakeResult:
    def __init__(self, value: float):
        self.qoi = {"uniformity": float(value)}
        self.diagnostics = {"poisson_residual_norm": 1.5}
        self.warnings = []


class _FakeEngine:
    def single_run(self, cond, geom, axis):  # noqa: ANN001
        return _FakeResult(sum(float(v) for v in cond.values()))

    def single_run_aggregated(self, cond, geom, axis):  # noqa: ANN001
        return self.single_run(cond=cond, geom=geom, axis=axis)


def test_optimize_runner_random_backend_reproducible():
    runner = OptimizeRunner(_FakeEngine())
    kwargs = {
        "space": {"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        "geom_space": {},
        "n_trials": 6,
        "geom_ref": {"geom_id": "default"},
        "axis": {"mode": "steady", "value": 0.0},
        "seed": 7,
        "backend": "random",
    }
    a = runner.run(**kwargs)
    b = runner.run(**kwargs)
    assert a.best_cond == b.best_cond
    assert a.best_value == b.best_value
    assert len(a.trials) == 6
    assert a.backend == "random"
    assert a.objective_key == "uniformity"


def test_optimize_runner_unknown_backend_raises():
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(ValueError, match="Unsupported optimize backend"):
        runner.run(
            space={"c0": (0.0, 1.0)},
            geom_space={},
            n_trials=2,
            geom_ref={"geom_id": "default"},
            backend="unknown_backend",
        )


def test_optimize_runner_rejects_zero_trials():
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        runner.run(
            space={"c0": (0.0, 1.0)},
            geom_space={},
            n_trials=0,
            geom_ref={"geom_id": "default"},
            backend="random",
        )


def test_optimize_runner_optuna_unavailable_raises(monkeypatch):
    real_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "optuna":
            raise ImportError("optuna not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(RuntimeError, match="optuna backend is not available"):
        runner.run(
            space={"c0": (0.0, 1.0)},
            geom_space={},
            n_trials=2,
            geom_ref={"geom_id": "default"},
            backend="optuna",
        )


def test_resolve_optimize_backend_legacy_sampler_mapping():
    assert _resolve_optimize_backend({"sampler": "random"}) == "random"
    assert _resolve_optimize_backend({"sampler": "optuna_grid"}) == "optuna"
    assert _resolve_optimize_backend({"sampler": "csv"}) == "csv"


def test_optimize_runner_csv_backend(tmp_path: Path):
    csv_path = tmp_path / "candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["c0", "c1"])
        writer.writeheader()
        writer.writerow({"c0": "0.2", "c1": "0.3"})
        writer.writerow({"c0": "0.1", "c1": "0.1"})
        writer.writerow({"c0": "0.1", "c1": "0.1"})  # duplicate

    runner = OptimizeRunner(_FakeEngine())
    out = runner.run(
        space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        geom_space={},
        n_trials=10,
        geom_ref={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
        backend="csv",
        backend_cfg={"csv_path": str(csv_path), "deduplicate": True},
    )
    assert out.backend == "csv"
    assert out.best_cond == {"c0": 0.1, "c1": 0.1}
    assert len(out.trials) == 2
    assert out.trials[0]["poisson_residual_norm"] == pytest.approx(1.5)


def test_cond_space_from_stats_uses_preprocessing_bounds():
    space = cond_space_from_stats(
        ["PP0", "Td"],
        {"PP0": {"min": 1.0, "max": 5.0}, "Td": {"min": 0.03, "max": 0.3}},
    )
    assert space == {"PP0": (1.0, 5.0), "Td": (0.03, 0.3)}


def test_cond_space_from_stats_requires_complete_bounds():
    with pytest.raises(ValueError, match="cond_stats is missing bounds"):
        cond_space_from_stats(["PP0", "Td"], {"PP0": {"min": 1.0, "max": 5.0}})
