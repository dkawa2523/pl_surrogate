from __future__ import annotations

import builtins
import csv
from pathlib import Path

import pytest

from plasma_surrogate.infer.optimize import OptimizeRunner, cond_space_from_stats
from plasma_surrogate.cli.workflows import _resolve_optimize_backend


class _FakeResult:
    def __init__(self, value: float):
        self.qoi = {"uniformity": float(value)}
        self.diagnostics = {"poisson_residual_norm": 1.5, "boundary_operator_proxy_loss": 2.0}


class _FakeEngine:
    def single_run(self, cond, geom, axis):  # noqa: ANN001
        return _FakeResult(sum(float(v) for v in cond.values()))

    def single_run_aggregated(self, cond, geom, axis):  # noqa: ANN001
        return self.single_run(cond=cond, geom=geom, axis=axis)


class _StaticResult:
    def __init__(self, qoi: dict[str, float], diagnostics: dict[str, float] | None = None):
        self.qoi = dict(qoi)
        self.diagnostics = dict(diagnostics or {})


class _StaticEngine:
    def __init__(self, qoi: dict[str, float], diagnostics: dict[str, float] | None = None):
        self.result = _StaticResult(qoi, diagnostics)

    def single_run_aggregated(self, cond, geom, axis):  # noqa: ANN001
        return self.result


class _SequenceEngine:
    def __init__(self, results: list[_StaticResult]):
        self.results = list(results)
        self.calls = 0
        self.save_outputs_flags: list[bool] = []

    def single_run_aggregated(self, cond, geom, axis, save_outputs=True):  # noqa: ANN001
        self.save_outputs_flags.append(bool(save_outputs))
        result = self.results[self.calls % len(self.results)]
        self.calls += 1
        return result


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
    assert a.objective_value == b.objective_value
    assert len(a.trials) == 6
    assert a.backend == "random"
    assert a.objective_mode == "weighted_sum"


def test_optimize_runner_random_backend_evaluates_initial_cond_first():
    runner = OptimizeRunner(_FakeEngine())
    out = runner.run(
        space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        geom_space={},
        n_trials=3,
        geom_ref={"geom_id": "default"},
        seed=7,
        backend="random",
        backend_cfg={"initial_cond": {"c0": 0.25, "c1": 0.75}},
    )

    assert out.trials[0]["cond"] == {"c0": 0.25, "c1": 0.75}
    assert out.backend_cfg["effective_sampler"] == "random"


def test_optimize_runner_rejects_unknown_optuna_sampler():
    pytest.importorskip("optuna")
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(ValueError, match="sampler must be one of"):
        runner.run(
            space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
            geom_space={},
            n_trials=2,
            geom_ref={"geom_id": "default"},
            backend="optuna",
            backend_cfg={"sampler": "not-cmaes"},
        )


def test_optimize_runner_cmaes_sampler_smoke():
    pytest.importorskip("optuna")
    pytest.importorskip("cmaes")
    runner = OptimizeRunner(_FakeEngine())
    initial = {"c0": 0.5, "c1": 0.5}
    out = runner.run(
        space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        geom_space={},
        n_trials=10,
        geom_ref={"geom_id": "default"},
        seed=7,
        backend="optuna",
        backend_cfg={
            "sampler": "cmaes",
            "initial_cond": initial,
            "x0": initial,
            "sigma0": 0.25,
            "popsize": 4,
        },
    )

    assert len(out.trials) == 10
    assert out.trials[0]["cond"] == initial
    assert out.backend_cfg["effective_sampler"] == "cmaes"


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


def test_optimize_runner_rejects_legacy_objective_keys():
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(ValueError, match="legacy objective keys"):
        runner.run(
            space={"c0": (0.0, 1.0)},
            geom_space={},
            n_trials=1,
            geom_ref={"geom_id": "default"},
            backend="random",
            objective_cfg={"main": "uniformity"},
        )


def test_optimize_runner_two_stage_backend_runs_global_then_local():
    runner = OptimizeRunner(_FakeEngine())
    out = runner.run(
        space={"c0": (0.0, 1.0), "c1": (0.0, 1.0)},
        geom_space={},
        n_trials=6,
        geom_ref={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
        seed=3,
        backend="two_stage",
        backend_cfg={"n_initial": 2, "top_k": 2, "local_trials_per_seed": 2, "local_radius_frac": 0.2},
    )
    assert out.backend == "two_stage"
    assert len(out.trials) == 6
    assert out.objective_value == pytest.approx(min(t["objective_value"] for t in out.trials))


def test_optimize_runner_two_stage_backend_records_geom_space_trials():
    runner = OptimizeRunner(_FakeEngine())
    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={"part.coil_01.tx": (-0.2, 0.2)},
        n_trials=5,
        geom_ref={"geom_id": "default"},
        axis={"mode": "steady", "value": 0.0},
        seed=4,
        backend="two_stage",
        backend_cfg={"n_initial": 2, "top_k": 1, "local_trials_per_seed": 3, "local_radius_frac": 0.25},
    )

    assert out.backend == "two_stage"
    assert len(out.trials) == 5
    assert all("part.coil_01.tx" in t["geom_param"] for t in out.trials)
    assert "part.coil_01.tx" in out.best_geom_param


def test_optimize_runner_two_stage_uses_feasible_search_value_for_local_seed():
    engine = _SequenceEngine(
        [
            _StaticResult({"uniformity": 0.1}, {"poisson_residual_norm": 1.0}),
            _StaticResult({"uniformity": 0.5}, {"poisson_residual_norm": 0.0}),
            _StaticResult({"uniformity": 0.4}, {"poisson_residual_norm": 0.0}),
        ]
    )
    runner = OptimizeRunner(engine)

    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={},
        n_trials=3,
        geom_ref={"geom_id": "default"},
        seed=9,
        backend="two_stage",
        backend_cfg={"n_initial": 2, "top_k": 1, "local_trials_per_seed": 1, "local_radius_frac": 0.0},
        constraints_cfg=[{"key": "poisson_residual_norm", "upper": 0.05}],
    )

    assert out.trials[0]["feasible"] is False
    assert out.trials[1]["feasible"] is True
    assert out.trials[2]["cond"] == out.trials[1]["cond"]


def test_optimize_runner_two_stage_rejects_nonfinite_local_radius():
    runner = OptimizeRunner(_FakeEngine())
    with pytest.raises(ValueError, match="local_radius_frac"):
        runner.run(
            space={"c0": (0.0, 1.0)},
            geom_space={},
            n_trials=2,
            geom_ref={"geom_id": "default"},
            backend="two_stage",
            backend_cfg={"local_radius_frac": float("nan")},
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


def test_resolve_optimize_backend_uses_backend_key():
    assert _resolve_optimize_backend({"backend": "two_stage"}) == "two_stage"


def test_resolve_optimize_backend_rejects_legacy_sampler():
    with pytest.raises(ValueError, match="sampler is removed"):
        _resolve_optimize_backend({"sampler": "random"})


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
    assert out.trials[0]["diagnostic_poisson_residual_norm"] == pytest.approx(1.5)


def test_optimize_runner_prefers_feasible_trial_over_lower_infeasible_objective():
    engine = _SequenceEngine(
        [
            _StaticResult({"uniformity": 0.1}, {"poisson_residual_norm": 1.0}),
            _StaticResult({"uniformity": 0.5}, {"poisson_residual_norm": 0.0}),
        ]
    )
    runner = OptimizeRunner(engine)

    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={},
        n_trials=2,
        geom_ref={"geom_id": "default"},
        backend="random",
        constraints_cfg=[{"key": "poisson_residual_norm", "upper": 0.05}],
    )

    assert out.feasible is True
    assert out.objective_value == pytest.approx(0.5)
    assert out.search_value == pytest.approx(0.5)
    assert out.constraint_violation_total == pytest.approx(0.0)
    assert out.trials[0]["feasible"] is False
    assert out.trials[0]["violated_constraints"] == ["poisson_residual_norm"]
    assert out.trials[0]["constraint_violation_total"] > 0.0
    assert out.trials[0]["search_value"] > out.trials[1]["search_value"]
    assert out.trials[1]["qoi_uniformity"] == pytest.approx(0.5)
    assert out.trials[1]["diagnostic_poisson_residual_norm"] == pytest.approx(0.0)


def test_optimize_runner_all_infeasible_uses_lowest_search_value():
    engine = _SequenceEngine(
        [
            _StaticResult({"uniformity": 0.1}, {"poisson_residual_norm": 1.0}),
            _StaticResult({"uniformity": 0.5}, {"poisson_residual_norm": 0.2}),
        ]
    )
    runner = OptimizeRunner(engine)

    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={},
        n_trials=2,
        geom_ref={"geom_id": "default"},
        backend="random",
        constraints_cfg=[{"key": "poisson_residual_norm", "upper": 0.05}],
    )

    assert out.feasible is False
    assert out.objective_value == pytest.approx(0.5)
    assert out.constraint_violation_total == pytest.approx(3.0)
    assert out.search_value == pytest.approx(3_000_000.5)
    assert out.search_value < out.trials[0]["search_value"]


def test_optimize_runner_records_weighted_terms():
    runner = OptimizeRunner(
        _StaticEngine(
            {"uniformity": 0.5, "boundary_gamma_uniformity": 0.2},
            {"poisson_residual_norm": 0.1},
        )
    )

    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={},
        n_trials=1,
        geom_ref={"geom_id": "default"},
        backend="random",
        objective_cfg={
            "mode": "weighted_sum",
            "terms": [
                {"key": "uniformity", "direction": "min", "weight": 1.0},
                {"key": "poisson_residual_norm", "direction": "min", "weight": 0.2},
            ],
        },
    )

    assert out.objective_mode == "weighted_sum"
    assert out.trials[0]["objective_value"] == pytest.approx(0.52)
    assert out.trials[0]["search_value"] == pytest.approx(0.52)
    assert "objective_key" not in out.trials[0]
    assert out.trials[0]["objective_term_uniformity"] == pytest.approx(0.5)
    assert out.trials[0]["objective_term_poisson_residual_norm"] == pytest.approx(0.02)


def test_optimize_runner_top_k_output_mode_suppresses_field_saves_during_search():
    engine = _SequenceEngine(
        [
            _StaticResult({"uniformity": 0.3}),
            _StaticResult({"uniformity": 0.2}),
            _StaticResult({"uniformity": 0.1}),
        ]
    )
    runner = OptimizeRunner(engine)

    out = runner.run(
        space={"c0": (0.0, 1.0)},
        geom_space={},
        n_trials=3,
        geom_ref={"geom_id": "default"},
        backend="random",
        output_cfg={"save_fields": "top_k", "top_k": 1},
    )

    assert out.output_cfg == {"save_fields": "top_k", "top_k": 1}
    assert engine.save_outputs_flags == [False, False, False]


def test_cond_space_from_stats_uses_preprocessing_bounds():
    space = cond_space_from_stats(
        ["PP0", "Td"],
        {"PP0": {"min": 1.0, "max": 5.0}, "Td": {"min": 0.03, "max": 0.3}},
    )
    assert space == {"PP0": (1.0, 5.0), "Td": (0.03, 0.3)}


def test_cond_space_from_stats_requires_complete_bounds():
    with pytest.raises(ValueError, match="cond_stats is missing bounds"):
        cond_space_from_stats(["PP0", "Td"], {"PP0": {"min": 1.0, "max": 5.0}})
