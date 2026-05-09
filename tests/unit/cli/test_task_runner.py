from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.pipeline.task_contract import TaskContext, TaskResult
from plasma_surrogate.pipeline.task_runner import TaskRunner


def test_task_runner_dispatches_preprocess(monkeypatch):
    calls: list[str] = []

    def fake_preprocess(config_path: str | Path):
        calls.append(str(config_path))
        return {"ok": True, "task": "preprocess"}

    monkeypatch.setattr("plasma_surrogate.pipeline.task_runner.run_preprocess", fake_preprocess)
    out = TaskRunner().run("preprocess", "dummy.yaml")
    assert out["ok"] is True
    assert out["stages"] == ["data_cleaning", "featurization", "preprocessing"]
    assert calls == ["dummy.yaml"]


def test_task_runner_dispatches_benchmark_run(monkeypatch, tmp_path: Path):
    class _Result:
        leaderboard_path = tmp_path / "leaderboard.csv"

    class _Runner:
        def run(self):
            return _Result()

    monkeypatch.setattr("plasma_surrogate.pipeline.task_runner.BenchmarkRunner.from_yaml", lambda _: _Runner())
    out = TaskRunner().run("benchmark.run", "bench.yaml")
    assert out["leaderboard"] == str(tmp_path / "leaderboard.csv")
    assert out["stages"] == ["benchmark"]


def test_task_runner_dispatches_benchmark_sweep(monkeypatch, tmp_path: Path):
    class _Result:
        summary_path = tmp_path / "summary.json"
        best_trial = {"trial_id": "t1", "score": 0.1}
        locked_config_path = tmp_path / "locked.yaml"
        locked_leaderboard_path = tmp_path / "leaderboard_locked.csv"

    class _Runner:
        def run_sweep(self):
            return _Result()

    monkeypatch.setattr("plasma_surrogate.pipeline.task_runner.BenchmarkRunner.from_yaml", lambda _: _Runner())
    out = TaskRunner().run("benchmark.sweep", "bench.yaml")
    assert out["summary"] == str(tmp_path / "summary.json")
    assert out["best_trial"] == {"trial_id": "t1", "score": 0.1}
    assert out["locked_config"] == str(tmp_path / "locked.yaml")
    assert out["locked_leaderboard"] == str(tmp_path / "leaderboard_locked.csv")
    assert out["stages"] == ["benchmark", "sweep"]


def test_task_runner_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        TaskRunner().run("invalid.task", "cfg.yaml")


def test_task_contract_context_defaults():
    ctx = TaskContext(task_name="preprocess", config_path=Path("dummy.yaml"))
    out = TaskResult(payload={"ok": True})
    assert ctx.stages == ()
    assert out.payload["ok"] is True
