"""Thin task dispatch adapter for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.cli.workflows import (
    run_cleanse,
    run_evaluate,
    run_feature,
    run_infer,
    run_pipeline,
    run_preprocess,
    run_train,
    run_viz,
)
from plasma_surrogate.pipeline.task_contract import BaseTask, TaskContext, TaskResult


class _CallableTask:
    """Adapter for function-based tasks."""

    def __init__(self, fn: Callable[[str | Path], dict[str, Any]]):
        self._fn = fn

    def run(self, ctx: TaskContext) -> TaskResult:
        return TaskResult(payload=self._fn(ctx.config_path))


class TaskRunner:
    """Execute top-level tasks while keeping CLI handlers minimal."""

    def __init__(self) -> None:
        self._stage_map: dict[str, tuple[str, ...]] = {
            "cleanse": ("data_cleaning",),
            "feature": ("featurization",),
            "preprocess": ("data_cleaning", "featurization", "preprocessing"),
            "train": ("train", "eval"),
            "infer": ("inference",),
            "evaluate": ("eval",),
            "pipeline.run": ("pipeline",),
            "viz": ("viz",),
            "benchmark.run": ("benchmark",),
            "benchmark.sweep": ("benchmark", "sweep"),
        }
        self._task_map: dict[str, BaseTask] = {
            "cleanse": _CallableTask(run_cleanse),
            "feature": _CallableTask(run_feature),
            "preprocess": _CallableTask(run_preprocess),
            "train": _CallableTask(run_train),
            "infer": _CallableTask(run_infer),
            "evaluate": _CallableTask(run_evaluate),
            "pipeline.run": _CallableTask(run_pipeline),
            "viz": _CallableTask(run_viz),
            "benchmark.run": _CallableTask(self._benchmark_run),
            "benchmark.sweep": _CallableTask(self._benchmark_sweep),
        }

    @staticmethod
    def _benchmark_run(config_path: str | Path) -> dict[str, Any]:
        result = BenchmarkRunner.from_yaml(config_path).run()
        return {"leaderboard": str(result.leaderboard_path)}

    @staticmethod
    def _benchmark_sweep(config_path: str | Path) -> dict[str, Any]:
        result = BenchmarkRunner.from_yaml(config_path).run_sweep()
        return {
            "summary": str(result.summary_path),
            "best_trial": result.best_trial,
            "locked_config": str(result.locked_config_path),
            "locked_leaderboard": str(result.locked_leaderboard_path),
        }

    def run(self, task_name: str, config_path: str | Path) -> dict[str, Any]:
        key = str(task_name)
        task = self._task_map.get(key)
        if task is None:
            raise ValueError(f"Unknown task: {task_name}")
        ctx = TaskContext(
            task_name=key,
            config_path=Path(config_path),
            stages=self._stage_map.get(key, ()),
        )
        payload = dict(task.run(ctx).payload)
        payload.setdefault("stages", list(ctx.stages))
        return payload
