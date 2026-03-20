"""Internal task contracts for lightweight pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskContext:
    task_name: str
    config_path: Path
    stages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaskResult:
    payload: dict[str, Any]


class BaseTask(Protocol):
    def run(self, ctx: TaskContext) -> TaskResult:
        ...

