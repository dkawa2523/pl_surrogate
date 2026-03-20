"""Pipeline task dispatch package."""

__all__ = ["TaskRunner"]


def __getattr__(name: str):
    if name == "TaskRunner":
        from plasma_surrogate.pipeline.task_runner import TaskRunner

        return TaskRunner
    raise AttributeError(name)
