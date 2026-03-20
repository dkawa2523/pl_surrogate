from __future__ import annotations

import pytest

from plasma_surrogate.core.task_spec import TaskSpecV1


def test_task_spec_v1_validate_ok(task_spec_dict: dict):
    spec = TaskSpecV1.from_dict(task_spec_dict)
    spec.validate()


def test_task_spec_v1_validate_missing_output(task_spec_dict: dict):
    broken = dict(task_spec_dict)
    broken["outputs"] = [o for o in broken["outputs"] if o["name"] != "phi"]
    with pytest.raises(ValueError):
        TaskSpecV1.from_dict(broken)
