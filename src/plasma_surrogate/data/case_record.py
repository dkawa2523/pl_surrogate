"""Case-level metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseRecord:
    case_id: str
    cond_raw: dict[str, Any]
    geom_ref: dict[str, Any]
    axis: dict[str, Any] = field(default_factory=lambda: {"mode": "steady", "value": 0.0})
    y_paths: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
