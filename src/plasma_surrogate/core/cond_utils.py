"""Shared conditioning helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


def build_cond_matrix_with_axis(
    cases: list[dict[str, Any]],
    cond_schema: CondSchema,
    axis_schema: AxisSchema,
) -> np.ndarray:
    """Encode case cond and axis into [N, Ccond+Caxis] matrix."""

    rows: list[np.ndarray] = []
    for case in cases:
        base = cond_schema.encode(case["cond"])
        axis_vec = axis_schema.encode(float(case.get("axis", 0.0)))
        rows.append(np.concatenate([base, axis_vec], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)
