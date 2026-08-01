"""Cross-validation result aggregation for benchmark runs."""

from __future__ import annotations

from typing import Any

import numpy as np


class CvBenchmarkRunner:
    @staticmethod
    def attach_summary(
        *,
        leaderboard: list[dict[str, Any]],
        cv_rows: dict[str, list[dict[str, float]]],
    ) -> None:
        row_by_model = {str(row["model_id"]): row for row in leaderboard}
        for model_name, rows in cv_rows.items():
            if not rows:
                continue
            out_row = row_by_model.get(str(model_name))
            if out_row is None:
                continue
            numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
            for key in numeric_keys:
                vals = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
                out_row[f"cv_mean_{key}"] = float(np.mean(vals))
                out_row[f"cv_std_{key}"] = float(np.std(vals))


__all__ = ["CvBenchmarkRunner"]
