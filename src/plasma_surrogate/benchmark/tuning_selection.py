"""Validation-only outer-selection helpers for benchmark and tuning workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from plasma_surrogate.train.selection import SPATIAL_SELECTION_MODE


VALIDATION_VALUE_KEY = "validation_selection_value"


def validation_selection_from_history(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Resolve one finite validation objective without consulting test metrics."""

    rows = [dict(row) for row in history]
    if not rows:
        return {
            "validation_selection_metric": "",
            VALIDATION_VALUE_KEY: float("nan"),
            "validation_selection_mode": "min",
            "validation_selection_reliable": False,
            "validation_selected_epoch": -1,
            "validation_selection_objective_version": "",
            "validation_selection_invalid_reason": "empty_history",
        }

    selected_rows = [row for row in rows if float(row.get("selected_epoch_flag", 0.0) or 0.0) > 0.5]
    selected = selected_rows[-1] if selected_rows else rows[-1]
    selection_mode = str(
        selected.get("selection_mode_effective", rows[-1].get("selection_mode_effective", "last"))
    ).strip().lower()
    if selection_mode in {"best_val_allvars_balance", "best_val_group_balance"}:
        metric = "selected_epoch_score"
        objective_mode = "max"
        raw_value = selected.get("selected_epoch_score", selected.get("val_balance_score"))
    elif selection_mode == SPATIAL_SELECTION_MODE:
        metric = "selected_epoch_score"
        objective_mode = "min"
        raw_value = selected.get("selected_epoch_score", selected.get("val_balance_score"))
    else:
        metric = "val_loss"
        objective_mode = "min"
        raw_value = (
            selected.get("selected_epoch_score")
            if selection_mode in {"best_val_loss", "best_val_data_loss"}
            else selected.get("val_loss")
        )
        if raw_value is None:
            raw_value = selected.get("val_loss")

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")
    try:
        epoch = int(float(selected.get("epoch", -1)))
    except (TypeError, ValueError):
        epoch = -1
    selection_valid = selected.get("selection_valid_flag")
    reliable = bool(np.isfinite(value))
    if selection_valid is not None and selection_mode != "last":
        reliable = reliable and float(selection_valid) > 0.5
    return {
        "validation_selection_metric": metric,
        VALIDATION_VALUE_KEY: value,
        "validation_selection_mode": objective_mode,
        "validation_selection_reliable": reliable,
        "validation_selected_epoch": epoch,
        "validation_selection_objective_version": str(
            selected.get("selection_objective_version", "") or ""
        ),
        "validation_selection_invalid_reason": "" if reliable else "nonfinite_or_invalid_validation_selection",
    }


def require_validation_objective(row: dict[str, Any]) -> tuple[float, str]:
    """Return the validation objective and direction, failing closed on legacy/test-only rows."""

    if VALIDATION_VALUE_KEY not in row:
        raise ValueError(
            "Validation-only tuning requires leaderboard.validation_selection_value; "
            "this result predates the leakage-safe protocol and must be retrained."
        )
    mode = str(row.get("validation_selection_mode", "")).strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError("validation_selection_mode must be one of: min, max")
    try:
        value = float(row[VALIDATION_VALUE_KEY])
    except (TypeError, ValueError) as exc:
        raise ValueError("validation_selection_value must be numeric") from exc
    if not bool(row.get("validation_selection_reliable", True)) or not np.isfinite(value):
        raise ValueError("Validation objective is unavailable or unreliable")
    return value, mode


__all__ = ["VALIDATION_VALUE_KEY", "require_validation_objective", "validation_selection_from_history"]
