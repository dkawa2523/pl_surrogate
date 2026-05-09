"""Shared target and selection contracts for train dispatch lanes."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_true_eval(
    y: np.ndarray,
    idx: np.ndarray,
    y_vars: list[str],
    *,
    source_y_vars: list[str] | None = None,
) -> dict[str, np.ndarray]:
    source = list(source_y_vars or y_vars)
    pos = {name: i for i, name in enumerate(source)}
    return {name: y[idx, pos[name] : pos[name] + 1] for name in y_vars}


def resolve_target_vars(raw: Any, *, available: list[str], cfg_key: str) -> list[str]:
    if raw is None:
        return list(available)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{cfg_key} must be a non-empty list")
    target = [str(v) for v in raw]
    unknown = [v for v in target if v not in set(available)]
    if unknown:
        raise ValueError(f"{cfg_key} contains unknown vars: {unknown}; available={available}")
    if len(set(target)) != len(target):
        raise ValueError(f"{cfg_key} must not contain duplicates")
    return target


def resolve_allvars_target_family(raw: Any, *, cfg_key: str) -> str:
    family = str(raw if raw is not None else "allvars").strip().lower()
    if family != "allvars":
        raise ValueError(f"{cfg_key} must be allvars for mainline")
    return family


def resolve_allvars_target_vars_for_family(
    *,
    family: str,
    raw_target_vars: Any,
    available: list[str],
    cfg_key: str,
) -> list[str]:
    default = list(available)
    if raw_target_vars is None:
        return default
    target = resolve_target_vars(raw_target_vars, available=available, cfg_key=cfg_key)
    if target != default:
        raise ValueError(f"{cfg_key} must match target_family=allvars: expected={default}, got={target}")
    return target


def resolve_mainline_selection_weights(
    *,
    selection_cfg: dict[str, Any],
    target_vars: list[str],
    cfg_prefix: str,
) -> dict[str, float]:
    raw_weights = dict(selection_cfg.get("weights", {}))
    if len(raw_weights) == 0:
        uniform = 1.0 / float(max(len(target_vars), 1))
        return {str(name): float(uniform) for name in target_vars}
    expected = set(str(v) for v in target_vars)
    unknown = sorted(set(str(k) for k in raw_weights.keys()) - expected)
    missing = sorted(expected - set(str(k) for k in raw_weights.keys()))
    if unknown:
        raise ValueError(
            f"{cfg_prefix}.selection.weights contains unknown vars: {unknown}; expected={sorted(expected)}"
        )
    if missing:
        raise ValueError(
            f"{cfg_prefix}.selection.weights missing vars: {missing}; expected={sorted(expected)}"
        )
    out: dict[str, float] = {}
    total = 0.0
    for name in target_vars:
        w = float(raw_weights.get(name, 0.0))
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"{cfg_prefix}.selection.weights[{name}] must be finite and >= 0; got={w}")
        out[str(name)] = float(w)
        total += float(w)
    if total <= 0.0:
        raise ValueError(f"{cfg_prefix}.selection.weights must sum to > 0")
    return out


__all__ = [
    "resolve_allvars_target_family",
    "resolve_allvars_target_vars_for_family",
    "resolve_mainline_selection_weights",
    "resolve_target_vars",
    "to_true_eval",
]
