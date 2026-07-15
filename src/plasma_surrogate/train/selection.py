"""Validation objectives used to choose model checkpoints.

The product objective is intentionally case-macro: every case is scored first,
then targets (or target families) are combined.  Legacy pooled-R2 helpers remain
available for reproducing older runs, but new spatial-field runs should use
``best_val_spatial_objective``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from plasma_surrogate.core.deeponet_contract import masked_r2_score
from plasma_surrogate.core.target_groups import TargetGroup, resolve_target_groups


GROUP_BALANCE_SELECTION_MODE = "best_val_group_balance"
SPATIAL_SELECTION_MODE = "best_val_spatial_objective"
SPATIAL_SELECTION_OBJECTIVE_VERSION = "case_macro_spatial_rmse_v2"


def resolve_spatial_selection_config(raw_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve the small, explicit config for the case-macro spatial objective."""

    raw = dict(raw_cfg or {})
    spatial = dict(raw.get("spatial", {}) or {})
    case_aggregation = dict(raw.get("case_aggregation", {}) or {})
    allowed_spatial = {
        "point_weight",
        "gradient_weight",
        "boundary_weight",
        "boundary_band_px",
        "gradient_spacing",
        "gradient_normalization",
        "gradient_epsilon",
    }
    unknown_spatial = sorted(set(str(key) for key in spatial) - allowed_spatial)
    if unknown_spatial:
        raise ValueError(f"selection.spatial contains unsupported keys: {unknown_spatial}")
    allowed_case = {"median_weight", "p90_weight", "worst_weight"}
    unknown_case = sorted(set(str(key) for key in case_aggregation) - allowed_case)
    if unknown_case:
        raise ValueError(f"selection.case_aggregation contains unsupported keys: {unknown_case}")

    def _nonnegative(payload: Mapping[str, Any], key: str, default: float) -> float:
        value = float(payload.get(key, default))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"selection.{key} must be finite and >= 0")
        return value

    spacing_raw = spatial.get("gradient_spacing", [1.0, 1.0])
    if not isinstance(spacing_raw, (list, tuple)) or len(spacing_raw) != 2:
        raise ValueError("selection.spatial.gradient_spacing must be [dy, dx]")
    spacing = tuple(float(value) for value in spacing_raw)
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("selection.spatial.gradient_spacing values must be finite and > 0")
    boundary_band_px = float(spatial.get("boundary_band_px", 2.0))
    if not np.isfinite(boundary_band_px) or boundary_band_px < 0.0:
        raise ValueError("selection.spatial.boundary_band_px must be finite and >= 0")
    gradient_normalization = str(spatial.get("gradient_normalization", "none")).strip().lower()
    if gradient_normalization not in {"none", "target_rms"}:
        raise ValueError("selection.spatial.gradient_normalization must be one of: none, target_rms")
    gradient_epsilon = float(spatial.get("gradient_epsilon", 0.05))
    if not np.isfinite(gradient_epsilon) or gradient_epsilon <= 0.0:
        raise ValueError("selection.spatial.gradient_epsilon must be finite and > 0")
    resolved = {
        "objective_version": SPATIAL_SELECTION_OBJECTIVE_VERSION,
        "point_weight": _nonnegative(spatial, "point_weight", 1.0),
        "gradient_weight": _nonnegative(spatial, "gradient_weight", 0.10),
        "boundary_weight": _nonnegative(spatial, "boundary_weight", 0.25),
        "boundary_band_px": boundary_band_px,
        "gradient_spacing": spacing,
        "gradient_normalization": gradient_normalization,
        "gradient_epsilon": gradient_epsilon,
        "median_weight": _nonnegative(case_aggregation, "median_weight", 1.0),
        "p90_weight": _nonnegative(case_aggregation, "p90_weight", 0.25),
        "worst_weight": _nonnegative(case_aggregation, "worst_weight", 0.10),
    }
    if sum(float(resolved[key]) for key in ("point_weight", "gradient_weight", "boundary_weight")) <= 0.0:
        raise ValueError("selection.spatial weights must contain at least one positive value")
    if sum(float(resolved[key]) for key in ("median_weight", "p90_weight", "worst_weight")) <= 0.0:
        raise ValueError("selection.case_aggregation weights must contain at least one positive value")
    return resolved


def _align_bhw(values: Any, *, batch_size: int, height: int, width: int, key: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, ...], (batch_size, height, width))
    elif arr.ndim == 3 and int(arr.shape[0]) == 1 and batch_size > 1:
        arr = np.broadcast_to(arr, (batch_size, height, width))
    elif arr.ndim == 4 and int(arr.shape[1]) == 1:
        arr = arr[:, 0]
    if arr.shape != (batch_size, height, width):
        raise ValueError(f"{key} must align to [B,H,W]={batch_size,height,width}; got={arr.shape}")
    return np.asarray(arr)


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return float("inf")
    return float(np.sqrt(np.mean(arr * arr)))


def _case_target_spatial_parts(
    *,
    error: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    distance: np.ndarray | None,
    cfg: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    point = _rmse(error[active])
    gradient_terms: list[np.ndarray] = []
    target_gradient_terms: list[np.ndarray] = []
    dy, dx = (float(value) for value in cfg["gradient_spacing"])
    if error.shape[0] > 1:
        pair = active[1:, :] & active[:-1, :]
        if np.any(pair):
            gradient_terms.append(np.diff(error, axis=0)[pair] / dy)
            target_gradient_terms.append(np.diff(target, axis=0)[pair] / dy)
    if error.shape[1] > 1:
        pair = active[:, 1:] & active[:, :-1]
        if np.any(pair):
            gradient_terms.append(np.diff(error, axis=1)[pair] / dx)
            target_gradient_terms.append(np.diff(target, axis=1)[pair] / dx)
    gradient = _rmse(np.concatenate(gradient_terms)) if gradient_terms else 0.0
    if gradient_terms and str(cfg["gradient_normalization"]) == "target_rms":
        target_gradient_rms = _rmse(np.concatenate(target_gradient_terms))
        gradient /= max(target_gradient_rms, float(cfg["gradient_epsilon"]))
    boundary = 0.0
    if float(cfg["boundary_weight"]) > 0.0:
        if distance is None:
            raise ValueError(
                "selection.mode=best_val_spatial_objective with boundary_weight>0 requires distance_any"
            )
        boundary_active = active & (np.abs(np.asarray(distance, dtype=np.float64)) <= float(cfg["boundary_band_px"]))
        if not np.any(boundary_active):
            raise ValueError("spatial validation objective resolved an empty boundary band")
        boundary = _rmse(error[boundary_active])
    total = (
        float(cfg["point_weight"]) * point
        + float(cfg["gradient_weight"]) * gradient
        + float(cfg["boundary_weight"]) * boundary
    )
    return float(total), float(point), float(gradient), float(boundary)


def case_macro_spatial_objective(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    y_vars: list[str],
    plasma_mask: np.ndarray,
    distance_any: np.ndarray | None,
    cfg: Mapping[str, Any] | None = None,
    target_weights: Mapping[str, float] | None = None,
    groups: Mapping[str, TargetGroup] | None = None,
    group_weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return a lower-is-better spatial objective with explicit tail penalties.

    ``pred`` and ``target`` are expected in the same train-fitted normalized
    field space.  This keeps target scales comparable while avoiding the
    case/pixel pooling that made legacy R2 optimistic.
    """

    pred_arr = np.asarray(pred, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    if pred_arr.shape != target_arr.shape or pred_arr.ndim != 4:
        raise ValueError(f"spatial selection expects matching [B,C,H,W] arrays; got={pred_arr.shape}/{target_arr.shape}")
    if int(pred_arr.shape[1]) != len(y_vars):
        raise ValueError("spatial selection target channel count does not match y_vars")
    resolved = resolve_spatial_selection_config(cfg)
    batch_size, _channels, height, width = pred_arr.shape
    mask = _align_bhw(
        plasma_mask,
        batch_size=int(batch_size),
        height=int(height),
        width=int(width),
        key="plasma_mask",
    ).astype(bool)
    distance = None
    if distance_any is not None:
        distance = _align_bhw(
            distance_any,
            batch_size=int(batch_size),
            height=int(height),
            width=int(width),
            key="distance_any",
        ).astype(np.float64)

    parts: dict[str, Any] = {
        "selection_objective_version": SPATIAL_SELECTION_OBJECTIVE_VERSION,
    }
    target_scores: dict[str, float] = {}
    for target_idx, name in enumerate(y_vars):
        totals: list[float] = []
        points: list[float] = []
        gradients: list[float] = []
        boundaries: list[float] = []
        for case_idx in range(int(batch_size)):
            active = np.asarray(mask[case_idx], dtype=bool)
            if not np.any(active):
                raise ValueError(f"spatial selection has an empty plasma mask for case index={case_idx}")
            error = np.asarray(pred_arr[case_idx, target_idx] - target_arr[case_idx, target_idx], dtype=np.float64)
            if not np.all(np.isfinite(error[active])):
                raise ValueError(
                    "spatial selection contains non-finite prediction or target values "
                    f"inside the active mask: case={case_idx}, target={name}"
                )
            values = _case_target_spatial_parts(
                error=error,
                target=np.asarray(target_arr[case_idx, target_idx], dtype=np.float64),
                active=active,
                distance=None if distance is None else distance[case_idx],
                cfg=resolved,
            )
            totals.append(values[0])
            points.append(values[1])
            gradients.append(values[2])
            boundaries.append(values[3])
        totals_arr = np.asarray(totals, dtype=np.float64)
        median = float(np.median(totals_arr))
        p90 = float(np.percentile(totals_arr, 90.0))
        worst = float(np.max(totals_arr))
        target_score = (
            float(resolved["median_weight"]) * median
            + float(resolved["p90_weight"]) * p90
            + float(resolved["worst_weight"]) * worst
        )
        target_scores[str(name)] = float(target_score)
        parts[f"spatial_{name}"] = float(target_score)
        parts[f"spatial_point_median_{name}"] = float(np.median(points))
        parts[f"spatial_gradient_median_{name}"] = float(np.median(gradients))
        parts[f"spatial_boundary_median_{name}"] = float(np.median(boundaries))
        parts[f"spatial_p90_{name}"] = p90
        parts[f"spatial_worst_{name}"] = worst

    if groups:
        assignments: dict[str, list[str]] = {str(name): [] for name in y_vars}
        for group_name, group in groups.items():
            for target_name in group.targets:
                target_key = str(target_name)
                if target_key in assignments:
                    assignments[target_key].append(str(group_name))
        missing = [name for name, owners in assignments.items() if not owners]
        duplicated = {name: owners for name, owners in assignments.items() if len(owners) > 1}
        if missing or duplicated:
            raise ValueError(
                "spatial selection target groups must cover every output exactly once; "
                f"missing={missing}, duplicated={duplicated}"
            )
        score_sum = 0.0
        weight_sum = 0.0
        weights = dict(group_weights or {})
        for group_name, group in groups.items():
            values = [target_scores[str(name)] for name in group.targets if str(name) in target_scores]
            if not values:
                continue
            group_score = float(np.mean(values))
            parts[f"spatial_group_{group_name}"] = group_score
            weight = float(weights.get(str(group_name), 1.0))
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"spatial selection group weight must be finite and >=0: {group_name}")
            score_sum += weight * group_score
            weight_sum += weight
        if weight_sum <= 0.0:
            return float("nan"), parts
        return float(score_sum / weight_sum), parts

    weights = dict(target_weights or {})
    if not weights:
        weights = {str(name): 1.0 for name in y_vars}
    score_sum = 0.0
    weight_sum = 0.0
    for name in y_vars:
        weight = float(weights.get(str(name), 0.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"spatial selection target weight must be finite and >=0: {name}")
        score_sum += weight * target_scores[str(name)]
        weight_sum += weight
    if weight_sum <= 0.0:
        return float("nan"), parts
    return float(score_sum / weight_sum), parts


def resolve_selection_target_groups(
    *,
    output_vars: list[str],
    target_role_schema: dict[str, Any] | None = None,
    model_target_groups: Mapping[str, TargetGroup] | None = None,
) -> dict[str, TargetGroup]:
    """Resolve groups for validation selection without target-name special cases."""

    output_names = [str(value) for value in output_vars]
    output_set = set(output_names)
    if model_target_groups:
        groups: dict[str, TargetGroup] = {}
        for raw_name, group in dict(model_target_groups).items():
            targets = tuple(str(target) for target in group.targets if str(target) in output_set)
            if not targets:
                continue
            group_name = str(raw_name)
            groups[group_name] = TargetGroup(
                name=group_name,
                targets=targets,
                source=str(group.source),
                field_family=group.field_family,
                roles=tuple(group.roles),
            )
    else:
        schema = dict(target_role_schema or {})
        if not schema.get("targets"):
            raise ValueError("selection group aggregation requires target_role_schema.targets")
        selected = set(output_names)
        raw_targets = schema.get("targets", [])
        schema["targets"] = [
            dict(raw)
            for raw in raw_targets
            if isinstance(raw, Mapping) and str(raw.get("id", "")).strip() in selected
        ]
        groups = resolve_target_groups(output_vars=output_names, target_role_schema=schema)
    if not groups:
        raise ValueError("selection group aggregation resolved no usable target groups")

    assignments: dict[str, list[str]] = {name: [] for name in output_names}
    for group_name, group in groups.items():
        for target in group.targets:
            target_name = str(target)
            if target_name in assignments:
                assignments[target_name].append(str(group_name))
    missing = [name for name, owners in assignments.items() if not owners]
    duplicated = {name: owners for name, owners in assignments.items() if len(owners) > 1}
    if missing or duplicated:
        raise ValueError(
            "selection target groups must cover every output exactly once; "
            f"missing={missing}, duplicated={duplicated}"
        )
    return groups


def resolve_selection_group_weights(
    raw_weights: Mapping[str, Any] | None,
    *,
    groups: Mapping[str, TargetGroup],
    cfg_prefix: str,
) -> dict[str, float]:
    """Return finite non-negative group weights, defaulting to uniform groups."""

    group_names = [str(name) for name, group in groups.items() if group.targets]
    if not group_names:
        raise ValueError("selection group weights require at least one non-empty target group")
    raw = dict(raw_weights or {})
    if not raw:
        uniform = 1.0 / float(len(group_names))
        return {name: float(uniform) for name in group_names}
    expected = set(group_names)
    unknown = sorted(set(str(k) for k in raw) - expected)
    missing = sorted(expected - set(str(k) for k in raw))
    if unknown:
        raise ValueError(f"{cfg_prefix}.selection.group_weights contains unknown groups: {unknown}")
    if missing:
        raise ValueError(f"{cfg_prefix}.selection.group_weights missing groups: {missing}")
    out: dict[str, float] = {}
    total = 0.0
    for name in group_names:
        weight = float(raw.get(name, 0.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{cfg_prefix}.selection.group_weights[{name}] must be finite and >= 0")
        out[name] = float(weight)
        total += float(weight)
    if total <= 0.0:
        raise ValueError(f"{cfg_prefix}.selection.group_weights must sum to > 0")
    return out


def group_plasma_balance_score(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    y_vars: list[str],
    plasma_mask: np.ndarray,
    groups: Mapping[str, TargetGroup],
    group_weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Score validation predictions as a finite weighted mean of group R2 values."""

    y_order = [str(v) for v in y_vars]
    y_pos = {name: idx for idx, name in enumerate(y_order)}
    mask = np.asarray(plasma_mask, dtype=bool)
    parts: dict[str, float] = {}
    target_r2: dict[str, float] = {}
    for name in y_order:
        idx = y_pos[name]
        r2_val = masked_r2_score(target[:, idx], pred[:, idx], mask)
        target_r2[name] = float(r2_val)
        parts[f"r2_{name}_plasma"] = float(r2_val)

    score_sum = 0.0
    score_weight = 0.0
    for group_name, group in groups.items():
        values = [target_r2[str(target)] for target in group.targets if str(target) in target_r2]
        finite_values = [float(value) for value in values if np.isfinite(float(value))]
        if not finite_values:
            continue
        group_score = float(np.mean(finite_values))
        parts[f"r2_group_{group_name}_plasma"] = group_score
        weight = float(group_weights.get(str(group_name), 0.0))
        if weight > 0.0 and np.isfinite(group_score):
            score_sum += weight * group_score
            score_weight += weight
    if score_weight <= 0.0:
        return float("nan"), parts
    return float(score_sum / score_weight), parts


__all__ = [
    "GROUP_BALANCE_SELECTION_MODE",
    "SPATIAL_SELECTION_MODE",
    "SPATIAL_SELECTION_OBJECTIVE_VERSION",
    "case_macro_spatial_objective",
    "group_plasma_balance_score",
    "resolve_selection_group_weights",
    "resolve_selection_target_groups",
    "resolve_spatial_selection_config",
]
