"""Preprocessing split-plan builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_surrogate.preprocessing.split import (
    build_casewise_splits,
    build_extrapolation_split,
    build_interpolation_overlap_split_with_status,
    build_interpolation_split,
)


@dataclass(frozen=True)
class SplitPlan:
    random: dict[str, list[str]]
    interp_marginal: dict[str, list[str]]
    interp_overlap: dict[str, list[str]]
    interp: dict[str, list[str]]
    extrap: dict[str, list[str]]
    structure_holdout: dict[str, list[str]]


class SplitPlanBuilder:
    def __init__(self, split_cfg: dict[str, Any]):
        self.cfg = dict(split_cfg or {})

    def build(self, *, cases: list[dict[str, Any]], cond_order: list[str]) -> SplitPlan:
        ratios = self.cfg.get("ratios", [0.7, 0.15, 0.15])
        case_ids = [str(c["case_id"]) for c in cases]
        split_groups = [str(c.get("split_group", c["case_id"])) for c in cases]
        random_split = build_casewise_splits(
            case_ids,
            split_groups=split_groups,
            seed=int(self.cfg.get("seed", 0)),
            ratios=tuple(ratios),
        )
        removed_extrap_keys = [
            key
            for key in ("pressure_extrap_key", "pressure_extrap_holdout_ratio", "pressure_extrap_val_ratio")
            if key in self.cfg
        ]
        if removed_extrap_keys:
            raise ValueError(
                "split.pressure_extrap_* keys are removed; use split.extrapolation."
                f" Removed keys: {removed_extrap_keys}"
            )
        cond_by_case = {str(c["case_id"]): c["cond"] for c in cases}
        extrap_cfg = dict(self.cfg.get("extrapolation", {}) or {})
        extrap_split = build_extrapolation_split(
            case_ids,
            cond_values=cond_by_case,
            key=str(extrap_cfg.get("key", cond_order[0])),
            direction=str(extrap_cfg.get("direction", "high")).strip().lower(),
            holdout_ratio=float(extrap_cfg.get("holdout_ratio", 0.2)),
            val_ratio_within_remain=float(extrap_cfg.get("val_ratio", 0.2)),
        )
        interp_mode = str(self.cfg.get("interp_mode", "marginal")).strip().lower()
        if interp_mode not in {"marginal", "overlap"}:
            raise ValueError("split.interp_mode must be one of: marginal, overlap")
        interp_marginal = build_interpolation_split(
            case_ids,
            cond_values=cond_by_case,
            keys=cond_order,
            seed=int(self.cfg.get("seed", 0)),
            ratios=tuple(ratios),
            mode="marginal",
        )
        overlap_status = build_interpolation_overlap_split_with_status(
            case_ids,
            cond_values=cond_by_case,
            keys=cond_order,
            seed=int(self.cfg.get("seed", 0)),
            ratios=tuple(ratios),
        )
        interp_overlap = dict(overlap_status["split"])
        if interp_mode == "overlap" and not bool(overlap_status.get("feasible", True)):
            reason = str(overlap_status.get("reason", "")) or "unknown"
            raise ValueError(f"split.interp_mode=overlap is not feasible: {reason}")
        return SplitPlan(
            random=random_split,
            interp_marginal=interp_marginal,
            interp_overlap=interp_overlap,
            interp=interp_overlap if interp_mode == "overlap" else interp_marginal,
            extrap=extrap_split,
            structure_holdout=random_split,
        )


__all__ = ["SplitPlan", "SplitPlanBuilder"]
