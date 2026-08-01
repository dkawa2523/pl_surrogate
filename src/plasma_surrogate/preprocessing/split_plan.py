"""Preprocessing split-plan builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_surrogate.preprocessing.split import (
    build_condition_grouped_splits,
    build_extrapolation_split,
    build_interpolation_overlap_split_with_status,
    build_interpolation_split,
    build_structure_holdout_split,
)


@dataclass(frozen=True)
class SplitPlan:
    random: dict[str, list[str]]
    interp_marginal: dict[str, list[str]]
    interp_overlap: dict[str, list[str]]
    interp: dict[str, list[str]]
    extrap: dict[str, list[str]]
    structure_holdout: dict[str, list[str]]
    structure_holdout_meta: dict[str, Any]


class SplitPlanBuilder:
    def __init__(self, split_cfg: dict[str, Any]):
        self.cfg = dict(split_cfg or {})

    @staticmethod
    def _case_value(case: dict[str, Any], key: str) -> Any:
        if key in case:
            return case[key]
        cond = dict(case.get("cond", {}) or {})
        if key in cond:
            return cond[key]
        raise KeyError(key)

    def _structure_holdout(
        self,
        *,
        cases: list[dict[str, Any]],
        case_ids: list[str],
        ratios: tuple[float, float, float],
        fallback: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        raw_cfg = self.cfg.get("structure_holdout", {})
        if raw_cfg is None:
            raw_cfg = {}
        if not isinstance(raw_cfg, dict):
            raise TypeError("split.structure_holdout must be a mapping")
        cfg = dict(raw_cfg)
        enabled_raw = cfg.get("enabled", "auto")
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower()
            if enabled not in {"auto", "true", "false"}:
                raise ValueError("split.structure_holdout.enabled must be one of: auto, true, false")
        else:
            enabled = "true" if bool(enabled_raw) else "false"
        required = bool(cfg.get("required", False))
        if enabled == "false":
            if required:
                raise ValueError("split.structure_holdout.required=true conflicts with enabled=false")
            return fallback, {
                "is_real_structure_holdout": False,
                "group_source": "",
                "group_keys": [],
                "n_structure_groups": 0,
                "fallback_reason": "disabled",
                "group_by_case": {},
            }

        group_keys_raw = cfg.get("group_keys", [])
        if isinstance(group_keys_raw, str):
            group_keys = [group_keys_raw]
        else:
            group_keys = [str(value) for value in list(group_keys_raw or [])]
        group_key = str(cfg.get("group_key", "")).strip()
        group_source = ""
        groups: list[Any] | None = None

        if group_keys:
            try:
                groups = [tuple(self._case_value(case, key) for key in group_keys) for case in cases]
            except KeyError as exc:
                raise ValueError(
                    f"split.structure_holdout.group_keys references missing case/condition key: {exc.args[0]!r}"
                ) from exc
            group_source = "configured_group_keys"
        elif group_key:
            try:
                groups = [self._case_value(case, group_key) for case in cases]
            except KeyError as exc:
                raise ValueError(
                    f"split.structure_holdout.group_key references missing case/condition key: {group_key!r}"
                ) from exc
            group_keys = [group_key]
            group_source = "configured_group_key"
        else:
            # Prefer explicit geometry identities.  A structure_npz path is deliberately not
            # accepted here: ICP stores one pack per operating case, so using that path would
            # silently turn a structure holdout back into a case-wise split.
            for candidate in ("base_name", "structure_group", "geom_id"):
                if all(str(case.get(candidate, "")).strip() for case in cases):
                    groups = [str(case[candidate]) for case in cases]
                    group_keys = [candidate]
                    group_source = f"auto:{candidate}"
                    break

        distinct = len(set(groups or []))
        if groups is None or distinct < 3:
            reason = "missing_structure_group" if groups is None else f"insufficient_structure_groups:{distinct}"
            if required or enabled == "true":
                raise ValueError(
                    "Real structure holdout requested but unavailable; provide at least 3 groups via "
                    "case.base_name/structure_group/geom_id or split.structure_holdout.group_key(s). "
                    f"reason={reason}"
                )
            return fallback, {
                "is_real_structure_holdout": False,
                "group_source": group_source,
                "group_keys": group_keys,
                "n_structure_groups": distinct,
                "fallback_reason": reason,
                "group_by_case": {},
            }

        structure_split = build_structure_holdout_split(
            case_ids,
            structure_groups=groups,
            seed=int(cfg.get("seed", self.cfg.get("seed", 0))),
            ratios=tuple(cfg.get("ratios", ratios)),
        )
        group_by_case = {cid: str(group) for cid, group in zip(case_ids, groups)}
        memberships = {
            split_name: {group_by_case[cid] for cid in structure_split[split_name]}
            for split_name in ("train", "val", "test")
        }
        if (
            memberships["train"] & memberships["val"]
            or memberships["train"] & memberships["test"]
            or memberships["val"] & memberships["test"]
        ):
            raise RuntimeError("Structure holdout group leakage detected after split construction")
        return structure_split, {
            "is_real_structure_holdout": True,
            "group_source": group_source,
            "group_keys": group_keys,
            "n_structure_groups": distinct,
            "fallback_reason": "",
            "group_by_case": group_by_case,
        }

    def build(self, *, cases: list[dict[str, Any]], cond_order: list[str]) -> SplitPlan:
        ratios = self.cfg.get("ratios", [0.7, 0.15, 0.15])
        case_ids = [str(c["case_id"]) for c in cases]
        split_groups = [str(c.get("split_group", c["case_id"])) for c in cases]
        cond_by_case = {str(c["case_id"]): c["cond"] for c in cases}
        condition_group_ties = bool(self.cfg.get("condition_group_ties", True))
        if not condition_group_ties:
            raise ValueError(
                "split.condition_group_ties=false is unsafe and unsupported; "
                "complete condition tuples must remain in one partition"
            )
        random_split = build_condition_grouped_splits(
            case_ids,
            cond_values=cond_by_case,
            keys=cond_order,
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
            split_groups=split_groups,
        )
        overlap_status = build_interpolation_overlap_split_with_status(
            case_ids,
            cond_values=cond_by_case,
            keys=cond_order,
            seed=int(self.cfg.get("seed", 0)),
            ratios=tuple(ratios),
            split_groups=split_groups,
        )
        interp_overlap = dict(overlap_status["split"])
        if interp_mode == "overlap" and not bool(overlap_status.get("feasible", True)):
            reason = str(overlap_status.get("reason", "")) or "unknown"
            raise ValueError(f"split.interp_mode=overlap is not feasible: {reason}")
        structure_holdout, structure_holdout_meta = self._structure_holdout(
            cases=cases,
            case_ids=case_ids,
            ratios=tuple(ratios),
            fallback=random_split,
        )
        return SplitPlan(
            random=random_split,
            interp_marginal=interp_marginal,
            interp_overlap=interp_overlap,
            interp=interp_overlap if interp_mode == "overlap" else interp_marginal,
            extrap=extrap_split,
            structure_holdout=structure_holdout,
            structure_holdout_meta=structure_holdout_meta,
        )


__all__ = ["SplitPlan", "SplitPlanBuilder"]
