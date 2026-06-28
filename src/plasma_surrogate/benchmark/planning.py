"""Benchmark planning helpers for product mainline runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_surrogate.core.contracts import EvaluationProtocol


@dataclass(frozen=True)
class EvalProtocolPlan:
    mode: str
    scope: str
    frozen_ref_tag: str
    primary_split: str
    interp_weight: float
    extrap_weight: float
    interp_mode: str
    primary_metric: str
    primary_mode: str
    target_family_for_score: str
    target_vars_for_score: list[str]
    region_bands: dict[str, Any]

    def as_resolved_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scope_effective": self.scope,
            "frozen_ref_tag": self.frozen_ref_tag,
            "primary_split": self.primary_split,
            "interp_weight": self.interp_weight,
            "extrap_weight": self.extrap_weight,
            "interp_mode": self.interp_mode,
        }

    def as_contract(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            mode=self.mode,
            scope=self.scope,
            primary_split=self.primary_split,
            primary_metric=self.primary_metric,
            objective_mode=self.primary_mode,
            target_vars=tuple(self.target_vars_for_score),
            region_bands=dict(self.region_bands),
            interp_weight=self.interp_weight,
            extrap_weight=self.extrap_weight,
        )


class BenchmarkPlanBuilder:
    def __init__(self, benchmark_cfg: dict[str, Any]):
        self.cfg = dict(benchmark_cfg or {})

    def profile_name(self) -> str:
        return str(self.cfg.get("profile", "m7_global_frozen_ref"))

    def validate_profile_lock(self, profile_lock: dict[str, Any]) -> None:
        requested_axis_mode = str(
            self.cfg.get(
                "axis_mode",
                self.cfg.get("preprocessing", {}).get("axis_schema", {}).get("mode", "steady"),
            )
        )
        requested_phi_mode = self.cfg.get("phi_mode")
        if requested_phi_mode is not None and str(requested_phi_mode) != str(profile_lock["phi_mode"]):
            raise ValueError(
                f"Benchmark profile lock violation: phi_mode={requested_phi_mode} "
                f"must be {profile_lock['phi_mode']}"
            )
        infer_refine = int(self.cfg.get("inference", {}).get("poisson_refine", {}).get("iters", 0))
        if infer_refine != int(profile_lock["poisson_refine_iters"]):
            raise ValueError(
                f"Benchmark profile lock violation: poisson_refine.iters={infer_refine} "
                f"must be {profile_lock['poisson_refine_iters']}"
            )
        if requested_axis_mode != str(profile_lock.get("axis_mode", "steady")):
            raise ValueError(
                f"Benchmark profile lock violation: axis_mode={requested_axis_mode} "
                f"must be {profile_lock.get('axis_mode', 'steady')}"
            )
        requested_primary_qoi = str(
            self.cfg.get("boundary_operator", {}).get(
                "primary_qoi_key",
                profile_lock.get("primary_qoi_key", "Gamma_i"),
            )
        )
        if str(profile_lock.get("primary_qoi_key", requested_primary_qoi)) != requested_primary_qoi:
            raise ValueError(
                f"Benchmark profile lock violation: boundary_operator.primary_qoi_key={requested_primary_qoi} "
                f"must be {profile_lock.get('primary_qoi_key')}"
            )

    def eval_protocol_plan(self, *, y_vars: list[str], scopes: tuple[str, ...]) -> EvalProtocolPlan:
        eval_protocol_cfg = dict(self.cfg.get("eval_protocol", {}))
        eval_cfg = dict(self.cfg.get("eval", {}))
        mode = str(eval_protocol_cfg.get("mode", "single")).strip().lower()
        if mode not in {"single", "primary_axis", "dual_axis"}:
            raise ValueError("benchmark.eval_protocol.mode must be one of: single, primary_axis, dual_axis")
        scope = str(eval_protocol_cfg.get("scope", "common")).strip().lower()
        if scope not in set(scopes):
            raise ValueError("benchmark.eval_protocol.scope must be one of: " f"{', '.join(scopes)}")
        interp_mode = str(eval_cfg.get("interp_mode", eval_protocol_cfg.get("interp_mode", "marginal"))).strip().lower()
        if interp_mode not in {"marginal", "overlap"}:
            raise ValueError("benchmark.eval.interp_mode must be one of: marginal, overlap")
        primary_split = str(eval_protocol_cfg.get("primary_split", "interp")).strip().lower()
        if primary_split not in {"interp", "extrap", "structure_holdout"}:
            raise ValueError("benchmark.eval_protocol.primary_split must be one of: interp, extrap, structure_holdout")
        if mode == "dual_axis" and primary_split == "structure_holdout":
            raise ValueError(
                "benchmark.eval_protocol.primary_split=structure_holdout requires "
                "benchmark.eval_protocol.mode=primary_axis"
            )
        target_family = str(eval_cfg.get("target_family_for_score", "auto")).strip().lower()
        if target_family not in {"auto", "allvars"}:
            raise ValueError("benchmark.eval.target_family_for_score must be one of: auto, allvars")
        primary_metric, primary_mode = resolve_primary_metric_config(eval_cfg)
        return EvalProtocolPlan(
            mode=mode,
            scope=scope,
            frozen_ref_tag=str(eval_protocol_cfg.get("frozen_ref_tag", "")).strip(),
            primary_split=primary_split,
            interp_weight=float(eval_protocol_cfg.get("interp_weight", 0.5)),
            extrap_weight=float(eval_protocol_cfg.get("extrap_weight", 0.5)),
            interp_mode=interp_mode,
            primary_metric=primary_metric,
            primary_mode=primary_mode,
            target_family_for_score=target_family,
            target_vars_for_score=_resolve_target_vars_for_score(eval_cfg, y_vars),
            region_bands=_resolve_region_bands(eval_cfg),
        )

    def inference_ood_cfg(self) -> dict[str, Any]:
        inference_cfg = dict(self.cfg.get("inference", {}) or {})
        ood_cfg = dict(inference_cfg.get("ood", {"poisson_residual_limit": 1e2}) or {})
        for key in ("qoi", "postprocess", "diagnostics"):
            if key in inference_cfg:
                ood_cfg[key] = dict(inference_cfg.get(key, {}) or {})
        return ood_cfg


def build_leaderboard_header(*, target_vars: list[str], leaderboard: list[dict[str, Any]] | None = None) -> list[str]:
    header = ["model_id"]
    for var_name in target_vars:
        header.extend(
            [
                f"test_rmse_{var_name}",
                f"test_rmse_{var_name}_plasma",
                f"test_r2_{var_name}",
                f"test_r2_{var_name}_plasma",
            ]
        )
    header.extend(
        [
            "surrogate_quality_score",
            "primary_metric",
            "primary_metric_value",
            "target_metrics_valid",
            "primary_metric_reliable",
        ]
    )
    extra_keys = sorted({k for row in list(leaderboard or []) for k in row.keys() if k not in header})
    return header + extra_keys


def build_core_leaderboard_header(*, full_header: list[str], core_keys: set[str]) -> list[str]:
    if not core_keys:
        return list(full_header)
    return [key for key in full_header if key in core_keys]


def resolve_primary_metric_config(eval_cfg: dict[str, Any]) -> tuple[str, str]:
    cfg = dict(eval_cfg or {})
    primary_metric = str(cfg.get("primary_metric", "surrogate_quality_score")).strip()
    if primary_metric == "":
        raise ValueError("benchmark.eval.primary_metric must be a non-empty string")
    primary_mode = str(cfg.get("objective_mode", cfg.get("primary_mode", "min"))).strip().lower()
    if primary_mode not in {"min", "max"}:
        raise ValueError("benchmark.eval.objective_mode must be one of: min, max")
    return primary_metric, primary_mode


def _resolve_region_bands(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    region_bands_cfg = dict(eval_cfg.get("region_bands", {}))
    mode = str(region_bands_cfg.get("mode", "fixed_px")).strip().lower()
    if mode not in {"fixed_px", "signed_quantile"}:
        raise ValueError("benchmark.eval.region_bands.mode must be one of: fixed_px, signed_quantile")
    return {
        "mode": mode,
        "boundary_in_px": float(region_bands_cfg.get("boundary_in_px", 2.0)),
        "deep_plasma_px": float(region_bands_cfg.get("deep_plasma_px", 10.0)),
        "boundary_q": float(region_bands_cfg.get("boundary_q", 0.15)),
        "deep_q": float(region_bands_cfg.get("deep_q", 0.70)),
    }


def _resolve_target_vars_for_score(eval_cfg: dict[str, Any], y_vars: list[str]) -> list[str]:
    raw = eval_cfg.get("target_vars_for_score", "auto")
    if raw in (None, "auto", "all"):
        return list(y_vars)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("benchmark.eval.target_vars_for_score must be a non-empty list or 'auto'")
    resolved = [str(v) for v in raw]
    unknown = [v for v in resolved if v not in set(y_vars)]
    if unknown:
        raise ValueError(
            "benchmark.eval.target_vars_for_score contains vars not present in output layout: "
            f"{unknown}; available={y_vars}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("benchmark.eval.target_vars_for_score must not contain duplicates")
    return resolved


__all__ = [
    "BenchmarkPlanBuilder",
    "EvalProtocolPlan",
    "build_core_leaderboard_header",
    "build_leaderboard_header",
    "resolve_primary_metric_config",
]
