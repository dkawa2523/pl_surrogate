"""Simple optimization loop for inference objective."""

from __future__ import annotations

import csv
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from plasma_surrogate.core.input_modes import TABLE_ONLY, TABLE_PLUS_STRUCTURE
from plasma_surrogate.data.geometry_provider import PROVIDER_MODE_FIXED, PROVIDER_MODE_PARAMETRIC_PARTS
from plasma_surrogate.infer.objectives import (
    ObjectiveEvaluation,
    evaluate_objective,
    objective_identity_from_config,
)

_PART_KEY_RE = re.compile(r"^part\.[A-Za-z0-9_\-]+\.(tx|ty|scale_x|scale_y|rotation_deg|fillet)$")
_LAYOUT_KEY_RE = re.compile(r"^layout\.[A-Za-z0-9_\-]+\.(r_center|z_center|width|height)$")
_GAP_KEY_RE = re.compile(r"^gap\.[A-Za-z0-9_\-]+$")
_OFFSET_KEY_RE = re.compile(r"^offset\.(x|y|tx|ty)$")


def _normalize_search_space(
    space: dict[str, tuple[float, float]] | None,
    *,
    label: str,
    allow_empty: bool,
    check_geom_keys: bool = False,
) -> dict[str, tuple[float, float]]:
    raw = dict(space or {})
    if not raw and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    out: dict[str, tuple[float, float]] = {}
    for key, bounds in raw.items():
        name = str(key).strip()
        if not name:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{label}[{name!r}] must be [lo, hi]")
        lo = float(bounds[0])
        hi = float(bounds[1])
        if (not np.isfinite(lo)) or (not np.isfinite(hi)):
            raise ValueError(f"{label}[{name!r}] bounds must be finite; got={bounds!r}")
        if lo > hi:
            raise ValueError(f"{label}[{name!r}] requires lo <= hi; got={bounds!r}")
        if check_geom_keys:
            if (
                _PART_KEY_RE.match(name) is None
                and _LAYOUT_KEY_RE.match(name) is None
                and _GAP_KEY_RE.match(name) is None
                and _OFFSET_KEY_RE.match(name) is None
            ):
                raise ValueError(
                    "geom_space keys must be low-dimensional geometry params: "
                    "part.<id>.(tx|ty|scale_x|scale_y|rotation_deg|fillet), "
                    "layout.<id>.(r_center|z_center|width|height), gap.<name>, offset.(x|y|tx|ty). "
                    f"got={name!r}"
                )
        out[name] = (lo, hi)
    return {k: out[k] for k in sorted(out.keys())}


def cond_space_from_stats(
    cond_order: list[str],
    cond_stats: dict[str, Any] | None,
    *,
    cfg_key: str = "inference.optimize.space",
) -> dict[str, tuple[float, float]]:
    """Build safe condition bounds from preprocessing stats."""

    stats_map = dict(cond_stats or {})
    out: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for key in cond_order:
        stats = stats_map.get(key)
        if not isinstance(stats, dict) or "min" not in stats or "max" not in stats:
            missing.append(str(key))
            continue
        lo = float(stats["min"])
        hi = float(stats["max"])
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo > hi:
            raise ValueError(f"{cfg_key} stats for {key!r} must provide finite min <= max")
        out[str(key)] = (lo, hi)
    if missing:
        raise ValueError(
            f"{cfg_key} is required because preprocessing cond_stats is missing bounds for: {missing}"
        )
    return {k: out[k] for k in sorted(out.keys())}


def _materialize_geom_ref(geom_ref: dict[str, Any], geom_param: dict[str, float]) -> dict[str, Any]:
    out = {"geom_id": str(geom_ref.get("geom_id", "default"))}
    if "geom_param" in geom_ref:
        if geom_param:
            raise ValueError(
                "optimize.geom.geom_param cannot be combined with optimize.geom_space; "
                "provide only one source of geometry params"
            )
        raw = geom_ref.get("geom_param")
        if not isinstance(raw, dict):
            raise TypeError("geom.geom_param must be dict when provided")
        out["geom_param"] = {str(k): float(v) for k, v in sorted(dict(raw).items())}
        return out
    if geom_param:
        out["geom_param"] = {k: float(v) for k, v in sorted(geom_param.items())}
    return out


def validate_optimize_geom_contract(
    *,
    input_mode: str,
    provider_mode: str,
    geom_space: dict[str, tuple[float, float]] | None,
    geom_ref: dict[str, Any] | None,
    label_prefix: str = "inference.optimize",
) -> dict[str, tuple[float, float]]:
    """Validate input-mode/provider contract for geometry optimization knobs."""

    mode = str(input_mode).strip().lower()
    provider = str(provider_mode).strip().lower() or PROVIDER_MODE_FIXED
    geom_space_norm = dict(geom_space or {})
    has_geom_space = len(geom_space_norm) > 0
    has_geom_param = isinstance(geom_ref, dict) and ("geom_param" in geom_ref)

    if has_geom_space and mode == TABLE_ONLY:
        raise ValueError("runtime.input_mode=table_only does not allow inference.optimize.geom_space")
    if has_geom_space and mode != TABLE_PLUS_STRUCTURE:
        raise ValueError(f"{label_prefix}.geom_space requires runtime.input_mode=table_plus_structure")
    if has_geom_space and provider != PROVIDER_MODE_PARAMETRIC_PARTS:
        raise ValueError(
            f"{label_prefix}.geom_space requires provider_mode=parametric_parts; "
            f"got={provider!r}"
        )
    if has_geom_space and has_geom_param:
        raise ValueError(f"{label_prefix}.geom.geom_param cannot be combined with {label_prefix}.geom_space")

    return geom_space_norm


@dataclass
class OptimizeResult:
    best_cond: dict[str, float]
    best_geom_param: dict[str, float]
    trials: list[dict[str, Any]]
    backend: str = "random"
    backend_cfg: dict[str, Any] = field(default_factory=dict)
    objective_key: str = "uniformity"
    objective_mode: str = "weighted_sum"
    best_objective_value: float | None = None
    best_search_value: float | None = None
    best_feasible: bool = True
    best_violated_constraints: list[str] = field(default_factory=list)
    output_cfg: dict[str, Any] = field(default_factory=dict)
    invalid_trial_count: int = 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw or {})


def _trial_record(
    *,
    cond: dict[str, float],
    geom_param: dict[str, float],
    evaluation: ObjectiveEvaluation,
    result: Any,
) -> dict[str, Any]:
    qoi = dict(getattr(result, "qoi", {}) or {})
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    record = {
        "cond": cond,
        "geom_param": geom_param,
        "objective_value": float(evaluation.objective_value),
        "search_value": float(evaluation.search_value),
        "objective_mode": str(evaluation.objective_mode),
        "objective_key": str(evaluation.objective_key),
        "feasible": bool(evaluation.feasible),
        "violated_constraints": list(evaluation.violated_constraints),
        "constraint_violation_total": float(evaluation.constraint_violation_total),
    }
    for key, raw_value in qoi.items():
        scalar = _float_or_none(raw_value)
        if scalar is not None:
            record[f"qoi_{key}"] = float(scalar)
    for key, raw_value in diagnostics.items():
        scalar = _float_or_none(raw_value)
        if scalar is not None:
            record[f"diagnostic_{key}"] = float(scalar)
    for key, raw_value in dict(evaluation.parts or {}).items():
        record[str(key)] = float(raw_value)
    for key, raw_value in dict(evaluation.constraint_values or {}).items():
        record[f"constraint_{key}"] = float(raw_value)
    return record


def _trial_error_record(
    *,
    cond: dict[str, float],
    geom_param: dict[str, float],
    error: Exception,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cond": cond,
        "geom_param": geom_param,
        "objective_value": float("inf"),
        "search_value": float("inf"),
        "feasible": False,
        "violated_constraints": ["trial_error"],
        "constraint_violation_total": float("inf"),
        "error": str(error),
    }
    return record


def _normalize_output_cfg(output_cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(output_cfg or {})
    mode = str(raw.get("save_fields", "all")).strip().lower() or "all"
    if mode not in {"all", "top_k", "none"}:
        raise ValueError("inference.optimize.output.save_fields must be one of: all, top_k, none")
    top_k = int(raw.get("top_k", 3))
    if top_k < 1:
        raise ValueError("inference.optimize.output.top_k must be >= 1")
    return {"save_fields": mode, "top_k": int(top_k)}


def _single_run_aggregated(
    engine: Any,
    *,
    cond: dict[str, float],
    geom: dict[str, Any],
    axis: dict[str, Any],
    save_outputs: bool,
) -> Any:
    fn = engine.single_run_aggregated
    try:
        params = inspect.signature(fn).parameters
        supports_save_outputs = "save_outputs" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        supports_save_outputs = False
    if supports_save_outputs:
        return fn(cond=cond, geom=geom, axis=axis, save_outputs=save_outputs)
    return fn(cond=cond, geom=geom, axis=axis)


def _trial_objective_value(record: dict[str, Any]) -> float:
    value = _float_or_none(record.get("objective_value"))
    return float(value if value is not None else float("inf"))


def _trial_search_value(record: dict[str, Any]) -> float:
    value = _float_or_none(record.get("search_value"))
    if value is None:
        value = _trial_objective_value(record)
    return float(value if value is not None else float("inf"))


def _best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {"cond": {}, "geom_param": {}, "objective_value": float("inf"), "feasible": False}
    feasible = [t for t in trials if bool(t.get("feasible", False))]
    pool = feasible if feasible else list(trials)
    return min(pool, key=_trial_objective_value)


def _evaluate_candidate(
    engine: Any,
    *,
    cond: dict[str, float],
    geom_param: dict[str, float],
    geom_ref: dict[str, Any],
    axis: dict[str, Any],
    objective_cfg: dict[str, Any],
    constraints_cfg: Any,
    save_outputs: bool,
) -> tuple[dict[str, Any], bool]:
    try:
        result = _single_run_aggregated(
            engine,
            cond=cond,
            geom=_materialize_geom_ref(geom_ref, geom_param),
            axis=axis,
            save_outputs=save_outputs,
        )
    except ValueError as exc:
        return (
            _trial_error_record(cond=cond, geom_param=geom_param, error=exc),
            True,
        )
    evaluation = evaluate_objective(result, objective_cfg=objective_cfg, constraints_cfg=constraints_cfg)
    invalid = not np.isfinite(float(evaluation.objective_value))
    return _trial_record(cond=cond, geom_param=geom_param, evaluation=evaluation, result=result), invalid


def _result_from_trials(
    *,
    trials: list[dict[str, Any]],
    backend: str,
    backend_cfg: dict[str, Any] | None,
    objective_cfg: dict[str, Any],
    output_cfg: dict[str, Any],
    invalid_trial_count: int,
) -> OptimizeResult:
    best = _best_trial(trials)
    best_objective_value = _trial_objective_value(best)
    best_search_value = _trial_search_value(best)
    objective_key, objective_mode = objective_identity_from_config(objective_cfg)
    return OptimizeResult(
        best_cond=dict(best.get("cond", {}) or {}),
        best_geom_param=dict(best.get("geom_param", {}) or {}),
        best_objective_value=float(best_objective_value),
        best_search_value=float(best_search_value),
        best_feasible=bool(best.get("feasible", False)),
        best_violated_constraints=[str(v) for v in list(best.get("violated_constraints", []) or [])],
        trials=trials,
        backend=backend,
        backend_cfg=dict(backend_cfg or {}),
        objective_key=str(best.get("objective_key", objective_key) or objective_key),
        objective_mode=str(best.get("objective_mode", objective_mode) or objective_mode),
        output_cfg=dict(output_cfg),
        invalid_trial_count=int(invalid_trial_count),
    )


def _sample_uniform(
    rng: np.random.Generator,
    space: dict[str, tuple[float, float]],
) -> dict[str, float]:
    return {k: float(rng.uniform(v[0], v[1])) for k, v in space.items()}


def _shrink_space_around(
    center: dict[str, float],
    space: dict[str, tuple[float, float]],
    *,
    radius_frac: float,
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    radius = max(float(radius_frac), 0.0)
    for key, bounds in space.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        width = hi - lo
        c = float(center.get(key, 0.5 * (lo + hi)))
        half = 0.5 * width * radius
        out[key] = (max(lo, c - half), min(hi, c + half))
    return out


class _RandomBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any],
        seed: int,
        backend_cfg: dict[str, Any] | None = None,
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        rng = np.random.default_rng(seed)
        objective = _dict_or_empty(objective_cfg)
        output_norm = _normalize_output_cfg(output_cfg)
        save_outputs = output_norm["save_fields"] == "all"
        trials: list[dict[str, Any]] = []
        invalid_trial_count = 0

        for _ in range(n_trials):
            cond = _sample_uniform(rng, space)
            geom_param = _sample_uniform(rng, geom_space)
            record, invalid = _evaluate_candidate(
                engine,
                cond=cond,
                geom_param=geom_param,
                geom_ref=geom_ref,
                axis=axis,
                objective_cfg=objective,
                constraints_cfg=constraints_cfg,
                save_outputs=save_outputs,
            )
            trials.append(record)
            invalid_trial_count += int(bool(invalid))
        return _result_from_trials(
            trials=trials,
            backend="random",
            backend_cfg=dict(backend_cfg or {}),
            objective_cfg=objective,
            output_cfg=output_norm,
            invalid_trial_count=invalid_trial_count,
        )


class _OptunaBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any],
        seed: int,
        backend_cfg: dict[str, Any] | None = None,
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("optuna backend is not available; install optuna to use backend='optuna'") from exc

        cfg = dict(backend_cfg or {})
        objective_cfg_norm = _dict_or_empty(objective_cfg)
        output_norm = _normalize_output_cfg(output_cfg)
        save_outputs = output_norm["save_fields"] == "all"
        sampler_name = str(cfg.get("sampler", "tpe"))
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=int(cfg.get("n_startup_trials", 5)),
                multivariate=bool(cfg.get("multivariate", False)),
            )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        trials: list[dict[str, Any]] = []
        invalid_trial_count = 0

        def objective(trial: Any) -> float:
            nonlocal invalid_trial_count
            cond = {k: float(trial.suggest_float(k, v[0], v[1])) for k, v in space.items()}
            geom_param = {k: float(trial.suggest_float(k, v[0], v[1])) for k, v in geom_space.items()}
            record, invalid = _evaluate_candidate(
                engine,
                cond=cond,
                geom_param=geom_param,
                geom_ref=geom_ref,
                axis=axis,
                objective_cfg=objective_cfg_norm,
                constraints_cfg=constraints_cfg,
                save_outputs=save_outputs,
            )
            trials.append(record)
            invalid_trial_count += int(bool(invalid))
            return _trial_search_value(record)

        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        return _result_from_trials(
            trials=trials,
            backend="optuna",
            backend_cfg=dict(cfg),
            objective_cfg=objective_cfg_norm,
            output_cfg=output_norm,
            invalid_trial_count=invalid_trial_count,
        )


class _CsvBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any],
        seed: int,
        backend_cfg: dict[str, Any] | None = None,
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        del seed  # deterministic by CSV row order
        cfg = dict(backend_cfg or {})
        objective_cfg_norm = _dict_or_empty(objective_cfg)
        output_norm = _normalize_output_cfg(output_cfg)
        save_outputs = output_norm["save_fields"] == "all"
        csv_path = cfg.get("csv_path")
        if not csv_path:
            raise ValueError("csv backend requires backend_cfg.csv_path")
        cond_keys = list(space.keys())
        geom_keys = list(geom_space.keys())
        deduplicate = bool(cfg.get("deduplicate", True))

        path = Path(str(csv_path))
        if not path.exists():
            raise FileNotFoundError(f"csv backend file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("csv backend requires header row")
            missing = [k for k in [*cond_keys, *geom_keys] if k not in reader.fieldnames]
            if missing:
                raise ValueError(f"csv backend missing required columns: {missing}")
            rows = list(reader)

        candidates: list[tuple[dict[str, float], dict[str, float]]] = []
        seen: set[tuple[float, ...]] = set()
        for row in rows:
            cond: dict[str, float] = {}
            for key in cond_keys:
                try:
                    cond[key] = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"csv backend invalid value for column '{key}': {row.get(key)!r}") from exc
                if not np.isfinite(cond[key]):
                    raise ValueError(f"csv backend non-finite value for column '{key}': {cond[key]!r}")
            geom_param: dict[str, float] = {}
            for key in geom_keys:
                try:
                    geom_param[key] = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"csv backend invalid value for column '{key}': {row.get(key)!r}") from exc
                if not np.isfinite(geom_param[key]):
                    raise ValueError(f"csv backend non-finite value for column '{key}': {geom_param[key]!r}")
            signature = tuple([*(cond[k] for k in cond_keys), *(geom_param[k] for k in geom_keys)])
            if deduplicate and signature in seen:
                continue
            seen.add(signature)
            candidates.append((cond, geom_param))

        if n_trials > 0:
            candidates = candidates[: int(n_trials)]
        if len(candidates) == 0:
            raise ValueError("csv backend produced no candidate rows")

        trials: list[dict[str, Any]] = []
        invalid_trial_count = 0
        for cond, geom_param in candidates:
            record, invalid = _evaluate_candidate(
                engine,
                cond=cond,
                geom_param=geom_param,
                geom_ref=geom_ref,
                axis=axis,
                objective_cfg=objective_cfg_norm,
                constraints_cfg=constraints_cfg,
                save_outputs=save_outputs,
            )
            trials.append(record)
            invalid_trial_count += int(bool(invalid))

        return _result_from_trials(
            trials=trials,
            backend="csv",
            backend_cfg={
                "csv_path": str(path),
                "deduplicate": deduplicate,
                "n_candidates": len(candidates),
            },
            objective_cfg=objective_cfg_norm,
            output_cfg=output_norm,
            invalid_trial_count=invalid_trial_count,
        )


class _TwoStageBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any],
        seed: int,
        backend_cfg: dict[str, Any] | None = None,
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        cfg = dict(backend_cfg or {})
        objective_cfg_norm = _dict_or_empty(objective_cfg)
        output_norm = _normalize_output_cfg(output_cfg)
        save_outputs = output_norm["save_fields"] == "all"
        rng = np.random.default_rng(seed)
        n_initial_cfg = int(cfg.get("n_initial", max(1, n_trials // 2)))
        n_initial = max(1, min(int(n_trials), n_initial_cfg))
        top_k = max(1, int(cfg.get("top_k", 8)))
        local_trials_per_seed = max(1, int(cfg.get("local_trials_per_seed", 8)))
        radius_frac = float(cfg.get("local_radius_frac", 0.25))
        if not np.isfinite(radius_frac) or radius_frac < 0.0:
            raise ValueError("backend_cfg.local_radius_frac must be finite and >= 0")
        trials: list[dict[str, Any]] = []
        invalid_trial_count = 0

        def eval_one(cond: dict[str, float], geom_param: dict[str, float]) -> dict[str, Any]:
            nonlocal invalid_trial_count
            record, invalid = _evaluate_candidate(
                engine,
                cond=cond,
                geom_param=geom_param,
                geom_ref=geom_ref,
                axis=axis,
                objective_cfg=objective_cfg_norm,
                constraints_cfg=constraints_cfg,
                save_outputs=save_outputs,
            )
            invalid_trial_count += int(bool(invalid))
            return record

        for _ in range(n_initial):
            trials.append(eval_one(_sample_uniform(rng, space), _sample_uniform(rng, geom_space)))

        finite_trials = sorted(
            [t for t in trials if np.isfinite(_trial_search_value(t))],
            key=lambda r: (0 if bool(r.get("feasible", False)) else 1, _trial_search_value(r)),
        )
        seeds = finite_trials[: min(top_k, len(finite_trials))]
        if seeds:
            for seed_record in seeds:
                cond_center = dict(seed_record.get("cond", {}) or {})
                geom_center = dict(seed_record.get("geom_param", {}) or {})
                cond_local_space = _shrink_space_around(cond_center, space, radius_frac=radius_frac)
                geom_local_space = _shrink_space_around(geom_center, geom_space, radius_frac=radius_frac)
                for _ in range(local_trials_per_seed):
                    if len(trials) >= int(n_trials):
                        break
                    trials.append(eval_one(_sample_uniform(rng, cond_local_space), _sample_uniform(rng, geom_local_space)))
                if len(trials) >= int(n_trials):
                    break

        return _result_from_trials(
            trials=trials,
            backend="two_stage",
            backend_cfg=dict(cfg),
            objective_cfg=objective_cfg_norm,
            output_cfg=output_norm,
            invalid_trial_count=invalid_trial_count,
        )


class OptimizeRunner:
    def __init__(self, engine: Any):
        self.engine = engine

    def run(
        self,
        space: dict[str, tuple[float, float]],
        geom_space: dict[str, tuple[float, float]] | None,
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any] | None = None,
        seed: int = 0,
        backend: str = "random",
        backend_cfg: dict[str, Any] | None = None,
        objective_cfg: dict[str, Any] | None = None,
        constraints_cfg: Any = None,
        output_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        n_trials_int = int(n_trials)
        if n_trials_int < 1:
            raise ValueError("n_trials must be >= 1")
        space_norm = _normalize_search_space(space, label="space", allow_empty=False, check_geom_keys=False)
        geom_space_norm = _normalize_search_space(
            geom_space,
            label="geom_space",
            allow_empty=True,
            check_geom_keys=True,
        )
        overlap = sorted(set(space_norm.keys()) & set(geom_space_norm.keys()))
        if overlap:
            raise ValueError(f"space and geom_space keys must be disjoint; overlap={overlap}")
        axis_payload = axis or {"mode": "steady", "value": 0.0}
        key = str(backend).lower()
        if key == "random":
            return _RandomBackend.run(
                self.engine,
                space=space_norm,
                geom_space=geom_space_norm,
                n_trials=n_trials_int,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
                objective_cfg=objective_cfg,
                constraints_cfg=constraints_cfg,
                output_cfg=output_cfg,
            )
        if key == "optuna":
            return _OptunaBackend.run(
                self.engine,
                space=space_norm,
                geom_space=geom_space_norm,
                n_trials=n_trials_int,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
                objective_cfg=objective_cfg,
                constraints_cfg=constraints_cfg,
                output_cfg=output_cfg,
            )
        if key == "csv":
            return _CsvBackend.run(
                self.engine,
                space=space_norm,
                geom_space=geom_space_norm,
                n_trials=n_trials_int,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
                objective_cfg=objective_cfg,
                constraints_cfg=constraints_cfg,
                output_cfg=output_cfg,
            )
        if key == "two_stage":
            return _TwoStageBackend.run(
                self.engine,
                space=space_norm,
                geom_space=geom_space_norm,
                n_trials=n_trials_int,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
                objective_cfg=objective_cfg,
                constraints_cfg=constraints_cfg,
                output_cfg=output_cfg,
            )
        raise ValueError(f"Unsupported optimize backend: {backend}")
