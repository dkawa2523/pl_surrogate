"""Simple optimization loop for inference objective."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from plasma_surrogate.core.input_modes import TABLE_ONLY, TABLE_PLUS_STRUCTURE
from plasma_surrogate.data.geometry_provider import PROVIDER_MODE_FIXED, PROVIDER_MODE_PARAMETRIC_PARTS

_PART_KEY_RE = re.compile(r"^part\.[A-Za-z0-9_\-]+\.(tx|ty|scale_x|scale_y|rotation_deg|fillet)$")
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
            if _PART_KEY_RE.match(name) is None and _GAP_KEY_RE.match(name) is None and _OFFSET_KEY_RE.match(name) is None:
                raise ValueError(
                    "geom_space keys must be low-dimensional geometry params: "
                    "part.<id>.(tx|ty|scale_x|scale_y|rotation_deg|fillet), gap.<name>, offset.(x|y|tx|ty). "
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
    best_value: float
    trials: list[dict[str, Any]]
    backend: str = "random"
    backend_cfg: dict[str, Any] = field(default_factory=dict)
    objective_key: str = "uniformity"
    invalid_trial_count: int = 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _trial_record(
    *,
    cond: dict[str, float],
    geom_param: dict[str, float],
    value: float,
    result: Any,
) -> dict[str, Any]:
    qoi = dict(getattr(result, "qoi", {}) or {})
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    warnings = [str(w) for w in list(getattr(result, "warnings", []) or [])]
    return {
        "cond": cond,
        "geom_param": geom_param,
        "value": float(value),
        "boundary_gamma_uniformity": _float_or_none(qoi.get("boundary_gamma_uniformity")),
        "poisson_residual_norm": _float_or_none(diagnostics.get("poisson_residual_norm")),
        "bc_phi_mae": _float_or_none(diagnostics.get("bc_phi_mae")),
        "boundary_operator_proxy_loss": _float_or_none(diagnostics.get("boundary_operator_proxy_loss")),
        "warnings": ";".join(warnings),
    }


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
    ) -> OptimizeResult:
        rng = np.random.default_rng(seed)
        trials: list[dict[str, Any]] = []
        best_value = float("inf")
        best_cond: dict[str, float] = {}
        best_geom_param: dict[str, float] = {}
        invalid_trial_count = 0

        for _ in range(n_trials):
            cond = {k: float(rng.uniform(v[0], v[1])) for k, v in space.items()}
            geom_param = {k: float(rng.uniform(v[0], v[1])) for k, v in geom_space.items()}
            result = engine.single_run_aggregated(cond=cond, geom=_materialize_geom_ref(geom_ref, geom_param), axis=axis)
            value = float(result.qoi["uniformity"])
            if not np.isfinite(value):
                invalid_trial_count += 1
                value = float("inf")
            trials.append(_trial_record(cond=cond, geom_param=geom_param, value=value, result=result))
            if (not best_cond) or value < best_value:
                best_value = value
                best_cond = cond
                best_geom_param = geom_param
        return OptimizeResult(
            best_cond=best_cond,
            best_geom_param=best_geom_param,
            best_value=best_value,
            trials=trials,
            backend="random",
            backend_cfg=dict(backend_cfg or {}),
            objective_key="uniformity",
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
    ) -> OptimizeResult:
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("optuna backend is not available; install optuna to use backend='optuna'") from exc

        cfg = dict(backend_cfg or {})
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
            result = engine.single_run_aggregated(
                cond=cond,
                geom=_materialize_geom_ref(geom_ref, geom_param),
                axis=axis,
            )
            value = float(result.qoi["uniformity"])
            if not np.isfinite(value):
                invalid_trial_count += 1
                value = float("inf")
            trials.append(_trial_record(cond=cond, geom_param=geom_param, value=value, result=result))
            return value

        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        return OptimizeResult(
            best_cond={k: float(study.best_params[k]) for k in space.keys() if k in study.best_params},
            best_geom_param={k: float(study.best_params[k]) for k in geom_space.keys() if k in study.best_params},
            best_value=float(study.best_value),
            trials=trials,
            backend="optuna",
            backend_cfg=dict(cfg),
            objective_key="uniformity",
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
    ) -> OptimizeResult:
        del seed  # deterministic by CSV row order
        cfg = dict(backend_cfg or {})
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
        best_value = float("inf")
        best_cond: dict[str, float] = {}
        best_geom_param: dict[str, float] = {}
        invalid_trial_count = 0
        for cond, geom_param in candidates:
            result = engine.single_run_aggregated(
                cond=cond,
                geom=_materialize_geom_ref(geom_ref, geom_param),
                axis=axis,
            )
            value = float(result.qoi["uniformity"])
            if not np.isfinite(value):
                invalid_trial_count += 1
                value = float("inf")
            trials.append(_trial_record(cond=cond, geom_param=geom_param, value=value, result=result))
            if (not best_cond) or value < best_value:
                best_value = value
                best_cond = cond
                best_geom_param = geom_param

        return OptimizeResult(
            best_cond=best_cond,
            best_geom_param=best_geom_param,
            best_value=best_value,
            trials=trials,
            backend="csv",
            backend_cfg={
                "csv_path": str(path),
                "deduplicate": deduplicate,
                "n_candidates": len(candidates),
            },
            objective_key="uniformity",
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
            )
        raise ValueError(f"Unsupported optimize backend: {backend}")
