"""Simple optimization loop for inference objective."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OptimizeResult:
    best_cond: dict[str, float]
    best_value: float
    trials: list[dict[str, Any]]
    backend: str = "random"
    backend_cfg: dict[str, Any] = field(default_factory=dict)
    objective_key: str = "uniformity"


class _RandomBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
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

        for _ in range(n_trials):
            cond = {k: float(rng.uniform(v[0], v[1])) for k, v in space.items()}
            result = engine.single_run_aggregated(cond=cond, geom=geom_ref, axis=axis)
            value = float(result.qoi["uniformity"])
            trials.append({"cond": cond, "value": value})
            if value < best_value:
                best_value = value
                best_cond = cond
        return OptimizeResult(
            best_cond=best_cond,
            best_value=best_value,
            trials=trials,
            backend="random",
            backend_cfg=dict(backend_cfg or {}),
            objective_key="uniformity",
        )


class _OptunaBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any],
        seed: int,
        backend_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        try:
            import optuna
        except Exception as exc:  # pragma: no cover - env dependent
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

        def objective(trial: Any) -> float:
            cond = {k: float(trial.suggest_float(k, v[0], v[1])) for k, v in space.items()}
            result = engine.single_run_aggregated(cond=cond, geom=geom_ref, axis=axis)
            value = float(result.qoi["uniformity"])
            trials.append({"cond": cond, "value": value})
            return value

        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        return OptimizeResult(
            best_cond={k: float(v) for k, v in study.best_params.items()},
            best_value=float(study.best_value),
            trials=trials,
            backend="optuna",
            backend_cfg=dict(cfg),
            objective_key="uniformity",
        )


class _CsvBackend:
    @staticmethod
    def run(
        engine: Any,
        *,
        space: dict[str, tuple[float, float]],
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
        deduplicate = bool(cfg.get("deduplicate", True))

        path = Path(str(csv_path))
        if not path.exists():
            raise FileNotFoundError(f"csv backend file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("csv backend requires header row")
            missing = [k for k in cond_keys if k not in reader.fieldnames]
            if missing:
                raise ValueError(f"csv backend missing required columns: {missing}")
            rows = list(reader)

        candidates: list[dict[str, float]] = []
        seen: set[tuple[float, ...]] = set()
        for row in rows:
            cond: dict[str, float] = {}
            for key in cond_keys:
                try:
                    cond[key] = float(row[key])
                except Exception as exc:
                    raise ValueError(f"csv backend invalid value for column '{key}': {row.get(key)!r}") from exc
                if not np.isfinite(cond[key]):
                    raise ValueError(f"csv backend non-finite value for column '{key}': {cond[key]!r}")
            signature = tuple(cond[k] for k in cond_keys)
            if deduplicate and signature in seen:
                continue
            seen.add(signature)
            candidates.append(cond)

        if n_trials > 0:
            candidates = candidates[: int(n_trials)]
        if len(candidates) == 0:
            raise ValueError("csv backend produced no candidate rows")

        trials: list[dict[str, Any]] = []
        best_value = float("inf")
        best_cond: dict[str, float] = {}
        for cond in candidates:
            result = engine.single_run_aggregated(cond=cond, geom=geom_ref, axis=axis)
            value = float(result.qoi["uniformity"])
            trials.append({"cond": cond, "value": value})
            if value < best_value:
                best_value = value
                best_cond = cond

        return OptimizeResult(
            best_cond=best_cond,
            best_value=best_value,
            trials=trials,
            backend="csv",
            backend_cfg={
                "csv_path": str(path),
                "deduplicate": deduplicate,
                "n_candidates": len(candidates),
            },
            objective_key="uniformity",
        )


class OptimizeRunner:
    def __init__(self, engine: Any):
        self.engine = engine

    def run(
        self,
        space: dict[str, tuple[float, float]],
        n_trials: int,
        geom_ref: dict[str, Any],
        axis: dict[str, Any] | None = None,
        seed: int = 0,
        backend: str = "random",
        backend_cfg: dict[str, Any] | None = None,
    ) -> OptimizeResult:
        axis_payload = axis or {"mode": "steady", "value": 0.0}
        key = str(backend).lower()
        if key == "random":
            return _RandomBackend.run(
                self.engine,
                space=space,
                n_trials=n_trials,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
            )
        if key == "optuna":
            return _OptunaBackend.run(
                self.engine,
                space=space,
                n_trials=n_trials,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
            )
        if key == "csv":
            return _CsvBackend.run(
                self.engine,
                space=space,
                n_trials=n_trials,
                geom_ref=geom_ref,
                axis=axis_payload,
                seed=seed,
                backend_cfg=backend_cfg,
            )
        raise ValueError(f"Unsupported optimize backend: {backend}")
