"""Observation wrappers for profile-based surrogate data assimilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_surrogate.infer.profiles import extract_radial_profile


@dataclass(frozen=True)
class LinearNeProfileObservation:
    """Electron-density observations and a fixed linear-error covariance."""

    r_m: np.ndarray
    z_m: float
    ne_obs: np.ndarray
    density_scale: float
    covariance_scaled: np.ndarray
    case_id: str = ""

    def __post_init__(self) -> None:
        r = np.asarray(self.r_m, dtype=np.float64).reshape(-1)
        ne = np.asarray(self.ne_obs, dtype=np.float64).reshape(-1)
        cov = np.asarray(self.covariance_scaled, dtype=np.float64)
        if r.size == 0 or r.shape != ne.shape:
            raise ValueError("observation r_m and ne_obs must be non-empty vectors of equal length")
        if cov.shape != (r.size, r.size):
            raise ValueError(f"observation covariance must have shape={(r.size, r.size)}; got={cov.shape}")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(ne)):
            raise ValueError("observation locations and electron densities must be finite")
        if not np.isfinite(float(self.density_scale)) or float(self.density_scale) <= 0.0:
            raise ValueError("observation density_scale must be finite and > 0")
        if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("observation covariance must be finite and symmetric")
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("observation covariance must be positive definite") from exc


def build_linear_profile_covariance(
    ne_obs: np.ndarray,
    *,
    point_rel_sigma: float,
    calibration_rel_sigma: float,
    floor_rel_sigma: float,
    surrogate_sigma_scaled: np.ndarray | None = None,
    jitter: float = 1.0e-10,
) -> tuple[float, np.ndarray]:
    """Build a fixed covariance without transforming electron density to log space."""

    ne = np.asarray(ne_obs, dtype=np.float64).reshape(-1)
    if ne.size == 0 or not np.all(np.isfinite(ne)):
        raise ValueError("ne_obs must be a non-empty finite vector")
    scale = float(np.sqrt(np.mean(ne * ne)))
    if scale <= 0.0:
        raise ValueError("ne_obs RMS must be > 0")
    point = float(point_rel_sigma)
    calibration = float(calibration_rel_sigma)
    floor = float(floor_rel_sigma)
    if min(point, calibration, floor) < 0.0:
        raise ValueError("profile uncertainty scales must be >= 0")
    if surrogate_sigma_scaled is None:
        surrogate = np.zeros_like(ne)
    else:
        surrogate = np.asarray(surrogate_sigma_scaled, dtype=np.float64).reshape(-1)
        if surrogate.shape != ne.shape or np.any(surrogate < 0.0) or not np.all(np.isfinite(surrogate)):
            raise ValueError("surrogate_sigma_scaled must be finite, non-negative, and match ne_obs")
    ne_scaled = ne / scale
    diagonal = (point * np.abs(ne_scaled)) ** 2 + floor**2 + surrogate**2
    common = calibration * ne_scaled
    cov = np.diag(diagonal) + np.outer(common, common)
    cov += np.eye(ne.size, dtype=np.float64) * float(jitter)
    return scale, cov


class LinearNeProfileAssimilationEngine:
    """Attach a fixed, physical-unit electron-density objective to an engine."""

    def __init__(
        self,
        base_engine: Any,
        *,
        observation: LinearNeProfileObservation,
        fixed_cond: dict[str, float],
        setpoints: dict[str, float],
        nuisance_rel_sigma: dict[str, float],
    ) -> None:
        self.base_engine = base_engine
        self.observation = observation
        self.fixed_cond = {str(k): float(v) for k, v in fixed_cond.items()}
        self.setpoints = {str(k): float(v) for k, v in setpoints.items()}
        self.nuisance_rel_sigma = {str(k): float(v) for k, v in nuisance_rel_sigma.items()}
        if set(self.setpoints) != set(self.nuisance_rel_sigma):
            raise ValueError("setpoints and nuisance_rel_sigma must have the same keys")
        for key, value in self.setpoints.items():
            if not np.isfinite(value) or value == 0.0:
                raise ValueError(f"setpoint {key!r} must be finite and non-zero")
            sigma = self.nuisance_rel_sigma[key]
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"nuisance relative sigma for {key!r} must be finite and > 0")
        self._chol = np.linalg.cholesky(np.asarray(observation.covariance_scaled, dtype=np.float64))

    def single_run_aggregated(
        self,
        cond: dict[str, float],
        geom: dict[str, Any],
        axis: dict[str, Any],
        save_outputs: bool = False,
    ) -> Any:
        overlap = sorted(set(cond) & set(self.fixed_cond))
        if overlap:
            raise ValueError(f"searched conditions overlap fixed assimilation conditions: {overlap}")
        merged = {**self.fixed_cond, **{str(k): float(v) for k, v in cond.items()}}
        result = self.base_engine.single_run_aggregated(
            cond=merged,
            geom=geom,
            axis=axis,
            save_outputs=save_outputs,
        )
        geom_ref = self.base_engine._validate_geom_ref(geom)
        geom_ctx = self.base_engine._get_geom_ctx(geom_ref, axis=axis)
        profile = extract_radial_profile(
            result.fields_phys["ne"],
            geom_ctx,
            z_m=self.observation.z_m,
            r_m=self.observation.r_m,
            require_plasma=True,
        )
        if not np.all(profile.valid) or not np.all(np.isfinite(profile.values)):
            raise ValueError("candidate electron-density profile is invalid at one or more observation points")

        obs = np.asarray(self.observation.ne_obs, dtype=np.float64)
        pred = np.asarray(profile.values, dtype=np.float64)
        residual_scaled = (pred - obs) / float(self.observation.density_scale)
        whitened = np.linalg.solve(self._chol, residual_scaled)
        chi2 = float(np.dot(whitened, whitened))
        prior_parts: dict[str, float] = {}
        for key, setpoint in self.setpoints.items():
            relative_offset = (float(merged[key]) / float(setpoint)) - 1.0
            prior_parts[key] = float((relative_offset / self.nuisance_rel_sigma[key]) ** 2)
        prior_total = float(sum(prior_parts.values()))
        rel_l2 = float(np.linalg.norm(pred - obs) / max(np.linalg.norm(obs), 1.0e-30))
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))

        qoi = dict(getattr(result, "qoi", {}) or {})
        qoi.update(
            {
                "assimilation_objective_linear": chi2 + prior_total,
                "profile_linear_chi2": chi2,
                "profile_linear_reduced_chi2": chi2 / float(obs.size),
                "profile_linear_rel_l2": rel_l2,
                "profile_linear_rmse": rmse,
                "nuisance_prior_total": prior_total,
            }
        )
        for key, value in prior_parts.items():
            qoi[f"nuisance_prior_{key}"] = float(value)
        result.qoi = qoi
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        diagnostics["assimilation_profile_sample_count"] = float(obs.size)
        result.diagnostics = diagnostics
        return result


__all__ = [
    "LinearNeProfileAssimilationEngine",
    "LinearNeProfileObservation",
    "build_linear_profile_covariance",
]
