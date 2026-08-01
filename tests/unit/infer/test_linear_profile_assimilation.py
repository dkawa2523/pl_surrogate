from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_surrogate.data.geometry_context import GeometryContext
from plasma_surrogate.infer.assimilation import (
    LinearNeProfileAssimilationEngine,
    LinearNeProfileObservation,
    build_linear_profile_covariance,
)
from plasma_surrogate.infer.profiles import extract_radial_profile


def _geom(mask: np.ndarray | None = None) -> GeometryContext:
    r = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    z = np.asarray([10.0, 20.0], dtype=np.float32)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    plasma = np.ones((2, 3), dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
    return GeometryContext(
        mask_plasma=plasma,
        distance_any=np.ones_like(plasma),
        dist0=np.ones_like(plasma),
        eps=np.ones_like(plasma),
        coord_grid=np.stack([rr, zz], axis=0),
    )


def test_extract_radial_profile_exact_and_between_rows():
    geom = _geom()
    field = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    exact = extract_radial_profile(field, geom, z_m=10.0, r_m=[0.5, 2.0])
    between = extract_radial_profile(field, geom, z_m=15.0, r_m=[0.5, 2.0])

    assert exact.valid.tolist() == [True, True]
    assert exact.values.tolist() == pytest.approx([10.5, 12.0])
    assert between.values.tolist() == pytest.approx([15.5, 17.0])


def test_extract_radial_profile_rejects_non_plasma_stencil():
    geom = _geom(mask=np.asarray([[1, 0, 1], [1, 1, 1]], dtype=np.float32))
    field = np.arange(6, dtype=np.float64).reshape(2, 3)

    profile = extract_radial_profile(field, geom, z_m=15.0, r_m=[0.5, 2.0])

    assert profile.valid.tolist() == [False, True]
    assert np.isnan(profile.values[0])


def test_linear_covariance_uses_fixed_physical_density_scale():
    obs = np.asarray([2.0e15, 4.0e15], dtype=np.float64)
    scale, cov = build_linear_profile_covariance(
        obs,
        point_rel_sigma=0.05,
        calibration_rel_sigma=0.05,
        floor_rel_sigma=0.02,
        surrogate_sigma_scaled=np.asarray([0.01, 0.02]),
    )

    assert scale == pytest.approx(np.sqrt(np.mean(obs**2)))
    assert cov.shape == (2, 2)
    assert cov[0, 1] > 0.0
    np.linalg.cholesky(cov)


class _FakeBaseEngine:
    def __init__(self) -> None:
        self.geom = _geom()

    def _validate_geom_ref(self, geom):  # noqa: ANN001
        return geom

    def _get_geom_ctx(self, geom_ref, axis):  # noqa: ANN001
        return self.geom

    def single_run_aggregated(self, cond, geom, axis, save_outputs=False):  # noqa: ANN001
        value = float(cond["PP0"]) * 10.0
        field = np.full((2, 3), value, dtype=np.float64)
        return SimpleNamespace(fields_phys={"ne": field}, qoi={}, diagnostics={})


def test_linear_assimilation_objective_uses_no_log_transform():
    obs_values = np.asarray([30.0, 30.0], dtype=np.float64)
    observation = LinearNeProfileObservation(
        r_m=np.asarray([0.0, 2.0]),
        z_m=15.0,
        ne_obs=obs_values,
        density_scale=30.0,
        covariance_scaled=np.eye(2, dtype=np.float64),
    )
    engine = LinearNeProfileAssimilationEngine(
        _FakeBaseEngine(),
        observation=observation,
        fixed_cond={"Td": 0.03},
        setpoints={"PP0": 3.0},
        nuisance_rel_sigma={"PP0": 0.05},
    )

    at_truth = engine.single_run_aggregated(
        cond={"PP0": 3.0}, geom={"geom_id": "base4"}, axis={"mode": "steady", "value": 0.0}
    )
    offset = engine.single_run_aggregated(
        cond={"PP0": 3.3}, geom={"geom_id": "base4"}, axis={"mode": "steady", "value": 0.0}
    )

    assert at_truth.qoi["assimilation_objective_linear"] == pytest.approx(0.0)
    # Linear residual: two entries of (33-30)/30=0.1, plus a 2-sigma PP0 prior.
    assert offset.qoi["profile_linear_chi2"] == pytest.approx(0.02)
    assert offset.qoi["nuisance_prior_PP0"] == pytest.approx(4.0)
    assert offset.qoi["assimilation_objective_linear"] == pytest.approx(4.02)
