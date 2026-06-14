from __future__ import annotations

import pytest

from plasma_surrogate.infer.objectives import evaluate_objective


class _Result:
    def __init__(self, qoi: dict[str, float] | None = None, diagnostics: dict[str, float] | None = None):
        self.qoi = dict(qoi or {})
        self.diagnostics = dict(diagnostics or {})


def test_weighted_sum_uniformity_matches_single_term_objective():
    result = _Result(qoi={"uniformity": 0.4})

    evaluation = evaluate_objective(
        result,
        objective_cfg={
            "mode": "weighted_sum",
            "terms": [{"key": "uniformity", "direction": "min", "weight": 1.0}],
        },
    )

    assert evaluation.objective_value == pytest.approx(0.4)
    assert evaluation.search_value == pytest.approx(0.4)
    assert evaluation.feasible is True
    assert evaluation.objective_key == "uniformity"
    assert evaluation.parts["objective_term_uniformity"] == pytest.approx(0.4)


def test_weighted_sum_can_mix_qoi_and_diagnostics():
    result = _Result(
        qoi={"uniformity": 0.5},
        diagnostics={"poisson_residual_norm": 0.1},
    )

    evaluation = evaluate_objective(
        result,
        objective_cfg={
            "mode": "weighted_sum",
            "terms": [
                {"key": "uniformity", "direction": "min", "weight": 1.0},
                {"key": "poisson_residual_norm", "direction": "min", "weight": 0.2},
            ],
        },
    )

    assert evaluation.objective_value == pytest.approx(0.52)
    assert evaluation.objective_key == "weighted_sum"
    assert evaluation.parts["objective_term_poisson_residual_norm"] == pytest.approx(0.02)


def test_weighted_sum_max_direction_is_negative_contribution():
    result = _Result(qoi={"yield": 2.0})

    evaluation = evaluate_objective(
        result,
        objective_cfg={
            "mode": "weighted_sum",
            "terms": [{"key": "yield", "direction": "max", "weight": 0.5}],
        },
    )

    assert evaluation.objective_value == pytest.approx(-1.0)
    assert evaluation.parts["objective_term_yield"] == pytest.approx(-1.0)


def test_scale_and_log1p_abs_transform_normalize_term_values():
    result = _Result(qoi={"residual": -3.0})

    evaluation = evaluate_objective(
        result,
        objective_cfg={
            "mode": "weighted_sum",
            "terms": [
                {
                    "key": "residual",
                    "direction": "min",
                    "weight": 2.0,
                    "scale": 4.0,
                    "transform": "log1p_abs",
                }
            ],
        },
    )

    normalized = 0.25 * 1.3862943611198906
    assert evaluation.term_values["residual"] == pytest.approx(normalized)
    assert evaluation.objective_value == pytest.approx(2.0 * normalized)


def test_constraints_mark_infeasible_and_add_search_penalty():
    result = _Result(
        qoi={"uniformity": 0.3},
        diagnostics={"poisson_residual_norm": 0.2},
    )

    evaluation = evaluate_objective(
        result,
        objective_cfg={"terms": [{"key": "uniformity"}]},
        constraints_cfg=[{"key": "poisson_residual_norm", "upper": 0.05}],
    )

    assert evaluation.feasible is False
    assert evaluation.violated_constraints == ["poisson_residual_norm"]
    assert evaluation.constraint_values["poisson_residual_norm"] == pytest.approx(0.2)
    assert evaluation.constraint_violation_total == pytest.approx(3.0)
    assert evaluation.search_value == pytest.approx(3_000_000.3)


def test_missing_term_key_fails_fast():
    with pytest.raises(ValueError, match="objective metric key"):
        evaluate_objective(_Result(qoi={"uniformity": 0.1}), objective_cfg={"terms": [{"key": "missing"}]})


def test_nonfinite_term_key_fails_fast():
    with pytest.raises(ValueError, match="objective metric key"):
        evaluate_objective(_Result(qoi={"uniformity": float("nan")}), objective_cfg={"terms": [{"key": "uniformity"}]})


def test_missing_constraint_key_fails_fast():
    with pytest.raises(ValueError, match="objective metric key"):
        evaluate_objective(
            _Result(qoi={"uniformity": 0.1}),
            objective_cfg={"terms": [{"key": "uniformity"}]},
            constraints_cfg=[{"key": "missing", "upper": 1.0}],
        )


@pytest.mark.parametrize(
    "legacy_cfg",
    [
        {"main": "uniformity"},
        {"negative_penalty": {"enabled": True}},
        {"physics_penalty": {"enabled": True}},
        {"boundary_penalty": {"enabled": True}},
        {"floor_penalty": {"enabled": True}},
        {"reward": {"enabled": True}},
        {"main": "qoi_ratio", "denominator_key": "peak"},
    ],
)
def test_legacy_objective_keys_are_rejected(legacy_cfg: dict[str, object]):
    with pytest.raises(ValueError, match="legacy objective keys"):
        evaluate_objective(_Result(qoi={"uniformity": 1.0}), objective_cfg=legacy_cfg)
