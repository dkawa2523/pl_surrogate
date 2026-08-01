from __future__ import annotations

import pytest

from plasma_surrogate.benchmark.runner import BenchmarkRunner, _primary_metric_value


def test_primary_metric_value_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric='missing'.*not present"):
        _primary_metric_value(
            {"model_id": "fno", "surrogate_quality_score": 1.0},
            primary_metric="missing",
            model_name="fno",
        )


def test_primary_metric_value_rejects_non_numeric_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric='surrogate_quality_score'.*must be numeric"):
        _primary_metric_value(
            {"model_id": "fno", "surrogate_quality_score": ""},
            primary_metric="surrogate_quality_score",
            model_name="fno",
        )


def test_validate_eval_scope_train_sections_common_allows_mixed_train_cfg() -> None:
    BenchmarkRunner._validate_eval_scope_train_sections(
        scope="common",
        train_cfg={
            "unet": {"epochs": 1},
            "global_mlp": {"epochs": 1},
            "fno": {"epochs": 1},
            "deeponet_plasma": {"epochs": 1},
        },
    )


@pytest.mark.parametrize(
    ("scope", "train_cfg"),
    [
        ("unet_isolated", {"global_mlp": {"epochs": 1}}),
        ("unetpp_isolated", {"unet": {"epochs": 1}}),
        ("unetpp_attn_isolated", {"unetpp": {"epochs": 1}}),
        ("fno_isolated", {"unet": {"epochs": 1}}),
        ("ffno_isolated", {"fno": {"epochs": 1}}),
        ("deeponet_isolated", {"fno": {"epochs": 1}}),
        ("global_frozen", {"deeponet_plasma": {"epochs": 1}}),
    ],
)
def test_validate_eval_scope_train_sections_rejects_mixed(scope: str, train_cfg: dict) -> None:
    with pytest.raises(ValueError, match="forbids non-empty sections"):
        BenchmarkRunner._validate_eval_scope_train_sections(scope=scope, train_cfg=train_cfg)
