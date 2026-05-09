from __future__ import annotations

import pytest

from plasma_surrogate.benchmark.runner import BenchmarkRunner, _primary_metric_value
from tests._config_presets import runtime_table_only


def _runner_with_guardrails(*, mode: str = "warn") -> BenchmarkRunner:
    return BenchmarkRunner(
        {
            "runtime": runtime_table_only(),
            "benchmark": {
                "guardrails": {
                    "enabled": True,
                    "mode": mode,
                    "checks": {"global_disabled_boost": True, "effective_steps_floor": True},
                },
                "effective_steps_floor": {
                    "global_mlp": 60,
                    "fno": 60,
                },
            }
        }
    )


def test_guardrail_detects_global_disabled_boost() -> None:
    runner = _runner_with_guardrails(mode="warn")
    warnings = runner._evaluate_guardrails(
        phase="pre",
        train_cfg={
            "global_mlp": {
                "grad_scale": {"mode": "off"},
                "layer_lr_multiplier": {"output": 1.0},
                "output_head_refresh": {"enabled": False},
            }
        },
        model_names=["global_mlp"],
        effective_steps_per_model=None,
    )
    assert any("global_disabled_boost" in msg for msg in warnings)


def test_guardrail_detects_effective_steps_floor_violations() -> None:
    runner = _runner_with_guardrails(mode="warn")
    warnings = runner._evaluate_guardrails(
        phase="post",
        train_cfg={},
        model_names=["global_mlp", "fno"],
        effective_steps_per_model={
            "global_mlp": {"interp": 40, "extrap": 60},
            "fno": 55,
        },
    )
    assert any("global_mlp.interp=40 (<60)" in msg for msg in warnings)
    assert any("fno=55 (<60)" in msg for msg in warnings)


def test_guardrail_error_mode_raises() -> None:
    runner = _runner_with_guardrails(mode="error")
    with pytest.raises(ValueError, match="benchmark.guardrails violations"):
        runner._evaluate_guardrails(
            phase="pre",
            train_cfg={
                "global_mlp": {
                    "grad_scale": {"mode": "off"},
                    "layer_lr_multiplier": {"output": 1.0},
                    "output_head_refresh": {"enabled": False},
                }
            },
            model_names=["global_mlp"],
            effective_steps_per_model=None,
        )


def test_primary_metric_value_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric='missing'.*not present"):
        _primary_metric_value(
            {"model_id": "fno", "score_total": 1.0},
            primary_metric="missing",
            model_name="fno",
        )


def test_primary_metric_value_rejects_non_numeric_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric='score_total'.*must be numeric"):
        _primary_metric_value(
            {"model_id": "fno", "score_total": ""},
            primary_metric="score_total",
            model_name="fno",
        )


def test_aggregate_unet_contract_effective_uses_explicit_model_key() -> None:
    runner = BenchmarkRunner({"runtime": runtime_table_only(), "benchmark": {}})
    out = runner._aggregate_unet_contract_effective(
        train_cfg={
            "unetpp": {
                "target_family": "allvars",
                "input_features": {"mode": "geom_feature_pack", "features": ["x", "y"]},
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {"conv_cfg": {"upsample_mode": "deconv"}, "output_heads": {"mode": "shared"}},
            },
            "unetpp_attn": {
                "target_family": "allvars",
                "input_features": {
                    "mode": "geom_feature_pack",
                    "features": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                },
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {
                    "conv_cfg": {"upsample_mode": "bilinear", "attention_cfg": {"enabled": True}},
                    "output_heads": {"mode": "shared"},
                },
            },
        },
        y_vars=["ne", "ni", "Te", "phi"],
        model_key="unetpp_attn",
        unet_contract_samples=[],
    )
    assert out["unet_feature_contract_effective"]["upsample_mode"] == "bilinear"
    assert out["unet_feature_contract_effective"]["input_feature_channels"] == [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
    ]
    assert out["selection_weights_effective"] == {}
    assert out["unet_optimizer_effective"]["weight_decay"] == pytest.approx(0.0)


def test_aggregate_unet_contract_effective_defaults_geom_mode_and_keeps_weights() -> None:
    runner = BenchmarkRunner({"runtime": runtime_table_only(), "benchmark": {}})
    out = runner._aggregate_unet_contract_effective(
        train_cfg={
            "unetpp": {
                "target_family": "allvars",
                "selection": {"mode": "best_val_allvars_balance", "weights": {"ne": 0.4, "ni": 0.6}},
                "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 5e-4},
                "model_cfg": {"output_heads": {"mode": "shared"}},
            }
        },
        y_vars=["ne", "ni", "Te", "phi"],
        model_key="unetpp",
        unet_contract_samples=[],
    )
    assert out["unet_feature_contract_effective"]["input_features_mode"] == "geom_feature_pack"
    assert out["selection_weights_effective"] == {"ne": 0.4, "ni": 0.6}
    assert out["unet_optimizer_effective"]["weight_decay"] == pytest.approx(5e-4)


def test_aggregate_deeponet_contract_effective_keeps_missing_policy_and_dot_skip_mode() -> None:
    runner = BenchmarkRunner({"runtime": runtime_table_only(), "benchmark": {}})
    out = runner._aggregate_deeponet_contract_effective(
        train_cfg={
            "deeponet_plasma": {
                "selection": {"mode": "best_val_allvars_balance"},
                "model_cfg": {
                    "output_path": {"mode": "fused", "dot_skip_mode": "learned_per_var"},
                    "missing_geom_feature_policy": "error",
                },
            }
        },
        y_vars=["ne", "ni", "Te", "phi"],
        deeponet_contract_samples=[],
    )
    assert out["output_path_dot_skip_mode_effective"] == "learned_per_var"
    assert out["missing_geom_feature_policy_effective"] == "error"


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


def test_validate_eval_scope_train_sections_allows_cno_operator_unet_scope() -> None:
    BenchmarkRunner._validate_eval_scope_models(
        scope="cno_operator_unet_isolated",
        model_names=["cno_operator_unet"],
    )
    BenchmarkRunner._validate_eval_scope_train_sections(
        scope="cno_operator_unet_isolated",
        train_cfg={"cno_operator_unet": {"epochs": 1}},
    )


def test_validate_eval_scope_train_sections_allows_unet_operator_v2_scope() -> None:
    BenchmarkRunner._validate_eval_scope_models(
        scope="unet_operator_v2_isolated",
        model_names=["unet_operator_v2"],
    )
    BenchmarkRunner._validate_eval_scope_train_sections(
        scope="unet_operator_v2_isolated",
        train_cfg={"unet_operator_v2": {"epochs": 1}},
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
        ("cno_operator_unet_isolated", {"cno": {"epochs": 1}}),
        ("unet_operator_v2_isolated", {"unetpp": {"epochs": 1}}),
        ("global_frozen", {"deeponet_plasma": {"epochs": 1}}),
    ],
)
def test_validate_eval_scope_train_sections_rejects_mixed(scope: str, train_cfg: dict) -> None:
    with pytest.raises(ValueError, match="forbids non-empty sections"):
        BenchmarkRunner._validate_eval_scope_train_sections(scope=scope, train_cfg=train_cfg)
