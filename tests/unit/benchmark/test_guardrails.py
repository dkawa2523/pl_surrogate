from __future__ import annotations

import pytest

from plasma_surrogate.benchmark.runner import BenchmarkRunner


def _runner_with_guardrails(*, mode: str = "warn") -> BenchmarkRunner:
    return BenchmarkRunner(
        {
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


def test_aggregate_unet_contract_effective_uses_explicit_model_key() -> None:
    runner = BenchmarkRunner({"benchmark": {}})
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
