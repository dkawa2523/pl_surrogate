from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "experimental" / "head_mode_comparison"


def test_head_mode_comparison_configs_are_minimal_and_distinct() -> None:
    expected = {
        "benchmark_fno_shared_n8.yaml": "shared",
        "benchmark_fno_role_grouped_n8.yaml": "role_grouped",
        "benchmark_fno_custom_groups_n8.yaml": "custom_groups",
    }
    for filename, mode in expected.items():
        payload = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8")) or {}
        benchmark = dict(payload["benchmark"])
        fno_cfg = dict(benchmark["train"]["fno"])
        output_heads = dict(fno_cfg["model_cfg"]["output_heads"])

        assert benchmark["profile"] == "m7_fno_isolated"
        assert benchmark["dataset"]["type"] == "synthetic"
        assert benchmark["dataset"]["n_cases"] == 8
        assert benchmark["eval"]["primary_metric"] == "surrogate_quality_score"
        assert benchmark["eval_protocol"]["mode"] == "single"
        assert benchmark["eval_protocol"]["scope"] == "fno_isolated"
        assert output_heads["mode"] == mode
        if mode == "custom_groups":
            assert output_heads["strict"] is True
            assert output_heads["groups"]["density"]["targets"] == ["ne", "ni"]
