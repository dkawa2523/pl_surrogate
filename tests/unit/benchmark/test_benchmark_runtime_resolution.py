from __future__ import annotations

from pathlib import Path

import pytest

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.benchmark.runtime_context import resolve_effective_benchmark_cfg
from tests._config_presets import runtime_table_only, runtime_table_plus_structure


def test_benchmark_runtime_prefers_benchmark_section(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        {
            "benchmark": {
                "output_dir": str(tmp_path / "bench"),
                "runtime": runtime_table_plus_structure(),
            },
        }
    )
    assert runner.benchmark_cfg["runtime"]["input_mode"] == "table_plus_structure"
    assert runner.input_mode_meta["input_mode_effective"] == "table_plus_structure"
    assert runner.input_mode_meta["structure_feature_profile_effective"] == "geom_v1_mainline"


def test_benchmark_runtime_falls_back_to_top_level(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        {
            "runtime": runtime_table_only(),
            "benchmark": {
                "output_dir": str(tmp_path / "bench"),
            },
        }
    )
    assert runner.benchmark_cfg["runtime"]["input_mode"] == "table_only"
    assert runner.input_mode_meta["input_mode_effective"] == "table_only"
    assert runner.input_mode_meta["structure_feature_profile_effective"] == "none"


def test_resolve_effective_benchmark_cfg_prefers_benchmark_runtime() -> None:
    cfg = resolve_effective_benchmark_cfg(
        {
            "benchmark": {
                "runtime": runtime_table_plus_structure(),
            },
        }
    )
    assert cfg["runtime"]["input_mode"] == "table_plus_structure"
    assert cfg["runtime"]["structure"]["feature_profile"] == "geom_v1_mainline"


def test_benchmark_runtime_missing_everywhere_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="runtime.input_mode=table_plus_structure requires runtime.structure.feature_profile",
    ):
        BenchmarkRunner({"benchmark": {"output_dir": str(tmp_path / "bench")}})


def test_resolve_effective_benchmark_cfg_conflict_strict_error_fails() -> None:
    with pytest.raises(ValueError, match="benchmark.runtime conflicts with top-level runtime"):
        resolve_effective_benchmark_cfg(
            {
                "runtime": runtime_table_only(),
                "benchmark": {"runtime": runtime_table_plus_structure()},
            }
        )


def test_resolve_effective_benchmark_cfg_rejects_non_error_strict_mode() -> None:
    with pytest.raises(ValueError, match="requires runtime.strict_input_mode='error'"):
        resolve_effective_benchmark_cfg(
            {
                "benchmark": {
                    "runtime": runtime_table_plus_structure(strict_input_mode="warn"),
                },
            }
        )


def test_resolve_effective_benchmark_cfg_rejects_allow_mode_fallback_true() -> None:
    with pytest.raises(ValueError, match="forbids runtime.allow_mode_fallback=true"):
        resolve_effective_benchmark_cfg(
            {
                "benchmark": {
                    "runtime": runtime_table_plus_structure(allow_mode_fallback=True),
                },
            }
        )
