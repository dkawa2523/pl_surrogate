from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from tests._runtime_requirements import require_torch_runtime


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.benchmark_slow


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_benchmark_table_plus_structure_cno_runtime_metadata(tmp_path: Path) -> None:
    require_torch_runtime()
    fixture = ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_plus_structure_cno_smoke.yaml"
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_plus_structure_cno")
    cfg_path = tmp_path / "bench_table_plus_structure_cno.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)

    assert rows
    row = rows[0]
    assert row["model_id"] == "cno"
    assert row["input_mode_effective"] == "table_plus_structure"
    assert row["structure_feature_profile_effective"] == "geom_v1_mainline"
    assert row["structure_adapter_mode_effective"] == "grid_pack"
    assert row["geometry_provider_mode_effective"] == "fixed"
    assert row["target_schema_hash"]
    assert row["feature_schema_hash"]
    assert row["surrogate_quality_score"]
