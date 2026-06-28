from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from tests._runtime_requirements import require_torch_runtime


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "experimental" / "head_mode_comparison"

pytestmark = [pytest.mark.benchmark_slow, pytest.mark.torch_runtime]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("benchmark_fno_shared_n8.yaml", "shared"),
        ("benchmark_fno_role_grouped_n8.yaml", "role_grouped"),
        ("benchmark_fno_custom_groups_n8.yaml", "custom_groups"),
    ],
)
def test_head_mode_benchmark_configs_emit_default_metric_columns(tmp_path: Path, name: str, mode: str) -> None:
    require_torch_runtime()
    payload = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) or {}
    payload["benchmark"]["output_dir"] = str(tmp_path / mode)
    cfg_path = tmp_path / name
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    core_rows = _read_rows(Path(payload["benchmark"]["output_dir"]) / "core_metrics.csv")

    assert rows
    assert core_rows
    row = rows[0]
    assert row["model_id"] == "fno"
    assert row["surrogate_quality_score"]
    assert "test_rmse_ne" in row
    assert "test_r2_ne" in row
    assert "test_rmse_group_density" in row
    assert "test_r2_group_density" in row
    assert "test_rmse_group_density_plasma" in row
    assert "test_rmse_ne_boundary_band" not in row
    assert "test_rmse_group_density_boundary_band" not in row
