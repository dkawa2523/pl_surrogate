from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np
import yaml

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from tests._runtime_requirements import require_torch_runtime


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.benchmark_slow


def _load_benchmark_payload(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "benchmark" not in payload:
        raise ValueError(f"{path} must define benchmark root config")
    return payload


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_parametric_parts_artifacts(output_dir: Path) -> None:
    geom_dir = output_dir / "dataset" / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    h, w = 8, 8
    part_stack = np.zeros((2, h, w), dtype=np.float32)
    part_stack[0, 2:4, 2:4] = 1.0
    part_stack[1, 4:6, 5:7] = 1.0
    np.savez_compressed(
        geom_dir / "parts_pack.npz",
        mask_stack=part_stack,
        part_ids=np.asarray(["p0", "p1"], dtype=object),
    )
    manifest = {
        "part_ids": ["p0", "p1"],
        "param_specs": {
            "part.p0.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
            "part.p1.tx": {"default": 0.0, "min": -0.2, "max": 0.2},
        },
    }
    (geom_dir / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_benchmark_dual_mode_table_only_smoke_emits_input_mode_columns(tmp_path: Path) -> None:
    payload = _load_benchmark_payload(
        ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_only_global_mlp_smoke.yaml"
    )
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_only")
    cfg_path = tmp_path / "bench_table_only.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    assert rows, "leaderboard.csv must contain at least one model row"
    row = rows[0]
    assert row["input_mode_effective"] == "table_only"
    assert row["structure_feature_profile_effective"] == "none"
    assert row["structure_descriptor_profile_effective"] == "none"
    assert row["structure_latent_profile_effective"] == "none"
    assert row["structure_adapter_mode_effective"] == "none"
    assert row["geometry_provider_mode_effective"] == "fixed"


def test_benchmark_dual_mode_table_plus_structure_smoke_emits_input_mode_columns(tmp_path: Path) -> None:
    require_torch_runtime()

    payload = _load_benchmark_payload(
        ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_plus_structure_deeponet_pod_smoke.yaml"
    )
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_plus_structure")
    cfg_path = tmp_path / "bench_table_plus_structure.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    assert rows, "leaderboard.csv must contain at least one model row"
    row = rows[0]
    assert row["input_mode_effective"] == "table_plus_structure"
    assert row["structure_feature_profile_effective"] == "geom_v1_mainline"
    assert row["structure_descriptor_profile_effective"] == "none"
    assert row["structure_latent_profile_effective"] == "none"
    assert row["structure_adapter_mode_effective"] == "none"
    assert row["geometry_provider_mode_effective"] == "fixed"
    ckpt_meta_path = Path(payload["benchmark"]["output_dir"]) / "models" / "deeponet_pod" / "checkpoints" / "meta.json"
    ckpt_meta = json.loads(ckpt_meta_path.read_text(encoding="utf-8"))
    assert ckpt_meta["deeponet_pod_descriptor_profile_effective"] == "none"
    assert int(ckpt_meta["deeponet_pod_descriptor_dim_effective"]) == 0
    assert ckpt_meta["deeponet_pod_latent_profile_effective"] == "none"
    assert ckpt_meta["deeponet_pod_latent_hook_effective"] is False


def test_benchmark_dual_mode_table_plus_structure_uno_smoke_emits_input_mode_columns(tmp_path: Path) -> None:
    require_torch_runtime()

    payload = _load_benchmark_payload(
        ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_plus_structure_u_no_smoke.yaml"
    )
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_plus_structure_uno")
    cfg_path = tmp_path / "bench_table_plus_structure_uno.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    assert rows, "leaderboard.csv must contain at least one model row"
    row = rows[0]
    assert row["model_id"] == "u_no"
    assert row["input_mode_effective"] == "table_plus_structure"
    assert row["structure_feature_profile_effective"] == "geom_v1_mainline"
    assert row["structure_descriptor_profile_effective"] == "none"
    assert row["structure_latent_profile_effective"] == "none"
    assert row["structure_adapter_mode_effective"] == "grid_pack"
    assert row["geometry_provider_mode_effective"] == "fixed"


def test_benchmark_dual_mode_table_plus_structure_cno_smoke_emits_input_mode_columns(tmp_path: Path) -> None:
    require_torch_runtime()

    payload = _load_benchmark_payload(
        ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_plus_structure_cno_smoke.yaml"
    )
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_plus_structure_cno")
    cfg_path = tmp_path / "bench_table_plus_structure_cno.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    assert rows, "leaderboard.csv must contain at least one model row"
    row = rows[0]
    assert row["model_id"] == "cno"
    assert row["input_mode_effective"] == "table_plus_structure"
    assert row["structure_feature_profile_effective"] == "geom_v1_mainline"
    assert row["structure_descriptor_profile_effective"] == "none"
    assert row["structure_latent_profile_effective"] == "none"
    assert row["structure_adapter_mode_effective"] == "grid_pack"
    assert row["geometry_provider_mode_effective"] == "fixed"


def test_benchmark_dual_mode_table_plus_structure_geom_deeponet_siren_smoke_emits_input_mode_columns(
    tmp_path: Path,
) -> None:
    require_torch_runtime()

    payload = _load_benchmark_payload(
        ROOT / "tests" / "fixtures" / "benchmark_dual_mode_table_plus_structure_geom_deeponet_siren_smoke.yaml"
    )
    payload["benchmark"]["output_dir"] = str(tmp_path / "bench_table_plus_structure_geom_deeponet_siren")
    _write_parametric_parts_artifacts(Path(payload["benchmark"]["output_dir"]))
    cfg_path = tmp_path / "bench_table_plus_structure_geom_deeponet_siren.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = BenchmarkRunner.from_yaml(cfg_path).run()
    rows = _read_rows(result.leaderboard_path)
    assert rows, "leaderboard.csv must contain at least one model row"
    row = rows[0]
    assert row["model_id"] == "geom_deeponet_siren"
    assert row["input_mode_effective"] == "table_plus_structure"
    assert row["structure_feature_profile_effective"] == "geom_v1_mainline"
    assert row["structure_descriptor_profile_effective"] == "struct_desc_v1"
    assert row["structure_latent_profile_effective"] == "none"
    assert row["structure_adapter_mode_effective"] == "hybrid_pack_descriptor"
    assert row["geometry_provider_mode_effective"] == "parametric_parts"
