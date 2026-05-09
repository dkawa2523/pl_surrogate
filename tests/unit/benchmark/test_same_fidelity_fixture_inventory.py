from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.config_catalog


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _target_ids(bench_cfg: dict) -> list[str]:
    dataset_cfg = dict(bench_cfg.get("dataset", {}))
    targets = list(dataset_cfg.get("targets", []))
    return [str(dict(t).get("id", "")).strip() for t in targets if str(dict(t).get("id", "")).strip()]


def test_same_fidelity_benchmark_fixture_inventory_is_consistent() -> None:
    expected = {
        "benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml": ("m7_unetpp_isolated", "unetpp"),
        "benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml": ("m7_unetpp_attn_isolated", "unetpp_attn"),
        "benchmark_periodic_real_m7_ffno_isolated_mainline.yaml": ("m7_ffno_isolated", "ffno"),
        "benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml": ("m7_coord_mlp_fourier_experimental", "coord_mlp_fourier"),
        "benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml": ("m7_coord_mlp_siren_experimental", "coord_mlp_siren"),
        "benchmark_periodic_real_m7_deeponet_pod_experimental.yaml": ("m7_deeponet_pod_experimental", "deeponet_pod"),
    }
    fixture_dir = ROOT / "tests" / "fixtures"
    for filename, (profile, model_name) in expected.items():
        payload = _load_yaml(fixture_dir / filename)
        bench = dict(payload.get("benchmark", {}))
        assert bench["profile"] == profile
        train_cfg = dict(bench.get("train", {}))
        assert model_name in train_cfg
        target_ids = _target_ids(bench)
        assert target_ids, f"{filename}: benchmark.dataset.targets[].id must be defined"
        scalers_cfg = dict(dict(bench.get("preprocessing", {})).get("scalers", {}))
        target_transforms = dict(scalers_cfg.get("target_transforms", {}))
        assert target_transforms, f"{filename}: preprocessing.scalers.target_transforms must be defined"
        assert set(target_transforms.keys()) == set(target_ids), (
            f"{filename}: target_transforms keys must match dataset.targets[].id "
            f"(got={sorted(target_transforms.keys())}, expected={sorted(target_ids)})"
        )


def test_deeponet_pod_fixture_has_required_runtime_blocks() -> None:
    payload = _load_yaml(ROOT / "tests" / "fixtures" / "benchmark_periodic_real_m7_deeponet_pod_experimental.yaml")
    bench = dict(payload.get("benchmark", {}))
    assert isinstance(bench.get("dataset"), dict)
    assert isinstance(bench.get("split"), dict)
    assert isinstance(bench.get("preprocessing"), dict)
    assert isinstance(bench.get("train"), dict)
    assert "deeponet_pod" in dict(bench.get("train", {}))


def test_same_fidelity_config_template_inventory_is_consistent() -> None:
    expected = {
        "unetpp_template.yaml": "unetpp",
        "unetpp_attn_template.yaml": "unetpp_attn",
        "ffno_template.yaml": "ffno",
        "coord_mlp_fourier_template.yaml": "coord_mlp_fourier",
        "coord_mlp_siren_template.yaml": "coord_mlp_siren",
        "deeponet_pod_template.yaml": "deeponet_pod",
    }
    cfg_dir = ROOT / "configs" / "experimental" / "same_fidelity_low_data"
    for filename, model_name in expected.items():
        payload = _load_yaml(cfg_dir / filename)
        assert dict(payload.get("model", {})).get("name") == model_name
        assert model_name in dict(payload.get("train", {}))
