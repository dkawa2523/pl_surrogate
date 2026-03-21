from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
