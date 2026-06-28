from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "run_benchmarkrun_ext0520.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_benchmarkrun_ext0520", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_ext0520_defaults_cover_trustworthy_full_matrix(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = module._parse_args()

    assert args.sizes == [27, 54, 78]
    assert args.config_root == "runs/gec_ccp_trustworthy_v1/configs"
    assert args.run_root == "runs/gec_ccp_trustworthy_v1"
    assert args.status_csv == "runs/gec_ccp_trustworthy_v1/run_status.csv"
    assert "unet_operator_v2" in args.models
    assert "deeponet_pod" in args.models
    assert "deeponet_plasma_pod" not in args.models
    assert len(args.models) == 17
