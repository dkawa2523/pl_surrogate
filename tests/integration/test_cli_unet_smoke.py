from __future__ import annotations

from pathlib import Path

import yaml

from plasma_surrogate.cli.main import main


def test_cli_unet_train_and_infer_smoke(tmp_path: Path):
    run_dir = tmp_path / "m3_unet_run"
    cfg = {
        "run_dir": str(run_dir),
        "dataset": {"type": "synthetic", "n_cases": 10, "height": 8, "width": 8, "cond_dim": 3, "seed": 2},
        "preprocessing": {"split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]}},
        "model": {"name": "unet", "phi_mode": "poisson_hybrid", "phi_hybrid_alpha": 0.15, "phi_hybrid_steps": 1},
        "train": {"epochs": 3, "lr": 0.02, "physics": {"enabled": True, "lambda_poisson": 0.02, "lambda_bc": 0.01}},
        "inference": {"single": {"enabled": True, "cond": {"c0": 0.3, "c1": 0.4, "c2": 0.5}}},
    }
    cfg_path = tmp_path / "m3_unet.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert main(["preprocess", "--config", str(cfg_path)]) == 0
    assert main(["train", "--config", str(cfg_path)]) == 0
    assert main(["infer", "--config", str(cfg_path)]) == 0

    single_dirs = list((run_dir / "inference" / "single").glob("*"))
    assert single_dirs
    assert (single_dirs[0] / "fields_model.npz").exists()
    assert (single_dirs[0] / "fields_phys.npz").exists()

