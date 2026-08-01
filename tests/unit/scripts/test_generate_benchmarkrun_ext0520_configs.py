from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "generate_benchmarkrun_ext0520_configs.py"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_generate_ext0520_trustworthy_configs_with_metadata_and_new_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "gec_ccp_trustworthy_v1"
    out_root = run_root / "configs"
    dataset_root = tmp_path / "data" / "outputs_merged_td_csv_periodic_ext0520"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sizes",
            "27",
            "--models",
            "global_mlp",
            "unet_operator_v2",
            "--run-root",
            str(run_root),
            "--out-root",
            str(out_root),
            "--dataset-root",
            str(dataset_root),
        ],
        cwd=str(ROOT),
        check=True,
    )

    for model in ("global_mlp", "unet_operator_v2"):
        cfg_path = out_root / "n27" / f"benchmark_ext0520_{model}_n27.yaml"
        assert cfg_path.exists()
        cfg = _load_yaml(cfg_path)
        benchmark = cfg["benchmark"]
        assert benchmark["output_dir"] == (run_root / "n27" / model).as_posix()

        eval_cfg = benchmark["eval"]
        assert eval_cfg["primary_metric"] == "surrogate_quality_score"
        assert eval_cfg["objective_mode"] == "min"
        assert "primary_mode" not in eval_cfg
        assert eval_cfg["target_vars_for_score"] == ["ne", "ni", "Te", "phi"]
        assert eval_cfg["region_bands"] == {
            "mode": "fixed_px",
            "boundary_in_px": 2.0,
            "deep_plasma_px": 10.0,
        }

        eval_protocol = benchmark["eval_protocol"]
        assert eval_protocol["mode"] == "dual_axis"
        if model == "unet_operator_v2":
            assert eval_protocol["scope"] == "common"
        assert eval_protocol["interp_weight"] == 0.5
        assert eval_protocol["extrap_weight"] == 0.5
        assert eval_protocol["interp_mode"] == "overlap"

        dataset = benchmark["dataset"]
        assert dataset["root"] == str(dataset_root)
        assert dataset["index_csv"] == "index_27.csv"
        targets = {target["id"]: target for target in dataset["targets"]}
        assert targets["ne"] == {
            "id": "ne",
            "source_key": "ne",
            "value_transform": "identity",
            "units": "m^-3",
            "dtype": "float32",
            "role": "density_electron",
            "positive": True,
            "field_family": "density",
            "default_region": "plasma_only",
        }
        assert targets["ni"]["source_key"] == "ni"
        assert targets["ni"]["value_transform"] == "identity"
        assert targets["ni"]["role"] == "density_ion"
        assert targets["Te"]["role"] == "temperature_electron"
        assert targets["Te"]["positive"] is True
        assert targets["phi"]["role"] == "potential"
        assert targets["phi"]["positive"] is False

        transforms = benchmark["preprocessing"]["scalers"]["target_transforms"]
        for target_id in ("ne", "ni", "Te", "phi"):
            assert transforms[target_id]["value_transform"] == "identity"
            assert transforms[target_id]["scaler"] == "zscore"
            assert transforms[target_id]["fit_scope"] == "plasma_only"
        assert benchmark["preprocessing"]["coord_grid_contract"]["require_requested_source"] == "off"
        assert benchmark["train"]["loss"]["supervised"] == {"type": "mse", "mask": "plasma_only"}
        assert benchmark["train"][model]["selection"]["mode"] == "best_val_allvars_balance"
        assert benchmark["train"][model]["selection"]["weights"] == {
            "ne": 0.25,
            "ni": 0.25,
            "Te": 0.25,
            "phi": 0.25,
        }

        assert benchmark["physics"]["enabled"] is False
        assert benchmark["inference"]["ood"]["physics"] == {
            "enabled": True,
            "symbols": {"density": "ne", "temperature": "Te", "potential": "phi"},
        }
        assert benchmark["inference"]["ood"]["qoi"]["uniformity"] == {"target": "ne"}
