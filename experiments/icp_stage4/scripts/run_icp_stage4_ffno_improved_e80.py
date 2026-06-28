from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE_CONFIG = Path(
    "configs/experimental/icp_stage4/generated_multifield_8field_part_sdf_lite_v1_e80/"
    "full/benchmark_icp_stage4_core4_full_ffno.yaml"
)
PLASMA_TARGETS = ["ne", "ni", "Te", "phi"]


def _set_path(root: dict[str, Any], keys: list[str], value: Any) -> None:
    node: dict[str, Any] = root
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _filter_keys(raw: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {name: raw[name] for name in keys if name in raw}


def build_config(base_path: Path, output_dir: Path, *, n_modes: int, epochs: int) -> dict[str, Any]:
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = dict(cfg.get("benchmark", {}) or {})
    train = dict(bench.get("train", {}) or {})
    ffno = dict(train.get("ffno", {}) or {})
    loss = dict(train.get("loss", {}) or {})
    supervised = dict(loss.get("supervised", {}) or {})
    multitask = dict(loss.get("multitask", {}) or {})
    eval_cfg = dict(bench.get("eval", {}) or {})

    bench["output_dir"] = output_dir.as_posix()
    bench["profile"] = "m7_ffno_isolated"
    eval_cfg["protocol_variant"] = f"icp_stage4_plasma_ffno_modes{int(n_modes)}_e{int(epochs)}"
    eval_cfg["target_family_for_score"] = "auto"
    eval_cfg["target_vars_for_score"] = list(PLASMA_TARGETS)
    audit_cfg = dict(eval_cfg.get("spatial_distribution_audit", {}) or {})
    audit_cfg["vars"] = list(PLASMA_TARGETS)
    eval_cfg["spatial_distribution_audit"] = audit_cfg
    plots_cfg = dict(eval_cfg.get("plots", {}) or {})
    plots_cfg["loss_yscale"] = "log"
    eval_cfg["plots"] = plots_cfg
    bench["eval"] = eval_cfg

    ffno["epochs"] = int(epochs)
    ffno["target_vars"] = list(PLASMA_TARGETS)
    ffno["selection"] = {
        "mode": "best_val_allvars_balance",
        "eval_every_n_epochs": 1,
        "warmup_epochs": 0,
        "weights": {name: 1.0 / float(len(PLASMA_TARGETS)) for name in PLASMA_TARGETS},
    }
    model_cfg = dict(ffno.get("model_cfg", {}) or {})
    model_cfg["n_modes"] = int(n_modes)
    model_cfg["fno_n_modes"] = int(n_modes)
    spectral_cfg = dict(model_cfg.get("spectral_cfg", {}) or {})
    spectral_cfg.setdefault("width", 64)
    spectral_cfg.setdefault("n_layers", 4)
    spectral_cfg.setdefault("dropout", 0.0)
    model_cfg["spectral_cfg"] = spectral_cfg
    ffno["model_cfg"] = model_cfg
    train["ffno"] = ffno

    supervised["spatial_consistency"] = {
        "enabled": True,
        "vars": list(PLASMA_TARGETS),
        "mode": "grad_huber",
        "lambda": 0.005,
        "delta": 1.0,
        "apply_region": "target_region",
        "normalize_by_var_scale": True,
        "multiscale": {"enabled": False, "scales": [1, 2, 4], "scale_weights": [1.0, 0.7, 0.5]},
    }
    supervised["target_weights"] = _filter_keys(dict(supervised.get("target_weights", {}) or {}), PLASMA_TARGETS)
    supervised["target_region_by_var"] = {name: "plasma_only" for name in PLASMA_TARGETS}
    supervised["boundary_weight"] = {
        "enabled": True,
        "alpha": 2.0,
        "tau": 2.0,
        "vars": list(PLASMA_TARGETS),
    }
    loss["supervised"] = supervised
    multitask["fixed_weights_by_var"] = _filter_keys(
        dict(multitask.get("fixed_weights_by_var", {}) or {}),
        PLASMA_TARGETS,
    )
    loss["multitask"] = multitask
    train["loss"] = loss
    bench["train"] = train
    cfg["benchmark"] = bench
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICP_stage4 FFNO plasma-field improved external experiment.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--n-modes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--tag", default="")
    parser.add_argument("--stack-dump-seconds", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    if int(args.stack_dump_seconds) > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(int(args.stack_dump_seconds), repeat=True)
    stamp = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(
        f"runs/icp_stage4_multifield_plasma_ffno_modes{int(args.n_modes)}_e80/full/"
        f"ffno_plasma_modes{int(args.n_modes)}_grad005_gpu80_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args.base_config, run_dir, n_modes=int(args.n_modes), epochs=int(args.epochs))
    cfg_path = run_dir / "run_config_effective.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    from plasma_surrogate.benchmark.runner import BenchmarkRunner

    started = time.perf_counter()
    BenchmarkRunner.from_yaml(cfg_path).run()
    elapsed = time.perf_counter() - started
    with (run_dir / "run_walltime.json").open("w", encoding="utf-8") as f:
        json.dump({"seconds": elapsed, "config": str(cfg_path)}, f, indent=2)
    print(run_dir.as_posix())


if __name__ == "__main__":
    main()
