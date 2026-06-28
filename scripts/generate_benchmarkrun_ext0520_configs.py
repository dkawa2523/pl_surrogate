from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


MODEL_ORDER = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "unet_operator_v2",
    "fno",
    "ffno",
    "coord_mlp_fourier",
    "coord_mlp_siren",
    "coord_mlp_pod_residual",
    "u_no",
    "cno",
    "cno_operator_unet",
    "geom_deeponet_pod",
    "geom_deeponet_siren",
    "deeponet_plasma",
)

TARGETS = (
    {
        "id": "ne",
        "source_key": "ne",
        "value_transform": "identity",
        "units": "m^-3",
        "dtype": "float32",
        "role": "density_electron",
        "positive": True,
        "field_family": "density",
        "default_region": "plasma_only",
    },
    {
        "id": "ni",
        "source_key": "ni",
        "value_transform": "identity",
        "units": "m^-3",
        "dtype": "float32",
        "role": "density_ion",
        "positive": True,
        "field_family": "density",
        "default_region": "plasma_only",
    },
    {
        "id": "Te",
        "source_key": "Te",
        "value_transform": "identity",
        "units": "eV",
        "dtype": "float32",
        "role": "temperature_electron",
        "positive": True,
        "field_family": "temperature",
        "default_region": "plasma_only",
    },
    {
        "id": "phi",
        "source_key": "phi",
        "value_transform": "identity",
        "units": "V",
        "dtype": "float32",
        "role": "potential",
        "positive": False,
        "field_family": "electrostatic",
        "default_region": "plasma_only",
    },
)

TARGET_VARS = [str(item["id"]) for item in TARGETS]
INFERENCE_PHYSICS_SYMBOLS = {"density": "ne", "temperature": "Te", "potential": "phi"}
INFERENCE_QOI_UNIFORMITY = {"target": "ne"}
COMMON_SCOPE_MODELS = {"coord_mlp_pod_residual", "unet_operator_v2", "cno_operator_unet"}
COMMON_SELECTION_WEIGHTS = {name: 0.25 for name in TARGET_VARS}


def _standard_loss_cfg() -> dict[str, Any]:
    return {
        "supervised": {
            "type": "mse",
            "mask": "plasma_only",
        }
    }


def _identity_zscore_target_transforms() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "value_transform": "identity",
            "scaler": "zscore",
            "fit_scope": "plasma_only",
            "clip": {"mode": "none"},
        }
        for name in TARGET_VARS
    }


def _config_path(*parts: str | Path) -> str:
    return Path(*[str(part) for part in parts]).as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ext0520 benchmark configs from curated templates.")
    parser.add_argument("--template-dir", default="configs/benchmarkrun_ext0520/templates")
    parser.add_argument("--run-root", default="runs/gec_ccp_trustworthy_v1")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--dataset-root", default="data/outputs_merged_td_csv_periodic_ext0520")
    parser.add_argument("--sizes", nargs="+", type=int, default=[27, 54, 78])
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--protocol-prefix", default="gec_ccp_trustworthy_v1")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "benchmark" not in payload:
        raise ValueError(f"{path} must contain a benchmark section")
    return payload


def _normalize_gec_ccp_trustworthy_cfg(
    cfg: dict[str, Any],
    *,
    dataset_root: str,
    run_root: str,
    model: str,
    size: int,
    protocol_prefix: str,
) -> dict[str, Any]:
    benchmark = cfg["benchmark"]
    benchmark["output_dir"] = _config_path(run_root, f"n{size}", model)

    eval_cfg = benchmark.setdefault("eval", {})
    eval_cfg["primary_metric"] = "surrogate_quality_score"
    eval_cfg["objective_mode"] = "min"
    eval_cfg.pop("primary_mode", None)
    eval_cfg["target_family_for_score"] = "allvars"
    eval_cfg["target_vars_for_score"] = list(TARGET_VARS)
    eval_cfg["quality_score"] = {
        "mode": "spatial_huber",
        "delta": 1.0,
        "boundary_alpha": 2.0,
        "boundary_band_px": 2.0,
        "avgpool_lambda": 0.05,
        "pool_scale": 3,
    }
    eval_cfg["protocol_variant"] = f"{model}_{protocol_prefix}_n{size}"
    eval_cfg["region_bands"] = {
        "mode": "fixed_px",
        "boundary_in_px": 2.0,
        "deep_plasma_px": 10.0,
    }

    eval_protocol = benchmark.setdefault("eval_protocol", {})
    eval_protocol["mode"] = "dual_axis"
    eval_protocol["interp_weight"] = 0.5
    eval_protocol["extrap_weight"] = 0.5
    eval_protocol["interp_mode"] = "overlap"
    if model in COMMON_SCOPE_MODELS:
        eval_protocol["scope"] = "common"

    dataset = benchmark.setdefault("dataset", {})
    dataset["root"] = str(dataset_root)
    dataset["index_csv"] = f"index_{size}.csv"
    dataset["targets"] = [dict(item) for item in TARGETS]

    preprocessing = benchmark.setdefault("preprocessing", {})
    scalers = preprocessing.setdefault("scalers", {})
    scalers["enforce_target_transforms"] = True
    scalers["y_fit_policy"] = "plasma_only"
    scalers["target_transforms"] = _identity_zscore_target_transforms()
    coord_grid_contract = preprocessing.setdefault("coord_grid_contract", {})
    coord_grid_contract["require_requested_source"] = "off"

    train = benchmark.setdefault("train", {})
    train["loss"] = _standard_loss_cfg()
    for section_name, section in list(train.items()):
        if not isinstance(section, dict) or section_name == "loss":
            continue
        if section_name not in set(MODEL_ORDER) | {"global_mlp", "deeponet_plasma"}:
            continue
        selection = section.setdefault("selection", {})
        selection["mode"] = "best_val_allvars_balance"
        selection["weights"] = dict(COMMON_SELECTION_WEIGHTS)

    physics_train = benchmark.setdefault("physics", {})
    physics_train["enabled"] = False

    inference = benchmark.setdefault("inference", {})
    ood = inference.setdefault("ood", {})
    physics = ood.setdefault("physics", {})
    physics["enabled"] = True
    physics["symbols"] = dict(INFERENCE_PHYSICS_SYMBOLS)
    qoi = ood.setdefault("qoi", {})
    qoi["uniformity"] = dict(INFERENCE_QOI_UNIFORMITY)
    return cfg


def main() -> int:
    args = _parse_args()
    template_dir = Path(args.template_dir)
    run_root = Path(args.run_root)
    out_root = Path(args.out_root) if args.out_root is not None else run_root / "configs"
    for size in [int(v) for v in args.sizes]:
        for model in [str(v) for v in args.models]:
            src = template_dir / f"benchmark_ext0520_{model}.yaml"
            if not src.exists():
                raise FileNotFoundError(f"missing benchmark template: {src}")
            cfg = _load_yaml(src)
            cfg = _normalize_gec_ccp_trustworthy_cfg(
                cfg,
                dataset_root=str(args.dataset_root),
                run_root=str(run_root),
                model=model,
                size=size,
                protocol_prefix=str(args.protocol_prefix),
            )
            out = out_root / f"n{size}" / f"benchmark_ext0520_{model}_n{size}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
