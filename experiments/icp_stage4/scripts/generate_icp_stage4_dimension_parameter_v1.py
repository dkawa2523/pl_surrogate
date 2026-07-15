"""Generate the dimension-conditioned ICP training and optimization cases.

This is the controlled counterpart of the case-varying coil-SDF study.  Coil
geometry varies only through five scalar dimensions in ``dataset.cond_columns``;
the spatial pack contains the fixed chamber coordinate/mask fields only.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "configs/experimental/icp_stage4/generated_axisymmetric_energy_weighted_v1"
TRAIN_DIR = ROOT / "configs/experimental/icp_stage4/generated_dimension_parameter_v1"
OPT_DIR = ROOT / "configs/experimental/icp_stage4/generated_dimension_parameter_bohm_opt_v1"
RUN_ROOT = "runs/icp_stage4_dimension_parameter_v1"

STUDY = "icp_stage4_dimension_parameter_v1"
OPT_STUDY = "icp_stage4_dimension_parameter_bohm_opt_v1"
COND_COLUMNS = ["llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0"]
STATIC_PROFILE = "geom_v1_mainline"
STATIC_FEATURES = ["x", "y", "mask_plasma", "distance_signed", "distance_any"]

# Dataset central 90% (empirical q05--q95).  nncoil is intentionally fixed:
# the generic continuous optimizer must not generate a fractional coil count.
OPT_SPACE = {
    "llcoil": [0.5457, 1.4440],
    "rrc": [2.3979, 9.5068],
    "nncoil": [4.0, 4.0],
    "rrce": [20.3595, 29.3805],
    "zzc": [0.2423, 4.6792],
    "pp": [586.464, 2902.557],
    "pp0": [0.003807, 0.086737],
}

REFERENCE_COND = {
    "llcoil": 0.9982,
    "rrc": 5.9389,
    "nncoil": 4.0,
    "rrce": 24.9477,
    "zzc": 2.4968,
    "pp": 1747.712,
    "pp0": 0.017126,
}


def _load(name: str) -> dict[str, Any]:
    with (SOURCE / f"benchmark_icp_stage4_axisymmetric_energy_weighted_v1_{name}.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def _dimension_inputs(cfg: dict[str, Any], *, model: str) -> None:
    bench = cfg["benchmark"]
    dataset = bench["dataset"]
    dataset["cond_columns"] = list(COND_COLUMNS)
    dataset.pop("structure_npz_column", None)

    coord = bench["preprocessing"]["coord_features"]
    coord.clear()
    coord.update(
        {
            "enabled": True,
            "output": "features/coord_feature_pack.npz",
            "channels_from_profile": STATIC_PROFILE,
            "distance_transform_stats": {"enabled": True},
            "scaling": {
                "enabled": True,
                "mode": "zscore",
                "fit_scope": "train_split",
                "mask_scope": "plasma_plus_band",
            },
        }
    )

    family = bench["train"][model]
    family["input_features"] = {
        "mode": "geom_feature_pack",
        "features": list(STATIC_FEATURES),
        "require_pack": "error",
        "distance_transform": {"mode": "bounded_auto"},
    }

    runtime = {
        "input_mode": "table_plus_structure",
        "strict_input_mode": "error",
        "structure": {
            "feature_profile": STATIC_PROFILE,
            "descriptor_profile": "none",
            "latent_profile": "none",
            "adapter_mode": "auto",
            "provider_mode": "fixed",
        },
    }
    bench["runtime"] = runtime
    cfg["runtime"] = deepcopy(runtime)


def _metadata(cfg: dict[str, Any], *, model: str) -> None:
    cfg["benchmark"]["metadata"] = {
        "study": STUDY,
        "case_name": f"{STUDY}_{model}",
        "comparison_role": "dimension_parameter_counterpart_to_coil_sdf",
        "representation": "scalar_coil_dimensions_plus_fixed_chamber_grid",
        "structure_condition_columns": COND_COLUMNS[:5],
        "process_condition_columns": COND_COLUMNS[5:],
        "case_varying_structure_maps": False,
        "target_space": "physical_linear_zscore",
        "training_objective": "axisymmetric_volume_and_electron_energy_weighted_mse",
    }


def _base_model(name: str) -> dict[str, Any]:
    source_name = "uno" if name == "uno" else "unet"
    cfg = _load(source_name)
    bench = cfg["benchmark"]
    if name == "coord_mlp":
        old = bench["train"].pop("unet")
        bench["profile"] = "m7_coord_mlp_fourier_experimental"
        bench["eval_protocol"]["scope"] = "coord_mlp_isolated"
        bench["train"]["unet_like"]["batch_size_cases"] = 4
        bench["train"]["coord_mlp_fourier"] = {
            "epochs": 70,
            "lr": 3.0e-4,
            "optimizer": deepcopy(old["optimizer"]),
            "target_family": "allvars",
            "target_vars": ["ne", "ni", "Te", "phi"],
            "input_features": {},
            "selection": {"mode": "best_val_loss"},
            "model_cfg": {
                "cond_hidden": [128, 128],
                "latent_dim": 128,
                "decoder_hidden": [256, 256, 256],
                "decoder_activation": "gelu",
                "embedding": {
                    "type": "fourier",
                    "n_frequencies": 8,
                    "include_raw": True,
                    "frequency_scale": 10.0,
                },
                "decoder_input_norm": {"enabled": True},
                "residual_head": {"enabled": True, "init_scale": 0.0},
            },
        }
        family_name = "coord_mlp_fourier"
    else:
        family_name = "u_no" if name == "uno" else "unet"

    _dimension_inputs(cfg, model=family_name)
    bench["output_dir"] = f"{RUN_ROOT}/{name}"
    bench["eval"]["protocol_variant"] = f"{STUDY}_{name}"
    _metadata(cfg, model=name)
    return cfg


def _optimization_config(train_cfg: dict[str, Any], *, model: str) -> dict[str, Any]:
    cfg = deepcopy(train_cfg)
    bench = cfg["benchmark"]
    infer = bench["inference"]
    infer["qoi"] = {
        "uniformity": {
            "target": "ni",
            "preferred_targets": ["ni", "ne", "Te"],
            "region": "wafer_near",
            "mid_height_band_px": 2,
        },
        "bohm_flux": {"density_target": "ni", "temperature_target": "Te"},
    }
    infer["optimize"] = {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 192,
        "seed": 411,
        "space": deepcopy(OPT_SPACE),
        "geom": {"geom_id": "default"},
        "backend_cfg": {"sampler": "tpe", "n_startup_trials": 32, "multivariate": False},
        "objective": {
            "mode": "weighted_sum",
            "terms": [{"key": "bohm_flux_area_cv", "direction": "min", "weight": 1.0}],
        },
        "constraints": [{"key": "bohm_flux_area_mean", "lower": 6.0249874e16}],
        "output": {"save_fields": "top_k", "top_k": 5},
    }
    infer["single"] = {
        "enabled": True,
        "case_id": "dimension_parameter_reference",
        "cond": deepcopy(REFERENCE_COND),
        "geom": {"geom_id": "default"},
        "axis": {"mode": "steady", "value": 0.0},
    }
    metadata = bench["metadata"]
    metadata.update(
        {
            "optimization_study": OPT_STUDY,
            "optimization_case_name": f"{OPT_STUDY}_{model}",
            "optimization_variables": "four_continuous_coil_dimensions_and_pp_pp0",
            "fixed_nncoil": 4,
            "optimization_objective": "wafer_near_axisymmetric_bohm_flux_cv",
            "minimum_bohm_flux_area_mean_ratio": 0.8,
        }
    )
    return cfg


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=False)


def main() -> None:
    for name in ("unet", "uno", "coord_mlp"):
        train = _base_model(name)
        _dump(TRAIN_DIR / f"benchmark_{STUDY}_{name}.yaml", train)
        _dump(OPT_DIR / f"benchmark_{OPT_STUDY}_{name}.yaml", _optimization_config(train, model=name))
    print(f"generated training configs: {TRAIN_DIR.relative_to(ROOT)}")
    print(f"generated optimization configs: {OPT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
