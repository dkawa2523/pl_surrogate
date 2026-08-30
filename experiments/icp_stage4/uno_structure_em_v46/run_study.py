#!/usr/bin/env python
"""Generate, warm-start, and run the isolated ICP UNO v46 ablation ladder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The repository's CLI establishes the package import order expected by the
# benchmark stack; importing a leaf model first exposes an existing package
# circularity unrelated to this experiment.
from plasma_surrogate.cli.main import main as _cli_import_guard  # noqa: E402,F401
from experimental_uno import ExperimentalUNOBaseline  # noqa: E402


DATASET = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
SOURCE_RUN = ROOT / "runs/icp_conference_continuity_v45/final/sdf_vacuum_structure/seed_1237"
SOURCE_CONFIG = SOURCE_RUN / "resolved_config.yaml"
SOURCE_WEIGHTS = SOURCE_RUN / "models/u_no/eval_protocol/interp/checkpoints/weights.npz"
SOURCE_META = SOURCE_WEIGHTS.with_name("meta.json")
CONFIG_ROOT = HERE / "configs"
RUN_ROOT = ROOT / "runs/icp_uno_structure_em_v46"
SEED = 1237

STATIC = ("x", "y", "mask_plasma", "distance_signed", "distance_any")
GEOMETRY = ("part_sdf_union", "part_source_mean", "part_second_proximity", "part_competition", "solid_proximity")
EM = ("vacuum_aphi_unit", "vacuum_br_unit", "vacuum_bz_unit", "vacuum_bmag_unit")
PER_COIL = tuple(f"sdf_coil_{slot:02d}" for slot in range(1, 7))
# Keep the adopted v45 compact-pack order so its deterministic preprocessing
# artifacts can be reused exactly for non-per-coil variants.
BASE_FEATURES = (
    *STATIC,
    "part_sdf_union",
    "part_source_mean",
    *EM,
    "part_second_proximity",
    "part_competition",
    "solid_proximity",
)
EM_SHAPE_FEATURES = tuple(name for name in BASE_FEATURES if name != "vacuum_bmag_unit")
PER_COIL_FEATURES = (*BASE_FEATURES, *PER_COIL)
ALL_FEATURES = tuple(name for name in (*BASE_FEATURES, *PER_COIL) if name != "vacuum_bmag_unit")
SOURCE_LABELS = ("pp", "pp0", *BASE_FEATURES)


VARIANTS: dict[str, dict[str, Any]] = {
    "A_multires": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {},
        "question": "Does a true U-shaped multi-resolution UNO improve structural response over the flat adopted trunk?",
    },
    "AB_separate_fusion": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {"separate_lifting": True},
        "question": "Does separating process, geometry, and EM lifting prevent weak structural channels from being ignored?",
    },
    "ABC_adaptive_mix": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {"separate_lifting": True, "adaptive_local_global": True},
        "question": "Does spatially adaptive local/global mixing help both coil-near and chamber-wide responses?",
    },
    "ABD_shared_coils": {
        "features": PER_COIL_FEATURES,
        "profile": "icp_uno_structure_em_v46_percoil",
        "flags": {"separate_lifting": True, "permutation_invariant_coils": True},
        "question": "Does a shared permutation-invariant per-coil encoder improve variable layout learning?",
    },
    "ABE_em_shape_amplitude": {
        "features": EM_SHAPE_FEATURES,
        "profile": "icp_uno_structure_em_v46_emshape",
        "flags": {"separate_lifting": True, "em_reparameterization": True},
        "question": "Does bounded vector-field shape/amplitude encoding make the deterministic EM carrier learnable?",
    },
    "ABF_structure_delta": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {
            "separate_lifting": True,
            "structure_delta_weight": 0.15,
            "sobolev_weight": 0.05,
            "multiscale_weight": 0.10,
        },
        "question": "Does direct supervision of structural response plus H1/multiscale loss improve geometry sensitivity?",
    },
    "ABG_sdf_em_aux": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {"separate_lifting": True, "em_reconstruction_weight": 0.05},
        "question": "Does SDF-to-EM auxiliary reconstruction force the geometry latent to retain electromagnetic meaning?",
    },
    "ABH_target_decoders": {
        "features": BASE_FEATURES,
        "profile": "icp_uno_structure_em_v46_base",
        "flags": {"separate_lifting": True, "target_specific_decoders": True},
        "question": "Do density, temperature, and potential-specific decoders reduce destructive target interference?",
    },
    "ALL_combined": {
        "features": ALL_FEATURES,
        "profile": "icp_uno_structure_em_v46_all",
        "flags": {
            "separate_lifting": True,
            "adaptive_local_global": True,
            "permutation_invariant_coils": True,
            "em_reparameterization": True,
            "structure_delta_weight": 0.15,
            "sobolev_weight": 0.05,
            "multiscale_weight": 0.10,
            "em_reconstruction_weight": 0.05,
            "target_specific_decoders": True,
        },
        "question": "Are the individually motivated upgrades compatible when combined in one UNO?",
    },
}


def _load_source_config() -> dict[str, Any]:
    payload = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid source config: {SOURCE_CONFIG}")
    return payload


def _model_cfg(variant: str, spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    model_cfg = deepcopy(source["train"]["u_no"]["model_cfg"])
    checkpoint_meta = json.loads(SOURCE_META.read_text(encoding="utf-8"))
    model_cfg["operator_response_cfg"] = deepcopy(checkpoint_meta["operator_response_cfg"])
    model_cfg["uno_cfg"].update(
        {
            "experimental_variant": variant,
            "multiresolution": True,
            "separate_lifting": False,
            "adaptive_local_global": False,
            "permutation_invariant_coils": False,
            "em_reparameterization": False,
            "structure_delta_weight": 0.0,
            "sobolev_weight": 0.0,
            "multiscale_weight": 0.0,
            "em_reconstruction_weight": 0.0,
            "target_specific_decoders": False,
            "em_asinh_scale": 1.0,
            **dict(spec.get("flags", {})),
        }
    )
    return model_cfg


def _build_fresh_model(variant: str, spec: dict[str, Any], source: dict[str, Any]) -> ExperimentalUNOBaseline:
    model_cfg = _model_cfg(variant, spec, source)
    return ExperimentalUNOBaseline(
        input_dim=2,
        grid_shape=(440, 600),
        out_channels=4,
        output_keys=["ne", "ni", "Te", "phi"],
        seed=SEED,
        with_rho_eff_head=False,
        n_modes=int(model_cfg.get("n_modes", 16)),
        head_mlp=dict(model_cfg.get("head_mlp", {})),
        input_feature_channels=list(spec["features"]),
        uno_cfg=dict(model_cfg["uno_cfg"]),
        backend="torch",
        output_heads=dict(model_cfg.get("output_heads", {})),
        target_role_schema=dict(model_cfg.get("target_role_schema", {})),
        operator_response_cfg=dict(model_cfg["operator_response_cfg"]),
    )


def _copy_if_shape(state: dict[str, np.ndarray], source: dict[str, np.ndarray], target_key: str, source_key: str) -> int:
    if target_key not in state or source_key not in source:
        return 0
    if state[target_key].shape != source[source_key].shape:
        return 0
    state[target_key] = np.asarray(source[source_key], dtype=np.float32).copy()
    return int(state[target_key].size)


def _warm_start(variant: str, spec: dict[str, Any], source_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    model = _build_fresh_model(variant, spec, source_cfg)
    state = model.state_dict_numpy()
    with np.load(SOURCE_WEIGHTS, allow_pickle=False) as stored:
        source = {name: np.asarray(stored[name], dtype=np.float32) for name in stored.files}
    copied = 0
    copied_keys: list[str] = []
    for key in list(state):
        count = _copy_if_shape(state, source, key, key)
        if count:
            copied += count
            copied_keys.append(key)

    # Reuse the five learned formal UNO blocks across the U-shaped encoder.
    target_blocks = ["enc_blocks.0", "low_blocks.0", "bottleneck_blocks.0"]
    suffixes = (
        "global_mix.weight_pos",
        "global_mix.weight_neg",
        "local_mix.0.weight",
        "local_mix.0.bias",
        "local_mix.2.weight",
        "local_mix.2.bias",
        "skip.weight",
        "skip.bias",
        "norm.weight",
        "norm.bias",
    )
    for block_index, target_prefix in enumerate(target_blocks):
        for suffix in suffixes:
            target_key = f"torch::backbone.{target_prefix}.{suffix}"
            source_key = f"torch::backbone.blocks.{block_index}.{suffix}"
            count = _copy_if_shape(state, source, target_key, source_key)
            if count:
                copied += count
                copied_keys.append(target_key)

    target_labels = ("pp", "pp0", *tuple(spec["features"]))
    source_input = source["torch::backbone.in_proj.weight"]
    direct_key = "torch::backbone.in_proj.weight"
    separate = bool(spec.get("flags", {}).get("separate_lifting", False))
    if not separate and direct_key in state and int(state[direct_key].shape[0]) == int(source_input.shape[0]):
        direct = np.asarray(state[direct_key], dtype=np.float32).copy()
        for target_index, label in enumerate(target_labels):
            if label in SOURCE_LABELS and target_index < direct.shape[1]:
                direct[:, target_index] = source_input[:, SOURCE_LABELS.index(label)]
        state[direct_key] = direct
    elif separate:
        group_labels = {
            "condition": ("pp", "pp0"),
            "static": tuple(name for name in spec["features"] if name in STATIC),
            "geometry": tuple(name for name in spec["features"] if name in GEOMETRY),
            "electromagnetic": tuple(name for name in spec["features"] if name in EM),
            "coil_slots": tuple(name for name in spec["features"] if name in PER_COIL),
        }
        for group, labels in group_labels.items():
            key = f"torch::backbone.group_lifts.{group}.weight"
            if key not in state:
                continue
            weight = np.asarray(state[key], dtype=np.float32).copy()
            for local_index, label in enumerate(labels):
                if label in SOURCE_LABELS and local_index < weight.shape[1]:
                    weight[:, local_index] = source_input[:, SOURCE_LABELS.index(label)]
            state[key] = weight
            bias_key = f"torch::backbone.group_lifts.{group}.bias"
            if bias_key in state:
                state[bias_key].fill(0.0)
        condition_bias = "torch::backbone.group_lifts.condition.bias"
        if condition_bias in state and "torch::backbone.in_proj.bias" in source:
            state[condition_bias] = source["torch::backbone.in_proj.bias"].copy()
        fuse_key = "torch::backbone.in_proj.weight"
        fuse_bias = "torch::backbone.in_proj.bias"
        if fuse_key in state:
            fuse = np.zeros_like(state[fuse_key])
            width = int(fuse.shape[0])
            group_count = int(fuse.shape[1] // width)
            for group_index in range(group_count):
                scale = 0.1 if group_index == group_count - 1 and "sdf_coil_01" in spec["features"] else 1.0
                fuse[:, group_index * width : (group_index + 1) * width, 0, 0] = np.eye(width, dtype=np.float32) * scale
            state[fuse_key] = fuse
        if fuse_bias in state:
            state[fuse_bias].fill(0.0)

    out = CONFIG_ROOT / "initial_weights" / f"{variant}_seed{SEED}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **state)
    total = int(sum(np.asarray(value).size for key, value in state.items() if not key.endswith(("density_means", "density_scales", "density_reference_scale"))))
    metadata = {
        "variant": variant,
        "source_checkpoint": str(SOURCE_WEIGHTS.relative_to(ROOT)),
        "target_checkpoint": str(out.relative_to(ROOT)),
        "source_input_labels": list(SOURCE_LABELS),
        "target_input_labels": list(target_labels),
        "exact_or_block_copied_keys": sorted(set(copied_keys)),
        "copied_parameter_elements_lower_bound": int(copied),
        "target_parameter_elements": total,
        "copied_fraction_lower_bound": float(copied / max(total, 1)),
        "new_modules_randomly_initialized": True,
        "all_layers_trainable": True,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out, metadata


def _config_for(variant: str, spec: dict[str, Any], source: dict[str, Any], epochs: int, weights: Path) -> dict[str, Any]:
    cfg = deepcopy(source)
    run_dir = RUN_ROOT / "final" / variant / f"seed_{SEED}"
    cfg["output_dir"] = run_dir.relative_to(ROOT).as_posix()
    cfg["seed"] = SEED
    cfg["dataset"]["root"] = DATASET.relative_to(ROOT).as_posix()
    cfg["dataset"]["index_csv"] = "index.csv"
    cfg["dataset"]["cond_columns"] = ["pp", "pp0"]
    cfg["split"]["fixed_interp"] = {
        "path": (DATASET / "split_membership.json").relative_to(ROOT).as_posix(),
        "allow_unassigned": False,
    }
    cfg["split"]["structure_holdout"] = {"enabled": False, "required": False}
    cfg["eval_protocol"]["primary_split"] = "interp"
    cfg["eval_protocol"]["scope"] = "u_no_isolated"
    coord = cfg["preprocessing"]["coord_features"]
    # The isolated launcher maps this known compact-pack route to the exact
    # variant channel tuple in-process.  No registry file is edited.
    coord["channels_from_profile"] = "icp_coil_sdf_source_mean_vacuum_structure_v1"
    coord["static_output"] = "features/static_spatial_feature_pack.npz"
    coord["case_structure_output"] = "features/case_structure_feature_pack.npz"
    coord["require_case_variation"] = True
    train = cfg["train"]["u_no"]
    train["epochs"] = int(epochs)
    train["batch_size_cases"] = 16
    train["initial_weights_path"] = weights.relative_to(ROOT).as_posix()
    train["input_features"]["features"] = list(spec["features"])
    train["model_cfg"] = _model_cfg(variant, spec, source)
    train["selection"]["warmup_epochs"] = min(5, max(0, int(epochs) // 4))
    # Fixed screening budget: all candidates see the same epoch count and the
    # same validation cadence.  Full 15-epoch studies validate every three
    # epochs to keep validation overhead comparable; the two-epoch contract
    # smoke still validates every epoch.
    train["selection"]["eval_every_n_epochs"] = 1 if int(epochs) <= 2 else 3
    spatial = cfg["train"]["loss"]["supervised"]["spatial"]
    if bool(spec.get("spatial_loss", False)):
        spatial["gradient_weight"] = 0.05
        spatial["multiscale_weight"] = 0.10
        spatial["multiscale_scales"] = [2, 4]
    else:
        spatial["gradient_weight"] = 0.0
        spatial["multiscale_weight"] = 0.0
    cfg["eval"]["protocol_variant"] = f"icp_uno_structure_em_v46_{variant}_seed{SEED}"
    cfg["runtime"]["structure"]["feature_profile"] = "icp_coil_sdf_source_mean_vacuum_structure_v1"
    cfg["runtime"]["structure"]["provider_mode"] = "parametric_parts"
    cfg.setdefault("metadata", {}).update(
        {
            "study": "icp_uno_structure_em_v46",
            "experimental_only": True,
            "formal_conference_model_untouched": True,
            "variant": variant,
            "hypothesis": str(spec["question"]),
            "seed": SEED,
            "epochs": int(epochs),
            "comparison_source": "conference_continuity_v45/sdf_vacuum_structure",
            "frozen_test_contract": "v43_G4_plus_legacy_source_test",
            "experimental_profile_channels": list(
                PER_COIL_FEATURES if "sdf_coil_01" in spec["features"] else BASE_FEATURES
            ),
        }
    )
    return {"benchmark": cfg, "runtime": deepcopy(cfg["runtime"])}


def generate(epochs: int, smoke: bool) -> list[dict[str, Any]]:
    if not SOURCE_WEIGHTS.is_file():
        raise FileNotFoundError(SOURCE_WEIGHTS)
    source = _load_source_config()
    effective_epochs = 2 if smoke else int(epochs)
    suffix = "smoke" if smoke else "final"
    manifest: list[dict[str, Any]] = []
    for variant, spec in VARIANTS.items():
        weights, warm_meta = _warm_start(variant, spec, source)
        payload = _config_for(variant, spec, source, effective_epochs, weights)
        payload["benchmark"]["output_dir"] = (
            RUN_ROOT / suffix / variant / f"seed_{SEED}"
        ).relative_to(ROOT).as_posix()
        path = CONFIG_ROOT / suffix / variant / f"seed_{SEED}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        manifest.append(
            {
                "variant": variant,
                "config": str(path.relative_to(ROOT)),
                "run": payload["benchmark"]["output_dir"],
                "epochs": effective_epochs,
                "question": spec["question"],
                "warm_start_fraction_lower_bound": warm_meta["copied_fraction_lower_bound"],
            }
        )
    manifest_path = CONFIG_ROOT / f"manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run(manifest: list[dict[str, Any]], selected: set[str]) -> None:
    env = dict(os.environ, PLASMA_SURROGATE_ENABLE_TORCH="1", PYTHONPATH=str(ROOT / "src"))
    python = ROOT / ".venv-torch/Scripts/python.exe"
    entry = HERE / "run_entry.py"
    for item in manifest:
        if selected and item["variant"] not in selected:
            continue
        run_dir = ROOT / item["run"]
        if (run_dir / "leaderboard.csv").is_file():
            print(f"SKIP completed {item['variant']}: {run_dir}", flush=True)
            continue
        command = [str(python), str(entry), "benchmark", "run", "--config", str(ROOT / item["config"])]
        print(f"RUN {item['variant']} ({item['epochs']} epochs)", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "run", "all"), default="generate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    args = parser.parse_args()
    manifest = generate(max(1, int(args.epochs)), bool(args.smoke))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"run", "all"}:
        run(manifest, set(args.variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
